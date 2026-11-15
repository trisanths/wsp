"""Cross-tokenizer skill tops: a donor from another tokenizer family behind a vocab bridge.

wsp/tops.py needs the donor to share the base's token ids, because the
donor's head scores the donor's vocabulary and the top must return logits in
the base's vocabulary. This module removes that limit on the output side.
The input side needs nothing: the trunk runs on the base's tokens, the
stitch maps the trunk state to the donor's width, and the donor's layers
never see token ids.

The bridge. The donor head gives one logit per donor token. The bridge turns
that row into one logit per base token:

  exact     the base token's bytes equal a donor token's bytes. The base token
            takes that donor logit. Bytes come from a normalized token string:
            the byte-level marker (the "G with dot" character) and the
            SentencePiece marker (the "lower one eighth block" character) both
            become a space, SentencePiece byte tokens (<0x41>) become one byte,
            and byte-level tokens are decoded through the GPT-2 byte table. So
            the three spellings of "the with a leading space" give one key.
  special   a short list of end-of-turn and end-of-text pairs (and the two
            tokenizers' eos tokens) map to each other, so a donor can stop.
  fallback  a base token with no exact match is tokenized by the donor
            tokenizer. The first donor token's logit plus a learned per-token
            bias is its logit (fallback="bias"). Option fallback="linear": a
            small linear map from the donor's normed final state to the
            base's final state, read out by the base's own unembedding, plus a
            learned scale, shift, and bias. A base token whose bytes the
            donor cannot start (a partial UTF-8 byte with no donor byte token)
            and a base special token with no match are masked to the dtype
            minimum.

The bias starts at minus the log of (1 + the number of fallback tokens that
share the same donor token), so the tokens that share a donor start do not
double count. Exact tokens are never changed by the bias.

Identity. When check_vocab passes (the two tokenizers give the same ids to
every shared token), the bridge is the identity on ids below both head widths
and masks what the existing path masks, so a same-tokenizer donor gives the
logits of the existing path.

Fit objective for a cross-tokenizer top (fit_cross_stitch). The donor's
distribution cannot be compared token by token, because the donor and the base
cut the same text in different places. The fit uses two terms.
  Text target. The donor writes its own greedy answer to each pool prompt in
  its own chat format. The base tokenizer re-tokenizes prompt plus answer, and
  the top is trained with cross entropy on the answer tokens. This matches what
  the donor says, in the base's tokens, through the bridge (including the
  fallback biases).
  Shared-token KL. At every pair of positions where a base token and a donor
  token end at the same character of the answer (offsets measured from the
  start of the answer), both models predict the next token from the same text
  prefix. There the donor's logits over the donor ids that have an exact match,
  and the top's logits over the matching base ids, are each renormalized over
  that shared set, and the KL(donor || top) is the loss.
Limits. The KL sees only the exact-matched tokens, so the distribution over
fallback tokens is trained by the text target alone. The donor prefix uses the
donor's chat template and the base prefix uses the base's, so a position's
context is the same text but not the same tokens. The donor's greedy answers
are one sample of its behavior, so the top copies that sample, and a skill the
prompts do not exercise is not trained. The penalty is the top's negative log
likelihood of the answer minus the donor's, summed over the answer and divided
by the donor's answer token count (nats per donor token); it compares two
tokenizations of one text, so a split that costs bits (a base token the bridge
reaches only by fallback) shows up as penalty.
"""

import math
import re
import time

import torch
from torch import nn
from torch.nn import functional as F

from wsp import stitch as stitch_lib
from wsp.align import match_offsets, ridge_fit

FALLBACKS = ("bias", "linear")
SPECIAL_PAIRS = [("<|im_end|>", "<|eot_id|>"), ("<|endoftext|>", "<|end_of_text|>"),
                 ("<|im_start|>", "<|start_header_id|>")]
BYTE_LEVEL_MARK = "Ġ"
SENTENCEPIECE_MARK = "▁"
MARK_MIN = 3


# ---------------------------------------------------------------- token bytes


def byte_decoder():
    """The GPT-2 table inverted: one printable character back to its byte."""
    kept = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
    chars = kept[:]
    extra = 0
    for value in range(256):
        if value not in kept:
            kept.append(value)
            chars.append(256 + extra)
            extra += 1
    out = {}
    for value, char in zip(kept, chars):
        out[chr(char)] = value
    return out


BYTE_DECODER = byte_decoder()


def vocab_kind(tokens):
    """"bytelevel", "sentencepiece", or "plain", from the tokens that start with a space marker."""
    byte_level = 0
    sentence = 0
    for token in tokens:
        if token.startswith(BYTE_LEVEL_MARK):
            byte_level += 1
        elif token.startswith(SENTENCEPIECE_MARK):
            sentence += 1
    floor = max(MARK_MIN, len(tokens) // 200)
    if byte_level >= floor and byte_level >= sentence:
        return "bytelevel"
    if sentence >= floor:
        return "sentencepiece"
    return "plain"


def token_bytes(token, kind):
    """The bytes a token stands for. The marked and the plain spelling of one token give the same bytes."""
    if kind == "bytelevel":
        try:
            return bytes([BYTE_DECODER[char] for char in token])
        except KeyError:
            return token.encode("utf-8")
    if kind == "sentencepiece":
        found = re.fullmatch(r"<0x([0-9A-Fa-f]{2})>", token)
        if found:
            return bytes([int(found.group(1), 16)])
        return token.replace(SENTENCEPIECE_MARK, " ").encode("utf-8")
    return token.encode("utf-8")


def special_ids(tokenizer):
    """Ids of added tokens flagged special, plus the tokenizer's named special ids."""
    out = set()
    decoder = getattr(tokenizer, "added_tokens_decoder", None) or {}
    for index, added in decoder.items():
        if getattr(added, "special", False):
            out.add(int(index))
    for index in tokenizer.all_special_ids:
        out.add(int(index))
    return out


def vocab_table(tokenizer):
    """Return (kind, ids_by_bytes, bytes_by_id, specials) for a tokenizer.

    Special tokens key on their raw string. When two ids share bytes the
    lowest id keeps the key.
    """
    vocab = tokenizer.get_vocab()
    kind = vocab_kind(list(vocab.keys()))
    specials = special_ids(tokenizer)
    by_bytes = {}
    by_id = {}
    for token, index in sorted(vocab.items(), key=lambda item: item[1]):
        if index in specials:
            raw = token.encode("utf-8")
        else:
            raw = token_bytes(token, kind)
        by_id[index] = raw
        if raw not in by_bytes:
            by_bytes[raw] = index
    return kind, by_bytes, by_id, specials


# ---------------------------------------------------------------- the bridge module


class VocabBridge(nn.Module):
    """Donor head logits to base-vocabulary logits.

    gather[b] is the donor id whose logit base id b takes. kind[b] is 0
    (masked), 1 (exact or special), or 2 (fallback). With fallback "linear"
    the fallback logits come from the base's own unembedding.
    """

    def __init__(self, gather, kind, fallback="bias", d_donor=0, d_base=0, base_head=None, bias_init=None,
                 shared_base=None, shared_donor=None, report=None, identity=False):
        super().__init__()
        if fallback not in FALLBACKS:
            raise ValueError(f"fallback must be one of {FALLBACKS}, got {fallback!r}")
        rows = int(gather.shape[0])
        self.rows = rows
        self.fallback = fallback
        self.identity = bool(identity)
        self.report = report or {}
        self.register_buffer("gather", gather.long(), persistent=False)
        self.register_buffer("kind", kind.to(torch.int8), persistent=False)
        self.register_buffer("is_fallback", kind == 2, persistent=False)
        self.register_buffer("is_masked", kind == 0, persistent=False)
        if shared_base is None:
            shared_base = torch.nonzero(kind == 1).flatten()
            shared_donor = gather[shared_base]
        self.register_buffer("shared_base", shared_base.long(), persistent=False)
        self.register_buffer("shared_donor", shared_donor.long(), persistent=False)
        self.has_fallback = bool((kind == 2).any())
        self.has_masked = bool((kind == 0).any())
        bias = torch.zeros(rows) if bias_init is None else bias_init.float().clone()
        self.bias = nn.Parameter(bias, requires_grad=False)
        self.map = None
        if fallback == "linear":
            if base_head is None or d_donor == 0 or d_base == 0:
                raise ValueError("fallback linear needs d_donor, d_base, and the base head weight")
            self.map = nn.Linear(d_donor, d_base, bias=False)
            nn.init.zeros_(self.map.weight)
            self.map.weight.requires_grad_(False)
            self.shift = nn.Parameter(torch.zeros(()), requires_grad=False)
            self.scale = nn.Parameter(torch.ones(()), requires_grad=False)
        self._head = [base_head]

    def forward(self, normed, head):
        """normed: the donor's normed final state. head: the donor head (Linear to the donor vocabulary)."""
        donor_logits = head(normed.to(head.weight.dtype))
        out = donor_logits.index_select(-1, self.gather.to(donor_logits.device))
        if self.has_fallback:
            if self.map is not None:
                state = self.map(normed.to(self.map.weight.dtype))
                weight = self._head[0].to(device=state.device, dtype=state.dtype)
                z = F.linear(state, weight).to(out.dtype)
                fallback = z * self.scale.to(out.dtype) + self.shift.to(out.dtype) + self.bias.to(out.dtype)
            else:
                fallback = out + self.bias.to(out.dtype)
            out = torch.where(self.is_fallback, fallback, out)
        if self.has_masked:
            out = out.masked_fill(self.is_masked, torch.finfo(out.dtype).min)
        return out

    def trainable(self):
        params = [self.bias]
        if self.map is not None:
            params += [self.map.weight, self.shift, self.scale]
        return params

    def counts(self):
        """Exact, fallback, and masked base ids."""
        return {"exact": int((self.kind == 1).sum()), "fallback": int((self.kind == 2).sum()),
                "masked": int((self.kind == 0).sum())}


def tokenize_first(tokenizer, strings):
    """The first token id of each string under the donor tokenizer, without special tokens. -1 for an empty result."""
    out = []
    step = 4096
    for start in range(0, len(strings), step):
        enc = tokenizer(strings[start:start + step], add_special_tokens=False)["input_ids"]
        for ids in enc:
            out.append(int(ids[0]) if len(ids) > 0 else -1)
    return out


def longest_prefix(raw, donor_by_bytes):
    """The donor id of the longest prefix of raw that is a donor token, or -1."""
    for end in range(len(raw), 0, -1):
        found = donor_by_bytes.get(raw[:end])
        if found is not None:
            return found
    return -1


def build_bridge(base_tokenizer, donor_tokenizer, base_rows, donor_rows, d_donor=0, base_head=None,
                 fallback="bias", special_pairs=None, strict_identity=True):
    """Build the bridge from donor ids to base ids. report holds the coverage of the vocabulary.

    base_rows and donor_rows are the widths of the two heads. Ids of the base
    head that no base token uses (padding rows) are masked, and so are base
    tokens the donor cannot start.
    """
    from wsp.tops import check_vocab

    report = {"fallback": fallback, "base_rows": int(base_rows), "donor_rows": int(donor_rows)}
    identity_report = None
    if strict_identity:
        try:
            identity_report = check_vocab(base_tokenizer, donor_tokenizer, donor_rows, base_rows)
        except ValueError:
            identity_report = None
    if identity_report is not None:
        gather = torch.arange(base_rows).clamp(max=max(0, donor_rows - 1))
        kind = torch.ones(base_rows, dtype=torch.int8)
        kind[min(base_rows, donor_rows):] = 0
        for index in identity_report["masked_ids"]:
            kind[index] = 0
        report.update({"identity": True, "base_tokens": len(base_tokenizer.get_vocab()),
                       "exact": int((kind == 1).sum()), "fallback": 0, "masked": int((kind == 0).sum()),
                       "exact_share": 1.0 - identity_report["base_only"] / max(1, len(base_tokenizer.get_vocab()))})
        return VocabBridge(gather, kind, fallback, d_donor, int(base_head.shape[1]) if base_head is not None else 0,
                           base_head, report=report, identity=True)

    base_vocab = base_tokenizer.get_vocab()
    base_kind, base_by_bytes, base_bytes, base_special = vocab_table(base_tokenizer)
    donor_kind, donor_by_bytes, donor_bytes, donor_special = vocab_table(donor_tokenizer)
    gather = torch.zeros(base_rows, dtype=torch.long)
    kind = torch.zeros(base_rows, dtype=torch.int8)
    exact_special = 0
    need = []
    for token, index in base_vocab.items():
        if index >= base_rows:
            continue
        raw = base_bytes[index]
        found = donor_by_bytes.get(raw)
        if found is not None and found < donor_rows:
            gather[index] = found
            kind[index] = 1
            if index in base_special:
                exact_special += 1
        else:
            need.append(index)
    # Explicit special pairs (and the eos pair) override the string match.
    pairs = list(SPECIAL_PAIRS if special_pairs is None else special_pairs)
    for name in ("eos_token", "bos_token"):
        donor_name = getattr(donor_tokenizer, name, None)
        base_name = getattr(base_tokenizer, name, None)
        if donor_name and base_name:
            pairs.append((donor_name, base_name))
    donor_vocab = donor_tokenizer.get_vocab()
    mapped_special = []
    for first, second in pairs:
        for donor_name, base_name in ((first, second), (second, first)):
            if donor_name in donor_vocab and base_name in base_vocab:
                d_id = donor_vocab[donor_name]
                b_id = base_vocab[base_name]
                # Only special tokens pair up; a plain token that spells "<s>" is just text.
                if d_id not in donor_special or b_id not in base_special:
                    continue
                if d_id < donor_rows and b_id < base_rows and kind[b_id] != 1:
                    gather[b_id] = d_id
                    kind[b_id] = 1
                    mapped_special.append((base_name, donor_name))
    need = [index for index in need if kind[index] != 1]

    # Fallback: the first donor token of the base token's own text.
    strings = []
    todo = []
    fallback_masked = 0
    for index in need:
        if index in base_special:
            fallback_masked += 1
            continue
        raw = base_bytes[index]
        try:
            strings.append(raw.decode("utf-8"))
            todo.append(index)
        except UnicodeDecodeError:
            found = longest_prefix(raw, donor_by_bytes)
            if found >= 0 and found < donor_rows:
                gather[index] = found
                kind[index] = 2
            else:
                fallback_masked += 1
    firsts = tokenize_first(donor_tokenizer, strings)
    for index, first in zip(todo, firsts):
        if first >= 0 and first < donor_rows and first not in donor_special:
            gather[index] = first
            kind[index] = 2
        else:
            fallback_masked += 1
    shared = {}
    for index in range(base_rows):
        if kind[index] != 0:
            shared[int(gather[index])] = shared.get(int(gather[index]), 0) + 1
    bias_init = torch.zeros(base_rows)
    group = {}
    for index in range(base_rows):
        if kind[index] == 2:
            group[int(gather[index])] = group.get(int(gather[index]), 0) + 1
    for index in range(base_rows):
        if kind[index] == 2:
            bias_init[index] = -math.log(1.0 + group[int(gather[index])])
    n_base = len(base_vocab)
    counts = {"exact": int((kind == 1).sum()), "fallback": int((kind == 2).sum()), "masked": int((kind == 0).sum())}
    report.update({"identity": False, "base_kind": base_kind, "donor_kind": donor_kind, "base_tokens": n_base,
                   "donor_tokens": len(donor_vocab), "exact": counts["exact"], "fallback": counts["fallback"],
                   "masked": counts["masked"], "exact_share": counts["exact"] / max(1, n_base),
                   "fallback_share": counts["fallback"] / max(1, n_base),
                   "exact_special": exact_special, "mapped_special": mapped_special[:12],
                   "fallback_masked_tokens": fallback_masked})
    d_base = int(base_head.shape[1]) if base_head is not None else 0
    return VocabBridge(gather, kind, fallback, d_donor, d_base, base_head, bias_init=bias_init, report=report)


# ---------------------------------------------------------------- coverage


def text_coverage(bridge, base_tokenizer, donor_tokenizer, texts):
    """Share of token occurrences on sample text that the bridge covers.

    base_exact: base tokens (tokenized by the base tokenizer) with an exact donor match.
    base_fallback: base tokens reached by fallback. donor_shared: donor tokens
    (tokenized by the donor tokenizer) that have an exact base match. The
    last two numbers say how much of the text the shared vocabulary carries
    on each side. The model-side mass is shared_mass_of_model.
    """
    kind = bridge.kind.cpu()
    shared_donor = set(bridge.shared_donor.cpu().tolist())
    base_total = 0
    base_exact = 0
    base_fallback = 0
    donor_total = 0
    donor_shared = 0
    for text in texts:
        for index in base_tokenizer(text, add_special_tokens=False)["input_ids"]:
            base_total += 1
            if index < kind.shape[0]:
                if kind[index] == 1:
                    base_exact += 1
                elif kind[index] == 2:
                    base_fallback += 1
        for index in donor_tokenizer(text, add_special_tokens=False)["input_ids"]:
            donor_total += 1
            if index in shared_donor:
                donor_shared += 1
    return {"texts": len(texts), "base_tokens": base_total, "donor_tokens": donor_total,
            "base_exact": base_exact / max(1, base_total), "base_fallback": base_fallback / max(1, base_total),
            "donor_shared": donor_shared / max(1, donor_total)}


@torch.no_grad()
def shared_mass_of_model(donor_model, donor_tokenizer, bridge, texts, max_length=256):
    """Mean donor probability mass on donor ids that have an exact base match, over the positions of sample text."""
    shared = bridge.shared_donor.to(next(donor_model.parameters()).device)
    total = 0.0
    n = 0
    for text in texts:
        enc = donor_tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        device = next(donor_model.parameters()).device
        logits = donor_model(input_ids=enc["input_ids"].to(device),
                             attention_mask=enc["attention_mask"].to(device)).logits[0].float()
        prob = torch.softmax(logits, dim=-1)
        total += float(prob[:, shared].sum(dim=-1).sum())
        n += prob.shape[0]
    return total / max(1, n)


def coverage_report(bridge, base_tokenizer, donor_tokenizer, texts, donor_model=None):
    """The bridge's vocabulary report plus the sample-text shares (and the model mass when a donor is given)."""
    out = dict(bridge.report)
    out["text"] = text_coverage(bridge, base_tokenizer, donor_tokenizer, texts)
    if donor_model is not None:
        out["donor_mass_on_shared"] = shared_mass_of_model(donor_model, donor_tokenizer, bridge, texts)
    return out


# ---------------------------------------------------------------- encoding with offsets


def strip_bos(tokenizer, text):
    """The text without a leading BOS string. A chat template can write the BOS as text, and the tokenizer adds
    its own, so the model would see two."""
    bos = getattr(tokenizer, "bos_token", None)
    if bos and text.startswith(bos):
        return text[len(bos):]
    return text


def encode_offsets(tokenizer, texts, seq_len, side):
    """Tokenize a batch with character offsets. Return (ids, mask, offsets) with padding on the given side.

    A BOS string at the start of a text is dropped before tokenizing (the tokenizer adds one BOS), and the
    offsets are shifted back, so they stay in the coordinates of the original text.
    """
    shifts = []
    stripped = []
    for text in texts:
        cut = strip_bos(tokenizer, text)
        shifts.append(len(text) - len(cut))
        stripped.append(cut)
    texts = stripped
    old = tokenizer.padding_side
    tokenizer.padding_side = side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        kwargs = {"return_tensors": "pt", "padding": True, "return_offsets_mapping": True}
        if seq_len is not None:
            kwargs["truncation"] = True
            kwargs["max_length"] = seq_len
        enc = tokenizer(list(texts), **kwargs)
    finally:
        tokenizer.padding_side = old
    offsets = enc["offset_mapping"].clone()
    for row in range(len(shifts)):
        if shifts[row] > 0:
            real = offsets[row, :, 1] > offsets[row, :, 0]
            offsets[row][real] += shifts[row]
    return enc["input_ids"], enc["attention_mask"], offsets


def real_tokens(mask_row, offsets_row):
    """(index, start, end) of every real token that has characters."""
    out = []
    for index in range(mask_row.shape[0]):
        if int(mask_row[index]) != 1:
            continue
        start = int(offsets_row[index][0])
        end = int(offsets_row[index][1])
        if end > start:
            out.append((index, start, end))
    return out


def answer_weights(mask, offsets, answer_start, skip_first):
    """Weight (batch, seq-1) of the predictions that count: the next token starts in the answer.

    The prediction at position i of the next token i+1 counts when both are
    real, token i+1 has characters and starts at or after the answer start,
    and (skip_first) i is not the row's first real token.
    """
    batch, length = mask.shape
    weight = torch.zeros(batch, length - 1)
    for row in range(batch):
        first = int(mask[row].long().argmax())
        for index in range(length - 1):
            if int(mask[row, index]) != 1 or int(mask[row, index + 1]) != 1:
                continue
            start = int(offsets[row, index + 1, 0])
            end = int(offsets[row, index + 1, 1])
            if end <= start or start < answer_start[row]:
                continue
            if skip_first and index == first:
                continue
            weight[row, index] = 1.0
    return weight


def aligned_positions(base_mask, base_offsets, base_start, donor_mask, donor_offsets, donor_start):
    """Position pairs where a base token and a donor token end at the same answer character.

    Offsets are measured from each side's own answer start, so two different
    prompts line up on the answer. Tokens that end before the answer do not
    count. Returns four lists: base row, base index, donor row, donor index.
    """
    base_rows = []
    base_index = []
    donor_rows = []
    donor_index = []
    for row in range(base_mask.shape[0]):
        base_tokens = real_tokens(base_mask[row], base_offsets[row])
        donor_tokens = real_tokens(donor_mask[row], donor_offsets[row])
        b_ends = []
        b_pos = []
        for index, _, end in base_tokens:
            if end - base_start[row] >= 0:
                b_ends.append(end - base_start[row])
                b_pos.append(index)
        d_ends = []
        d_pos = []
        for index, _, end in donor_tokens:
            if end - donor_start[row] >= 0:
                d_ends.append(end - donor_start[row])
                d_pos.append(index)
        for i, j in match_offsets(b_ends, d_ends):
            base_rows.append(row)
            base_index.append(b_pos[i])
            donor_rows.append(row)
            donor_index.append(d_pos[j])
    return base_rows, base_index, donor_rows, donor_index


# ---------------------------------------------------------------- loss terms


def make_items(texts, donor_texts=None, answers=None):
    """Items for the loss terms. texts are base prompts (or full texts); answers are appended when given.

    donor_texts are the same prompts in the donor's format. Without answers an
    item is a full text and the whole text counts.
    """
    items = []
    for index, text in enumerate(texts):
        donor_text = donor_texts[index] if donor_texts is not None else text
        answer = answers[index] if answers is not None else ""
        items.append({"base": text + answer, "donor": donor_text + answer, "base_start": len(text) if answers
                      is not None else 0, "donor_start": len(donor_text) if answers is not None else 0,
                      "answer": answer})
    return items


@torch.no_grad()
def donor_answers(donor_model, donor_tokenizer, prompts, max_new_tokens=96, batch_size=16):
    """The donor's greedy answers to prompts in its own format, decoded to text."""
    old = donor_tokenizer.padding_side
    donor_tokenizer.padding_side = "left"
    if donor_tokenizer.pad_token is None:
        donor_tokenizer.pad_token = donor_tokenizer.eos_token
    device = next(donor_model.parameters()).device
    out = []
    try:
        for start in range(0, len(prompts), batch_size):
            enc = donor_tokenizer(list(prompts[start:start + batch_size]), return_tensors="pt", padding=True)
            ids = enc["input_ids"].to(device)
            made = donor_model.generate(input_ids=ids, attention_mask=enc["attention_mask"].to(device),
                                        max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                        top_p=None, top_k=None, pad_token_id=donor_tokenizer.pad_token_id)
            out += donor_tokenizer.batch_decode(made[:, ids.shape[1]:], skip_special_tokens=True)
    finally:
        donor_tokenizer.padding_side = old
    return out


def item_terms(net, top, donor_model, items, seq_len, grad, kl=True):
    """Loss terms of a batch of items. Returns sums and counts (see keys below).

    ce_sum, ce_n: the top's cross entropy on the answer tokens (base ids).
    donor_sum, donor_n: the donor's own cross entropy on its answer tokens.
    kl_sum, kl_n: KL(donor || top) over the shared ids at aligned positions.
    """
    from wsp.tops import first_index, model_device, positions_of

    bridge = top.bridge
    donor_tok = top.donor_tokenizer
    b_ids, b_mask, b_off = encode_offsets(net.tokenizer, [i["base"] for i in items], seq_len, "left")
    d_ids, d_mask, d_off = encode_offsets(donor_tok, [i["donor"] for i in items], seq_len, "right")
    device = model_device(net.base)
    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)
    pos = positions_of(b_mask)
    with torch.no_grad():
        h = net.run_trunk(b_ids, b_mask, pos, top.cut_host, None, 0)
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        x = top.run(h, b_mask, pos, None, 0, first_index(b_mask))
        logits = top.logits(x)
        base_start = [i["base_start"] for i in items]
        donor_start = [i["donor_start"] for i in items]
        weight = answer_weights(b_mask.cpu(), b_off, base_start, top.has_sink).to(device)
        ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), b_ids[:, 1:].reshape(-1),
                             reduction="none").reshape(weight.shape)
        out = {"ce_sum": (ce * weight).sum(), "ce_n": float(weight.sum())}
    donor_device = model_device(donor_model)
    with torch.no_grad():
        donor_logits = donor_model(input_ids=d_ids.to(donor_device), attention_mask=d_mask.to(donor_device)).logits
        d_weight = answer_weights(d_mask, d_off, donor_start, False).to(donor_device)
        d_ce = F.cross_entropy(donor_logits[:, :-1].float().reshape(-1, donor_logits.shape[-1]),
                               d_ids[:, 1:].to(donor_device).reshape(-1), reduction="none").reshape(d_weight.shape)
        out["donor_sum"] = (d_ce * d_weight).sum().to(device)
        out["donor_n"] = float(d_weight.sum())
    out["kl_sum"] = torch.zeros((), device=device)
    out["kl_n"] = 0.0
    if kl and bridge.shared_base.numel() > 0:
        br, bi, dr, di = aligned_positions(b_mask.cpu(), b_off, base_start, d_mask, d_off, donor_start)
        if top.has_sink:
            keep = []
            firsts = first_index(b_mask).cpu()
            for k in range(len(br)):
                keep.append(bi[k] != int(firsts[br[k]]))
            br = [v for v, ok in zip(br, keep) if ok]
            bi = [v for v, ok in zip(bi, keep) if ok]
            dr = [v for v, ok in zip(dr, keep) if ok]
            di = [v for v, ok in zip(di, keep) if ok]
        if len(br) > 0:
            with context:
                student = logits[torch.tensor(br, device=device), torch.tensor(bi, device=device)]
                student = student.index_select(-1, bridge.shared_base.to(device)).float()
            with torch.no_grad():
                teacher = donor_logits[torch.tensor(dr, device=donor_device), torch.tensor(di, device=donor_device)]
                teacher = teacher.index_select(-1, bridge.shared_donor.to(donor_device)).float().to(device)
            t = torch.log_softmax(teacher, dim=-1)
            s = torch.log_softmax(student, dim=-1)
            out["kl_sum"] = (t.exp() * (t - s)).sum()
            out["kl_n"] = float(len(br))
    return out


def cross_penalty(net, top, texts, donor_model, seq_len=256, native=None):
    """Top NLL minus donor NLL of the same text, divided by the donor's token count (nats per donor token).

    texts are plain strings (the whole text counts) or items from
    make_items (only the answer counts). The same keys as SkillTops.penalty
    return, plus kl_shared, the mean KL over the shared ids at aligned
    positions, and the answer characters.
    """
    items = []
    for text in texts:
        items.append(text if isinstance(text, dict) else make_items([text])[0])
    donor_sum = 0.0
    top_sum = 0.0
    donor_n = 0.0
    top_n = 0.0
    kl_sum = 0.0
    kl_n = 0.0
    chars = 0
    used = 0
    with torch.no_grad():
        for item in items:
            terms = item_terms(net, top, donor_model, [item], seq_len, False)
            if terms["donor_n"] == 0 or terms["ce_n"] == 0:
                continue
            donor_sum += float(terms["donor_sum"])
            top_sum += float(terms["ce_sum"])
            donor_n += terms["donor_n"]
            top_n += terms["ce_n"]
            kl_sum += float(terms["kl_sum"])
            kl_n += terms["kl_n"]
            chars += len(item["base"]) - item["base_start"]
            used += 1
    if donor_n == 0:
        raise ValueError("no text has an answer token on both sides")
    penalty = (top_sum - donor_sum) / donor_n
    return {"native_loss": donor_sum / donor_n, "top_loss": top_sum / donor_n, "penalty": penalty,
            "penalty_all": penalty, "tokens": int(donor_n), "base_tokens": int(top_n), "texts": used,
            "drop_first": top.has_sink, "native_ce": [], "kl_shared": kl_sum / max(1.0, kl_n), "kl_positions": kl_n,
            "chars": chars, "nats_per_char_donor": donor_sum / max(1, chars),
            "nats_per_char_top": top_sum / max(1, chars)}


# ---------------------------------------------------------------- linear fallback init


@torch.no_grad()
def init_linear_fallback(net, top, donor_model, strings, lam=1e-3, max_rows=4000, shift_rows=512):
    """Ridge the donor's normed final state onto the base's normed final state, then set shift and scale.

    Rows are offset-matched tokens (the first real token of each side is
    skipped, since it is the attention sink). The shift makes the mean
    log-sum-exp of the fallback logits sit below the exact ones by the share of
    fallback tokens in the text, so the two groups start in proportion.
    """
    from wsp.tops import model_device

    bridge = top.bridge
    donor_tok = top.donor_tokenizer
    xs = []
    ys = []
    fallback_tokens = 0
    all_tokens = 0
    for text in strings:
        if sum(x.shape[0] for x in xs) >= max_rows:
            break
        b_ids, b_mask, b_off = encode_offsets(net.tokenizer, [text], None, "left")
        d_ids, d_mask, d_off = encode_offsets(donor_tok, [text], None, "right")
        b_tokens = real_tokens(b_mask[0], b_off[0])[1:]
        d_tokens = real_tokens(d_mask[0], d_off[0])[1:]
        for index in b_ids[0].tolist():
            if index < bridge.rows:
                all_tokens += 1
                if int(bridge.kind[index]) == 2:
                    fallback_tokens += 1
        pairs = match_offsets([t[2] for t in b_tokens], [t[2] for t in d_tokens])
        if len(pairs) == 0:
            continue
        base_state = net.base.model(input_ids=b_ids.to(model_device(net.base)),
                                    attention_mask=b_mask.to(model_device(net.base))).last_hidden_state[0]
        donor_state = donor_model.model(input_ids=d_ids.to(model_device(donor_model)),
                                        attention_mask=d_mask.to(model_device(donor_model))).last_hidden_state[0]
        ys.append(base_state.float().cpu()[torch.tensor([b_tokens[i][0] for i, _ in pairs])])
        xs.append(donor_state.float().cpu()[torch.tensor([d_tokens[j][0] for _, j in pairs])])
    if len(xs) == 0:
        raise ValueError("no matching tokens for the linear fallback fit")
    x = torch.cat(xs, dim=0)[:max_rows]
    y = torch.cat(ys, dim=0)[:max_rows]
    w, r2, r2_dim = ridge_fit(x, y, lam=lam)
    bridge.map.weight.copy_(w.to(bridge.map.weight.dtype))
    # Calibrate the scale and shift on the same rows.
    head = top.head.weight.detach().float().cpu()
    exact_donor = bridge.shared_donor.cpu()
    base_head = net.base.get_output_embeddings().weight.detach().float().cpu()
    fb = torch.nonzero(bridge.is_fallback.cpu()).flatten()
    rows = min(shift_rows, x.shape[0])
    diff = []
    for start in range(0, rows, 128):
        chunk = x[start:min(rows, start + 128)]
        donor_lse = torch.logsumexp(chunk @ head[exact_donor].T, dim=-1)
        pred = chunk @ w.T.float()
        fb_lse = torch.logsumexp(pred @ base_head[fb].T, dim=-1)
        diff.append(donor_lse - fb_lse)
    diff = torch.cat(diff)
    prior = min(0.5, max(1e-3, fallback_tokens / max(1, all_tokens)))
    shift = float(diff.mean()) + math.log(prior / (1.0 - prior))
    bridge.shift.fill_(shift)
    return {"r2": float(r2), "r2_dim": float(r2_dim), "pairs": int(x.shape[0]), "shift": shift,
            "fallback_prior": prior}


# ---------------------------------------------------------------- the fit


def fit_cross_stitch(net, top, prompts, steps=800, lr=None, batch_size=8, seq_len=256, seed=0, eval_texts=None,
                     donor_model=None, ridge_texts=None, max_penalty=0.25, lora_rank=0, lora_alpha=None,
                     log_every=0, log_name="stitch", objective="kd", lm_weight=0.0, donor_prompts=None,
                     eval_donor_prompts=None, answer_tokens=96, ce_weight=1.0, kl_weight=1.0, answers=None,
                     eval_answers=None):
    """Fit the stitch of a cross-tokenizer top (and the bridge's learned part) on the donor's own answers.

    prompts are strings in the base's format. donor_prompts are the same
    prompts in the donor's format (default: the same strings). The donor
    writes a greedy answer to each donor prompt (unless answers are given).
    The loss is ce_weight times the top's cross entropy on the answer tokens
    plus kl_weight times the shared-token KL at aligned positions. objective
    "lm" turns the KL off. Reports the penalty before and after on eval_texts
    (prompts too). Flags the top the way SkillTops.fit_stitch does.
    """
    from wsp.tops import LR_BY_KIND, count, model_device

    start = time.time()
    if donor_model is None:
        donor_model = net.load_donor_model(top)
    donor_model.eval()
    for param in donor_model.parameters():
        param.requires_grad_(False)
    donor_tok = top.donor_tokenizer
    if donor_tok is None:
        raise ValueError("a cross-tokenizer top needs its donor tokenizer")
    if objective == "lm":
        kl_weight = 0.0
    if len(prompts) < 4:
        raise ValueError("fit_cross_stitch needs at least four prompts")
    if eval_texts is None:
        hold = max(2, min(100, len(prompts) // 10))
        eval_texts = prompts[-hold:]
        eval_donor_prompts_all = donor_prompts[-hold:] if donor_prompts is not None else None
        prompts = prompts[:-hold]
        donor_prompts = donor_prompts[:-hold] if donor_prompts is not None else None
        answers = answers[:-hold] if answers is not None else None
        eval_donor_prompts = eval_donor_prompts_all
    gen_start = time.time()
    if answers is None:
        answers = donor_answers(donor_model, donor_tok, donor_prompts or prompts, answer_tokens)
    if eval_answers is None:
        eval_answers = donor_answers(donor_model, donor_tok, eval_donor_prompts or eval_texts, answer_tokens)
    train_items = [i for i in make_items(prompts, donor_prompts, answers) if len(i["answer"].strip()) > 0]
    eval_items = [i for i in make_items(eval_texts, eval_donor_prompts, eval_answers)
                  if len(i["answer"].strip()) > 0]
    if len(train_items) < 2 or len(eval_items) < 1:
        raise ValueError("the donor wrote too few non-empty answers to fit on")
    report = {"steps": steps, "batch_size": batch_size, "seq_len": seq_len, "seed": seed,
              "train_texts": len(train_items), "eval_texts": len(eval_items), "max_penalty": max_penalty,
              "lora_rank": lora_rank, "stitch_kind": top.spec["stitch_kind"], "cross_tokenizer": True,
              "answer_tokens": answer_tokens, "ce_weight": ce_weight, "kl_weight": kl_weight,
              "answer_seconds": round(time.time() - gen_start, 1), "fallback": top.bridge.fallback}
    ridge_items = train_items[:300] if ridge_texts is None else make_items(ridge_texts)
    with torch.no_grad():
        if top.cut_donor > 0:
            sink = stitch_lib.donor_sink(donor_model, donor_tok, top.cut_donor,
                                         [strip_bos(donor_tok, i["donor"]) for i in ridge_items])
            top.sink.copy_(sink.to(top.sink.device))
            top.has_sink = True
        if top.cut_host > 0:
            report["ridge"] = stitch_lib.ridge_init(top.stitch, net.base, net.tokenizer, top.cut_host - 1,
                                                    donor_model, donor_tok, top.cut_donor,
                                                    [strip_bos(net.tokenizer, i["base"]) for i in ridge_items])
        if top.bridge.map is not None:
            report["linear_init"] = init_linear_fallback(net, top, donor_model, [i["base"] for i in ridge_items])
    before = cross_penalty(net, top, eval_items, donor_model, seq_len)
    before.pop("native_ce")
    report["before"] = before
    if lora_rank > 0:
        from wsp.lora import apply_lora, default_alpha

        alpha = lora_alpha if lora_alpha is not None else default_alpha(lora_rank)
        apply_lora(top, lora_rank, alpha)
        top.lora = {"rank": int(lora_rank), "alpha": alpha}
    if lr is None:
        lr = LR_BY_KIND.get(top.spec["stitch_kind"], 3e-4)
    params = top.trainable()
    for param in params:
        param.requires_grad_(True)
    report["lr"] = lr
    report["trainable_params"] = count(params)
    start_state = [p.detach().clone() for p in params]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    next_batch = stitch_lib.text_sampler(train_items, batch_size, seed)
    losses = []
    parts = []
    diverged = False
    fit_start = time.time()
    early = None
    for step in range(steps):
        batch = next_batch()
        terms = item_terms(net, top, donor_model, batch, seq_len, True, kl=kl_weight > 0)
        ce = terms["ce_sum"] / max(1.0, terms["ce_n"])
        kl = terms["kl_sum"] / max(1.0, terms["kl_n"])
        loss = ce_weight * ce + kl_weight * kl
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        value = float(loss.detach())
        losses.append(value)
        parts.append((float(ce.detach()), float(kl.detach())))
        if log_every and (step + 1) % log_every == 0:
            print(f"[{log_name}] step {step + 1}/{steps} loss {value:.4f} ce {parts[-1][0]:.4f} "
                  f"kl {parts[-1][1]:.4f}", flush=True)
        if not (value == value) or value == float("inf"):
            diverged = True
        if step == 24:
            early = sum(losses) / len(losses)
        if step >= 49 and step % 25 == 24 and early is not None:
            if sum(losses[-25:]) / 25.0 > 2.0 * early:
                diverged = True
        if diverged:
            break
    if diverged:
        with torch.no_grad():
            for param, saved in zip(params, start_state):
                param.copy_(saved)
    for param in params:
        param.requires_grad_(False)
    report["fit_seconds"] = round(time.time() - fit_start, 1)
    report["loss_first"] = losses[0] if losses else None
    report["loss_last25"] = sum(losses[-25:]) / max(1, len(losses[-25:]))
    report["ce_last25"] = sum(p[0] for p in parts[-25:]) / max(1, len(parts[-25:]))
    report["kl_last25"] = sum(p[1] for p in parts[-25:]) / max(1, len(parts[-25:]))
    report["steps_run"] = len(losses)
    report["diverged"] = diverged
    after = cross_penalty(net, top, eval_items, donor_model, seq_len)
    after.pop("native_ce")
    report["after"] = after
    report["penalty"] = after["penalty"]
    top.accepted = (after["penalty"] <= max_penalty) and not diverged
    top.flag = None
    if diverged:
        top.flag = "training diverged"
    elif after["penalty"] > max_penalty:
        top.flag = f"penalty {after['penalty']:.3f} nats is above {max_penalty}"
    report["accepted"] = top.accepted
    report["flag"] = top.flag
    report["seconds"] = round(time.time() - start, 1)
    top.fit_report = report
    return report
