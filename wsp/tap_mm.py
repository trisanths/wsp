"""Multimodal taps: a text-only host reads a vision-language donor's states.

The donor is a VLM (for example Qwen2.5-VL). It reads the image and the
question natively, with its own processor and chat template, and runs its
language model only up to the read layer. The host reads only the question
text, plus the donor memory through the usual wsp.tap branches.

wsp.tap.donor_memories only knows two kinds of donors: a (model, tokenizer)
pair and a CachedDonor. A VLM needs the image next to each prompt, so this
module has a parallel path that reuses wsp.tap's branch, hooks, visibility,
and training internals:

    donor = MultimodalDonor(vlm, processor, read_layer=20)
    branch = TapBranch(d_host, donor.hidden_size, donor_id="vlm", donor_layer=20)
    attach_taps(host, [(6, branch, 0)])
    donors = {"vlm": donor}
    losses = tap_train_mm(host, host_tokenizer, donors, examples, tap_parameters(host))
    inputs = prepare_mm(host, host_tokenizer, donors, [(prompt, image)])
    logits = host(**inputs).logits

examples are (prompt text, image or list of images, completion). The donors
dict can also hold plain text donors and CachedDonors; they read the prompt
text only. save_taps and load_taps from wsp.tap work unchanged, since they
store only the branches.

Visibility. MultimodalDonor.lookup returns memory_ends in the same form as
wsp.tap.run_donor, against the prompt text the host reads:
- Image tokens and template tokens before the prompt (system turn, user
  header) get end -1 and valid True. wsp.tap.build_visibility compares
  memory end <= host end, so they are visible to every real host token.
  Host pads and special tokens (end -1) still see nothing.
- Prompt text tokens get their character end in the prompt, so a host
  token reads them only once it has reached that character.
- Template tokens after the prompt (end of turn, assistant header) get
  end len(prompt): visible from the last prompt character on.
- Pads get end -1 and valid False.
The images must come before the prompt text in the template (the default
here). Fallback: when the processor's tokenizer gives no offsets, or the
templated text does not hold the prompt verbatim, or the donor tokens after
the images do not match a plain tokenization of the templated text, every
real donor token in that row gets end -1, so all donor tokens are visible
to all host tokens. MultimodalDonor.fallback_rows lists those rows of the
last lookup.
"""

import contextlib
import time

import torch

from wsp.parts import find_decoder_layers
from wsp.tap import (_StopDonor, build_visibility, clear_tap_context, donor_requests, encode_pairs, flat_params,
                     host_device, host_token_ends, is_cached_donor, run_donor_layers, set_tap_context,
                     train_visibility)

LETTERS = "ABCDE"


def image_list(images):
    """Return images as a list: None gives [], a list or tuple is copied, one image gives [image]."""
    if images is None:
        return []
    if isinstance(images, (list, tuple)):
        return list(images)
    return [images]


class MultimodalDonor:
    """A frozen VLM used as a tap donor. See the module docstring for visibility.

    model: the VLM (for example AutoModelForImageTextToText). processor: its
    processor, with apply_chat_template, a tokenizer attribute, and
    __call__(text=..., images=..., padding=True, return_tensors="pt").
    read_layer: the default language model layer for memory() (0-based).
    system: an optional system message. format_text: an optional function
    (prompt, number of images) -> donor text, used in place of the chat
    template. image_token_id: the image placeholder id, found from the
    model config or the processor when None.
    """

    def __init__(self, model, processor, read_layer=None, system=None, format_text=None, image_token_id=None):
        self.model = model
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.read_layer = read_layer
        self.system = system
        self.format_fn = format_text
        self.image_token_id = image_token_id
        if self.image_token_id is None:
            self.image_token_id = find_image_token_id(model, processor, self.tokenizer)
        self.fallback_rows = []
        self.image_mask = None

    def __repr__(self):
        return f"MultimodalDonor({type(self.model).__name__}, read_layer={self.read_layer})"

    @property
    def hidden_size(self):
        """The width of the language model states, for TapBranch d_donor."""
        embeddings = self.model.get_input_embeddings()
        return int(embeddings.weight.shape[1])

    @property
    def layers_total(self):
        _, layers = find_decoder_layers(self.model, 0)
        return len(layers)

    def format_text(self, prompt, image_count):
        """Return the donor text for one prompt: the chat template, images first, then the prompt."""
        if self.format_fn is not None:
            return self.format_fn(prompt, image_count)
        content = []
        for index in range(image_count):
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        messages = []
        if self.system is not None:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system}]})
        messages.append({"role": "user", "content": content})
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def encode(self, items):
        """Run the processor on (prompt, images) items, right padded when the processor allows it.

        Returns (encoded, ends, valid). encoded holds the processor outputs
        as tensors. ends and valid are (batch, m), as described in the
        module docstring.
        """
        texts = []
        prompts = []
        all_images = []
        for prompt, images in items:
            images = image_list(images)
            prompts.append(prompt)
            texts.append(self.format_text(prompt, len(images)))
            for image in images:
                all_images.append(image)

        tokenizer = self.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        old_side = tokenizer.padding_side
        tokenizer.padding_side = "right"
        kwargs = {"text": texts, "padding": True, "return_tensors": "pt"}
        if len(all_images) > 0:
            kwargs["images"] = all_images
        try:
            # Some processors take the padding side from the tokenizer's
            # saved settings, so we also ask for right padding by name.
            try:
                encoded = self.processor(padding_side="right", **kwargs)
            except TypeError:
                encoded = self.processor(**kwargs)
        finally:
            tokenizer.padding_side = old_side

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        valid = attention_mask == 1
        ends = torch.full(input_ids.shape, -1, dtype=torch.long)
        self.fallback_rows = []
        for row in range(len(texts)):
            real_ids = input_ids[row][valid[row]].tolist()
            row_ends = self.row_ends(texts[row], prompts[row], real_ids)
            if row_ends is None:
                # Fallback: every donor token is visible to every host token.
                self.fallback_rows.append(row)
                continue
            # The real positions in order; this works for either padding side.
            positions = valid[row].nonzero().flatten()
            ends[row, positions] = torch.tensor(row_ends, dtype=torch.long)
        ends[~valid] = -1
        if self.image_token_id is not None:
            self.image_mask = (input_ids == self.image_token_id) & valid
        else:
            self.image_mask = None
        return encoded, ends, valid

    def row_ends(self, text, prompt, real_ids):
        """Return the end offset of each real donor token in one row, or None for the fallback.

        We tokenize the templated text (before image expansion) with
        offsets. The tokens from the first prompt token to the end hold no
        image placeholder, so they must equal the last tokens of the
        processor row. Earlier tokens (template and images) get -1.
        """
        if not getattr(self.tokenizer, "is_fast", False):
            return None
        start = text.rfind(prompt)
        if start < 0 or len(prompt) == 0:
            return None
        stop = start + len(prompt)
        try:
            plain = self.tokenizer(text, return_offsets_mapping=True)
        except (NotImplementedError, ValueError, TypeError):
            return None
        ids = list(plain["input_ids"])
        offsets = plain["offset_mapping"]
        first = None
        for index in range(len(ids)):
            begin, end = offsets[index]
            if end > start and begin < stop and end > begin:
                first = index
                break
        if first is None:
            return None
        tail = ids[first:]
        if len(tail) > len(real_ids) or real_ids[len(real_ids) - len(tail):] != tail:
            return None

        row_ends = [-1] * len(real_ids)
        shift = len(real_ids) - len(tail)
        for k in range(len(tail)):
            begin, end = offsets[first + k]
            if end > start and begin < stop and end > begin:
                end_in_prompt = end
                if end_in_prompt > stop:
                    end_in_prompt = stop
                row_ends[shift + k] = end_in_prompt - start
            else:
                # A template token after the prompt.
                row_ends[shift + k] = len(prompt)
        return row_ends

    def model_inputs(self, encoded):
        """Move the processor outputs to the model device; floating tensors get the model dtype."""
        param = next(self.model.parameters())
        inputs = {}
        for name in encoded:
            value = encoded[name]
            if not torch.is_tensor(value):
                inputs[name] = value
            elif torch.is_floating_point(value):
                inputs[name] = value.to(device=param.device, dtype=param.dtype)
            else:
                inputs[name] = value.to(param.device)
        return inputs

    def lookup(self, items, layers=None, device=None):
        """Return (memories, ends, valid) like wsp.tap.run_donor_layers, for (prompt, images) items.

        The VLM runs its vision encoder and its language model up to the
        deepest layer in layers, then stops. memories is a dict of layer ->
        (batch, m, d) float32.
        """
        if layers is None:
            if self.read_layer is None:
                raise ValueError("give the layers to read, or set read_layer")
            layers = [self.read_layer]
        encoded, ends, valid = self.encode(items)
        wanted = sorted(set(layers))
        last = wanted[-1]
        _, donor_layers = find_decoder_layers(self.model, last)
        store = {}
        handles = []
        for layer in wanted:

            def hook(module, inputs, output, layer=layer):
                if isinstance(output, tuple):
                    output = output[0]
                store[layer] = output.detach().to(torch.float32)
                if layer == last:
                    raise _StopDonor()

            handles.append(donor_layers[layer].register_forward_hook(hook))
        try:
            with torch.no_grad():
                self.model(**self.model_inputs(encoded), use_cache=False)
        except _StopDonor:
            pass
        finally:
            for handle in handles:
                handle.remove()
        if last not in store:
            raise RuntimeError(f"donor layer {last} did not run")
        if device is not None:
            for layer in store:
                store[layer] = store[layer].to(device)
            ends = ends.to(device)
            valid = valid.to(device)
        return store, ends, valid

    def memory(self, items, layer=None, device=None):
        """Return (memory, memory_ends, valid) for one layer, like wsp.tap.run_donor."""
        if layer is None:
            layer = self.read_layer
        memories, ends, valid = self.lookup(items, [layer], device=device)
        return memories[layer], ends, valid

    def last_logits(self, items):
        """Run the whole VLM and return the next-token logits after each row, (batch, vocab) float32."""
        encoded, ends, valid = self.encode(items)
        with torch.no_grad():
            out = self.model(**self.model_inputs(encoded), use_cache=False)
        logits = out.logits
        rows = []
        for row in range(len(items)):
            last = int(valid[row].nonzero().flatten()[-1])
            rows.append(logits[row, last].float())
        return torch.stack(rows).cpu()


def find_image_token_id(model, processor, tokenizer):
    """Return the image placeholder token id from the model config or the processor, or None."""
    config = getattr(model, "config", None)
    for name in ["image_token_id", "image_token_index"]:
        value = getattr(config, name, None)
        if isinstance(value, int):
            return value
    token = getattr(processor, "image_token", None)
    if isinstance(token, str):
        found = tokenizer.convert_tokens_to_ids(token)
        if isinstance(found, int) and found != tokenizer.unk_token_id:
            return found
    return None


def is_multimodal_donor(source):
    return isinstance(source, MultimodalDonor)


def mismatch_images(items):
    """Return items with the images rolled by one: item i gets the images of item i - 1.

    Each item is a tuple whose second field is the images, or a dict with
    an "images" key. The other fields stay. This is the mismatched-image
    control.
    """
    if len(items) < 2:
        raise ValueError("mismatch_images needs at least 2 items")
    out = []
    for index in range(len(items)):
        previous = items[index - 1]
        if isinstance(items[index], dict):
            item = dict(items[index])
            item["images"] = previous["images"]
            out.append(item)
        else:
            item = list(items[index])
            item[1] = previous[1]
            out.append(tuple(item))
    return out


def donor_memories_mm(host_model, donors, items, derange=False):
    """Like wsp.tap.donor_memories, for (prompt, images) items.

    A MultimodalDonor reads the prompt and the images. A (model,
    tokenizer) pair or a CachedDonor reads the prompt text only. Each
    donor runs once, up to the deepest layer its taps need. derange=True
    rolls each memory by one along the batch.
    """
    requests = donor_requests(host_model)
    layers_by_donor = {}
    for donor_id, donor_layer in requests:
        if donor_id not in donors:
            raise KeyError(f"no donor named {donor_id!r}; donors given: {sorted(donors)}")
        if donor_id not in layers_by_donor:
            layers_by_donor[donor_id] = []
        layers_by_donor[donor_id].append(donor_layer)
    if derange and len(items) < 2:
        raise ValueError("derange needs at least 2 prompts in a batch")

    prompts = []
    for prompt, images in items:
        prompts.append(prompt)
    device = host_device(host_model)
    stats = host_model.wsp_tap_stats
    runs = {}
    for donor_id in layers_by_donor:
        source = donors[donor_id]
        layers = layers_by_donor[donor_id]
        start = time.time()
        if is_multimodal_donor(source):
            memories, ends, valid = source.lookup(items, layers, device=device)
            note = {"layers_ran": max(layers) + 1, "layers_total": source.layers_total, "multimodal": True,
                    "fallback_rows": len(source.fallback_rows)}
        elif is_cached_donor(source):
            memories, ends, valid = source.lookup(prompts, layers, device=device)
            note = {"layers_ran": max(layers) + 1, "layers_total": source.layers_total, "cached": True}
        else:
            donor_model, donor_tokenizer = source
            memories, ends, valid = run_donor_layers(donor_model, donor_tokenizer, prompts, layers)
            _, all_layers = find_decoder_layers(donor_model, 0)
            note = {"layers_ran": max(layers) + 1, "layers_total": len(all_layers)}
        stats["donor_seconds"] += time.time() - start
        stats["donor_runs"] += 1
        stats["donor_tokens"] += int(valid.sum())
        stats["donors"][donor_id] = note
        runs[donor_id] = (memories, ends, valid)

    out = []
    for donor_id, donor_layer in requests:
        memories, ends, valid = runs[donor_id]
        memory = memories[donor_layer]
        if derange:
            memory = torch.roll(memory, shifts=1, dims=0)
            ends = torch.roll(ends, shifts=1, dims=0)
            valid = torch.roll(valid, shifts=1, dims=0)
        out.append((memory, ends, valid))
    return out


def has_taps(host_model):
    return getattr(host_model, "wsp_taps", None) is not None


def prepare_mm(host_model, host_tokenizer, donors, items, derange=False):
    """Run the donors on (prompt, images) items, set the tap context, and return the host inputs.

    The host reads only the prompt text. Returns input_ids and
    attention_mask on the host device, padded with the host tokenizer's
    own padding side. A host without taps gets plain inputs.
    """
    prompts = []
    for prompt, images in items:
        prompts.append(prompt)
    input_ids, attention_mask, ends = host_token_ends(host_tokenizer, prompts)
    if has_taps(host_model):
        entries = []
        for memory, memory_ends, memory_valid in donor_memories_mm(host_model, donors, items, derange):
            visibility = build_visibility(ends, memory_ends, memory_valid)
            entries.append((memory, memory_valid, visibility))
        set_tap_context(host_model, entries)
    device = host_device(host_model)
    return {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}


def train_batch_mm(host_model, host_tokenizer, donors, batch, seq_len, derange=False):
    """Encode (prompt, images, completion) examples and set the tap context.

    Returns input_ids, attention_mask, and labels on the host device. The
    labels keep only completion tokens and the EOS (wsp.tap.encode_pairs).
    """
    pairs = []
    items = []
    prompt_chars = []
    for prompt, images, completion in batch:
        pairs.append((prompt, completion))
        items.append((prompt, images))
        prompt_chars.append(len(prompt))
    input_ids, attention_mask, labels, ends = encode_pairs(host_tokenizer, pairs, seq_len)
    if int((labels != -100).sum()) == 0:
        raise ValueError("no completion tokens fit in seq_len for this batch")
    if has_taps(host_model):
        prompt_chars = torch.tensor(prompt_chars, dtype=torch.long)
        entries = []
        for memory, memory_ends, memory_valid in donor_memories_mm(host_model, donors, items, derange):
            visibility = train_visibility(ends, prompt_chars, memory_ends, memory_valid)
            entries.append((memory, memory_valid, visibility))
        set_tap_context(host_model, entries)
    device = host_device(host_model)
    return {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device),
            "labels": labels.to(device)}


def donor_modules(donors):
    """Return the donor models in a donors dict (CachedDonors have none)."""
    models = []
    for donor_id in donors:
        source = donors[donor_id]
        if is_multimodal_donor(source):
            models.append(source.model)
        elif is_cached_donor(source):
            continue
        else:
            models.append(source[0])
    return models


def tap_train_mm(host_model, host_tokenizer, donors, examples, params, steps=300, lr=1e-4, batch_size=8,
                 seq_len=384, seed=0, derange=False, mismatch=False, log_every=10):
    """Train only params on (prompt, images, completion) examples. Return the per-step losses.

    The same loop as wsp.tap.tap_train. A host with no taps trains on the
    prompt text alone (the text-only LoRA baseline), with donors unused.
    derange=True rolls the donor memory by one along each batch.
    mismatch=True rolls only the images, so each donor reads the right
    question with another example's image (the mismatched-image control).
    """
    flat = flat_params(params)
    for param in host_model.parameters():
        param.requires_grad_(False)
    for model in donor_modules(donors):
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
    for param in flat:
        param.requires_grad_(True)

    device = host_device(host_model)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    generator = torch.Generator()
    generator.manual_seed(seed)
    use_autocast = device.type == "cuda"

    host_model.train()
    losses = []
    order = []
    try:
        for step in range(steps):
            batch = []
            while len(batch) < batch_size:
                if len(order) == 0:
                    order = torch.randperm(len(examples), generator=generator).tolist()
                batch.append(examples[order.pop()])
            if mismatch:
                batch = mismatch_images(batch)

            inputs = train_batch_mm(host_model, host_tokenizer, donors, batch, seq_len, derange)
            optimizer.zero_grad()
            if use_autocast:
                autocast = torch.autocast("cuda", dtype=torch.bfloat16)
            else:
                autocast = contextlib.nullcontext()
            with autocast:
                out = host_model(**inputs)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(flat, 1.0)
            optimizer.step()

            loss = float(out.loss.detach())
            losses.append(loss)
            if log_every and (step + 1) % log_every == 0:
                print(f"tap_mm step {step + 1}/{steps} loss {loss:.4f}")
    finally:
        if has_taps(host_model):
            clear_tap_context(host_model)
        host_model.eval()
    return losses


def mcq_prompt(question, choices, context=None):
    """Return the multiple-choice prompt the host and the donor read, ending in "Answer:"."""
    lines = []
    if context:
        lines.append("Context: " + str(context).strip())
    lines.append("Question: " + str(question).strip())
    for index in range(len(choices)):
        lines.append(LETTERS[index] + ". " + str(choices[index]).strip())
    return "\n".join(lines) + "\nAnswer:"


def mcq_example(item):
    """Return the training example (prompt, images, " X") of an item dict with prompt, images, gold."""
    return (item["prompt"], item["images"], " " + LETTERS[item["gold"]])


def letter_ids(tokenizer, letters=LETTERS, space=True):
    """Return the last token id of " X" (or "X" when space is False) for each letter."""
    ids = []
    for letter in letters:
        text = letter
        if space:
            text = " " + letter
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids.append(encoded[-1])
    return ids


def choice_record(item, scores):
    """Turn the per-choice scores of one item into a record with probs, pred, and correct."""
    probs = torch.softmax(scores.float(), dim=-1)
    pred = int(probs.argmax())
    rounded = []
    for value in probs.tolist():
        rounded.append(round(value, 5))
    return {"id": item.get("id"), "gold": item["gold"], "pred": pred, "probs": rounded,
            "correct": pred == item["gold"]}


def score_choices(host_model, host_tokenizer, items, donors=None, derange=False, batch_size=16):
    """Score multiple-choice items with the host, by the log-probs of " A", " B", ... after the prompt.

    items: dicts with prompt, images, gold, n_choices (and id). The host
    reads only the prompt. With donors, the taps read the donors on
    (prompt, images). Pass mismatch_images(items) for the control. Returns one record per item.
    """
    old_side = host_tokenizer.padding_side
    host_tokenizer.padding_side = "left"
    if host_tokenizer.pad_token is None:
        host_tokenizer.pad_token = host_tokenizer.eos_token
    all_ids = letter_ids(host_tokenizer)
    use_taps = donors is not None and has_taps(host_model)
    records = []
    try:
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            pairs = []
            for item in batch:
                pairs.append((item["prompt"], item["images"]))
            if use_taps:
                encoded = prepare_mm(host_model, host_tokenizer, donors, pairs, derange=derange)
            else:
                prompts = []
                for item in batch:
                    prompts.append(item["prompt"])
                plain = host_tokenizer(prompts, return_tensors="pt", padding=True)
                device = host_device(host_model)
                encoded = {"input_ids": plain["input_ids"].to(device),
                           "attention_mask": plain["attention_mask"].to(device)}
            with torch.no_grad():
                logits = host_model(**encoded).logits[:, -1, :].float().cpu()
            if use_taps:
                clear_tap_context(host_model)
            for row in range(len(batch)):
                item = batch[row]
                ids = all_ids[:item["n_choices"]]
                records.append(choice_record(item, logits[row, ids]))
    finally:
        host_tokenizer.padding_side = old_side
    return records


def score_vlm_choices(donor, items, batch_size=8, prompt_key="prompt"):
    """Score multiple-choice items with the VLM alone (the ceiling).

    The VLM reads the images and item[prompt_key] in its chat template.
    Each choice scores the log-sum-exp of the next-token log-probs of "X"
    and " X", since after an assistant header the letter often comes with
    no space.
    """
    tokenizer = donor.tokenizer
    bare = letter_ids(tokenizer, space=False)
    spaced = letter_ids(tokenizer, space=True)
    records = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        pairs = []
        for item in batch:
            pairs.append((item[prompt_key], item["images"]))
        logits = donor.last_logits(pairs)
        logprobs = torch.log_softmax(logits, dim=-1)
        for row in range(len(batch)):
            item = batch[row]
            scores = []
            for index in range(item["n_choices"]):
                forms = [logprobs[row, bare[index]]]
                if spaced[index] != bare[index]:
                    forms.append(logprobs[row, spaced[index]])
                scores.append(torch.logsumexp(torch.stack(forms), dim=0))
            records.append(choice_record(item, torch.stack(scores)))
    return records


def accuracy(records):
    if len(records) == 0:
        return 0.0
    correct = 0
    for record in records:
        correct += int(record["correct"])
    return correct / len(records)
