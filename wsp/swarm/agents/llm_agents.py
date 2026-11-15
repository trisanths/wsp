"""LLM analyst and planners, through ctx["llm"] (the LLM contract in interfaces.py).

- LLMAnalyst starts from the scripted analyst's map, asks the LLM for a
  capability map, checks it against the catalog and merges it in. Any LLM
  failure keeps the scripted map and logs an event.
- LLMPlanner samples two parents from the frontier, recalls two angle
  queries from memory (level "low") on top of the swarm's round recall, reads
  the frozen literature priors, and asks the LLM for N single-edit proposals
  (prompts.planner_messages). Replies are parsed and checked with
  space.validate plus the planner rules below. Invalid output gets one
  repair retry with the errors and never reaches the env, so it never costs
  an eval. Any LLM failure returns an empty list and logs an event, so the
  loop never crashes. Each valid hypothesis gets at most one literature
  search, and the citations ride on the proposal into the log row.
- PlannerBatch sends the planners' prompts of one round as a single
  complete_many call. The factory gives every LLM planner one shared batch.

Planner rules beyond space.validate (the language spec states each one):
at least one graft, seed 0, control null, glue_rank from the allowed list.

Seeds: every LLM call carries a seed made from the run seed and the round,
and the planners' own choices use ctx["rng"], so a run with a deterministic
LLM repeats exactly.
"""

import json
import math
import threading
import time

from wsp.swarm import program_schema as ps
from wsp.swarm import prompts, space
from wsp.swarm.agents.base import Analyst, Planner
from wsp.swarm.agents.scripted import ScriptedAnalyst, main_task

DIRECTIVES = [
    "exploit: make small, careful edits to the best parents",
    "explore: try parts and donors with few or no evals so far, at depths with good fit",
    "placement fixer: move grafts to host layers with better fit and space out grafts that interfere",
]

# Edit menu with fixed odds per proposal slot (papers.md 7.2 and 8.5; the
# host-mask edit is left out because planners do not get mask cards).
EDIT_MENU = [("swap part", 0.25), ("move at", 0.20), ("add graft", 0.15), ("drop graft", 0.10),
             ("change heal steps", 0.10), ("change glue rank", 0.10), ("fresh draft", 0.10)]

PARENT_SHARPNESS = 10.0   # sigmoid slope of the parent weights
RECALL_LIMIT = 20
HYPOTHESIS_CHARS = 300
SEARCH_K = 3
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048
DEFAULT_BATCH_WAIT_S = 30.0


# Calling the LLM.

def llm_settings(config, round_index, offset):
    """Sampling settings for one call. The seed comes from the run seed, the round and an offset."""
    temperature = config.get("llm_temperature")
    if temperature is None:
        temperature = DEFAULT_TEMPERATURE
    max_tokens = config.get("llm_max_tokens")
    if max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS
    seed = config["seed"] * 100003 + round_index * 101 + offset
    return {"temperature": temperature, "seed": seed, "max_tokens": max_tokens, "schema": None}


def replies_from_error(error, count):
    """The raw replies a JSON error carries, or None.

    The clients in llm.py raise LLMJSONError, with the raw replies in
    .texts, when a reply asked for with a schema is not strict JSON (for
    example JSON inside a code fence). The planner parses and repairs those
    replies itself, so they are recovered instead of lost.
    """
    texts = getattr(error, "texts", None)
    if isinstance(texts, list) and len(texts) == count:
        return list(texts)
    return None


def call_one(llm, messages, settings):
    """One complete() call with the given settings. Returns the reply text."""
    try:
        return llm.complete(messages, schema=settings["schema"], temperature=settings["temperature"],
                            seed=settings["seed"], max_tokens=settings["max_tokens"])
    except Exception as error:
        texts = replies_from_error(error, 1)
        if texts is None:
            raise
        return texts[0]


def call_many(llm, list_of_messages, settings):
    """One reply per message list: one complete_many call, or a plain complete for a single list."""
    if len(list_of_messages) == 1 or not hasattr(llm, "complete_many"):
        replies = []
        for messages in list_of_messages:
            replies.append(call_one(llm, messages, settings))
        return replies
    try:
        return llm.complete_many(list_of_messages, schema=settings["schema"], temperature=settings["temperature"],
                                 seed=settings["seed"], max_tokens=settings["max_tokens"])
    except Exception as error:
        texts = replies_from_error(error, len(list_of_messages))
        if texts is None:
            raise
        return texts


def request_index(request):
    return request["index"]


class PlannerBatch:
    """Sends the planners' prompts of one round as one llm.complete_many call.

    The swarm runs its planners in parallel threads. Each planner calls
    ask() once per round, or skip() when it has nothing to send. When all
    `size` planners have arrived, the last one sends the batch in planner
    order and every planner gets its own reply. If a planner has not arrived
    after wait_s seconds, the waiting planners send what has arrived. With
    size 1 every ask is a plain single call. The sender holds the lock during
    the LLM call; the other planners of the batch are waiting on it anyway.
    Planners called one after another (not in threads) with a shared batch
    of size > 1 still work, but each waits wait_s before sending alone.
    """

    def __init__(self, size=1, wait_s=DEFAULT_BATCH_WAIT_S):
        self.size = max(1, int(size))
        self.wait_s = wait_s
        self.cond = threading.Condition()
        self.requests = []
        self.arrived = 0
        self.sent = []   # one entry per LLM call: the number of message lists it carried

    def ask(self, index, llm, messages, settings):
        """Queue one planner's messages and return its reply text. Raises what the LLM raised."""
        request = {"index": index, "llm": llm, "messages": messages, "settings": settings,
                   "done": False, "reply": None, "error": None}
        with self.cond:
            self.requests.append(request)
            self.arrive()
            deadline = time.monotonic() + self.wait_s
            while not request["done"]:
                left = deadline - time.monotonic()
                if left <= 0:
                    self.send()
                    break
                self.cond.wait(left)
        if request["error"] is not None:
            raise request["error"]
        return request["reply"]

    def skip(self):
        """Count a planner that sends nothing this round, so the others do not wait for it."""
        with self.cond:
            self.arrive()

    def arrive(self):
        """Count one arrival; send when everyone is here. The caller holds the lock."""
        self.arrived += 1
        if self.arrived >= self.size:
            self.send()

    def send(self):
        """Send every waiting request in one call, in planner order. The caller holds the lock."""
        batch = sorted(self.requests, key=request_index)
        self.requests = []
        self.arrived = 0
        if len(batch) == 0:
            return
        lists = []
        for request in batch:
            lists.append(request["messages"])
        first = batch[0]
        replies = None
        error = None
        try:
            replies = call_many(first["llm"], lists, first["settings"])
        except Exception as caught:
            error = caught
        self.sent.append(len(lists))
        for position in range(len(batch)):
            request = batch[position]
            if error is not None:
                request["error"] = error
            elif replies is not None and position < len(replies):
                request["reply"] = replies[position]
            else:
                request["reply"] = ""
            request["done"] = True
        self.cond.notify_all()


# Logging and memory.

def log_event(ctx, kind, text, role, round_index, extra=None):
    """Write an event row through the env. Never raises."""
    env = ctx.get("env")
    if env is None:
        return
    try:
        env.log_event(kind, text, role, round_index, extra)
    except Exception:
        pass


def error_text(error):
    return type(error).__name__ + ": " + prompts.clean_text(str(error), 300)


def recall(ctx, role, query):
    """memory.recall at level low. Returns [] on any failure."""
    memory = ctx.get("memory")
    if memory is None:
        return []
    try:
        found = memory.recall(role, query, level="low", limit=RECALL_LIMIT)
    except Exception:
        return []
    if found is None:
        return []
    return list(found)


def merge_lessons(first, second):
    """Both lists of lessons, without repeats and without past.dev "mention" items."""
    out = []
    seen = set()
    for lesson in list(first or []) + list(second or []):
        if not isinstance(lesson, dict):
            continue
        if lesson.get("kind") == "mention":
            continue
        key = lesson.get("source_id") or lesson.get("text")
        if key in seen:
            continue
        seen.add(key)
        out.append(lesson)
    return out


def part_sizes(log):
    """Part parameters by part id, from the compile reports already in the run log."""
    sizes = {}
    if log is None:
        return sizes
    for row in log.all_rows():
        for entry in row.get("graft_report") or []:
            if entry.get("control") is not None or entry.get("part_params") is None:
                continue
            if entry.get("id") not in sizes:
                sizes[entry["id"]] = entry["part_params"]
    return sizes


def ranked_part_ids(observation, cap_map):
    ids = []
    for record in prompts.ordered_parts(observation, cap_map):
        ids.append(record["id"])
    return ids


# Parsing and checking replies.

def parse_json(text):
    """Read the JSON object in a reply (a code fence around it is fine). Returns (data, error)."""
    if not isinstance(text, str) or text.strip() == "":
        return None, "the reply was empty"
    body = text.strip()
    start = body.find("{")
    end = body.rfind("}")
    if start < 0 or end <= start:
        return None, "the reply has no JSON object"
    try:
        data = json.loads(body[start:end + 1])
    except ValueError as error:
        return None, "the reply is not valid JSON: " + str(error)
    if not isinstance(data, dict):
        return None, "the reply must be a JSON object"
    return data, None


def planner_rule_errors(program, config):
    """The planner rules beyond space.validate. program is already normalized."""
    errors = []
    if len(ps.grafts(program)) == 0:
        errors.append("a proposal needs at least one graft")
    if ps.seed(program) != 0:
        errors.append("seed must be 0, got " + repr(ps.seed(program)))
    allowed = prompts.glue_ranks_allowed(config)
    for graft in ps.grafts(program):
        if ps.graft_control(graft) is not None:
            errors.append(str(ps.graft_id(graft)) + ": control must be null")
        if graft.get("glue_rank") not in allowed:
            errors.append(str(ps.graft_id(graft)) + ": glue_rank " + repr(graft.get("glue_rank")) +
                          " not in " + prompts.ranks_text(allowed))
    return errors


def program_errors(program, catalog, config):
    """space.validate plus the planner rules. A program the checks cannot read is one error, not a crash."""
    try:
        errors = list(space.validate(program, catalog, config))
        if len(errors) == 0:
            errors.extend(planner_rule_errors(ps.normalize(program), config))
    except Exception as error:
        return ["could not read program: " + error_text(error)]
    return errors


def item_errors(item, catalog, config):
    """Every problem with one proposal item, as plain strings."""
    if not isinstance(item, dict):
        return ["each proposal must be a JSON object"]
    program = item.get("program")
    if not isinstance(program, dict):
        return ["program must be a JSON object"]
    errors = program_errors(program, catalog, config)
    hypothesis = item.get("hypothesis")
    if not isinstance(hypothesis, str) or hypothesis.strip() == "":
        errors.append("hypothesis must be a non-empty string")
    return errors


def fit_unknown(program, observation, catalog, config):
    """True when any graft sits in a slot whose fit has not been measured."""
    for graft in ps.grafts(program):
        slot = space.graft_slot(graft, catalog, config)
        if slot is None or slot not in observation["known_fits"]:
            return True
    return False


def citation_for(value, literature):
    """A cite dict for one cited value: a label such as "L2", a URL, or a cite dict."""
    if isinstance(value, dict):
        url = str(value.get("url") or "")
        if url == "":
            return None
        return {"url": url, "title": str(value.get("title") or ""), "quote": str(value.get("quote") or "")}
    if not isinstance(value, str) or value.strip() == "":
        return None
    text = value.strip().strip("[]")
    for item in literature:
        if text == item["label"] or (item["url"] != "" and text == item["url"]):
            return {"url": item["url"], "title": item["title"], "quote": item["quote"]}
    if text.startswith("http://") or text.startswith("https://"):
        return {"url": text, "title": "", "quote": ""}
    return None


def add_citation(citations, cite):
    if cite is None:
        return
    for existing in citations:
        if existing["url"] == cite["url"] and existing["quote"] == cite["quote"]:
            return
    citations.append(cite)


def resolve_citations(values, literature):
    citations = []
    if not isinstance(values, list):
        return citations
    for value in values:
        add_citation(citations, citation_for(value, literature))
    return citations


def shared_grafts(program, parent_program):
    parent_ids = []
    for graft in ps.grafts(parent_program):
        parent_ids.append(ps.graft_id(graft))
    count = 0
    for graft in ps.grafts(program):
        if ps.graft_id(graft) in parent_ids:
            count += 1
    return count


def closest_parent(program, parents):
    """The hash of the parent sharing the most graft ids with the program, or None for a fresh draft."""
    best = None
    best_count = 0
    for parent in parents:
        count = shared_grafts(program, parent["program"])
        if count > best_count:
            best = parent
            best_count = count
    if best is None:
        return None
    return best["program_hash"]


def make_proposal(item, program, context):
    """The proposal dict the swarm expects, from one checked reply item."""
    observation = context["observation"]
    probe_first = item.get("probe_first") is True
    if fit_unknown(program, observation, context["catalog"], context["config"]):
        probe_first = True
    return {"program": program,
            "hypothesis": prompts.clean_text(item["hypothesis"], HYPOTHESIS_CHARS),
            "citations": resolve_citations(item.get("citations"), context["literature"]),
            "probe_first": probe_first,
            "parent": closest_parent(program, context["parents"])}


def check_reply(reply, context):
    """Parse a planner reply. Returns (valid proposals, errors). Never raises on bad input."""
    data, error = parse_json(reply)
    if error is not None:
        return [], [error]
    items = data.get("proposals")
    if not isinstance(items, list):
        return [], ["the reply must be {\"proposals\": [...]}"]
    proposals = []
    errors = []
    seen = set()
    for position in range(len(items)):
        if len(proposals) >= context["n"]:
            break
        item = items[position]
        problems = item_errors(item, context["catalog"], context["config"])
        if len(problems) > 0:
            for problem in problems:
                errors.append("proposal " + str(position + 1) + ": " + problem)
            continue
        try:
            program = ps.normalize(item["program"])
            program_hash = space.program_hash(program)
            proposal = make_proposal(item, program, context)
        except Exception as error:
            errors.append("proposal " + str(position + 1) + ": could not read program: " + error_text(error))
            continue
        if program_hash in seen:
            continue
        seen.add(program_hash)
        proposals.append(proposal)
    if len(items) == 0:
        errors.append("the reply has no proposals")
    return proposals, errors


def merge_proposals(first, second, n):
    """First's proposals, then second's new ones, at most n, one per program hash."""
    out = []
    seen = set()
    for proposal in list(first) + list(second):
        program_hash = space.program_hash(proposal["program"])
        if program_hash in seen or len(out) >= n:
            continue
        seen.add(program_hash)
        out.append(proposal)
    return out


# Parents and edit slots.

def child_counts(history):
    counts = {}
    for row in history:
        parent = row.get("parent")
        if parent is not None:
            counts[parent] = counts.get(parent, 0) + 1
    return counts


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def sigmoid(value):
    if value < -60:
        return 0.0
    if value > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))


def parent_weights(pool, history):
    """ShinkaEvolve-style weights: sigmoid of fitness around the median, times 1 / (1 + children)."""
    scores = []
    for row in pool:
        scores.append(row["combined"])
    middle = median(scores)
    children = child_counts(history)
    weights = []
    for row in pool:
        weight = sigmoid(PARENT_SHARPNESS * (row["combined"] - middle))
        weight = weight / (1.0 + children.get(row["program_hash"], 0))
        weights.append(max(weight, 1e-6))
    return weights


def sample_parents(observation, ctx, count=2):
    """Up to count distinct frontier rows (retired parents left out), sampled with parent_weights."""
    pool = []
    for row in observation["frontier"]:
        if row["program_hash"] not in ctx.get("retired", []) and row.get("combined") is not None:
            pool.append(row)
    if len(pool) == 0:
        return []
    weights = parent_weights(pool, observation["history"])
    rng = ctx["rng"]
    chosen = []
    while len(chosen) < count and len(pool) > 0:
        pick = rng.choices(range(len(pool)), weights=weights)[0]
        chosen.append(pool.pop(pick))
        weights.pop(pick)
    return chosen


def edit_slots(n, parents, rng):
    """One edit name per proposal slot, drawn from EDIT_MENU; all fresh drafts without parents."""
    slots = []
    for index in range(n):
        if len(parents) == 0:
            slots.append("fresh draft")
            continue
        roll = rng.random()
        total = 0.0
        name = EDIT_MENU[-1][0]
        for edit, rate in EDIT_MENU:
            total += rate
            if roll < total:
                name = edit
                break
        slots.append(name)
    return slots


def angle_parts(parents, observation, cap_map, catalog, config):
    """The parts a planner's angle queries ask about: its parents' grafts, else the map's top parts."""
    names = []
    depths = []
    for parent in parents:
        for graft in ps.grafts(parent["program"]):
            name = space.short_name(ps.graft_id(graft))
            if name not in names:
                names.append(name)
            at = space.resolved_at(graft, catalog, config)
            if at is not None:
                depths.append(at)
    if len(names) == 0:
        for part_id in ranked_part_ids(observation, cap_map)[:3]:
            names.append(space.short_name(part_id))
    return names[:4], depths


def angle_queries(parents, observation, cap_map, catalog, config):
    """Two recall queries (pastdev.md 6.3): what happened with these parts here, and their collapses."""
    names, depths = angle_parts(parents, observation, cap_map, catalog, config)
    where = "at their default depths"
    if len(depths) > 0:
        where = "at host layers " + str(min(depths)) + " to " + str(max(depths))
    parts = ", ".join(names)
    return ["what happened with " + parts + " " + where,
            "collapses and fit rejections involving " + parts + " or nearby host layers"]


def search_query(proposal, catalog):
    """The literature query for one hypothesis: the hypothesis plus the grafted parts' donors and kinds."""
    parts = []
    for graft in ps.grafts(proposal["program"]):
        record = catalog.records.get(ps.graft_id(graft)) or {}
        parts.append(str(record.get("model")) + " " + str(record.get("type")) + " " + prompts.layer_text(record))
    return proposal["hypothesis"] + " Parts: " + "; ".join(parts)


def add_searched_citations(proposals, ctx, role, catalog):
    """At most one literature search per hypothesis; found cites join the proposal's citations."""
    literature = ctx.get("literature")
    if literature is None or getattr(literature, "mode", "off") == "off":
        return 0
    searches = 0
    for proposal in proposals:
        searches += 1
        try:
            found = literature.search(role, search_query(proposal, catalog), k=SEARCH_K)
        except Exception:
            found = []
        for cite in found or []:
            add_citation(proposal["citations"], citation_for(cite, []))
    return searches


# The planner.

class LLMPlanner(Planner):
    def __init__(self, index, config=None, batch=None):
        self.index = index
        self.name = "planner-" + str(index + 1)
        self.directive = DIRECTIVES[index % len(DIRECTIVES)]
        if batch is None:
            batch = PlannerBatch(1)
        self.batch = batch
        self.calls = 0

    def propose(self, observation, cap_map, lessons, priors, n, ctx):
        """Ask the LLM for n single-edit proposals. Returns [] on any failure (and logs an event)."""
        self.calls += 1
        round_index = self.calls - 1
        try:
            context = self.prepare(observation, cap_map, lessons, priors, n, ctx)
        except Exception as error:
            self.batch.skip()
            log_event(ctx, "llm_error", self.name + " could not build its prompt: " + error_text(error),
                      self.name, round_index)
            return []
        settings = llm_settings(ctx["config"], round_index, 0)
        settings["schema"] = context["schema"]
        try:
            reply = self.batch.ask(self.index, ctx["llm"], context["messages"], settings)
        except Exception as error:
            log_event(ctx, "llm_error", self.name + " LLM call failed, no proposals this round: " +
                      error_text(error), self.name, round_index)
            return []
        try:
            return self.finish(reply, context, ctx, round_index)
        except Exception as error:
            log_event(ctx, "llm_error", self.name + " could not read the reply: " + error_text(error),
                      self.name, round_index)
            return []

    def prepare(self, observation, cap_map, lessons, priors, n, ctx):
        """Parents, recalls, literature and edit slots, then the prompt. Returns the context dict."""
        config = ctx["config"]
        env = ctx["env"]
        catalog = env.catalog
        parents = sample_parents(observation, ctx)
        recalled = []
        for query in angle_queries(parents, observation, cap_map, catalog, config):
            recalled.extend(recall(ctx, self.name, query))
        context = {
            "role": self.name,
            "directive": self.directive,
            "n": n,
            "slots": edit_slots(n, parents, ctx["rng"]),
            "observation": observation,
            "cap_map": cap_map,
            "lessons": merge_lessons(lessons, recalled),
            "literature": prompts.literature_items(priors, ranked_part_ids(observation, cap_map)),
            "parents": parents,
            "counters": env.budget.counters(),
            "sizes": part_sizes(ctx.get("log")),
            "config": config,
            "catalog": catalog,
            "schema": prompts.proposal_schema(config, catalog, n),
        }
        context["messages"] = prompts.planner_messages(context)
        return context

    def finish(self, reply, context, ctx, round_index):
        """Check the reply, repair once if needed, attach searched citations, log a summary."""
        if not isinstance(reply, str) or reply.strip() == "":
            log_event(ctx, "llm_error", self.name + " got an empty reply, no proposals this round",
                      self.name, round_index)
            return []
        proposals, errors = check_reply(reply, context)
        repaired = False
        repair_errors = []
        if len(errors) > 0:
            repaired = True
            more, repair_errors = self.repair(reply, errors, context, ctx, round_index)
            proposals = merge_proposals(proposals, more, context["n"])
        searches = add_searched_citations(proposals, ctx, self.name, context["catalog"])
        text = (self.name + ": " + str(len(proposals)) + " valid proposals of " + str(context["n"]) + " asked")
        if repaired:
            text += ", repaired once after " + str(len(errors)) + " errors"
        log_event(ctx, "llm_plan", text, self.name, round_index,
                  {"proposals": len(proposals), "errors": errors[:10], "repair_errors": repair_errors[:10],
                   "repaired": repaired, "searches": searches})
        return proposals

    def repair(self, reply, errors, context, ctx, round_index):
        """The one repair retry. Returns (valid proposals, errors left)."""
        messages = prompts.repair_messages(context["messages"], reply, errors)
        settings = llm_settings(ctx["config"], round_index, self.index + 1)
        settings["schema"] = context["schema"]
        try:
            second = call_one(ctx["llm"], messages, settings)
        except Exception as error:
            log_event(ctx, "llm_error", self.name + " repair call failed: " + error_text(error),
                      self.name, round_index)
            return [], ["repair call failed"]
        return check_reply(second, context)


# The analyst.

def is_known_id(value, known):
    """True when value is one of the known part ids. A list or object id is never known."""
    try:
        return value in known
    except TypeError:
        return False


def capmap_errors(data, observation, n_host):
    """Problems with an analyst reply, as plain strings. Never raises on bad input."""
    errors = []
    known = set()
    for record in observation["parts"]:
        known.add(record["id"])
    if not isinstance(data.get("text"), str):
        errors.append("text must be a string")
    by_task = data.get("by_task")
    if not isinstance(by_task, dict):
        errors.append("by_task must be an object of task -> [part ids]")
    else:
        for task in sorted(by_task):
            if task not in observation["tasks"]:
                errors.append("by_task has unknown task " + str(task))
                continue
            if not isinstance(by_task[task], list):
                errors.append("by_task." + str(task) + " must be a list")
                continue
            for part_id in by_task[task]:
                if not is_known_id(part_id, known):
                    errors.append("unknown part id " + prompts.clean_text(str(part_id), 120))
    placements = data.get("placements")
    if not isinstance(placements, dict):
        errors.append("placements must be an object of part id -> [host layers]")
    else:
        for part_id in sorted(placements):
            if not is_known_id(part_id, known):
                errors.append("placements has unknown part id " + str(part_id))
                continue
            values = placements[part_id]
            if not isinstance(values, list):
                errors.append("placements." + str(part_id) + " must be a list")
                continue
            for at in values:
                if isinstance(at, bool) or not isinstance(at, int) or at < 0 or at > n_host - 1:
                    errors.append("placements." + str(part_id) + ": host layer " + repr(at) +
                                  " is outside 0.." + str(n_host - 1))
    return errors


def merge_ranking(llm_ids, base_ids):
    out = []
    for part_id in list(llm_ids) + list(base_ids):
        if part_id in base_ids and part_id not in out:
            out.append(part_id)
    return out


def merge_capmap(base, data):
    """The scripted map with the LLM's rankings, placements, text and notes put first."""
    merged = dict(base)
    by_task = {}
    for task in base["by_task"]:
        by_task[task] = merge_ranking((data.get("by_task") or {}).get(task) or [], base["by_task"][task])
    merged["by_task"] = by_task
    placements = {}
    for part_id in base["placements"]:
        chosen = []
        for at in (data.get("placements") or {}).get(part_id) or []:
            if at not in chosen:
                chosen.append(at)
        for at in base["placements"][part_id]:
            if at not in chosen:
                chosen.append(at)
        placements[part_id] = chosen
    merged["placements"] = placements
    merged["text"] = prompts.clean_text(data.get("text"), 1500)
    if merged["text"] == "":
        merged["text"] = base["text"]
    notes = []
    for note in (data.get("notes") or [])[:5]:
        notes.append(prompts.clean_text(note))
    merged["notes"] = notes + list(base["notes"])
    merged["source"] = "llm"
    return merged


class LLMAnalyst(Analyst):
    def __init__(self, config=None):
        self.name = "analyst"
        self.fallback = ScriptedAnalyst(config)
        self.calls = 0

    def build_map(self, observation, ctx):
        """The LLM's capability map merged over the scripted one; the scripted map on any failure."""
        self.calls += 1
        base = self.fallback.build_map(observation, ctx)
        base["source"] = "scripted"
        try:
            return self.ask(observation, base, ctx)
        except Exception as error:
            log_event(ctx, "llm_error", "analyst LLM call failed, using the scripted map: " + error_text(error),
                      self.name, None)
            return base

    def ask(self, observation, base, ctx):
        config = ctx["config"]
        catalog = ctx["env"].catalog
        n_host = space.host_layers(config, catalog)
        lessons = recall(ctx, self.name, "placement notes: which host layers gave good fit_r2 and scores for " +
                         main_task(observation["tasks"]))
        context = {
            "observation": observation,
            "lessons": merge_lessons(lessons, []),
            "literature": prompts.literature_items(ctx["literature"].priors(), ranked_part_ids(observation, base)),
            "sizes": part_sizes(ctx.get("log")),
            "config": config,
            "catalog": catalog,
            "schema": prompts.capmap_schema(observation["tasks"]),
        }
        messages = prompts.analyst_messages(context)
        settings = llm_settings(config, self.calls - 1, 50000)
        settings["schema"] = context["schema"]
        reply = call_one(ctx["llm"], messages, settings)
        data, error = parse_json(reply)
        errors = [error]
        if data is not None:
            errors = capmap_errors(data, observation, n_host)
        if len(errors) > 0 and isinstance(reply, str) and reply.strip() != "":
            settings["seed"] += 1
            reply = call_one(ctx["llm"], prompts.repair_messages(messages, reply, errors), settings)
            data, error = parse_json(reply)
            errors = [error]
            if data is not None:
                errors = capmap_errors(data, observation, n_host)
        if len(errors) > 0:
            log_event(ctx, "llm_error", "analyst reply unusable, using the scripted map: " +
                      prompts.clean_text("; ".join(errors), 300), self.name, None)
            return base
        return merge_capmap(base, data)
