"""LoRA and full fine-tuning arms: the plan for real runs, and STUB runners.

Why. The brief's "versus regular training" comparison: LoRA on the host and
full fine-tuning of the host, trained on the same data as the glue, at the
same eval budget and the same compute seconds as every other arm, compared
on the compute axis (design rule 10, papers.md 8.3).

Built here
- TrainingArm, the driver. It plans a few checkpoints, charges every
  training second to the run's compute axis (ledger phase "train"), spends
  one eval token (1.0 FEE, the full search split) per checkpoint eval, and
  logs one eval row per checkpoint with method "lora" or "full_ft". It ranks
  the checkpoints with the swarm judge's rule and scores the best one once
  on held-out, from the held-out pool. It uses the run's env for the budget,
  the log, the ledger and the host's base scores, so the counters are the
  same as for every other arm. It cannot use env.step, because a training
  run is not a graft program: rows have "program": {} and the training
  setup sits in extra["training"]. The cascade does not apply: every
  checkpoint eval is one full-fidelity eval.
- StubTrainer, a STUB trainer. Instant, deterministic, labeled STUB.

STUB cost model, in seconds, reported and never slept
    lora      0.5 s per optimizer step. Assumption: the stub charges 0.5 s per
              heal step per graft (stub.py) at the same batch 8 and sequence
              384, and a LoRA step does the same work (a forward and backward
              pass through the frozen 1B host, a small trainable set).
    full_ft   1.5 s per optimizer step. Assumption: 3 times a LoRA step, for
              gradients on all 1.24B weights plus AdamW states and the update.
    eval      120 s per checkpoint eval (stub.eval_cost(1.0), as for every arm).
Both per-step numbers are placeholders to be replaced by timings from a
pilot (papers.md 8.3: set the compute budget from a pilot).

STUB learning curve (made-up plumbing, never a result): the target score
rises by gain * (1 - exp(-step / tau)) over the host's base score, a guard
falls by forget * (1 - exp(-step / tau)), plus fixed noise of about 0.01.
LoRA: gain 0.06, tau 400, forget 0.01. Full FT: gain 0.09, tau 800, forget
0.06. These constants were picked so the curves are visible, nothing more.

Checkpoint plan. count = min(train_checkpoints (4), whole tokens left).
Training gets the compute budget minus count evals; total steps =
floor(training seconds / seconds per step); checkpoints sit at
total * i // count for i = 1..count. Before each chunk the driver checks
that the chunk plus its eval fit in the compute left, and shortens the
chunk if not, so the planned run never overshoots the compute axis.

Real runs (not built here)
1. Data and recipe. Train on the heal data, gsm8k_train (GSM8K train minus
   the 200 search items, wsp/data.py gsm8k_splits), with the heal's batch
   size (8) and sequence length (384). Never train on the search or
   held-out split.
2. Tuning is charged. LoRA rank and learning rate, and the full-FT learning
   rate, are chosen on the search split, so every trial's training seconds
   and checkpoint evals come out of the same two budgets (papers.md 8.3).
   The grid is fixed before the run and written to the manifest. Draw the
   arm as an expected-best curve against GPU-seconds (Dodge et al. 2019).
3. Runners, each implementing the Trainer interface below:
   - TinkerLoRA (LoRA only, Tinker API). Check first that Tinker serves this
     host. Tinker does not report GPU-seconds on our GPU type, so for the
     compute axis either time the same steps on the Modal GPU in a pilot and
     charge those seconds, or use the Modal runner for the compute-axis
     comparison and Tinker as a cross-check.
   - ModalTrainer (LoRA through peft, and full FT). One Modal function call
     per chunk of steps on the run's fixed GPU type, checkpoints saved to
     the wsp-swarm-models volume, measured GPU-seconds returned as
     compute_s. Full FT holds weights, gradients and AdamW states at 16
     bytes per parameter, about 20 GB for 1.24B parameters, plus
     activations, so it needs a 40 GB or larger GPU. evaluate() calls the
     same eval harness, split and question set as every other arm.
4. Matched LoRA (the brief's control 2). rank = matched_lora_rank(part_params
   + glue_params) of the program it controls; the env logs that sum as
   extra["params"] on the program's rows. Arithmetic on results/README.md's
   round 2 numbers: a 2.4M-parameter part with 6.3M full glue gives rank 12
   on all linear layers of Llama-3.2-1B, and with 0.46M rank-64 glue rank 4.
   Round 3 also plans a standard rank-64 LoRA.
5. The driver owns every budget charge. A runner does the work and reports
   measured seconds; it never spends a token, and a retry reuses the token
   it was given (rule 6).
"""

import hashlib
import json
import math

from wsp.swarm import stub
from wsp.swarm.agents.scripted import rank_rows
from wsp.swarm.baselines import build_summary, setting
from wsp.swarm.config import fidelity_cost, guard_tasks, heal_data_allowed
from wsp.swarm.log import new_eval_row
from wsp.swarm.scoring import score_with_config

STEP_SECONDS = {"lora": 0.5, "full_ft": 1.5}
CHECKPOINTS = 4
LORA_RANK = 16
STUB_CURVES = {
    "lora": {"gain": 0.06, "tau": 400.0, "forget": 0.01},
    "full_ft": {"gain": 0.09, "tau": 800.0, "forget": 0.06},
}

# The host's shapes, from the public unsloth/Llama-3.2-1B-Instruct config:
# 16 layers, hidden 2048, MLP 8192, 32 query heads and 8 KV heads of size 64,
# vocabulary 128256, tied input and output embeddings.
LLAMA_32_1B = {
    "layers": 16,
    "hidden": 2048,
    "vocab": 128256,
    "linear": [("q_proj", 2048, 2048), ("k_proj", 2048, 512), ("v_proj", 2048, 512), ("o_proj", 2048, 2048),
               ("gate_proj", 2048, 8192), ("up_proj", 2048, 8192), ("down_proj", 8192, 2048)],
}


# Parameter counts.

def lora_params(rank, shapes=LLAMA_32_1B):
    """Trainable parameters of a rank-r LoRA on every linear layer: r * (d_in + d_out) per layer."""
    per_layer = 0
    for name, d_in, d_out in shapes["linear"]:
        per_layer += rank * (d_in + d_out)
    return per_layer * shapes["layers"]


def full_params(shapes=LLAMA_32_1B):
    """Every parameter of the host: linear layers, two norms per layer, the tied embedding, the final norm."""
    per_layer = 2 * shapes["hidden"]
    for name, d_in, d_out in shapes["linear"]:
        per_layer += d_in * d_out
    return per_layer * shapes["layers"] + shapes["vocab"] * shapes["hidden"] + shapes["hidden"]


def matched_lora_rank(target_params, shapes=LLAMA_32_1B):
    """The LoRA rank whose parameter count is closest to target_params (at least 1)."""
    rank = int(round(float(target_params) / lora_params(1, shapes)))
    if rank < 1:
        rank = 1
    return rank


def lora_rank_for(config):
    """config lora_match_params (a parameter count) wins; else config lora_rank; else 16."""
    match = setting(config, "lora_match_params", None)
    if match is not None:
        return matched_lora_rank(match)
    return setting(config, "lora_rank", LORA_RANK)


def plan_checkpoints(total_steps, count):
    """Checkpoint steps spread evenly over the run: total * i // count for i = 1..count."""
    targets = []
    for index in range(1, count + 1):
        targets.append(total_steps * index // count)
    return targets


# Runners.

class Trainer:
    """What a training runner implements. TrainingArm owns every budget charge.

    step_seconds() and eval_seconds() are the planned costs (from a pilot)
    and are used only to plan checkpoints; the charge is the measured
    compute_s each call returns.

    train(steps) runs `steps` more optimizer steps from where training
    stopped and returns {"ok", "error", "step": total steps so far,
    "checkpoint": a handle evaluate() accepts, "train_loss": [floats],
    "compute_s": {"train": seconds}}.

    evaluate(checkpoint, split, tasks, token, fidelity) scores a checkpoint
    with the same eval harness as every arm and returns {"ok", "error",
    "scores": {task: float}, "compute_s": {"eval": seconds}}. token is the
    eval token the driver already spent; never spend one here.
    """

    name = "base"

    def step_seconds(self):
        raise NotImplementedError

    def eval_seconds(self):
        raise NotImplementedError

    def trainable_params(self):
        raise NotImplementedError

    def describe(self):
        raise NotImplementedError

    def train(self, steps):
        raise NotImplementedError

    def evaluate(self, checkpoint, split, tasks, token, fidelity="full"):
        raise NotImplementedError

    def release(self, checkpoint):
        return None


def stub_checkpoint_scores(world, method, step, split, tasks, guards, fraction, stub_seed, run_seed=0):
    """STUB task scores for a checkpoint after `step` optimizer steps (see the module docstring).

    The run seed is in the noise key: a training run with another seed
    (another data order) is another model, as its checkpoint id already says.
    """
    base = world.base_scores(split, tasks)
    curve = STUB_CURVES[method]
    progress = 1.0 - math.exp(-step / curve["tau"])
    sigma = stub.EVAL_NOISE / math.sqrt(fraction)
    out = {}
    for task in tasks:
        if task in guards:
            score = base[task] - curve["forget"] * progress
        else:
            score = base[task] + curve["gain"] * progress
        noise_text = ("train|" + str(stub_seed) + "|" + str(run_seed) + "|" + method + "|" + str(step) + "|"
                      + split + "|" + task)
        score += stub.fixed_gauss(noise_text, sigma)
        out[task] = round(stub.clamp(score, 0.0, 1.0), 4)
    return out


class StubTrainer(Trainer):
    """STUB LoRA or full fine-tuning: the cost model and learning curve above, instant and deterministic."""

    name = "stub"

    def __init__(self, config, method):
        if method not in STEP_SECONDS:
            raise ValueError("no stub trainer for method " + str(method))
        self.config = config
        self.method = method
        self.world = stub.StubWorld(config["stub_seed"])
        self.guards = guard_tasks(config)
        self.rank = None
        if method == "lora":
            self.rank = lora_rank_for(config)
        self.step = 0

    def step_seconds(self):
        return STEP_SECONDS[self.method]

    def eval_seconds(self):
        return stub.eval_cost(1.0)

    def trainable_params(self):
        if self.method == "lora":
            return lora_params(self.rank)
        return full_params()

    def describe(self):
        if self.method == "lora":
            return "LoRA r" + str(self.rank) + " on all linear layers"
        return "full fine-tuning"

    def train(self, steps):
        self.step += steps
        return {"ok": True, "error": None, "step": self.step, "checkpoint": {"method": self.method, "step": self.step},
                "train_loss": [], "compute_s": {"train": steps * self.step_seconds()}, "stub": True}

    def evaluate(self, checkpoint, split, tasks, token, fidelity="full"):
        fraction = fidelity_cost(self.config, fidelity)
        scores = stub_checkpoint_scores(self.world, self.method, checkpoint["step"], split, tasks, self.guards,
                                        fraction, self.config["stub_seed"], self.config["seed"])
        return {"ok": True, "error": None, "scores": scores, "compute_s": {"eval": stub.eval_cost(fraction)},
                "stub": True}


def make_trainer(config, method):
    """The trainer for a training arm. Only the STUB exists; real runners are the plan above."""
    if config["backend"] == "stub":
        return StubTrainer(config, method)
    raise NotImplementedError("the " + method + " arm has only a STUB runner; the real run is a plan in "
                              "wsp/swarm/baselines/training.py")


# The driver.

class TrainingArm:
    def __init__(self, config, env, trainer):
        self.config = config
        self.env = env
        self.trainer = trainer
        self.agent = config["method"]
        self.checkpoints = int(setting(config, "train_checkpoints", CHECKPOINTS))
        self.data = heal_data_allowed(config)[0]
        self.rows = []
        self.trained = {}        # checkpoint row hash -> the train() result that made it
        self.planned_steps = 0

    def settings(self):
        return {"method": self.agent, "trainer": self.trainer.name, "describe": self.trainer.describe(),
                "trainable_params": self.trainer.trainable_params(), "step_seconds": self.trainer.step_seconds(),
                "eval_seconds": self.trainer.eval_seconds(), "checkpoints": self.checkpoints, "data": self.data,
                "batch_size": self.config["heal_batch_size_allowed"][0],
                "seq_len": self.config["heal_seq_len_allowed"][0],
                "cascade": "not used: each checkpoint eval is one full-fidelity eval"}

    def training_info(self):
        return {"method": self.agent, "describe": self.trainer.describe(), "data": self.data,
                "trainable_params": self.trainer.trainable_params(), "planned_steps": self.planned_steps,
                "batch_size": self.config["heal_batch_size_allowed"][0],
                "seq_len": self.config["heal_seq_len_allowed"][0], "seed": self.config["seed"]}

    def run(self):
        """Train with checkpoint evals, then score the best checkpoint once on held-out. Returns the summary."""
        env = self.env
        env.write_manifest({"baseline": self.settings(), "k": self.config["k"], "cascade": self.config["cascade"]})
        env.reset(self.config["seed"])
        self.train_and_check()
        env.end_search(self.agent + " finished after " + str(len(self.rows)) + " checkpoints")
        heldout = self.final_heldout()
        summary = build_summary(self.config, env, len(self.rows), self.frontier(), heldout, [])
        env.log.write_json("summary.json", summary)
        return summary

    def train_and_check(self):
        env = self.env
        budget = env.budget
        step_s = float(self.trainer.step_seconds())
        eval_s = float(self.trainer.eval_seconds())
        count = min(self.checkpoints, int(math.floor(budget.eval_remaining() + 1e-9)))
        if count < 1:
            env.log_event("train_plan", "no eval token left for a checkpoint; nothing trained", self.agent)
            return
        total_steps = int(math.floor((budget.compute_remaining() - count * eval_s) / step_s))
        if total_steps < count:
            env.log_event("train_plan", "too little compute for " + str(count) + " checkpoints; nothing trained",
                          self.agent)
            return
        self.planned_steps = total_steps
        targets = plan_checkpoints(total_steps, count)
        env.log_event("train_plan", self.trainer.describe() + " on " + self.data + ": " + str(total_steps) +
                      " steps at " + str(step_s) + " s per step, checkpoint evals at steps " + str(targets),
                      self.agent, 0, extra={"targets": targets, "step_seconds": step_s, "eval_seconds": eval_s})
        done = 0
        for index in range(len(targets)):
            if not budget.can_spend(1.0):
                break
            steps = targets[index] - done
            affordable = int(math.floor((budget.compute_remaining() - eval_s) / step_s))
            if steps > affordable:
                steps = affordable
            if steps <= 0:
                break
            trained, compute = self.train_chunk(steps)
            if not trained.get("ok"):
                env.log_event("train_error", "training failed: " + str(trained.get("error")), self.agent, index)
                break
            done = trained["step"]
            token = budget.spend_eval(self.agent, 1.0)
            if token is None:
                break
            row = self.score_checkpoint(trained, "search", budget, token, index, compute)
            self.rows.append(row)
            self.trained[row["program_hash"]] = trained
        env.check_overshoot()

    def train_chunk(self, steps):
        try:
            trained = self.trainer.train(steps)
        except Exception as error:
            trained = {"ok": False, "error": type(error).__name__ + ": " + str(error), "compute_s": {}}
        compute = {}
        self.charge(self.env.budget, trained.get("compute_s"), compute)
        return trained, compute

    def charge(self, budget, compute_s, into):
        if compute_s is None:
            return
        for phase in sorted(compute_s):
            seconds = float(compute_s[phase])
            budget.charge_compute(seconds, self.agent, phase)
            into[phase] = into.get(phase, 0.0) + seconds

    def checkpoint_hash(self, step):
        text = json.dumps({"training": self.training_info(), "step": step}, sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def score_checkpoint(self, trained, split, budget, token, index, compute):
        """Evaluate one checkpoint with an already spent token, charge its eval time, and log its row."""
        env = self.env
        try:
            result = self.trainer.evaluate(trained["checkpoint"], split, env.tasks, token, "full")
        except Exception as error:
            result = {"ok": False, "error": type(error).__name__ + ": " + str(error), "compute_s": {}}
        self.charge(budget, result.get("compute_s"), compute)
        step = trained["step"]
        row = new_eval_row()
        env.stamp(row, budget)
        row["agent"] = self.agent
        row["round"] = index
        row["hypothesis"] = (self.trainer.describe() + " on " + self.data + ", checkpoint at step " + str(step) +
                             " of " + str(self.planned_steps))
        row["split"] = split
        row["fidelity"] = "full"
        row["token"] = token
        row["eval_count"] = budget.eval_used()
        row["compute_s"] = dict(compute)
        row["program"] = {}
        row["program_hash"] = self.checkpoint_hash(step)
        row["describe"] = "host + " + self.trainer.describe() + " on " + self.data + ", step " + str(step)
        extra = {"training": self.training_info(), "step": step, "params": self.trainer.trainable_params(),
                 "train_loss": list(trained.get("train_loss") or []), "eval_cost": 1.0}
        scores = result.get("scores") or {}
        missing = []
        for task in env.tasks:
            if task not in scores:
                missing.append(task)
        if not result.get("ok") or len(missing) > 0:
            row["status"] = "infra_error"
            extra["error"] = result.get("error") or ("eval returned no score for " + ", ".join(missing))
        else:
            value, collapsed = score_with_config(scores, env.base_scores(split), self.config)
            row["scores"] = dict(scores)
            row["combined"] = value
            row["status"] = "ok"
            if collapsed:
                row["status"] = "collapsed"
        row["extra"] = extra
        env.emit(row)
        return row

    def frontier(self):
        candidates = []
        for row in self.rows:
            if row["status"] == "ok":
                candidates.append(row)
        return rank_rows(candidates, self.config["frontier_size"], self.config["tie_margin"], self.config)

    def final_heldout(self):
        """Score the best checkpoint once on held-out, from the held-out pool (rule 7)."""
        heldout_budget = self.env.heldout_budget
        frontier = self.frontier()
        if heldout_budget is None or len(frontier) == 0 or heldout_budget.compute_remaining() <= 0:
            return []
        token = heldout_budget.spend_eval(self.agent, 1.0)
        if token is None:
            return []
        best = frontier[0]
        row = self.score_checkpoint(self.trained[best["program_hash"]], "heldout", heldout_budget, token,
                                    best["round"], {})
        return [row]
