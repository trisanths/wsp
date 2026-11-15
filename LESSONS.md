# Lessons

Read this before launching a run or briefing an agent. Add a line when
something fails. Each entry: what went wrong, then the rule.

## Before any Modal launch

- Run ONE cheap cell first and wait for its first row. Four rounds died on
  setup before reaching the experiment on 2026-11-14 (16d, 20, 21 twice),
  each costing 20 to 40 minutes. CPU tests do not catch image, volume, or
  dataset problems. A fan-out comes only after a smoke cell prints a row.
- Check the image has every package `wsp` imports. Round 21 lacked
  `zstandard` and died at import. Copy the pip list from a script that
  has run (modal_round20.py), do not write a new one.
- Modal volumes: no hard links (`os.link` fails with PermissionError), and
  `volume.reload()` fails while any file is open (Hugging Face xet logs
  count). Use a plain rename, wrap reloads in try/except, and remember each
  workspace has its own volume, so shards in different workspaces cannot
  race.
- A workspace runs about 10 GPU containers. More than that queues with no
  message. Count running containers (`modal container list`) before
  launching, launch the most important round first, and do not start
  low-value rounds that will sit in front of it. Tell the mining session
  before a big launch; it shares the same three workspaces.
- Background processes started with `&` ignore SIGINT. Stop them with
  SIGTERM on their PIDs, then `modal app stop -y <id>`. In zsh, loop over
  PIDs with `while read`, not `for p in $VAR` (newlines stay in one word).
- Only trisanth, trisanthsrini, and ztrisanth work. trisanths,
  trisanthsrinivasan, and trisanthcollege were disabled by Modal.

## Data

- Verify a dataset by reading thousands of rows, not three. LogiQA2.0 has
  rows without options; three sample rows passed and the first real read
  crashed. Renderers must skip a row they cannot read.
- Never heal on dataset gold answers in a foreign style. A LoRA heal on
  GSM8K gold solutions cost the base 11 GSM8K points and halved MATH-500.
  Heal on the donor's or the base's own outputs or logits, with a KL to the
  frozen base on general text.
- Nothing that trains, fits, or selects for a compile may use a benchmark
  the result is scored on. Other benchmarks and data are fine and wanted.
  Mining may use any data. Use wsp/healdata.py with the request's reported
  list.
- The compiler merges the LoRA of a LoRA-only program into the host. A
  loaded LoRA-only model has no LoRA parameters to train.

## Experiments

- Never report one seed. GSM8K moved 8 points between seeds (run 16b), and
  a control moved 8 points on MMLU (run 14c). I called a 72.3 a win when
  the base was 68.7 and the second seed was 64.0.
- Always include the base in the first table. I compared with LoRA only
  and the LoRA was itself damaged by the heal.
- "Another layer range of the same donor" is not a random control. It is
  pretrained knowledge too. Use scrambled weights (wsp/scramble.py) to test
  whether donor knowledge is used, and several controls with a z-score
  (probe.content_z) to beat seed noise.
- Check lineage before choosing a method (wsp/lineage.py). Qwen2.5-Coder
  and -Math are far from Qwen2.5-Instruct (cosine 0.2 to 0.5): mixing,
  MLP experts, and layer swaps all gave chance. Weight mixing is only for
  true fine-tunes (cosine near 1, same rope_theta).
- Smoke-test new decision code on real Hub models, not only tiny fixtures.
  The lineage probe passed 30 CPU tests and then called OpenMath2 "foreign"
  (tokenizer hash too strict) and HuatuoGPT-o1 "same" (cosine threshold
  where equality was meant).
- Generation paths need their own test with batches, left padding, and two
  calls in a row. Forced whole-layer experts matched the specialist on one
  forward pass and scored 0.7 on GSM8K because of a cache bug.
- Check the eval protocol before trusting any baseline: chat template,
  answer format, and token limit. Runs 1 to 5 scored GSM8K with a bare
  prompt and 256 tokens, which cut instruct models off before the answer
  (Llama-1B read 0.23; the right protocol gives 0.58). Conclusions from
  those runs were drawn on truncated outputs. Use wsp.skills_eval and look
  at a few raw generations before a fan-out.
- Print the cell name on every score line. Round 19 printed bare scores and
  I could not tell which cell they belonged to for an hour.

## Working with agents and sessions

- Give every agent the current rules in its brief (heal data, benchmarks,
  controls). Agents briefed before a rule changed built code that broke
  it, and each needed a patch round.
- One owner per file. My studio agent and the studio session edited the
  same files and conflicted three times. The studio session owns studio/;
  the mining session owns the atlas, zoo, and giants.
- `git pull --rebase --autostash` can leave conflict markers in files an
  agent is editing. Check `git status` for `UU` after every pull.
- Do not relay another session's reading of the owner as a rule without
  checking it against what the owner told me. I stopped the atlas on a
  wrong relay and restarted it ten minutes later.
- Say plainly when a result is one seed, unlabelled, or diagnostic. Record
  failures on past.dev (scripts/team_memory.py, topic compiler) as they
  happen, so other sessions do not repeat them.
- Read a real file before describing its format in a brief. I told a build
  agent the miners' skill items were {prompt, answer}; they are
  {context, continuation, digits_only} under "search" and "heldout", and
  the owner of the format corrected me. Ask the owning session for the
  format, or open two real files, before writing the brief.
- Test data helpers at the sizes real scripts ask for. healdata.pairs took
  an even share from each source, so a 700-row source (SVAMP) left the math
  pool at 4168 pairs when round 22 asked for 5096, and every cell died. Its
  order also depended on the count, so two calls with different skip values
  were not disjoint. Pools now read round robin in a fixed order.
- A zero-started output map is not safe by itself. In the first feedback
  seam run the fresh model equalled the base, and ONE Adam step made the
  written vector many times larger than the embedding it was added to (the
  donor's states have huge outlier dimensions and nothing normalized them):
  KD to the donor went from 0.64 to 7.9 in one step and the healed model
  scored chance. Normalize what a component writes, tie its size to the
  host's own scale, start the gate at 0, warm up, and halt a heal whose loss
  jumps.
- Check stage 1 before paying for stage 2. That run's input stitch left a
  1.98-nat penalty (the stitch study reached 0.03) and stage 2 ran anyway.
  A compile must refuse to heal on a bad stitch.
- Run a tool's tests with the library versions of the image it runs on.
  wsp/giant.py passed 60 tests under transformers 4.57.1, but the giants
  image uses 5.23.0, where 29 of the 60 fail (5.x holds experts as fused
  3-D parameters, not per-expert tensors). Two GPU smokes died before the
  mining session found this. A verifier's "checked in a scratch 5.23
  environment" claim is not a test in the suite; put the check in the suite.
- Do not import a transformers model class for the first time inside
  `with torch.device("meta")`. Model modules import torch._dynamo, which
  fails with a circular import under an active device context, and the error
  reads "Could not find Qwen2ForCausalLM". Earlier rounds never hit it
  because they loaded a model first; round 24's prefetch step did not, and
  the mining session hit the same thing in the giants profiler. Import
  torch._dynamo and resolve the class before entering the context, and test
  such helpers in a fresh process.
- CPU tests in float32 cannot catch dtype bugs that appear in bf16 on a GPU.
  Skill tops passed 36 CPU tests and died on its first GPU step: the donor's
  norm returned float32 into a bf16 head. Add at least one bf16 test of each
  forward path (CPU supports bf16 matmuls), and cast explicitly at every
  module boundary that mixes a donor's and a base's parameters.
- A version port must keep working on the old version until every user
  moves. merge-tf5 imports transformers.conversion_mapping (5.x only) at the
  top of a code path in wsp/parts.py, so merging it into main broke 295
  tests under 4.57.1, the version every running round uses. Guard
  version-specific imports and run the suite under both versions before a
  port lands.
- Launch long Modal runs with `modal run --detach`. At 23:58 PT the Mac's
  network dropped for about a minute; every attached `modal run` lost its
  client, Modal stopped the ephemeral apps, and four weave searches
  (up to generation 9) and the skill tops smoke were lost. Detached apps
  keep running; scripts must write rows to the volume so a dead local
  entrypoint loses nothing (round 21 writes progress per generation).
- Key every cached artifact by the code version that made it. Round 24's
  rerun after an objective change restored the stitch the old objective had
  saved on the volume and reported the old numbers as if they were new. Put
  a fit or format version in every artifact path and bump it when the fit
  changes.
- A filter that matches nothing must fail loudly. Round 20c R1 was launched
  with `--only` set to full cell names, but the filter matched only arm or
  config names, so the run printed "0 cells" and exited, and I did not look
  for 50 minutes. Run `--estimate-only` with the same arguments before a
  launch, and make entrypoints exit with an error on zero cells.
