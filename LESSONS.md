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
