# Update 1 to the part 3 brief (from part 2, 2026-11-14)

Received by Rex from Trisanth and saved verbatim. It overrides `brief-R-rl-env-swarm.md` where they differ. Part 3's reading of it is in `docs/part3-design.md` (version 2).

---

To Rex (RL env and swarm)

Update to your brief (briefs/brief-R-rl-env-swarm.md). This overrides it where they differ. I saw part 3's commits and the status board (issue trisanths/wsp#1); this answers its questions for part 2.

The project is mixed family now. The compiler grafts parts from any donor model, in any family and at any size, into a host. The output is one fixed model per task type. Routing is out of scope.

Action space:
A program is a host plus a list of grafts, each a catalog part id with an optional host depth at=, plus an optional heal pass.
Draft JSON: {"host": ..., "grafts": [{"id": ..., "at": 9}], "heal": {"steps": 300, "data": "gsm8k_train"}}.
Mask expressions on the host still work.
I'll lock the exact schema in wsp/program.py and post it on issue trisanths/wsp#1.

Cheap signal: each compile reports a fit score per graft before any heal. Planners can drop bad placements without spending an eval.

Cost: compile plus heal takes minutes, and eval uses Hugging Face generation. Count both against the budget, and expect fewer steps per hour than with mask-only programs.

Baselines: evolutionary search over the same graft space, LoRA, and full fine-tuning, all at equal budget. Also add the "random cluster through the same glue" control.

Heal: the heal pass is part of the compiler build. It will be heal(steps, data) inside the program.

Fixed: the Windows catalog path bug (commit c74d2b6).
