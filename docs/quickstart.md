# Sapien quickstart

Three requests, each written as a program. Every program is a text file with
the `.sap` suffix. The reference for each statement is
[language.md](language.md). The results so far are in
[../results/README.md](../results/README.md). The notes under each program
say what that log shows for it, and what it does not show.

Install once, then use the same four steps for each program:

```bash
uv venv --python 3.12 && uv pip install -e ".[compile,dev]"
sapien check prog.sap                            # size estimate, no weights
sapien compile prog.sap -o out/prog              # build one model directory
sapien compile prog.sap -o out/prog_random --control random   # random components of the same size
sapien score out/prog --skills gsm8k,mmlu
```

`sapien check` reads only `config.json` of each model. `sapien compile`
writes `compile_report.json` with `params_total`, `params_active`, and
`heal_seconds`. Python code loads the result with
`wsp.load("out/prog")`, which returns `(model, tokenizer)`.

A result means something only next to its controls. For each program below,
also compile and score:

1. the base alone,
2. the base with a LoRA on the same heal data (delete the component lines,
   keep `lora` and `heal`),
3. the base with a distillation heal from the same source at the same heal
   budget (the `heal ... distill` line, no components),
4. the program with `--control random`.

## 1. A small specialist

Request: a math model that runs at most 1.6B parameters per token, from a
1.5B base, using the Qwen2.5-Math-1.5B-Instruct MLPs as a routed expert.

```sap
# math_specialist.sap
host Qwen/Qwen2.5-1.5B-Instruct
experts Qwen/Qwen2.5-Math-1.5B-Instruct top_k 1 balance on
lora rank 8
heal steps 1000 data gsm8k_train lr 1e-4 distill Qwen/Qwen2.5-Math-7B-Instruct kd_weight 1.0
budget total 3b active 1.6b heal_minutes 60
```

The compiled network holds the base MLPs and the Math MLPs in every layer
and a router in front of each pair. With `top_k 1` one MLP runs per token,
so `params_active` stays near the base's 1.54B while `params_total` is about
2.7B. The teacher shares the Qwen tokenizer, which the compile checks. The
budget makes the compile fail before the heal if the model is over a limit.

What the log shows: run 16b built a skill MoE like this one (without the
distillation). Its top-1 form scored 68.2 on GSM8K on average, against 68.7
for the base, while the Math specialist alone scored 81.7. The router barely
left the base, so the specialist's skill did not carry over. This program has
not been run.

## 2. A small base with skills written in, under a size budget

Request: a 0.5B base that gains the skill of a much larger model, in at most
7B total parameters.

```sap
# small_base_big_layers.sap
host Qwen/Qwen2.5-0.5B-Instruct
insert Qwen/Qwen2.5-14B-Instruct layers 14..34 at 7 stitch mlp part_lora_rank 8
lora rank 16
heal steps 1500 data mmlu_aux lr 1e-4 distill Qwen/Qwen2.5-14B-Instruct kd_weight 1.0
budget total 7b active 6.6b heal_minutes 90
```

The host runs layers 0 to 6, then 21 layers of the 14B model behind a
trained stitch, then the rest of the host. The gate starts closed, so the
fresh network equals the host. The heal opens it, trains the stitch, both
LoRAs, and distills toward the 14B model. `sapien check` estimates 6.35B for the network before the part LoRA, so
the `budget` line fits.

What the log shows: runs 14 and 14b wrote the same block into the same base.
The stitch is lossless (a penalty of +0.008 and -0.02 nats), but the base
gained 3 to 6 MMLU points and a random block of the same length gained as
much. The gain came from added depth, not from the donor's skill. Run 14c
(the distillation heal) is running and not in the log yet.

## 3. A multi-skill model

Request: one 1.5B-class model with math and code, no larger than 3.9B total.

```sap
# math_and_code.sap
host Qwen/Qwen2.5-1.5B-Instruct
experts Qwen/Qwen2.5-Math-1.5B-Instruct,Qwen/Qwen2.5-Coder-1.5B-Instruct top_k 2 balance on router_lr 0.001
lora rank 16
heal steps 1000 data gsm8k_train lr 1e-4
budget total 3.9b active 2.8b heal_minutes 60
```

Each layer routes over three MLPs: the base, Math, and Coder. With `top_k 2`
two run per token. `sapien check` estimates `params_active` 2.70B and `params_total` 3.86B.
Check the numbers first:

```bash
sapien check math_and_code.sap
```

For grafts that write blocks from several specialists into one host, give
each graft `gate on`, so each fires on its own tokens, and compile with a
catalog (`sapien compile prog.sap -o out --catalog catalog.jsonl`). The
`library` and `atlas` commands help find the components:

```bash
sapien library query --kind block --skill math --status validated
sapien atlas best --skill code --max-params-b 2
```

What the log shows: run 16b is this program with `top_k` 1, 2, and 3. No
arm beat both the base and the base plus LoRA by more than the seed spread
(top-1 GSM8K scored 64.0 and 72.3 on two seeds; the best MedMCQA arm, top-2
plus LoRA, scored 46.0 and 46.1 against 44.6 for the base and 45.4 and 45.0
for the LoRA alone). The LoRA heal on this data mix cost 11 GSM8K points. The router stayed on the base MLP for
92% of the routing slots on GSM8K prompts. The numbers for this exact
program do not exist yet.
