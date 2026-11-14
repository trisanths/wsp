# wsp: weight-space programming

Write a model as a program over components, then compile it into one model.

This repo holds the language and compiler (part 2) and the eval harness and extraction tools (part 1b, [wsp_eval/README.md](wsp_eval/README.md)). The team split and each
person's agent brief are in [briefs/](briefs/). Checked facts about Deletion
Tuning are in [VERIFY.md](VERIFY.md).

## What works now

- Mask algebra over Deletion Tuning masks on one shared base: union (`|`),
  intersection (`&`), difference (`-`), and `restrict(layers=..., modules=...)`.
- Task vectors: `weight * (fine-tuned - base)` on selected layers and modules.
  Native code, not mergekit.
- `compile()`: loads the base in bf16, adds task vectors, zeroes masked
  weights, and saves a standard Hugging Face model with `program.json` and
  `compile_report.json` beside it.
- A seed `catalog.jsonl` with the 37 public SFT masks for Llama-3.2-1B-Instruct
  (14 MMLU-Pro subjects, 23 BBEH tasks).

Not built yet: the heal pass, cross-family stitching, the text syntax, and a
deletion budget rule.

## Example

```python
from wsp import Catalog, Program, use

catalog = Catalog.load("catalog.jsonl")
bio = use("mask_mmlupro_biology_llama1b")
chem = use("mask_mmlupro_chemistry_llama1b")

program = Program(
    base="unsloth/Llama-3.2-1B-Instruct",
    masks=(bio | chem).restrict(layers=range(4, 16)),
)
program.compile("outputs/bio_chem", catalog)
```

Agents can send the same program as JSON: `program.to_dict()` and
`Program.from_dict(data)`.

## Base model

The catalog uses `unsloth/Llama-3.2-1B-Instruct`, an ungated copy of
`meta-llama/Llama-3.2-1B-Instruct`. The meta repo needs a Hugging Face access
token. All components in one program must name the program's base; the
compiler refuses a mixed-base program.

## Run

CPU tests (toy tensors only, no model downloads):

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]" && .venv/bin/pytest -q
```

Real models run on Modal, profile `trisanth`:

```bash
MODAL_PROFILE=trisanth .venv/bin/modal run modal_app.py::verify
```

```bash
MODAL_PROFILE=trisanth .venv/bin/modal run modal_app.py::compile_json --program-path program.json --name my_model
```

Compiled models go to the Modal volume `wsp-models` at `/vol/compiled/<name>`.

## Mask format

Masks use the Deletion Tuning `packbits_zstd` format (code from
agentic-learning-ai-lab/deleting-weights, MIT license). The file stores KEEP
bits. In this code, a mask is a dict of parameter name to a boolean tensor
where True means delete.
