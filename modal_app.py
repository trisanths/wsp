"""Run the compiler on Modal.

Commands (profile "trisanth"):
    MODAL_PROFILE=trisanth modal run modal_app.py::verify
    MODAL_PROFILE=trisanth modal run modal_app.py::compile_json --program-path program.json --name my_model

Compiled models go to the Modal volume "wsp-models" under /vol/compiled/<name>.
"""

import json

import modal

app = modal.App("wsp-compiler")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1",
        "transformers==5.23.0",
        "accelerate",
        "huggingface_hub",
        "safetensors",
        "zstandard",
        "numpy",
    )
    .env({"HF_HOME": "/vol/hf"})
    .add_local_python_source("wsp")
    .add_local_file("catalog.jsonl", "/root/catalog.jsonl")
)


@app.function(image=image, volumes={"/vol": volume}, cpu=4, memory=16384, timeout=1800)
def compile_remote(program_dict, name):
    from wsp import Catalog, Program

    catalog = Catalog.load("/root/catalog.jsonl")
    program = Program.from_dict(program_dict)
    output_dir = f"/vol/compiled/{name}"
    program.compile(output_dir, catalog)
    volume.commit()
    with open(f"{output_dir}/compile_report.json") as f:
        return json.load(f)


@app.local_entrypoint()
def compile_json(program_path: str, name: str):
    with open(program_path) as f:
        program_dict = json.load(f)
    report = compile_remote.remote(program_dict, name)
    print(json.dumps(report, indent=2))


@app.function(image=image, volumes={"/vol": volume}, gpu="L4", memory=32768, timeout=2400)
def verify_remote():
    """Hour-8 compiler checks on the real Llama-3.2-1B base and public masks."""
    import glob

    import torch
    import transformers
    from safetensors.torch import load_file

    from wsp import Catalog, Program, use
    from wsp import masks as mask_ops
    from wsp.compiler import make_mask_loader

    catalog = Catalog.load("/root/catalog.jsonl")
    base = "unsloth/Llama-3.2-1B-Instruct"
    bio_id = "mask_mmlupro_biology_llama1b"
    chem_id = "mask_mmlupro_chemistry_llama1b"
    results = {}

    def state(path):
        tensors = {}
        for file in sorted(glob.glob(f"{path}/*.safetensors")):
            tensors.update(load_file(file))
        return tensors

    # 1. An empty program reproduces the base exactly.
    empty_dir = Program(base=base).compile("/vol/compiled/verify_empty", catalog)
    base_model = transformers.AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    base_state = {}
    for name, param in base_model.named_parameters():
        base_state[name] = param.detach()
    empty_state = state(empty_dir)
    mismatched = []
    for name in base_state:
        if name in empty_state and not torch.equal(empty_state[name], base_state[name]):
            mismatched.append(name)
    results["empty_matches_base"] = len(mismatched) == 0
    results["empty_mismatched_params"] = mismatched[:5]

    # 2. Mask counts and union arithmetic.
    load_mask = make_mask_loader(catalog)
    bio = load_mask(bio_id)
    chem = load_mask(chem_id)
    with open(f"{catalog.local_path(bio_id)}/config.json") as f:
        bio_config = json.load(f)
    n_bio = mask_ops.count_deleted(bio)
    n_chem = mask_ops.count_deleted(chem)
    n_both = mask_ops.count_deleted(mask_ops.intersect(bio, chem))
    results["bio_deleted"] = n_bio
    results["bio_config_zero_params"] = bio_config.get("zero_params")
    results["chem_deleted"] = n_chem
    results["bio_and_chem"] = n_both
    results["jaccard_bio_chem"] = n_both / (n_bio + n_chem - n_both)

    per_layer = {}
    bio_layers = mask_ops.deleted_per_layer(bio)
    chem_layers = mask_ops.deleted_per_layer(chem)
    both_layers = mask_ops.deleted_per_layer(mask_ops.intersect(bio, chem))
    for layer in sorted(bio_layers):
        a = bio_layers[layer]
        b = chem_layers.get(layer, 0)
        c = both_layers.get(layer, 0)
        per_layer[layer] = round(c / max(1, a + b - c), 4)
    results["jaccard_per_layer"] = per_layer

    union_dir = Program(base=base, masks=use(bio_id) | use(chem_id)).compile("/vol/compiled/verify_bio_or_chem", catalog)
    with open(f"{union_dir}/compile_report.json") as f:
        union_report = json.load(f)
    results["union_deleted"] = union_report["mask"]["deleted"]
    results["union_expected"] = n_bio + n_chem - n_both
    results["union_fraction_of_all_params"] = union_report["mask"]["fraction_of_all_params"]

    # 3. Zeros in the saved union model equal the union mask exactly.
    union_state = state(union_dir)
    wrong = 0
    for name in bio:
        should_delete = bio[name] | chem[name]
        zero_now = union_state[name] == 0
        zero_before = base_state[name] == 0
        wrong += int(((zero_now & ~zero_before) != (should_delete & ~zero_before)).sum())
    results["union_zero_pattern_errors"] = wrong
    volume.commit()

    # 4. Sanity: loss on a fixed passage, and one short generation.
    text = (
        "Photosynthesis converts light energy into chemical energy. In plants, it takes place in the "
        "chloroplasts, where chlorophyll absorbs light. The reaction uses carbon dioxide and water and "
        "releases oxygen. An acid donates a proton, and a base accepts one."
    )
    question = "Which organelle carries out photosynthesis in plant cells? Answer in one word."
    tokenizer = transformers.AutoTokenizer.from_pretrained(base)
    sanity = {}
    for label, path in [("base", base), ("bio", None), ("bio_or_chem", union_dir)]:
        if label == "bio":
            path = Program(base=base, masks=use(bio_id)).compile("/vol/compiled/verify_bio", catalog)
        model = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda")
        model.eval()
        ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            loss = model(ids, labels=ids).loss.item()
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        with torch.no_grad():
            out = model.generate(chat, max_new_tokens=12, do_sample=False)
        answer = tokenizer.decode(out[0, chat.shape[1]:], skip_special_tokens=True)
        sanity[label] = {"loss": round(loss, 4), "answer": answer.strip()}
        del model
        torch.cuda.empty_cache()
    results["sanity"] = sanity
    volume.commit()
    return results


@app.local_entrypoint()
def verify():
    results = verify_remote.remote()
    print(json.dumps(results, indent=2))
    with open("outputs/verify_results.json", "w") as f:
        json.dump(results, f, indent=2)
