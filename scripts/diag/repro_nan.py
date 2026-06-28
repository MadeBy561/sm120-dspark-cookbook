"""Locate the DSpark training NaN: build the draft model exactly as the trainer does,
run ONE forward on real cache samples on a single GPU, and finite-check every stage.

  python3 repro_nan.py
"""
import json
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, "/work/DeepSpec")
from transformers import AutoConfig  # noqa: E402

from deepspec.data.target_cache_dataset import CacheDataset  # noqa: E402
from deepspec.data import CacheCollator  # noqa: E402
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel  # noqa: E402
from deepspec.modeling.dspark.qwen3.config import build_draft_config  # noqa: E402

MODEL = "/mnt/models/GLM-5.2-Int8Mix-NVFP4-REAP-594B"
CACHE = "/mnt/18tb_r1/dspark-data/cache"
DEV = "cuda:0"


class Args(dict):
    """attribute + `in` access, like the trainer's OmegaConf model_args."""
    def __getattr__(self, k):
        return self[k]


MODEL_ARGS = Args(
    num_draft_layers=5,
    target_layer_ids=[75, 76, 77],
    block_size=5,
    num_anchors=512,
    markov_rank=512,
    markov_head_type="vanilla",
    mask_token_id=154821,
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,
)


def finite(t):
    t = t.float()
    return int((~torch.isfinite(t)).sum().item()), float(t.abs().max().item())


print("building draft model ...")
target_config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
draft_config = build_draft_config(target_config, MODEL_ARGS)
model = Qwen3DSparkModel(draft_config).to(torch.bfloat16).to(DEV).eval()

# wire frozen embed + lm_head from the target safetensors (as base_trainer does)
idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))
wm = idx["weight_map"]
with safe_open(os.path.join(MODEL, wm["model.embed_tokens.weight"]), framework="pt", device="cpu") as f:
    emb = f.get_tensor("model.embed_tokens.weight")
with safe_open(os.path.join(MODEL, wm["lm_head.weight"]), framework="pt", device="cpu") as f:
    lmh = f.get_tensor("lm_head.weight")
with torch.no_grad():
    model.embed_tokens.weight.copy_(emb.to(model.embed_tokens.weight.dtype))
    model.lm_head.weight.copy_(lmh.to(model.lm_head.weight.dtype))
print("  embed/lm_head wired. fc weight absmax:", finite(model.fc.weight)[1])

# stage hooks
def mk(name):
    def hook(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        if torch.is_tensor(o):
            nf, am = finite(o)
            flag = "  <-- NON-FINITE" if nf else ""
            print(f"  stage {name:24s} non_finite={nf} absmax={am:.3e}{flag}")
    return hook

model.fc.register_forward_hook(mk("fc"))
model.hidden_norm.register_forward_hook(mk("hidden_norm"))
for i, layer in enumerate(model.layers):
    layer.register_forward_hook(mk(f"layer[{i}]"))
model.norm.register_forward_hook(mk("final_norm"))
model.lm_head.register_forward_hook(mk("lm_head"))

# one real batch
ds = CacheDataset(CACHE)
collate = CacheCollator()
batch = collate([ds[0], ds[1]])
batch = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in batch.items()}
batch["input_ids"] = batch["input_ids"].long()  # trainer/collator feeds long ids
print("batch:", {k: tuple(v.shape) for k, v in batch.items() if torch.is_tensor(v)})

print("forward ...")
with torch.no_grad():
    out = model(
        input_ids=batch["input_ids"],
        target_hidden_states=batch["target_hidden_states"],
        loss_mask=batch["loss_mask"],
        target_last_hidden_states=batch["target_last_hidden_states"],
    )
print("draft_logits        ", finite(out.draft_logits))
print("aligned_target_logits", finite(out.aligned_target_logits))
print("block_keep_mask kept blocks:", int(out.block_keep_mask.sum().item()), "/", out.block_keep_mask.numel())
print("eval_mask true:", int(out.eval_mask.sum().item()))
