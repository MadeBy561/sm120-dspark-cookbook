"""Stage 2: train-mode forward + real compute_dspark_loss + backward, to prove whether
the NaN is in the loss forward or the backward (fully-masked-row gradients).

  python3 repro_nan2.py
"""
import json
import os
import sys

import torch
import torch.distributed as dist
from safetensors import safe_open

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29577")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

sys.path.insert(0, "/work/DeepSpec")
from transformers import AutoConfig  # noqa: E402

from deepspec.data.target_cache_dataset import CacheDataset  # noqa: E402
from deepspec.data import CacheCollator  # noqa: E402
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel  # noqa: E402
from deepspec.modeling.dspark.qwen3.config import build_draft_config  # noqa: E402
from deepspec.modeling.dspark.loss import compute_dspark_loss  # noqa: E402

MODEL = "/mnt/models/GLM-5.2-Int8Mix-NVFP4-REAP-594B"
CACHE = "/mnt/18tb_r1/dspark-data/cache"
DEV = "cuda:0"


class Args(dict):
    def __getattr__(self, k):
        return self[k]


MODEL_ARGS = Args(
    num_draft_layers=5, target_layer_ids=[75, 76, 77], block_size=5, num_anchors=512,
    markov_rank=512, markov_head_type="vanilla", mask_token_id=154821,
    confidence_head_alpha=1.0, confidence_head_with_markov=True,
)


def nf(t):
    return int((~torch.isfinite(t.float())).sum().item())


dist.init_process_group("nccl", rank=0, world_size=1)
torch.cuda.set_device(0)

target_config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
draft_config = build_draft_config(target_config, MODEL_ARGS)
model = Qwen3DSparkModel(draft_config).to(torch.bfloat16).to(DEV).train()

idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))
wm = idx["weight_map"]
with safe_open(os.path.join(MODEL, wm["model.embed_tokens.weight"]), framework="pt", device="cpu") as f:
    emb = f.get_tensor("model.embed_tokens.weight")
with safe_open(os.path.join(MODEL, wm["lm_head.weight"]), framework="pt", device="cpu") as f:
    lmh = f.get_tensor("lm_head.weight")
with torch.no_grad():
    model.embed_tokens.weight.copy_(emb.to(model.embed_tokens.weight.dtype))
    model.lm_head.weight.copy_(lmh.to(model.lm_head.weight.dtype))
model.embed_tokens.weight.requires_grad_(False)
model.lm_head.weight.requires_grad_(False)

ds = CacheDataset(CACHE)
collate = CacheCollator()
batch = collate([ds[0], ds[1]])
batch = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in batch.items()}
batch["input_ids"] = batch["input_ids"].long()

out = model(
    input_ids=batch["input_ids"],
    target_hidden_states=batch["target_hidden_states"],
    loss_mask=batch["loss_mask"],
    target_last_hidden_states=batch["target_last_hidden_states"],
)
loss = compute_dspark_loss(
    outputs=out, loss_decay_gamma=4.0, ce_loss_alpha=0.1, l1_loss_alpha=0.9,
    confidence_head_alpha=1.0,
)
print(f"LOSS forward: value={loss.item():.6f} non_finite={nf(loss)}")

loss.backward()
bad = []
for name, p in model.named_parameters():
    if p.grad is not None and nf(p.grad) > 0:
        bad.append((name, nf(p.grad), p.grad.numel()))
print(f"params with NON-FINITE grad: {len(bad)}")
for name, n, tot in bad[:20]:
    print(f"  {name:50s} non_finite={n}/{tot}")
if not bad:
    print("  (all gradients finite -- NaN is NOT in this single backward)")
