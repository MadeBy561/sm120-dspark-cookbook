"""Profile ONE DSpark training step (fwd+loss+bwd) to find the per-sample bottleneck:
eager flex_attention vs the 154880-vocab lm_head projections vs MLP. Grounds the
efficiency plan (compile? fewer anchors? chunked vocab?).

  python3 profile_step.py
"""
import json
import os
import sys

import torch
import torch.distributed as dist
from safetensors import safe_open

os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29611")
os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "1")
sys.path.insert(0, "/work/DeepSpec")
from transformers import AutoConfig  # noqa: E402
from deepspec.data.target_cache_dataset import CacheDataset  # noqa: E402
from deepspec.data import CacheCollator  # noqa: E402
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel  # noqa: E402
from deepspec.modeling.dspark.qwen3.config import build_draft_config  # noqa: E402
from deepspec.modeling.dspark.loss import compute_dspark_loss  # noqa: E402

MODEL = "/mnt/models/GLM-5.2-Int8Mix-NVFP4-REAP-594B"
CACHE = "/mnt/18tb_r1/dspark-data/cache"


class A(dict):
    def __getattr__(self, k): return self[k]


def build(num_anchors):
    MA = A(num_draft_layers=5, target_layer_ids=[75, 76, 77], block_size=5, num_anchors=num_anchors,
           markov_rank=512, markov_head_type="vanilla", mask_token_id=154821,
           confidence_head_alpha=1.0, confidence_head_with_markov=True)
    cfg = build_draft_config(AutoConfig.from_pretrained(MODEL, trust_remote_code=True), MA)
    model = Qwen3DSparkModel(cfg).to(torch.bfloat16).cuda().train()
    idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
    for nm, attr in [("model.embed_tokens.weight", model.embed_tokens), ("lm_head.weight", model.lm_head)]:
        with safe_open(os.path.join(MODEL, idx[nm]), framework="pt", device="cuda") as f:
            attr.weight.data.copy_(f.get_tensor(nm).to(torch.bfloat16))
    model.embed_tokens.weight.requires_grad_(False); model.lm_head.weight.requires_grad_(False)
    return model


def one_step(model, b):
    out = model(input_ids=b["input_ids"], target_hidden_states=b["target_hidden_states"],
                loss_mask=b["loss_mask"], target_last_hidden_states=b["target_last_hidden_states"])
    loss = compute_dspark_loss(outputs=out, loss_decay_gamma=4.0, ce_loss_alpha=0.1,
                               l1_loss_alpha=0.9, confidence_head_alpha=1.0)
    loss.backward()
    model.zero_grad(set_to_none=True)
    return loss


dist.init_process_group("nccl", rank=0, world_size=1); torch.cuda.set_device(0)
ds = CacheDataset(CACHE); collate = CacheCollator()
# pick a median-ish sample (~600 tok) for a representative step
b = collate([ds[0]]); b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
b["input_ids"] = b["input_ids"].long()
print(f"sample seq_len={b['input_ids'].shape[1]}")


def timed(model, label, iters=5):
    for _ in range(2):  # warmup
        one_step(model, b)
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        one_step(model, b)
    en.record(); torch.cuda.synchronize()
    ms = st.elapsed_time(en) / iters
    print(f"  {label:28s} {ms:8.1f} ms/step  ({1000.0/ms:.2f} samples/s, 1 GPU)")
    return ms


for na in [512, 128, 64]:
    print(f"\n=== num_anchors={na} (eager) ===")
    m = build(na)
    timed(m, f"anchors={na}")
    del m; torch.cuda.empty_cache()

# detailed op breakdown at the real config (512) via torch.profiler
print("\n=== op breakdown (num_anchors=512, eager) — top CUDA ops ===")
from torch.profiler import profile, ProfilerActivity  # noqa: E402
m = build(512)
for _ in range(2):
    one_step(m, b)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
        one_step(m, b)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=14))
