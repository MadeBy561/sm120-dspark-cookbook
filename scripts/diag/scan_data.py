"""Scan all cache samples for degeneracy that can poison the DSpark backward:
zero supervised tokens, zero valid anchor candidates, tiny seq_len, non-finite targets.
Then run a forward+backward on the worst offenders to confirm the NaN.

  python3 scan_data.py
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
from deepspec.modeling.dspark.common import build_anchor_candidate_mask  # noqa: E402

CACHE = "/mnt/18tb_r1/dspark-data/cache"

ds = CacheDataset(CACHE)
n = len(ds)
print(f"scanning {n} samples ...")

zero_loss, zero_anchor, tiny, nonfinite = [], [], [], []
seqlens = []
for i in range(n):
    s = ds[i]
    ids = s["input_ids"]
    lm = s["loss_mask"]
    T = ids.shape[0]
    seqlens.append(T)
    valid = build_anchor_candidate_mask(seq_len=T, loss_mask=lm.unsqueeze(0)).sum().item()
    if lm.sum().item() == 0:
        zero_loss.append(i)
    if valid == 0:
        zero_anchor.append(i)
    if T < 8:
        tiny.append(i)
    th = s["target_hidden_states"]
    tl = s["target_last_hidden_states"]
    if (~torch.isfinite(th.float())).any() or (~torch.isfinite(tl.float())).any():
        nonfinite.append(i)

seqlens = torch.tensor(seqlens).float()
print(f"\nseq_len: min={int(seqlens.min())} p1={int(seqlens.kthvalue(max(1,n//100)).values)} "
      f"median={int(seqlens.median())} max={int(seqlens.max())} mean={seqlens.mean():.0f}")
print(f"zero supervised tokens (loss_mask all 0): {len(zero_loss)}  {zero_loss[:10]}")
print(f"zero valid anchor candidates:            {len(zero_anchor)}  {zero_anchor[:10]}")
print(f"tiny seq_len (<8):                       {len(tiny)}  {tiny[:10]}")
print(f"non-finite targets:                      {len(nonfinite)}  {nonfinite[:10]}")

suspects = sorted(set(zero_loss + zero_anchor + tiny))
if not suspects:
    print("\nNo degenerate samples found by cheap scan.")
    sys.exit(0)

# confirm the poison: forward+backward on a batch of suspects
print(f"\n=== confirming poison on {len(suspects)} suspect(s) ===")
import torch.distributed as dist
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29588")
os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "1")
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.modeling.dspark.qwen3.config import build_draft_config
from deepspec.modeling.dspark.loss import compute_dspark_loss

MODEL = "/mnt/models/GLM-5.2-Int8Mix-NVFP4-REAP-594B"


class Args(dict):
    def __getattr__(self, k):
        return self[k]


MA = Args(num_draft_layers=5, target_layer_ids=[75, 76, 77], block_size=5, num_anchors=512,
          markov_rank=512, markov_head_type="vanilla", mask_token_id=154821,
          confidence_head_alpha=1.0, confidence_head_with_markov=True)
dist.init_process_group("nccl", rank=0, world_size=1)
torch.cuda.set_device(0)
cfg = build_draft_config(AutoConfig.from_pretrained(MODEL, trust_remote_code=True), MA)
model = Qwen3DSparkModel(cfg).to(torch.bfloat16).cuda().train()
collate = CacheCollator()

for i in suspects[:6]:
    model.zero_grad(set_to_none=True)
    b = collate([ds[i]])
    b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
    b["input_ids"] = b["input_ids"].long()
    out = model(input_ids=b["input_ids"], target_hidden_states=b["target_hidden_states"],
                loss_mask=b["loss_mask"], target_last_hidden_states=b["target_last_hidden_states"])
    loss = compute_dspark_loss(outputs=out, loss_decay_gamma=4.0, ce_loss_alpha=0.1,
                               l1_loss_alpha=0.9, confidence_head_alpha=1.0)
    lf = int((~torch.isfinite(loss)).sum())
    loss.backward()
    gbad = sum(1 for _, p in model.named_parameters()
               if p.grad is not None and (~torch.isfinite(p.grad.float())).any())
    print(f"  sample[{i}] T={ds[i]['input_ids'].shape[0]} loss={loss.item():.4f} "
          f"loss_nonfinite={lf} params_with_nonfinite_grad={gbad}")
