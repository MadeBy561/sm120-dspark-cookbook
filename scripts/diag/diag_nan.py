"""Diagnose the DSpark training NaN: check (A) captured target hidden states for
non-finite / huge values, and (B) the embed_tokens + lm_head tensors used in the CE path.

  python3 diag_nan.py
"""
import json
import os
import sys

import torch
from safetensors import safe_open

MODEL = "/mnt/models/GLM-5.2-Int8Mix-NVFP4-REAP-594B"
CACHE = "/mnt/18tb_r1/dspark-data/cache"


def stat(name, t):
    t = t.float()
    finite = torch.isfinite(t)
    nf = (~finite).sum().item()
    fin = t[finite]
    mn = fin.min().item() if fin.numel() else float("nan")
    mx = fin.max().item() if fin.numel() else float("nan")
    amax = fin.abs().max().item() if fin.numel() else float("nan")
    print(f"  {name:28s} shape={tuple(t.shape)} dtype-in={t.dtype} "
          f"non_finite={nf} min={mn:+.3e} max={mx:+.3e} absmax={amax:.3e}")
    return nf


print("=== (B) embed_tokens + lm_head ===")
idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))
wm = idx["weight_map"]
for key in ["model.embed_tokens.weight", "lm_head.weight"]:
    shard = os.path.join(MODEL, wm[key])
    with safe_open(shard, framework="pt", device="cpu") as f:
        t = f.get_tensor(key)
        print(f"{key}: raw dtype={t.dtype}")
        stat(key, t)

print("\n=== (A) captured target cache ===")
sys.path.insert(0, "/work/DeepSpec")
from deepspec.data.target_cache_dataset import CacheDataset  # noqa: E402

manifest = json.load(open(os.path.join(CACHE, "manifest.json")))
print("manifest keys:", list(manifest.keys()))
print("num_samples:", manifest.get("num_samples"), "target_layer_ids:",
      manifest.get("target_layer_ids"), "hidden_size:", manifest.get("hidden_size"))

ds = CacheDataset(CACHE)
print("len(ds):", len(ds))
total_nf = 0
for i in range(min(4, len(ds))):
    s = ds[i]
    print(f"\nsample[{i}] keys:", list(s.keys()))
    for k, v in s.items():
        if torch.is_tensor(v):
            if v.dtype in (torch.long, torch.int, torch.int32, torch.int64, torch.bool):
                vmin = v.min().item()
                vmax = v.max().item()
                print(f"  {k:28s} shape={tuple(v.shape)} dtype={v.dtype} "
                      f"min={vmin} max={vmax}")
            else:
                total_nf += stat(k, v)
        else:
            print(f"  {k:28s} = {v}")

print(f"\nTOTAL non-finite in sampled target tensors: {total_nf}")
