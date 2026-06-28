"""Validate (and optionally filter) a DSpark target cache so no degenerate sample can
poison the training backward — at any scale, including the ~38 TB full runs.

It reads ONLY each sample's `seq_len`, `loss_mask` (uint8) and `input_ids` (int32) via the
fixed-size index records + shard offsets. It never reads the multi-GB hidden-state tensors,
so a full-corpus validate is seconds of small mmap reads, not a 38 TB scan.

Degeneracy criteria (any → drop):
  * seq_len            < MIN_SEQ_LEN          (too short to form context + one block)
  * supervised tokens  < MIN_SUPERVISED       (loss_mask.sum(); nothing to learn from)
  * valid anchors      < MIN_VALID_ANCHORS    (positions i with loss_mask[i]&loss_mask[i+1];
                                               an anchor needs its first draft target supervised)
  * input_ids out of [0, vocab)               (corrupt token → embedding / CE blow-up)

Usage:
  python3 validate_cache.py CACHE                 # report only
  python3 validate_cache.py CACHE --confirm       # + GPU forward/backward proof on flagged + control
  python3 validate_cache.py CACHE --apply         # rewrite samples.idx dropping degenerate (backs up)
"""
import argparse
import json
import mmap
import os
import shutil
import struct
import sys

import numpy as np

REC = struct.Struct("<QIIQQQQQ")
REC_FIELDS = [
    "sample_id", "shard_id", "seq_len", "input_ids_offset", "attention_mask_offset",
    "loss_mask_offset", "target_hidden_states_offset", "target_last_hidden_states_offset",
]

# Invariants. Samples 0/1 (441 tok, ~hundreds of valid anchors) train clean; the danger is the
# tail with near-zero supervised tokens. Defaults are conservative and cheap to satisfy.
MIN_SEQ_LEN = 16
MIN_SUPERVISED = 8
MIN_VALID_ANCHORS = 8


def load_manifest(cache):
    with open(os.path.join(cache, "manifest.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def shard_paths(cache, manifest):
    out = {}
    for sh in manifest["shards"]:
        out[int(sh["shard_id"])] = os.path.join(cache, sh["file_name"])
    return out


def classify(cache):
    """Fast pass: return (records, flags) where flags[i] is None (keep) or a drop-reason str."""
    manifest = load_manifest(cache)
    vocab = None  # filled from a config if available; else skip range check
    idx_path = os.path.join(cache, "samples.idx")
    raw = open(idx_path, "rb").read()
    n = len(raw) // REC.size
    assert len(raw) % REC.size == 0, "corrupt idx (size not a multiple of record size)"
    paths = shard_paths(cache, manifest)
    handles = {sid: open(p, "rb") for sid, p in paths.items()}  # keep alive for mmap fds
    mm = {sid: mmap.mmap(h.fileno(), 0, access=mmap.ACCESS_READ) for sid, h in handles.items()}

    records, flags = [], []
    stats = {"seq_len": [], "supervised": [], "valid_anchors": []}
    for i in range(n):
        rec = dict(zip(REC_FIELDS, REC.unpack_from(raw, i * REC.size)))
        records.append(rec)
        T = int(rec["seq_len"])
        sh = mm[int(rec["shard_id"])]
        loss_mask = np.frombuffer(sh, dtype=np.uint8, count=T, offset=int(rec["loss_mask_offset"])).copy()
        ids = np.frombuffer(sh, dtype=np.int32, count=T, offset=int(rec["input_ids_offset"])).copy()
        supervised = int((loss_mask > 0).sum())
        if T >= 2:
            lm = loss_mask > 0
            valid_anchors = int((lm[:-1] & lm[1:]).sum())
        else:
            valid_anchors = 0
        stats["seq_len"].append(T)
        stats["supervised"].append(supervised)
        stats["valid_anchors"].append(valid_anchors)

        reason = None
        if T < MIN_SEQ_LEN:
            reason = "short_seq"
        elif supervised < MIN_SUPERVISED:
            reason = "few_supervised"
        elif valid_anchors < MIN_VALID_ANCHORS:
            reason = "few_anchors"
        elif int(ids.min()) < 0 or (vocab is not None and int(ids.max()) >= vocab):
            reason = "bad_token"
        flags.append(reason)
    for sh in mm.values():
        sh.close()
    for h in handles.values():
        h.close()
    return records, flags, stats, manifest


def report(flags, stats):
    n = len(flags)
    from collections import Counter
    reasons = Counter(f for f in flags if f is not None)
    keep = sum(1 for f in flags if f is None)
    def pct(a, b):
        return f"{100.0 * a / b:.2f}%" if b else "0%"
    for k in ("seq_len", "supervised", "valid_anchors"):
        arr = np.array(stats[k])
        print(f"  {k:14s} min={arr.min():6d} p1={int(np.percentile(arr,1)):6d} "
              f"median={int(np.median(arr)):6d} max={arr.max():6d} mean={arr.mean():8.1f}")
    print(f"\n  total={n}  keep={keep} ({pct(keep,n)})  drop={n-keep} ({pct(n-keep,n)})")
    for reason, c in reasons.most_common():
        print(f"    drop[{reason:14s}] = {c}")
    print(f"\n  thresholds: MIN_SEQ_LEN={MIN_SEQ_LEN} MIN_SUPERVISED={MIN_SUPERVISED} "
          f"MIN_VALID_ANCHORS={MIN_VALID_ANCHORS}")
    return [i for i, f in enumerate(flags) if f is not None]


def apply_filter(cache, records, flags, manifest):
    idx_path = os.path.join(cache, "samples.idx")
    backup = idx_path + ".prefilter"
    if not os.path.exists(backup):
        shutil.copy2(idx_path, backup)
        print(f"  backed up original idx -> {backup}")
    keep = [(rec) for rec, f in zip(records, flags) if f is None]
    # rewrite dense, renumbering sample_id to position (CacheDataset asserts sample_id == index);
    # shard_id + offsets are preserved (they point to the real data; dropped samples leave holes).
    buf = bytearray()
    for new_id, rec in enumerate(keep):
        buf += REC.pack(new_id, int(rec["shard_id"]), int(rec["seq_len"]),
                        int(rec["input_ids_offset"]), int(rec["attention_mask_offset"]),
                        int(rec["loss_mask_offset"]), int(rec["target_hidden_states_offset"]),
                        int(rec["target_last_hidden_states_offset"]))
    tmp = idx_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(buf)
    os.replace(tmp, idx_path)
    manifest["num_samples"] = len(keep)
    with open(os.path.join(cache, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  rewrote samples.idx: {len(records)} -> {len(keep)} samples; manifest num_samples updated")


def confirm(cache, flags):
    """Prove the mechanism: forward+backward on flagged (degenerate) vs control (good) samples."""
    import torch
    import torch.distributed as dist
    sys.path.insert(0, "/work/DeepSpec")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29599")
    os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "1")
    from transformers import AutoConfig
    from deepspec.data.target_cache_dataset import CacheDataset
    from deepspec.data import CacheCollator
    from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
    from deepspec.modeling.dspark.qwen3.config import build_draft_config
    from deepspec.modeling.dspark.loss import compute_dspark_loss

    MODEL = "/mnt/models/GLM-5.2-Int8Mix-NVFP4-REAP-594B"

    class A(dict):
        def __getattr__(self, k): return self[k]
    MA = A(num_draft_layers=5, target_layer_ids=[75, 76, 77], block_size=5, num_anchors=512,
           markov_rank=512, markov_head_type="vanilla", mask_token_id=154821,
           confidence_head_alpha=1.0, confidence_head_with_markov=True)
    dist.init_process_group("nccl", rank=0, world_size=1); torch.cuda.set_device(0)
    cfg = build_draft_config(AutoConfig.from_pretrained(MODEL, trust_remote_code=True), MA)
    model = Qwen3DSparkModel(cfg).to(torch.bfloat16).cuda().train()
    ds = CacheDataset(cache); collate = CacheCollator()
    flagged = [i for i, f in enumerate(flags) if f is not None][:4]
    good = [i for i, f in enumerate(flags) if f is None][:4]
    for tag, idxs in (("DEGENERATE", flagged), ("CONTROL(good)", good)):
        for i in idxs:
            model.zero_grad(set_to_none=True)
            b = collate([ds[i]]); b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
            b["input_ids"] = b["input_ids"].long()
            out = model(input_ids=b["input_ids"], target_hidden_states=b["target_hidden_states"],
                        loss_mask=b["loss_mask"], target_last_hidden_states=b["target_last_hidden_states"])
            loss = compute_dspark_loss(outputs=out, loss_decay_gamma=4.0, ce_loss_alpha=0.1,
                                       l1_loss_alpha=0.9, confidence_head_alpha=1.0)
            lf = int((~torch.isfinite(loss)).sum()); loss.backward()
            gbad = sum(1 for _, p in model.named_parameters()
                       if p.grad is not None and (~torch.isfinite(p.grad.float())).any())
            print(f"  {tag:14s} sample[{i}] loss={loss.item():.4f} loss_nonfinite={lf} grad_nonfinite_params={gbad}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args()
    print(f"validating {args.cache} ...", flush=True)
    records, flags, stats, manifest = classify(args.cache)
    flagged = report(flags, stats)
    if args.confirm:
        print("\n=== forward/backward proof (flagged vs control) ===", flush=True)
        confirm(args.cache, flags)
    if args.apply:
        print("\n=== applying filter ===", flush=True)
        apply_filter(args.cache, records, flags, manifest)
    else:
        print(f"\n(dry-run; {len(flagged)} would be dropped. add --apply to rewrite idx)")


if __name__ == "__main__":
    main()
