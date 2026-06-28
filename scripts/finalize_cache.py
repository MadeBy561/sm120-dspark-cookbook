"""One-off: finalize a capture_hidden_states.py cache dir (local shards + samples.local.idx)
into the format CacheDataset needs (renamed shards + samples.idx + manifest.json), reusing
DeepSpec's own finalize functions. No re-capture — moves are renames on one filesystem.

  python3 finalize_cache.py /mnt/18tb_r1/dspark-data/cache
"""
import glob
import os
import shutil
import sys

sys.path.insert(0, "/work/DeepSpec")
from deepspec.data.target_cache_dataset import (  # noqa: E402
    INDEX_RECORD_SIZE,
    atomic_json_dump,
    build_global_target_cache_shard_map,
    build_target_cache_manifest,
    cleanup_target_cache_tmp_dir,
    finalize_target_cache_index,
    rename_local_target_cache_shards,
    write_target_cache_manifest,
)

OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/18tb_r1/dspark-data/cache"
AUX_LAYERS = [75, 76, 77]
HIDDEN = 6144
MODEL = "/mnt/models/GLM-5.2-Int8Mix-NVFP4-REAP-594B"

if os.path.exists(os.path.join(OUT, "manifest.json")):
    print("Already finalized:", OUT)
    sys.exit(0)

shard_files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(OUT, "shard-local-*.bin")))
idx_path = os.path.join(OUT, "samples.local.idx")
assert shard_files and os.path.exists(idx_path), f"no local shards/idx in {OUT}"
n_samples = os.path.getsize(idx_path) // INDEX_RECORD_SIZE
print(f"local shards={len(shard_files)} samples={n_samples}")

# stage into _tmp/rank_0 (renames, instant on the same fs)
rank_dir = os.path.join(OUT, "_tmp", "rank_0")
os.makedirs(rank_dir, exist_ok=True)
for f in shard_files + ["samples.local.idx"]:
    shutil.move(os.path.join(OUT, f), os.path.join(rank_dir, f))

summary = {
    "global_rank": 0,
    "source_sample_start": 0,
    "source_sample_end": n_samples,
    "num_local_samples": n_samples,
    "num_local_shards": len(shard_files),
    "local_shard_files": shard_files,
}
atomic_json_dump(summary, os.path.join(rank_dir, "summary.json"))
summaries = [summary]

shard_map, shards = build_global_target_cache_shard_map(summaries)
rename_local_target_cache_shards(output_dir=OUT, rank_dir=rank_dir, summary=summary, shard_map=shard_map)
n = finalize_target_cache_index(output_dir=OUT, summaries=summaries, shard_map=shard_map)
manifest = build_target_cache_manifest(
    num_samples=n, shards=shards, target_layer_ids=AUX_LAYERS, hidden_size=HIDDEN,
    extra_fields={"target_model_name_or_path": MODEL, "source": "capture_hidden_states.py"},
)
write_target_cache_manifest(output_dir=OUT, manifest=manifest)
cleanup_target_cache_tmp_dir(OUT)
print(f"FINALIZED {n} samples -> {OUT}")
print("files:", sorted(os.listdir(OUT))[:6], "...")
