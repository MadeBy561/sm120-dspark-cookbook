"""Convert a DeepSpec DSpark checkpoint (Qwen3DSparkModel, standard MHA, full vocab) into the
format vLLM's DFlash speculator (`DFlashDraftModel` / qwen3_dflash.py) loads.

vLLM's DFlash load_weights FUSES q/k/v->qkv and gate/up->gate_up at load, so our separate-proj
weights pass through unchanged — this is purely a config remap + a symlink of model.safetensors
(which also carries markov_head.* + confidence_head.* for the markov monkey-patch).

  python3 convert_to_dflash.py SRC_DIR OUT_DIR

Two things to VERIFY at the serve-test (can't be confirmed from config alone):
  1. aux-layer convention: we set dflash_config.target_layer_ids = [t-1 ...] so vLLM's i+1 maps
     back to the DeepSpec target_layer_ids [75,76,77]; confirm the served draft receives the SAME
     aux hidden states it trained on (input-of-layer vs output-of-layer).
  2. draft vocab: DFlash normally uses a reduced draft_vocab_size + t2d/d2t; we set
     draft_vocab_size = vocab_size (full, identity). Confirm DFlash serves a full-vocab draft
     (no t2d/d2t) without error; if it requires them, add identity maps here.
"""
import json
import os
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "/mnt/18tb_r1/dspark-ckpts/deepspec/dspark_block5_glm52_594b/step_100"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/mnt/18tb_r1/dspark-ckpts/dflash_glm52_step100"

cfg = json.load(open(os.path.join(SRC, "config.json")))
src_arch = cfg.get("architectures")
src_tli = cfg.get("target_layer_ids")
print(f"source: arch={src_arch} target_layer_ids={src_tli} vocab={cfg.get('vocab_size')} "
      f"layers={cfg.get('num_hidden_layers')} markov_rank={cfg.get('markov_rank')}")

# --- config remap (keep all dims; change arch + add DFlash/markov fields) ---
cfg["architectures"] = ["DFlashDraftModel"]
# vLLM does `i+1` on dflash_config.target_layer_ids -> target aux layers. Set t-1 so it round-trips
# to the DeepSpec target_layer_ids. (VERIFY input-vs-output convention at serve-test.)
cfg["dflash_config"] = {
    "target_layer_ids": [int(t) - 1 for t in src_tli],
    "use_aux_hidden_state": True,
}
cfg["draft_vocab_size"] = int(cfg["vocab_size"])          # full vocab, identity (no reduction)
cfg["dspark_markov_rank"] = int(cfg.get("markov_rank", 512))  # consumed by the markov monkey-patch
cfg["num_lookahead_tokens"] = int(cfg.get("block_size", 5))   # block_size proposals

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)

# --- weights: pass through via symlink (keeps markov_head.* + confidence_head.*) ---
dst_w = os.path.join(OUT, "model.safetensors")
if os.path.lexists(dst_w):
    os.remove(dst_w)
os.symlink(os.path.join(SRC, "model.safetensors"), dst_w)

print(f"converted -> {OUT}")
print(f"  architectures -> {cfg['architectures']}")
print(f"  dflash_config.target_layer_ids -> {cfg['dflash_config']['target_layer_ids']}  (i+1 => {src_tli})")
print(f"  draft_vocab_size -> {cfg['draft_vocab_size']} (full)  | dspark_markov_rank -> {cfg['dspark_markov_rank']}")
print("  model.safetensors -> symlinked (q/k/v fused at load; markov/confidence weights carried)")
print("NOTE: serve with the model PATH containing 'dflash' (auto-detect) OR --speculative-config method=dflash;")
print("      full DSpark needs the markov monkey-patch (next).")
