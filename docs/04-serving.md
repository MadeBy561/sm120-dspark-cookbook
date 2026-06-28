# Serving the trained DSpark draft (vLLM b12x)

## You need an image with BOTH GLM support AND DSpark

Two distinct capabilities, and not every image has both:

| image | `GlmMoeDsaForCausalLM` (serves GLM) | DFlash | full DSpark (markov) |
|---|---|---|---|
| `voipmonitor/vllm:eldritch-...` | ✅ | ✅ (`qwen3_dflash.py`) | ❌ (no `dspark.py`/markov) |
| `voipmonitor/vllm:ds4dspark-v7-...` | ✅ (registered) | ✅ | ✅ (`dspark.py`, `dspark/`, markov, `dspark_sparse_attn_tilelang.py`) |

**Use the `ds4dspark` image** to serve the full DSpark draft — it has GLM *and* the markov head.
The `eldritch` image can serve the **DFlash backbone only** (no markov), which is a fine *first*
accept measurement if your serving image story isn't sorted yet (DFlash now, markov later — that's
the documented DeepSeek path too).

Verify an image quickly:
```bash
docker run --rm --entrypoint bash <image> -c '
  V=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
  grep -l "class GlmMoeDsaForCausalLM" $V/model_executor/models/*.py     # GLM support
  find $V -iregex ".*\(dflash\|dspark\|markov\).*" -printf "%f\n" | sort  # speculator support
'
```

## sm120 / TP / DCP caveats

If you serve a large GLM on sm120 with FlashInfer-MLA-sparse at `TP>4` / DCP, watch for the
**head-chunk** issue: gathering >32 heads can overflow the FlashInfer workspace (illegal address).
The fix is to run the heads in contiguous 32-head chunks (an env like `FIMLA_HEAD_CHUNK=32` in
images that carry the patch). Confirm your image boots the target at your TP/DCP **before** wiring
the draft. (For the full bf16/fp8 GLM-5.2 on a fat box you may not hit this; it bit a 4-GPU
DCP4/TP4 quantized setup.)

## Wiring the draft

The DSpark checkpoint holds the DFlash backbone + markov head + confidence head. Serve it as the
speculative model via vLLM's speculative config pointing at the DSpark speculator (the
`ds4dspark` image registers it; check `vllm/model_executor/models/dspark.py` and the
`spec_decode` proposer entry in your exact build for the expected `method`/class name and the
config keys it wants — they vary by build).

Then read **per-position accept-length** from vLLM's spec-decode metrics on a few prompts, and
compare to the target's **stock MTP** baseline. That delta is the whole point of the draft.

## Reminder

The draft is bonded to the target it was **captured from**. Serve it on that same model (full
GLM-5.2 → the full-GLM draft). A draft captured from a REAP/pruned model will under-accept here,
and vice-versa (see README "What DSpark is" + gotchas).
