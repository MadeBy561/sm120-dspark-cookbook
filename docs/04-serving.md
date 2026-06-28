# Serving the trained DSpark draft on vLLM (MLA target)

> **Confidence: SERVED END-TO-END.** A standard-MHA DSpark/DFlash draft now boots and **generates
> tokens** on a GLM-5.2 (MLA) target in vLLM, via four monkey-patches (no image rebuild). An earlier
> version of this page claimed *"DFlash avoids the EAGLE3 KV wall"* — **that was wrong.** A DFlash
> draft is *also* standard multi-head attention, so it hits the **same** wall EAGLE3 does. The
> difference is that the wall is **patchable** — and the patches are below.

## Why an MHA draft fights an MLA target (the root cause)

GLM-5.2's target uses **MLA** attention: its KV cache is a single compact latent (~576 B/token,
`fp8_ds_mla` format). The open DeepSpec recipe builds a **standard-MHA** draft (`q/k/v_proj`, full
heads). vLLM's MLA-serving path was written assuming the draft is *also* MLA (DeepSeek's V4 draft
is). So an MHA draft trips three assumptions, each of which is fixable:

1. **KV dtype.** vLLM resolves `kv_cache_dtype` *globally* from the MLA target → hands the MHA draft
   `fp8_ds_mla`, which **no non-MLA attention backend supports**. *(This is the exact wall that
   "killed" EAGLE3 — it was never EAGLE3-specific.)*
2. **Load OOM.** The target nearly fills each GPU; `fastsafetensors`' sharded NCCL shuffle
   materializes each full (unsharded) draft tensor on rank-0 → OOM next to the target.
3. **KV-page grouping.** The MHA draft's KV page (e.g. `16 heads × 64 × 2(K,V) × 2 bytes = 2048`
   elem/token) is **much larger** than the MLA latent page (576 B/token), so vLLM can't pad it into
   an MLA "bucket" → `AssertionError: max(sm_page_sizes) <= max(all_page_sizes)`.

## The recipe: 4 monkey-patches + the converter

All four live in **`scripts/serve_patch/sitecustomize.py`** (auto-imported by every vLLM process,
including the TP workers, when its directory is on `PYTHONPATH` — no image rebuild) plus
**`scripts/convert_to_dflash.py`**:

| # | patch | what it does |
|---|---|---|
| 1 | `load_dflash_model` honors **`draft_kv_cache_dtype`** | the stock dflash loader ignores it; the patch gives the MHA draft its **own bf16 KV** (set `draft_kv_cache_dtype: "auto"` in `--speculative-config`) instead of the target's `fp8_ds_mla`. Set `draft_attention_backend: "FLASH_ATTN"` so the standard-attn draft gets a non-MLA backend. |
| 2 | draft loads with **`load_format="safetensors"`** | the patch overrides the draft's loader to the standard CPU-staged one (no rank-0 GPU spike from `fastsafetensors`' shuffle). |
| 3 | **strip** `embed_tokens` + `confidence_head` (+ `markov` for the DFlash-only backbone) | in the converter. `embed_tokens` is shared from the target (DFlash `load_weights` auto-skips it when absent); `confidence_head` is a training-only head vLLM never serves. Pass `KEEP_MARKOV=1` to retain the markov weights for full DSpark. **Also set `dflash_config.mask_token_id` = the training mask token** — `get_parallel_drafting_token_id` raises without it. |
| 4 | `_get_kv_cache_groups_uniform_groups` **own-group fallback** | on the page-size `AssertionError`, emit the draft as its **own** `KVCacheGroupSpec` (its native page) instead of forcing it into an MLA bucket → heterogeneous KV groups coexist. |

Then **size `--max-model-len` to fit two KV pools** (target MLA + draft MHA). The dual pools cost
memory, so you'll run a shorter context than the target's native max — vLLM tells you the ceiling
(`estimated maximum model length is N`); set `max_model_len` at or below it (we used 16k).

Boot with `serving/docker-compose.dflash-example.yml` (mounts `serve_patch/` on `PYTHONPATH`).

## Honest limits — read before you invest

- **Accept is training-driven, not architecture-driven.** An undertrained draft accepts ~nothing →
  it's *slower* than no draft (it runs every step for ~0 benefit). Finish training before judging.
- **The MHA draft needs its OWN KV pool** (it can't share the MLA latent). At long context that pool
  is large (≈5 GB/GPU at 245k tokens with a 5-layer draft) → you **lose context**. This is the MHA
  path's real cost.
- **Speed ceiling is the *target's* `forward_rate`, not the draft.** `t/s = forward_rate ×
  accept_length`. `forward_rate` is set by HBM bandwidth ÷ active params (and scales with GPU count);
  a draft only lifts `accept_length`. So the same draft that gives, say, 300 t/s on a 7.6B-active
  model gives roughly `300 ÷ (your_active_params / 7.6B)` on a heavier one. Pick your target and GPU
  count with that formula in mind — a heavy model on few GPUs caps low no matter how good the draft.
- **MHA vs MLA draft.** MHA **serves** (via these patches) but costs context (separate KV pool). An
  **MLA-backbone** draft shares the target's latent KV → no extra pool, full context, native
  `dspark.py` serving (no patches) — but it's a *different draft architecture*, and DeepSeek's
  `dspark.py` loader is V4-specific (a GLM target needs its own MLA-draft serving path). Same speed
  ceiling either way. **Choose MHA for short-context + zero-retrain; choose MLA only if you need long
  context with the draft** (it's a context optimization, not a speed one).

## Which image
The image must contain `GlmMoeDsa` (the target), the **DFlash speculator** (`qwen3_dflash.py`,
`DFlashSpeculator`) for the standard-MHA draft, and **`DSparkMarkovHead`** (in `dspark.py`) to port
the markov head. The plain DFlash backbone serves first (a DFlash-level accept number); add the
markov head for full DSpark.

## Reminder
The draft is bonded to the target it was **captured from**. Serve it on that exact model — a draft
captured from one checkpoint under-accepts on a different one.
