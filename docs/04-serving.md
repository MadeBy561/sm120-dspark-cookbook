# Serving the trained DSpark draft (vLLM b12x)

> **Confidence: SCOPED, not yet served end-to-end.** This page is now code-level (we read the
> serving sources), but the serve-test itself hasn't run yet. The capture → validate → train legs
> are battle-tested; the serving path below is the *correct, viable* one (the obvious alternative is
> a proven dead end — see below), with two config details to confirm when you actually load it.

## The key finding: vLLM's `dspark` method is **MLA-only** — don't use it for a DeepSpec draft

It's tempting to serve via vLLM's `dspark` speculator (`models/deepseek_v4/nvidia/dspark.py`). **It
won't load a DeepSpec draft.** That class (`DeepSeekV4DSparkLayer(DeepseekV4DecoderLayer)`) is
DeepSeek-V4 **MLA** attention (`wq_a`/`wq_b`/`wkv`/`kv_norm`, even in its `_dspark_use_separate_qkv`
mode — that's unfused-MLA, not standard MHA) and its `load_weights` only takes weights **bundled
under `mtp.*`** in the target checkpoint. A DeepSpec draft is the opposite: **standard MHA**
(`q_proj`/`k_proj`/`v_proj`/`o_proj`) in a **separate** `layers.N.*` checkpoint. Wrong attention,
wrong namespace — not a rename.

## The path that works: **DFlash backbone + markov monkey-patch**

`DFlashDraftModel → DFlashQwen3ForCausalLM` (`qwen3_dflash.py`) **is** standard MHA, and its
`load_weights` **fuses `q/k/v→qkv` and `gate/up→gate_up` at load** — so a DeepSpec draft's separate
projections + `fc`/`hidden_norm`/`norm`/`embed_tokens`/`lm_head` pass through nearly unchanged. The
markov head you trained is **identical** to the image's `DSparkMarkovHead` (`markov_w1`+`markov_w2`,
reads `dspark_markov_rank`). So full DSpark = **DFlash + port the markov head**, no image rebuild.

**(1) Convert** — `scripts/convert_to_dflash.py` (config remap + weight symlink):
```bash
python3 scripts/convert_to_dflash.py /path/to/deepspec/step_NNN /path/to/dflash_out
```
It sets `architectures=["DFlashDraftModel"]`, `dflash_config.target_layer_ids = [t-1 for t in
target_layer_ids]` (vLLM does `i+1` → your `[75,76,77]`; **`speculative.py:407`**), and
`draft_vocab_size = vocab_size` (full, identity). Weights are symlinked (the markov + confidence
tensors ride along for the patch).

**Two things to confirm at the serve-test** (can't be verified from config alone):
- **aux-layer convention** — DeepSpec captures the residual *entering* layers 75/76/77; confirm the
  served draft receives the *same* aux hidden it trained on (input-of-layer vs output-of-layer; the
  `i+1` is set to round-trip, but verify).
- **full vocab** — DFlash usually uses a *reduced* `draft_vocab_size` + `t2d`/`d2t` maps; we serve
  full-vocab (no maps). If your build requires the maps, add identity ones in the converter.

**(2) Markov monkey-patch** — a mounted `.py` on `PYTHONPATH` (no image rebuild) that adds
`DSparkMarkovHead` to the DFlash proposer and wires the per-block markov correction into its draft
sampling. (Build + validate this at the serve-test; the DFlash backbone alone serves first for a
DFlash-level accept number, then markov for full DSpark.)

## EAGLE3 is *also* a dead end on an MLA target in vLLM
A generic EAGLE3 Llama head (e.g. `AQ-MedAI/GLM-5.1-eagle3`) **loads** (dims match) but **can't
serve**: the MLA target requires `fp8_ds_mla` KV, and no standard attention backend supports that
for the Llama draft (vLLM resolves KV dtype globally from the target). It "works" only in llama.cpp.
This is why DSpark-via-DFlash (built around the standard-MHA draft) is the right path, not a
bolt-on EAGLE3.

## Which image
| image | serves GLM (`GlmMoeDsa`) | DFlash | markov head to port |
|---|---|---|---|
| `voipmonitor/vllm:eldritch-...` | ✅ | ✅ (`qwen3_dflash.py`) | ❌ |
| `voipmonitor/vllm:ds4dspark-v7-...` | ✅ | ✅ | ✅ (`DSparkMarkovHead` to copy) |

Use **`ds4dspark`** — it has DFlash *and* the `DSparkMarkovHead` source for the patch. Verify:
```bash
docker run --rm --entrypoint bash <image> -c '
  V=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
  grep -rl "class DFlashQwen3ForCausalLM" $V/model_executor/models/   # DFlash serving
  grep -rl "class DSparkMarkovHead" $V/models/ $V/model_executor/      # markov head to port
'
```

## sm120 / TP / DCP caveats
Serving a Llama-style draft alongside the MLA target: **don't force a global `--attention-backend`**
(the MLA backend is invalid for the standard-attention draft — let vLLM auto-select per model), and
**DCP works on B12X** (FlashInfer-MLA-sparse threw a DCP-vs-draft-KV-heads assertion; B12X didn't).
Confirm the target boots at your TP/DCP before wiring the draft.

## Reminder
The draft is bonded to the target it was **captured from**. Serve it on that same model (full
GLM-5.2 → the full-GLM draft); a REAP-captured draft under-accepts on full GLM-5.2 and vice-versa.
