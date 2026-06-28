# sm120-dspark-cookbook

Train a **DSpark speculative-decoding draft for GLM-5.2** (DeepSeek-V4 lineage: `GlmMoeDsa`
= MLA + DSA + MoE) on **Blackwell RTX PRO 6000 / sm120** boxes, **1:1 with DeepSeek's
DSpark recipe** — with every script, config, and hard-won gotcha from a full debugging run
already worked out.

This is the cookbook version of a real end-to-end build (capture → validate → train →
serve). Everything here ran on a 4×RTX-PRO-6000 box against a quantized GLM-5.2; it is
written so you can point it at the **full GLM-5.2** on a bigger sm120 box and turn the crank.

> **TL;DR for the impatient:** clone DeepSpec, `git apply deepspec-patches/deepspec-glm.patch`,
> drop in `config_dspark_glm52_full.py`, generate on-policy data, capture hidden states with
> `scripts/capture_hidden_states.py`, `scripts/finalize_cache.py`, **always run
> `scripts/validate_cache.py` before training**, then `train.py`. Read
> [`docs/02-gotchas.md`](docs/02-gotchas.md) first — it will save you the ~16 hours it cost to find these.

---

## What DSpark is (30 seconds)

DSpark is DeepSeek's speculative-decoding method (the successor to DFlash/EAGLE-3). The draft
is a small transformer that **consumes the target model's hidden states** (the residual stream
at a few late layers) and proposes the next `block_size` tokens, which the target then verifies.

- **DFlash backbone** (parallel block predictor) **+ Markov serial head** (rank-256/512 corrector)
  **+ confidence head**. The Markov head is what makes it *DSpark* and not just *DFlash*.
- Trained to match the target's output distribution (0.9·L1 distillation + 0.1·CE + 1.0·confidence).
- Accept length caps at `block_size + 1` (=6 at block 5).

**A draft is bonded to the exact target it was captured from.** A draft trained on full GLM-5.2
will *run* on a REAP/pruned GLM-5.2 but with degraded accept (different hidden states), and vice
versa. **Capture from the model you will actually serve.** (See [gotchas](docs/02-gotchas.md).)

---

## Why this is 1:1 with the paper — and why the GLM changes don't break that

Every difference from DeepSeek's literal config falls into one of four buckets. The
**method-defining knobs are identical**; everything else is either **forced by GLM's architecture**
(same role, different value) or a **training-logistics adaptation** that provably doesn't change
what the draft learns. We followed their research to the letter and only changed what the target
*made* us change.

### 1. Identical to the recipe — the knobs that DEFINE DSpark
Copied straight from DeepSeek's V4-Flash/Pro DSpark configs + DeepSpec's hyperparameters, unchanged:
`block_size=5` · `num_draft_layers=5` · `num_anchors=512` · `markov_rank=512` /
`markov_head_type=vanilla` · confidence head (with markov) · loss `0.9·L1 + 0.1·CE + 1.0·conf`,
`loss_decay_gamma=4.0` · `lr=6e-4`, `warmup_ratio=0.04`, `weight_decay=0`, `max_grad_norm=1.0` ·
`num_train_epochs=10` · on-policy data gen with `--disable-thinking` (their README's own example) ·
**the DSpark model itself** (DFlash backbone + Markov serial head + confidence head) reused
**verbatim** from DeepSpec.
**Why it's 1:1:** these *are* the architecture and the objective. We changed none of them.

### 2. Forced by GLM's architecture — same role, different value (NOT corner-cutting)
| knob | paper (V4) | GLM-5.2 | why it's forced |
|---|---|---|---|
| `target_layer_ids` | last-3 (Flash `[40,41,42]`, Pro `[58,59,60]`) | `[75,76,77]` | GLM has 78 layers; **same last-3 rule**, indices just follow the layer count |
| `mask_token_id` | `128799` (their vocab) | `154821` | **same role** (a reserved/unused id), value comes from GLM's 154880-token vocab |
| `hidden_size` / `vocab_size` | their model's | `6144` / `154880` | inherited — the draft deep-copies the target config; not a choice |

**Why it doesn't disrupt 1:1:** the recipe *specifies* "last-3 layers" and "a reserved mask id" —
we apply those rules to GLM's shape. Using GLM's real layer count and tokenizer is what makes it
faithful; pasting V4's literal `[40,41,42]` / `128799` onto GLM would be the actual deviation.

### 3. Forced by the vLLM / quantized target — same data + weights, different plumbing
DeepSpec's `prepare_target_cache.py` loads the target with transformers `AutoModelForCausalLM`.
A `GlmMoeDsa` (custom arch, frequently quantized) **cannot** load that way. So:
- **Capture** runs through the **vLLM-served** target via per-layer hooks
  (`capture_hidden_states.py` + `dspark_hooks.py`) instead of a transformers forward. The tensors
  are the *same* residual streams at layers 75/76/77 + the final hidden — identical in meaning,
  just read from the served model. *Bonus:* capturing from the actual quantized target makes the
  draft **quant-aware** (it learns the distribution it will serve) — that's *more* correct for that
  target, not a deviation.
- **Model build** (our `base_trainer` patch) reads the frozen `embed_tokens`/`lm_head` straight from
  the safetensors and builds the draft **on-GPU** — the *same* frozen target weights the recipe uses
  (bf16, unquantized in the checkpoint), just a different loader, and it dodges a host-RAM OOM.
- **GLM chat template** added to the parser (DeepSpec ships Qwen/Gemma only). It tokenizes
  conversations correctly and puts the loss on the right tokens — the *right* template is required
  for faithful training; the wrong one would be the deviation.
- **`GlmDSparkTrainer`** is a one-line `pass` subclass — the trainer is target-agnostic.

**Why it doesn't disrupt 1:1:** identical hidden-state data and identical frozen weights flow into
the identical objective. Only the *mechanism that extracts them* changed.

### 4. Training-logistics — scales with hardware/data, zero effect on what's learned
- **`global_batch_size`.** The recipe's 512 is calibrated to their **~1M-sample** dataset → ~19,500
  updates + ~780 warmup steps. On a *small* dataset, literal `gbs=512` gives only ~80 updates with
  ~3 warmup steps — degenerate, and the source of the bf16-accumulation NaN. **Scaling `gbs` down on
  small data reproduces the recipe's training *trajectory*** (right update count + warmup). Batch
  size is the one knob that must track dataset size; keeping it literal on 1/200th the data is the
  real deviation. With your 16-GPU box + abundant data, `gbs=512` (→ accum 32) is faithful — keep it.
- **`torch_compile=false`.** Compile is kernel fusion; the weights it produces are **bit-for-bit
  identical** to eager, and it has **zero** effect on serving speed. We run eager only because
  compile CantSplits on GLM's 64-head flex-attention.
- **GPU count / `no_shard` FSDP / `num_workers`.** Data-parallel replication and dataloading don't
  touch the objective.

**Why it doesn't disrupt 1:1:** none of these change the model or the loss — they change *how fast*
and *how noisily* you reach the same optimum. We set them so the trajectory matches the recipe's
intent for your data and hardware.

> The two **bugs** we fixed (NaN, host-RAM OOM) were artifacts of a constrained box, not the method.
> The fixes *restore* the recipe's intended behavior on small hardware — they don't alter it.
> Every why is documented in [`docs/02-gotchas.md`](docs/02-gotchas.md).

---

## The pipeline

```
 prompts ──▶ (1) GENERATE on-policy answers from the target  (DeepSpec generate_train_data.py)
         ──▶ (2) CAPTURE hidden states                       (scripts/capture_hidden_states.py + dspark_hooks.py)
         ──▶ (3) FINALIZE the cache                          (scripts/finalize_cache.py)
         ──▶ (4) VALIDATE the cache  ◀── DO NOT SKIP         (scripts/validate_cache.py)
         ──▶ (5) TRAIN the DSpark draft                      (DeepSpec train.py + our patch + config)
         ──▶ (6) SERVE on vLLM b12x                          (ds4dspark image; see docs/04-serving.md)
```

Detailed walkthrough: [`docs/01-pipeline.md`](docs/01-pipeline.md).

---

## Hardware reality on sm120 (read this before you size anything)

These cards (RTX PRO 6000 Blackwell, sm120) have **no NVLink**. That changes the math:

- **Multi-GPU data-parallel barely scales for draft training.** Measured on 4×6000:
  1 GPU ≈ 2.2, 2 GPU ≈ 2.3, **4 GPU ≈ 1.6 samples/s** — more GPUs got *slower* (PCIe all-reduce +
  per-step barrier waiting on the slowest variable-length rank). Your 16-GPU box will hit the same
  wall *for the training step*. Where 16 GPUs **do** win big: **parallel data-gen + capture** (run
  many target replicas), which is the real bottleneck at scale.
- **The training step is matmul/FLOP-bound, not attention-bound.** Profiled: `aten::mm` = **65%**
  of GPU time (the 6144→154880 vocab projection, done twice), flex-attention softmax only **6.6%**.
  ⇒ **`torch.compile` is only ~1.3× here** (and it CantSplits on GLM's 64 heads), so don't burn days on it.
  The real speed lever is **`num_anchors`** (linear): 512→128 is ~3.3× faster — but it's a *coverage*
  knob, safe to cut only when you have lots of data. See [gotchas](docs/02-gotchas.md).
- **RAM matters.** `train.py` spawns one process per visible GPU; each builds the draft (whose
  embed+lm_head are [154880,6144]) — naively that OOMs a small host. Our patch builds the draft
  **directly on the GPU**, so host RAM stays low. Keep that patch.

Full scaling notes + the profiler: [`docs/03-hardware-scaling.md`](docs/03-hardware-scaling.md).

---

## Quickstart

### 0. Environment
- A vLLM **b12x** image that has `GlmMoeDsaForCausalLM` registered (to serve the target for capture
  and to serve the draft later). The DSpark-capable one is
  `voipmonitor/vllm:ds4dspark-v7-...` (has `dspark.py` + markov + GLM). Details in
  [`docs/04-serving.md`](docs/04-serving.md).
- The DeepSpec training repo + our patch (below). DeepSpec deps: `pip install tensorboard PyYAML
  matplotlib` on top of the vLLM image — **do not** reinstall its pinned torch/transformers/triton
  (it clobbers vLLM).

### 1. Patch DeepSpec
```bash
git clone https://github.com/deepseek-ai/DeepSpec && cd DeepSpec
git apply /path/to/sm120-dspark-cookbook/deepspec-patches/deepspec-glm.patch
cp /path/to/sm120-dspark-cookbook/deepspec-patches/config_dspark_glm52_full.py config/dspark/
```
What the patch does and why: [`deepspec-patches/README.md`](deepspec-patches/README.md).

### 2. Generate on-policy data (faithful to DeepSpec's example)
Serve the **full GLM-5.2** target on vLLM, then regenerate answers (their README uses
`--disable-thinking`; that is on-recipe):
```bash
python scripts/data/generate_train_data.py --model GLM-5.2 \
  --server-address 127.0.0.1:8000 --concurrency 32 \
  --temperature 0.7 --top-p 0.8 --top-k 20 --min-p 0 --max-tokens 4096 \
  --disable-thinking --resume \
  --input-file-path train_datasets/perfectblend_train.jsonl \
  --output-file-path train_datasets/glm52/regen.jsonl
```
> Storage warning (DeepSpec's own): the hidden-state cache is **huge** (~38 TB for the full
> open-perfectblend set at 3 captured layers). Start with 50–100k samples; that's enough for a
> strong draft. Cache size scales with `#samples × seq_len × hidden × #layers`.

### 3. Capture hidden states (our vLLM path — the NVFP4/quant-aware part)
`prepare_target_cache.py` (transformers `AutoModel`) **cannot** load a `GlmMoeDsa` vLLM
checkpoint. We capture from the **vLLM-served target** instead, via per-layer hooks:
```bash
python scripts/capture_hidden_states.py   # edit MODEL / OUT / prompts at the top
python scripts/finalize_cache.py /path/to/cache
```
See [`docs/01-pipeline.md`](docs/01-pipeline.md) §Capture for the env vars (spawn, insecure-serialization, etc.).

### 4. Validate (never skip)
```bash
python scripts/validate_cache.py /path/to/cache            # report
python scripts/validate_cache.py /path/to/cache --confirm  # + GPU fwd/bwd proof on flagged samples
python scripts/validate_cache.py /path/to/cache --apply    # rewrite samples.idx dropping degenerates
```

### 5. Train
```bash
RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
CUDA_VISIBLE_DEVICES=0,1,2,3,...,15 \
python train.py --config config/dspark/dspark_glm52_full.py \
  --opts data.target_cache_path=/path/to/cache \
  --opts model.mask_token_id=154821 \
  --opts train.global_batch_size=512 \   # accum 32 on 16 GPUs; drop if you see loss=nan
  --opts train.torch_compile=false
```
Watch steps 1–20 for `loss=nan`. If it NaNs, see [gotchas](docs/02-gotchas.md) §NaN — it's almost
always grad-accumulation depth or warmup-too-short, not your data.

### 6. Serve
Load the checkpoint as the speculator on the `ds4dspark` vLLM image. See
[`docs/04-serving.md`](docs/04-serving.md).

---

## Repo map

| path | what |
|---|---|
| `scripts/capture_hidden_states.py` + `dspark_hooks.py` | capture target hidden states from **vLLM** (offline prefill + per-layer hooks) |
| `scripts/finalize_cache.py` | turn local capture shards into the DeepSpec cache format (manifest + samples.idx) |
| `scripts/validate_cache.py` | **the data gate** — fast at-scale validator/filter (reads only the small fields) |
| `scripts/profile_step.py` | profile one training step → find your bottleneck before optimizing |
| `scripts/diag/` | the NaN/degeneracy diagnostics used during the build (handy if training misbehaves) |
| `deepspec-patches/deepspec-glm.patch` | the DeepSpec changes (GLM trainer + chat template + NVFP4-safe model build) |
| `deepspec-patches/config_dspark_glm52_full.py` | the recipe config for **full GLM-5.2** |
| `deepspec-patches/config_dspark_glm52_REAP_reference.py` | the original REAP config (reference) |
| `docs/01-pipeline.md` | step-by-step walkthrough |
| `docs/02-gotchas.md` | **the 16-hours-of-debugging file — read first** |
| `docs/03-hardware-scaling.md` | sm120/no-NVLink scaling, the profiler results, GPU-count guidance |
| `docs/04-serving.md` | which vLLM image, how to wire the draft |

---

## Provenance

Built end-to-end against a quantized GLM-5.2 (`GlmMoeDsa`) on a 4×RTX-PRO-6000 (sm120) box.
The recipe matches DeepSeek's `DeepSeek-V4-Flash-DSpark` / `-Pro-DSpark` configs (block 5,
last-3 aux layers, markov 256/512). The DeepSpec framework is
[`deepseek-ai/DeepSpec`](https://github.com/deepseek-ai/DeepSpec).
