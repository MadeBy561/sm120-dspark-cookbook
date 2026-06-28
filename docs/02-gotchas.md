# Gotchas — the 16-hours file

Every one of these cost real time to find. Organized **symptom → cause → fix**. If your
training misbehaves, it is almost certainly one of these, not a flaw in DSpark.

---

## A. `loss=nan` — TWO distinct failure modes (the big one)

There are **two** NaN mechanisms with **different causes and different fixes**. You need both: the
config handles Mode 1, the patch handles Mode 2.

### Mode 1 — NaN *from step 1*: deep grad-accum + too-short warmup

**Symptom:** `loss=nan` from the very first logged step and never recovers.

**What it is NOT** (we ruled these out, in order — don't re-walk them):
- **Not the data.** Built `validate_cache.py` to scan every sample (finite check + supervised-token
  + valid-anchor counts). 0/4607 degenerate. A single fwd+bwd on real samples is clean (loss ~3.5,
  zero non-finite grads).
- **Not the optimizer.** DeepSpec's `BF16Optimizer` is correct — it keeps **fp32 master params** and
  runs AdamW on them (no eps-underflow). Verified.
- **Not the dist setup.** `init_dist` reads `RANK`/`WORLD_SIZE` as **node-level** (multi-node) and
  derives per-GPU rank from `device_count()`. So `RANK=0 WORLD_SIZE=1` on a single machine is
  *correct* (1 node, N GPUs → world_size=N).

**The actual cause — deep bf16 gradient accumulation + too-short warmup, both from large `gbs` on
small data.** The training loop accumulates gradients in the **bf16** `param.grad` across
`grad_accum` micro-batches under FSDP `no_sync()` before the reduce. At `gbs=512` on few GPUs that's
**accum=256** in bf16 → precision collapse → non-finite. *Confounded with it:* large `gbs` on a small
dataset also means very few total steps, so `warmup_ratio=0.04` yields only ~3 warmup steps → the lr
slams to full → divergence. Both push the same direction.

**Bisection that proves it:**
| GPUs | gbs | accum | result |
|---|---|---|---|
| 1 | 8 | 8 | clean |
| 2 | 16 | 8 | clean |
| 2 | 32 | 16 | clean |
| 2 or 4 | 512 | 256/128 | **NaN** |

**Fix:** keep grad-accumulation **shallow**. Choose `gbs` so `gbs/(world_size·local_batch)` is small
(≤ ~32). On 16 GPUs, `gbs=512` → accum 32 (likely fine — watch steps 1–20). On few GPUs or small
data, drop `gbs` (e.g. 16–128). See the README "Fidelity §4" for why this is *not* a deviation: on
small data, small `gbs` reproduces the recipe's update-count + warmup trajectory.

**If you want literal `gbs=512` on few GPUs:** implement **fp32 gradient accumulation** (accumulate
into an fp32 buffer instead of the bf16 `param.grad`). ~20 lines in the training loop. Not needed if
your accum depth is already shallow.

### Mode 2 — NaN *after clean training*: a transient bad batch (the sneaky one)

**Symptom:** trains **clean for hundreds of steps** (loss descending then stable), then **suddenly**
`loss=nan` and stays there. The loss was *not* diverging — it spiked from one step to the next.
(Observed on a clean accum-8 run: fine to step 210 at loss 2.59, **nan at step 220**.)

**Cause:** a single bad batch / numerical spike — e.g. as the draft converges, its logits sharpen
and a softmax in the loss overflows → an **inf/nan gradient**. `clip_grad_norm_` keeps it non-finite,
then `optimizer.step()` writes NaN into **every** weight → nan forever. **Shallow accumulation does
NOT prevent this** — it isn't an accumulation-precision problem, it's one poisoned step. Mode-1 fixes
won't save you here.

**Fix (in `deepspec-glm.patch`):** a **non-finite-gradient guard** in the training loop — when
`grad_norm` is inf/nan, **skip** `optimizer.step()`, zero the grads, and continue. Last-good weights
are preserved; you lose exactly one batch. This is standard EAGLE/spec-decode training hygiene, and
DeepSpec's loop ships without it. After the patch you'll see occasional
`[nan-guard] step N: ... skipped` lines instead of a dead run.

**If skips become frequent** (many guarded steps, training stalls), the model is genuinely unstable —
lower `lr` or clamp the logits in the loss. Rare skips (1 in hundreds) are harmless.

**Diagnostics to reach for:** `scripts/diag/repro_nan2.py` (one fwd+bwd, prints non-finite grads),
`scripts/diag/scan_data.py` (per-sample), `scripts/validate_cache.py --confirm` (fwd+bwd on flagged).

---

## B. `SIGKILL` during model build on a multi-GPU box (host-RAM OOM)

**Symptom:** `torch.multiprocessing.spawn ... process N terminated with signal SIGKILL`, before
training starts. Worse with more GPUs.

**Cause:** `train.py` spawns **one process per visible GPU**. Each builds `Qwen3DSparkModel`, whose
`embed_tokens` + `lm_head` are **[vocab, hidden] = [154880, 6144]** each. Built **on CPU in fp32**
(nn defaults) that's ~3.8 GB × 2 per process, plus loading the frozen target embed/lm_head — ~8–11 GB
host RAM **per process**. N processes blow past a small host (we had 62 GB; 4 procs OOM'd).

**Fix (in our `deepspec-glm.patch`):** build the draft **directly on the GPU**
(`with torch.device(self.device): draft_model = ...`) and load the frozen target `embed`/`lm_head`
**straight to the GPU** via `safe_open(..., device=cuda)` wrapped in a `SimpleNamespace` (skips the
fp32 `nn.Embedding`/`nn.Linear` intermediates). Host RAM then stays low (~38 GB for 4 procs). Keep
this patch even on a big-RAM box — it's free and avoids the trap entirely.

---

## C. "Why isn't 16 GPUs faster?" — sm120 has no NVLink

**Symptom:** adding GPUs doesn't speed up *training* (and can slow it down).

**Cause:** RTX PRO 6000 (Blackwell, sm120) has **no NVLink**. Data-parallel gradient all-reduce goes
over PCIe, and with `local_batch=1` + variable-length samples, every accumulation boundary is a
**barrier that waits on the slowest rank**. Measured on 4×6000: 1 GPU ≈ 2.2, 2 GPU ≈ 2.3, **4 GPU ≈
1.6 samples/s** — the watcher shows a rotating idle GPU (0%/100% flap).

**What to do:**
- For the **training step**, don't expect linear scaling. 2 GPUs ≈ 1 GPU here; 4 was worse. Use a
  modest GPU count for training and shallow accum.
- For **data-gen + capture** (the real scale bottleneck), 16 GPUs is a huge win — run **many target
  replicas in parallel** (each replica captures a shard of prompts). That's where your monster earns
  its keep.

---

## D. "Should I fix torch.compile for speed?" — probably not

**Symptom:** `torch.compile(model, dynamic=True)` throws `InductorError: CantSplit: 64*s20*s66 ...
not divisible`.

**Cause:** GLM's **64-head** flex-attention with **dynamic anchor shapes** + the data-dependent
`dspark_mask_mod` defeats Inductor's codegen.

**Why it's low-value anyway:** we profiled a step (`scripts/profile_step.py`). It's
**matmul/FLOP-bound, not attention-bound**: `aten::mm` = **65%** of GPU time (the 6144→154880 vocab
projection, computed twice — draft logits + the frozen-target distillation logits), flex-attention
softmax only **6.6%**. Those matmuls already call cuBLAS — compile would fuse the ~20% elementwise
ops for **~1.3×**, not 2–3×. **Run eager.** (And eager produces bit-identical weights — zero fidelity
or serving-speed cost.)

**If you truly need compile speed:** the fix is **length-bucketing** (pad each batch to a fixed
bucket so shapes are static), not chasing the CantSplit. But measure first — the bigger lever is below.

---

## E. The real speed lever: `num_anchors` (and its catch)

**Profiled, eager, 1×6000:** `num_anchors` scales the step cost ~linearly:

| num_anchors | ms/step | samples/s |
|---|---|---|
| 512 (recipe) | 246 | 4.1 |
| 128 | 74 | 13.6 |
| 64 | 47 | 21.5 |

**The catch — it's a *coverage* knob, not free.** `num_anchors` = positions trained per sequence per
epoch. Cutting it trains each position fewer times. **On small data this undertrains** (and can make
a smoke look like a method failure when it's just under-coverage). **Safe to cut only at scale**,
where coverage comes from having many sequences. Keep `512` (recipe) unless you have abundant data
and have confirmed accept holds.

---

## F. Capture (extracting hidden states from the vLLM target)

`prepare_target_cache.py` can't load `GlmMoeDsa`; we capture from the **vLLM-served** model. The
sharp edges:

- **CUDA-after-fork in TP workers** → set `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
- **`max_model_len` defaults to 1M** → KV demand explodes → set `max_model_len` (e.g. 16384) +
  `gpu_memory_utilization` for the offline `LLM(...)`.
- **`generate(prompt_token_ids=...)` is wrong** → use `generate(prompts=[{"prompt_token_ids": ids}])`.
- **`apply_model(func)` rejects function serialization** → `VLLM_ALLOW_INSECURE_SERIALIZATION=1`,
  and the hook fns **must be module-level** (a nested function isn't picklable to workers) — that's
  why they live in `dspark_hooks.py` on the `PYTHONPATH`.
- **The hooks** register `forward_pre_hook`s on `model.model.layers[L]` for each aux layer,
  capturing `args[1] + args[2]` (hidden_states + residual = the true residual stream entering the
  layer) + a `forward_hook` on `model.model.norm` for the final hidden. That `args[1]+args[2]` is
  the EAGLE/DSpark aux-hidden value; getting it wrong is silent.
- **Python output buffering** hides progress under docker → `open(..., buffering=1)` /
  `python3 -u`.

---

## G. Finalize + validate (don't skip either)

- **Capture writes local shards** (`shard-local-*.bin` + `samples.local.idx`). The
  `CacheDataset` the trainer reads needs `manifest.json` + renamed shards + a dense `samples.idx`.
  Run `scripts/finalize_cache.py /path/to/cache` (it reuses DeepSpec's own finalize functions; the
  moves are renames on one filesystem, no re-capture).
- **Always `validate_cache.py` before training.** It reads only `seq_len`/`loss_mask`/`input_ids`
  via the 56-byte index records (never the multi-GB hidden states), so it's seconds even on a 38 TB
  cache. `--apply` rewrites `samples.idx` to drop degenerate samples — note it **renumbers
  `sample_id` dense** because `CacheDataset` asserts `sample_id == position`, and updates
  `manifest.num_samples`. It backs up the original idx to `.prefilter`.

---

## H. Training-launch checklist (env + mounts)

- DeepSpec needs `pip install tensorboard PyYAML matplotlib` on top of the vLLM image. **Do not**
  reinstall its pinned torch/transformers/triton — it clobbers vLLM.
- Env: `RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500` (node-level; `init_dist` reads
  `RANK`).
- Mount the checkpoint dir the config writes to (it saves under `~/checkpoints/...` → mount it out
  of the container, or you lose the checkpoint).
- Data is **on-policy with `--disable-thinking`** — that matches DeepSpec's own README example, it's
  on-recipe (not a thinking-vs-no-think compromise).
- The final checkpoint saves at train end regardless of `checkpointing_steps`; lower
  `checkpointing_steps` to probe accept per-epoch instead of waiting for all 10.

---

## J. Checkpoint save crashes on GLM's per-layer config lists (the late-firing one)

**Symptom:** training runs **fine for a whole epoch**, then **crashes at the first checkpoint save**:
```
ValueError: `num_hidden_layers` (5) must be equal to the number of `mlp_layer_types` (78)
```
(or `indexer_types`, or another per-layer list). It only fires inside `save_pretrained`, so you
don't see it until the first `checkpointing_steps` boundary — **after burning an epoch.** The
crash leaves a partial `step_N/` dir with `training_state.rank*.pt` but **no `config.json` /
`model.safetensors`** (and no `step_latest` link, so a relaunch correctly starts fresh — but delete
the partial dir so it doesn't confuse you).

**Cause:** the draft config **deep-copies the GLM target config**, which carries per-layer LIST
fields (`mlp_layer_types`, `indexer_types`, ...) sized to GLM's **78** layers. The draft has
`num_draft_layers` (5), and transformers' config validator requires those lists to equal
`num_hidden_layers`. Qwen3/Gemma targets don't have these fields → upstream DeepSpec never hits it.

**Fix (in `deepspec-glm.patch`, `GlmDSparkTrainer._build_draft_model`):** after building the draft
config, **truncate every per-layer list (length == target layer count) to the draft depth**:
```python
n_draft = int(draft_config.num_hidden_layers); n_target = int(target_config.num_hidden_layers)
for name, val in list(vars(draft_config).items()):
    if isinstance(val, (list, tuple)) and len(val) == n_target:
        setattr(draft_config, name, type(val)(val[:n_draft]))
```
It's general (catches `mlp_layer_types` AND `indexer_types` AND any future per-layer field); the
draft model never reads them anyway. **Verify before a long run** with a quick
`draft_config.save_pretrained(tmpdir)` (see `scripts/diag/`-style check) — that single call exercises
the same validator and takes seconds, vs discovering it an epoch in.

## I. Honest expectations (so a weak smoke doesn't mislead you)

A draft trained **from scratch** on a small set (a few thousand samples) competes against a target
whose stock MTP head was pretrained on the *entire* corpus. Expect a **modest** first number — it may
not beat the stock baseline at small scale, and that's a **data-size artifact, not a DSpark failure**.
A clear win needs scale (**~50–100k samples**). Validate the pipeline small; judge the method at scale.
