# DeepSpec patch — what it changes and why

Apply against a fresh `git clone https://github.com/deepseek-ai/DeepSpec`:
```bash
git clone https://github.com/deepseek-ai/DeepSpec && cd DeepSpec
# the patch was cut against DeepSpec @ 0a03e19 ("first init"). If HEAD has moved and the patch
# doesn't apply cleanly, pin it:  git checkout 0a03e19   (or 3-way merge:  git apply --3way)
git apply /path/to/cookbook/deepspec-patches/deepspec-glm.patch
cp /path/to/cookbook/deepspec-patches/config_dspark_glm52_full.py config/dspark/
```

The patch is **small** (~60 lines across 4 files) and **adds GLM support without altering the
DSpark method**. Each change maps to a "Fidelity §3" item in the top-level README.

### `deepspec/trainer/dspark_trainer.py` — `GlmDSparkTrainer`
`GlmDSparkTrainer(Qwen3DSparkTrainer)` overrides `_build_draft_model` for one GLM-only reason: after
building the draft config (via the **unchanged** Qwen3 builder), it **truncates GLM's per-layer
config lists** (`mlp_layer_types`, `indexer_types`) from the target's 78 entries to the draft's 5, so
the first checkpoint `save_pretrained` doesn't crash on transformers' config validator (gotchas §J).
The DSpark *method* is untouched — the draft is still the standard `Qwen3DSparkModel` consuming the
target's hidden states; this is pure GLM-config compatibility, not a recipe change.

### `deepspec/trainer/__init__.py` — export it
One line so the config can `from deepspec.trainer import GlmDSparkTrainer`.

### `deepspec/data/parser.py` — GLM chat template
Registers a `"glm"` `ChatTemplate` (assistant/user headers, end-of-turn, the empty
`<think></think>` for thinking-off). DeepSpec ships Qwen/Gemma templates only. This makes the
tokenizer render GLM conversations correctly and puts the **loss mask on the right tokens**. Using
the *correct* template is required for faithful training — the wrong one would be the deviation.
Validate with `scripts` / `test_parser.py` style checks before a big run.

### `deepspec/trainer/base_trainer.py` — NVFP4-safe, low-RAM model build
DeepSpec's `build_models` loads the target via transformers `AutoModelForCausalLM` purely to copy
its frozen `embed_tokens` + `lm_head` into the draft. A `GlmMoeDsa` (custom arch + quant) **can't**
load that way. The patch:
1. Builds the draft **inside `with torch.device(self.device):`** so its [154880,6144] embed/lm_head
   allocate on the **GPU**, not in host RAM (fixes the multi-process SIGKILL — gotchas §B).
2. Reads the frozen `model.embed_tokens.weight` + `lm_head.weight` **straight from the target
   safetensors** with `safe_open(..., device=cuda)` and hands them in via a `SimpleNamespace`
   (skips the fp32 `nn.Embedding`/`nn.Linear` intermediates).

These are the **same frozen target weights** the recipe uses (bf16, unquantized in the checkpoint) —
just a loader that works for `GlmMoeDsa` and doesn't OOM. The draft, objective, and data are
unchanged. (On a **full bf16 GLM-5.2**, transformers might load the target directly — but this patch
works either way and is strictly safer on RAM, so keep it.)

It also adds a **non-finite-gradient guard** in the training loop: skip `optimizer.step()` when
`grad_norm` is inf/nan (drop the bad batch, preserve last-good weights) instead of poisoning every
weight. This is standard spec-decode training hygiene that DeepSpec's loop lacks — without it,
training dies on a single bad batch hundreds of steps in (gotchas §A, Mode 2). Lossless: one batch
skipped, the objective is untouched.

---

**Nothing in this patch touches** `block_size`, `num_anchors`, the markov head, the loss, the
optimizer, or the DSpark model. It is pure GLM/quant **enablement**.
