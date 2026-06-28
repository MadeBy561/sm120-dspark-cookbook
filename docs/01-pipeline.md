# Pipeline — step by step

Concrete commands for the full run. Paths/model names are placeholders — edit for your box.
Read [`02-gotchas.md`](02-gotchas.md) alongside this; the fixes there are baked into these commands.

Assumed layout:
```
DeepSpec/                     # cloned + patched (see deepspec-patches/README.md)
cookbook/scripts/             # this repo's scripts, on PYTHONPATH at /work
/cache/glm52/                 # the hidden-state cache (big; put on fast storage)
GLM-5.2/                      # the full target checkpoint
```

---

## 0. Image + deps

Use a vLLM **b12x** image with `GlmMoeDsaForCausalLM` (e.g. `voipmonitor/vllm:ds4dspark-v7-...`).
Inside it:
```bash
pip install tensorboard PyYAML matplotlib     # DeepSpec deps; do NOT touch torch/transformers/triton
```

## 1. Patch DeepSpec
```bash
git clone https://github.com/deepseek-ai/DeepSpec && cd DeepSpec
git apply /path/to/cookbook/deepspec-patches/deepspec-glm.patch
cp /path/to/cookbook/deepspec-patches/config_dspark_glm52_full.py config/dspark/
```

## 2. Generate on-policy answers
Serve the **full GLM-5.2** target (vLLM, OpenAI-compatible endpoint), then:
```bash
python scripts/data/download_and_split.py --dataset-name mlabonne/open-perfectblend \
  --test-size 0.05 --train-output-path train_datasets/perfectblend_train.jsonl \
  --test-output-dir eval_datasets --skip-existing

python scripts/data/generate_train_data.py --model GLM-5.2 \
  --server-address 127.0.0.1:8000 [more replicas...] \
  --concurrency 32 --temperature 0.7 --top-p 0.8 --top-k 20 --min-p 0 \
  --max-tokens 4096 --disable-thinking --resume \
  --input-file-path train_datasets/perfectblend_train.jsonl \
  --output-file-path train_datasets/glm52/regen.jsonl
```
> **Swap the sampling values for GLM's.** `--temperature/--top-p/--top-k/--min-p` above are
> **Qwen3-4B's** recommended settings (DeepSpec's example). Per their README, adjust these to the
> **target's** recommended sampling → use **GLM-5.2's** values. The flags (`--disable-thinking`,
> `--max-tokens 4096`, `--resume`, `--concurrency`) stay as-is. Match how you'll serve GLM-5.2.

On your 16-GPU box: run **many target replicas** and pass all their `--server-address`es — this
stage is the real bottleneck and it parallelizes cleanly. Start with **50–100k** samples.

## 3. Capture hidden states (from the vLLM target)
`scripts/capture_hidden_states.py` builds an offline `LLM(...)`, prefills each conversation, and
`dspark_hooks.setup_hooks` captures the residual stream at the aux layers + final norm, writing the
DeepSpec cache format. Edit the top of the script (`MODEL`, `OUT`, `AUX_LAYERS=[N-3,N-2,N-1]`, the
prompt source). Run it inside the image with:
```bash
docker run --rm --gpus all --network host --ipc host --init \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  -e PYTHONPATH=/work/DeepSpec:/work \
  -v /models:/models:ro -v /cache:/cache -v /path/to/cookbook:/work \
  -w /work <image> python3 -u scripts/capture_hidden_states.py
```
Key settings inside the script (see gotchas §F): `tensor_parallel_size`, `kv_cache_dtype`,
`enforce_eager=True`, **`max_model_len=16384`**, `gpu_memory_utilization=0.90`, and
`generate(prompts=[{"prompt_token_ids": ids}])`.

> **The shipped `capture_hidden_states.py` was tuned for a quantized (NVFP4) REAP target.** For a
> **full bf16 GLM-5.2**, remove/adjust the quant-specific `LLM(...)` args: drop `quantization=...`,
> and the `hf_overrides` `use_index_cache` / `index_topk_pattern` (those are REAP/quant-specific).
> The hook mechanism (`dspark_hooks.py`) is unchanged — it captures the same residual streams
> regardless of target precision. Just verify `AUX_LAYERS = [N-3, N-2, N-1]` for your layer count.

## 4. Finalize + validate
```bash
python3 scripts/finalize_cache.py /cache/glm52            # manifest + dense samples.idx + renamed shards
python3 scripts/validate_cache.py /cache/glm52            # report (degenerate samples?)
python3 scripts/validate_cache.py /cache/glm52 --apply    # drop degenerates (renumbers idx, backs up)
```

## 5. Profile (optional but recommended once)
```bash
python3 scripts/profile_step.py     # confirms matmul-bound + shows num_anchors timing on YOUR card
```

## 6. Train
```bash
docker run -d --name dspark-train --gpus all --network host --ipc host --init \
  --shm-size 32g --ulimit memlock=-1 \
  -e PYTHONPATH=/work/DeepSpec:/work \
  -e RANK=0 -e WORLD_SIZE=1 -e MASTER_ADDR=127.0.0.1 -e MASTER_PORT=29500 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -v /models:/models:ro -v /cache:/cache -v /work/DeepSpec:/work/DeepSpec \
  -v $HOME/checkpoints:/root/checkpoints \
  -w /work/DeepSpec <image> python3 train.py \
  --config config/dspark/dspark_glm52_full.py \
  --opts data.target_cache_path=/cache/glm52 \
  --opts model.mask_token_id=154821 \
  --opts train.global_batch_size=512 \
  --opts train.torch_compile=false \
  --opts logging.checkpointing_steps=2000
```
Watch `docker logs -f dspark-train` for the first 20 steps. Loss should descend (we saw ~3.5→2.7);
**if you see `loss=nan`, drop `global_batch_size`** (gotchas §A).

## 7. Serve
See [`04-serving.md`](04-serving.md): load the checkpoint as the speculator on the `ds4dspark`
image (DFlash first, then full DSpark with the markov head), and read the per-position accept-length
from vLLM's spec-decode metrics vs the stock MTP baseline.
