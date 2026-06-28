#!/usr/bin/env python3
"""
Build the DSpark target-hidden-state cache for GLM-5.2-REAP-594B (NVFP4, GlmMoeDsa)
via vLLM — the replacement for DeepSpec's scripts/data/prepare_target_cache.py.

WHY THIS FILE EXISTS
--------------------
DeepSpec's prepare_target_cache.py loads the target with transformers `AutoModel` +
forward hooks. Our target is an NVFP4 `glm_moe_dsa` model that ONLY loads in the b12x
vLLM fork (transformers has no such arch). So we capture the aux hidden states
[75,76,77] + the final hidden state through vLLM, then write the SAME cache format
DeepSpec's trainer (CacheCollator / target_cache_dataset) reads — fully lossless w.r.t.
the recipe.

MECHANISM (verified in the live image, deepseek_v2.py:1462-1516)
----------------------------------------------------------------
GlmMoeDsaForCausalLM.model is a DeepseekV2Model. Its forward, with
    self.aux_hidden_state_layers = (75, 76, 77)
set, returns:
    (final_hidden_post_norm,  [aux_75, aux_76, aux_77])
  - each aux_i = the residual stream ENTERING layer i  -> [num_tokens, hidden]   (= DeepSpec target_hidden_states, pre-concat)
  - final_hidden = post-final-norm hidden               -> [num_tokens, hidden]   (= DeepSpec target_last_hidden_states)
The residual stream is TP-replicated (full hidden_size on every rank), so rank 0's
capture is complete.

APPROACH
--------
Run the target in vLLM (TP4, same quant/backend as serving), install a forward hook on
the inner DeepseekV2Model via collective_rpc, then for each (prompt+response) sequence
run a 1-token prefill (generate max_tokens=1) so the hook sees ALL positions; pull
rank-0's capture back and write a DeepSpec cache record.

⚠️ GPU-GATED — loads a full 594B (4×~84 GiB). STOP the serving container first
(`docker stop glm52-reap`); they cannot share the GPUs. The [VERIFY-n] markers are the
first-run checkpoints (vLLM internals shift between versions).

RUN (when the GPUs are free)
----------------------------
  docker stop glm52-reap                       # free the 4 GPUs
  docker run --rm --gpus all --network host --ipc host --shm-size 32g \
    -v /mnt/models:/mnt/models:ro -v /mnt/18tb_r1:/mnt/18tb_r1 -v /home/reaper/dspark:/work \
    -e VLLM_USE_B12X_MOE=1 -e VLLM_USE_B12X_SPARSE_INDEXER=1 -e VLLM_USE_V2_MODEL_RUNNER=1 \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3 -w /work/DeepSpec -e PYTHONPATH=/work/DeepSpec \
    --entrypoint python3 dspark-gen:latest /work/capture_hidden_states.py \
      --regen /mnt/18tb_r1/dspark-data/regen/perfectblend_regen.jsonl \
      --out-dir /mnt/18tb_r1/dspark-data/cache --max-samples 5000
  docker start glm52-reap                       # resume serving/gen afterwards
"""
import argparse
import json
import os

import torch
from transformers import AutoTokenizer

MODEL = "/mnt/models/GLM-5.2-Int8Mix-NVFP4-REAP-594B"
AUX_LAYERS = (75, 76, 77)              # V4 last-3 recipe (GLM has 78 layers, 0-77)
INDEX_TOPK_PATTERN = "FFFSSS" + "FSSS" * 18 + "FSSS"  # [VERIFY-0] copy exact from serving compose


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--regen", required=True, help="generate_train_data output jsonl (conversations w/ role/content)")
    p.add_argument("--out-dir", required=True, help="cache output dir on the NAS")
    p.add_argument("--max-samples", type=int, default=0, help="0 = all")
    p.add_argument("--min-loss-tokens", type=int, default=14)
    return p.parse_args()


def build_llm():
    """Load the NVFP4 GlmMoeDsa target. [VERIFY-1] keep these args == the serving compose
    (compressed-tensors, fp8 kv, B12X_MLA_SPARSE, DCP4, index pattern) so captured hidden
    states match what we SERVE. Speculative/MTP is irrelevant here (we only prefill)."""
    from vllm import LLM
    return LLM(
        model=MODEL,
        quantization="compressed-tensors",
        tensor_parallel_size=4,
        kv_cache_dtype="fp8",
        trust_remote_code=True,
        enforce_eager=True,               # avoid cudagraph for the one-off prefills
        max_model_len=16384,              # capture seqs are <=8k; model default is 1M → 52GiB KV blowup
        gpu_memory_utilization=0.90,
        hf_overrides={"use_index_cache": True, "index_topk_pattern": INDEX_TOPK_PATTERN},
        # attention/dcp: vLLM picks B12X_MLA_SPARSE for glm_moe_dsa by default on this image.
    )


def install_capture_hook(llm):
    """Register the per-layer capture hooks on every TP worker. The hook fns live in the
    `dspark_hooks` module (on PYTHONPATH=/work) so vLLM's pickle-based rpc can reference them
    by name — the workers are spawned subprocesses that can import dspark_hooks but NOT the
    __main__ capture script, so nested/local fns (the previous approach) fail to pickle."""
    import dspark_hooks
    llm.apply_model(dspark_hooks.setup_hooks)


def fetch_capture(llm):
    """Return (final[L,hidden], aux=[a75,a76,a77]) from rank 0 (residual stream is TP-replicated,
    so rank 0 is complete). If aux dict is empty, the pre-hooks didn't fire on the prefill."""
    import dspark_hooks
    cap = llm.apply_model(dspark_hooks.get_capture)[0]
    aux = [cap["aux"][i] for i in AUX_LAYERS]
    return cap["final"], aux


_PARSER = None


def tokenize_with_loss_mask(tok, conversations, max_length=8192):
    """✅ VALIDATED (test_parser.py, 3/3): render with the GLM chat template + loss-mask the
    assistant body via DeepSpec's GeneralParser + the registered 'glm' template — the SAME path
    prepare_target_cache uses, so input_ids/loss_mask are recipe-faithful (lossless)."""
    global _PARSER
    if _PARSER is None:
        from deepspec.data.parser import GeneralParser, TEMPLATE_REGISTRY
        _PARSER = GeneralParser(tokenizer=tok, chat_template=TEMPLATE_REGISTRY.get("glm"))
    out = _PARSER.parse(conversations, max_length=max_length)
    return out["input_ids"].tolist(), out["loss_mask"].tolist()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    llm = build_llm()
    install_capture_hook(llm)

    from vllm import SamplingParams
    from deepspec.data.target_cache_dataset import AsyncTargetCacheWriter   # [VERIFY-5] reuse their writer
    writer = AsyncTargetCacheWriter(rank_dir=args.out_dir, max_shard_bytes=64 * 1024**3, max_queue_size=64)

    n = 0
    with open(args.regen) as f:
        for line in f:
            d = json.loads(line)
            if d.get("status") != "success":
                continue
            conv = d["conversations"]
            input_ids, loss_mask = tokenize_with_loss_mask(tok, conv)
            if sum(loss_mask) < args.min_loss_tokens:
                continue

            # 1-token prefill forces a full forward over input_ids -> hook captures all positions.
            # generate() takes `prompts` (PromptType); a TokensPrompt dict feeds raw ids (no re-templating).
            llm.generate(prompts=[{"prompt_token_ids": input_ids}],
                         sampling_params=SamplingParams(max_tokens=1, temperature=0.0))
            final, aux = fetch_capture(llm)
            L = len(input_ids)
            final = final[:L]                               # [L, hidden]
            target_hidden = torch.cat([a[:L] for a in aux], dim=-1)  # [L, 3*hidden]

            writer.write_sample(
                input_ids=torch.tensor(input_ids, dtype=torch.long),
                attention_mask=torch.ones(L, dtype=torch.long),
                loss_mask=torch.tensor(loss_mask, dtype=torch.long),
                target_hidden_states=target_hidden,
                target_last_hidden_states=final,
            )
            n += 1
            if n % 100 == 0:
                print(f"[capture] {n} samples", flush=True)
            if args.max_samples and n >= args.max_samples:
                break
    writer.close()
    print(f"[capture] DONE: {n} samples -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
