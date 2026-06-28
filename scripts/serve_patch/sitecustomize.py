# sitecustomize.py — auto-imported by CPython at interpreter startup (site init), so it runs in
# every vLLM process INCLUDING the spawned TP workers. No image rebuild. Patches two vLLM internals
# so a standard-MHA DSpark/DFlash draft can serve on the MLA GLM-5.2 target:
#
#  (1) dflash loader (vllm/v1/worker/gpu/spec_decode/dflash/utils.py:load_dflash_model)
#      - honor speculative_config.draft_kv_cache_dtype (stock dflash ignores it) so the MHA draft
#        gets its own (bf16) KV instead of inheriting the MLA target's fp8_ds_mla (no non-MLA backend
#        supports fp8_ds_mla).
#      - load the small draft via the standard CPU-staged loader, not fastsafetensors (whose sharded
#        NCCL shuffle materializes full tensors on rank 0 and OOMs next to the ~92GB/GPU target).
#
#  (2) KV-cache grouping (vllm/v1/core/kv_cache_utils.py:_get_kv_cache_groups_uniform_groups)
#      - the DeepseekV4 path pads every non-MLA group UP into an MLA page bucket, which is impossible
#        when the MHA draft's page (2048 KV elems/token) exceeds the MLA latent page (576 B/token).
#        EXPERIMENT: on that AssertionError, emit each group as its OWN KVCacheGroupSpec (its native
#        page) and let the general hybrid KV path handle heterogeneous pages. May break downstream
#        (the block pool may assume uniform pages) — this is the actual "fix vLLM" attempt.
import sys
import importlib.abc
import importlib.util

# We shadow the image's stock /usr/lib/python3.12/sitecustomize.py (apport hook only). Replicate it.
try:
    import apport_python_hook
except ImportError:
    pass
else:
    apport_python_hook.install()


def _patch_dflash_utils(M):
    from vllm.config import replace
    from vllm.distributed.parallel_state import get_pp_group
    from vllm.model_executor.model_loader import get_model
    from vllm.v1.worker.gpu.spec_decode.eagle.utils import _should_share

    def load_dflash_model(target_model, vllm_config):
        from vllm.compilation.backends import set_model_tag

        spec = vllm_config.speculative_config
        assert spec is not None
        dmc = spec.draft_model_config
        causal = M.get_dflash_causal(dmc)
        draft_backend = getattr(spec, "draft_attention_backend", None) or spec.attention_backend
        draft_vllm_config = replace(
            vllm_config,
            attention_config=replace(
                vllm_config.attention_config,
                use_non_causal=not causal,
                backend=draft_backend,
            ),
            load_config=replace(vllm_config.load_config, load_format="safetensors"),
        )
        dkv = getattr(spec, "draft_kv_cache_dtype", None)
        if dkv is not None:
            draft_vllm_config = replace(
                draft_vllm_config,
                cache_config=replace(draft_vllm_config.cache_config, cache_dtype=dkv),
            )
            print(
                f"[dspark-patch] draft KV dtype -> {dkv} "
                f"(target stays {vllm_config.cache_config.cache_dtype}); draft attn -> {draft_backend}",
                flush=True,
            )

        with set_model_tag("dflash_head"):
            dflash_model = get_model(vllm_config=draft_vllm_config, model_config=dmc)

        target_language_model = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        target_inner = target_language_model.model
        draft_inner = dflash_model.model
        if get_pp_group().world_size == 1:
            target_embed = getattr(target_inner, "embed_tokens", None) or getattr(
                target_inner, "embedding", None
            )
            draft_embed = getattr(draft_inner, "embed_tokens", None)
            if target_embed is not None and _should_share(
                dflash_model, "has_own_embed_tokens", draft_embed, target_embed
            ):
                if draft_embed is not None:
                    del draft_inner.embed_tokens
                draft_inner.embed_tokens = target_embed
        target_lm_head = getattr(target_language_model, "lm_head", None)
        draft_lm_head = getattr(dflash_model, "lm_head", None)
        if target_lm_head is not None and _should_share(
            dflash_model, "has_own_lm_head", draft_lm_head, target_lm_head
        ):
            if draft_lm_head is not None:
                del dflash_model.lm_head
            dflash_model.lm_head = target_lm_head
        return dflash_model

    M.load_dflash_model = load_dflash_model
    print("[dspark-patch] patched load_dflash_model (draft_kv_cache_dtype + standard loader)", flush=True)


def _patch_kv_cache_utils(M):
    _orig = M._get_kv_cache_groups_uniform_groups

    def _own_groups(grouped_specs):
        try:
            return _orig(grouped_specs)
        except AssertionError:
            from vllm.v1.kv_cache_interface import KVCacheGroupSpec

            print(
                "[dspark-patch] draft KV page exceeds MLA buckets -> emitting each spec as its OWN "
                "KV group (heterogeneous-page experiment)",
                flush=True,
            )
            return [
                KVCacheGroupSpec(
                    layer_names=list(gs.kv_cache_specs.keys()), kv_cache_spec=gs
                )
                for gs in grouped_specs
            ]

    M._get_kv_cache_groups_uniform_groups = _own_groups
    print("[dspark-patch] patched _get_kv_cache_groups_uniform_groups (own-group fallback)", flush=True)


_PATCHERS = {
    "vllm.v1.worker.gpu.spec_decode.dflash.utils": _patch_dflash_utils,
    "vllm.v1.core.kv_cache_utils": _patch_kv_cache_utils,
}


def _safe_patch(name, module):
    try:
        _PATCHERS[name](module)
    except Exception as e:  # pragma: no cover
        print(f"[dspark-patch] FAILED to patch {name}: {e!r}", flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name not in _PATCHERS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        _orig_exec = spec.loader.exec_module

        def exec_module(mod):
            _orig_exec(mod)
            _safe_patch(name, mod)

        spec.loader.exec_module = exec_module
        return spec


for _name in _PATCHERS:
    _existing = sys.modules.get(_name)
    if _existing is not None:
        _safe_patch(_name, _existing)

if not any(isinstance(f, _Finder) for f in sys.meta_path):
    sys.meta_path.insert(0, _Finder())
