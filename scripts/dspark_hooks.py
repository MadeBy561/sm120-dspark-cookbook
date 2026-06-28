"""Module-level hook functions for the DSpark hidden-state capture.

WHY A SEPARATE MODULE: vLLM's `apply_model`/collective_rpc serializes the function with
pickle (insecure path) to send it to the TP workers. pickle can serialize a module-level
function *by reference* (module name + qualname) but NOT a nested/local function or lambda.
The workers are spawned subprocesses that inherit PYTHONPATH but do NOT import the __main__
capture script — so the functions must live in an importable module (this file, with /work on
PYTHONPATH) for the worker to resolve them.

Captures, per the deepseek_v2.py forward: the residual stream ENTERING each AUX layer
(hidden_states + residual) + the final post-norm hidden, via per-layer hooks (no
aux_hidden_state_layers, which would change the model's return type and break the runner).
"""
import torch

AUX_LAYERS = (75, 76, 77)


def setup_hooks(model):
    """Run on each TP worker (via apply_model): register the per-layer capture hooks."""
    inner = model.model  # DeepseekV2Model
    model._dspark_cap = {"aux": {}, "final": None}

    def _make_pre(idx):
        def _pre(_mod, args):
            # layer.forward(positions, hidden_states, residual, llama_4_scaling)
            hs, res = args[1], args[2]
            rs = hs if res is None else (hs + res)
            model._dspark_cap["aux"][idx] = rs.detach().to(torch.float16).cpu()
        return _pre

    for idx in AUX_LAYERS:
        inner.layers[idx].register_forward_pre_hook(_make_pre(idx))

    def _norm_hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out  # RMSNorm-w-residual returns (normed, res)
        model._dspark_cap["final"] = h.detach().to(torch.float16).cpu()

    inner.norm.register_forward_hook(_norm_hook)
    return True


def get_capture(model):
    """Run on each TP worker (via apply_model): return the stashed (final, aux) capture."""
    return getattr(model, "_dspark_cap", None)
