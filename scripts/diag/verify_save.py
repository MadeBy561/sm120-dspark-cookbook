"""Verify the GlmDSparkTrainer per-layer-list truncation lets draft_config.save_pretrained pass
(the step-288 crash was: mlp_layer_types(78) != num_hidden_layers(5)). CPU-only, ~seconds.
"""
import sys
import tempfile

sys.path.insert(0, "/work/DeepSpec")
from transformers import AutoConfig
from deepspec.modeling.dspark.qwen3.config import build_draft_config

MODEL = "/mnt/models/GLM-5.2-Int8Mix-NVFP4-REAP-594B"


class A(dict):
    def __getattr__(self, k):
        return self[k]


MA = A(num_draft_layers=5, target_layer_ids=[75, 76, 77], block_size=5, num_anchors=512,
       markov_rank=512, markov_head_type="vanilla", mask_token_id=154821,
       confidence_head_alpha=1.0, confidence_head_with_markov=True)

tc = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
dc = build_draft_config(tc, MA)
n_draft = int(dc.num_hidden_layers)
n_target = int(tc.num_hidden_layers)

before = [k for k, v in vars(dc).items() if isinstance(v, (list, tuple)) and len(v) == n_target]
print(f"per-layer lists (len=={n_target}) BEFORE fix: {before}")

# the GlmDSparkTrainer fix:
for name, val in list(vars(dc).items()):
    if isinstance(val, (list, tuple)) and len(val) == n_target:
        setattr(dc, name, type(val)(val[:n_draft]))

after = [k for k, v in vars(dc).items() if isinstance(v, (list, tuple)) and len(v) == n_target]
print(f"per-layer lists (len=={n_target}) AFTER fix:  {after}")

with tempfile.TemporaryDirectory() as d:
    dc.save_pretrained(d)
    print("✅ save_pretrained PASSED — checkpoint save will work")
