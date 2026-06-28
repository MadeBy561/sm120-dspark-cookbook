"""DSpark draft config for GLM-5.2-Int8Mix-NVFP4-REAP-594B (GlmMoeDsa = MLA+DSA, V4-lineage).

Faithful to DeepSeek's V4-production DSpark recipe (read from DeepSeek-V4-Flash/Pro-DSpark
config.json) + DeepSpec's training hyperparameters. Differences from dspark_qwen3_4b.py are
DICTATED BY THE TARGET ARCH, not by us cutting corners:
  - block_size 5  (V4-Flash/Pro use 5, not the Qwen ref's 7)
  - target_layer_ids = LAST 3 layers  (V4 uses last-3: Flash [40,41,42], Pro [58,59,60];
    GLM has 78 layers → [75,76,77]), NOT the Qwen ref's spread [1,9,17,25,33]
  - markov_rank 512 (V4-Pro=512, hidden 7168 ≈ GLM 6144; V4-Flash=256 at hidden 4096)
  - mask_token_id MUST be an unused id in GLM's 154880 vocab (V4 uses 128799 in its vocab)

NB: GlmMoeDsa is NOT a vanilla-transformers model — `prepare_target_cache.py`'s
AutoModel.from_pretrained path will NOT load our NVFP4 vLLM checkpoint. The target
hidden states are captured from the **vLLM-served NVFP4 594B** via its eagle3 aux-hidden
mechanism (GlmMoeDsaForCausalLM already supports set_aux_hidden_state_layers) — this is
also what makes the draft NVFP4-AWARE (it learns the quantized target's real distribution).
See docs/dspark-build-checklist.md step 3.
"""
import os
from deepspec.trainer import GlmDSparkTrainer  # NEW: thin subclass of Qwen3DSparkTrainer (step 2)

BASE_TB_DIR = os.path.expanduser("~/tensorboard")
BASE_CKPT_DIR = os.path.expanduser("~/checkpoints")
project_name = "deepspec"
exp_name = "dspark_block5_glm52_594b"
seed = 42

model = dict(
    target_model_name_or_path="/mnt/models/GLM-5.2-Int8Mix-NVFP4-REAP-594B",
    block_size=5,                       # V4 production (accept caps at block+1 = 6)
    num_draft_layers=5,                 # DeepSpec default; tune 3–5
    target_layer_ids=[75, 76, 77],      # GLM 78L last-3 (V4 last-3 pattern)
    mask_token_id=None,                 # TODO: pick an unused/reserved id in GLM vocab 154880
    num_anchors=512,

    ## markov head (the DSpark serial corrector — reused verbatim from markov_head.py)
    markov_rank=512,                    # V4-Pro=512 (GLM hidden 6144 ≈ Pro 7168)
    markov_head_type='vanilla',         # V4 production

    ## confidence head (trained even though we skip the scheduler for single-stream)
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,

    ## loss (DeepSpec exact — L1 distillation dominates; L1 REQUIRES caching target_last_hidden)
    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,
)

train = dict(
    trainer_cls=GlmDSparkTrainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=512,
    num_train_epochs=10,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    torch_compile=True,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=3000,
)

data = dict(
    target_cache_path=None,
    chat_template="glm",                # NEW: add GLM chat template to ConversationCollator (step 2)
    max_length=4096,                    # draft training length (can raise; cache scales linearly)
    num_workers=4,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name = str(cfg['project_name'])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(BASE_CKPT_DIR, project_name, exp_name)
    logging_cfg["tensorboard_dir"] = os.path.join(BASE_TB_DIR, project_name, exp_name)
    cfg["logging"] = logging_cfg
    return cfg
