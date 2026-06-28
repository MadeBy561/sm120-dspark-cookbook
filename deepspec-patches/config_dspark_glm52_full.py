"""DSpark draft config for the FULL GLM-5.2 (GlmMoeDsa = MLA + DSA + MoE, DeepSeek-V4 lineage).

Drop this in DeepSpec at:  config/dspark/dspark_glm52_full.py

It is the DeepSeek V4-production DSpark recipe (block 5, last-3 aux layers, markov 512),
applied to the full GLM-5.2 instead of a REAP. The ONLY things that differ from
DeepSeek's own V4-Flash/Pro DSpark configs are dictated by GLM's architecture
(78 layers, hidden 6144, vocab 154880), not by cutting corners.

WHAT TO EDIT FOR YOUR BOX (see comments tagged >>EDIT):
  - target_model_name_or_path : path to your full GLM-5.2 checkpoint
  - target_layer_ids          : last-3 layers of YOUR model (verify the layer count)
  - global_batch_size         : size for your GPU count so grad-accum stays shallow (see GOTCHAS)
  - torch_compile             : leave False unless you implement length-bucketing (see GOTCHAS)
"""
import os
from deepspec.trainer import GlmDSparkTrainer  # added by deepspec-glm.patch

BASE_TB_DIR = os.path.expanduser("~/tensorboard")
BASE_CKPT_DIR = os.path.expanduser("~/checkpoints")
project_name = "deepspec"
exp_name = "dspark_block5_glm52_full"
seed = 42

model = dict(
    # >>EDIT: full GLM-5.2 (bf16 or fp8). NOT a REAP — a draft is bonded to the exact
    # target it is captured from (REAP and full have different hidden states). See GOTCHAS.
    target_model_name_or_path="/path/to/GLM-5.2",

    block_size=5,                  # V4 production. Accept length caps at block+1 = 6.
    num_draft_layers=5,            # DeepSpec default (tune 3-5).
    # >>EDIT: last-3 transformer layers. GLM-5.2 = 78 layers -> [75,76,77].
    # Verify your model's num_hidden_layers; use [N-3, N-2, N-1].
    target_layer_ids=[75, 76, 77],
    mask_token_id=154821,          # GLM tokenizer reserved [MASK]. Verify in YOUR tokenizer.
    num_anchors=512,               # positions trained per sequence per epoch. See GOTCHAS (coverage knob).

    # --- Markov serial head (the DSpark corrector; reused verbatim from DeepSpec) ---
    markov_rank=512,               # V4-Pro = 512 (hidden 7168 ~ GLM 6144). V4-Flash = 256 @ hidden 4096.
    markov_head_type="vanilla",    # V4 production.

    # --- Confidence head ---
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,

    # --- Loss (DeepSpec exact: L1 distillation dominates; L1 REQUIRES target_last_hidden in cache) ---
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
    # >>EDIT: global_batch_size / (world_size * local_batch_size) = grad-accumulation depth.
    # We measured: accum 8 trains clean, accum 256 -> bf16-accumulation NaN. Keep accum shallow.
    #   16 GPUs:  gbs=512 -> accum 32  (recipe-faithful; watch step 1-20 for loss=nan)
    #             gbs=128 -> accum 8   (proven-clean depth; safest)
    # warmup_ratio*total_steps must be >= ~30 steps or the lr ramp is too steep (also -> NaN on
    # tiny datasets). With lots of data this is automatic. See GOTCHAS.
    global_batch_size=512,
    num_train_epochs=10,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    # >>EDIT: torch.compile is only ~1.3x here (the step is matmul/vocab-projection-bound, not
    # attention-bound) AND it CantSplits on GLM's 64-head flex_attention with the dynamic anchor
    # shapes. Leave False unless you implement length-bucketing. See GOTCHAS + docs/03.
    torch_compile=False,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=2000,      # lower this to checkpoint per-epoch and probe accept early
)

data = dict(
    target_cache_path=None,        # set on the CLI: --opts data.target_cache_path=/path/to/cache
    chat_template="glm",           # added by deepspec-glm.patch
    max_length=4096,               # draft training length (cache size scales ~linearly with this)
    num_workers=8,                 # bump on a fat box
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name = str(cfg["project_name"])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(BASE_CKPT_DIR, project_name, exp_name)
    logging_cfg["tensorboard_dir"] = os.path.join(BASE_TB_DIR, project_name, exp_name)
    cfg["logging"] = logging_cfg
    return cfg
