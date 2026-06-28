# Hardware & scaling on sm120

The numbers here are measured on **RTX PRO 6000 (Blackwell, sm120, no NVLink)** — the same cards
in a 16-GPU box. Read this before you size a run.

## The step is matmul-bound, not attention-bound

`scripts/profile_step.py` on one card, eager, `num_anchors=512`:

```
aten::mm (matmuls)      472ms   65%   ← the 6144→154880 vocab projection, done twice
MmBackward0             266ms   37%
aten::linear / matmul   205ms         ← q/k/v/o + MLP projections
softmax (flex attn)      48ms   6.6%  ← NOT the bottleneck
```

Consequences:
- **`torch.compile` ≈ 1.3×** here (matmuls already cuBLAS; only the ~20% elementwise fuses). Don't
  chase the CantSplit — run eager.
- **Bigger `local_batch` doesn't help** the matmul efficiency — the lm_head matmul is already
  `M = num_anchors·block = 2560`, large and efficient. It's genuinely FLOP-bound.
- The one big lever is **`num_anchors`** (linear), but it's a coverage knob — see gotchas §E.

## Data-parallel barely scales (no NVLink)

Same draft, eager, by GPU count:

| GPUs | samples/s |
|---|---|
| 1 | ~2.2 |
| 2 | ~2.3 |
| 4 | ~1.6 |

More GPUs ≠ faster *training*. The gradient all-reduce is over PCIe, and with variable-length
samples every accumulation boundary waits on the slowest rank (rotating idle GPU). 4 was *slower*
than 2. Expect the same shape on 16.

**So how do you use 16 GPUs?**
- **Data-gen + capture in parallel** — run many target replicas, each capturing a shard of prompts.
  This is the actual bottleneck at scale and it scales near-linearly. This is where 16 GPUs wins.
- **Training:** use a modest replica count with **shallow grad-accum** (gotchas §A). Or, if you want
  literal `gbs=512` across 16 GPUs, that's accum 32 — likely fine; watch steps 1–20 for NaN.

## RAM

`train.py` spawns one process per GPU; each builds a [154880,6144] embed + lm_head. Our patch builds
on-GPU so host RAM stays low — keep it. Budget host RAM for `num_gpus × (a few GB)` regardless.

## Scale math (so you can plan the spend/time)

At ~13.6 samples/s (eager, `num_anchors=128`, 1 card) — or scale by your effective throughput:

| dataset | × epochs | sample-forwards | 1-card time @13.6/s |
|---|---|---|---|
| 100k | 10 | 1.0M | ~20 h |
| 1.5M | 10 | 15M | ~13 days |

100k samples is plenty for a strong draft. 1.5M is a research-scale run; only worth it if you're
chasing the last bit of accept. Parallelize capture across your 16 GPUs and the wall-clock drops
accordingly.
