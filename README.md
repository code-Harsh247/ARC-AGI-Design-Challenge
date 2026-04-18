# DL Assignment 2: ARC-AGI Open Architecture Design Challenge

**Team: Gradient Ascend** | CS60010 Deep Learning, Spring 2026

---

## Architecture Summary

**Iterative Refinement Transformer** — a bidirectional encoder–decoder that refines an output grid over T=5 steps using shared decoder weights.

| Component | Detail |
|---|---|
| Architecture | Context Encoder (8 layers) + Iterative Decoder (6 layers, shared) |
| Positional Encoding | 3D RoPE — separate axes for (row, col, pair_index) per attention head |
| Loss | Deep-supervised cross-entropy across all T refinement steps |
| Inference Attempt 1 | Greedy decoding from base model |
| Inference Attempt 2 | Test-Time Training (TTT) — fine-tune on demo pairs at inference time |
| Total Parameters | **36,754,176** (≤ 50M budget) |
| d_model | 384 |
| Augmentation | 8 dihedral × 2 color permutations = 16 augments per task |

---

## Files in This Submission

```
DL_Assignment_GradientAscend.zip
├── DL_Assignment_GradientAscend.ipynb   # Self-contained notebook (run this)
├── DL_Assignment_GradientAscend_report.pdf
└── README.md                            # This file
```

---

## How to Run the Notebook

### Environment
- **Platform**: Google Colab or any Linux machine with a CUDA GPU
- **GPU**: T4 (15 GB) minimum; RTX 4090 (24 GB) recommended for full run
- **Python**: 3.10+ with PyTorch 2.x

### Step-by-Step Cell Execution

The notebook is divided into 5 sections. Run all cells **top to bottom** in a fresh runtime.

#### Section 1 — Setup & Data Loading (Cells 1–7)
| Cell | Purpose |
|---|---|
| Cell 1 | Set `CHECKPOINT_DIR` — **edit this path** to your local drive or Google Drive mount |
| Cell 2 | (Empty — skip) |
| Cell 3 | Clones ARC-AGI dataset from GitHub into `LOCAL_ARC` — **edit this path** too |
| Cell 4 | Defines all data classes: `ARCTask`, `load_tasks`, `build_task_sequence`, tokenization |
| Cell 5 | Defines augmentation: dihedral transforms, color permutation, `ARCDataset`, `collate_arc_batch` |
| Cell 6 | Loads 400 training tasks, performs 360/40 train/val split, prints sequence length stats |

> **Before running**: update `CHECKPOINT_DIR` in Cell 1 and `LOCAL_ARC` in Cell 3 to valid paths on your machine.

#### Section 2 — Model Architecture (Cells 8–11)
| Cell | Purpose |
|---|---|
| Cell 8 | Defines `RoPE3D` and `apply_rope_to_qk` — 3D rotary positional encoding |
| Cell 9 | Defines `MultiHeadAttention`, `FeedForward`, `EncoderBlock`, `DecoderBlock` |
| Cell 10 | Defines full `ARCModel`: `ContextEncoder` + `IterativeDecoder` + `PerTaskEmbedding` |
| Cell 11 | **Instantiates model and prints parameter count** — verifies ≤ 50M budget |

#### Section 3 — Training (Cells 12–16)
| Cell | Purpose |
|---|---|
| Cell 12 | Defines `deep_supervised_loss` and `WSDScheduler` (Warmup-Stable-Decay LR) |
| Cell 13 | Defines `save_checkpoint` / `load_checkpoint` utilities |
| Cell 14 | Defines `validate` — exact-match accuracy on val set |
| Cell 15 | **Main training loop** — runs for `EPOCHS=200`, resumes from checkpoint if present. Saves `checkpoint_best.pt` when val accuracy improves |
| Cell 16 | Plots training loss and validation accuracy curves |

> **Training time**: ~3–5 hours for 200 epochs on T4 (spread across Colab sessions). Training **auto-resumes** from `checkpoint_best.pt` if it exists — just re-run Cell 15.

> **To retrain from scratch**: delete `checkpoint_best.pt` before running Cell 15.

#### Section 4 — Evaluation (Cells 17–18)
| Cell | Purpose |
|---|---|
| Cell 17 | Defines `test_time_train` (TTT) and `predict_task` inference functions |
| Cell 18 | **Final evaluation on 400 held-out tasks** — loads best checkpoint, runs greedy (attempt 1) then TTT (attempt 2), prints accuracy table |

> Only run Cell 18 once, on the final trained model. TTT takes ~100 gradient steps per task (~2–3 hours for all 400).

#### Section 5 — Ablation Experiments (Cells 19–22)
| Cell | Purpose |
|---|---|
| Cell 19 | Ablation A description: 2D RoPE vs 3D RoPE (zero out pair_index) |
| Cell 20 | **Ablation B**: defines `final_step_loss` (no deep supervision baseline) |
| Cell 21 | Ablation C description: no per-task embeddings baseline |
| Cell 22 | (Visualization) Plots sample predictions vs ground truth for val tasks |

> To run an ablation: follow the comments in each cell, instantiate a fresh model variant, retrain for ~50 epochs, and compare val accuracy to the full model.

---

## Checkpoints

Checkpoints are **not included in this ZIP** (each is ~420 MB). To reproduce results:
1. Run training (Section 3) to regenerate `checkpoint_best.pt`
2. Then run evaluation (Section 4)

Checkpoint files saved during training:
- `checkpoint_best.pt` — highest validation accuracy model
- `checkpoint_final.pt` — end-of-training model

---

## Hard Constraints Met

| Constraint | Status |
|---|---|
| ≤ 50M parameters | ✅ 36,754,176 params |
| Trained from scratch | ✅ Random init, no pretrained weights |
| ARC-1 training data only | ✅ Only `data/training/` (400 tasks) used |
| Evaluate on `data/evaluation/` | ✅ Outputs never inspected during development |
| 2 attempts per task | ✅ Attempt 1 = greedy, Attempt 2 = TTT |
