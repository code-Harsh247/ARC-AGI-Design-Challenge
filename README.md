# DL Assignment 2: ARC-AGI Open Architecture Design Challenge

**Team: Gradient Ascend** | CS60010 Deep Learning, Spring 2026

| Name | Roll Number |
|---|---|
| Harsh Chattar | 22CS30028 |
| Abhinav Kumar Singh | 22CS30005 |
| Saras Dipak Wagh | 22CS30048 |
| Pallav Agarwal | 22CS30040 |

---

## Architecture Summary

**Iterative Refinement Transformer (IRT)** — a bidirectional encoder–decoder that refines an output grid over T=5 steps using shared decoder weights and 3D Rotary Positional Encoding.

| Component | Detail |
|---|---|
| Architecture | Context Encoder (8 layers) + Iterative Decoder (6 layers, shared weights) |
| Positional Encoding | 3D RoPE — independent axes for (row, col, pair_index) per attention head |
| FFN | SwiGLU (`SiLU` gate × linear), d_ff = 1536 |
| Loss | Deep-supervised cross-entropy, exponentially weighted across all T=5 refinement steps |
| Inference Attempt 1 | Greedy decoding from base model (argmax at final step) |
| Inference Attempt 2 | Test-Time Training (TTT) — leave-one-out fine-tune on demo pairs before predicting |
| Total Parameters | **36,754,176** (≤ 50M budget ✅) |
| d_model | 384 |
| Heads | 8 (d_head = 48) |
| Augmentation | 8 dihedral (D4) × 2 color permutations = **16× per task** |
| Train / Val split | 360 / 40 tasks (random seed 42) |

### Results

| Metric | Value |
|---|---|
| Best validation exact-match (100 epochs) | **0.0750** (3 / 40 tasks) |
| Evaluation tasks solved — Attempt 1 (greedy) | 0 / 400 |
| Evaluation tasks solved — Attempt 2 (TTT) | 5 / 400 |
| **Final evaluation accuracy** | **1.25% (5 / 400)** |

---

## Files in This Submission

```
DL_Assignment_GradientAscend2.ipynb   # Self-contained notebook (run this)
report.tex / report.pdf               # LaTeX report
checkpoints/
    checkpoint_best.pt                # Best validation accuracy checkpoint
    checkpoint_final.pt               # End-of-training checkpoint
data_repo/                            # ARC-AGI dataset (cloned from GitHub)
README.md                             # This file
```

---

## Requirements

- **Python**: 3.10+
- **PyTorch**: 2.x with CUDA support
- **Other packages**: `numpy`, `matplotlib`, `pandas` (standard; all available in Colab/conda)
- **GPU**: CUDA GPU required (T4 15 GB minimum; A100/RTX 4090 recommended for faster training)

No extra `pip install` steps are needed beyond a standard PyTorch environment.

---

## How to Run the Notebook

The notebook `DL_Assignment_GradientAscend2.ipynb` is divided into **5 sections with 30 cells total**.  
Run all cells **top to bottom** in a single session (kernel must not be restarted between sections).

### ⚠️ Before You Start — Edit Two Paths

Open **Cell 1** and **Cell 3** and update the following variables to match your machine:

```python
# Cell 1
CHECKPOINT_DIR = '/your/path/to/checkpoints'

# Cell 3
LOCAL_ARC = '/your/path/to/ARC-AGI'
```

Everything else runs without modification.

---

### Section 1 — Setup & Data Loading (Cells 1–6)

| Cell # | What it does |
|---|---|
| **Cell 1** | Sets `CHECKPOINT_DIR`. **Edit this path** before running. Creates the directory if it doesn't exist. |
| **Cell 2** | Empty cell — skip. |
| **Cell 3** | Clones the ARC-AGI dataset from GitHub into `LOCAL_ARC`. **Edit this path**. Sets `TRAIN_DIR` and `EVAL_DIR`. Prints task counts (400 training, 400 evaluation). |
| **Cell 4** | Defines all data-loading code: `ARCTask`, `load_tasks`, `predict_output_shape`, `grid_to_tokens`, `build_task_sequence`. Prints `"Data loading code defined."` |
| **Cell 5** | Defines augmentation and dataset classes: `ARCDataset`, `augment_task`, `collate_arc_batch`. Prints `"Augmentation & dataset code defined."` |
| **Cell 6** | Loads 400 training tasks, performs 360 / 40 train/val split (seed 42), prints sequence length statistics. |

---

### Section 2 — Model Architecture (Cells 7–11)

> **Markdown header cell** (Cell 7 in notebook) — not executable, just a section title.

| Cell # | What it does |
|---|---|
| **Cell 8** | Defines `RoPE3D` and `apply_rope_to_qk` — 3D rotary positional encoding split across row, col, pair-index axes. Prints `"RoPE3D defined."` |
| **Cell 9** | Defines `MultiHeadAttention` (Flash Attention via `F.scaled_dot_product_attention`), `FeedForward` (SwiGLU), `EncoderBlock`, `DecoderBlock`. Prints `"Transformer blocks defined."` |
| **Cell 10** | Defines the full model: `ContextEncoder`, `IterativeDecoder`, `PerTaskEmbedding`, `ARCModel`. Prints `"Model architecture defined."` |
| **Cell 11** | **Instantiates the model** with `d_model=384, enc_layers=8, dec_layers=6, n_heads=8, d_ff=1536, refine_steps=5`. Prints the parameter breakdown table and confirms **36,754,176 params ≤ 50M**. |

Expected output of Cell 11:
```
Component                           Parameters
-----------------------------------------------
encoder                          18,892,800 ( 51.4%)
decoder                          17,723,136 ( 48.2%)
task_embed                          138,240 (  0.4%)
-----------------------------------------------
TOTAL                            36,754,176
Budget (50M)                     50,000,000
✓ Within 50M parameter budget
```

---

### Section 3 — Training (Cells 12–19)

> **Markdown header cell** (not executable).

| Cell # | What it does |
|---|---|
| **Cell 12** | Defines `deep_supervised_loss` (exponentially weighted CE across all T steps) and `WSDScheduler` (Warmup-Stable-Decay). Prints `"Loss and scheduler defined."` |
| **Cell 13** | Defines `save_checkpoint` and `load_checkpoint` utilities. Prints `"Checkpoint utils defined."` |
| **Cell 14** | Defines `validate` — computes exact-match and cell accuracy on the val loader. Prints `"Validation defined."` |
| **Cell 15** | **Main training loop** — trains for `EPOCHS=100`, effective batch size 32 (batch=2, grad_accum=16), bfloat16 AMP. Validates every 5 epochs. Saves `checkpoint_best.pt` on improvement. Prints loss and val metrics each epoch. |
| **Cell 16** | Plots training loss curve and validation exact-match / cell accuracy curves side-by-side. |

**Training time**: ~8–12 hours for 100 epochs on T4 (Colab). To resume a stopped run, just re-run Cell 15 — it will start fresh (no auto-resume; load `checkpoint_best.pt` manually via `load_checkpoint` if needed before re-running).

**To skip training** and use the provided checkpoints: load `checkpoint_best.pt` directly in Cell 18 (evaluation) without running Cell 15.

---

### Section 4 — Evaluation on 400 Held-Out Tasks (Cells 17–18)

> **Markdown header cell** (not executable).

| Cell # | What it does |
|---|---|
| **Cell 17** | Defines `test_time_train` (TTT: leave-one-out fine-tune for 100 steps on demo pairs) and `predict_task` (single-task inference). Prints `"TTT and prediction functions defined."` |
| **Cell 18** | **Final evaluation** — loads `checkpoint_best.pt`, runs greedy decoding (Attempt 1) then TTT (Attempt 2) on all 400 evaluation tasks. Prints running progress and a final accuracy table. |

Expected output of Cell 18:
```
==================================================
EVALUATION RESULTS
==================================================
Total tasks:     400
Solved:          5
Accuracy:        0.0125 (1.25%)
  Attempt 1:     0
  Attempt 2:     5 (additional via TTT)
==================================================
```

> TTT runs 100 gradient steps per task — allow ~2–4 hours for all 400 evaluation tasks on a T4.

---

### Section 5 — Ablation Experiments (Cells 19–30)

> **Markdown header cell** describing the 3 ablations (not executable).

Each ablation trains a model variant for **50 epochs** with one design choice removed.  
**Run Cell 19 (the big setup cell) first** — it re-defines all classes and loads data so ablations can run independently of the main training session.

| Cell # | Ablation | What changes | Result |
|---|---|---|---|
| **Cell 20** | **A — 2D RoPE** | `context_pairs` zeroed → pair-index axis carries no info | Val exact-match: 0.0750 (Δ = 0.0000) |
| **Cell 21** | **B — Final-step loss** | `final_step_loss` replaces `deep_supervised_loss` | Val exact-match: 0.0250 (Δ = **+0.0500**) |
| **Cell 22** | **C — No per-task embeddings** | `task_idx=None` during training | Val exact-match: 0.0500 (Δ = +0.0250) |
| **Cell 23** | Ablation summary table | Loads results from saved checkpoints and prints comparison | — |

#### Ablation summary

| # | Design choice | Full model | Ablation | Δ |
|---|---|---|---|---|
| A | 3D RoPE (row, col, pair_index) | 0.0750 | 2D RoPE: 0.0750 | +0.0000 |
| B | Deep supervision (all T steps) | 0.0750 | Final step only: 0.0250 | **+0.0500** |
| C | Per-task embeddings | 0.0750 | No per-task embed: 0.0500 | +0.0250 |

---

### Visualization (Cell 30)

Cell 30 plots sample predictions vs. ground truth for the first 3 validation tasks using the trained model. Requires the model to be loaded (run Cells 1–11 and load a checkpoint first).

---

## Checkpoints

Two checkpoints are included in the `checkpoints/` directory:

| File | Description |
|---|---|
| `checkpoint_best.pt` | Best validation exact-match model (epoch with highest val acc = 0.0750) |
| `checkpoint_final.pt` | End-of-training model (epoch 100) |

To load a checkpoint manually:
```python
load_checkpoint(model, '/path/to/checkpoints/checkpoint_best.pt', device=device)
```

---

## Hard Constraints Met

| Constraint | Status |
|---|---|
| ≤ 50M parameters | ✅ 36,754,176 params |
| Trained from scratch | ✅ Random init (Xavier uniform / normal), no pretrained weights |
| ARC-1 training data only | ✅ Only `data/training/` (400 tasks) used for training |
| Evaluation on `data/evaluation/` | ✅ Test outputs never inspected during development |
| 2 attempts per task | ✅ Attempt 1 = greedy decoding, Attempt 2 = TTT |
| Single notebook | ✅ All code in `DL_Assignment_GradientAscend2.ipynb` |
