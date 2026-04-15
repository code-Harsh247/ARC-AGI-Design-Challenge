# ARC-AGI Design Challenge — Team Gradient Ascend

## Overview
Iterative Refinement Transformer for ARC-AGI-1 visual reasoning puzzles.
- **Architecture**: Bidirectional Context Encoder + Iterative Decoder (~28M params)
- **Positional Encoding**: 3D RoPE (row, col, pair_index)
- **Loss**: Deep-supervised cross-entropy at every refinement step
- **Inference**: Greedy (attempt 1) + Test-Time Training (attempt 2)

## Project Structure
```
src/
  data.py          # Dataset loading, tokenization, augmentation
  model.py         # Full model architecture
  train.py         # Training loop, loss, optimizer setup
  evaluate.py      # Evaluation + TTT inference
  utils.py         # Param counting, checkpointing, visualization
DL_Assignment_GradientAscend.ipynb   # Final Colab notebook (all-in-one)
```

## How to Run (Colab)
1. Open `DL_Assignment_GradientAscend.ipynb` in Google Colab.
2. Set runtime to **GPU (T4)**.
3. Run cells **in order** from top to bottom.
4. **Section 1**: Clones ARC-AGI data, installs deps.
5. **Section 2**: Defines model architecture and prints parameter count.
6. **Section 3**: Trains the model (≈2–3 hours on T4). Checkpoints saved to Google Drive.
7. **Section 4**: Loads best checkpoint and runs evaluation on 400 eval tasks.
8. **Section 5**: Ablation experiments (run individually).

## Checkpoints
- `checkpoint_best.pt` — best validation accuracy during training
- `checkpoint_final.pt` — end-of-training weights

## Constraints
- ≤ 50M parameters (we use ~28.5M)
- Trained from scratch — no pretrained weights
- Only ARC-1 training data (400 tasks) used for training
- Evaluation on 400 held-out tasks (outputs never inspected during development)
