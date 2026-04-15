"""
Training loop, loss functions, and optimizer setup for ARC-AGI.

Key features:
  - Deep-supervised loss: CE at every refinement step with exponential upweighting
  - WSD learning rate schedule (warmup → linear decay)
  - BF16 mixed precision via torch.autocast
  - Gradient accumulation for large effective batch sizes
  - Checkpoint saving/loading to Google Drive
"""

import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from data import ARCDataset, ARCTask, collate_arc_batch, load_tasks
from model import ARCModel, make_debug_model, make_full_model


# ---------------------------------------------------------------------------
# Loss Function
# ---------------------------------------------------------------------------
def deep_supervised_loss(
    all_logits: list,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute deep-supervised cross-entropy loss across all refinement steps.

    The loss at later steps is weighted more heavily (exponentially):
        L = sum_t  2^(t - T) * CE_t

    This ensures the final step contributes most but early steps still
    receive training signal, which is the primary performance driver for
    iterative refinement models (per TRM / HRM literature).

    Args:
        all_logits: list of T tensors, each (B, L, C) where C = num_colors
        target: (B, L) ground-truth color indices
        mask: (B, L) True = valid cell, False = padding

    Returns:
        Scalar loss
    """
    T = len(all_logits)
    total_loss = torch.tensor(0.0, device=target.device)

    for t, logits in enumerate(all_logits):
        # Weight: 2^(t - T + 1) so last step gets weight 1.0, first gets 2^(1-T)
        weight = 2.0 ** (t - T + 1)
        # Flatten for cross-entropy: (B*L, C) vs (B*L,)
        B, L, C = logits.shape
        flat_logits = logits.reshape(-1, C)
        flat_target = target.reshape(-1)
        flat_mask = mask.reshape(-1).float()

        # Per-token CE, then mask out padding
        ce = F.cross_entropy(flat_logits, flat_target, reduction="none")
        masked_ce = (ce * flat_mask).sum() / flat_mask.sum().clamp(min=1)
        total_loss = total_loss + weight * masked_ce

    return total_loss


# ---------------------------------------------------------------------------
# Learning Rate Schedule: Warmup + Linear Decay (WSD)
# ---------------------------------------------------------------------------
class WSDScheduler:
    """Warmup-Stable-Decay scheduler."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            scale = self.step_count / max(1, self.warmup_steps)
        else:
            # Linear decay
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = max(self.min_lr_ratio, 1.0 - progress * (1.0 - self.min_lr_ratio))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * scale

    def get_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ---------------------------------------------------------------------------
# Checkpoint Management
# ---------------------------------------------------------------------------
def save_checkpoint(
    model: ARCModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    val_acc: float,
    path: str,
):
    """Save model + optimizer state to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "val_acc": val_acc,
    }, path)
    print(f"  Checkpoint saved to {path} (epoch={epoch}, val_acc={val_acc:.4f})")


def load_checkpoint(model: ARCModel, path: str, optimizer=None, device="cuda"):
    """Load model (+ optionally optimizer) state from disk."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"  Loaded checkpoint from {path} (epoch={ckpt.get('epoch', '?')}, "
          f"val_acc={ckpt.get('val_acc', '?')})")
    return ckpt


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(
    model: ARCModel,
    val_loader: DataLoader,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Compute exact-match accuracy on a validation set.

    A task is "solved" only if every cell in the predicted grid matches the target.
    """
    model.eval()
    total_tasks = 0
    solved_tasks = 0
    total_cell_correct = 0
    total_cells = 0

    for batch in val_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        all_logits = model(
            context_tokens=batch["context_tokens"],
            context_rows=batch["context_rows"],
            context_cols=batch["context_cols"],
            context_pairs=batch["context_pairs"],
            context_mask=batch["context_mask"],
            target_rows=batch["target_rows"],
            target_cols=batch["target_cols"],
            target_mask=batch["target_mask"],
            test_input_tokens=batch["test_input_tokens"],
            test_input_rows=batch["test_input_rows"],
            test_input_cols=batch["test_input_cols"],
            test_input_mask=batch["test_input_mask"],
        )

        # Use final step predictions
        final_logits = all_logits[-1]  # (B, L, C)
        preds = final_logits.argmax(dim=-1)  # (B, L)
        target = batch["target_tokens"]
        mask = batch["target_mask"]

        # Per-task exact match
        B = preds.shape[0]
        for i in range(B):
            valid = mask[i]
            pred_cells = preds[i][valid]
            true_cells = target[i][valid]
            correct = (pred_cells == true_cells).all().item()
            solved_tasks += correct
            total_tasks += 1
            total_cell_correct += (pred_cells == true_cells).sum().item()
            total_cells += valid.sum().item()

    model.train()
    return {
        "exact_match": solved_tasks / max(1, total_tasks),
        "cell_accuracy": total_cell_correct / max(1, total_cells),
        "solved": solved_tasks,
        "total": total_tasks,
    }


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train(
    model: ARCModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cuda",
    epochs: int = 50,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    grad_accum_steps: int = 8,
    checkpoint_dir: str = "./checkpoints",
    validate_every_n_epochs: int = 5,
    use_amp: bool = True,
):
    """
    Main training loop with deep supervision, AMP, gradient accumulation,
    and periodic validation + checkpointing.
    """
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
    )

    total_steps = epochs * len(train_loader) // grad_accum_steps
    warmup_steps = int(0.05 * total_steps)
    scheduler = WSDScheduler(optimizer, warmup_steps, total_steps)
    scaler = GradScaler(enabled=use_amp)

    best_val_acc = 0.0
    global_step = 0

    print(f"Training config: epochs={epochs}, lr={lr}, grad_accum={grad_accum_steps}")
    print(f"Total steps: {total_steps}, warmup: {warmup_steps}")
    print(f"Model params: {model.count_parameters():,}")
    print(f"Train batches/epoch: {len(train_loader)}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                all_logits = model(
                    context_tokens=batch["context_tokens"],
                    context_rows=batch["context_rows"],
                    context_cols=batch["context_cols"],
                    context_pairs=batch["context_pairs"],
                    context_mask=batch["context_mask"],
                    target_rows=batch["target_rows"],
                    target_cols=batch["target_cols"],
                    target_mask=batch["target_mask"],
                    test_input_tokens=batch["test_input_tokens"],
                    test_input_rows=batch["test_input_rows"],
                    test_input_cols=batch["test_input_cols"],
                    test_input_mask=batch["test_input_mask"],
                    task_idx=batch["task_idx"],
                )

                loss = deep_supervised_loss(
                    all_logits, batch["target_tokens"], batch["target_mask"]
                )
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            epoch_loss += loss.item() * grad_accum_steps
            epoch_tokens += batch["target_mask"].sum().item()

        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(1, len(train_loader))
        lr_now = scheduler.get_lr()[0]
        print(f"Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f} | "
              f"lr={lr_now:.2e} | time={elapsed:.1f}s")

        # Periodic validation
        if epoch % validate_every_n_epochs == 0 or epoch == epochs:
            val_metrics = validate(model, val_loader, device)
            print(f"  Val: exact_match={val_metrics['exact_match']:.4f} "
                  f"({val_metrics['solved']}/{val_metrics['total']}) "
                  f"cell_acc={val_metrics['cell_accuracy']:.4f}")

            if val_metrics["exact_match"] >= best_val_acc:
                best_val_acc = val_metrics["exact_match"]
                save_checkpoint(
                    model, optimizer, epoch, global_step, best_val_acc,
                    os.path.join(checkpoint_dir, "checkpoint_best.pt"),
                )

    # Save final checkpoint
    save_checkpoint(
        model, optimizer, epochs, global_step, best_val_acc,
        os.path.join(checkpoint_dir, "checkpoint_final.pt"),
    )
    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    return model
