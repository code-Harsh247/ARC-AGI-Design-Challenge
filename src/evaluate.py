"""
Evaluation and Test-Time Training (TTT) for ARC-AGI.

- TTT: fine-tune model on demo pairs of a single task at inference time
- Two-attempt strategy: attempt 1 = base model, attempt 2 = TTT-adapted model
- Final evaluation on 400 held-out tasks
"""

import copy
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import (
    ARCTask, ARCDataset, build_task_sequence, collate_arc_batch,
    load_tasks, predict_output_shape, NUM_COLORS, PAD_TOKEN
)
from model import ARCModel


# ---------------------------------------------------------------------------
# Test-Time Training (TTT)
# ---------------------------------------------------------------------------
def test_time_train(
    model: ARCModel,
    task: ARCTask,
    n_steps: int = 100,
    lr: float = 1e-4,
    device: str = "cuda",
) -> ARCModel:
    """
    Fine-tune model on demo pairs of a single task at inference time.

    For each demo pair, treats the input as the test input and the output as
    the target. This teaches the model the task-specific transformation rule
    without seeing the actual test output.

    Args:
        model: trained model to adapt
        task: the task being evaluated
        n_steps: number of gradient steps
        lr: learning rate for TTT
        device: cuda or cpu

    Returns:
        TTT-adapted model (a deep copy, original unchanged)
    """
    # Deep copy to avoid modifying the base model
    ttt_model = copy.deepcopy(model)
    ttt_model.to(device)
    ttt_model.train()

    optimizer = torch.optim.AdamW(ttt_model.parameters(), lr=lr, weight_decay=0.0)

    # Build training data from demo pairs:
    # For each demo pair, treat it as a "task" — use the other pairs as context
    # and the current pair's output as the target
    for step in range(n_steps):
        total_loss = torch.tensor(0.0, device=device)
        n_demos = len(task.train_pairs)

        for hold_out_idx in range(n_demos):
            # Create a sub-task: hold out one pair as "test", rest as context
            held_pair = task.train_pairs[hold_out_idx]
            context_pairs = (
                task.train_pairs[:hold_out_idx] + task.train_pairs[hold_out_idx + 1:]
            )
            if len(context_pairs) == 0:
                context_pairs = task.train_pairs  # fall back to using all

            sub_task = ARCTask(
                task_id=task.task_id,
                train_pairs=context_pairs,
                test_pairs=[held_pair],
            )

            seq = build_task_sequence(sub_task, test_idx=0)
            # Move to device and add batch dimension
            batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor)
                     else v for k, v in seq.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                all_logits = ttt_model(
                    context_tokens=batch["context_tokens"],
                    context_rows=batch["context_rows"],
                    context_cols=batch["context_cols"],
                    context_pairs=batch["context_pairs"],
                    context_mask=torch.ones_like(batch["context_tokens"], dtype=torch.bool),
                    target_rows=batch["target_rows"],
                    target_cols=batch["target_cols"],
                    target_mask=torch.ones(1, batch["target_tokens"].shape[1],
                                           dtype=torch.bool, device=device),
                    test_input_tokens=batch["test_input_tokens"],
                    test_input_rows=batch["test_input_rows"],
                    test_input_cols=batch["test_input_cols"],
                    test_input_mask=torch.ones_like(batch["test_input_tokens"], dtype=torch.bool),
                )

                # Loss on final step only for TTT (speed)
                logits = all_logits[-1]
                target = batch["target_tokens"]
                ce = F.cross_entropy(
                    logits.reshape(-1, NUM_COLORS), target.reshape(-1)
                )
                total_loss = total_loss + ce / n_demos

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(ttt_model.parameters(), max_norm=1.0)
        optimizer.step()

    ttt_model.eval()
    return ttt_model


# ---------------------------------------------------------------------------
# Single-Task Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_task(
    model: ARCModel,
    task: ARCTask,
    test_idx: int = 0,
    device: str = "cuda",
    T: Optional[int] = None,
) -> List[List[int]]:
    """
    Generate prediction for a single task's test input.

    Returns:
        2D grid (list of lists) — the predicted output
    """
    model.eval()
    seq = build_task_sequence(task, test_idx=test_idx)
    batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor)
             else v for k, v in seq.items()}

    all_logits = model(
        context_tokens=batch["context_tokens"],
        context_rows=batch["context_rows"],
        context_cols=batch["context_cols"],
        context_pairs=batch["context_pairs"],
        context_mask=torch.ones_like(batch["context_tokens"], dtype=torch.bool),
        target_rows=batch["target_rows"],
        target_cols=batch["target_cols"],
        target_mask=torch.ones(1, batch["target_tokens"].shape[1],
                               dtype=torch.bool, device=device),
        test_input_tokens=batch["test_input_tokens"],
        test_input_rows=batch["test_input_rows"],
        test_input_cols=batch["test_input_cols"],
        test_input_mask=torch.ones_like(batch["test_input_tokens"], dtype=torch.bool),
        T=T,
    )

    final_logits = all_logits[-1]  # (1, H*W, C)
    preds = final_logits.argmax(dim=-1).squeeze(0).cpu().numpy()  # (H*W,)

    out_h = seq["output_h"]
    out_w = seq["output_w"]
    grid = preds[:out_h * out_w].reshape(out_h, out_w).tolist()
    return grid


# ---------------------------------------------------------------------------
# Full Evaluation Pipeline
# ---------------------------------------------------------------------------
def evaluate_on_tasks(
    model: ARCModel,
    tasks: List[ARCTask],
    device: str = "cuda",
    use_ttt: bool = True,
    ttt_steps: int = 100,
    ttt_lr: float = 1e-4,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate model on a list of tasks with 2 attempts per task.

    Attempt 1: base model (greedy)
    Attempt 2: TTT-adapted model (greedy)

    Args:
        model: trained ARCModel
        tasks: list of ARCTask objects
        use_ttt: whether to use TTT for attempt 2
        ttt_steps: gradient steps for TTT
        ttt_lr: learning rate for TTT

    Returns:
        dict with accuracy metrics and per-task results
    """
    model.eval()
    model.to(device)

    results = []
    solved_count = 0

    for i, task in enumerate(tasks):
        task_result = {
            "task_id": task.task_id,
            "solved": False,
            "attempt1_correct": False,
            "attempt2_correct": False,
        }

        for test_idx in range(len(task.test_pairs)):
            true_output = task.test_pairs[test_idx]["output"]

            # Attempt 1: base model
            pred1 = predict_task(model, task, test_idx=test_idx, device=device)
            if pred1 == true_output:
                task_result["attempt1_correct"] = True
                task_result["solved"] = True

            # Attempt 2: TTT-adapted model
            if use_ttt and not task_result["solved"]:
                ttt_model = test_time_train(
                    model, task, n_steps=ttt_steps, lr=ttt_lr, device=device
                )
                pred2 = predict_task(ttt_model, task, test_idx=test_idx, device=device)
                del ttt_model
                torch.cuda.empty_cache()

                if pred2 == true_output:
                    task_result["attempt2_correct"] = True
                    task_result["solved"] = True
            elif not use_ttt:
                # Attempt 2 without TTT: use more refinement steps
                pred2 = predict_task(model, task, test_idx=test_idx, device=device,
                                     T=model.refine_steps * 2)
                if pred2 == true_output:
                    task_result["attempt2_correct"] = True
                    task_result["solved"] = True

        if task_result["solved"]:
            solved_count += 1
        results.append(task_result)

        if verbose and (i + 1) % 20 == 0:
            print(f"  Evaluated {i+1}/{len(tasks)} tasks, "
                  f"solved so far: {solved_count}/{i+1} "
                  f"({solved_count/(i+1)*100:.1f}%)")

    accuracy = solved_count / max(1, len(tasks))
    attempt1_solved = sum(1 for r in results if r["attempt1_correct"])
    attempt2_solved = sum(1 for r in results if r["attempt2_correct"] and not r["attempt1_correct"])

    summary = {
        "total_tasks": len(tasks),
        "solved": solved_count,
        "accuracy": accuracy,
        "attempt1_solved": attempt1_solved,
        "attempt2_solved": attempt2_solved,
        "per_task": results,
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"Total tasks:     {summary['total_tasks']}")
        print(f"Solved:          {summary['solved']}")
        print(f"Accuracy:        {summary['accuracy']:.4f} ({summary['accuracy']*100:.2f}%)")
        print(f"  Attempt 1:     {summary['attempt1_solved']}")
        print(f"  Attempt 2:     {summary['attempt2_solved']} (additional)")
        print(f"{'='*50}")

    return summary


# ---------------------------------------------------------------------------
# Evaluation-only runner (loads eval tasks, never peeks at outputs)
# ---------------------------------------------------------------------------
def run_evaluation(
    model: ARCModel,
    eval_dir: str,
    device: str = "cuda",
    use_ttt: bool = True,
    ttt_steps: int = 100,
):
    """
    Run final evaluation on the 400 held-out evaluation tasks.

    IMPORTANT: This loads eval task inputs only. The outputs are present in
    the JSON files but are used only for scoring — never inspected during
    model development or training.
    """
    print(f"Loading evaluation tasks from {eval_dir}...")
    tasks = load_tasks(eval_dir)
    print(f"Loaded {len(tasks)} evaluation tasks.")

    summary = evaluate_on_tasks(
        model, tasks, device=device,
        use_ttt=use_ttt, ttt_steps=ttt_steps,
    )
    return summary


# ---------------------------------------------------------------------------
# Grid Visualization (for notebook display)
# ---------------------------------------------------------------------------
ARC_COLORS = {
    0: "#000000",  # black
    1: "#0074D9",  # blue
    2: "#FF4136",  # red
    3: "#2ECC40",  # green
    4: "#FFDC00",  # yellow
    5: "#AAAAAA",  # gray
    6: "#F012BE",  # magenta
    7: "#FF851B",  # orange
    8: "#7FDBFF",  # cyan
    9: "#870C25",  # maroon
}


def grid_to_ascii(grid: List[List[int]]) -> str:
    """Convert a grid to a colored ASCII representation."""
    lines = []
    for row in grid:
        lines.append(" ".join(str(c) for c in row))
    return "\n".join(lines)


def print_task(task: ARCTask, predictions: Optional[List[List[int]]] = None):
    """Print a task's demo pairs and optionally a prediction."""
    print(f"Task: {task.task_id}")
    print(f"Demo pairs: {len(task.train_pairs)}")

    for i, pair in enumerate(task.train_pairs):
        print(f"\n--- Demo {i+1} ---")
        print("Input:")
        print(grid_to_ascii(pair["input"]))
        print("Output:")
        print(grid_to_ascii(pair["output"]))

    print(f"\n--- Test ---")
    print("Input:")
    print(grid_to_ascii(task.test_pairs[0]["input"]))
    print("Expected:")
    print(grid_to_ascii(task.test_pairs[0]["output"]))

    if predictions is not None:
        print("Predicted:")
        print(grid_to_ascii(predictions))
        match = predictions == task.test_pairs[0]["output"]
        print(f"Exact match: {match}")
