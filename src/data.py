"""
ARC-AGI Data Loading, Tokenization, and Augmentation.

Handles:
  - Loading ARC-AGI JSON tasks
  - Converting grids to token sequences with positional metadata
  - Geometric + color augmentation (dihedral group + color permutation)
  - Collating variable-length tasks into padded batches
"""

import json
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_GRID_DIM = 30          # ARC grids are at most 30×30
NUM_COLORS = 10            # Colors 0–9
MAX_SEQ_LEN = 512          # Token budget per training example

# Special token IDs (colors occupy 0–9)
PAD_TOKEN = 10
ROW_SEP = 11              # Separates rows within a grid
GRID_SEP = 12             # Separates input grid from output grid within a pair
PAIR_SEP = 13             # Separates demo pairs / marks start of test
VOCAB_SIZE = 14            # 0–9 colors + 4 special tokens


# ---------------------------------------------------------------------------
# Task Loading
# ---------------------------------------------------------------------------
class ARCTask:
    """Represents a single ARC-AGI task with demo pairs and test pair(s)."""

    def __init__(self, task_id: str, train_pairs: List[Dict], test_pairs: List[Dict]):
        self.task_id = task_id
        self.train_pairs = train_pairs   # list of {"input": grid, "output": grid}
        self.test_pairs = test_pairs     # list of {"input": grid, "output": grid}

    @classmethod
    def from_json(cls, filepath: str) -> "ARCTask":
        with open(filepath, "r") as f:
            data = json.load(f)
        task_id = Path(filepath).stem
        return cls(
            task_id=task_id,
            train_pairs=data["train"],
            test_pairs=data["test"],
        )

    def __repr__(self):
        return (
            f"ARCTask(id={self.task_id}, "
            f"demos={len(self.train_pairs)}, tests={len(self.test_pairs)})"
        )


def load_tasks(directory: str) -> List[ARCTask]:
    """Load all ARC tasks from a directory of JSON files."""
    tasks = []
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".json"):
            tasks.append(ARCTask.from_json(os.path.join(directory, fname)))
    return tasks


# ---------------------------------------------------------------------------
# Output Shape Prediction
# ---------------------------------------------------------------------------
def predict_output_shape(task: ARCTask, test_idx: int = 0) -> Tuple[int, int]:
    """
    Heuristically predict the test output grid shape.

    Rules (in priority order):
      1. If all outputs have the same shape as their corresponding inputs,
         the test output has the same shape as the test input.
      2. If all outputs share the same shape, use that shape.
      3. Fall back to the test input shape.
    """
    demos = task.train_pairs
    test_input = task.test_pairs[test_idx]["input"]

    # Rule 1: output shape == input shape for all demos?
    same_as_input = all(
        len(d["output"]) == len(d["input"]) and len(d["output"][0]) == len(d["input"][0])
        for d in demos
    )
    if same_as_input:
        return len(test_input), len(test_input[0])

    # Rule 2: all outputs share the same shape?
    out_shapes = [(len(d["output"]), len(d["output"][0])) for d in demos]
    if len(set(out_shapes)) == 1:
        return out_shapes[0]

    # Rule 3: fall back to test input shape
    return len(test_input), len(test_input[0])


# ---------------------------------------------------------------------------
# Grid Tokenization
# ---------------------------------------------------------------------------
def grid_to_tokens(grid: List[List[int]]) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Flatten a 2D grid into a 1D token list with (row, col) positions.

    Returns:
        tokens: list of integer token IDs (color values 0–9, + ROW_SEP between rows)
        positions: list of (row, col) tuples; ROW_SEP gets (row, -1)
    """
    tokens = []
    positions = []
    for r, row in enumerate(grid):
        if r > 0:
            tokens.append(ROW_SEP)
            positions.append((r, -1))
        for c, val in enumerate(row):
            tokens.append(val)
            positions.append((r, c))
    return tokens, positions


def build_task_sequence(
    task: ARCTask,
    test_idx: int = 0,
    max_len: int = MAX_SEQ_LEN,
) -> Dict[str, torch.Tensor]:
    """
    Build the full input sequence for a task.

    Layout:
      [PAIR_SEP] [in1 tokens] [GRID_SEP] [out1 tokens]
      [PAIR_SEP] [in2 tokens] [GRID_SEP] [out2 tokens]
      ...
      [PAIR_SEP] [test_input tokens] [GRID_SEP]

    Returns dict with:
      - context_tokens: (L_ctx,) int tensor — demo pair sequence
      - context_rows:   (L_ctx,) int tensor — row index per token
      - context_cols:   (L_ctx,) int tensor — col index per token
      - context_pairs:  (L_ctx,) int tensor — pair index per token
      - target_tokens:  (H*W,) int tensor — flattened target output cells
      - target_rows:    (H*W,) int tensor
      - target_cols:    (H*W,) int tensor
      - test_input_tokens: (H_in*W_in,) — flattened test input cells (no separators)
      - test_input_rows:   (H_in*W_in,)
      - test_input_cols:   (H_in*W_in,)
      - output_h: int — predicted output height
      - output_w: int — predicted output width
    """
    ctx_tokens = []
    ctx_rows = []
    ctx_cols = []
    ctx_pairs = []

    # Encode demo pairs
    for pair_idx, pair in enumerate(task.train_pairs):
        # Pair separator
        ctx_tokens.append(PAIR_SEP)
        ctx_rows.append(0)
        ctx_cols.append(0)
        ctx_pairs.append(pair_idx)

        # Input grid
        itok, ipos = grid_to_tokens(pair["input"])
        ctx_tokens.extend(itok)
        ctx_rows.extend([p[0] for p in ipos])
        ctx_cols.extend([p[1] for p in ipos])
        ctx_pairs.extend([pair_idx] * len(itok))

        # Grid separator
        ctx_tokens.append(GRID_SEP)
        ctx_rows.append(0)
        ctx_cols.append(0)
        ctx_pairs.append(pair_idx)

        # Output grid
        otok, opos = grid_to_tokens(pair["output"])
        ctx_tokens.extend(otok)
        ctx_rows.extend([p[0] for p in opos])
        ctx_cols.extend([p[1] for p in opos])
        ctx_pairs.extend([pair_idx] * len(otok))

    # Test input (as final "pair")
    test_pair_idx = len(task.train_pairs)
    ctx_tokens.append(PAIR_SEP)
    ctx_rows.append(0)
    ctx_cols.append(0)
    ctx_pairs.append(test_pair_idx)

    test_input = task.test_pairs[test_idx]["input"]
    titok, tipos = grid_to_tokens(test_input)
    ctx_tokens.extend(titok)
    ctx_rows.extend([p[0] for p in tipos])
    ctx_cols.extend([p[1] for p in tipos])
    ctx_pairs.extend([test_pair_idx] * len(titok))

    ctx_tokens.append(GRID_SEP)
    ctx_rows.append(0)
    ctx_cols.append(0)
    ctx_pairs.append(test_pair_idx)

    # Truncate context to max_len
    ctx_tokens = ctx_tokens[:max_len]
    ctx_rows = ctx_rows[:max_len]
    ctx_cols = ctx_cols[:max_len]
    ctx_pairs = ctx_pairs[:max_len]

    # Target output (flat, no separators — just cell values)
    test_output = task.test_pairs[test_idx]["output"]
    out_h, out_w = predict_output_shape(task, test_idx)
    target_tokens = []
    target_rows = []
    target_cols = []
    for r in range(out_h):
        for c in range(out_w):
            if r < len(test_output) and c < len(test_output[r]):
                target_tokens.append(test_output[r][c])
            else:
                target_tokens.append(0)  # pad with black
            target_rows.append(r)
            target_cols.append(c)

    # Test input flat (no separators)
    ti_flat_tokens = []
    ti_flat_rows = []
    ti_flat_cols = []
    for r, row in enumerate(test_input):
        for c, val in enumerate(row):
            ti_flat_tokens.append(val)
            ti_flat_rows.append(r)
            ti_flat_cols.append(c)

    return {
        "context_tokens": torch.tensor(ctx_tokens, dtype=torch.long),
        "context_rows": torch.tensor(ctx_rows, dtype=torch.long),
        "context_cols": torch.tensor(ctx_cols, dtype=torch.long),
        "context_pairs": torch.tensor(ctx_pairs, dtype=torch.long),
        "target_tokens": torch.tensor(target_tokens, dtype=torch.long),
        "target_rows": torch.tensor(target_rows, dtype=torch.long),
        "target_cols": torch.tensor(target_cols, dtype=torch.long),
        "test_input_tokens": torch.tensor(ti_flat_tokens, dtype=torch.long),
        "test_input_rows": torch.tensor(ti_flat_rows, dtype=torch.long),
        "test_input_cols": torch.tensor(ti_flat_cols, dtype=torch.long),
        "output_h": out_h,
        "output_w": out_w,
    }


# ---------------------------------------------------------------------------
# Data Augmentation
# ---------------------------------------------------------------------------
def rotate_grid_90(grid: List[List[int]]) -> List[List[int]]:
    """Rotate grid 90° clockwise."""
    h, w = len(grid), len(grid[0])
    return [[grid[h - 1 - c][r] for c in range(h)] for r in range(w)]


def flip_grid_horizontal(grid: List[List[int]]) -> List[List[int]]:
    """Flip grid along the vertical axis (left-right)."""
    return [row[::-1] for row in grid]


def apply_dihedral(grid: List[List[int]], transform_id: int) -> List[List[int]]:
    """
    Apply one of 8 dihedral group transforms (D4) to a grid.

    transform_id 0: identity
    transform_id 1–3: 90°, 180°, 270° rotation
    transform_id 4: horizontal flip
    transform_id 5–7: flip + 90°, 180°, 270° rotation
    """
    g = [row[:] for row in grid]  # copy
    if transform_id >= 4:
        g = flip_grid_horizontal(g)
    for _ in range(transform_id % 4):
        g = rotate_grid_90(g)
    return g


def permute_colors(grid: List[List[int]], perm: Dict[int, int]) -> List[List[int]]:
    """Apply a color permutation to a grid. Color 0 (black) stays fixed."""
    return [[perm.get(val, val) for val in row] for row in grid]


def random_color_permutation() -> Dict[int, int]:
    """Generate a random permutation of non-zero colors 1–9."""
    colors = list(range(1, 10))
    shuffled = colors[:]
    random.shuffle(shuffled)
    return {orig: new for orig, new in zip(colors, shuffled)}


def augment_task(
    task: ARCTask,
    dihedral_id: int = 0,
    color_perm: Optional[Dict[int, int]] = None,
) -> ARCTask:
    """
    Return a new ARCTask with the specified geometric + color augmentation
    applied consistently to ALL grids in the task.
    """
    def transform(grid):
        g = apply_dihedral(grid, dihedral_id)
        if color_perm is not None:
            g = permute_colors(g, color_perm)
        return g

    new_train = [
        {"input": transform(p["input"]), "output": transform(p["output"])}
        for p in task.train_pairs
    ]
    new_test = [
        {"input": transform(p["input"]), "output": transform(p["output"])}
        for p in task.test_pairs
    ]
    return ARCTask(
        task_id=task.task_id,
        train_pairs=new_train,
        test_pairs=new_test,
    )


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class ARCDataset(Dataset):
    """
    ARC-AGI dataset with on-the-fly augmentation.

    Each __getitem__ returns the tokenized sequence for one (possibly augmented)
    task, ready for model consumption.
    """

    def __init__(
        self,
        tasks: List[ARCTask],
        augment: bool = True,
        n_dihedral: int = 8,
        n_color_perms: int = 2,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        self.tasks = tasks
        self.augment = augment
        self.n_dihedral = n_dihedral if augment else 1
        self.n_color_perms = n_color_perms if augment else 1
        self.max_seq_len = max_seq_len

        # Total virtual dataset size = tasks × dihedral × color_perms
        self.augments_per_task = self.n_dihedral * self.n_color_perms

    def __len__(self):
        return len(self.tasks) * self.augments_per_task

    def __getitem__(self, idx):
        task_idx = idx // self.augments_per_task
        aug_idx = idx % self.augments_per_task
        dihedral_id = aug_idx // self.n_color_perms
        color_idx = aug_idx % self.n_color_perms

        task = self.tasks[task_idx]

        if self.augment and (dihedral_id > 0 or color_idx > 0):
            color_perm = random_color_permutation() if color_idx > 0 else None
            task = augment_task(task, dihedral_id=dihedral_id, color_perm=color_perm)

        seq = build_task_sequence(task, test_idx=0, max_len=self.max_seq_len)
        seq["task_idx"] = task_idx
        return seq


# ---------------------------------------------------------------------------
# Collation (pad variable-length sequences)
# ---------------------------------------------------------------------------
def collate_arc_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate a list of task sequences into a padded batch.

    Pads context_* and target_* tensors to the max length in the batch.
    Returns attention masks for context tokens.
    """
    # Find max lengths
    max_ctx = max(b["context_tokens"].size(0) for b in batch)
    max_tgt = max(b["target_tokens"].size(0) for b in batch)
    max_ti = max(b["test_input_tokens"].size(0) for b in batch)

    B = len(batch)
    result = {
        "context_tokens": torch.full((B, max_ctx), PAD_TOKEN, dtype=torch.long),
        "context_rows": torch.zeros(B, max_ctx, dtype=torch.long),
        "context_cols": torch.zeros(B, max_ctx, dtype=torch.long),
        "context_pairs": torch.zeros(B, max_ctx, dtype=torch.long),
        "context_mask": torch.zeros(B, max_ctx, dtype=torch.bool),
        "target_tokens": torch.zeros(B, max_tgt, dtype=torch.long),
        "target_rows": torch.zeros(B, max_tgt, dtype=torch.long),
        "target_cols": torch.zeros(B, max_tgt, dtype=torch.long),
        "target_mask": torch.zeros(B, max_tgt, dtype=torch.bool),
        "test_input_tokens": torch.full((B, max_ti), PAD_TOKEN, dtype=torch.long),
        "test_input_rows": torch.zeros(B, max_ti, dtype=torch.long),
        "test_input_cols": torch.zeros(B, max_ti, dtype=torch.long),
        "test_input_mask": torch.zeros(B, max_ti, dtype=torch.bool),
        "output_h": torch.zeros(B, dtype=torch.long),
        "output_w": torch.zeros(B, dtype=torch.long),
        "task_idx": torch.zeros(B, dtype=torch.long),
    }

    for i, b in enumerate(batch):
        lc = b["context_tokens"].size(0)
        result["context_tokens"][i, :lc] = b["context_tokens"]
        result["context_rows"][i, :lc] = b["context_rows"]
        result["context_cols"][i, :lc] = b["context_cols"]
        result["context_pairs"][i, :lc] = b["context_pairs"]
        result["context_mask"][i, :lc] = True

        lt = b["target_tokens"].size(0)
        result["target_tokens"][i, :lt] = b["target_tokens"]
        result["target_rows"][i, :lt] = b["target_rows"]
        result["target_cols"][i, :lt] = b["target_cols"]
        result["target_mask"][i, :lt] = True

        li = b["test_input_tokens"].size(0)
        result["test_input_tokens"][i, :li] = b["test_input_tokens"]
        result["test_input_rows"][i, :li] = b["test_input_rows"]
        result["test_input_cols"][i, :li] = b["test_input_cols"]
        result["test_input_mask"][i, :li] = True

        result["output_h"][i] = b["output_h"]
        result["output_w"][i] = b["output_w"]
        result["task_idx"][i] = b["task_idx"]

    return result
