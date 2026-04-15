"""Utility functions: parameter counting, visualization, misc helpers."""

import torch
import torch.nn as nn
from typing import Dict


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_breakdown(model: nn.Module) -> Dict[str, int]:
    """Parameter count per top-level submodule."""
    breakdown = {}
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters() if p.requires_grad)
        breakdown[name] = n
    breakdown["__total__"] = count_parameters(model)
    return breakdown


def print_parameter_table(model: nn.Module):
    """Pretty-print parameter budget table."""
    bd = parameter_breakdown(model)
    total = bd.pop("__total__")
    print(f"{'Component':<30} {'Parameters':>15}")
    print("-" * 47)
    for name, count in bd.items():
        pct = count / total * 100
        print(f"{name:<30} {count:>12,} ({pct:5.1f}%)")
    print("-" * 47)
    print(f"{'TOTAL':<30} {total:>12,}")
    budget = 50_000_000
    print(f"{'Budget (50M)':<30} {budget:>12,}")
    print(f"{'Remaining':<30} {budget - total:>12,}")
    assert total <= budget, f"OVER BUDGET! {total:,} > {budget:,}"
    print("✓ Within parameter budget")


def format_num(n: int) -> str:
    """Format large numbers: 14200000 → '14.2M'."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)
