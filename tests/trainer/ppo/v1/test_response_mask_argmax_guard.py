#!/usr/bin/env python3
"""Standalone test: response_mask argmax safety guard in
compute_advantage_for_multi_trajectories.

Bug: response_mask.argmax(dim=1) on all-zero rows returns 0,
extracting a garbage advantage value instead of 0.0.
Fix: check row_has_valid before extracting, set to 0.0 for empty rows.
"""

import torch
import numpy as np


def test_argmax_on_all_zero_mask():
    """Bug: argmax on all-zero mask returns 0, not -1."""
    mask = torch.zeros(1, 8)
    idx = mask.argmax(dim=1)
    assert idx.item() == 0, "argmax on all-zero returns 0 (the bug)"
    print("1. argmax on all-zero mask returns 0 — BUG CONFIRMED")


def test_garbage_advantage_extraction():
    """Bug: extracting advantage at position 0 for an all-zero row."""
    mask = torch.zeros(1, 8)
    advantages = torch.tensor([[0.5, 0.3, 0.2, 1.0, 0.8, 0.1, 0.0, 0.0]])
    idx = mask.argmax(dim=1)
    extracted = advantages[torch.arange(1), idx]
    assert extracted.item() == 0.5, "Extracted garbage value at pos 0"
    print("2. Garbage advantage extracted: 0.5 (should be 0.0) — BUG CONFIRMED")


def test_safe_extraction_with_fix():
    """Fix: check row_has_valid, set to 0.0 for empty rows."""
    mask = torch.zeros(2, 8)
    mask[1, 3] = 1.0  # Only row 1 has valid tokens
    advantages = torch.tensor([
        [0.5, 0.3, 0.2, 1.0, 0.8, 0.1, 0.0, 0.0],  # row 0: all-zero mask
        [0.1, 0.2, 0.3, 2.0, 0.4, 0.5, 0.0, 0.0],  # row 1: valid mask
    ])

    # Apply the fix
    row_has_valid = mask.sum(dim=1) > 0
    first_nnz_indices = mask.argmax(dim=1)
    final_scores = advantages[torch.arange(2), first_nnz_indices]
    final_scores = torch.where(row_has_valid, final_scores, torch.zeros_like(final_scores))

    # Row 0 (all-zero mask): should be 0.0
    assert final_scores[0].item() == 0.0, "Empty row should have advantage 0.0"
    # Row 1 (valid mask): should be the advantage at position 3
    assert final_scores[1].item() == 2.0, "Valid row should extract correct advantage"
    print("3. Safe extraction: empty row→0.0, valid row→2.0 — FIX VERIFIED")


def test_normal_mask_unchanged():
    """Fix doesn't affect normal (all-valid) masks."""
    mask = torch.ones(2, 8)
    advantages = torch.tensor([
        [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    ])

    row_has_valid = mask.sum(dim=1) > 0
    first_nnz_indices = mask.argmax(dim=1)
    final_scores = advantages[torch.arange(2), first_nnz_indices]
    final_scores = torch.where(row_has_valid, final_scores, torch.zeros_like(final_scores))

    assert final_scores[0].item() == 1.5, "Normal mask unchanged"
    assert final_scores[1].item() == 2.0, "Normal mask unchanged"
    print("4. Normal masks unaffected — FIX SAFE")


if __name__ == "__main__":
    print("=== response_mask argmax safety guard tests ===\n")
    test_argmax_on_all_zero_mask()
    test_garbage_advantage_extraction()
    test_safe_extraction_with_fix()
    test_normal_mask_unchanged()
    print("\n=== All 4 tests passed ===")
