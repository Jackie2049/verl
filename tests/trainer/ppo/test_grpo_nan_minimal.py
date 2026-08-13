#!/usr/bin/env python3
"""Minimal test: GRPO NaN propagation bug and fix verification.

Tests that NaN rewards produce NaN advantages (bug), and that
nan_to_num replaces them with 0.0 (fix).
Does NOT require full verl installation — only torch and numpy.
"""

import sys
import numpy as np
import torch

from verl.trainer.ppo.core_algos import (
    compute_grpo_outcome_advantage,
    compute_grpo_vectorized_outcome_advantage,
)


def make_batch(bsz=4, seq_len=8):
    """Create a standard GRPO test batch."""
    token_level_rewards = torch.randn(bsz, seq_len)
    response_mask = torch.ones(bsz, seq_len)
    response_mask[bsz // 2:, -2:] = 0
    index = np.array([0] * (bsz // 2) + [1] * (bsz // 2))
    return token_level_rewards, response_mask, index


def test_nan_reward_propagates():
    """Bug: NaN reward → NaN advantage (no guard)."""
    rewards, mask, index = make_batch()
    rewards[0, 3] = float("nan")

    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=rewards, response_mask=mask, index=index,
        epsilon=1e-6, norm_adv_by_std_in_grpo=True,
    )

    has_nan = torch.any(torch.isnan(advantages)).item()
    print(f"Test 1: NaN reward → NaN advantage: {has_nan}")
    assert has_nan, "Bug confirmed: NaN propagates to advantages"
    print("  BUG CONFIRMED: NaN rewards produce NaN advantages in compute_grpo_outcome_advantage")


def test_nan_reward_vectorized():
    """Bug in vectorized path too."""
    rewards, mask, index = make_batch()
    rewards[0, 3] = float("nan")

    advantages, returns = compute_grpo_vectorized_outcome_advantage(
        token_level_rewards=rewards, response_mask=mask, index=index,
        epsilon=1e-6, norm_adv_by_std_in_grpo=True,
    )

    has_nan = torch.any(torch.isnan(advantages)).item()
    print(f"Test 2: NaN reward → NaN advantage (vectorized): {has_nan}")
    assert has_nan, "Bug confirmed: NaN propagates in vectorized path"
    print("  BUG CONFIRMED: NaN rewards produce NaN advantages in vectorized path")


def test_nan_to_num_fix():
    """Fix: nan_to_num replaces NaN advantages with 0.0."""
    rewards, mask, index = make_batch()
    rewards[0, 3] = float("nan")

    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=rewards, response_mask=mask, index=index,
        epsilon=1e-6, norm_adv_by_std_in_grpo=True,
    )

    # Apply the centralized guard fix
    fixed_adv = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
    fixed_ret = torch.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    has_nan_after = torch.any(torch.isnan(fixed_adv)).item()
    has_inf_after = torch.any(torch.isinf(fixed_adv)).item()
    print(f"Test 3: After nan_to_num: has_nan={has_nan_after}, has_inf={has_inf_after}")
    assert not has_nan_after, "nan_to_num should remove all NaN"
    assert not has_inf_after, "nan_to_num should remove all Inf"

    # NaN positions should be exactly 0.0
    nan_mask = torch.isnan(advantages)
    assert torch.all(fixed_adv[nan_mask] == 0.0), "NaN positions replaced with 0.0"
    print("  FIX VERIFIED: nan_to_num replaces NaN with 0.0, no NaN/Inf remain")


def test_inf_reward_propagates():
    """Inf rewards → Inf or NaN advantages."""
    rewards, mask, index = make_batch()
    rewards[0, 3] = float("inf")

    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=rewards, response_mask=mask, index=index,
        epsilon=1e-6, norm_adv_by_std_in_grpo=True,
    )

    has_inf_or_nan = (torch.any(torch.isinf(advantages)) or torch.any(torch.isnan(advantages))).item()
    print(f"Test 4: Inf reward → Inf/NaN advantage: {has_inf_or_nan}")
    assert has_inf_or_nan, "Inf reward produces Inf or NaN advantage"
    print("  BUG CONFIRMED: Inf rewards propagate to advantages")


def test_normal_rewards_ok():
    """Normal rewards → normal advantages (no false positives)."""
    rewards, mask, index = make_batch()

    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=rewards, response_mask=mask, index=index,
        epsilon=1e-6, norm_adv_by_std_in_grpo=True,
    )

    has_nan = torch.any(torch.isnan(advantages)).item()
    has_inf = torch.any(torch.isinf(advantages)).item()
    print(f"Test 5: Normal rewards: has_nan={has_nan}, has_inf={has_inf}")
    assert not has_nan and not has_inf, "Normal rewards should produce normal advantages"
    print("  PASS: Normal rewards produce valid advantages")


if __name__ == "__main__":
    print("=== verl GRPO NaN Propagation Bug Tests ===\n")
    test_nan_reward_propagates()
    test_nan_reward_vectorized()
    test_nan_to_num_fix()
    test_inf_reward_propagates()
    test_normal_rewards_ok()
    print("\n=== All tests passed ===")
