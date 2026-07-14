#!/usr/bin/env python3
"""Standalone test: GRPO NaN propagation bug and fix verification.

This test implements the GRPO advantage computation inline (no verl dependency)
to verify that NaN rewards produce NaN advantages, and that nan_to_num fixes it.
"""

import numpy as np
import torch


def compute_grpo_advantage_standalone(token_level_rewards, response_mask, index, epsilon=1e-6):
    """Standalone GRPO advantage computation (mirrors verl's logic)."""
    from collections import defaultdict
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1) * response_mask
    return scores, scores


def make_batch(bsz=4, seq_len=8):
    token_level_rewards = torch.randn(bsz, seq_len)
    response_mask = torch.ones(bsz, seq_len)
    response_mask[bsz // 2:, -2:] = 0
    index = np.array([0] * (bsz // 2) + [1] * (bsz // 2))
    return token_level_rewards, response_mask, index


def test_nan_reward():
    rewards, mask, index = make_batch()
    rewards[0, 3] = float("nan")
    adv, ret = compute_grpo_advantage_standalone(rewards, mask, index)
    has_nan = torch.any(torch.isnan(adv)).item()
    print(f"1. NaN reward → NaN advantage: {has_nan} ✓ (bug confirmed)")
    assert has_nan


def test_inf_reward():
    rewards, mask, index = make_batch()
    rewards[0, 3] = float("inf")
    adv, ret = compute_grpo_advantage_standalone(rewards, mask, index)
    has_problem = (torch.any(torch.isinf(adv)) or torch.any(torch.isnan(adv))).item()
    print(f"2. Inf reward → Inf/NaN advantage: {has_problem} ✓ (bug confirmed)")
    assert has_problem


def test_nan_to_num_fix():
    rewards, mask, index = make_batch()
    rewards[0, 3] = float("nan")
    adv, ret = compute_grpo_advantage_standalone(rewards, mask, index)
    fixed = torch.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)
    no_nan = not torch.any(torch.isnan(fixed)).item()
    no_inf = not torch.any(torch.isinf(fixed)).item()
    print(f"3. nan_to_num removes NaN/Inf: no_nan={no_nan}, no_inf={no_inf} ✓ (fix works)")
    assert no_nan and no_inf
    # NaN positions are exactly 0.0
    nan_mask = torch.isnan(adv)
    assert torch.all(fixed[nan_mask] == 0.0)


def test_normal_rewards():
    rewards, mask, index = make_batch()
    adv, ret = compute_grpo_advantage_standalone(rewards, mask, index)
    no_nan = not torch.any(torch.isnan(adv)).item()
    no_inf = not torch.any(torch.isinf(adv)).item()
    print(f"4. Normal rewards → normal advantages: no_nan={no_nan}, no_inf={no_inf} ✓")
    assert no_nan and no_inf


def test_singleton_group():
    """Single sample in group → mean=0, std=1 → advantage ≈ score (not zero)."""
    rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # 1 sample
    mask = torch.ones(1, 4)
    index = np.array([0])
    adv, ret = compute_grpo_advantage_standalone(rewards, mask, index)
    # With singleton: mean=0, std=1 → (1.0-0)/(1+eps) ≈ 1.0
    first_val = adv[0, 0].item()
    print(f"5. Singleton group → advantage ≈ raw score: {first_val:.4f}")
    assert abs(first_val - 1.0) < 0.01, "Singleton should give advantage ≈ score"


if __name__ == "__main__":
    print("=== GRPO NaN Propagation Bug & Fix Tests ===\n")
    test_nan_reward()
    test_inf_reward()
    test_nan_to_num_fix()
    test_normal_rewards()
    test_singleton_group()
    print("\n=== All 5 tests passed ===")
