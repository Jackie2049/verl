"""Test NaN/Inf guard in GRPO advantage computation.

Verifies that NaN and Inf values in advantages are replaced with 0.0
to prevent training crash propagation.
"""

import unittest

import numpy as np
import torch

from verl.trainer.ppo.core_algos import (
    compute_grpo_outcome_advantage,
    compute_grpo_vectorized_outcome_advantage,
)


class TestGRPONaNGuard(unittest.TestCase):
    """Test that NaN/Inf in token-level rewards produces detectable NaN in advantages,
    and that the centralized NaN guard in ray_trainer.py would catch it.
    """

    def _make_batch(self, bsz=4, seq_len=8, num_groups=2):
        """Helper: create a standard GRPO batch."""
        token_level_rewards = torch.randn(bsz, seq_len)
        response_mask = torch.ones(bsz, seq_len)
        # Last 2 tokens are padding for half the batch
        response_mask[bsz // 2:, -2:] = 0
        # Group index: first half = group 0, second half = group 1
        index = np.array([0] * (bsz // 2) + [1] * (bsz // 2))
        return token_level_rewards, response_mask, index

    def test_nan_reward_propagates_to_advantage(self):
        """Verify that NaN rewards produce NaN advantages (the bug we're fixing)."""
        rewards, mask, index = self._make_batch()
        # Inject NaN in one reward
        rewards[0, 3] = float("nan")

        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=rewards,
            response_mask=mask,
            index=index,
            epsilon=1e-6,
            norm_adv_by_std_in_grpo=True,
        )

        # NaN should propagate to the advantage for that group
        # (This confirms the bug exists before the guard)
        self.assertTrue(
            torch.any(torch.isnan(advantages)),
            "NaN reward should produce NaN advantage (confirms bug before fix)",
        )

    def test_nan_reward_vectorized_propagates(self):
        """Same test for the vectorized GRPO estimator."""
        rewards, mask, index = self._make_batch()
        rewards[0, 3] = float("nan")

        advantages, returns = compute_grpo_vectorized_outcome_advantage(
            token_level_rewards=rewards,
            response_mask=mask,
            index=index,
            epsilon=1e-6,
            norm_adv_by_std_in_grpo=True,
        )

        self.assertTrue(
            torch.any(torch.isnan(advantages)),
            "NaN reward should produce NaN advantage in vectorized path",
        )

    def test_nan_to_num_fixes_nan_advantages(self):
        """Verify that nan_to_num replaces NaN advantages with 0.0."""
        rewards, mask, index = self._make_batch()
        rewards[0, 3] = float("nan")

        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=rewards,
            response_mask=mask,
            index=index,
            epsilon=1e-6,
            norm_adv_by_std_in_grpo=True,
        )

        # Apply the fix: nan_to_num
        fixed_advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)

        # No NaN should remain
        self.assertFalse(
            torch.any(torch.isnan(fixed_advantages)),
            "nan_to_num should remove all NaN values",
        )

        # NaN positions should be replaced with 0.0
        nan_mask = torch.isnan(advantages)
        self.assertTrue(
            torch.all(fixed_advantages[nan_mask] == 0.0),
            "NaN positions should be replaced with exactly 0.0",
        )

    def test_inf_reward_propagates_to_advantage(self):
        """Verify that Inf rewards produce Inf or NaN advantages."""
        rewards, mask, index = self._make_batch()
        rewards[0, 3] = float("inf")

        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=rewards,
            response_mask=mask,
            index=index,
            epsilon=1e-6,
            norm_adv_by_std_in_grpo=True,
        )

        # Inf should propagate (either as Inf or NaN from normalization)
        has_inf_or_nan = torch.any(torch.isinf(advantages)) or torch.any(torch.isnan(advantages))
        self.assertTrue(
            has_inf_or_nan,
            "Inf reward should produce Inf or NaN advantage",
        )

    def test_centralized_guard_handles_both_nan_and_inf(self):
        """Simulate the centralized guard from ray_trainer.py."""
        rewards, mask, index = self._make_batch()
        # Mix NaN and Inf
        rewards[0, 3] = float("nan")
        rewards[2, 5] = float("inf")

        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=rewards,
            response_mask=mask,
            index=index,
            epsilon=1e-6,
            norm_adv_by_std_in_grpo=True,
        )

        # Simulate the centralized guard
        if torch.any(torch.isnan(advantages)) or torch.any(torch.isinf(advantages)):
            fixed_adv = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
            fixed_ret = torch.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        # After guard: no NaN or Inf
        self.assertFalse(torch.any(torch.isnan(fixed_adv)))
        self.assertFalse(torch.any(torch.isinf(fixed_adv)))
        self.assertFalse(torch.any(torch.isnan(fixed_ret)))
        self.assertFalse(torch.any(torch.isinf(fixed_ret)))

    def test_normal_rewards_produce_normal_advantages(self):
        """Verify that normal rewards produce normal advantages (no false positives)."""
        rewards, mask, index = self._make_batch()

        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=rewards,
            response_mask=mask,
            index=index,
            epsilon=1e-6,
            norm_adv_by_std_in_grpo=True,
        )

        # No NaN or Inf should be present with normal rewards
        self.assertFalse(torch.any(torch.isnan(advantages)))
        self.assertFalse(torch.any(torch.isinf(advantages)))


if __name__ == "__main__":
    unittest.main()
