from __future__ import annotations

import unittest

import torch

from module.layout_prompt import QwenPromptPriorAdapter


def _prior(batch: int = 4) -> torch.Tensor:
    prior = torch.zeros(batch, 21)
    prior[:, 0] = 1.0
    prior[:, 5:10] = torch.tensor([0.7, 0.1, 0.1, 0.05, 0.05])
    return prior


class ConfidenceGateTest(unittest.TestCase):
    def test_default_adapter_preserves_a3_direct_context(self) -> None:
        adapter = QwenPromptPriorAdapter(
            context_dim=8,
            hidden_dim=16,
            use_layout_tokens=False,
        ).eval()
        contexts = adapter(_prior(2))
        self.assertIsNone(adapter.unknown_context)
        self.assertIsNone(contexts.gate_confidence)
        self.assertTrue(torch.equal(contexts.deg_context, contexts.qwen_context))

    def test_interpolation_and_overrides(self) -> None:
        adapter = QwenPromptPriorAdapter(
            context_dim=8,
            hidden_dim=16,
            use_layout_tokens=False,
            use_confidence_gate=True,
        ).eval()
        assert adapter.unknown_context is not None
        with torch.no_grad():
            adapter.unknown_context.fill_(2.0)
        prior = _prior(2)

        normal = adapter(prior)
        expected = 0.7 * normal.qwen_context + 0.3 * torch.full_like(normal.qwen_context, 2.0)
        self.assertTrue(torch.allclose(normal.deg_context, expected, atol=1e-6))

        force_zero = adapter(prior, confidence_override=0.0)
        self.assertTrue(torch.allclose(force_zero.deg_context, torch.full_like(force_zero.deg_context, 2.0)))
        force_one = adapter(prior, confidence_override=1.0)
        self.assertTrue(torch.allclose(force_one.deg_context, force_one.qwen_context))

    def test_condition_dropout_forces_unknown_and_trains_it(self) -> None:
        adapter = QwenPromptPriorAdapter(
            context_dim=8,
            hidden_dim=16,
            use_layout_tokens=False,
            use_confidence_gate=True,
            condition_dropout_probability=1.0,
        ).train()
        contexts = adapter(_prior())
        self.assertTrue(contexts.condition_dropout_mask.all())
        self.assertFalse(contexts.prior_corruption_mask.any())
        self.assertTrue(torch.equal(contexts.gate_confidence, torch.zeros(4, 1)))
        contexts.deg_context.sum().backward()
        assert adapter.unknown_context is not None
        self.assertIsNotNone(adapter.unknown_context.grad)
        self.assertGreater(float(adapter.unknown_context.grad.abs().sum()), 0.0)

    def test_corruption_is_exclusive_and_forces_zero_confidence(self) -> None:
        adapter = QwenPromptPriorAdapter(
            context_dim=8,
            hidden_dim=16,
            use_layout_tokens=False,
            use_confidence_gate=True,
            prior_corruption_probability=1.0,
        ).train()
        contexts = adapter(_prior())
        self.assertFalse(contexts.condition_dropout_mask.any())
        self.assertTrue(contexts.prior_corruption_mask.all())
        self.assertTrue(torch.equal(contexts.gate_confidence, torch.zeros(4, 1)))
        self.assertFalse(bool((contexts.condition_dropout_mask & contexts.prior_corruption_mask).any()))

    def test_planned_rates_are_mutually_exclusive(self) -> None:
        torch.manual_seed(1234)
        adapter = QwenPromptPriorAdapter(
            context_dim=8,
            hidden_dim=16,
            use_layout_tokens=False,
            use_confidence_gate=True,
            condition_dropout_probability=0.2,
            prior_corruption_probability=0.1,
        ).train()
        contexts = adapter(_prior(10000))
        dropout_rate = float(contexts.condition_dropout_mask.float().mean())
        corruption_rate = float(contexts.prior_corruption_mask.float().mean())
        self.assertLess(abs(dropout_rate - 0.2), 0.02)
        self.assertLess(abs(corruption_rate - 0.1), 0.02)
        self.assertFalse(bool((contexts.condition_dropout_mask & contexts.prior_corruption_mask).any()))


if __name__ == "__main__":
    unittest.main()
