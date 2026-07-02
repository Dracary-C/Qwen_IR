from __future__ import annotations

import unittest

import torch

from module.layout_prompt import StructuredPriorV2, normalize_severity


def sample_payload() -> dict:
    return {
        "qwen_parsed": {
            "noise": "mild",
            "blur": "severe",
            "haze": "none",
            "rain": "moderate",
            "low_light": "serious",
            "degradation_layout": {
                "global": True,
                "local_region": True,
                "object_specific": False,
                "continuous": True,
                "discrete": False,
                "directional": True,
                "depth_dependent": False,
                "shadow_dependent": True,
                "texture_dependent": True,
                "uncertain": False,
            },
        },
        "condition_scoring": {
            "scores": [
                {"candidate": "blur", "avg_logprob": -0.1},
                {"candidate": "noise", "avg_logprob": -2.0},
                {"candidate": "haze", "avg_logprob": -3.0},
                {"candidate": "rain", "avg_logprob": -4.0},
                {"candidate": "low_light", "avg_logprob": -5.0},
            ]
        },
    }


class StructuredPriorV2Test(unittest.TestCase):
    def test_serious_and_severe_are_equivalent(self) -> None:
        self.assertEqual(normalize_severity("serious"), 1.0)
        self.assertEqual(normalize_severity("severe"), 1.0)

    def test_calibrated_probs_and_confidence(self) -> None:
        prior = StructuredPriorV2.from_qwen_payload(sample_payload(), temperature=2.0)
        self.assertEqual(tuple(prior.main_logits_5.shape), (5,))
        self.assertAlmostEqual(float(prior.main_probs_5.sum()), 1.0, places=6)
        self.assertAlmostEqual(
            float(prior.calibrated_confidence[0]),
            float(prior.main_probs_5.max()),
            places=6,
        )
        self.assertLess(float(prior.calibrated_confidence[0]), float(torch.softmax(prior.main_logits_5, 0).max()))

    def test_a2_zeros_only_severity_and_layout(self) -> None:
        prior = StructuredPriorV2.from_qwen_payload(sample_payload(), temperature=2.0)
        vector = prior.to_model_vector("qwen_probs")
        self.assertEqual(tuple(vector.shape), (21,))
        self.assertTrue(torch.equal(vector[0:5], torch.zeros(5)))
        self.assertTrue(torch.allclose(vector[5:10], prior.main_probs_5))
        self.assertTrue(torch.equal(vector[10:20], torch.zeros(10)))

    def test_a3_preserves_all_five_severities(self) -> None:
        prior = StructuredPriorV2.from_qwen_payload(sample_payload(), temperature=2.0)
        vector = prior.to_model_vector("qwen_probs_severity")
        expected = torch.tensor([1.0 / 3.0, 1.0, 0.0, 2.0 / 3.0, 1.0])
        self.assertTrue(torch.allclose(vector[0:5], expected))
        self.assertGreater(int((vector[0:5] > 0).sum()), 1)

    def test_a4_uses_the_same_prior_values_as_a3(self) -> None:
        prior = StructuredPriorV2.from_qwen_payload(sample_payload(), temperature=2.0)
        self.assertTrue(torch.equal(
            prior.to_model_vector("confidence_gate"),
            prior.to_model_vector("qwen_probs_severity"),
        ))

    def test_legacy_roundtrip(self) -> None:
        vector = torch.tensor([
            0, 1, 0, 0, 0,
            0.1, 0.6, 0.1, 0.1, 0.1,
            1, 0, 0, 1, 0, 1, 0, 0, 1, 0,
            0.5,
        ], dtype=torch.float32)
        rebuilt = StructuredPriorV2.from_legacy_vector(vector).to_legacy_vector()
        self.assertTrue(torch.allclose(rebuilt, vector))


if __name__ == "__main__":
    unittest.main()
