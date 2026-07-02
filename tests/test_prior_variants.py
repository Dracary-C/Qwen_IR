from __future__ import annotations

import unittest

import torch
from torch.utils.data import Dataset

from module.layout_prompt import DEGRADATION_ORDER
from module.pipeline.assess_tpgd import StructuredPriorVariantDataset


class _FiveTaskDataset(Dataset):
    def __len__(self) -> int:
        return len(DEGRADATION_ORDER)

    def __getitem__(self, index: int) -> dict:
        prior = torch.zeros(21)
        prior[index] = 1.0
        prior[5 + index] = 0.8
        prior[5 + ((index + 1) % 5)] = 0.2
        return {
            "structured_prior": prior,
            "degradation": DEGRADATION_ORDER[index],
        }


class PriorVariantTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base = _FiveTaskDataset()

    def test_correct_is_unchanged(self) -> None:
        actual = StructuredPriorVariantDataset(self.base, "correct")[0]["structured_prior"]
        self.assertTrue(torch.equal(actual, self.base[0]["structured_prior"]))

    def test_zero(self) -> None:
        actual = StructuredPriorVariantDataset(self.base, "zero")[0]["structured_prior"]
        self.assertTrue(torch.equal(actual, torch.zeros(21)))

    def test_uniform(self) -> None:
        actual = StructuredPriorVariantDataset(self.base, "uniform")[0]["structured_prior"]
        self.assertTrue(torch.equal(actual[:5], torch.zeros(5)))
        self.assertTrue(torch.allclose(actual[5:10], torch.full((5,), 0.2)))
        self.assertTrue(torch.equal(actual[10:], torch.zeros(11)))

    def test_shuffled_uses_next_task_block(self) -> None:
        actual = StructuredPriorVariantDataset(self.base, "shuffled")[0]["structured_prior"]
        self.assertTrue(torch.equal(actual, self.base[1]["structured_prior"]))

    def test_forced_wrong_preserves_distribution_and_is_wrong(self) -> None:
        for index, degradation in enumerate(DEGRADATION_ORDER):
            actual = StructuredPriorVariantDataset(self.base, "forced_wrong")[index]["structured_prior"]
            self.assertNotEqual(int(actual[5:10].argmax()), index)
            self.assertTrue(torch.equal(
                torch.sort(actual[5:10]).values,
                torch.sort(self.base[index]["structured_prior"][5:10]).values,
            ))
            self.assertEqual(int((actual[:5] > 0).sum()), 1)


if __name__ == "__main__":
    unittest.main()
