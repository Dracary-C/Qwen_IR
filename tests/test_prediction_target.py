import pytest
import torch

from module.pipeline.assess_tpgd import _direct_restored, _matching_loss


def test_image_prediction_is_returned_unchanged() -> None:
    prediction = torch.randn(2, 3, 4, 4)
    lq = torch.randn_like(prediction)

    restored = _direct_restored(prediction, lq, "image")

    assert restored is prediction


def test_residual_prediction_adds_lq_and_preserves_l1_equivalence() -> None:
    lq = torch.randn(2, 3, 4, 4)
    residual = torch.randn_like(lq)
    gt = torch.randn_like(lq)

    restored = _direct_restored(residual, lq, "residual")

    torch.testing.assert_close(restored, lq + residual)
    image_loss = _matching_loss(restored, gt, "l1")
    residual_loss = _matching_loss(residual, gt - lq, "l1")
    torch.testing.assert_close(image_loss, residual_loss)


def test_invalid_prediction_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported prediction_target"):
        _direct_restored(torch.zeros(1), torch.zeros(1), "unknown")
