import pytest
import torch

from tts_impl.functional.f0_interpolation import interpolate_f0


def test_all_voiced():
    """Test that all-voiced input is returned unchanged."""
    f0 = torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0])
    result = interpolate_f0(f0)
    assert torch.allclose(result, f0)


def test_all_unvoiced():
    """Test that all-unvoiced input is returned unchanged."""
    f0 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    result = interpolate_f0(f0)
    assert torch.allclose(result, f0)


def test_leading_unvoiced():
    """Test that leading unvoiced regions are filled with first voiced value."""
    f0 = torch.tensor([0.0, 0.0, 100.0, 200.0, 300.0])
    result = interpolate_f0(f0)
    expected = torch.tensor([100.0, 100.0, 100.0, 200.0, 300.0])
    assert torch.allclose(result, expected)


def test_trailing_unvoiced():
    """Test that trailing unvoiced regions are filled with last voiced value."""
    f0 = torch.tensor([100.0, 200.0, 300.0, 0.0, 0.0])
    result = interpolate_f0(f0)
    expected = torch.tensor([100.0, 200.0, 300.0, 300.0, 300.0])
    assert torch.allclose(result, expected)


def test_middle_unvoiced_linear_interpolation():
    """Test that middle unvoiced regions are linearly interpolated."""
    f0 = torch.tensor([100.0, 0.0, 0.0, 0.0, 200.0])
    result = interpolate_f0(f0)
    expected = torch.tensor([100.0, 125.0, 150.0, 175.0, 200.0])
    assert torch.allclose(result, expected)


def test_multiple_unvoiced_regions():
    """Test interpolation with multiple unvoiced regions."""
    f0 = torch.tensor([100.0, 0.0, 200.0, 0.0, 0.0, 300.0])
    result = interpolate_f0(f0)
    expected = torch.tensor([100.0, 150.0, 200.0, 233.33333, 266.66666, 300.0])
    assert torch.allclose(result, expected, atol=1e-4)


def test_low_pitch_treated_as_unvoiced():
    """Test that values below f0_min (20.0 Hz) are treated as unvoiced."""
    f0 = torch.tensor([100.0, 10.0, 5.0, 15.0, 200.0])
    result = interpolate_f0(f0)
    expected = torch.tensor([100.0, 125.0, 150.0, 175.0, 200.0])
    assert torch.allclose(result, expected)


def test_low_pitch_at_edges():
    """Test that low pitch values at edges are handled correctly."""
    f0 = torch.tensor([5.0, 10.0, 100.0, 200.0, 15.0, 0.0])
    result = interpolate_f0(f0)
    expected = torch.tensor([100.0, 100.0, 100.0, 200.0, 200.0, 200.0])
    assert torch.allclose(result, expected)


def test_custom_f0_min():
    """Test with custom f0_min threshold."""
    f0 = torch.tensor([100.0, 50.0, 30.0, 200.0])
    # With default f0_min=20.0, all values are voiced
    result_default = interpolate_f0(f0)
    assert torch.allclose(result_default, f0)

    # With f0_min=60.0, 50.0 and 30.0 are treated as unvoiced
    result_custom = interpolate_f0(f0, f0_min=60.0)
    expected = torch.tensor([100.0, 133.33333, 166.66666, 200.0])
    assert torch.allclose(result_custom, expected, atol=1e-4)


def test_2d_tensor():
    """Test interpolation with 2D tensor (batch)."""
    f0 = torch.tensor([
        [100.0, 0.0, 200.0],
        [50.0, 0.0, 0.0],
    ])
    result = interpolate_f0(f0)
    expected = torch.tensor([
        [100.0, 150.0, 200.0],
        [50.0, 50.0, 50.0],
    ])
    assert torch.allclose(result, expected)


def test_1d_tensor():
    """Test interpolation with 1D tensor."""
    f0 = torch.tensor([100.0, 0.0, 200.0])
    result = interpolate_f0(f0)
    expected = torch.tensor([100.0, 150.0, 200.0])
    assert torch.allclose(result, expected)


def test_single_voiced_value():
    """Test with only one voiced value."""
    f0 = torch.tensor([0.0, 0.0, 100.0, 0.0, 0.0])
    result = interpolate_f0(f0)
    expected = torch.tensor([100.0, 100.0, 100.0, 100.0, 100.0])
    assert torch.allclose(result, expected)


def test_preserves_device():
    """Test that output tensor is on the same device as input."""
    f0 = torch.tensor([100.0, 0.0, 200.0])
    result = interpolate_f0(f0)
    assert result.device == f0.device


def test_preserves_dtype():
    """Test that output tensor has the same dtype as input."""
    f0 = torch.tensor([100.0, 0.0, 200.0], dtype=torch.float64)
    result = interpolate_f0(f0)
    assert result.dtype == f0.dtype


def test_does_not_modify_input():
    """Test that input tensor is not modified."""
    f0 = torch.tensor([100.0, 0.0, 200.0])
    f0_original = f0.clone()
    _ = interpolate_f0(f0)
    assert torch.allclose(f0, f0_original)


def test_invalid_ndim():
    """Test that invalid tensor dimensions raise an error."""
    f0 = torch.tensor([[[100.0, 0.0, 200.0]]])
    with pytest.raises(ValueError):
        interpolate_f0(f0)


def test_boundary_value_exactly_f0_min():
    """Test that value exactly equal to f0_min is treated as voiced."""
    f0 = torch.tensor([100.0, 20.0, 200.0])
    result = interpolate_f0(f0)
    # 20.0 is exactly f0_min, should be treated as voiced
    assert torch.allclose(result, f0)


def test_boundary_value_just_below_f0_min():
    """Test that value just below f0_min is treated as unvoiced."""
    f0 = torch.tensor([100.0, 19.99, 200.0])
    result = interpolate_f0(f0)
    expected = torch.tensor([100.0, 150.0, 200.0])
    assert torch.allclose(result, expected, atol=1e-2)
