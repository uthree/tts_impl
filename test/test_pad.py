import pytest
import torch
from tts_impl.functional.pad import (
    adjust_size,
    adjust_size_1d,
    adjust_size_2d,
    adjust_size_3d,
)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("input_length", [100, 256])
@pytest.mark.parametrize("target_length", [50, 100, 300])
def test_adjust_size_1d(
    batch_size: int, channels: int, input_length: int, target_length: int
):
    """Test adjust_size_1d function with various input and target sizes."""
    x = torch.randn(batch_size, channels, input_length)
    result = adjust_size_1d(x, target_length)

    assert result.shape[0] == batch_size
    assert result.shape[1] == channels
    assert result.shape[2] == target_length


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("input_size", [(64, 64), (100, 200)])
@pytest.mark.parametrize("target_size", [(32, 32), (64, 64), (128, 128), (50, 100)])
def test_adjust_size_2d(
    batch_size: int, channels: int, input_size: tuple, target_size: tuple
):
    """Test adjust_size_2d function with various input and target sizes."""
    h, w = input_size
    x = torch.randn(batch_size, channels, h, w)
    result = adjust_size_2d(x, target_size)

    target_h, target_w = target_size
    assert result.shape[0] == batch_size
    assert result.shape[1] == channels
    assert result.shape[2] == target_h
    assert result.shape[3] == target_w


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("input_size", [(16, 16, 16), (32, 32, 32)])
@pytest.mark.parametrize("target_size", [(8, 8, 8), (16, 16, 16), (48, 48, 48)])
def test_adjust_size_3d(
    batch_size: int, channels: int, input_size: tuple, target_size: tuple
):
    """Test adjust_size_3d function with various input and target sizes."""
    d, h, w = input_size
    x = torch.randn(batch_size, channels, d, h, w)
    result = adjust_size_3d(x, target_size)

    target_d, target_h, target_w = target_size
    assert result.shape[0] == batch_size
    assert result.shape[1] == channels
    assert result.shape[2] == target_d
    assert result.shape[3] == target_h
    assert result.shape[4] == target_w


@pytest.mark.parametrize(
    "input_shape,target_size",
    [
        ((100,), 50),
        ((100,), 150),
        ((2, 100), 50),
        ((2, 100), 150),
        ((2, 3, 100), 50),
        ((2, 3, 100), 150),
        ((2, 3, 64, 64), (32, 32)),
        ((2, 3, 64, 64), (128, 128)),
        ((2, 3, 16, 16, 16), (8, 8, 8)),
        ((2, 3, 16, 16, 16), (32, 32, 32)),
    ],
)
def test_adjust_size_generic(input_shape: tuple, target_size):
    """Test the generic adjust_size function with various tensor dimensions."""
    x = torch.randn(*input_shape)
    result = adjust_size(x, target_size)

    # Check that the last dimension(s) match the target size
    if isinstance(target_size, int):
        assert result.shape[-1] == target_size
    elif len(target_size) == 2:
        assert result.shape[-2] == target_size[0]
        assert result.shape[-1] == target_size[1]
    elif len(target_size) == 3:
        assert result.shape[-3] == target_size[0]
        assert result.shape[-2] == target_size[1]
        assert result.shape[-1] == target_size[2]


def test_adjust_size_pad():
    """Test that padding works correctly when input is smaller than target."""
    x = torch.ones(1, 1, 10)
    result = adjust_size_1d(x, 20)

    # First 10 elements should be 1, remaining should be 0 (padded)
    assert torch.all(result[:, :, :10] == 1)
    assert torch.all(result[:, :, 10:] == 0)


def test_adjust_size_crop():
    """Test that cropping works correctly when input is larger than target."""
    x = torch.arange(20).float().reshape(1, 1, 20)
    result = adjust_size_1d(x, 10)

    # Should keep only the first 10 elements
    assert torch.all(result == torch.arange(10).float().reshape(1, 1, 10))


def test_adjust_size_no_change():
    """Test that tensor is unchanged when input size equals target size."""
    x = torch.randn(2, 3, 100)
    result = adjust_size_1d(x, 100)

    assert torch.equal(x, result)
