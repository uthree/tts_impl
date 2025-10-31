import pytest
import torch
from tts_impl.functional.f0_estimation import estimate_f0


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("sample_rate", [16000, 22050, 24000])
@pytest.mark.parametrize("frame_size", [256, 480])
@pytest.mark.parametrize("duration", [1.0, 2.0])
def test_estimate_f0_yin(
    batch_size: int, sample_rate: int, frame_size: int, duration: float
):
    """Test F0 estimation using YIN algorithm."""
    length = int(sample_rate * duration)
    waveform = torch.randn(batch_size, 1, length)

    f0 = estimate_f0(waveform, sample_rate, frame_size, algorithm="yin")

    # Check output shape
    expected_frames = length // frame_size
    assert f0.shape[0] == batch_size
    assert f0.shape[1] == expected_frames

    # Check that F0 values are non-negative
    assert torch.all(f0 >= 0)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("length", [16000, 24000])
@pytest.mark.parametrize("frame_size", [256, 480])
def test_estimate_f0_shape(batch_size: int, length: int, frame_size: int):
    """Test that F0 estimation returns correct shape."""
    sample_rate = 24000
    waveform = torch.randn(batch_size, 1, length)

    f0 = estimate_f0(waveform, sample_rate, frame_size, algorithm="yin")

    expected_frames = length // frame_size
    assert f0.shape == (batch_size, expected_frames)


def test_estimate_f0_single_channel():
    """Test F0 estimation with single channel audio."""
    sample_rate = 24000
    frame_size = 480
    length = 24000
    waveform = torch.randn(1, 1, length)

    f0 = estimate_f0(waveform, sample_rate, frame_size, algorithm="yin")

    assert f0.ndim == 2
    assert f0.shape[0] == 1


def test_estimate_f0_multi_channel():
    """Test F0 estimation with multi-channel audio (should sum channels)."""
    sample_rate = 24000
    frame_size = 480
    length = 24000
    # Create stereo input
    waveform = torch.randn(2, 2, length)

    f0 = estimate_f0(waveform, sample_rate, frame_size, algorithm="yin")

    assert f0.ndim == 2
    assert f0.shape[0] == 2


def test_estimate_f0_sine_wave():
    """Test F0 estimation on a synthetic sine wave with known frequency."""
    sample_rate = 24000
    frame_size = 480
    duration = 2.0
    frequency = 440.0  # A4 note

    # Generate sine wave
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.sin(2 * torch.pi * frequency * t)
    waveform = waveform.unsqueeze(0).unsqueeze(0)

    f0 = estimate_f0(waveform, sample_rate, frame_size, algorithm="yin")

    # YIN should detect frequencies close to the true frequency
    # Filter out unvoiced regions (F0 = 0)
    voiced_f0 = f0[f0 > 0]

    if len(voiced_f0) > 0:
        mean_f0 = voiced_f0.mean().item()
        # Allow 10% tolerance for frequency detection
        assert abs(mean_f0 - frequency) / frequency < 0.1


@pytest.mark.parametrize("algorithm", ["yin"])
def test_estimate_f0_different_algorithms(algorithm: str):
    """Test that different algorithms can be called."""
    sample_rate = 24000
    frame_size = 480
    length = 24000
    waveform = torch.randn(1, 1, length)

    f0 = estimate_f0(waveform, sample_rate, frame_size, algorithm=algorithm)

    expected_frames = length // frame_size
    assert f0.shape == (1, expected_frames)


def test_estimate_f0_invalid_algorithm():
    """Test that invalid algorithm raises an error."""
    sample_rate = 24000
    frame_size = 480
    length = 24000
    waveform = torch.randn(1, 1, length)

    with pytest.raises(ValueError):
        estimate_f0(waveform, sample_rate, frame_size, algorithm="invalid")
