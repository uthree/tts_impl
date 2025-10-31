import pytest
import torch

from tts_impl.transforms.log_mel_spectrogram import LogMelSpectrogram
from tts_impl.transforms.pitch_estimation import PitchEstimation


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("sample_rate", [16000, 22050, 24000])
@pytest.mark.parametrize("n_fft", [512, 1024, 2048])
@pytest.mark.parametrize("hop_length", [128, 256, 512])
@pytest.mark.parametrize("n_mels", [40, 80, 128])
def test_log_mel_spectrogram_shape(
    batch_size: int,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int
):
    """Test LogMelSpectrogram output shape."""
    duration = 2.0  # seconds
    length = int(sample_rate * duration)

    mel_spec = LogMelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    waveform = torch.randn(batch_size, length)
    output = mel_spec(waveform)

    # Check output shape
    # Output frames depend on padding and hop_length
    # Approximate: length // hop_length
    expected_time_frames = length // hop_length
    assert output.shape[0] == batch_size
    assert output.shape[1] == n_mels
    # Allow some tolerance due to padding
    assert abs(output.shape[2] - expected_time_frames) <= 10


def test_log_mel_spectrogram_values():
    """Test that LogMelSpectrogram produces valid values."""
    sample_rate = 22050
    n_fft = 1024
    hop_length = 256
    n_mels = 80

    mel_spec = LogMelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    waveform = torch.randn(1, 22050)  # 1 second
    output = mel_spec(waveform)

    # Log mel spectrogram should produce finite values
    assert torch.all(torch.isfinite(output))

    # Values should be in a reasonable range (log of magnitude)
    # Since we're using log, values can be negative
    assert output.max() < 100  # Reasonable upper bound
    assert output.min() > -100  # Reasonable lower bound


def test_log_mel_spectrogram_fmin_fmax():
    """Test LogMelSpectrogram with custom fmin and fmax."""
    sample_rate = 22050
    mel_spec = LogMelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        fmin=80.0,
        fmax=8000.0,
    )

    waveform = torch.randn(1, 22050)
    output = mel_spec(waveform)

    assert output.ndim == 3
    assert output.shape[1] == 80


def test_log_mel_spectrogram_safe_log():
    """Test safe_log method with zero and negative values."""
    mel_spec = LogMelSpectrogram(eps=1e-8)

    # Test with zeros (should be clamped to eps before log)
    x = torch.zeros(1, 80, 100)
    result = mel_spec.safe_log(x)
    assert torch.all(torch.isfinite(result))
    assert torch.all(result == torch.log(torch.tensor(1e-8)))

    # Test with positive values
    x = torch.ones(1, 80, 100)
    result = mel_spec.safe_log(x)
    assert torch.all(torch.isfinite(result))
    assert torch.allclose(result, torch.log(torch.ones_like(result)))


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("sample_rate", [16000, 22050, 24000])
@pytest.mark.parametrize("frame_size", [256, 480])
@pytest.mark.parametrize("algorithm", ["yin"])
def test_pitch_estimation_shape(
    batch_size: int,
    sample_rate: int,
    frame_size: int,
    algorithm: str
):
    """Test PitchEstimation output shape."""
    duration = 2.0
    length = int(sample_rate * duration)

    pitch_estimator = PitchEstimation(
        sample_rate=sample_rate,
        frame_size=frame_size,
        algorithm=algorithm,
    )

    waveform = torch.randn(batch_size, 1, length)
    pitch = pitch_estimator(waveform)

    expected_frames = length // frame_size
    assert pitch.shape[0] == batch_size
    assert pitch.shape[1] == expected_frames


def test_pitch_estimation_values():
    """Test that PitchEstimation produces valid pitch values."""
    sample_rate = 24000
    frame_size = 480

    pitch_estimator = PitchEstimation(
        sample_rate=sample_rate,
        frame_size=frame_size,
        algorithm="yin",
    )

    # Generate a sine wave with known frequency
    frequency = 440.0  # A4 note
    duration = 1.0
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.sin(2 * torch.pi * frequency * t)
    waveform = waveform.unsqueeze(0).unsqueeze(0)

    pitch = pitch_estimator(waveform)

    # Check that output is non-negative
    assert torch.all(pitch >= 0)

    # Check that detected pitch is close to actual frequency
    voiced_pitch = pitch[pitch > 0]
    if len(voiced_pitch) > 0:
        mean_pitch = voiced_pitch.mean().item()
        # Allow 10% tolerance
        assert abs(mean_pitch - frequency) / frequency < 0.1


def test_pitch_estimation_multi_channel():
    """Test PitchEstimation with multi-channel input."""
    sample_rate = 24000
    frame_size = 480
    length = 24000

    pitch_estimator = PitchEstimation(
        sample_rate=sample_rate,
        frame_size=frame_size,
        algorithm="yin",
    )

    # Stereo input
    waveform = torch.randn(2, 2, length)
    pitch = pitch_estimator(waveform)

    expected_frames = length // frame_size
    assert pitch.shape == (2, expected_frames)


def test_pitch_estimation_config():
    """Test PitchEstimation configuration."""
    sample_rate = 22050
    frame_size = 256
    algorithm = "yin"

    pitch_estimator = PitchEstimation(
        sample_rate=sample_rate,
        frame_size=frame_size,
        algorithm=algorithm,
    )

    assert pitch_estimator.sample_rate == sample_rate
    assert pitch_estimator.frame_size == frame_size
    assert pitch_estimator.algorithm == algorithm


@pytest.mark.parametrize("n_mels", [40, 80, 128])
def test_mel_spectrogram_different_n_mels(n_mels: int):
    """Test LogMelSpectrogram with different numbers of mel bands."""
    sample_rate = 22050
    mel_spec = LogMelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=n_mels,
    )

    waveform = torch.randn(1, 22050)
    output = mel_spec(waveform)

    assert output.shape[1] == n_mels


def test_mel_spectrogram_padding():
    """Test that LogMelSpectrogram applies padding correctly."""
    sample_rate = 22050
    n_fft = 1024
    hop_length = 256

    mel_spec = LogMelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=80,
    )

    # Very short signal
    waveform = torch.randn(1, 512)
    output = mel_spec(waveform)

    # Should still produce output due to padding
    assert output.shape[0] == 1
    assert output.shape[1] == 80
    assert output.shape[2] > 0
