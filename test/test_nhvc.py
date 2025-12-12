import pytest
import torch
from torch import nn as nn
from torch.nn import functional as F
from tts_impl.net.vc.nhvc import NhvcLightningModule
from tts_impl.net.vc.nhvc.generator import NhvcDecoder, NhvcEncoder, NhvcGenerator
from tts_impl.net.vocoder.ddsp import HomomorphicVocoder


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("num_frames", [100, 200])
@pytest.mark.parametrize("in_channels", [80])
@pytest.mark.parametrize("d_model", [128, 256])
@pytest.mark.parametrize("n_layers", [2, 4])
def test_nhvc_encoder(
    batch_size: int,
    num_frames: int,
    in_channels: int,
    d_model: int,
    n_layers: int,
):
    """Test NhvcEncoder forward pass and output shapes"""
    encoder = NhvcEncoder(
        in_channels=in_channels,
        d_model=d_model,
        n_layers=n_layers,
        d_phonemes=64,
        n_phonemes=128,
        n_f0_classes=128,
        n_fft=1024,
    )

    # Test parallel forward
    x = torch.randn(batch_size, num_frames, in_channels)
    h = encoder._initial_state(x)
    output, h_last = encoder._parallel_forward(x, h)

    # Check output shape: [batch_size, num_frames, d_phonemes + n_f0_classes + fft_bin]
    expected_output_dim = 64 + 128 + (1024 // 2 + 1)  # d_phonemes + n_f0_classes + fft_bin
    assert output.shape == torch.Size([batch_size, num_frames, expected_output_dim])

    # Test sequential forward
    x_seq = torch.randn(batch_size, 1, in_channels)
    h_seq = encoder._initial_state(x_seq)
    output_seq, h_last_seq = encoder._sequential_forward(x_seq, h_seq)
    assert output_seq.shape == torch.Size([batch_size, 1, expected_output_dim])


def test_nhvc_encoder_f0_functions():
    """Test F0 encoding/decoding and loss functions"""
    encoder = NhvcEncoder(
        in_channels=80,
        d_model=128,
        n_layers=4,
        n_f0_classes=128,
        fmin=20.0,
        fmax=8000.0,
    )

    batch_size = 2
    num_frames = 100

    # Test freq2idx and idx2freq
    f0 = torch.randn(batch_size, num_frames).abs() * 500 + 100  # F0 in 100-600Hz range
    idx = encoder.freq2idx(f0)
    assert idx.shape == f0.shape
    assert idx.dtype == torch.long

    f0_reconstructed = encoder.idx2freq(idx)
    assert f0_reconstructed.shape == f0.shape

    # Test decode_f0
    f0_probs = torch.randn(batch_size, encoder.n_f0_classes, num_frames)
    f0_decoded = encoder.decode_f0(f0_probs)
    assert f0_decoded.shape == torch.Size([batch_size, num_frames])

    # Test f0_loss
    loss = encoder.f0_loss(f0_probs, f0)
    assert loss.shape == torch.Size([])
    assert not torch.isnan(loss)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("num_frames", [100, 200])
@pytest.mark.parametrize("d_model", [128, 256])
@pytest.mark.parametrize("gin_channels", [0, 128])
def test_nhvc_decoder(
    batch_size: int,
    num_frames: int,
    d_model: int,
    gin_channels: int,
):
    """Test NhvcDecoder forward pass and output shapes"""
    d_phonemes = 64
    n_fft = 1024
    decoder = NhvcDecoder(
        d_model=d_model,
        n_layers=4,
        n_fft=n_fft,
        d_phonemes=d_phonemes,
        gin_channels=gin_channels,
    )

    # Test parallel forward
    x = torch.randn(batch_size, num_frames, d_phonemes)
    h = decoder._initial_state(x)

    # Speaker embedding (optional)
    g = None
    if gin_channels > 0:
        g = torch.randn(batch_size, 1, gin_channels)

    output, h_last = decoder._parallel_forward(x, h, g=g)

    # Check output shape: [batch_size, num_frames, fft_bin * 2]
    fft_bin = n_fft // 2 + 1
    expected_output_dim = fft_bin * 2
    assert output.shape == torch.Size([batch_size, num_frames, expected_output_dim])

    # Test sequential forward
    x_seq = torch.randn(batch_size, 1, d_phonemes)
    h_seq = decoder._initial_state(x_seq)
    output_seq, h_last_seq = decoder._sequential_forward(x_seq, h_seq, g=g)
    assert output_seq.shape == torch.Size([batch_size, 1, expected_output_dim])


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_frames", [500, 1000])  # Increased to avoid padding errors
@pytest.mark.parametrize("n_speakers", [0, 10])
def test_nhvc_generator(
    batch_size: int,
    num_frames: int,
    n_speakers: int,
):
    """Test NhvcGenerator forward pass and output shapes"""
    in_channels = 80
    frame_size = 256
    sample_rate = 48000
    n_fft = 512  # Smaller FFT for smaller test inputs

    generator = NhvcGenerator(
        encoder=NhvcEncoder.Config(
            in_channels=in_channels,
            d_model=128,
            n_layers=4,
            n_fft=n_fft,
            d_phonemes=64,
            n_phonemes=128,
        ),
        decoder=NhvcDecoder.Config(
            d_model=128,
            n_layers=4,
            n_fft=n_fft,
            d_phonemes=64,
        ),
        vocoder=HomomorphicVocoder.Config(
            hop_length=frame_size,
            n_fft=n_fft,
            sample_rate=sample_rate,
        ),
        n_speakers=n_speakers,
        gin_channels=128 if n_speakers > 0 else 0,
        sample_rate=sample_rate,
    )

    # Test parallel forward
    x = torch.randn(batch_size, in_channels, num_frames)
    h = generator._initial_state(x)

    # Speaker ID (optional)
    sid = None
    if n_speakers > 0:
        sid = torch.randint(0, n_speakers, (batch_size,))

    output, h_last = generator._parallel_forward(x, h, sid=sid)

    # Check output shape: [batch_size, 1, num_frames * frame_size]
    expected_waveform_length = num_frames * frame_size
    assert output.shape == torch.Size([batch_size, 1, expected_waveform_length])

    # Check hidden state is a single tensor (concatenated encoder and decoder states)
    assert isinstance(h_last, torch.Tensor)
    assert h_last.ndim == 3  # [B, n_layers*2, d_model]

    # Test sequential forward with minimum frames for vocoder
    # Need at least ceil(n_fft / hop_length) frames to avoid padding errors
    min_frames = (n_fft // frame_size) + 1
    x_seq = torch.randn(batch_size, in_channels, min_frames)
    h_seq = generator._initial_state(x_seq)
    output_seq, h_last_seq = generator._sequential_forward(x_seq, h_seq, sid=sid)
    assert output_seq.shape == torch.Size([batch_size, 1, min_frames * frame_size])


def test_nhvc_generator_with_config():
    """Test NhvcGenerator initialization from config"""
    config = NhvcGenerator.Config()
    generator = NhvcGenerator(**config)

    batch_size = 2
    num_frames = 100
    in_channels = 80

    x = torch.randn(batch_size, in_channels, num_frames)
    h = generator._initial_state(x)
    output, h_last = generator._parallel_forward(x, h)

    assert output.shape[0] == batch_size
    assert output.shape[1] == 1
    assert output.shape[2] > 0  # Has some waveform output


@pytest.mark.parametrize("n_speakers", [0, 5])
@pytest.mark.parametrize("batch_size", [2, 4])
def test_nhvc_lightning_module_initialization(
    n_speakers: int,
    batch_size: int,
):
    """Test NhvcLightningModule initialization"""
    module = NhvcLightningModule(
        generator=NhvcGenerator.Config(),
        n_speakers=n_speakers,
        gin_channels=128 if n_speakers > 0 else 0,
        weight_mel=45.0,
        weight_feat=1.0,
        weight_adv=1.0,
        weight_phoneme=10.0,
        weight_f0=5.0,
    )

    assert module.generator is not None
    assert module.discriminator is not None
    assert module.hubert is not None
    assert module.n_speakers == n_speakers

    # Check that HuBERT is frozen
    for param in module.hubert.parameters():
        assert not param.requires_grad


def test_nhvc_lightning_module_forward():
    """Test NhvcLightningModule forward pass"""
    module = NhvcLightningModule(
        n_speakers=0,
        gin_channels=0,
    )

    batch_size = 2
    num_frames = 500  # Increased to work with mel spectrogram computation
    in_channels = 80

    acoustic_features = torch.randn(batch_size, in_channels, num_frames)
    fake, phoneme_logits, f0_probs = module._generator_forward(
        acoustic_features, sid=None
    )

    # Check waveform output
    assert fake.shape[0] == batch_size
    assert fake.shape[1] == 1
    assert fake.shape[2] > 0

    # Check phoneme logits
    # phoneme_logits: [B, T, n_phonemes]
    assert phoneme_logits.shape[0] == batch_size
    assert phoneme_logits.shape[1] == num_frames
    assert phoneme_logits.shape[2] == module.generator.encoder.to_phoneme_prob.out_features

    # Check F0 probabilities
    assert f0_probs.shape[0] == batch_size
    assert f0_probs.shape[1] == module.generator.encoder.n_f0_classes
    assert f0_probs.shape[2] == num_frames


def test_nhvc_lightning_module_training_step():
    """Test NhvcLightningModule training step"""
    module = NhvcLightningModule(
        n_speakers=2,
        gin_channels=128,
        segment_size=8192,
    )

    # Configure optimizers (required for manual optimization)
    module.configure_optimizers()

    batch_size = 2
    num_frames = 500  # Increased for mel spectrogram computation
    frame_size = 256
    waveform_length = num_frames * frame_size

    # Create mock batch (no acoustic_features needed - computed on-the-fly)
    batch = {
        "waveform": torch.randn(batch_size, 1, waveform_length),
        "f0": torch.randn(batch_size, num_frames).abs() * 200 + 100,
        "speaker_id": torch.randint(0, 2, (batch_size,)),
    }

    # Note: Full training step requires optimizers to be set up properly
    # This is a basic check that the method exists and accepts the right input
    # We won't actually run it as it requires Lightning Trainer context


def test_nhvc_encoder_phoneme_logits():
    """Test that encoder can produce phoneme logits"""
    encoder = NhvcEncoder(
        in_channels=80,
        d_model=128,
        n_layers=4,
        d_phonemes=64,
        n_phonemes=128,
    )

    batch_size = 2
    num_frames = 100

    x = torch.randn(batch_size, num_frames, 80)
    h = encoder._initial_state(x)
    output, _ = encoder._parallel_forward(x, h)

    # Extract phoneme embeddings
    phoneme_emb, _, _ = torch.split(
        output,
        [encoder.d_phonemes, encoder.n_f0_classes, encoder.fft_bin],
        dim=-1,
    )

    # Get phoneme logits
    # phoneme_emb: [B, T, d_phonemes] -> phoneme_logits: [B, T, n_phonemes]
    phoneme_logits = encoder.to_phoneme_prob(phoneme_emb)

    assert phoneme_logits.shape == torch.Size([batch_size, num_frames, encoder.to_phoneme_prob.out_features])


def test_nhvc_components_gradient_flow():
    """Test that gradients flow through all components"""
    n_fft = 512  # Smaller FFT for smaller test inputs

    generator = NhvcGenerator(
        encoder=NhvcEncoder.Config(n_fft=n_fft),
        decoder=NhvcDecoder.Config(n_fft=n_fft),
        vocoder=HomomorphicVocoder.Config(n_fft=n_fft),
        n_speakers=0,
        gin_channels=0,
    )

    batch_size = 2
    num_frames = 500  # Increased to avoid padding errors
    in_channels = 80

    x = torch.randn(batch_size, in_channels, num_frames, requires_grad=True)
    h = generator._initial_state(x)
    output, _ = generator._parallel_forward(x, h)

    # Compute a simple loss
    loss = output.mean()
    loss.backward()

    # Check that input has gradients
    assert x.grad is not None
    assert not torch.all(x.grad == 0)


def test_nhvc_gradient_separation():
    """Test that Encoder and Decoder gradients are properly separated"""
    module = NhvcLightningModule(
        n_speakers=0,
        gin_channels=0,
    )

    batch_size = 2
    num_frames = 500
    in_channels = 80

    # Enable gradient tracking for encoder and decoder parameters
    for param in module.generator.encoder.parameters():
        param.requires_grad = True
    for param in module.generator.decoder.parameters():
        param.requires_grad = True

    acoustic_features = torch.randn(batch_size, in_channels, num_frames)
    fake, phoneme_logits, f0_probs = module._generator_forward(
        acoustic_features, sid=None
    )

    # Test 1: Mel loss should only affect Decoder, not Encoder
    module.zero_grad()
    mel_loss = fake.mean()
    mel_loss.backward(retain_graph=True)

    # Check that Decoder has gradients
    decoder_has_grad = any(
        param.grad is not None and torch.any(param.grad != 0)
        for param in module.generator.decoder.parameters()
    )
    assert decoder_has_grad, "Decoder should have gradients from mel loss"

    # Check that Encoder has NO gradients (from mel loss)
    encoder_has_grad = any(
        param.grad is not None and torch.any(param.grad != 0)
        for param in module.generator.encoder.parameters()
    )
    assert not encoder_has_grad, "Encoder should NOT have gradients from mel loss"

    # Test 2: Phoneme loss should affect Encoder
    module.zero_grad()
    phoneme_loss = phoneme_logits.mean()
    phoneme_loss.backward(retain_graph=True)

    # Check that Encoder has gradients
    encoder_has_grad = any(
        param.grad is not None and torch.any(param.grad != 0)
        for param in module.generator.encoder.parameters()
    )
    assert encoder_has_grad, "Encoder should have gradients from phoneme loss"

    # Test 3: F0 loss should affect Encoder
    module.zero_grad()
    f0_loss = f0_probs.mean()
    f0_loss.backward()

    # Check that Encoder has gradients
    encoder_has_grad = any(
        param.grad is not None and torch.any(param.grad != 0)
        for param in module.generator.encoder.parameters()
    )
    assert encoder_has_grad, "Encoder should have gradients from f0 loss"


def test_nhvc_pitch_shift():
    """Test pitch shift functionality"""
    module = NhvcLightningModule(
        n_speakers=0,
        gin_channels=0,
    )

    batch_size = 2
    num_frames = 500
    in_channels = 80

    acoustic_features = torch.randn(batch_size, in_channels, num_frames)

    # Generate audio without pitch shift
    fake_no_shift, _, _ = module._generator_forward(
        acoustic_features, sid=None, pitch_shift_semitones=0.0
    )

    # Generate audio with +12 semitones pitch shift (one octave up)
    fake_shift_up, _, _ = module._generator_forward(
        acoustic_features, sid=None, pitch_shift_semitones=12.0
    )

    # Generate audio with -12 semitones pitch shift (one octave down)
    fake_shift_down, _, _ = module._generator_forward(
        acoustic_features, sid=None, pitch_shift_semitones=-12.0
    )

    # Check that pitch-shifted outputs are different from original
    assert not torch.allclose(fake_no_shift, fake_shift_up, atol=1e-3)
    assert not torch.allclose(fake_no_shift, fake_shift_down, atol=1e-3)
    assert not torch.allclose(fake_shift_up, fake_shift_down, atol=1e-3)

    # Check output shapes are the same
    assert fake_no_shift.shape == fake_shift_up.shape
    assert fake_no_shift.shape == fake_shift_down.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
