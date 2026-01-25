import pytest
import torch

from tts_impl.net.base.stateful import sanity_check_stateful_module
from tts_impl.net.common.mamba import MambaLayer, complex_parallel_scan, complex_step


class TestComplexParallelScan:
    """Tests for the complex parallel scan function."""

    def test_single_step_matches_complex_step(self):
        """Test that parallel scan with seq=1 matches single step."""
        batch, d_state = 2, 16
        log_mag = torch.randn(batch, 1, d_state) * 0.1
        angle = torch.randn(batch, 1, d_state)
        Bx = torch.randn(batch, 1, d_state)
        h0_r = torch.randn(batch, 1, d_state)
        h0_i = torch.randn(batch, 1, d_state)

        # Parallel scan
        h_r_par, h_i_par = complex_parallel_scan(log_mag, angle, Bx, h0_r, h0_i)

        # Single step
        mag = torch.exp(log_mag)
        cos_A = mag * torch.cos(angle)
        sin_A = mag * torch.sin(angle)
        h_r_seq, h_i_seq = complex_step(cos_A, sin_A, Bx, h0_r, h0_i)

        assert torch.allclose(h_r_par, h_r_seq, atol=1e-5)
        assert torch.allclose(h_i_par, h_i_seq, atol=1e-5)

    def test_output_shape(self):
        """Test that output shape matches input sequence length."""
        batch, seq, d_state = 2, 32, 16
        log_mag = torch.randn(batch, seq, d_state) * 0.1
        angle = torch.randn(batch, seq, d_state)
        Bx = torch.randn(batch, seq, d_state)
        h0_r = torch.randn(batch, 1, d_state)
        h0_i = torch.randn(batch, 1, d_state)

        h_r, h_i = complex_parallel_scan(log_mag, angle, Bx, h0_r, h0_i)

        assert h_r.shape == (batch, seq, d_state)
        assert h_i.shape == (batch, seq, d_state)


class TestMamba:
    """Tests for the Mamba module."""

    @pytest.mark.parametrize("d_model", [32, 64])
    @pytest.mark.parametrize("d_state", [None, 16, 32])
    def test_output_shape(self, d_model: int, d_state: int | None):
        """Test that output shape matches input shape."""
        model = MambaLayer(d_model=d_model, d_state=d_state)
        model.eval()

        batch, seq = 2, 16
        x = torch.randn(batch, seq, d_model)

        y, h = model(x)

        assert y.shape == (batch, seq, d_model)
        expected_state_dim = (d_state if d_state is not None else d_model) * 2
        assert h.shape == (batch, 1, expected_state_dim)

    @pytest.mark.parametrize("d_model", [32, 64])
    @pytest.mark.parametrize("d_state", [None, 32])
    def test_parallel_sequential_consistency(self, d_model: int, d_state: int | None):
        """Test that parallel and sequential forward give same results."""
        model = MambaLayer(d_model=d_model, d_state=d_state)
        model.eval()

        x = torch.randn(2, 16, d_model)

        sanity_check_stateful_module(model, x, atol=1e-4)

    def test_stateful_streaming(self):
        """Test that stateful streaming works correctly."""
        model = MambaLayer(d_model=32, d_state=16)
        model.eval()

        batch, seq = 2, 10

        # Full sequence
        x_full = torch.randn(batch, seq, 32)
        with torch.no_grad():
            y_full, _ = model(x_full)

        # Step by step
        y_steps = []
        h = None
        with torch.no_grad():
            for t in range(seq):
                x_t = x_full[:, t : t + 1, :]
                y_t, h = model(x_t, h)
                y_steps.append(y_t)
        y_streamed = torch.cat(y_steps, dim=1)

        assert torch.allclose(y_full, y_streamed, atol=1e-4)

    def test_state_decay(self):
        """Test that state decays with zero input."""
        model = MambaLayer(d_model=32, d_state=16)
        model.eval()

        # Initialize with non-zero input
        x_init = torch.randn(1, 1, 32) * 10
        with torch.no_grad():
            _, h = model(x_init)
        initial_norm = h.norm().item()

        # Feed zero inputs
        x_zero = torch.zeros(1, 1, 32)
        norms = []
        with torch.no_grad():
            for _ in range(10):
                _, h = model(x_zero, h)
                norms.append(h.norm().item())

        # State should decay
        assert norms[-1] < initial_norm * 0.1

    def test_initial_state_zeros(self):
        """Test that initial state is zeros."""
        model = MambaLayer(d_model=32, d_state=16)
        x = torch.randn(2, 1, 32)

        h0 = model._initial_state(x)

        assert torch.all(h0 == 0)
        assert h0.shape == (2, 1, 32)  # d_state * 2 = 32

    def test_preserves_device(self):
        """Test that output is on the same device as input."""
        model = MambaLayer(d_model=32)
        x = torch.randn(1, 8, 32)

        y, h = model(x)

        assert y.device == x.device
        assert h.device == x.device

    def test_preserves_dtype(self):
        """Test that output has same dtype as input."""
        model = MambaLayer(d_model=32).double()
        x = torch.randn(1, 8, 32, dtype=torch.float64)

        y, h = model(x)

        assert y.dtype == x.dtype
        assert h.dtype == x.dtype

    def test_different_dt_rank(self):
        """Test with different dt_rank values."""
        for dt_rank in [4, 8, 16]:
            model = MambaLayer(d_model=32, dt_rank=dt_rank)
            x = torch.randn(1, 8, 32)

            y, h = model(x)

            assert y.shape == (1, 8, 32)

    def test_custom_initial_period(self):
        """Test with custom initial_period."""
        model1 = MambaLayer(d_model=32, initial_period=1000)
        model2 = MambaLayer(d_model=32, initial_period=100000)

        # Phase frequencies should be different
        assert not torch.allclose(model1.phase_A, model2.phase_A)

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = MambaLayer(d_model=32, d_state=16)
        x = torch.randn(2, 8, 32, requires_grad=True)

        y, h = model(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

        # Check that model parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_batch_independence(self):
        """Test that batches are processed independently."""
        model = MambaLayer(d_model=32, d_state=16)
        model.eval()

        x1 = torch.randn(1, 8, 32)
        x2 = torch.randn(1, 8, 32)
        x_batch = torch.cat([x1, x2], dim=0)

        with torch.no_grad():
            y1, _ = model(x1)
            y2, _ = model(x2)
            y_batch, _ = model(x_batch)

        assert torch.allclose(y_batch[0], y1[0], atol=1e-5)
        assert torch.allclose(y_batch[1], y2[0], atol=1e-5)

    def test_long_sequence(self):
        """Test with longer sequences."""
        model = MambaLayer(d_model=32, d_state=16)
        model.eval()

        x = torch.randn(1, 256, 32)

        with torch.no_grad():
            y, h = model(x)

        assert y.shape == (1, 256, 32)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_numerical_stability(self):
        """Test numerical stability with extreme inputs."""
        model = MambaLayer(d_model=32, d_state=16)
        model.eval()

        # Test with large inputs
        x_large = torch.randn(1, 16, 32) * 100
        with torch.no_grad():
            y, h = model(x_large)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

        # Test with small inputs
        x_small = torch.randn(1, 16, 32) * 1e-6
        with torch.no_grad():
            y, h = model(x_small)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()
