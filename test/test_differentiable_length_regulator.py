import pytest
import torch

from tts_impl.net.tts.length_regurator import DifferentiableLengthRegulator


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("channels", [64, 192])
@pytest.mark.parametrize("text_length", [10, 50])
@pytest.mark.parametrize("sigma_scale", [0.5, 1.0, 2.0])
def test_differentiable_length_regulator_basic(
    batch_size, channels, text_length, sigma_scale
):
    """Test basic functionality without masks."""
    lr = DifferentiableLengthRegulator(sigma_scale=sigma_scale)
    x = torch.randn(batch_size, channels, text_length)
    w = torch.randint(1, 10, (batch_size, text_length)).float()

    out = lr.forward(x, w)

    expected_length = int(w.sum(dim=1).max().item())
    assert out.shape[0] == batch_size
    assert out.shape[1] == channels
    assert out.shape[2] == expected_length


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [64])
@pytest.mark.parametrize("text_length", [20])
def test_differentiable_length_regulator_with_x_mask(batch_size, channels, text_length):
    """Test with input mask (x_mask)."""
    lr = DifferentiableLengthRegulator()
    x = torch.randn(batch_size, channels, text_length)
    w = torch.randint(1, 10, (batch_size, text_length)).float()
    x_mask = torch.ones(batch_size, 1, text_length)
    # Mask out the last few positions
    x_mask[:, :, -5:] = 0

    out = lr.forward(x, w, x_mask=x_mask)

    expected_length = int(w.sum(dim=1).max().item())
    assert out.shape[0] == batch_size
    assert out.shape[1] == channels
    assert out.shape[2] == expected_length


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [64])
@pytest.mark.parametrize("text_length", [20])
@pytest.mark.parametrize("output_length", [100, 200])
def test_differentiable_length_regulator_with_y_mask(
    batch_size, channels, text_length, output_length
):
    """Test with output mask (y_mask)."""
    lr = DifferentiableLengthRegulator()
    x = torch.randn(batch_size, channels, text_length)
    w = torch.randint(1, 10, (batch_size, text_length)).float()
    y_mask = torch.ones(batch_size, 1, output_length)
    # Mask out some positions
    y_mask[:, :, -20:] = 0

    out = lr.forward(x, w, y_mask=y_mask)

    # Output length should match the mask length
    assert out.shape[0] == batch_size
    assert out.shape[1] == channels
    assert out.shape[2] == output_length
    # Check that masked positions are zero
    assert torch.allclose(out[:, :, -20:], torch.zeros_like(out[:, :, -20:]))


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [64])
@pytest.mark.parametrize("text_length", [20])
@pytest.mark.parametrize("output_length", [150])
def test_differentiable_length_regulator_with_both_masks(
    batch_size, channels, text_length, output_length
):
    """Test with both input and output masks."""
    lr = DifferentiableLengthRegulator()
    x = torch.randn(batch_size, channels, text_length)
    w = torch.randint(1, 10, (batch_size, text_length)).float()
    x_mask = torch.ones(batch_size, 1, text_length)
    y_mask = torch.ones(batch_size, 1, output_length)
    x_mask[:, :, -3:] = 0
    y_mask[:, :, -10:] = 0

    out = lr.forward(x, w, x_mask=x_mask, y_mask=y_mask)

    assert out.shape[0] == batch_size
    assert out.shape[1] == channels
    assert out.shape[2] == output_length


def test_differentiable_length_regulator_gradient():
    """Test that gradients can be computed through the module."""
    lr = DifferentiableLengthRegulator()
    x = torch.randn(2, 64, 20, requires_grad=True)
    w = torch.randint(1, 10, (2, 20)).float()
    w.requires_grad = True

    out = lr.forward(x, w)
    loss = out.sum()
    loss.backward()

    # Check that gradients are computed
    assert x.grad is not None
    assert w.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(w.grad).any()


def test_differentiable_length_regulator_sigma_scale_learnable():
    """Test that sigma_scale is a learnable parameter."""
    lr = DifferentiableLengthRegulator(sigma_scale=1.0)

    # Check that sigma_scale is a parameter
    assert hasattr(lr, "sigma_scale")
    assert isinstance(lr.sigma_scale, torch.nn.Parameter)

    # Check that it can be updated via gradient descent
    x = torch.randn(2, 64, 20)
    w = torch.randint(1, 5, (2, 20)).float()

    optimizer = torch.optim.SGD(lr.parameters(), lr=0.1)
    initial_sigma = lr.sigma_scale.clone()

    out = lr.forward(x, w)
    loss = out.sum()
    loss.backward()
    optimizer.step()

    # sigma_scale should have been updated
    assert not torch.allclose(lr.sigma_scale, initial_sigma)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_differentiable_length_regulator_consistent_output(batch_size):
    """Test that the output is consistent for the same input."""
    lr = DifferentiableLengthRegulator(sigma_scale=1.0)
    lr.eval()  # Set to eval mode for deterministic behavior

    x = torch.randn(batch_size, 64, 20)
    w = torch.randint(1, 10, (batch_size, 20)).float()

    out1 = lr.forward(x, w)
    out2 = lr.forward(x, w)

    assert torch.allclose(out1, out2)


def test_differentiable_length_regulator_zero_duration():
    """Test behavior with zero duration (edge case)."""
    lr = DifferentiableLengthRegulator()
    x = torch.randn(1, 64, 5)
    # All durations are 1 (minimum)
    w = torch.ones(1, 5)

    out = lr.forward(x, w)

    # Output length should be 5 (sum of durations)
    assert out.shape[2] == 5


def test_differentiable_length_regulator_variable_duration_per_batch():
    """Test with different total durations per batch item."""
    lr = DifferentiableLengthRegulator()
    x = torch.randn(2, 64, 10)
    w = torch.ones(2, 10)
    # First batch item has longer duration
    w[0, :] = 5.0
    # Second batch item has shorter duration
    w[1, :] = 2.0

    out = lr.forward(x, w)

    # Output length should match the maximum total duration
    expected_length = int(w.sum(dim=1).max().item())
    assert out.shape[2] == expected_length
