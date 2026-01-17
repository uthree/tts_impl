import torch
from torch import nn as nn
from torch.nn import functional as F

from tts_impl.net.base.stateful import StatefulModule
from tts_impl.utils.config import derive_config


@torch.jit.script
def complex_parallel_scan(
    log_mag: torch.Tensor,
    angle: torch.Tensor,
    Bx: torch.Tensor,
    h0_r: torch.Tensor,
    h0_i: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Parallel scan for complex SSM using cumsum.

    Computes h_t = A_t * h_{t-1} + (Bx_t, 0) where A_t = exp(log_mag_t) * exp(i * angle_t)

    The solution is: h_t = P_t * (h_0 + cumsum(P^{-1} * b)[t])
    where P_t = A_1 * A_2 * ... * A_t

    Uses numerically stable computation by normalizing with the final cumulative magnitude.

    Args:
        log_mag: (batch, seq, d_state) - log magnitude of A (should be negative for decay)
        angle: (batch, seq, d_state) - angle/phase of A
        Bx: (batch, seq, d_state) - real input (B * x), added to real part only
        h0_r: (batch, 1, d_state) - initial state real part
        h0_i: (batch, 1, d_state) - initial state imaginary part

    Returns:
        h_r: (batch, seq, d_state) - real part of states
        h_i: (batch, seq, d_state) - imaginary part of states
    """
    # P_t = A_1 * A_2 * ... * A_t (cumulative product)
    # log|P_t| = cumsum(log_mag), arg(P_t) = cumsum(angle)
    log_mag_cum = torch.cumsum(log_mag, dim=1)
    angle_cum = torch.cumsum(angle, dim=1)

    # Numerical stability: normalize by the final magnitude to prevent overflow
    # log_mag_cum is typically negative (decay), so -log_mag_cum is positive and can overflow
    # We use the final value as a reference point for normalization
    log_mag_final = log_mag_cum[:, -1:, :]  # (batch, 1, d_state)
    angle_final = angle_cum[:, -1:, :]  # (batch, 1, d_state)

    # Compute normalized magnitudes (relative to final)
    # log_mag_cum - log_mag_final >= 0 (since log_mag <= 0, earlier values are larger)
    # log_mag_final - log_mag_cum <= 0 (safe for exp)
    # Clamp to prevent overflow: exp(88) ≈ 1.6e38 (float32 limit)
    # For very long sequences with strong decay, early inputs have negligible effect anyway
    log_mag_rel = torch.clamp(log_mag_cum - log_mag_final, max=80.0)
    log_mag_inv_rel = log_mag_final - log_mag_cum  # <= 0, safe for exp

    # P_t / P_T = exp(log_mag_rel) * exp(i * (angle_cum - angle_final))
    mag_rel = torch.exp(log_mag_rel)
    angle_rel = angle_cum - angle_final

    # P_T / P_s = exp(log_mag_inv_rel) * exp(i * (angle_final - angle_cum))
    # = exp(log_mag_inv_rel) * exp(-i * (angle_cum - angle_final))
    inv_mag_rel = torch.exp(log_mag_inv_rel)
    inv_angle_rel = angle_final - angle_cum  # = -(angle_cum - angle_final)

    # c_s = (P_T / P_s) * b_s where b_s = (Bx_s, 0) (real input)
    # Complex multiplication: (cos + i*sin) * (Bx + 0i) = (cos*Bx, sin*Bx)
    inv_cos = inv_mag_rel * torch.cos(inv_angle_rel)
    inv_sin = inv_mag_rel * torch.sin(inv_angle_rel)
    c_r = inv_cos * Bx
    c_i = inv_sin * Bx

    # cumsum(c) for the summation term
    c_r_cum = torch.cumsum(c_r, dim=1)
    c_i_cum = torch.cumsum(c_i, dim=1)

    # h_0 contribution scaled by P_T
    # P_T * h_0 = exp(log_mag_final) * exp(i * angle_final) * (h0_r + i*h0_i)
    mag_final = torch.exp(log_mag_final)
    cos_final = mag_final * torch.cos(angle_final)
    sin_final = mag_final * torch.sin(angle_final)

    # P_T * h_0 in complex multiplication
    Ph0_r = cos_final * h0_r - sin_final * h0_i
    Ph0_i = sin_final * h0_r + cos_final * h0_i

    # total = P_T * h_0 + cumsum((P_T / P_s) * b_s)
    total_r = Ph0_r + c_r_cum
    total_i = Ph0_i + c_i_cum

    # h_t = (P_t / P_T) * total
    cos_rel = mag_rel * torch.cos(angle_rel)
    sin_rel = mag_rel * torch.sin(angle_rel)

    h_r = cos_rel * total_r - sin_rel * total_i
    h_i = sin_rel * total_r + cos_rel * total_i

    return h_r, h_i


@torch.jit.script
def complex_step(
    cos_A: torch.Tensor,
    sin_A: torch.Tensor,
    Bx: torch.Tensor,
    h_r: torch.Tensor,
    h_i: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Single step of complex SSM: h_new = A * h_old + (Bx, 0)

    Args:
        cos_A: (batch, 1, d_state) - mag * cos(angle)
        sin_A: (batch, 1, d_state) - mag * sin(angle)
        Bx: (batch, 1, d_state) - real input
        h_r: (batch, 1, d_state) - current state real part
        h_i: (batch, 1, d_state) - current state imaginary part

    Returns:
        new_h_r, new_h_i: updated state
    """
    # Complex rotation + real input
    new_h_r = cos_A * h_r - sin_A * h_i + Bx
    new_h_i = sin_A * h_r + cos_A * h_i
    return new_h_r, new_h_i


@derive_config
class Mamba(StatefulModule):
    """
    Mamba-3 (https://openreview.net/forum?id=HwCvaJOiCj) unofficial implementation.

    Uses complex-valued state space model with real tensor operations for ONNX compatibility.
    The complex state is represented as (real, imag) parts stored separately.
    """

    def __init__(
        self,
        d_model: int,
        dt_rank: int | None = None,
        d_state: int | None = None,
        initial_period: int = 10000,
    ):
        """
        Args:
            d_model: int, input/output dimension
            dt_rank: int | None, rank for delta projection. If None, defaults to d_model // 4
            d_state: int | None, SSM state dimension. If None, defaults to d_model
            initial_period: int, base period for phase initialization (like RoPE)
        """
        super().__init__()
        if d_state is None:
            d_state = d_model
        if dt_rank is None:
            dt_rank = max(d_model // 4, 1)

        self.d_model = d_model
        self.dt_rank = dt_rank
        self.d_state = d_state

        # A parameters in polar form: A = exp(log_mag) * exp(i * phase)
        # log_mag controls decay (initialized to small negative for stability)
        # phase controls rotation frequency
        self.log_mag_A = nn.Parameter(torch.zeros(d_state))

        # Phase initialization: exponentially spaced frequencies (like S4/HiPPO)
        # Higher state indices have lower frequencies
        freqs = 1.0 / (initial_period ** (torch.arange(d_state).float() / d_state))
        self.phase_A = nn.Parameter(freqs)

        # Project input to: B (d_state), C (d_state), delta_inner (dt_rank)
        self.x_proj = nn.Linear(d_model, d_state * 2 + dt_rank)
        self.dt_proj = nn.Linear(dt_rank, d_state)

        # Input/output projections if d_model != d_state
        if d_model != d_state:
            self.in_proj = nn.Linear(d_model, d_state)
            self.out_proj = nn.Linear(d_state, d_model)
        else:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()

        # Initialize dt_proj bias to make delta start positive
        with torch.no_grad():
            self.dt_proj.bias.fill_(1.0)

    def _initial_state(self, x: torch.Tensor) -> torch.Tensor:
        """
        Initialize complex state as zeros.

        Returns:
            h: (batch, 1, d_state * 2) - [real part, imag part] concatenated
        """
        batch = x.shape[0]
        # Store as (batch, 1, d_state * 2) with first half real, second half imag
        return torch.zeros(batch, 1, self.d_state * 2, device=x.device, dtype=x.dtype)

    def _compute_A_components(
        self, delta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute discretized A matrix components.

        A_bar = exp(-exp(log_mag_A) * delta) * exp(i * phase_A * delta)

        Args:
            delta: (batch, seq, d_state) - time step

        Returns:
            log_mag: (batch, seq, d_state) - log magnitude (negative for decay)
            angle: (batch, seq, d_state) - rotation angle
        """
        # Decay factor: exp(-decay_rate * delta) where decay_rate = exp(log_mag_A)
        # For stability, we want the magnitude < 1, so log_mag should be negative
        log_mag = -F.softplus(self.log_mag_A) * delta

        # Rotation angle
        angle = self.phase_A * delta

        return log_mag, angle

    def _sequential_forward(
        self, x: torch.Tensor, h: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sequential forward for single time step.

        Args:
            x: (batch, 1, d_model)
            h: (batch, 1, d_state * 2) - [real, imag] concatenated

        Returns:
            y: (batch, 1, d_model)
            h_new: (batch, 1, d_state * 2)
        """
        # Project input to SSM space
        x_ssm = self.in_proj(x)  # (batch, 1, d_state)

        # Get time-varying parameters
        proj = self.x_proj(x)  # (batch, 1, d_state*2 + dt_rank)
        B, C, delta_inner = proj.split(
            [self.d_state, self.d_state, self.dt_rank], dim=-1
        )

        # Compute delta (time step)
        log_delta = self.dt_proj(delta_inner)  # (batch, 1, d_state)
        delta = F.softplus(log_delta)

        # Compute A components
        log_mag, angle = self._compute_A_components(delta)
        mag = torch.exp(log_mag)
        cos_A = mag * torch.cos(angle)
        sin_A = mag * torch.sin(angle)

        # Split state into real and imaginary parts
        h_r, h_i = h.chunk(2, dim=-1)  # each (batch, 1, d_state)

        # Input: B * x_ssm (real, added to real part only)
        Bx = B * x_ssm

        # State update
        new_h_r, new_h_i = complex_step(cos_A, sin_A, Bx, h_r, h_i)

        # Output: C * h_real (use only real part)
        y = C * new_h_r
        y = self.out_proj(y)

        # Concatenate state
        new_h = torch.cat([new_h_r, new_h_i], dim=-1)

        return y, new_h

    def _parallel_forward(
        self, x: torch.Tensor, h: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parallel forward using cumsum-based scan.

        Args:
            x: (batch, seq, d_model)
            h: (batch, 1, d_state * 2) - [real, imag] concatenated

        Returns:
            y: (batch, seq, d_model)
            h_last: (batch, 1, d_state * 2)
        """
        batch, seq, _ = x.shape

        # Project input to SSM space
        x_ssm = self.in_proj(x)  # (batch, seq, d_state)

        # Get time-varying parameters
        proj = self.x_proj(x)
        B, C, delta_inner = proj.split(
            [self.d_state, self.d_state, self.dt_rank], dim=-1
        )

        # Compute delta
        log_delta = self.dt_proj(delta_inner)
        delta = F.softplus(log_delta)

        # Compute A components
        log_mag, angle = self._compute_A_components(delta)

        # Input contribution
        Bx = B * x_ssm  # (batch, seq, d_state)

        # Split initial state
        h0_r, h0_i = h.chunk(2, dim=-1)  # each (batch, 1, d_state)

        # Parallel scan
        h_r, h_i = complex_parallel_scan(log_mag, angle, Bx, h0_r, h0_i)

        # Output: C * h_real
        y = C * h_r
        y = self.out_proj(y)

        # Last hidden state
        h_last = torch.cat([h_r[:, -1:, :], h_i[:, -1:, :]], dim=-1)

        return y, h_last
