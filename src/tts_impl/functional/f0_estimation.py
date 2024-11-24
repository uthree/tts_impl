import warnings
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pyworld as pw

    PYWORLD_AVAILABLE = True
except ModuleNotFoundError:
    PYWORLD_AVAILABLE = False

try:
    from torchfcpe import spawn_bundled_infer_model

    TORCHFCPE_AVAILABLE = True
except ModuleNotFoundError:
    TORCHFCPE_AVAILABLE = False


def estimate_f0_dio(
    wf,
    sample_rate: int,
    frame_size: int = 480,
    f0_min: float = 20.0,
    f0_max: float = 20000.0,
):
    if not PYWORLD_AVAILABLE:
        raise "pyworld is not installed in this python environment. install pyworld if you need use this F0 estimation method."

    with torch.no_grad():
        if wf.ndim == 1:
            device = wf.device
            signal = wf.detach().cpu().numpy()
            signal = signal.astype(np.double)
            _f0, t = pw.dio(signal, sample_rate, f0_floor=f0_min, f0_ceil=f0_max)
            f0 = pw.stonemask(signal, _f0, t, sample_rate)
            f0 = torch.from_numpy(f0).to(torch.float)
            f0 = f0.to(device)
            f0 = f0.unsqueeze(0).unsqueeze(0)
            f0 = F.interpolate(f0, wf.shape[0] // frame_size, mode="linear")
            f0 = f0.squeeze(0).detach()
            return f0
        elif wf.ndim == 2:
            waves = wf.split(1, dim=0)
            pitchs = [
                estimate_f0_dio(wave[0], sample_rate, frame_size) for wave in waves
            ]
            pitchs = torch.stack(pitchs, dim=0)
            return pitchs.detach()


def estimate_f0_harvest(
    wf,
    sample_rate: int,
    frame_size: int = 480,
    f0_min: float = 20.0,
    f0_max: float = 20000.0,
):
    if not PYWORLD_AVAILABLE:
        raise "pyworld is not installed in this python environment. install pyworld if you need use this F0 estimation method."

    with torch.no_grad():
        if wf.ndim == 1:
            device = wf.device
            signal = wf.detach().cpu().numpy()
            signal = signal.astype(np.double)
            f0, t = pw.harvest(signal, sample_rate, f0_floor=f0_min, f0_ceil=f0_max)
            f0 = torch.from_numpy(f0).to(torch.float)
            f0 = f0.to(device)
            f0 = f0.unsqueeze(0).unsqueeze(0)
            f0 = F.interpolate(f0, wf.shape[0] // frame_size, mode="linear")
            f0 = f0.squeeze(0)
            return f0.detach()
        elif wf.ndim == 2:
            waves = wf.split(1, dim=0)
            pitchs = [
                estimate_f0_harvest(wave[0], sample_rate, frame_size) for wave in waves
            ]
            pitchs = torch.stack(pitchs, dim=0)
            return pitchs.detach()


global torchfcpe_model
torchfcpe_model = {}


def estimate_f0_fcpe(wf, sample_rate=24000, frame_size=480, f0_min=20, f0_max=20000):
    if not TORCHFCPE_AVAILABLE:
        raise "torchfcpe is not installed in this python environment. install torchfcpe if you need use this F0 estimation method."

    with torch.no_grad():
        if wf.device not in torchfcpe_model:
            warnings.warn(
                "When you estimate f0 with torchfcpe, the model will remain in memory. To unload it, use unload_torchfcpe()."
            )
            torchfcpe_model[wf.device] = spawn_bundled_infer_model(wf.device)
        f0 = torchfcpe_model[wf.device].infer(wf.unsqueeze(2), sample_rate)
        f0 = f0.transpose(1, 2)
    return f0.detach()


def unload_torchfcpe(device: Optional[torch.device]):
    if device is not None:
        del torchfcpe_model[device]
    else:
        torchfcpe_model = {}


def estimate_f0(
    waveform,
    sample_rate: int,
    frame_size: int,
    algorithm: Literal["harvest", "dio", "fcpe"] = "harvest",
) -> torch.Tensor:
    """
    Estimate fundamental frequency (F0).

    Args:
        waveform: Tensor, shape=(N, C, L).
        sample_rate: int, sample rate of waveform.
        frame_size: int, length of one frame.
        algorithm: str, algorithm type

    Returns:
        pitch_envelope: Tensor, shape=(N, L // frame_size)
    """
    l = waveform.shape[2]
    waveform = waveform.sum(dim=1)

    if algorithm == "harvest":
        f0 = estimate_f0_harvest(waveform, sample_rate)
    elif algorithm == "dio":
        f0 = estimate_f0_dio(waveform, sample_rate)
    elif algorithm == "fcpe":
        f0 = estimate_f0_fcpe(waveform, sample_rate)
    else:
        raise ValueError("invalid algorithm")

    f0_resized = F.interpolate(f0, l // frame_size, mode="linear").squeeze(1).detach()
    return f0_resized
