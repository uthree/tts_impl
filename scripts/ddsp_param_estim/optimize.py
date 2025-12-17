import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from torch.export.exported_program import PassType
from torchaudio.functional import resample
from tqdm import tqdm

from tts_impl.functional.ddsp import (
    estimate_minimum_phase,
    impulse_train,
    spectral_envelope_filter,
)
from tts_impl.functional.pad import adjust_size
from tts_impl.transforms import LogMelSpectrogram


class VoiceSpeaker(nn.Module):
    def __init__(
        self,
        n_frames=1000,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        initial_f0=100.0,
        post_filter_size: int = 8192,
    ) -> None:
        super().__init__()
        self.n_frames = n_frames
        self.fft_bin = n_fft // 2 + 1
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.log_f0 = nn.Parameter(torch.ones(1, n_frames) * math.log(initial_f0))
        self.spec_env_noi = nn.Parameter(torch.randn(1, self.fft_bin, n_frames) - 6.0)
        self.spec_env_per = nn.Parameter(torch.randn(1, self.fft_bin, n_frames) - 6.0)

    def forward(self) -> Tensor:
        device = self.log_f0.device
        noise = torch.randn(1, self.hop_length * self.n_frames, device=device)
        pulse = impulse_train(
            torch.exp(self.log_f0),
            hop_length=self.hop_length,
            sample_rate=self.sample_rate,
        )
        noi = spectral_envelope_filter(
            noise, self.spec_env_noi.exp(), n_fft=self.n_fft, hop_length=self.hop_length
        )
        per = spectral_envelope_filter(
            pulse,
            estimate_minimum_phase(self.spec_env_per.exp()),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        return noi + per


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("loading file...")
    wf, sr = torchaudio.load("scripts/ddsp_param_estim/target.wav")
    target = resample(wf, sr, 24000).to(device)
    target = adjust_size(target, 256 * 1000)
    print("initializing model...")
    model = VoiceSpeaker(n_frames=1000).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epoch = 50000
    logmel = LogMelSpectrogram(
        sample_rate=24000,
        n_fft=1024,
        hop_length=256,
    ).to(device)
    print("optimizing...")
    bar = tqdm(total=num_epoch)
    for epoch in range(num_epoch):
        optimizer.zero_grad()
        output = model.forward()
        loss = (logmel(output) - logmel(target)).abs().mean()
        loss.backward()
        optimizer.step()
        bar.set_description(f"Epoch #{epoch}, loss: {loss.item():.4f}")
        bar.update(1)
    output = model.forward()
    torchaudio.save("scripts/ddsp_param_estim/output.wav", output.detach().cpu(), 24000)


if __name__ == "__main__":
    main()
