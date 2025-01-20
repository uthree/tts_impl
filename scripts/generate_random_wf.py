from tts_impl.net.vocoder.ddsp.ultralight import UltraLighweightDdsp
import torchaudio
import torch


dsp = UltraLighweightDdsp()
f0 = torch.full((1, 100), 440.0)
p = torch.ones((1, 12, 100))
v = torch.rand((1, 257, 100))
wf = dsp.forward(f0, p, v)
torchaudio.save("a.wav", wf, 24000)