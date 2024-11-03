import torch
import torch.nn as nn
import torch.nn.functional as F

from tts_impl.net.tts.vits.lightning import VitsGenerator


def test_vits_generator():
    G = VitsGenerator(256, 513, 32)