import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import melscale_fbanks

from tts_impl.functional.ddsp import impulse_train
