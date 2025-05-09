import torch
import torch.nn as nn
import torch.nn.functional as F
from tts_impl.functional.parallel_scan import parallel_scan_log


def g(x):
    return torch.where(x >= 0, x+0.5, torch.sigmoid(x))

def log_g(x):
    return torch.where(x >= 0, (F.relu(x)+0.5).log(), -F.softplus(-x))

class MinGRU(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.linear_h = nn.Linear(d_model, d_model, bias=False)
        self.linear_z = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, h_prev = None):
        if h_prev is None:
            h_prev = torch.zeros(x.shape[0], 1, self.d_model)
        if x.shape[1] == 1:
            return self._sequential_forward(x, h_prev)
        else:
            return self._parallel_forward(x, h_prev)

    def _sequential_forward(self, x, h_prev):
        z = torch.sigmoid(self.linear_z(x))
        h_tilde = g(self.linear_h(x))
        h = (1 - z) * h_prev + z * h_tilde
        return h
    
    def _parallel_forward(self, x, h_0):
        k = self.linear_z(x)
        log_z = -F.softplus(-k)
        log_coeffs = -F.softplus(k)
        log_h_0 = log_g(h_0)
        log_tilde_h = log_g(self.linear_h(x))
        h = parallel_scan_log(F.pad(log_coeffs, (0,0,1,0)), torch.cat([log_h_0, log_z + log_tilde_h], dim=1))
        h = h[:, 1:]
        return h