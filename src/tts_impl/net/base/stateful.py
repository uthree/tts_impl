from typing import Optional, Sequence, Tuple

import torch
from torch import nn as nn
from torch.nn import functional as F


class StatefulModule(nn.Module):
    """
    An abstract class that represents stateful behavior for sequences. It is used in RNNs, GRUs, etc.
    """

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        default implementation of forward pass

        Args:
            x: [batch_size, seq_len, d_model]
            h: [batch_size, 1, d_state]

        Returns:
            x: [batch_size, seq_len, d_model]
            h: [batch_size, 1, d_state]
        """
        assert x.shape[1] != 0, "input sequence should be longer than 0."

        # initialize state if initial state is not given.
        if h is None:
            h = self._initial_state(x)

        # if sequence length is 1, use sequential forward implementation. otherwise, use parallel implementation.
        if x.shape[1] == 1:
            x, h = self._sequential_forward(x, h, *args, **kwargs)
        else:
            x, h = self._parallel_forward(x, h, *args, **kwargs)

        return x, h

    def _sequential_forward(
        self, x: torch.Tensor, h: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sequential mode forward pass.

        Args:
            x: [batch_size, 1, d_model]
            h: [batch_size, 1, d_state]

        Returns:
            x: [batch_size, 1, d_model]
            h: [batch_size, 1, d_state]
        """
        return self._parallel_forward(x, h, *args, **kwargs)

    def _parallel_forward(
        self, x: torch.Tensor, h: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parallel mode forward pass.

        Args:
            x: [batch_size, seq_len, d_model]
            h: [batch_size, 1, d_state]

        Returns:
            x: [batch_size, seq_len, d_model]
            h: [batch_size, 1, d_state]
        """
        raise NotImplemented

    def _initial_state(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Initialize state.

        Args:
            x: Tensor

        Returns:
            h: [batch_size, 1, d_state]
        """
        raise NotImplemented


class PointwiseModule:
    """
    An abstract class that indicates that the module is independent of the time-sequential direction.
    """

    pass


class StatefulModuleSequential(StatefulModule):
    def __init__(self, stateful_modules: Sequence[StatefulModule]):
        super().__init__()
        self.stateful_modules = nn.ModuleList(stateful_modules)
        self.h_dim_list = []

    def _initial_state(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        hs = []
        h_dim_list = []
        for module in self.stateful_modules:
            h = module._initial_state(x, *args, **kwargs)
            hs.append(h)
            h_dim_list.append(h.shape[2])
        self.h_dim_list = h_dim_list
        return torch.cat(hs, dim=2)

    def _parallel_forward(
        self, x: torch.Tensor, h: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # calculate sum of all hidden state dimension if it's not already calculated.
        if self.h_dim_list is None:
            self._initial_state(x)

        # forward pass each layer
        hs = list(torch.split(h, self.h_dim_list, dim=2))
        for i in range(len(self.stateful_modules)):
            x, hs[i] = self.stateful_modules[i](x, hs[i], *args, **kwargs)
        hs = torch.cat(hs, dim=2)
        return x, hs

    def append(self, m):
        self.stateful_modules.append(m)


def sanity_check_stateful_module(
    stateful_module: StatefulModule,
    x: torch.Tensor,
    h_0: Optional[torch.Tensor] = None,
    atol: float = 1e-4,
):
    with torch.no_grad():
        y_par, _ = stateful_module(x, h_0)
        y_seq = []
        h_t = h_0
        for x_t in x.unbind(dim=1):
            x_t = x_t[:, None]
            y_t, h_t = stateful_module(x_t, h_t)
            y_seq.append(y_t)
        y_seq = torch.cat(y_seq, dim=1)
        print((y_seq - y_par).mean())
        assert torch.allclose(y_par, y_seq, atol=atol)
