import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class StatefulModule(nn.Module):
    """
    An abstract class that represents stateful behavior for sequences. It is used in RNNs, GRUs, etc.
    """

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
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
            x, h = self._sequential_forward(x, h)
        else:
            x, h = self._parallel_forward(x, h)

        return x, h

    def _sequential_forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplemented

    def _parallel_forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplemented

    def _initial_state(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplemented


class PointwiseModule:
    """
    An abstract class that indicates that the module is independent of the series direction.
    """

    pass
