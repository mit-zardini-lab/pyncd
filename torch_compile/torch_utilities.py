import torch
from torch import nn
import math
from typing import *

def to_tuple[T](x: T | tuple[T, ...]) -> tuple[T, ...]:
    return x if isinstance(x, tuple) else (x,)

# A wrapper for properly initialized weights.
class Weights(nn.Module):
    def __init__(self,
                 size: tuple[int, ...],
                 bias_size: None | tuple[int, ...] = None,
                 device = None, dtype = None):
        """Wrapper for weights initialized akin to nn.Linear

        Args:
            size (tuple[int, ...]): The size of the weights
            bias_size (None | tuple[int, ...], optional): The size of a bias (if needed). Defaults to None.
            device (_type_, optional): The device on which the weights are placed. Defaults to None.
            dtype (_type_, optional): The dtype of the weights. Defaults to None.
        """
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.size = size
        self.weight = nn.Parameter(torch.empty(size, **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(bias_size, **factory_kwargs)) if bias_size else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
    def __repr__(self):
        return f'{type(self).__qualname__}(size={self.size})'

class Multilinear(nn.Module):
    def __init__(self,
        in_size  : tuple[int, ...] | int,
        out_size : tuple[int, ...] | int, 
        bias     : bool = True,
        device   : Any | None = None,
        dtype    : Any | None = None) -> None:
        """A learned linear layer which supports tuple axis sizes.

        Args:
            in_size (tuple[int, ...] | int): The size of input axes.
            out_size (tuple[int, ...] | int): The size of output axes.
            bias (bool, optional): Whether to incorporate a bias on the outputs. Defaults to True.
            device (Any | None, optional): _description_. Defaults to None.
            dtype (Any | None, optional): _description_. Defaults to None.
        """        
        super().__init__()
        in_size, out_size = to_tuple(in_size), to_tuple(out_size)
        self.in_size, self.out_size , self.bias = in_size, out_size, bias
        self.weights = Weights(in_size + out_size, out_size if bias else None, device, dtype)
    
    def forward(self, x_in : torch.Tensor) -> torch.Tensor:
        # Reshape the input. The last axes should match, else there's an error.
        x_shape = x_in.shape
        assert tuple(x_shape[-len(self.in_size):]) == self.in_size
        x_out = torch.tensordot(x_in, self.weights.weight, len(self.in_size)) # type: ignore
        if self.bias:
            x_out += self.weights.bias
        return x_out
    
    def extra_repr(self) -> str:
        return f'{self.in_size} -> {self.out_size}{', bias={self.bias}' if self.bias else ''}'