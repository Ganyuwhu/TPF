import torch
import torch.nn.functional as F
from typing import Optional
from torch import nn, Tensor
from typing import Callable


class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


def exists(val):
    return val is not None


def default(val, default_val):
    return default_val if val is None else val


def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.0)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.0)


class GLU(nn.Module):
    """
    GLU (Gated Linear Unit) module.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension.
        activation (Callable[[Tensor], Tensor]): Activation function to be applied to the gate.
        mult_bias (bool, optional): Whether to multiply the bias term. Defaults to False.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        activation: Callable[[Tensor], Tensor],
        mult_bias: bool = False,
    ):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)
        self.mult_bias = nn.Parameter(torch.ones(dim_out)) if mult_bias else 1.0

    def forward(self, x: Tensor) -> Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate) * self.mult_bias


class SwiGLU(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def forward(self, x):
        """Forward

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: Optional[int] = None,
        dim_out: Optional[int] = None,
        mult: Optional[int] = 4,
        glu: Optional[bool] = False,
        glu_mult_bias: Optional[bool] = False,
        swish: Optional[bool] = False,
        relu_squared: Optional[bool] = False,
        post_act_ln: Optional[bool] = False,
        dropout: Optional[float] = 0.0,
        no_bias: Optional[bool] = False,
        zero_init_output: Optional[bool] = False,
        custom_act: Optional[nn.Module] = None,
        swiglu: Optional[bool] = False,
        triton_kernels_on: bool = False,
    ):
        """
        FeedForward module that applies a series of linear transformations and activations.

        Args:
            dim (int): Input dimension.
            dim_out (int, optional): Output dimension. Defaults to None.
            mult (int, optional): Multiplier for the inner dimension. Defaults to 4.
            glu (bool, optional): Whether to use Gated Linear Units (GLU). Defaults to False.
            glu_mult_bias (bool, optional): Whether to use bias in the GLU operation. Defaults to False.
            swish (bool, optional): Whether to use Swish activation. Defaults to False.
            relu_squared (bool, optional): Whether to use squared ReLU activation. Defaults to False.
            post_act_ln (bool, optional): Whether to apply Layer Normalization after the activation. Defaults to False.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            no_bias (bool, optional): Whether to use bias in the linear transformations. Defaults to False.
            zero_init_output (bool, optional): Whether to initialize the last linear layer to 0. Defaults to False.
            custom_act (nn.Module, optional): Custom activation module. Defaults to None.
            swiglu (bool, optional): Whether to use SwiGLU activation. Defaults to False.
        """
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.mult = mult
        self.glu = glu
        self.glu_mult_bias = glu_mult_bias
        self.swish = swish
        self.relu_squared = relu_squared
        self.post_act_ln = post_act_ln
        self.dropout = dropout
        self.no_bias = no_bias
        self.zero_init_output = zero_init_output
        self.custom_act = custom_act
        self.swiglu = swiglu
        self.triton_kernels_on = triton_kernels_on

        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if relu_squared:
            activation = ReluSquared()
        elif swish:
            activation = nn.SiLU()
        elif custom_act is not None:
            activation = custom_act
        elif swiglu:
            activation = SwiGLU()
        else:
            activation = nn.GELU()

        if glu:
            project_in = GLU(
                dim, inner_dim, activation, mult_bias=glu_mult_bias
            )
        else:
            project_in = nn.Sequential(
                nn.Linear(dim, inner_dim, bias=not no_bias), activation
            )

        if post_act_ln:
            self.ff = nn.Sequential(
                project_in,
                nn.LayerNorm(inner_dim),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim_out, bias=no_bias),
            )
        else:
            self.ff = nn.Sequential(
                project_in,
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim_out, bias=not no_bias),
            )

        # init last linear layer to 0
        if zero_init_output:
            init_zero_(self.ff[-1])

    def forward(self, x):
        """
        Forward pass of the feedforward network

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        return self.ff(x)