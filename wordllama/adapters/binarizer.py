import torch
from torch import nn
from torch.autograd import Function
import math


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors

        # Gradient is passed through unchanged where x values are between -1 and 1
        grad_x = grad_output.clone()

        # Restrict to the pass-through region
        grad_x[x.abs() > 1] = 0
        return grad_x


class TanhSTEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sign(torch.tanh(x))

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_x = grad_output.clone() * (1 - torch.tanh(x) ** 2)  # derivative of tanh
        return grad_x


# ReSTE
class ReSTEFunction(Function):
    """https://github.com/DravenALG/ReSTE"""

    @staticmethod
    def forward(ctx, x, t, o):
        ctx.save_for_backward(x, t, o)
        out = torch.sign(x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, t, o = ctx.saved_tensors

        interval = 0.1

        tmp = torch.zeros_like(x)
        mask1 = (x <= t) & (x > interval)
        tmp[mask1] = (1 / o) * torch.pow(x[mask1], (1 - o) / o)
        mask2 = (x >= -t) & (x < -interval)
        tmp[mask2] = (1 / o) * torch.pow(-x[mask2], (1 - o) / o)
        tmp[(x <= interval) & (x >= 0)] = approximate_function(interval, o) / interval
        tmp[(x <= 0) & (x >= -interval)] = (
            -approximate_function(-interval, o) / interval
        )

        # calculate the final gradient
        grad_x = tmp * grad_output.clone()

        return grad_x


def approximate_function(x, o):
    if x >= 0:
        return math.pow(x, 1 / o)
    else:
        return -math.pow(-x, 1 / o)


class Binarizer(nn.Module):
    def __init__(self, ste="tanh"):
        super().__init__()
        assert ste in ["ste", "reste", "stochastic", "tanh"]

        self.ste = ste
        if ste == "reste":
            self.t = torch.tensor(1.5).float()
            self.o = torch.tensor(1).float()

    def forward(self, tensors):
        x = tensors["sentence_embedding"]

        # Apply Straight-through estimator
        if self.ste == "reste":
            x = ReSTEFunction.apply(x, self.t.to(x.device), self.o.to(x.device))
        elif self.ste == "tanh":
            x = TanhSTEFunction.apply(x)
        else:
            x = STEFunction.apply(x)

        tensors.update({"sentence_embedding": x})
        return tensors

    def save(self, *args, **kwargs):
        pass  # nothing to save
