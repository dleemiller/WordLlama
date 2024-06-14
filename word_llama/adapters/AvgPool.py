from torch import nn


class AvgPool(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, tensors):
        mask = tensors["attention_mask"].unsqueeze(dim=-1)
        x = (tensors["x"] * mask).sum(dim=1) / mask.sum(dim=1)
        return x
