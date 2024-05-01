from torch import Tensor
from torch import nn
import torch
from torchvision import models

class TFModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.tf_model = nn.Transformer()

    def forward(self, x: Tensor) -> Tensor:
        out = self.tf_model(x[0], x[1])
        return x

if __name__ == '__main__':
    model = TFModel()
    src = torch.rand((10, 32, 512))
    tgt = torch.rand((20, 32, 512))
    out = model([src, tgt])
