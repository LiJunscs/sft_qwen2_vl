import torch
import torch.nn as nn

from .dummy import dummyMerger

__all__ = ['Projector']

class Projector(nn.Module):
    def __init__(self, args):
        super(Projector, self).__init__()
        self.MODEL_MAP = {
            'dummy': dummyMerger
        }
         
        cls = self.MODEL_MAP[args['projector_cls']]
        self.projector = cls(**args['kwargs'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)