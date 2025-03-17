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

    def initialize_model(self):

        for m in self.projector.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)