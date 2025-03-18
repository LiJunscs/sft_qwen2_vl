from typing import Optional
import torch
import torch.nn as nn

# from .dummy import dummyMerger
from .downsample_2x2_s2 import S2Compressor

__all__ = ['Projector']

class IdentityProjector(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, visual_embeds: torch.Tensor, grid_thw: torch.Tensor, input_ids: torch.LongTensor, position_ids: torch.LongTensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs):
        return visual_embeds, grid_thw, input_ids, position_ids, attention_mask, labels

class Projector(nn.Module):
    def __init__(self, args):
        super(Projector, self).__init__()
        self.MODEL_MAP = {
            # 'dummy': dummyMerger,
            "identity": IdentityProjector,
            "s2": S2Compressor
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

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.projector(x, *args, **kwargs)