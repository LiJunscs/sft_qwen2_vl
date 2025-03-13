import torch
from torch import nn
from typing import Dict
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel

from .base_compressor import BaseCompressor, IdentityCompressor, S2Compressor


class CompressorConfig(PretrainedConfig):
    model_type = "compressor_config"

    def __init__(self, compressor_name: str, **kwargs):
        super().__init__(**kwargs)
        self.compressor_name = compressor_name
        for key, item in kwargs.items():
            setattr(self, str(key), item)

class OptionalCompressor(PreTrainedModel):
    config_class = CompressorConfig

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.compressor_name = config.compressor_name

        self.compressor_config = config

        if self.compressor_name == "s2":
            self.layers = nn.Sequential(
                S2Compressor(),
                nn.LayerNorm(config.hidden_size * 4),
                nn.Linear(config.hidden_size * 4, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
        else:
            self.layers = nn.Sequential(IdentityCompressor())
    
    def forward(self, input_embeds, position_ids, attention_mask, labels, **kwargs):
        visual_start = kwargs["visual_start"]
        visual_end = kwargs["visual_end"]
        video_grid_thw = kwargs["video_grid_thw"]

        visual_part = input_embeds[:, visual_start: visual_end]

        visual_part, position_ids, attention_mask, labels, n_token_per_frames_compressed = self.layers[0](visual_part, position_ids, attention_mask, labels, **kwargs)
        for layer in self.layers[1:]:
            visual_part = layer(visual_part)
        input_embeds = torch.cat([input_embeds[:, :visual_start], visual_part, input_embeds[:, visual_end:]], dim=1)
        input_embeds = input_embeds.contiguous()
        return input_embeds, position_ids, attention_mask, labels, n_token_per_frames_compressed



