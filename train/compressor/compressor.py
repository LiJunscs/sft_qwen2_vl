import torch
from torch import nn
from typing import Dict
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel

from .base_compressor import BaseCompressor, IdentityCompressor, S2Compressor


class CompressorConfig(PretrainedConfig):
    model_type = "compressor_config"

    def __init__(self, compressor_name: str, image_token_id: int, video_token_id: int, padding_token_id: int, spatial_merge_size: int, hidden_size: int, **kwargs):
        super().__init__(**kwargs)
        self.compressor_name = compressor_name
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.padding_token_id = padding_token_id
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size = hidden_size
        for key, item in kwargs.items():
            setattr(self, str(key), item)

class OptionalCompressor(PreTrainedModel):
    config_class = CompressorConfig

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.compressor_name = config.compressor_name

        self.compressor_config = config

        if self.compressor_name == "2x2_s2":
            self.layers = nn.Sequential(
                S2Compressor(config.image_token_id, config.video_token_id, config.padding_token_id, config.spatial_merge_size),
                nn.LayerNorm(config.hidden_size * 4),
                nn.Linear(config.hidden_size * 4, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
        else:
            self.layers = nn.Sequential(IdentityCompressor())
    
    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, input_ids: torch.Tensor, position_ids, attention_mask, labels=None, scale: bool = False, **kwargs):
        if scale:
            origin_data = (pixel_values, grid_thw, input_ids, position_ids, attention_mask, labels)
        else:
            origin_data = None
        compressor_layer = self.layers[0]
        pixel_values, grid_thw, input_ids, position_ids, attention_mask, labels = compressor_layer(
            pixel_values=pixel_values,
            grid_thw=grid_thw,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        hidden_state = pixel_values
        dtype = hidden_state.dtype
        hidden_state = hidden_state.float()
        for layer in self.layers[1:]:
            hidden_state = layer(hidden_state)
        hidden_state = hidden_state.to(dtype=dtype)
        return ((hidden_state, grid_thw, input_ids, position_ids, attention_mask, labels), origin_data)


