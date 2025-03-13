from abc import ABC
from dataclasses import field
import torch
from torch import nn


class BaseCompressor(nn.Module):
    compressor_name: str

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        """
            Args:
                x: torch.Tensor, shape = [batch, seqlen, hidden_size]
            Returns:
                torch.Tensor, shape = [batch, seqlen_compress, hidden_size]
        """
        raise NotImplementedError


class IdentityCompressor(BaseCompressor):
    compressor_name: str = "identity"

    def __init__(self):
        super().__init__()

    def forward(self, input_embeds: torch.Tensor, position_ids: torch.Tensor, attention_mask: torch.Tensor, labels, **kwargs):
        video_grid_thw = kwargs["video_grid_thw"]
        spatial_merge_size = kwargs["spatial_merge_size"]
        w, h = video_grid_thw[0][1], video_grid_thw[0][2]
        n_token_per_frame_compressed = w // spatial_merge_size * h // spatial_merge_size
        return input_embeds, position_ids, attention_mask, labels, n_token_per_frame_compressed

class S2Compressor(BaseCompressor):
    compressor_name: str = "S2"

    def __init__(self):
        super().__init__()

    def flat_square_2x2(self, visual_part: torch.Tensor, video_grid_thw: torch.Tensor, spatial_merge_size: int):
        batch, seqlen, hidden = visual_part.shape
        t, h, w = video_grid_thw[0][0], video_grid_thw[0][1] // spatial_merge_size, video_grid_thw[0][2] // spatial_merge_size
        visual_part = visual_part.contiguous()
        visual_part = visual_part.view(batch, t, w * h, hidden)
        if w * h % 4 != 0:
            visual_part = torch.cat([visual_part, torch.zeros(batch, t, 4 - w * h % 4, hidden, device=visual_part.device, dtype=visual_part.dtype)], dim=2)
            visual_part = visual_part.contiguous()
        visual_part = visual_part.view(batch, -1, hidden * 4)
        return visual_part

    def forward(self, input_embeds: torch.Tensor, position_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None, **kwargs):
        video_grid_thw = kwargs["video_grid_thw"]
        spatial_merge_size = kwargs["spatial_merge_size"]
        visual_start = kwargs["visual_start"]
        visual_end = kwargs["visual_end"]
        batch, visual_len, hidden = input_embeds.shape
        n_frames = video_grid_thw[0][0].item()
        visual_part = input_embeds

        visual_part = self.flat_square_2x2(visual_part, video_grid_thw, spatial_merge_size)
        visual_mask = torch.arange(0, visual_len, 4, device=visual_part.device) + visual_start

        seqlen = attention_mask.shape[1]
        mask = torch.zeros(batch, seqlen, device=input_embeds.device, dtype=torch.bool)
        mask[:, :visual_start] = True
        mask[:, visual_end:] = True
        mask[:, visual_mask] = True
        position_ids = position_ids[:, mask]
        if position_ids.dim() == 2:
            position_ids = position_ids.unsqueeze(dim=1)
        attention_mask = attention_mask[mask]
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(dim=0)
            

        n_tokens_per_frames_compressed = visual_part.shape[1] // n_frames
        position_ids = position_ids.contiguous()
        attention_mask = attention_mask.contiguous()
        if labels is not None:
            labels = labels[mask]
            labels = labels.contiguous()
        return visual_part, position_ids, attention_mask, labels, n_tokens_per_frames_compressed


