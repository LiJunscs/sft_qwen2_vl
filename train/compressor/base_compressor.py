from abc import ABC
from dataclasses import field
import torch
from torch import nn
from typing import Optional, Dict


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


def flat_square_3x3(x):
    n, w, h, c = x.size()
    if w % 3 != 0:
        x = torch.concat([x, torch.zeros((n, 3 - (w % 3), h, c), dtype=x.dtype).to(x.device)], dim=1).contiguous()
        n, w, h, c = x.size()
    x = x.contiguous()
    if h % 3 != 0:
        x = torch.concat([x, torch.zeros((n, w, 3 - (h % 3), c), dtype=x.dtype).to(x.device)], dim=2).contiguous()
        n, w, h, c = x.size()
    x = x.view(n, w, int(h / 3), int(c * 3))
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(n, int(h / 3), int(w / 3), int(c * 9))
    x = x.permute(0, 2, 1, 3).contiguous()
    return x

class S2Compressor(BaseCompressor):
    compressor_name: str = "2x2_S2"

    def __init__(self, image_token_id: int, video_token_id: int):
        super().__init__()
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

    def flat_square_2x2(self, x: torch.Tensor, thw: torch.Tensor):
        _, c = x.shape
        t, h, w = thw[0], thw[1], thw[2]
        x = x.view(t, h, w, c)
        if w % 2 == 1:
            x = torch.concat([x, torch.zeros((t, h, 1, c), dtype=x.dtype).to(x.device)], dim=1).contiguous()
            t, h, w, c = x.size()
        x = x.contiguous()
        if h % 2 == 1:
            x = torch.concat([x, torch.zeros((t, 1, h, c), dtype=x.dtype).to(x.device)], dim=2).contiguous()
            t, h, w, c = x.size()
        x = x.view(t, int(h / 2), w, int(c * 2))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(t, int(w / 2), int(h / 2), int(c * 4))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, input_ids: torch.LongTensor, position_ids: torch.LongTensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None, *kwargs) -> Dict[str, torch.Tensor]:
        """
        Args:
            pixel_values: shape = (visual_seqlen, hidden)
            grid_thw: shape = (batch, 3)
            input_ids: shape = (batch, seqlen)
            position_ids: shape = (3, batch, seqlen)
            attention_mask: shape = (batch, seqlen)
            labels: Optional, If not None, shape = (batch, seqlen)
        Return:
            pixel_values: Tensor(visual_length_compressed, hidden),
            grid_thw: Tensor(batch, 3)
            input_ids: Tensor(1 seqlen_compressed_flatten)
            position_ids: Tensor(3, 1, seqlen_compressed_flatten)
            attention_mask: None
            cu_seqlens_q: Tensor(batch + 1, )
            max_seqlen_q: int
            labels: Tensor(1, seqlen_compressed_flatten) if labels else None

        NOTE: 
            1. 压缩后将全部的tensor在seqlen维度拼接, 避免大量的padding, 后期使用flash_attn_varlen计算. 
            2. 返回值的attention_mask将永远被置为None, 计算flash_attn_varlen的数据将后续手动计算
        """
        ## 处理多batch的情况, 由于pixel_values是经过flatten的, 且batch中每个数据的thw不一定相同, 所以暂时采用循环完成
        input_ids_compressed = []
        position_ids_compressed = []
        pixel_values_compressed = []
        grid_thw_compressed = []
        if labels is not None:
            labels_compressed = []
        last_media = 0
        device = input_ids.device
        batch, seqlen = input_ids.shape

        cu_seqlens_q = [0]
        max_seqlen_q = 0
        for i in range(batch):
            thw = grid_thw[i]
            n_visual_tokens = t * h * w
            visual_compressed = self.flat_square_2x2(pixel_values[last_media : last_media + n_visual_tokens], thw=thw)
            pixel_values_compressed.append(visual_compressed)
            # 处理thw, input_ids中的visual_pad, position_ids, labels
            t, h, w = thw[0], thw[1], thw[2]
            grid_thw_compressed.append(torch.tensor([t, h // 2, w //2], device=device, dtype=torch.int32))
            if t == 1:
                # 图片
                image_token_indices = (input_ids[i] == self.image_token_id).squeeze()
                image_start = image_token_indices[0]
                image_end = image_token_indices[-1]
                mask = torch.zeros(seqlen, device=device, dtype=torch.bool)
                selectec_indices = torch.arange(image_start, image_end + 1, 4, device=device)
                mask[: image_start] = True
                mask[selectec_indices] = True
                mask[image_end + 1 : ] = True
                # 根据attention_mask去掉padding部分
                mask[~attention_mask[i]] = False

                tokens = input_ids[i][mask]
                q_len = tokens.shape[0]
                cu_seqlens_q.append(q_len)
                max_seqlen_q = max(max_seqlen_q, q_len)
                input_ids_compressed.append(tokens)

                # position_ids = shape(3, batch, seqlen, hidden)
                position_ids_compressed.append(position_ids[:, i, mask])

                # label = shape (batch, seqlen)
                if labels is not None:
                    labels_compressed.append(labels[i][mask])
            else:
                # 视频
                video_token_indices = (input_ids[i] == self.video_token_id).squeeze()
                video_start = video_token_indices[0]
                video_end = video_token_indices[-1]

                mask = torch.zeros(seqlen, device=device, dtype=torch.bool)
                selectec_indices = torch.arange(video_start, video_end + 1, 4, device=device)
                mask[: video_start] = True
                mask[selectec_indices] = True
                mask[video_end + 1 : ] = True
                # 根据attention_mask去掉padding部分
                mask[~attention_mask[i]] = False

                tokens = input_ids[i][mask]
                q_len = tokens.shape[0]
                cu_seqlens_q.append(q_len)
                max_seqlen_q = max(max_seqlen_q, q_len)
                input_ids_compressed.append(tokens)

                # position_ids = shape(3, batch, seqlen)
                position_ids_compressed.append(position_ids[:, i, mask])

                # label = shape (batch, seqlen)
                if labels is not None:
                    labels_compressed.append(labels[i][mask])
        # 将全部的tensor在seqlen维度拼接，避免大量的padding, 后期使用flash_attn_varlen计算
        pixel_values_compressed = torch.cat(pixel_values_compressed, dim=0)
        grid_thw_compressed = torch.stack(grid_thw_compressed, dim=0)

        input_ids_compressed = torch.cat(input_ids_compressed, dim=0)
        input_ids_compressed = input_ids_compressed.unsqueeze(dim=0)
        input_ids_compressed = input_ids_compressed.contiguous()

        position_ids_compressed = torch.cat(position_ids_compressed, dim=-1)
        position_ids_compressed = position_ids_compressed.unsqueeze(dim=1)
        position_ids_compressed = position_ids_compressed.contiguous()

        cu_seqlens_q = torch.tensor(cu_seqlens_q, device=device, dtype=torch.int64).cumsum(dim=0)

        if labels_compressed is not None:
            labels_compressed = torch.cat(labels_compressed, dim=0)
            labels_compressed = labels_compressed.unsqueeze(dim=0)
            labels_compressed = labels_compressed.contiguous()
        if labels is None:
            labels_compressed = None
        return pixel_values_compressed, grid_thw_compressed, input_ids_compressed, position_ids_compressed, None, cu_seqlens_q, max_seqlen_q, labels_compressed



