from abc import ABC
from dataclasses import field
import torch
from torch import nn
from typing import Optional, Dict
from torch.nn import functional as F


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

    def __init__(self, image_token_id: int, video_token_id: int, padding_token_id: int, spatial_merge_size: int, ):
        super().__init__()
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.spatial_merge_size = spatial_merge_size
        self.padding_token_id = padding_token_id

    def flat_square_2x2(self, x: torch.Tensor, thw: torch.Tensor):
        _, c = x.shape
        t, h, w = thw[0], thw[1], thw[2]
        x = x.view(t, h, w, c)
        kernel_size = 2 * self.spatial_merge_size
        if h % kernel_size != 0:
            x = torch.concat([x, torch.zeros((t, kernel_size - (h % kernel_size), w, c), dtype=x.dtype).to(x.device)], dim=1).contiguous()
            t, h, w, c = x.size()
        x = x.contiguous()
        if w % kernel_size != 0:
            x = torch.concat([x, torch.zeros((t, h, kernel_size - (w % kernel_size), c), dtype=x.dtype).to(x.device)], dim=2).contiguous()
            t, h, w, c = x.size()
        x = x.view(t, int(h / 2), w, int(c * 2))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(t, int(w / 2), int(h / 2), int(c * 4))
        x = x.permute(0, 2, 1, 3).contiguous()
        t, h, w, c = x.shape
        thw = torch.tensor([t, h, w], device=x.device, dtype=torch.int32)
        x = x.view(-1, c)
        return x, thw

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

        """
        ## 处理多batch的情况, 由于pixel_values是经过flatten的, 且batch中每个数据的thw不一定相同, 所以暂时采用循环完成
        input_ids_compressed = []
        position_ids_compressed = []
        pixel_values_compressed = []
        grid_thw_compressed = []
        attention_mask_compressed = []
        if labels is not None:
            labels_compressed = []

        last_media = 0
        device = input_ids.device
        batch, seqlen = input_ids.shape

        max_padding_length = 0
        for i in range(batch):
            thw = grid_thw[i]
            t, h, w = thw
            n_visual_tokens = t * h * w
            visual_compressed, thw = self.flat_square_2x2(pixel_values[last_media : last_media + n_visual_tokens], thw=thw)
            t, h, w = thw
            last_media += n_visual_tokens
            pixel_values_compressed.append(visual_compressed)
            # 处理thw, input_ids中的visual_pad, position_ids, labels
            grid_thw_compressed.append(thw)
            if t == 1:
                # 图片
                ## NOTE: 暂时不清楚为什么在图片中出现了video_token_id
                image_token_indices = torch.nonzero((input_ids[i] == self.image_token_id) | (input_ids[i] == self.video_token_id)).squeeze()
                image_start = image_token_indices[0]
                image_end = image_token_indices[-1]
                mask = torch.zeros(seqlen, device=device, dtype=torch.bool)
                ## NOTE: 存在问题，nvila的s2压缩会导致h, w维度顺序的混乱
                selectec_indices = torch.linspace(image_start, image_end, h  // self.spatial_merge_size * w // self.spatial_merge_size, device=device, dtype=torch.int32)
                mask[: image_start] = True
                mask[selectec_indices] = True
                mask[image_end + 1 : ] = True
                # 去掉padding的token
                mask[attention_mask[i] == 0] = False


            else:
                # 视频
                video_token_indices = torch.nonzero((input_ids[i] == self.image_token_id) | (input_ids[i] == self.video_token_id)).squeeze()
                video_start = video_token_indices[0]
                video_end = video_token_indices[-1]

                mask = torch.zeros(seqlen, device=device, dtype=torch.bool)
                selectec_indices = torch.linspace(video_start, video_end,  t * h  // self.spatial_merge_size * w // self.spatial_merge_size, device=device, dtype=torch.int32)
                mask[: video_start] = True
                mask[selectec_indices] = True
                mask[video_end + 1 : ] = True
                # 去掉padding的token
                mask[attention_mask[i] == 0] = False

            
            input_ids_compressed_i = input_ids[i][mask]
            input_ids_compressed.append(input_ids_compressed_i)
            q_len = input_ids_compressed_i.shape[0]

            max_padding_length = max(max_padding_length, q_len)

            # position_ids = shape(3, batch, seqlen)
            position_ids_compressed_i = position_ids[:, i, mask]
            position_ids_compressed.append(position_ids_compressed_i)
            # attention_mask = shape(batch, seqlen)
            attention_mask_compressed_i = attention_mask[i, mask]
            attention_mask_compressed.append(attention_mask_compressed_i)
            # label = shape (batch, seqlen)
            if labels is not None:
                labels_compressed_i = labels[i, mask]
                labels_compressed.append(labels_compressed_i)
            

        for i in range(batch):
            input_ids_compressed_i = input_ids_compressed[i]
            position_ids_compressed_i = position_ids_compressed[i]
            attention_mask_compressed_i = attention_mask_compressed[i]
            if labels is not None:
                labels_compressed_i = labels_compressed[i]
            q_len = input_ids_compressed_i.shape[0]
            if q_len >= max_padding_length:
                input_ids_compressed_i = input_ids_compressed_i[-max_padding_length:]
                position_ids_compressed_i = position_ids_compressed_i[:, -max_padding_length:]
                attention_mask_compressed_i = attention_mask_compressed_i[-max_padding_length:]
                if labels is not None:
                    labels_compressed_i = labels_compressed_i[-max_padding_length:]
            else:
                input_ids_compressed_i = torch.cat([torch.full((max_padding_length - q_len, ), self.padding_token_id, device=device, dtype=torch.int32), input_ids_compressed_i], dim=0)
                position_ids_compressed_i = torch.cat([torch.full((3, max_padding_length - q_len, ), 0, device=device, dtype=torch.int64), position_ids_compressed_i], dim=1)
                attention_mask_compressed_i = torch.cat([torch.full((max_padding_length - q_len, ), 0, device=device, dtype=torch.int32), attention_mask_compressed_i], dim=0)
                if labels is not None:
                    labels_compressed_i = torch.cat([torch.full((max_padding_length - q_len, ), -100, device=device, dtype=torch.int32), labels_compressed_i], dim=0)
                
            input_ids_compressed[i] = input_ids_compressed_i
            position_ids_compressed[i] = position_ids_compressed_i
            attention_mask_compressed[i] = attention_mask_compressed_i
            if labels is not None:
                labels_compressed[i] = labels_compressed_i
 

        pixel_values_compressed = torch.cat(pixel_values_compressed, dim=0)
        grid_thw_compressed = torch.stack(grid_thw_compressed, dim=0)

        input_ids_compressed = torch.stack(input_ids_compressed, dim=0)
        input_ids_compressed = input_ids_compressed.contiguous()                    # shape (batch, max_padding_length)

        position_ids_compressed = torch.stack(position_ids_compressed, dim=1)
        position_ids_compressed = position_ids_compressed.contiguous()              # shape (3, batch, max_padding_length)

        attention_mask_compressed = torch.stack(attention_mask_compressed, dim=0)
        attention_mask_compressed = attention_mask_compressed.contiguous()          # shape (batch, max_padding_length)

        if labels is not None:
            labels_compressed = torch.stack(labels_compressed, dim=0)               # shape (batch, max_padding_length)
            labels_compressed = labels_compressed.contiguous()
        else:
            labels_compressed = None
        return pixel_values_compressed, grid_thw_compressed, input_ids_compressed, position_ids_compressed, attention_mask_compressed, labels_compressed


def test_compressor():
    config = {
        "image_token_id": 151655,
        "video_token_id": 151656,
        "padding_token_id": 151643,
        "spatial_merge_size": 2,
    }

    # 对长度不够的 tensor 进行左填充
    def left_pad(tensor, max_length, pad_value):
        pad_length = max_length - tensor.size(0)
        if pad_length > 0:
            padding = torch.full((pad_length,), pad_value, dtype=tensor.dtype, device=tensor.device)
            return torch.cat((padding, tensor), dim=0)
        return tensor
    # 根据填充情况生成 attention_mask
    def generate_attention_mask(tensor, pad_value):
        return (tensor != pad_value).to(torch.int32)
    
    compressor = S2Compressor(**config)
    n_tokens = 10 * 8 * 8 + 5 * 4 * 4
    hidden = 10
    grid_thw = torch.tensor([[10, 8, 8], [5, 4, 4]], device="cuda", dtype=torch.int32)
    pixel_values = torch.randint(0, 10, (n_tokens, hidden), device="cuda", dtype=torch.int32)

    pre_prompt = torch.randint(0, 100, (10, ), device="cuda", dtype=torch.int32)
    post_prompt = torch.randint(0, 100, (10, ), device="cuda", dtype=torch.int32)
    tensor_1 = torch.full((160,), config["video_token_id"], dtype=torch.int32, device="cuda")
    tensor_1 = torch.cat([pre_prompt, tensor_1, post_prompt], dim=0)
    tensor_2 = torch.full((20,), config["video_token_id"],dtype=torch.int32, device="cuda")
    tensor_2 = torch.cat([pre_prompt, tensor_2, post_prompt], dim=0)

    max_length = 180
    pad_value = 151643
    tensor_2 = left_pad(tensor_2, max_length, pad_value)

    input_ids = torch.stack([tensor_1, tensor_2], dim=0)
    attention_mask = generate_attention_mask(input_ids, pad_value)

    position_ids = torch.randint(0, 10000, (3, 2, max_length), device="cuda", dtype=torch.int64)

    out = compressor(pixel_values, grid_thw, input_ids, position_ids, attention_mask)
    print()

def test_flat_square_2x2():
    t, h, w, c, = 1, 6, 4, 2
    x = torch.arange(1, t * h * w * c + 1)
    x = x.view(-1, c)
    y = x.clone()

    compressor = S2Compressor(1000, 1001, 1002, 2)
    x, thw = compressor.flat_square_2x2(x, (t, h, w))
    print()

if __name__ =="__main__":
    test_flat_square_2x2()

    
    


