import os
import sys
root_dir = "/home/lijun2/multimodal/sft_qwen2_vl/"
if root_dir not in sys.path:
    sys.path.append(root_dir)
    print(sys.path)
import torch
from torch import nn
from typing import Dict, Optional


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
    
class S2Compressor(nn.Module):
    compressor_name: str = "2x2_S2"

    def __init__(self, dim: int, image_token_id: int, video_token_id: int, pad_token_id: int, spatial_merge_size: int):
        super().__init__()
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.pad_token_id = pad_token_id
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size = dim * 4
        
        self.ln_q = Qwen2RMSNorm(self.hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def flat_square_2x2(self, x: torch.Tensor, thw: torch.Tensor):
        _, c = x.shape
        t, h, w = thw[0], thw[1], thw[2]
        h = h // self.spatial_merge_size
        w = w // self.spatial_merge_size
        x = x.view(t, h, w, c)
        if h % 2 != 0:
            x = torch.concat([x, torch.zeros((t, 1, w, c), dtype=x.dtype).to(x.device)], dim=1).contiguous()
            t, h, w, c = x.size()
        x = x.contiguous()
        if w % 2 != 0:
            x = torch.concat([x, torch.zeros((t, h, 1, c), dtype=x.dtype).to(x.device)], dim=2).contiguous()
            t, h, w, c = x.size()
        x = x.view(t, int(h / 2), w, int(c * 2))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(t, int(w / 2), int(h / 2), int(c * 4))
        x = x.permute(0, 2, 1, 3).contiguous()
        t, h, w, c = x.shape
        h *= self.spatial_merge_size
        w *= self.spatial_merge_size
        thw = torch.tensor([t, h, w], device=x.device, dtype=torch.int32)
        x = x.view(-1, c)
        return x, thw

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor, input_ids: torch.LongTensor, position_ids: torch.LongTensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
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
        batch, seqlen = input_ids.shape
        device = input_ids.device
        visual_idx = 0
        last_visual_end = 0

        # 纯文本的情况
        if x is None:
            return x, grid_thw, input_ids, position_ids, attention_mask, labels
        image_or_video = grid_thw[0, 0] == 1
        # image_or_video mean that visual_embeds is image or video. True is image, or not video
        visual_token_id = self.image_token_id if image_or_video else self.video_token_id
        visual_embeds_compressed = []
        for i in range(batch):
            if torch.any(input_ids[i] == visual_token_id):
                t, h, w = grid_thw[visual_idx]
                n_visual_tokens = t * h * w // (self.spatial_merge_size ** 2)
                visual_tokens, thw = self.flat_square_2x2(x[last_visual_end : last_visual_end + n_visual_tokens], grid_thw[visual_idx])
                visual_embeds_compressed.append(visual_tokens)
                t, h, w = thw
                grid_thw[visual_idx] = thw

                # 处理input_ids
                mask = input_ids[i] != visual_token_id
                visual_token_start_idx = torch.nonzero(mask == False)[0]
                n_visual_tokens_compressed = t * h * w // (self.spatial_merge_size ** 2)

                padding_length = n_visual_tokens - n_visual_tokens_compressed
                # 将部分visual_token_id取出，其他的去掉
                mask[visual_token_start_idx : visual_token_start_idx + n_visual_tokens_compressed] = True

                # 取出input_ids, 并进行padding
                input_ids[i] = torch.cat(
                    [
                        torch.full((padding_length, ), self.pad_token_id, device=device, dtype=input_ids.dtype),
                        input_ids[i][mask]
                    ],
                    dim=0
                )

                # 取出positon_ids，并进行padding, position_ids的mask需要修改，不能与input_ids的相同，但是qwen2.5vl将positiion的计算放在了visual后，暂时不会出现问题
                if position_ids is not None:
                    position_ids[:, i] = torch.cat(
                        [
                            torch.zeros((3, padding_length), device=device, dtype=position_ids.dtype),
                            position_ids[:, i, mask]
                        ],
                        dim=1
                    )

                # 取出attention_mask，并进行padding
                attention_mask[i] = torch.cat(
                    [
                        torch.full((padding_length, ), 0, device=device, dtype=attention_mask.dtype),
                        attention_mask[i][mask]
                    ],
                    dim=0
                )

                # 取出labels(如果存在)，并进行padding
                if labels is not None:
                    labels[i] = torch.cat(
                        [
                            torch.full((padding_length, ), -100, device=device, dtype=labels.dtype),
                            labels[i][mask]
                        ],
                        dim=0
                    )

                last_visual_end += n_visual_tokens
                visual_idx += 1
        x = torch.cat(visual_embeds_compressed, dim=0)
        x = x.contiguous()

        input_ids = input_ids.contiguous()
        if position_ids is not None:
            position_ids = position_ids.contiguous()
        attention_mask = attention_mask.contiguous()
        if labels is not None:
            labels = labels.contiguous()
        

        hidden_states = self.ln_q(x)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, grid_thw, input_ids, position_ids, attention_mask, labels


def test_s2_comress():
    n_visual = 96
    visual_1 = 30
    visual_2 = 66
    llm_dim = 5
    image_token_id = 1
    video_token_id = 2
    padding_token_id = 3
    spatial_merge_size = 2
    seqlen = visual_2 + 5 + 5
    grid_thw = [[1, 10, 12], [1, 22, 12]]
    device = "cuda"
    dtype = torch.int32
    grid_thw = torch.tensor(grid_thw, device=device, dtype=dtype)

    visual_embeds = torch.rand((n_visual, llm_dim), device=device, dtype=torch.float32)

    pre_prompt = torch.randint(10, 19, (5, ), device=device, dtype=dtype)
    post_prompt = torch.randint(10, 99, (5, ), device=device, dtype=dtype)

    input_1 = torch.full((visual_1, ), image_token_id, device=device, dtype=dtype)
    input_1 = torch.cat([torch.full((visual_2 - visual_1, ), padding_token_id, device=device, dtype=dtype), pre_prompt, input_1, post_prompt], dim=0)
    input_2 = torch.full((visual_2, ), image_token_id, device=device, dtype=dtype)
    input_2 = torch.cat([pre_prompt, input_2, post_prompt], dim=0)

    input_ids = torch.stack([input_1, input_2], dim=0)

    position_ids = torch.randint(0, 9, (3, 2, seqlen), device=device, dtype=dtype)
    attention_mask = (input_ids != padding_token_id).to(dtype=dtype)

    compressor = S2Compressor(dim=llm_dim, image_token_id=image_token_id, video_token_id=video_token_id, pad_token_id=padding_token_id, spatial_merge_size=spatial_merge_size)
    compressor = compressor.to(device).eval()
    out = compressor(visual_embeds, True, grid_thw, input_ids, position_ids, attention_mask)
    print(1)

if __name__ == "__main__":
    test_s2_comress()