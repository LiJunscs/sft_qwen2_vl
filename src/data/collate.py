from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch
from transformers import AutoProcessor

from src.constants import IGNORE_INDEX
from src.utils.logging import logger
from qwen_vl_utils import process_vision_info

__all__ = ["DataCollator"]


@dataclass
class DataCollator:
    processor: AutoProcessor

    def __init__(self, processor: AutoProcessor):
        super().__init__()
        self.processor = processor

    def construct_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        
        batch_size, seq_len = input_ids.shape
        labels = torch.full((batch_size, seq_len), IGNORE_INDEX, dtype=torch.long)

        for i in range(batch_size):

            assistant_start = torch.nonzero(input_ids[i] == self.processor.tokenizer.encode("assistant")[0]).squeeze()
            assert assistant_start.numel() == 1, "only one assistant token is supported."
            assistant_start_indices = assistant_start.item() + 2
            im_end = torch.nonzero(input_ids[i] == self.processor.tokenizer.encode("<|im_end|>")[0]).squeeze()
            assistant_end_indices = im_end[-1].item() + 1

            labels[i,assistant_start_indices:assistant_end_indices] = input_ids[i,assistant_start_indices:assistant_end_indices]

        return labels

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        # Gather everything from the batch
        conversations = self.processor.apply_chat_template(instances, tokenize=False)
        image_inputs, video_inputs = process_vision_info(instances)

        instances = self.processor(
            text=conversations,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side='left',
            return_tensors="pt",   
        )
        instances.update({'labels': self.construct_labels(instances.input_ids)})

        return instances