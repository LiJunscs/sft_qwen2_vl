import random
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from src.constants import IGNORE_INDEX
from src.train.args import DataArguments
from src.utils.logging import logger

__all__ = ["BaseDataset"]

class BaseDataset(Dataset):
    def __init__(
        self,
        processor: AutoProcessor,
        data_args: DataArguments,
        global_batch_size: int,
        no_system_prompt: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.processor = processor
        self.data_args = data_args
        self.no_system_prompt = no_system_prompt
        self.instances = []
        self.enable_dynamic_res = False
        self.enable_dynamic_res_s2 = False
        self.global_batch_size = global_batch_size

    def process(self, instance: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Dict[str, Any]:
        instance = self.instances[index]

        try:
            # Process instance to conversation
            conversation = self.process(instance)
            text = self.processor.apply_chat_template(conversation, tokenize=False)
            # test = self.processor.apply_chat_template(conversation, tokenize=True, return_dict=True, return_assistant_tokens_mask = True)
            image_inputs, video_inputs = process_vision_info(conversation)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",   
            )
            assistant_start = torch.nonzero(inputs['input_ids'][0] == self.processor.tokenizer.encode("assistant")[0]).squeeze()
            assert assistant_start.numel() == 1, "only one assistant token is supported."
            assistant_start_indices = assistant_start.item() + 2
            im_end = torch.nonzero(inputs['input_ids'][0] == self.processor.tokenizer.encode("<|im_end|>")[0]).squeeze()
            assistant_end_indices = im_end[-1].item() + 1

            labels = torch.ones_like(inputs['input_ids']) * IGNORE_INDEX
            labels[:,assistant_start_indices:assistant_end_indices] = inputs['input_ids'][:,assistant_start_indices:assistant_end_indices]
            inputs.update({'labels': labels})
        except Exception as e:
            logger.exception(f"Error processing instance '{instance}': '{e}'. Resampling.")
            return self.__getitem__(random.randint(0, len(self.instances) - 1))

        return inputs

    def __len__(self) -> int:
        return len(self.instances)
