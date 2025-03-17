import sys
sys.path.append('/home/yuanziqi/Work25/sft_qwen2_vl')
import torch

from src.model.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from data.dataset import make_supervised_data_module
from src.train.args import DataArguments, ModelArguments, TrainingArguments
from transformers import HfArgumentParser
from qwen_vl_utils import process_vision_info
from transformers import AutoTokenizer, AutoProcessor
from torch.utils.data import DataLoader, RandomSampler
from transformers.loss.loss_utils import ForCausalLMLoss

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        '/data/public/multimodal/yuanziqi/models/Qwen2.5-VL-7B-Instruct',
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    data_args.image_processor = AutoProcessor.from_pretrained('/data/public/multimodal/yuanziqi/models/Qwen2.5-VL-7B-Instruct')

    processor = AutoProcessor.from_pretrained('/data/public/multimodal/yuanziqi/models/Qwen2.5-VL-7B-Instruct')
    dataset = make_supervised_data_module(processor=processor, data_args=data_args, training_args=training_args)
    dataset, collator = dataset['train_dataset'], dataset['data_collator']

    dataloader_params = {
        "batch_size": 4,
        "collate_fn": collator,
        "num_workers": 0,
        "pin_memory": True,
        "sampler": RandomSampler(dataset)
        # "persistent_workers": self.args.dataloader_persistent_workers,
    }

    dataloader = DataLoader(dataset, **dataloader_params)

    for i, inputs in enumerate(dataloader):

        res = model(
            input_ids=inputs.input_ids.cuda(),
            attention_mask=inputs.attention_mask.cuda(),
            pixel_values=inputs.pixel_values.cuda(),
            image_grid_thw=inputs.image_grid_thw.cuda(),
            pixel_values_videos=inputs.pixel_values_videos.cuda(),
            video_grid_thw=inputs.video_grid_thw.cuda(),
        )
        loss = ForCausalLMLoss(res.logits, inputs.labels, vocab_size=res.logits.shape[-1], num_items_in_batch = (inputs.labels!=-100).sum(), ignore_index=-100)
        print(1111)

    print(11111)

