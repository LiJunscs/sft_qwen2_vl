import sys
sys.path.append('/home/yuanziqi/Work25/sft_qwen2_vl')

from data.dataset import make_supervised_data_module
from src.train.args import DataArguments, ModelArguments, TrainingArguments
from transformers import HfArgumentParser

from transformers import AutoTokenizer, AutoProcessor

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    data_args.image_processor = AutoProcessor.from_pretrained('/data/public/multimodal/yuanziqi/models/Qwen2.5-VL-7B-Instruct')

    processor = AutoProcessor.from_pretrained('/data/public/multimodal/yuanziqi/models/Qwen2.5-VL-7B-Instruct')
    dataset = make_supervised_data_module(processor=processor, data_args=data_args, training_args=training_args)
    dataset, collator = dataset['train_dataset'], dataset['data_collator']

    res = dataset[0]
    res_v = dataset[-1]


    print(11111)

