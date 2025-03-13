import datetime
import os
import sys
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)
from typing import List, Optional, Union, Tuple
from transformers import AutoTokenizer, AutoProcessor, AutoConfig
from qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from safetensors.torch import save_file, load_file


from torch.nn.parallel import DistributedDataParallel as DDP
from train.arguments import TrainingArgumentsCustom

from transformers import logging
def get_model(model_args, train_args: Optional[TrainingArgumentsCustom]) -> Tuple[Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor]:
    def load_safetensors_from_dir(dir_path):
        # 初始化一个空的 state_dict
        merged_state_dict = {}
        # 遍历指定目录下的所有文件
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                # 检查文件是否为 .safetensors 文件
                if file.endswith('.safetensors'):
                    file_path = os.path.join(root, file)
                    try:
                        # 加载当前 .safetensors 文件的 state_dict
                        state_dict = load_file(file_path)
                        # 将当前 state_dict 合并到 merged_state_dict 中
                        merged_state_dict.update(state_dict)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
        return merged_state_dict

    load_resume = model_args
    model_name_or_path = model_args.model_name_or_path
    config_name = model_args.config_name
    if config_name is None:
        config_name = model_name_or_path
    tokenizer_name_or_path = model_args.tokenizer_name
    if tokenizer_name_or_path is None:
        tokenizer_name_or_path = model_name_or_path
    processor_name_or_path = model_args.processor_name
    if processor_name_or_path is None:
        processor_name_or_path = model_name_or_path

    qwen2_vl_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=config_name)
    if train_args.bf16:
        dtype = torch.bfloat16
    elif train_args.fp16:
        dtype=torch.float16
    else:
        dtype=torch.float32
    qwen2_vl_config.torch_dtype = dtype

    if load_resume:
        # weights = load_safetensors_from_dir(dir_path=model_name_or_path)
        # model.load_state_dict(state_dict=weights, strict=False)
        model = Qwen2VLForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
        model.to(dtype)
    else:
        model = Qwen2VLForConditionalGeneration(config=qwen2_vl_config)
        model.to(dtype)
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_name_or_path)
    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=processor_name_or_path)
    return model, tokenizer, processor


def get_optimizer(model: Union[torch.nn.Module, DDP], train_args: TrainingArgumentsCustom):
    freeze_llm = train_args.freeze_llm
    llm_learning_rate = train_args.llm_learning_rate

    freeze_encoder = train_args.freeze_encoder
    encoder_learning_rate = train_args.encoder_learning_rate

    freeze_projector = train_args.freeze_projector
    projector_learning_rate = train_args.projector_learning_rate

    freeze_compressor = train_args.freeze_compressor
    compressor_learning_rate = train_args.compressor_learning_rate

    encoder_params = []
    projector_params = []
    compressor_params = []
    llm_params = []

    if hasattr(model, "module"):
        model = model.module
    for name, param in model.named_parameters():
        if "merger" in name:
            if freeze_projector:
                param.requires_grad = False
            else:
                projector_params.append(param)
        elif "visual" in name:
            if freeze_encoder:
                param.requires_grad = False
            else:
                encoder_params.append(param)
        elif "compressor" in name:
            if freeze_compressor:
                param.requires_grad= False
            else:
                compressor_params.append(param)
        else:
            if freeze_llm:
                param.requires_grad = False
            else:
                llm_params.append(param)
    
    optimizer_grouped_parameters =[
        {"params": encoder_params, "lr": encoder_learning_rate},
        {"params": projector_params, "lr": projector_learning_rate},
        {"params": compressor_params, "lr": compressor_learning_rate},
        {"params": llm_params, "lr": llm_learning_rate}
    ]
    optimizer = AdamW(optimizer_grouped_parameters)

    lr_scheduler = []
    if len(encoder_params) > 0:
        lr_scheduler.append(CosineAnnealingLR(optimizer=optimizer, T_max=train_args.encoder_step_max, eta_min=encoder_learning_rate / train_args.encoder_step_max))
    if len(projector_params) > 0:
        lr_scheduler.append(CosineAnnealingLR(optimizer=optimizer, T_max=train_args.projector_step_max, eta_min=projector_learning_rate / train_args.projector_step_max))
    if len(compressor_params) > 0:
        lr_scheduler.append(CosineAnnealingLR(optimizer=optimizer, T_max=train_args.compressor_step_max, eta_min=compressor_learning_rate / train_args.compressor_step_max))
    if len(llm_params) > 0:
        lr_scheduler.append(CosineAnnealingLR(optimizer=optimizer, T_max=train_args.llm_step_max, eta_min=llm_learning_rate / train_args.llm_step_max))
    return optimizer, lr_scheduler