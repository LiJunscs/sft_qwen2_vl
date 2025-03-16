import datetime
import os
import sys
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)
import re
import string
from typing import List, Optional, Union, Tuple
from transformers import AutoTokenizer, AutoProcessor, AutoConfig
from qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from accelerate import Accelerator
import argparse
import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
from transformers import Trainer, TrainingArguments
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from safetensors.torch import save_file, load_file

import torch.distributed as dist
from transformers import HfArgumentParser, PreTrainedModel
from dataclasses import dataclass, field
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from data.data_preprocess import process_data
from torch.nn import CrossEntropyLoss
from data.llava_rap import CustomDataset
from tqdm import tqdm

tran_dir = "/home/lijun2/multimodal/sft_qwen2_vl/"
if tran_dir not in sys.path:
    sys.path.append(tran_dir)
from transformers import logging
from train.arguments import TrainingArgumentsCustom, ModelArguments, DataArguments
from train.utils import get_model, get_optimizer
from src.data.dataset import make_supervised_data_module

# 获取 logger 实例
logger = logging.get_logger(__name__)


def get_dataset(data_args, training_args, processor):
    return make_supervised_data_module(processor=processor, data_args=data_args, training_args=training_args)


def save_model(model: Union[torch.nn.Module, DDP, PreTrainedModel], optimizer: torch.optim.Optimizer, iter: int = -1, ckpt_name: Optional[str] = "ckpt", output_dir: Optional[str] = None, train_args: Optional[TrainingArgumentsCustom] = None):
    output_dir = train_args.output_dir
    if output_dir is None or output_dir == "":
        output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    if hasattr(model, "module"):
        model = model.module
    # 获取模型的所有参数
    all_params = list(model.named_parameters())
    num_params = len(all_params)
    # 假设我们将参数均匀分成 4 块，可根据需要调整
    num_chunks = 4
    chunk_size = num_params // num_chunks

    if iter == -1:
        iter = "final"
    else:
        iter = f"iter{iter}"
    save_dir = f"{ckpt_name}_{iter}"
    
    save_dir = os.path.join(output_dir, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    model.save_pretrained(save_dir, max_shard_size="5GB", safe_serialization=True)

    # 保存模型配置
    model.config.save_pretrained(save_dir)
    print_rank_0(f"Save checkpoint of iter {iter} to directory {save_dir}")

    # 获取优化器状态字典
    # optimizer_state = optimizer.state_dict()

    # 保存优化器状态为 safetensors 文件
    # save_opt_file = os.path.join(save_dir, 'optimizer_state.safetensors')
    # save_file(optimizer_state, save_opt_file)


def setup():
    """
    初始化分布式环境
    :param rank: 当前进程的全局排名
    :param world_size: 总的进程数
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # 初始化进程组
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)


def cleanup():
    """
    销毁进程组
    """
    dist.destroy_process_group()

def collator_fn(batch):
    new_batch = {"text": [], "visuals": [], "answer": []}
    for item in batch:
        new_batch["text"].append(item["text"])
        new_batch["visuals"].append(item["visuals"])
        new_batch["answer"].append(item["answer"])
    return new_batch

def pretrain_loss_func(logits: torch.Tensor, labels: torch.Tensor, vocab_size: int):
    loss = None
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss

def sft_loss_func(logits: torch.Tensor, labels: torch.Tensor, vocab_size: int):
    pass

def train(train_args: TrainingArgumentsCustom, model_args, data_args):
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    model, tokenizer, processor = get_model(model_args=model_args, train_args=train_args)
    dataset = get_dataset(data_args, train_args, processor)
    dataset, collator = dataset['train_dataset'], dataset['data_collator']

    # 创建分布式采样器
    train_sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank
    )

    train_dataloader = DataLoader(
        dataset,
        sampler=train_sampler,
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=collator,
        drop_last=True,
        pin_memory=True
    )
    
    model.to(rank)
    optimizer, lr_schedulers = get_optimizer(model, train_args)
    ddp_model = DDP(module=model, device_ids=[rank])


    epochs = int(train_args.num_train_epochs)
    grad_accumulation_steps = int(train_args.gradient_accumulation_steps)
    data_len = len(train_dataloader)


    ddp_save_step = train_args.save_steps // world_size
    # train
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        ddp_model.train()
        for i, item in enumerate(train_dataloader):
            cache_position = torch.arange(0, item.input_ids.shape[1], device=rank)
            labels = item.labels
            # prefill 
            inputs = ddp_model.module.prepare_inputs_for_generation(**item, use_cache=True, cache_position=cache_position)
            inputs.update({
                "labels": labels,
                "compress": True
            })
            outputs = ddp_model(**inputs, output_attentions=False, output_hidden_states=False)
            logits = outputs.logits
            loss = outputs.loss

            loss.backward()
            if (i + 1) % grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), train_args.max_grad_norm)
                optimizer.step()
                for lr_scheduler in lr_schedulers:
                    lr_scheduler.step()
                optimizer.zero_grad()
            if (i + 1) % train_args.logging_steps == 0:
                log_str = f"[Epoch {epoch}/{epochs}, Iter {i}/{data_len}: loss: {loss.item()} | grad_norm: {train_args.max_grad_norm}"
                print_with_rank_and_time(log_str)
            if (i + 1) % ddp_save_step == 0 and rank == 0 and not train_args.just_debug:
                model = ddp_model.module
                save_model(model=model, optimizer=optimizer, iter=i + 1)

    cleanup()
    save_model(model=ddp_model.module, optimizer=optimizer, train_args=train_args)
    tokenizer.save_pretrained(train_args.output_dir)
    processor.save_pretrained(train_args.output_dir)
    logger.info("Training success ...")


def print_args(args, args_type: Optional[str]=None):
    """
    以类似Megatron-LM的格式打印参数
    """
    print(f'------------------------ {args_type} arguments  ------------------------')
    for arg in sorted(vars(args)):
        dots = '.' * (48 - len(arg))
        print(f'{arg} {dots} {getattr(args, arg)}')
    print(f'-------------------- {args_type} end of arguments  ---------------------')

def print_rank_0(message):
    """
    如果是分布式训练，仅在rank为0的进程中打印消息；如果不是分布式训练，则直接打印消息。
    """
    if torch.distributed.is_initialized():
        if dist.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def print_with_rank_and_time(message):
    """
    该函数在各个 rank 进程中输出包含当前时间和 rank 数的信息。
    :param message: 要输出的额外消息
    """
    # 检查分布式训练是否初始化
    if dist.is_initialized():
        # 获取当前进程的 rank 数
        rank = dist.get_rank()
    else:
        rank = 0

    # 获取当前时间
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 构造要输出的完整消息
    full_message = f"[{current_time}] [Rank {rank}]: {message}"
    print(full_message, flush=True)

if __name__ == "__main__":
    # 创建参数解析器
    parser = HfArgumentParser((ModelArguments, TrainingArgumentsCustom, DataArguments))

    # 解析命令行参数
    model_args, training_args, data_args = parser.parse_args_into_dataclasses()
    # 设置日志级别为 INFO
    logging.set_verbosity_debug()

    print_rank_0("Prepare arguments for training...")
    # 打印解析后的参数
    print_args(model_args, "Model")
    print_args(training_args, "Train")
    print_args(data_args, "Data")
    train(training_args, model_args, data_args)
