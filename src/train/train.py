import os
import torch
import transformers

import pathlib

from functools import partial
from transformers import AutoConfig, AutoProcessor, set_seed
from transformers.loss.loss_utils import ForCausalLMLoss
from safetensors.torch import save_file, load_file

from src.data import make_supervised_data_module
from src.constants import IGNORE_INDEX
from src.utils.utils import rank0_print
from src.train.trainer import CustomTrainer
from src.model.projector import Projector
from src.model.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from src.train.args import DataArguments, ModelArguments, TrainingArguments

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

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir, _internal_call=True)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def train(attn_implementation="flash_attention_2"):

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    if training_args.verbose_logging:
        rank0_print(f"Inspecting experiment hyperparameters:\n")
        rank0_print(f"model_args = {vars(model_args)}\n")
        rank0_print(f"data_args = {vars(data_args)}\n")
        rank0_print(f"training_args = {vars(training_args)}\n")

    # local_rank = training_args.local_rank 
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    set_seed(training_args.seed)

    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    model_config._attn_implementation = model_config.vision_config._attn_implementation = attn_implementation
    model = Qwen2_5_VLForConditionalGeneration(model_config)
    #TODO. mock patch model.visual.merger here.
    mock_projector = Projector({
        'projector_cls': 'dummy', 
        'kwargs': {
            'context_dim': model_config.vision_config.hidden_size, 
            'dim': model_config.vision_config.out_hidden_size, 
            'spatial_merge_size': model_config.vision_config.spatial_merge_size
        }
    })
    model.visual.merger = mock_projector
    model.visual.merger.initialize_model()

    weights = load_safetensors_from_dir(dir_path=model_args.model_name_or_path)
    model.load_state_dict(state_dict=weights, strict=False, assign=True)
    
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     model_args.model_name_or_path,
    #     torch_dtype=compute_dtype,
    #     attn_implementation=attn_implementation,
    #     # device_map="auto",
    # )

    rank0_print(model)

    processor = AutoProcessor.from_pretrained(model_args.processor_name_or_path)
    
    
    # model = get_model(model_args, training_args, bnb_model_from_pretrained_args)
    # model.config.use_cache = False
    # if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
    #     model.config.rope_scaling = {
    #         "factor": model_args.rope_scaling_factor,
    #         "type": model_args.rope_scaling_type,
    #     }

    # if model_args.freeze_backbone:
    #     model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        # TODO. 
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


    ### Deciding train which part of the model
    mm_tunable_parts = []
    for attr_name, attr_value in training_args.__dict__.items():
        if attr_name.startswith('tune_') and attr_value:
            mm_tunable_parts.append(attr_name.replace('tune_', ''))
    training_args.mm_tunable_parts = ",".join(mm_tunable_parts)
   
    rank0_print(f"Using tunable_parts: {training_args.mm_tunable_parts}")
    # Set the entire model to not require gradients by default
    model.requires_grad_(False)
    # Parse the mm_tunable_parts to decide which parts to unfreeze
    projector_param_name, vm_param_name, lm_param_name = [], [], []
    if "projector" in mm_tunable_parts:
        for name, param in model.named_parameters():
            if "visual" in name and "merger" in name:
                param.requires_grad_(True)
                projector_param_name.append(name)
    if "vision_tower" in mm_tunable_parts:
        for name, param in model.named_parameters():
            if "visual" in name and "merger" not in name:
                param.requires_grad_(True)
                vm_param_name.append(name)
    if "language_model" in mm_tunable_parts:
        for name, param in model.named_parameters():
            if "visual" not in name:
                param.requires_grad_(True)
                lm_param_name.append(name)
    
    training_args.projector_param_name = projector_param_name
    training_args.vm_param_name = vm_param_name
    training_args.lm_param_name = lm_param_name
    total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
    trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(f"Total parameters: ~{total_params/1e6:.2f} MB)")
    rank0_print(f"Trainable parameters: ~{trainable_params/1e6:.2f} MB)")

    data_module = make_supervised_data_module(processor=processor, data_args=data_args, training_args=training_args)
    compute_loss_func = partial(ForCausalLMLoss, vocab_size=model.config.vocab_size, ignore_index=IGNORE_INDEX)
    trainer = CustomTrainer(model=model, processing_class=processor, compute_loss_func=compute_loss_func, args=training_args, **data_module)

    rank0_print(f"model_config after before train: {model.config}")

    rank0_print(
        "length of dataloader:",
        len(trainer.get_train_dataloader()),
        len(trainer.train_dataset),
    )
    rank0_print(
        "[GPU memory] before trainer",
        torch.cuda.memory_allocated() / 1024 / 1024 / 1024,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
