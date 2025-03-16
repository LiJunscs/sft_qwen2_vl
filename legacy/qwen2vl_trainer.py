from transformers.trainer import ALL_LAYERNORM_LAYERS  # ShardedDDPOption,
from transformers.trainer import get_parameter_names, has_length, is_sagemaker_mp_enabled, logger
from transformers import Trainer
from typing import Optional, Dict
import torch
import os
import torch.distributed as dist
import json
import torch.nn as nn
class Qwen2VLTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = True
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels", None)
        if hasattr(model, "module"):
            model = model.module
        input_ids = inputs["input_ids"]
        cache_position = torch.arange(0, input_ids.shape[1], device=input_ids.device)
        inputs = model.prepare_inputs_for_generation(**inputs, use_cache=False, cache_position=cache_position)
        inputs.update({
            "labels": labels,
            "compress": True
        })
        outputs = model(**inputs, output_attentions=False, output_hidden_states=False)
        loss = outputs.loss
        return loss

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
        #     return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            elif self.args.vision_tower_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "vision_tower" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.vision_tower_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def save_model(self, output_dir: Optional[str], _internal_call: bool):
        ## save tuned model separately
        if self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(self.deepspeed)
        else:
            # TODO(ligeng): fix save_model for multi-node training on large models (e.g., Llama-70b)
            state_dict = self.model.state_dict()

        # if self.args.lora_enable:
        #     non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters())
        #     os.makedirs(output_dir, exist_ok=True)
        #     torch.save(
        #         non_lora_state_dict,
        #         os.path.join(output_dir, "non_lora_trainables.bin"),
        #     )
        #     # config
        #     self.model._name_or_path = output_dir
        #     self.model.architectures = [self.model.__class__.__name__]
        #     self.model.config.save_pretrained(output_dir)

        if self.args.should_save:
            return self.model.save_pretrained(output_dir, state_dict=state_dict)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)

        if self.args.debug_e2e and self.control.should_training_stop:

            # Only save log history if the current process is rank 0
            if dist.get_rank() == 0:
                with open(f"{self.args.output_dir}/log_history.json", "w") as f:
                    json.dump(self.state.log_history, f, indent=4)

        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)


