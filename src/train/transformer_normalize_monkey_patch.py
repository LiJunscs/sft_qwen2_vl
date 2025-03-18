# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import transformers
from transformers.image_transforms import (
    ChannelDimension,
    Iterable,
    Optional,
    Union,
    get_channel_dimension_axis,
    infer_channel_dimension_format,
    np,
    to_channel_dimension_format,
)


import os

import torch
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import logging, is_apex_available

if is_apex_available():
    from apex import amp

TRAINER_STATE_NAME = "trainer_state.json"
logger = logging.get_logger(__name__)


def _save_checkpoint(self, model, trial, metrics=None):
    # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
    # want to save except FullyShardedDDP.
    # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

    # Save model checkpoint
    checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

    if self.hp_search_backend is None and trial is None:
        self.store_flos()

    run_dir = self._get_output_dir(trial=trial)
    output_dir = os.path.join(run_dir, checkpoint_folder)

    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        logger.warning(
            f"Checkpoint destination directory {output_dir} already exists and is non-empty."
            "Saving will proceed but saved results may be invalid."
        )
        staging_output_dir = output_dir
    else:
        staging_output_dir = os.path.join(run_dir, f"tmp-{checkpoint_folder}")

    self.save_model(staging_output_dir, _internal_call=True)

    if not self.args.save_only_model:
        # Save optimizer and scheduler
        self._save_optimizer_and_scheduler(staging_output_dir)
        # Save RNG state
        self._save_rng_state(staging_output_dir)

    # Determine the new best metric / best model checkpoint
    if metrics is not None and self.args.metric_for_best_model is not None:
        metric_to_check = self.args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics[metric_to_check]

        operator = np.greater if self.args.greater_is_better else np.less
        if (
            self.state.best_metric is None
            or self.state.best_model_checkpoint is None
            or operator(metric_value, self.state.best_metric)
        ):
            self.state.best_metric = metric_value
            self.state.best_model_checkpoint = staging_output_dir

    # Save the Trainer state
    if self.args.should_save:
        self.state.save_to_json(os.path.join(staging_output_dir, TRAINER_STATE_NAME))

    if self.args.push_to_hub:
        self._push_from_checkpoint(staging_output_dir)

    torch.distributed.barrier()
    if staging_output_dir != output_dir:
        with self.args.main_process_first(
            desc="Renaming model checkpoint folder to true location", local=self.args.save_on_each_node
        ):
            if os.path.exists(staging_output_dir):
                os.rename(staging_output_dir, output_dir)

    # Maybe delete some older checkpoints.
    if self.args.should_save:
        # Solely rely on numerical checkpoint id for rotation.
        # mtime is not reliable especially on some fuse fs in cloud environments.
        self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)


from typing import Any, Dict, Union

from torch import nn
from transformers.training_args import OptimizerNames
from transformers.utils import (
    is_sagemaker_mp_enabled,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)


def training_step(
    self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
) -> torch.Tensor:
    """
    Perform a training step on a batch of inputs.
    Subclass and override to inject custom behavior.
    Args:
        model (`nn.Module`):
            The model to train.
        inputs (`Dict[str, Union[torch.Tensor, Any]]`):
            The inputs and targets of the model.
            The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
            argument `labels`. Check your model's documentation for all accepted arguments.
    Return:
        `torch.Tensor`: The tensor with training loss on this batch.
    """
    ## NOTE: 根据当前step设置随机数种子，后续随机选择当前batch是否进行compress
    if self.state.global_step % (self.args.save_steps // 4) == 0:
        torch.manual_seed(self.state.global_step)
    model.train()
    if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
        self.optimizer.train()

    inputs = self._prepare_inputs(inputs)
    compress = torch.rand(1).item() >= 0.5
    inputs.update({
        "compress": compress
    })
    # if is_sagemaker_mp_enabled():
    #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
    #     return loss_mb.reduce_mean().detach().to(self.args.device)

    with self.compute_loss_context_manager():
        loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

    del inputs
    if (
        self.args.torch_empty_cache_steps is not None
        and self.state.global_step % self.args.torch_empty_cache_steps == 0
    ):
        if is_torch_xpu_available():
            torch.xpu.empty_cache()
        elif is_torch_mlu_available():
            torch.mlu.empty_cache()
        elif is_torch_musa_available():
            torch.musa.empty_cache()
        elif is_torch_npu_available():
            torch.npu.empty_cache()
        elif is_torch_mps_available(min_version="2.0"):
            torch.mps.empty_cache()
        else:
            torch.cuda.empty_cache()

    kwargs = {}

    # For LOMO optimizers you need to explicitly use the learnign rate
    if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
        kwargs["learning_rate"] = self._get_learning_rate()

    if self.args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel training

    if self.use_apex:
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        if num_items_in_batch is not None:
            if self.compute_loss_func or self.model_accepts_loss_kwargs:
                loss *= self.args.gradient_accumulation_steps
            # Average tokens across devices is orthogonal to gradient accumulation
            loss *= self.args.world_size
        self.accelerator.backward(loss, **kwargs)

    return loss.detach() / self.args.gradient_accumulation_steps


def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    """
    How the loss is computed by Trainer. By default, all models return the loss in the first element.
    Subclass and override for custom behavior.
    """
    if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
        labels = inputs.pop("labels")
    else:
        labels = None
    if num_items_in_batch is not None:
        num_items_in_batch_tensor = torch.tensor(num_items_in_batch, device=self.args.device)
        num_items_in_batch = int(self.accelerator.gather(num_items_in_batch_tensor).sum().cpu())
    if self.model_accepts_loss_kwargs:
        loss_kwargs = {}
        if num_items_in_batch is not None:
            loss_kwargs["num_items_in_batch"] = num_items_in_batch
        inputs = {**inputs, **loss_kwargs}
    outputs = model(**inputs)
    ## NOTE: 由于随机进行compress的原因，所以可能部分参数不存在梯度，导致DDP出错，这里将不存在梯度的参数的梯度设置为0。不过这个是否真的起到作用存疑
    # 遍历优化器中的所有参数组
    for param_group in self.optimizer.param_groups:
        # 遍历每个参数组中的参数
        for param in param_group['params']:
            # 检查参数是否需要梯度且梯度为 None
            if param.requires_grad and param.grad is None:
                    # 手动将梯度设置为相同形状的零张量
                    param.grad = torch.zeros_like(param)
    # Save past state if it exists
    # TODO: this needs to be fixed and made cleaner later.
    if self.args.past_index >= 0:
        self._past = outputs[self.args.past_index]

    if labels is not None:
        unwrapped_model = self.accelerator.unwrap_model(model)
        # if _is_peft_model(unwrapped_model):
        #     model_name = unwrapped_model.base_model.model._get_name()
        # else:
        #     model_name = unwrapped_model._get_name()
        model_name = unwrapped_model._get_name()
        # User-defined compute_loss function
        ## NOTE: 由于存在compress，所以input_ids的长度可能会变化，对应labels跟着改变
        seqlen = outputs.logits.shape[1]
        labels = labels[:, -seqlen:]
        if self.compute_loss_func is not None:
            loss = self.compute_loss_func(outputs.logits, labels, num_items_in_batch=num_items_in_batch)
        # elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        #     loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            loss = self.label_smoother(outputs, labels)
    else:
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
    print(f"lm loss: {loss:.5f}")
    return (loss, outputs) if return_outputs else loss


