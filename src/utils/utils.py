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
import os
from typing import Any, List
import torch.distributed as dist
from safetensors.torch import save_file, load_file

__all__ = ["make_list", "disable_torch_init"]


def make_list(obj: Any) -> List:
    return obj if isinstance(obj, list) else [obj]

def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def load_safetensors_from_dir(dir_path):
        merged_state_dict = {}

        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.safetensors'):
                    file_path = os.path.join(root, file)
                    try:
                        state_dict = load_file(file_path)
                        merged_state_dict.update(state_dict)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
        return merged_state_dict

