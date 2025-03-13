import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import os
from typing import Optional
import json
from PIL import Image, ImageEnhance
from tqdm import tqdm
import random

class CustomDataset(Dataset):
    def __init__(self, data_args):
        self.data_dirs = data_args.data_dir  # 现在 data_dir 是一个列表
        self.prompt_file = data_args.prompt_file
        self.image_dir = data_args.image_dir
        self.video_dir = data_args.video_dir
        self.image_pre_prompt = data_args.image_pre_prompt
        self.video_pre_prompt = data_args.video_pre_prompt
        self.post_prompt = data_args.post_prompt

        self.data = []
        for data_dir in self.data_dirs:
            prompt_file_path = os.path.join(data_dir, self.prompt_file)
            image_dir_path = os.path.join(data_dir, self.image_dir)
            video_dir_path = os.path.join(data_dir, self.video_dir)

            if data_args.data_type == "json":
                with open(prompt_file_path, "r") as f:
                    data = json.load(f)
            elif data_args.data_type == "parquet":
                data = load_dataset(data_dir)
            else:
                raise NotImplementedError("No support yet data type")

            # 截取前100个样本用于调试
            if data_args.debug_dataset:
                debug_dataset = {"train": None}
                debug_dataset["train"] = [data["train"][i] for i in range(20)]
                data = debug_dataset

            if data_args.num_samples > 0:
                train_data_len = len(data["train"])
                sample_data = {"train": []}
                num_sample = min(train_data_len, data_args.num_samples)
                pbar = tqdm(range(0, train_data_len, train_data_len // num_sample), "Sample datasets")
                sample_data["train"] = [data["train"][i] for i in pbar]
                data = sample_data

            pbar = tqdm(range(len(data["train"])), desc=f"Loading dataset from {data_dir}")
            for item in data["train"]:
                visual_path = None
                if "image" in item:
                    if isinstance(item["image"], str):
                        visual_path = os.path.join(image_dir_path, item["image"])
                        visual_path = Image.open(visual_path)
                        if data_args.repeat > 1:
                            visual_list = [visual_path]
                            visual_copy = [visual_path.copy() for _ in range(data_args.repeat - 1)]
                            if data_args.enhance:
                                for i in range(1, data_args.repeat - 1):
                                    img = visual_copy[i]
                                    # 随机调整对比度（0.8 到 1.2 之间）
                                    contrast_factor = random.uniform(0.8, 1.2)
                                    contrast_enhancer = ImageEnhance.Contrast(img)
                                    img = contrast_enhancer.enhance(contrast_factor)

                                    # 随机调整饱和度（0.7 到 1.3 之间）
                                    saturation_factor = random.uniform(0.7, 1.3)
                                    color_enhancer = ImageEnhance.Color(img)
                                    img = color_enhancer.enhance(saturation_factor)
                                    visual_copy[i] = img
                                visual_path = visual_list + visual_copy

                    elif isinstance(item["image"], Image.Image):
                        visual_path = item["image"]
                elif "video" in item:
                    video_name = item["video"]
                    if not video_name.endswith(".mp4"):
                        video_name += ".mp4"
                    visual_path = os.path.join(video_dir_path, video_name)
                if visual_path is not None and not isinstance(visual_path, list):
                    visual_path = [visual_path]

                text = []
                labels = []
                answer = []
                for conv in item["conversations"]:
                    role = conv["from"]
                    if role == "human":
                        if "image" in item:
                            text.append(self.image_pre_prompt + conv["value"] + self.post_prompt)
                        elif "video" in item:
                            text.append(self.video_pre_prompt + conv["value"] + self.post_prompt)
                    elif role == "gpt":
                        answer.append(conv["value"])
                    else:
                        raise RuntimeError

                self.data.append({
                    "text": text,
                    "visuals": visual_path,
                    "labels": text,
                    "answer": answer
                })
                pbar.update(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]