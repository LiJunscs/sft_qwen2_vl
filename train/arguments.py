# 定义模型参数类
from dataclasses import dataclass, field
from typing import List
from transformers import TrainingArguments

@dataclass
class ModelArguments:
    """
    模型相关的参数
    """
    model_name_or_path: str = field(
        default="qwen2_vl",
        metadata={"help": "预训练模型的名称或路径"}
    )
    config_name: str = field(
        default=None,
        metadata={"help": "配置文件的名称或路径，如果未指定则使用 model_name_or_path"}
    )
    tokenizer_name: str = field(
        default=None,
        metadata={"help": "分词器的名称或路径，如果未指定则使用 model_name_or_path"}
    )

    processor_name: str = field(
        default=None,
        metadata={"help": "处理器的名称或路径，如果未指定则使用 model_name_or_path"}
    )

    load_resume: bool = field(
        default=True,
        metadata={"help": "从一个预训练好的模型加载"}
    )

    max_new_tokens: int = field(
        default=512,
    )


# 定义训练参数类
@dataclass
class TrainingArgumentsCustom(TrainingArguments):
    """
    训练相关的参数
    """

    encoder_learning_rate: float = field(
        default=5e-5,
        metadata={"help": "encoder学习率"}
    )

    encoder_step_max: int = field(
        default=5000,
        metadata={"help": "余弦学习率调度器执行步骤"}
    )

    freeze_encoder: bool = field(
        default=True
    )

    projector_learning_rate: float = field(
        default=5e-5,
        metadata={"help": "projector学习率"}
    )

    projector_step_max: int = field(
        default=5000,
        metadata={"help": "余弦学习率调度器执行步骤"}
    )

    freeze_projector: bool = field(
        default=False
    )

    compressor_learning_rate: float = field(
        default=5e-5,
        metadata={"help": "compressor学习率"}
    )

    compressor_step_max: int = field(
        default=5000,
        metadata={"help": "余弦学习率调度器执行步骤"}
    )

    freeze_compressor: bool = field(
        default=False
    )

    llm_learning_rate: float = field(
        default=5e-5,
        metadata={"help": "llm学习率"}
    )

    llm_step_max: int = field(
        default=5000,
        metadata={"help": "余弦学习率调度器执行步骤"}
    )
    freeze_llm: bool = field(
        default=True
    )

    pretrain: bool = field(
        default=True
    )

    just_debug: bool = field(
        default=False
    )

    sample_lens: int = field(
        default=20,
    )



# 定义数据参数类
@dataclass
class DataArguments:
    """
    数据相关的参数
    """
    data_dir: List[str] = field(
        default_factory=list,
        metadata={"help": "数据文件的路径列表"}
    )

    data_type: str = field(
        default="parquet",
        metadata={"help": "数据集文件的类型, 比如json, parquet"}
    )
    prompt_file: str = field(
        default="dataset.json",
        metadata={"help": "prompt信息存放的文件名"}
    )
    image_dir: str = field(
        default="./images",
        metadata={"help": "图片的存放路径"}
    )

    video_dir: str = field(
        default="./videos",
        metadata={"help": "视频的存放路径"}
    )


    image_pre_prompt: str = field(
        default="Please describe the image.\n",
    )

    video_pre_prompt: str = field(
        default="Please describe the video.\n"
    )

    post_prompt: str = field(
        default="",
    )


    enhance: bool = field(
        default=False,
        metadata={"help": "对图片进行数据增强"}
    )

    repeat: int = field(
        default=1,
        metadata={"help": "是否对单张图片进行复制"}
    )

    data_mixture: str = "videochat_flash_pretrain"

    num_video_frames: int = field(
        default=256
    )

    max_pixels: int = field(
        default=200476
    )

    fps: float = 0.0  # 0.0 means we do not use fps at all. Always sample the same number of frames.