from train.qwen2vl_trainer import Qwen2VLTrainer
from train.arguments import TrainingArgumentsCustom, DataArguments, ModelArguments
from train.utils import get_model
from src.data.builder import build_dataset


def train(train_args: TrainingArgumentsCustom, data_args: DataArguments, model_args: ModelArguments):

    model, tokenizer, processor = get_model(model_args=model_args, train_args=train_args)

