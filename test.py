from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from train.data.data_preprocess import process_data
import torch

if __name__ == "__main__":
    model = Qwen2VLForConditionalGeneration.from_pretrained("/home/lijun2/multimodal/checkpoints/Qwen2-VL-7B-Instruct", torch_dtype=torch.float16, device_map="auto", attn_implementation="flash_attention_2").eval()
    qwen_processor = AutoProcessor.from_pretrained("/home/lijun2/multimodal/checkpoints/Qwen2-VL-7B-Instruct")
    contexts = ["Hello", "Who are you"]
    visuals = ["/home/lijun2/multimodal/dataset/video-mme/videomme/data/zxKPjD8urG4.mp4", "/home/lijun2/multimodal/dataset/video-mme/videomme/data/ZXoaMa6jlO4.mp4"]
    inputs = process_data(contexts=contexts, visuals=visuals, processor=qwen_processor)
    inputs = inputs.to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=32, use_cache=True, output_attentions=False, output_hidden_states=False)
    print()