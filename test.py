from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
from train.data.data_preprocess import process_data, get_assistant_mask
import torch

def test_process_data():
    model = Qwen2VLForConditionalGeneration.from_pretrained("/home/lijun2/multimodal/checkpoints/Qwen2-VL-7B-Instruct", torch_dtype=torch.float16, device_map="auto", attn_implementation="flash_attention_2").eval()
    qwen_processor = AutoProcessor.from_pretrained("/home/lijun2/multimodal/checkpoints/Qwen2-VL-7B-Instruct")
    contexts = ["Hello", "Who are you", "Thank you", "Good luck"]
    img1 = Image.open("/data/public/multimodal/yuanziqi/datasets/pretraining_datasets/OCR-VQA-200K/images_o/B016X4I0JY.jpg")
    img2 = Image.open("/data/public/multimodal/yuanziqi/datasets/pretraining_datasets/OCR-VQA-200K/images_o/B0170JXKQE.jpg")
    visuals = [img1, img2, "/home/lijun2/multimodal/dataset/video-mme/videomme/data/zxKPjD8urG4.mp4", "/home/lijun2/multimodal/dataset/video-mme/videomme/data/ZXoaMa6jlO4.mp4"]
    inputs = process_data(contexts=contexts, visuals=visuals, processor=qwen_processor)
    inputs = inputs.to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=32, use_cache=True, output_attentions=False, output_hidden_states=False)
    print()

def test_get_assistant_mask():
    qwen_processor = AutoProcessor.from_pretrained("/home/lijun2/multimodal/checkpoints/Qwen2-VL-7B-Instruct")
    input_ids = torch.randint(0, 100, (2, 20), device="cuda", dtype=torch.int32)
    input_ids[0, [2, 10, 15]] = 151644
    input_ids[0, [5, 14]] = 151645
    input_ids[1, [3, 13, 17]] = 151644
    input_ids[1, [7, 16, ]] = 151645
    mask = get_assistant_mask(input_ids, qwen_processor)

if __name__ == "__main__":
    test_get_assistant_mask()