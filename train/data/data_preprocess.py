import base64
from io import BytesIO
import decord
from PIL import Image
from qwen_vl_utils import process_vision_info
import numpy as np
import torch
from transformers import AutoProcessor
from typing import List, Union, Optional


def process_data(contexts: List[str], visuals: List[Union[Image.Image, str]], processor: AutoProcessor, answers: Optional[List[str]] = None, max_pixels: int = 200476, max_num_frames: int = 256, sft: bool = False):
    if sft:
        assert answers is not None, "SFT need answers for assistant prompt"
    messages = []
    for i, context in enumerate(contexts):
        if "<image>" in context:
            context = context.replace("<image>", "")

        message = [{"role": "system", "content": "You are a helpful assistant."}]
        if len(visuals) > 0:
            visual = visuals[i] if i < len(visuals) else None
            if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                vr = decord.VideoReader(visual)
                first_frame = vr[0].asnumpy()
                height, width = first_frame.shape[:2]
                # max_pixels = height * width
                message.append({"role": "user", "content": [{"type": "video", "video": visual, "max_pixels": max_pixels}, {"type": "text", "text": context}]})
            elif isinstance(visual, Image.Image):  # Single image
                base64_image = visual.convert("RGB")
                buffer = BytesIO()
                base64_image.save(buffer, format="JPEG")
                base64_bytes = base64.b64encode(buffer.getvalue())
                base64_string = base64_bytes.decode("utf-8")
                message.append({"role": "user", "content": [{"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"}, {"type": "text", "text": context}]})
            elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  # Multiple images
                image_content = []
                for v in visual:
                    base64_image = v.convert("RGB")
                    buffer = BytesIO()
                    base64_image.save(buffer, format="JPEG")
                    base64_bytes = base64.b64encode(buffer.getvalue())
                    base64_string = base64_bytes.decode("utf-8")
                    image_content.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"})
                message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
            else:
                message.append({"role": "user", "content": [{"type": "text", "text": context}]})
        else:
            message.append({"role": "user", "content": [{"type": "text", "text": context}]})
        if answers is not None:
            answer = answers[i]
            if "<image>" in answer:
                answer = answer.replace("<image>", "")
            assistant = {"role": "assistant", "content": answer}
            message.append(assistant)
            messages.append(message)
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                if total_frames > max_num_frames:
                    indices = np.linspace(0, total_frames - 1, max_num_frames, dtype=int)
                    # Append the last frame index if not already included
                    if total_frames - 1 not in indices:
                        indices = np.append(indices, total_frames - 1)
                    video_inputs[0] = video_inputs[0][indices]
    inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    if sft:
        labels_mask = get_assistant_mask(input_ids=inputs.input_ids, processor=processor)
        labels = inputs.input_ids.clone()
        labels[labels_mask] = -100  # hard code, because I forget the ignore token
        labels = labels.contiguous()
        inputs.update({
            "labels": labels
        })
    return inputs

def get_assistant_mask(input_ids: torch.Tensor, processor: AutoProcessor):
    batch, seqlen = input_ids.shape
    assert batch == 1, "Only support batch size 1 yet"
    im_start = processor.tokenizer.encode("<|im_start|>")[0]
    im_end = processor.tokenizer.encode("<|im_end|>")[0]

    im_start_indices = torch.nonzero(input_ids[0] == im_start).squeeze()
    if len(im_start_indices) < 2:
        return torch.ones(batch, seqlen, device=input_ids.device, dtype=torch.bool) 
    im_start_assistance = im_start_indices[-2].item()

    im_end_indices = torch.nonzero(input_ids[0] == im_end).squeeze()
    if len(im_end_indices) < 2:
        return torch.ones(batch, seqlen, device=input_ids.device, dtype=torch.bool) 
    im_end_assistance = im_end_indices[-1].item()

    mask = torch.ones(batch, seqlen, device=input_ids.device, dtype=torch.bool)
    mask[:, im_start_assistance : im_end_assistance + 1] = False
    return mask