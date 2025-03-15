import copy
# import glob
import os
import random
from typing import Any, Dict, List, Optional

# from src.constants import DEFAULT_IMAGE_TOKEN
from src.data.base import BaseDataset
from src.constants import MEDIA_TOKENS
from src.utils import io, make_list

# __all__ = ["LLaVADataset", "LLaVANextDataset", "LLaVANextVideoDataset"]
__all__ = ["LLaVADataset"]


# def _remove_media_tokens(text: str) -> str:
#     for token in ["<image>", "<video>"]:
#         text = text.replace(token + "\n", "").replace("\n" + token, "").replace(token, "")
#     return text.strip()

# def tokenize_conversation(
#     messages: Sequence[Dict[str, str]],
#     tokenizer: transformers.PreTrainedTokenizer,
#     add_generation_prompt: bool = False,
#     overrides: Optional[Dict[str, str]] = None,
#     no_system_prompt: bool = False,
# ) -> torch.Tensor:
#     # Normalize the conversation before tokenization
#     for message in messages:
#         message["value"] = message["value"].strip()

#     if conversation_lib.default_conversation.sep_style != conversation_lib.SeparatorStyle.AUTO:
#         return tokenize_conversation_legacy(
#             messages,
#             tokenizer,
#             add_generation_prompt=add_generation_prompt,
#             overrides=overrides,
#             no_system_prompt=no_system_prompt,
#         )

#     conversation = []
#     for m in messages:
#         message = {}
#         if m["from"] == "human":
#             message["role"] = "user"
#         elif m["from"] == "gpt":
#             message["role"] = "assistant"
#         else:
#             raise ValueError(f"Unexpected sender '{m['from']}' in conversation entry.")

#         message["content"] = m["value"]
#         if overrides is not None and m["from"] in overrides:
#             message["content"] = overrides[m["from"]]
#         conversation.append(message)

#     if no_system_prompt:
#         conversation = [{"role": "system", "content": None}] + conversation

#     text = tokenizer.apply_chat_template(
#         conversation,
#         add_generation_prompt=add_generation_prompt,
#         tokenize=False,
#     )
#     return tokenizer_image_token(text, tokenizer, return_tensors="pt")


class LLaVADataset(BaseDataset):
    def __init__(self, data_path: str, media_dir: Optional[str] = None, is_video=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.data_path = data_path
        self.media_dir = media_dir
        self.instances = io.load(self.data_path)
        global_batch_size = kwargs["global_batch_size"]
        self.is_video = is_video or any(["video" in instance for instance in self.instances])
        self.role_list = {'human': 'user', 'gpt': 'assistant'}
        # self.enable_dynamic_res = self.data_args.image_aspect_ratio == "dynamic" and not self.is_video
        # self.enable_dynamic_res_s2 = self.data_args.image_aspect_ratio == "dynamic_s2" and not self.is_video

        residual = global_batch_size - len(self.instances) % global_batch_size
        if residual != global_batch_size:
            if global_batch_size // len(self.instances) >= 2:
                self.instances = self.instances * (global_batch_size // len(self.instances))
                residual = global_batch_size - len(self.instances) % global_batch_size
            selected_elements = random.sample(range(len(self.instances)), residual)
            additional_instance = [self.instances[i] for i in selected_elements]
            self.instances.extend(additional_instance)

    def process_visual(self, visual_path):

        if self.is_video:
            kwargs = {'type': 'video', 'video': os.path.join(self.media_dir, visual_path)}
            if self.data_args.fps:
                kwargs.update({'fps': self.data_args.fps})
            else:
                assert self.data_args.num_video_frames, "fps and num_video_frames must be set."
                kwargs.update({'nframes': self.data_args.num_video_frames})
        else:
            kwargs = {'type': 'image', 'image': os.path.join(self.media_dir, visual_path)}

        visual_dict = {
            **kwargs,
            'max_pixels': self.data_args.max_pixels,
        }
        return visual_dict


    def process(self, instance: Dict[str, Any]) -> List[Dict[str, Any]]:
        
        # TODO. currently only support one-turn dialogues.
        conversation_count = len(instance['conversations'])
        assert conversation_count == 2, "currently only support one-turn dialogues."

        # TODO. Implement multiturn dialogue.

        image = instance.get('video', None) if self.is_video else instance.get('image', None)
        if image is not None:
            image_list = make_list(image)
            image_count = len(image_list)
            assert image_count == 1, "currently only support one-image per data."

        user_text = instance['conversations'][0]['value'].replace(MEDIA_TOKENS, '').strip()
        user_content = [
            self.process_visual(image),
            {"type": "text", "text": user_text}
        ]
        user_message = {
            'role': 'user',
            'content': user_content
        }
        assistant_message = {
            'role': 'assistant',
            'content': [{"type": "text", "text": instance['conversations'][1]['value']}]
        }
        messages = [user_message, assistant_message]
        return messages



# class LLaVANextDataset(BaseDataset):
#     def __init__(self, data_path: str, media_dir: str, is_video=False, **kwargs) -> None:
#         super().__init__(**kwargs)
#         self.data_path = data_path
#         self.media_dir = media_dir
#         self.instances = io.load(self.data_path)
#         self.is_video = is_video or any(["video" in instance for instance in self.instances])
#         self.enable_dynamic_res = self.data_args.image_aspect_ratio == "dynamic" and not self.is_video
#         self.enable_dynamic_res_s2 = self.data_args.image_aspect_ratio == "dynamic_s2" and not self.is_video

#     def process(self, instance: Dict[str, Any]) -> List[Dict[str, Any]]:
#         datasource = instance.get("datasource", None)
#         messages = instance["conversations"]

#         if "image" in instance:
#             img_list = []
#             for img_path in instance["image"]:
#                 img_list.append(Image(os.path.join(self.media_dir, img_path)))

#             # replace all <image> tokens in the messages
#             for idx1, msg in enumerate(messages):
#                 # value = messages[0]["value"]
#                 value = messages[idx1]["value"]
#                 img_tok_len = len(DEFAULT_IMAGE_TOKEN)
#                 new_value = []

#                 while value.find(DEFAULT_IMAGE_TOKEN) >= 0:  # still has <image>
#                     idx = value.find(DEFAULT_IMAGE_TOKEN)
#                     if idx > 0:
#                         new_value.append(value[:idx])
#                     new_value.append(img_list.pop(0))
#                     value = value[idx + img_tok_len :]
#                 new_value.append(value)
#                 messages[idx1]["value"] = new_value

#                 # FIXME(ligeng): this is an interesting bug... if we feed [{"from": "gpt"}, {"from": "user"}] to the model, it will throw errors.
#                 if datasource == "twitter_post":
#                     # warnings.warn(f"{index} {datasource} enforcing the role for twitter_post datasource")
#                     role = "human" if idx1 % 2 == 0 else "gpt"
#                     messages[idx1]["from"] = role

#             assert (
#                 len(img_list) == 0
#             ), f"#Num of <images> does not match the number of images in the instance. {instance}"
#         return messages


# class LLaVANextVideoDataset(BaseDataset):
#     def __init__(self, data_path: str, media_dir: str, **kwargs) -> None:
#         super().__init__(**kwargs)
#         self.data_path = data_path
#         self.media_dir = media_dir
#         self.instances = io.load(self.data_path)

#     def process(self, instance: Dict[str, Any]) -> List[Dict[str, Any]]:
#         messages = instance["conversations"]

#         if "video" in instance:
#             img_flist = glob.glob(os.path.join(self.media_dir, instance["video"]) + "/*.jpeg")
#             vpath = os.path.join(self.media_dir, instance["video"])

#             assert len(img_flist) > 0, f"no images found in {vpath}"
#             value = messages[0]["value"]
#             img_list = [Image(img_path) for img_path in img_flist]
#             new_value = [*img_list, value.replace(DEFAULT_IMAGE_TOKEN, "").strip()]
#             messages[0]["value"] = new_value
#         return messages


