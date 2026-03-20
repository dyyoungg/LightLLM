import os
import re
import json
import math
import numpy as np
from functools import lru_cache

from lightllm.models.registry import ModelRegistry
from lightllm.common.basemodel.multimodal_tokenizer import BaseMultiModalTokenizer
from transformers import AutoTokenizer
from lightllm.models.qwen2.model import Qwen2TpPartModel
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import (
    LlamaMultimodalPreLayerInfer,
)
from lightllm.server.multimodal_params import MultimodalParams, ImageItem, AudioItem
from lightllm.common.build_utils import repair_config
from transformers import AutoConfig
from lightllm.server.core.objs import SamplingParams
from lightllm.utils.infer_utils import calculate_cpu_time_sync

# Warp of the origal tokenizer


class LlavaQWen25AudioVLTokenizer(BaseMultiModalTokenizer):
    def __init__(self, tokenizer, model_cfg):
        super().__init__(tokenizer)

        self.model_cfg = model_cfg
        self.image_token = model_cfg.get("image_token", "<image>")
        self.audio_token = model_cfg.get("audio_token", "<audio>")
        self.audio_start_id = tokenizer.convert_tokens_to_ids("<|audio_start|>")
        self.audio_end_id = tokenizer.convert_tokens_to_ids("<|audio_end|>")
        self.skip_start = model_cfg.get("skip_start", True)
        self.patch_size = model_cfg.get("mm_patch_size", 14)
        self.merge_size = model_cfg.get("mm_merge_size", 2)
        self.audio_downsample_ratio = model_cfg.get("audio_downsample_ratio", 10)
        self.audio_frame_length = model_cfg.get("audio_frame_length", 320)
        # image token length
        # height, width = 364, 644
        # h, w = self.get_adaptive_pool_size(height, width, scale=int(self.model_cfg.get("mm_downsample_ratio", 16)))
        # self.image_length = h * w // 2

    @lru_cache(maxsize=10)
    def get_adaptive_pool_size(self, h, w, scale=16):
        M = h / self.patch_size / self.merge_size
        N = w / self.patch_size / self.merge_size
        r = 1 / math.sqrt(scale)
        Mh = max(1, int(np.round(M * r)))
        Nw = max(1, int(np.round(N * r)))
        return Mh, Nw

    def get_image_token_length(self, img: ImageItem):
        width = img.image_w
        height = img.image_h
        h, w = self.get_adaptive_pool_size(height, width, scale=self.model_cfg["mm_downsample_ratio"])
        image_length = h * w // 2  # 每两张图需要合并
        return image_length

    def get_audio_token_length(self, audio: AudioItem):
        feature_len = audio.audio_length // self.audio_frame_length
        token_num = (feature_len + self.audio_downsample_ratio - 1) // self.audio_downsample_ratio
        return token_num

    def init_imageitem_extral_params(
        self,
        img: ImageItem,
        multi_params: MultimodalParams,
        sampling_params: SamplingParams,
    ):
        return

    def init_audioitem_extral_params(
        self,
        audio: AudioItem,
        multi_params: MultimodalParams,
        sampling_params: SamplingParams,
    ):
        return

    def encode(self, prompt, multimodal_params: MultimodalParams = None, **kwargs):

        pattern = f"({re.escape(self.image_token)}|{re.escape(self.audio_token)})"
        chunks = re.split(pattern, prompt)
        input_ids = []
        image_id = 0
        audio_id = 0
        idx = 0
        while idx < len(chunks):
            text = chunks[idx]
            if text:
                ids = self.tokenizer(text).input_ids
                if len(input_ids) == 0:
                    input_ids.extend(ids)
                else:
                    if len(ids) > 0 and ids[0] == self.tokenizer.bos_token_id and self.skip_start:
                        ids = ids[1:]
                    input_ids.extend(ids)
            idx += 1
            if idx < len(chunks):
                token = chunks[idx]
                if token == self.image_token:
                    assert multimodal_params is not None and image_id < len(
                        multimodal_params.images
                    ), "Not enough images in multimodal_params"
                    token_id = multimodal_params.images[image_id].token_id
                    token_num = multimodal_params.images[image_id].token_num
                    input_ids.extend(range(token_id, token_id + token_num))
                    image_id += 1
                elif token == self.audio_token:
                    assert multimodal_params is not None and audio_id < len(
                        multimodal_params.audios
                    ), "Not enough audios in multimodal_params"
                    token_id = multimodal_params.audios[audio_id].token_id
                    token_num = multimodal_params.audios[audio_id].token_num
                    if self.audio_start_id is not None and self.audio_end_id is not None:
                        audio_input_ids = (
                            [self.audio_start_id] + list(range(token_id, token_id + token_num)) + [self.audio_end_id]
                        )
                    else:
                        audio_input_ids = list(range(token_id, token_id + token_num))
                    input_ids.extend(audio_input_ids)
                    audio_id += 1
                idx += 1
        if multimodal_params:
            assert image_id == len(
                multimodal_params.images
            ), f"invalid image tag num: {len(multimodal_params.images)} vs {image_id}!"
            assert audio_id == len(
                multimodal_params.audios
            ), f"invalid audio tag num: {len(multimodal_params.audios)} vs {audio_id}!"
        return input_ids

    def __getattr__(self, name):
        if name != "encode":
            return getattr(self.tokenizer, name)
        return self.encode


@ModelRegistry("llavaqwen2", is_multimodal=True)
class LlavaQwen2TpPartModel(Qwen2TpPartModel):

    # infer class
    pre_layer_infer_class = LlamaMultimodalPreLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_config(self):
        super()._init_config()
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        return


if __name__ == "__main__":

    tokenizer_path = "0728_llava_omni_qwen25vl_14B_16x_4k_st2_kimiwhisper_10x_unfreezeaudio_omnidata_text500w_lr2e-6"
    config_path = f"{tokenizer_path}/config.json"

    with open(config_path, "r") as f:
        model_cfg = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    multimodal_params = {
        "images": [{"type": "base64", "data": "dGVzdF9pbWFnZV9kYXRh"}],
        "audios": [{"type": "base64", "data": "dGVzdF9pbWFnZV9kYXRh"}],
    }

    multimodal_params = MultimodalParams(**multimodal_params)
    for img in multimodal_params.images:
        img.uuid = 1
        img.token_id = 50000
        img.token_num = 9

    for audio in multimodal_params.audios:
        audio.uuid = 2
        audio.token_id = 60000
        audio.token_num = 10

    prompt = "这是一段文本<image>然后是图片后的文本<audio>最后是音频后的文本"

    my_tokenizer = LlavaQWen25AudioVLTokenizer(tokenizer, model_cfg)
    input_ids = my_tokenizer.encode(prompt, multimodal_params)
    print("input_ids:", input_ids)
