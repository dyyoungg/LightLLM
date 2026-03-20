import json
import os
from PIL import Image
from typing import List, Union
from types import SimpleNamespace
from safetensors import safe_open
from io import BytesIO

import torch
import torch.nn.functional as F
import torch.nn as nn
import transformers
from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    SiglipVisionModel,
    SiglipImageProcessor,
    CLIPVisionConfig,
)
from typing import Optional, Tuple
from flash_attn import flash_attn_func
from lightllm.server.embed_cache.utils import read_shm, get_shm_name_data
from lightllm.utils.log_utils import init_logger
from lightllm.utils.infer_utils import calculate_cpu_time_sync
from concurrent.futures import ThreadPoolExecutor
from lightllm.models.llavaqwen_avgpool.image_processor_opt import (
    OpimizedCLIPImageProcessor,
)

logger = init_logger(__name__)


def my_CLIPVisionEmbeddings_init(self, config: "CLIPVisionConfig"):
    super(self.__class__, self).__init__()
    self.config = config
    self.embed_dim = config.hidden_size
    self.image_size = config.image_size
    self.patch_size = config.patch_size

    self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
    self.patch_embedding = nn.Conv2d(
        in_channels=config.num_channels,
        out_channels=self.embed_dim,
        kernel_size=self.patch_size,
        stride=self.patch_size,
        bias=False,
    )

    if isinstance(self.image_size, int):
        self.image_size = {"width": self.image_size, "height": self.image_size}
    self.num_patches = (self.image_size["width"] // self.patch_size) * (
        self.image_size["height"] // self.patch_size
    )  # btnkij
    self.num_positions = self.num_patches + 1
    self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
    self.register_buffer(
        "position_ids",
        torch.arange(self.num_positions).expand((1, -1)),
        persistent=False,
    )


def fix_embedding_forward(self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=False) -> torch.Tensor:
    batch_size, _, height, width = pixel_values.shape
    target_dtype = self.patch_embedding.weight.dtype
    patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
    patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

    class_embeds = self.class_embedding.expand(batch_size, 1, -1)
    embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
    if interpolate_pos_encoding:
        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
    else:
        embeddings = embeddings + self.position_embedding(self.position_ids)
    return embeddings


def my_CLIPAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    causal_attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""

    bsz, tgt_len, embed_dim = hidden_states.size()
    assert embed_dim == self.num_heads * self.head_dim

    # get query proj
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim)
    key_states = key_states.view(bsz, tgt_len, self.num_heads, self.head_dim)
    value_states = value_states.view(bsz, tgt_len, self.num_heads, self.head_dim)

    # apply the causal_attention_mask first
    if causal_attention_mask is not None:
        raise NotImplementedError

    if attention_mask is not None:
        raise NotImplementedError

    if output_attentions:
        raise NotImplementedError
    else:
        attn_weights_reshaped = None

    attn_output = flash_attn_func(
        query_states,
        key_states,
        value_states,
        dropout_p=(self.dropout if self.training else 0),
        softmax_scale=self.scale,
        causal=False,
    )
    # attn_output.shape == (bsz, tgt_len, num_heads, head_dim)
    attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights_reshaped


transformers.models.clip.modeling_clip.CLIPVisionEmbeddings.__init__ = my_CLIPVisionEmbeddings_init
transformers.models.clip.modeling_clip.CLIPAttention.forward = my_CLIPAttention_forward
transformers.models.clip.modeling_clip.CLIPVisionEmbeddings.forward = fix_embedding_forward


class LlavaAvgpoolVisionModel:
    def __init__(self):
        # self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self._compiled_forward = None  # 添加编译后的forward引用

    def load_model(self, weight_dir):
        config_file = os.path.join(weight_dir, "config.json")
        config = json.load(open(config_file))

        self.load_bin_model(config, weight_dir)

        self.vision_tower.requires_grad_(False)
        self.mm_projector.requires_grad_(False)
        self.device = torch.device("cpu")

    def load_hf_model(self, config, weight_dir):
        raise NotImplementedError("can't load model from huggingface model format")

    def load_bin_model(self, config, weight_dir):
        self.select_layer = config.get("mm_vision_select_layer", -2)
        self.select_feature = config.get("mm_vision_select_feature", "patch")

        # load clip vision model by cfg['mm_vision_tower']:
        #   huggingface_name or path_of_clip_relative_to_llava_model_dir
        vision_path = config.get("mm_vision_tower", "openai/clip-vit-large-patch14-336")
        if isinstance(vision_path, list):
            vision_path = vision_path[0]
        if vision_path.startswith("./"):
            vision_path = os.path.join(weight_dir, vision_path)

        from .avgpool import PoolPojector

        if "siglip" in vision_path.lower():
            # self.image_processor = SiglipImageProcessor.from_pretrained(vision_path)
            self.image_processor = OpimizedCLIPImageProcessor.from_pretrained(vision_path)
            self.vision_tower = SiglipVisionModel.from_pretrained(vision_path).half()  # 精度

        elif "clip" in vision_path.lower():
            # self.image_processor = CLIPImageProcessor.from_pretrained(vision_path)

            self.image_processor = OpimizedCLIPImageProcessor.from_pretrained(vision_path)
            self.vision_tower = CLIPVisionModel.from_pretrained(vision_path).half()

        assert isinstance(config["proj_output_size"], dict) and len(config["proj_output_size"]) == 2
        config["mm_hidden_size"] = self.vision_tower.config.hidden_size

        self.mm_projector = PoolPojector(config)
        self.token_num = int(config["proj_output_size"]["width"] * config["proj_output_size"]["height"] + 1)
        # load projector weights
        self.projector_weights = {}
        self.vision_model_weights = {}
        for f in os.listdir(weight_dir):
            if f.endswith(".safetensors"):
                d = safe_open(os.path.join(weight_dir, f), "pt", "cpu")
                for k in d.keys():
                    if "model.mm_projector" in k:
                        self.projector_weights[k.replace("model.mm_projector.", "")] = d.get_tensor(k)
                    elif "model.vision_tower.vision_tower" in k:
                        self.vision_model_weights[k.replace("model.vision_tower.vision_tower.", "")] = d.get_tensor(k)

        self.mm_projector.load_state_dict(self.projector_weights)
        self.vision_tower.load_state_dict(self.vision_model_weights)

    def cuda(self):
        self.vision_tower = self.vision_tower.cuda()
        self.mm_projector = self.mm_projector.cuda()

        # self.vision_tower = torch.compile(
        #                 self.vision_tower,
        #                 mode="reduce-overhead",
        #                 fullgraph=True,
        #             )
        # self.mm_projector = torch.compile(
        #                 self.mm_projector,
        #                 mode="reduce-overhead",
        #                 fullgraph=True,
        # )
        return self
        # if self._compiled_forward is None:

        #     def _forward_impl(x):
        #         x = x.half().cuda()
        #         x = self.vision_tower(x, output_hidden_states=True)
        #         x = x.hidden_states[self.select_layer]
        #         x = x[:, 1:].contiguous()
        #         B, L, N = x.shape
        #         x = x.float()
        #         x = self.mm_projector(x)
        #         x = x.view(B, self.token_num, -1)
        #         return x

        #     try:
        #         self._compiled_forward = torch.compile(
        #             _forward_impl,
        #             mode="reduce-overhead",
        #             fullgraph=True,
        #         )
        #         print("Compile successfully!")
        #     except Exception as e:
        #         print(f"Compilation failed: {e}")
        #         self._compiled_forward = None

        # return self

    @torch.no_grad()
    def forward(self, x):
        # if self._compiled_forward is not None:
        #     return self._compiled_forward(x)

        x = x.half().cuda()
        vision_outputs = self.vision_tower(x, output_hidden_states=True)
        x = vision_outputs.hidden_states[self.select_layer]
        del vision_outputs
        # if self.select_feature == "patch" or self.select_feature == "default":
        x = x[:, 1:].contiguous()
        B, L, N = x.shape
        x = x.float()
        x = self.mm_projector(x)
        x = x.view(B, self.token_num, -1)
        return x

    # @calculate_cpu_time_sync(show=True)
    def process_image(self, image_data):
        t = self.image_processor.preprocess(image_data, return_tensors="pt")["pixel_values"]
        return t

    # @calculate_cpu_time_sync(show=True)
    def get_image_tensor(self, image_uuids):

        uuids = []
        valid_id = 0
        valid_ids = []

        batch_images = []
        for i, item in enumerate(image_uuids):
            if isinstance(item, int):
                uuids.append(item)
                image_data = read_shm(get_shm_name_data(item))
                # print("image data length", len(image_data))
                image_data = Image.open(BytesIO(image_data)).convert("RGB")
                batch_images.append(image_data)
            else:
                raise Exception("Unsupport input types: {} for {}".format(type(item), item))

        img_tensors = self.process_image(batch_images)
        cur_num = img_tensors.shape[0]
        valid_ids = [[i, i + 1] for i in range(cur_num)]

        return img_tensors, uuids, valid_ids

    def encode(self, image_uuids: List):
        img, uuids, valid_ids = self.get_image_tensor(image_uuids)
        all_img_embeds = self.forward(img)
        return all_img_embeds, uuids, valid_ids


class FLOPCounter:
    def __init__(self, model):
        self.model = model

    def method1_thop(self, input_tensor):
        try:
            from thop import profile

            flops, params = profile(self.model, inputs=(input_tensor.clone(),), verbose=False)

            print(f"✓ thop - FLOPs: {flops/1e9:.2f} GFLOPs, Params: {params/1e6:.2f}M")
            return flops

        except Exception as e:
            print(f"thop失败: {e}")
            return None


if __name__ == "__main__":
    import numpy as np
    import time

    model_path = (
        "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/checkpoints/llava_qwen_audio_firered_channelupscale_down3_mixall_0224"
    )

    pt_model = LlavaAvgpoolVisionModel()

    pt_model.load_model(model_path)
    pt_model.cuda()

    batch_sizes = [1, 2, 4, 8, 16, 20, 128, 256]
    warmup_runs = 3
    test_runs = 20

    # from .clip_trt import CLIPVisionModelTRT

    # trt_model = CLIPVisionModelTRT.from_pretrained("/mnt/afs/zhouhang/model/clip-vit-large-patch14-644x364", use_tensorrt=True, projector=False)

    def run_test(model, image_tensor, bs, warmup_runs, test_runs):
        flop_counter = FLOPCounter(model.vision_tower)
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model.forward(image_tensor.cuda())

        single_input = image_tensor[:1].cuda()
        total_flops = flop_counter.method1_thop(single_input)

        torch.cuda.synchronize()
        start_time = time.time()

        for _ in range(test_runs):
            with torch.no_grad():
                x = pt_model.forward(image_tensor.cuda())

        torch.cuda.synchronize()
        end_time = time.time()

        # 计算性能指标
        total_time = end_time - start_time
        avg_time = (total_time / test_runs) * 1000  # 转换为毫秒
        throughput = bs * test_runs / total_time

        if total_flops:
            batch_flops = total_flops * bs
            flops_per_second = batch_flops * test_runs / total_time
            efficiency = flops_per_second / 1e12  # TFLOP/s
            print(f"Batch {bs}: {avg_time:.2f}ms, {throughput:.2f}img/s, {efficiency:.2f}TFLOP/s")
        return avg_time, throughput, total_flops

    # 存储性能测试结果
    results = []

    def test_performance():
        for bs in batch_sizes:
            images = []
            for _ in range(bs):
                height, width = 364, 644
                fake_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                image = Image.fromarray(fake_img)
                images.append(image)
            print(f"testing bs:{bs}")
            t = pt_model.image_processor.preprocess(images, return_tensors="pt")["pixel_values"]

            avg_pt, throughput1, flops = run_test(pt_model, t, bs, warmup_runs, test_runs)

            # avg_trt, throughput2 = run_test(trt_model, t, bs, warmup_runs, test_runs)

            # 存储结果
            results.append(
                {
                    "batch_size": bs,
                    "avg_time_pt": avg_pt,
                    "throughput_pt": throughput1,
                    "flops": flops,
                    # 'avg_time_trt': avg_trt,
                    # 'throughput_trt': throughput2,
                }
            )

        # 打印性能测试报告
        print("\n性能测试报告:")
        print("=" * 80)
        print(f"{'批次大小':^15} | {'平均推理时间(PT/TRT)(ms)':^20} | {'吞吐量(PT/TRT)(img/s)':^15} | FLOPS")
        print("-" * 80)

        for result in results:
            print(
                f"{result['batch_size']:^15} | "
                f"{result['avg_time_pt']:^10.2f}| "
                f"{result['throughput_pt']:^7.2f}| "
            )

        print("=" * 80)

    test_performance()
