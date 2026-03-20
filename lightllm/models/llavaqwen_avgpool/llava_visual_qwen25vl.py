import json
import os
from PIL import Image
from typing import List, Union, Dict, Any, Optional, Tuple
from types import SimpleNamespace
from safetensors import safe_open
from io import BytesIO
import time
from rich.console import Console
from rich.table import Table
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import transformers
from transformers.activations import ACT2FN
from lightllm.server.embed_cache.utils import read_shm, get_shm_name_data
from lightllm.utils.log_utils import init_logger
from lightllm.utils.infer_utils import calculate_cpu_time_sync
from lightllm.models.vit.triton_kernel.rms_norm_vit import rms_norm
from lightllm.models.vit.triton_kernel.flashattention_nopad import flash_attention_fwd
from lightllm.models.qwen2_vl.triton_kernel.rotary_pos_emb import (
    apply_rotary_pos_emb_triton,
)
from lightllm.models.llavaqwen_avgpool.image_processor_opt import (
    Qwen25VLImageProcessorOptimized,
)
from lightllm.models.llavaqwen_avgpool.avgpool import (
    Qwen25VLAvgPoolProjector,
    get_adaptive_pool_size,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2RMSNorm,
)
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from lightllm.server.multimodal_params import ImageItem


logger = init_logger(__name__)


class Qwen2_5_VLPatchMerger(nn.Module):
    def __init__(self, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x).view(-1, self.hidden_size)
        return x


class Qwen25ViTPretrainedModel(Qwen2_5_VisionTransformerPretrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.merger = Qwen2_5_VLPatchMerger(context_dim=config.hidden_size, spatial_merge_size=config.spatial_merge_size)


class LlavaQwen25AvgpoolVisionModelAnyRes:
    def __init__(self):
        # self.thread_pool = ThreadPoolExecutor(max_workers=4)
        pass
        self.spatial_patch_size = 14
        self.spatial_merge_size = 2

    def load_model(self, weight_dir):
        config_file = os.path.join(weight_dir, "config.json")
        config = json.load(open(config_file))
        config = SimpleNamespace(**config)
        self.load_bin_model(config, weight_dir)

        self.vision_tower.requires_grad_(False)
        self.mm_projector.requires_grad_(False)
        self.device = torch.device("cpu")

    def load_hf_model(self, config, weight_dir):
        raise NotImplementedError("can't load model from huggingface model format")

    def load_bin_model(self, config, weight_dir):

        vision_path = getattr(config, "mm_vision_tower", "/mnt/afs/share/qwen25_vl_encoder")
        if isinstance(vision_path, list):
            vision_path = vision_path[0]
        if vision_path.startswith("./"):
            vision_path = os.path.join(weight_dir, vision_path)

        vision_config_path = os.path.join(vision_path, "config.json")
        vision_config = Qwen2_5_VLConfig.from_json_file(vision_config_path)
        vision_config._attn_implementation = "flash_attention_2"

        self.mm_downsample_ratio = config.mm_downsample_ratio
        self.hidden_size = config.mm_hidden_size

        self.image_processor = Qwen25VLImageProcessorOptimized.from_pretrained(vision_path)

        self.vision_tower = Qwen25ViTPretrainedModel._from_config(vision_config).to(torch.bfloat16)  # 精度

        self.spatial_patch_size = vision_config.spatial_patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size

        self.mm_projector = Qwen25VLAvgPoolProjector(config)

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

    def cuda(self, use_compile: bool = False):
        self.vision_tower = self.vision_tower.cuda().eval()
        self.mm_projector = self.mm_projector.cuda().eval()
        self.device = torch.device("cuda")
        self.cuda_stream = torch.cuda.Stream()
        if use_compile:
            logger.info("使用 torch.compile 编译模型...")
            self.vision_tower = torch.compile(self.vision_tower, mode="max-autotune")
            self.mm_projector = torch.compile(self.mm_projector, mode="max-autotune")
            logger.info("编译完成。")
        return self

    @torch.no_grad()
    def forward(self, images, images_thw):

        images_gpu = images.to(device=self.device, dtype=torch.bfloat16, non_blocking=True)
        images_thw_gpu = images_thw.to(device=self.device, dtype=torch.int32, non_blocking=True)

        image_embeds = self.vision_tower(images_gpu, images_thw_gpu)
        image_embeds = image_embeds.float()
        image_embeds, token_len_list = self.mm_projector(image_embeds, images_thw)
        # image_embeds = image_embeds.float()
        return image_embeds

    # @calculate_cpu_time_sync(show=True)
    def process_image(self, image_data):
        image_data = self.image_processor.preprocess(image_data, return_tensors="pt")
        images = image_data["pixel_values"]
        image_thw = image_data["grid_thw"]
        return images, image_thw

    @calculate_cpu_time_sync(show=True)
    def get_image_tensor(self, images: List[ImageItem]):

        image_uuids = [img.uuid for img in images]
        uuids = []
        valid_id = 0
        valid_ids = []

        img_tensors = []
        img_grids = []
        image_sizes = []

        total_imgs = len(images)
        i = 0
        flag = False
        while i < total_imgs:

            if i + 1 < total_imgs:
                items = [image_uuids[i], image_uuids[i + 1]]
            else:
                items = [image_uuids[i], image_uuids[i]]  # 奇数情况复制一份
                flag = True

            images = []
            try:
                for item in items:
                    if not isinstance(item, int):
                        raise Exception("Unsupport input types: {} for {}".format(type(item), item))
                    uuids.append(item)
                    image_data = read_shm(get_shm_name_data(item))
                    image = Image.open(BytesIO(image_data)).convert("RGB")
                    logger.info(f"image size: {image.size}")
                    images.append(image)

                # 一次性处理两张图，添加错误处理
                pixel_values, image_thw = self.process_image(images)
                img_tensors.append(pixel_values)
                img_grids.append(image_thw)

                # 假设每组的尺寸一致
                width, height = images[0].size
                image_sizes.append((width, height))

                M = width / self.spatial_patch_size / self.spatial_merge_size
                N = height / self.spatial_patch_size / self.spatial_merge_size
                m, n = get_adaptive_pool_size(M, N, scale=self.mm_downsample_ratio)

                cur_num = int(m * n)
                single_image_token_num = cur_num // 2

                if not flag:
                    for _ in range(2):
                        valid_ids.append([valid_id, valid_id + single_image_token_num])
                        valid_id += single_image_token_num
                else:
                    valid_ids.append([valid_id, valid_id + single_image_token_num])
                    valid_id += single_image_token_num

            except Exception as e:

                error_msg = f"处理图片失败: {str(e)}"
                logger.error(error_msg)

            i += 2

        if not img_tensors:
            return None

        if flag:
            uuids.pop()
        imgs = torch.cat(img_tensors, dim=0)
        grid_thw = torch.cat(img_grids, dim=0)
        return imgs, grid_thw, uuids, valid_ids

    def encode(self, image_uuids: List):
        img, img_thw, uuids, valid_ids = self.get_image_tensor(image_uuids)
        torch.cuda.synchronize()
        start = time.time()
        all_img_embeds = self.forward(img, img_thw)
        torch.cuda.synchronize()
        end = time.time()
        logger.debug(f"[encode image] 推理耗时: {end - start:.4f} 秒 image num {round(len(valid_ids) / 2) * 2}")

        return all_img_embeds, uuids, valid_ids


def count_custom_module(module, x, y):
    """
    A generic counter for any custom module.
    It iterates through all standard layers within the module that thop *does* recognize
    (like nn.Linear, nn.LayerNorm) and sums their FLOPs.
    """
    total_ops = 0
    for sub_module in module.modules():
        # Check if the submodule is a leaf and has a FLOPs count attached by thop
        if len(list(sub_module.children())) == 0 and hasattr(sub_module, "total_ops"):
            total_ops += sub_module.total_ops
    module.total_ops = total_ops


class FLOPCounter:
    def __init__(self, model):
        self.model = model

    def measure_vit_flops(self, input_tensor, images_thw):
        """
        专门测量ViT部分的FLOPs
        只测量self.vision_tower，不包含其他组件
        """
        try:
            from thop import profile

            # 只测量vision_tower部分，传入正确的参数格式
            vit_model = self.model.vision_tower

            # 确保输入格式正确，vision_tower期望的输入格式
            # 与forward方法中的调用保持一致
            images_input = input_tensor.to(device=torch.device("cuda"), dtype=torch.bfloat16)
            images_thw_input = images_thw.to(device=torch.device("cuda"), dtype=torch.int32)

            flops, params = profile(vit_model, inputs=(images_input, images_thw_input), verbose=True)

            return flops, params

        except Exception as e:
            print(f"ViT FLOPs测量失败: {e}")
            return None, None

    def get_flops_breakdown(self, input_tensor, images_thw):

        self.measure_vit_flops(input_tensor, images_thw)
        import operator
        from functools import reduce

        vit_model = self.model.vision_tower
        module_flops = {}

        def sum_module_flops(module):
            total_ops = reduce(operator.add, (getattr(d, "total_ops", 0) for d in module.modules()))

            # 💡 **HERE IS THE FIX** 💡
            # Ensure the final result is a Python number, not a tensor.
            if isinstance(total_ops, torch.Tensor):
                return total_ops.item()
            return total_ops

        # 1. Patch Embedding
        if hasattr(vit_model, "patch_embed"):
            module_flops["patch_embed"] = sum_module_flops(vit_model.patch_embed)

        if hasattr(vit_model, "rotary_pos_emb"):
            module_flops["rotary_pos_emb_module"] = sum_module_flops(vit_model.rotary_pos_emb)

        if hasattr(vit_model, "blocks") and isinstance(vit_model.blocks, nn.ModuleList):
            for i, block in enumerate(vit_model.blocks):
                block_name = f"block_{i}"
                module_flops[block_name] = sum_module_flops(block)

        if hasattr(vit_model, "merger"):
            module_flops["merger"] = sum_module_flops(vit_model.merger)

        return module_flops


class BenchmarkRunner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results: Dict[str, Dict[int, Dict[str, float]]] = {}
        self.console = Console()
        self.models_to_test: Dict[str, Any] = {}
        self.flop_counters: Dict[str, FLOPCounter] = {}

    def _prepare_data(self, batch_size: int, image_processor: Any) -> tuple:

        images = []
        for _ in range(batch_size):
            fake_img_np = np.random.randint(
                0,
                255,
                (self.config["image_height"], self.config["image_width"], 3),
                dtype=np.uint8,
            )
            images.append(Image.fromarray(fake_img_np))
        image_tensors, images_thw = image_processor(images)
        return image_tensors.cuda(), images_thw.cuda()

    def _execute_benchmarks(self):
        """执行所有已设置模型的基准测试"""
        self.console.print("[bold cyan]🚀 开始基准测试...[/bold cyan]")
        any_model = next(iter(self.models_to_test.values()))
        image_processor = any_model.process_image

        for bs in self.config["batch_sizes"]:
            self.console.print(f"\n[yellow]正在测试 Batch Size: {bs}...[/yellow]")
            image_tensors, images_thw = self._prepare_data(bs, image_processor)

            for model_name, model in self.models_to_test.items():
                self.console.print("-> 评估模型: [bold magenta]{model_name}[/bold magenta]")
                # try:
                # 测量FLOPs (只在第一个batch size时测量一次)
                if bs == self.config["batch_sizes"][0]:
                    self.console.print("-> 测量ViT FLOPs...")
                    flop_counter = self.flop_counters[model_name]
                    vit_flops, vit_params = flop_counter.measure_vit_flops(image_tensors, images_thw)
                    flops_breakdown = flop_counter.get_flops_breakdown(image_tensors, images_thw)

                    if vit_flops is not None:
                        self.console.print(f"  -> ViT FLOPs: {vit_flops/1e9:.2f} GFLOPs, Params: {vit_params/1e6:.2f}M")
                        if model_name not in self.results:
                            self.results[model_name] = {}
                        self.results[model_name]["flops"] = vit_flops
                        self.results[model_name]["params"] = vit_params
                        self.results[model_name]["flops_breakdown"] = flops_breakdown

                for _ in range(self.config["warmup_runs"]):
                    with torch.no_grad():
                        _ = model.forward(image_tensors, images_thw)

                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(self.config["test_runs"]):
                    with torch.no_grad():
                        _ = model.forward(image_tensors, images_thw)
                torch.cuda.synchronize()
                end_time = time.time()

                total_time = end_time - start_time
                avg_latency_ms = (total_time / self.config["test_runs"]) * 1000
                throughput = bs * self.config["test_runs"] / total_time

                if model_name not in self.results:
                    self.results[model_name] = {}
                self.results[model_name][bs] = {
                    "latency": avg_latency_ms,
                    "throughput": throughput,
                }

                # except Exception as e:
                #     self.console.print(f"[bold red]  -> 错误: 模型 {model_name} 在 BS={bs} 时运行失败: {e}[/bold red]")
                #     if model_name not in self.results: self.results[model_name] = {}
                #     self.results[model_name][bs] = {"latency": float('inf'), "throughput": 0}

                torch.cuda.empty_cache()

    def report(self, baseline_name: str = None):
        """
        生成并打印性能对比报告。
        :param baseline_name: 用于计算加速比的基准模型名称。如果为None，则使用第一个模型作为基准。
        """
        self.console.print("\n[bold green]📊 性能测试报告[/bold green]")
        model_names = list(self.results.keys())
        if not model_names:
            self.console.print("[bold red]没有可报告的结果。[/bold red]")
            return

        if baseline_name and baseline_name not in model_names:
            self.console.print(f"[bold red]警告: 基准模型 '{baseline_name}' 不在测试结果中。将使用第一个模型 '{model_names[0]}' 作为替代。[/bold red]")
            baseline_name = None

        if baseline_name is None:
            baseline_name = model_names[0]

        # 首先显示FLOPs和参数信息
        self.console.print("\n[bold blue]🔍 模型复杂度分析[/bold blue]")
        flops_table = Table(title="ViT FLOPs和参数统计")
        flops_table.add_column("模型", justify="center", style="cyan")
        flops_table.add_column("ViT FLOPs (GFLOPs)", justify="center", style="magenta")
        flops_table.add_column("ViT Params (M)", justify="center", style="green")

        for name in model_names:
            if "flops" in self.results[name]:
                flops_g = self.results[name]["flops"] / 1e9
                params_m = self.results[name]["params"] / 1e6
                flops_table.add_row(name, f"{flops_g:.2f}", f"{params_m:.2f}")

        self.console.print(flops_table)

        self.console.print("\n[bold blue]🔬 ViT 模块 FLOPs 分解[/bold blue]")
        breakdown_table = Table(title="各模块 FLOPs 占比")
        breakdown_table.add_column("模型", justify="left", style="cyan")
        breakdown_table.add_column("模块", justify="left", style="white")
        breakdown_table.add_column("FLOPs (G)", justify="right", style="magenta")
        breakdown_table.add_column("占比 (%)", justify="right", style="green")

        for name, model_results in self.results.items():
            if "flops_breakdown" in model_results:
                total_flops = model_results.get("flops", 1)  # Avoid division by zero
                breakdown = model_results["flops_breakdown"]

                # Sort by FLOPs descending for better readability
                sorted_breakdown = sorted(breakdown.items(), key=lambda item: item[1], reverse=True)

                is_first_row = True
                for module_name, module_flops in sorted_breakdown:
                    flops_g = module_flops / 1e9
                    percentage = (module_flops / total_flops) * 100
                    if is_first_row:
                        # Show model name only on the first row for this model
                        breakdown_table.add_row(
                            f"[bold]{name}[/bold]",
                            module_name,
                            f"{flops_g:.2f}",
                            f"{percentage:.1f}%",
                        )
                        is_first_row = False
                    else:
                        breakdown_table.add_row("", module_name, f"{flops_g:.2f}", f"{percentage:.1f}%")
                # Add a separator line between models
                if len(self.results) > 1:
                    breakdown_table.add_row(end_section=True)

        self.console.print(breakdown_table)

        # 性能对比表格
        table = Table(title=f"模型性能对比 (基准: {baseline_name})")
        table.add_column("Batch Size", justify="center", style="cyan")
        for name in model_names:
            table.add_column(f"{name}\nLatency (ms)", justify="center", style="magenta")
            table.add_column(f"{name}\nThroughput (img/s)", justify="center", style="green")

        if len(model_names) > 1:
            table.add_column("Speedup 🚀", justify="center", style="yellow")

        for bs in self.config["batch_sizes"]:
            row_data = [str(bs)]
            baseline_latency = self.results[baseline_name][bs].get("latency", float("inf"))

            for name in model_names:
                res = self.results[name].get(bs, {"latency": float("inf"), "throughput": 0})
                row_data.extend([f"{res['latency']:.2f}", f"{res['throughput']:.2f}"])

            if len(model_names) > 1:
                optimized_model_name = next(n for n in model_names if n != baseline_name)
                optimized_latency = self.results[optimized_model_name][bs].get("latency", float("inf"))
                speedup = baseline_latency / optimized_latency if optimized_latency > 0 else float("inf")
                row_data.append(f"{speedup:.2f}x")

            table.add_row(*row_data)

        self.console.print(table)

    def run_and_report(self, models_to_test: Dict[str, Any], baseline_name: str = None):
        """
        一站式完成所有模型的基准测试并生成报告。
        """
        self.models_to_test = models_to_test

        # 为每个模型创建FLOP计数器
        for model_name, model in models_to_test.items():
            self.flop_counters[model_name] = FLOPCounter(model)

        self._execute_benchmarks()
        self.report(baseline_name=baseline_name)


if __name__ == "__main__":
    torch._dynamo.config.capture_scalar_outputs = True
    # --- 1. 配置中心 ---
    model_path = "0706_llava_omni_qwen25vl_14B_16x_8k_vitlora_kimiwhisper_10x_conv_channelupscale"
    BENCHMARK_CONFIG = {
        "model_path": model_path,
        "batch_sizes": [1, 2, 4, 8, 16, 24, 32, 48, 64],
        "warmup_runs": 5,
        "test_runs": 10,
        "image_height": 364,
        "image_width": 644,
    }

    # --- 2. 准备待测试的模型 ---
    console = Console()
    console.print("[bold blue]1. 正在加载和准备模型...[/bold blue]")

    # 模型A: 标准的 PyTorch Eager 模型

    model_eager = LlavaQwen25AvgpoolVisionModelAnyRes()
    model_eager.load_model(BENCHMARK_CONFIG["model_path"])
    model_eager.cuda(use_compile=False)
    console.print("[green]  -> Eager 模型准备就绪。[/green]")

    # 模型B: 使用 torch.compile 优化的模型
    # model_compiled = LlavaQwen25AvgpoolVisionModel()
    # model_compiled.load_model(BENCHMARK_CONFIG["model_path"])
    # model_compiled.cuda(use_compile=True) # 假设您的 .cuda() 方法已按建议修改
    # console.print("[green]  -> Compiled 模型准备就绪。[/green]")

    models_to_compare = {
        "PyTorch Eager": model_eager,
        # "torch.compile": model_compiled
    }

    # --- 3. 运行并报告 ---
    console.print("\n[bold blue]2. 开始执行基准测试...[/bold blue]")
    runner = BenchmarkRunner(BENCHMARK_CONFIG)
    runner.run_and_report(models_to_compare, baseline_name="PyTorch Eager")
    image_tensors, images_thw = runner._prepare_data(2, model_eager.process_image)

    for i in range(10):
        image_embed = model_eager.forward(image_tensors, images_thw)
