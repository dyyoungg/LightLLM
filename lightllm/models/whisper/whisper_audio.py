import torch
import librosa
from io import BytesIO
from typing import List, Union, Dict, Any
import numpy as np
import time
from torch import nn
from safetensors.torch import load_file
from torch.nn.utils.rnn import pad_sequence
from whisper.audio import pad_or_trim, log_mel_spectrogram
import concurrent.futures
import pickle

import json
import torch.nn.functional as F
import math
import os
import rpyc
from rich.console import Console
from rich.table import Table
from lightllm.utils.log_utils import init_logger
from lightllm.server.multimodal_params import AudioItem
from lightllm.models.whisper.modeling_whisper import WhisperModel
from lightllm.models.whisper.defaults import MIN_AUDIO_LEN, MAX_AUDIO_LEN
from lightllm.server.embed_cache.utils import (
    tensor2bytes,
    read_shm,
    create_shm,
    get_shm_name_data,
    get_shm_name_embed,
)
from lightllm.utils.infer_utils import calculate_cpu_time_sync

logger = init_logger(__name__)


class AudioConvUpScaleProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.audio_hidden_size = config.audio_hidden_size
        self.afeat_1d_conv = nn.Conv1d(
            in_channels=config.audio_hidden_size,
            out_channels=config.audio_hidden_size,
            kernel_size=2,
            stride=2,
            padding=0,
        )  # 50Hz -> 25Hz
        self.compress_ratio = config.audio_downsample_ratio // 2  # conv已经压缩了2倍
        self.linear1 = nn.Linear(
            int(self.audio_hidden_size * self.compress_ratio),
            self.hidden_size,
            bias=True,
        )
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def forward(self, x, feature_length):
        # x: [bs, seq_len, audio_hidden_size]
        # feature length: List[int]

        x = self.afeat_1d_conv(x.transpose(1, 2)).transpose(1, 2)  # Process Whisper features with 1D conv: (B x T x D) -> (B x T//2 x D')
        bs, seq_len, audio_hidden_size = x.size()

        target_seq_len = (seq_len + self.compress_ratio - 1) // self.compress_ratio * self.compress_ratio
        pad_len = target_seq_len - seq_len

        if pad_len > 0:
            pad_tensor = torch.zeros(bs, pad_len, audio_hidden_size, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad_tensor], dim=1)  # 在时间维度 padding

        new_seq_len = target_seq_len // self.compress_ratio
        x = x.reshape(bs, new_seq_len, audio_hidden_size * self.compress_ratio)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        compress_ratio = self.config.audio_downsample_ratio
        num_tokens = [(cur_l + compress_ratio - 1) // compress_ratio for cur_l in feature_length]
        return x, num_tokens


class WhisperAudioModel:
    def __init__(self, kvargs):
        self.max_seconds = 30
        self.mel_bins = 128
        self.sampling_rate = 16000
        self.max_length = self.max_seconds * self.sampling_rate
        self.cache_port = kvargs["cache_port"]
        self.cache_client = rpyc.connect("localhost", self.cache_port, config={"allow_pickle": True})
        data_type = kvargs["data_type"]
        if data_type in ["bf16", "bfloat16"]:
            self.data_type = torch.bfloat16
        else:
            self.data_type = torch.float16
        self.audio_projector_dtype = torch.float32
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def cuda(self):
        self.audio_model = self.audio_model.cuda()
        self.audio_projector = self.audio_projector.cuda()
        self.device = torch.device("cuda")
        return self

    def load_model(self, weight_dir, config):
        self.audio_model = WhisperModel.from_pretrained(config.audio_encoder).encoder.to(self.data_type)
        self.audio_projector = AudioConvUpScaleProjector(config).to(self.audio_projector_dtype)

        self.load_weight(weight_dir)

    def load_weight(self, weight_dir):
        weight_path = os.path.join(weight_dir, "model.safetensors.index.json")
        weight_map = json.load(open(weight_path, "r"))["weight_map"]
        params_map = {}
        audio_weight = {}
        audio_projector_weight = {}

        for k, v in weight_map.items():

            if "audio_encoder" not in k and "audio_projector" not in k:
                continue

            filename = weight_map[k]
            if filename not in params_map:
                tensor_data = load_file(os.path.join(weight_dir, filename))
                params_map[filename] = tensor_data
            if k.startswith("audio_encoder.audio_projector."):
                audio_projector_weight[k.replace("audio_encoder.audio_projector.", "")] = params_map[filename][k].to(self.data_type)
            elif k.startswith("audio_encoder.") and not k.startswith("audio_encoder.audio_projector."):
                audio_weight[k.replace("audio_encoder.", "")] = params_map[filename][k].to(self.data_type)
            elif "audio_projector" in k:
                audio_projector_weight[k.replace("model.audio_projector.", "")] = params_map[filename][k].to(self.data_type)

            elif "audio_encoder" in k:
                audio_weight[k.replace("model.audio_encoder.model.", "")] = params_map[filename][k].to(self.data_type)

        self.audio_model.load_state_dict(audio_weight)
        self.audio_projector.load_state_dict(audio_projector_weight)

    @torch.no_grad()
    def forward(self, batch_audios, audio_lens):
        # batch audios : List[np.ndarray]
        # audio length: List[int]

        batch_audios = [torch.tensor(a) if not isinstance(a, torch.Tensor) else a for a in batch_audios]
        padded_audio = pad_sequence(batch_audios, batch_first=True, padding_value=0)
        audio = pad_or_trim(padded_audio)
        mel = log_mel_spectrogram(audio, n_mels=self.mel_bins)

        mel = mel.to(self.data_type).to(device=self.device)
        audio_features = self.audio_model(mel).last_hidden_state
        audio_features = audio_features.to(self.audio_projector_dtype)

        audio_features, feature_len = self.audio_projector(audio_features, audio_lens)

        return audio_features, feature_len

    def encode(self, audio_items: List[AudioItem]):
        batch_audios = []
        batch_audio_lens = np.zeros(len(audio_items), dtype=np.int32)
        uuids = []
        for i, item in enumerate(audio_items):
            if isinstance(item, AudioItem):
                uuids.append(item.uuid)
                audio_data = read_shm(get_shm_name_data(item.uuid))
                audio = BytesIO(audio_data)
                audio, _ = librosa.load(audio, sr=16000)
            else:
                raise ValueError(f"cannot read audio which type is {type(item)}!")

            # padding to min audio len
            if audio.shape[0] < MIN_AUDIO_LEN:
                audio = np.pad(
                    audio,
                    (0, MIN_AUDIO_LEN - len(audio)),
                    mode="constant",
                    constant_values=0.0,
                )
            elif audio.shape[0] > MAX_AUDIO_LEN:
                audio = audio[:MAX_AUDIO_LEN]

            batch_audio_lens[i] = min(audio.shape[0], self.max_length)
            batch_audios.append(audio)

        audio_len = [len(audio) // 320 for audio in batch_audios]
        torch.cuda.synchronize()
        start = time.time()
        audios, audio_token_num = self.forward(batch_audios, audio_len)
        audios = audios.to(torch.device("cpu"))
        torch.cuda.synchronize()
        end = time.time()
        logger.debug(f"whisper encode time: {end - start:.4f}s, audio num: {len(audio_len)}")

        # self.alloc_audio_resources(uuids, audios, audio_token_num)
        self.alloc_audio_resources_batch(uuids, audios, audio_token_num)

    @calculate_cpu_time_sync(show=True)
    def alloc_audio_resources(self, uuids, audio_features, audio_token_num):
        for i in range(len(uuids)):
            if not self.cache_client.root.get_item_embed(uuids[i]):
                cur_embed_bytes = tensor2bytes(audio_features[i][: audio_token_num[i]])
                create_shm(get_shm_name_embed(uuids[i]), cur_embed_bytes)
                self.cache_client.root.set_item_embed(uuids[i])

    def create_shm_for_item(self, uuid, audio_feature, audio_token_num):
        """为单个item创建共享内存的函数"""
        cur_embed_bytes = tensor2bytes(audio_feature[:audio_token_num])
        create_shm(get_shm_name_embed(uuid), cur_embed_bytes)
        return uuid

    @calculate_cpu_time_sync(show=True)
    def alloc_audio_resources_batch(self, uuids, audio_features, audio_token_num):

        uuids_blob = pickle.dumps(uuids)
        embed_status = self.cache_client.root.get_items_embed_v2(uuids_blob)
        embed_status = pickle.loads(embed_status)

        tasks = [(uuids[i], audio_features[i], audio_token_num[i]) for i in range(len(uuids)) if not embed_status[i]]

        if not tasks:
            return  # 所有items都已经embed了

        futures = [self.thread_pool.submit(self.create_shm_for_item, uuid, audio, token_num) for uuid, audio, token_num in tasks]

        created_uuids = []
        for future in concurrent.futures.as_completed(futures):
            try:
                uuid = future.result()
                created_uuids.append(uuid)
            except Exception as e:
                print(f"创建共享内存失败: {e}")

        if len(created_uuids):
            created_uuids_blob = pickle.dumps(created_uuids)
            self.cache_client.root.set_items_embed_v2(created_uuids_blob)


class WhisperAudioBenchmarkRunner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results: Dict[str, Dict[int, Dict[str, float]]] = {}
        self.console = Console()
        self.models_to_test: Dict[str, Any] = {}

    def _prepare_data(self, batch_size: int) -> tuple:
        """准备测试音频数据"""
        batch_audios = []
        audio_lens = []

        for _ in range(batch_size):
            # 随机生成音频长度 (1-20秒)
            length = np.random.randint(16000, 320000)  # 1-20秒
            audio = np.random.randn(length).astype(np.float32)
            batch_audios.append(audio)
            audio_lens.append(len(audio) // 320)

        return batch_audios, audio_lens

    def _execute_benchmarks(self):
        """执行所有已设置模型的基准测试"""
        self.console.print("[bold cyan]🚀 开始Whisper音频模型基准测试...[/bold cyan]")

        for bs in self.config["batch_sizes"]:
            self.console.print(f"\n[yellow]正在测试 Batch Size: {bs}...[/yellow]")

            for model_name, model in self.models_to_test.items():
                self.console.print(f"  -> 评估模型: [bold magenta]{model_name}[/bold magenta]")
                try:
                    # 预热
                    for _ in range(self.config["warmup_runs"]):
                        test_audios, test_lens = self._prepare_data(bs)
                        with torch.no_grad():
                            _ = model.forward(test_audios, test_lens)

                    # 正式测试
                    torch.cuda.synchronize()
                    start_time = time.time()

                    for _ in range(self.config["test_runs"]):
                        test_audios, test_lens = self._prepare_data(bs)
                        with torch.no_grad():
                            _ = model.forward(test_audios, test_lens)

                    torch.cuda.synchronize()
                    end_time = time.time()

                    total_time = end_time - start_time
                    avg_latency_ms = (total_time / self.config["test_runs"]) * 1000

                    # 计算吞吐量 (音频秒数/秒)
                    total_audio_length = sum(len(audio) for audio in test_audios) / 16000  # 转换为秒
                    throughput = (bs * total_audio_length * self.config["test_runs"]) / total_time

                    if model_name not in self.results:
                        self.results[model_name] = {}
                    self.results[model_name][bs] = {
                        "latency": avg_latency_ms,
                        "throughput": throughput,
                        "audio_seconds_per_sec": throughput,
                    }

                except Exception as e:
                    self.console.print(f"[bold red]  -> 错误: 模型 {model_name} 在 BS={bs} 时运行失败: {e}[/bold red]")
                    if model_name not in self.results:
                        self.results[model_name] = {}
                    self.results[model_name][bs] = {
                        "latency": float("inf"),
                        "throughput": 0,
                        "audio_seconds_per_sec": 0,
                    }

                torch.cuda.empty_cache()

    def report(self, baseline_name: str = None):
        """
        生成并打印性能对比报告。
        :param baseline_name: 用于计算加速比的基准模型名称。如果为None，则使用第一个模型作为基准。
        """
        self.console.print("\n[bold green]📊 Whisper音频模型性能测试报告[/bold green]")
        model_names = list(self.results.keys())
        if not model_names:
            self.console.print("[bold red]没有可报告的结果。[/bold red]")
            return

        if baseline_name and baseline_name not in model_names:
            self.console.print(f"[bold red]警告: 基准模型 '{baseline_name}' 不在测试结果中。将使用第一个模型 '{model_names[0]}' 作为替代。[/bold red]")
            baseline_name = None

        if baseline_name is None:
            baseline_name = model_names[0]

        table = Table(title=f"Whisper音频模型性能对比 (基准: {baseline_name})")
        table.add_column("Batch Size", justify="center", style="cyan")
        for name in model_names:
            table.add_column(f"{name}\nLatency (ms)", justify="center", style="magenta")
            table.add_column(f"{name}\nThroughput\n(audio sec/s)", justify="center", style="green")

        if len(model_names) > 1:
            table.add_column("Speedup 🚀", justify="center", style="yellow")

        for bs in self.config["batch_sizes"]:
            row_data = [str(bs)]
            baseline_latency = self.results[baseline_name][bs].get("latency", float("inf"))

            for name in model_names:
                res = self.results[name].get(
                    bs,
                    {
                        "latency": float("inf"),
                        "throughput": 0,
                        "audio_seconds_per_sec": 0,
                    },
                )
                row_data.extend([f"{res['latency']:.2f}", f"{res['audio_seconds_per_sec']:.2f}"])

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
        self._execute_benchmarks()
        self.report(baseline_name=baseline_name)


if __name__ == "__main__":
    # 简单的测试示例
    torch._dynamo.config.capture_scalar_outputs = True
    import types

    TORCH_COMPILE_AVAILABLE = hasattr(torch, "compile")
    root_path = "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/checkpoints/omni_models"
    # --- 1. 配置中心 ---
    BENCHMARK_CONFIG = {
        "model_path": os.path.join(
            root_path,
            "0803_llava_omni_qwen25vl_14B_16x_4k_st2_kimiwhisper_10x_unfreezeaudio_omnidata_text500w_8k",
        ),
        "batch_sizes": [1, 2, 4, 8, 16, 32],
        "warmup_runs": 5,
        "test_runs": 20,
    }

    # --- 2. 准备待测试的模型 ---
    console = Console()
    console.print("[bold blue]1. 正在加载和准备Whisper音频模型...[/bold blue]")

    # 加载配置
    config_path = os.path.join(BENCHMARK_CONFIG["model_path"], "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = types.SimpleNamespace(**config_dict)

    kvargs = {"cache_port": 12345, "data_type": "bfloat16"}

    # 模型A: 标准的 PyTorch Eager 模型
    model_eager = WhisperAudioModel(kvargs)
    model_eager.load_model(BENCHMARK_CONFIG["model_path"], config)
    model_eager.cuda()
    console.print("[green]  -> Eager Whisper模型准备就绪。[/green]")
    models_to_compare = {"PyTorch Eager": model_eager}

    # --- 3. 运行并报告 ---
    console.print("\n[bold blue]2. 开始执行Whisper音频模型基准测试...[/bold blue]")
    runner = WhisperAudioBenchmarkRunner(BENCHMARK_CONFIG)
    baseline_name = "PyTorch Eager"
    runner.run_and_report(models_to_compare, baseline_name=baseline_name)
