import os
import json
import rpyc
import librosa
import numpy as np
import types
import torch
from torch import nn
import torch.nn.functional as F
from io import BytesIO
from typing import List, Union
from safetensors.torch import load_file
from torch.nn.utils.rnn import pad_sequence
from whisper.audio import pad_or_trim, log_mel_spectrogram
from transformers.processing_utils import ProcessorMixin
from lightllm.server.embed_cache.utils import read_shm, get_shm_name_data
from lightllm.server.multimodal_params import AudioItem
from rpyc.utils.classic import obtain
from lightllm.server.embed_cache.embed_cache_client import CpuEmbedCacheClient
import concurrent.futures
import pickle
from lightllm.utils.infer_utils import calculate_cpu_time_sync
from lightllm.models.whisper.modeling_whisper import WhisperModel


# tokenizer_class removed
class WhisperProcessor(ProcessorMixin):
    r"""
    Constructs a Whisper processor which wraps a Whisper feature extractor and a Whisper tokenizer into a single
    processor.

    [`WhisperProcessor`] offers all the functionalities of [`WhisperFeatureExtractor`] and [`WhisperTokenizer`]. See
    the [`~WhisperProcessor.__call__`] and [`~WhisperProcessor.decode`] for more information.

    Args:
        feature_extractor (`WhisperFeatureExtractor`):
            An instance of [`WhisperFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`WhisperTokenizer`):
            An instance of [`WhisperTokenizer`]. The tokenizer is a required input.
    """
    attributes = ["feature_extractor"]
    feature_extractor_class = "WhisperFeatureExtractor"

    def __init__(self, feature_extractor):
        super().__init__(feature_extractor)
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False

    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        return self.tokenizer.get_decoder_prompt_ids(task=task, language=language, no_timestamps=no_timestamps)

    def get_T_after_cnn(self, L_in, dilation=1):
        for (padding, kernel_size, stride) in eval("[(1,3,1)] + [(1,3,2)] "):
            L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
            L_out = 1 + L_out // stride
            L_in = L_out
        return L_out

    def __call__(self, audios, audio_lens, *args, **kwargs):
        """
        Forwards the `audios` argument to WhisperFeatureExtractor's [`~WhisperFeatureExtractor.__call__`] and the `text`
        argument to [`~WhisperTokenizer.__call__`]. Please refer to the doctsring of the above two methods for more
        information.
        """
        # For backward compatibility
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        sampling_rate = kwargs.pop("sampling_rate", 16000)

        audio_lens = np.where(audio_lens <= 480000, audio_lens, 480000)
        audio_lens = audio_lens // 160
        audio_lens_after_cnn = self.get_T_after_cnn(audio_lens)
        padded_inputs = self.feature_extractor(audios, *args, sampling_rate=sampling_rate, **kwargs)

        return padded_inputs["input_features"], audio_lens_after_cnn

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to WhisperTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to WhisperTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def get_prompt_ids(self, text: str, return_tensors="np"):
        return self.tokenizer.get_prompt_ids(text, return_tensors=return_tensors)


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

        x = self.afeat_1d_conv(x.transpose(1, 2)).transpose(
            1, 2
        )  # Process Whisper features with 1D conv: (B x T x D) -> (B x T//2 x D')
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
        # self.audio_processor = WhisperProcessor.from_pretrained(weight_dir)
        # from lightllm.models.whisper.modeling_whisper import WhisperEncoder, WhisperConfig
        if isinstance(config, dict):
            config = types.SimpleNamespace(**config)
        self.audio_model = WhisperModel.from_pretrained(config.audio_encoder).encoder.to(self.data_type)
        self.audio_projector = AudioConvUpScaleProjector(config).to(self.audio_projector_dtype)

        # self.device = torch.device("cpu")
        # self.projector_weights = {}
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
            if "audio_projector" in k:
                audio_projector_weight[k.replace("model.audio_projector.", "")] = params_map[filename][k].to(
                    self.data_type
                )

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

    def encode(self, audio_items: List[AudioItem], cpu_embed_cache_client: CpuEmbedCacheClient):
        # 每个元素是一个chunk
        batch_audios = []
        batch_audio_lens = []
        uuids = []
        items: List[AudioItem] = []
        # 记录每个chunk属于哪个audio_items下标
        chunk_owner_index = []
        for i, item in enumerate(audio_items):
            if isinstance(item, AudioItem):
                uuids.append(item.uuid)
                items.append(item)
                audio_data = read_shm(get_shm_name_data(item.uuid))
                audio = BytesIO(audio_data)
                audio, _ = librosa.load(audio, sr=16000)
            else:
                raise ValueError(f"cannot read audio which type is {type(item)}!")

            # padding to min audio len
            from .defaults import MIN_AUDIO_LEN

            if audio.shape[0] < MIN_AUDIO_LEN:
                audio = np.pad(audio, (0, MIN_AUDIO_LEN - len(audio)), mode="constant", constant_values=0.0)

            if audio.shape[0] > self.max_length:
                start = 0
                while start < audio.shape[0]:
                    end = min(start + self.max_length, audio.shape[0])
                    chunk = audio[start:end]

                    if chunk.shape[0] < MIN_AUDIO_LEN:
                        chunk = np.pad(chunk, (0, MIN_AUDIO_LEN - chunk.shape[0]), mode="constant", constant_values=0.0)
                    batch_audios.append(chunk)
                    batch_audio_lens.append(min(chunk.shape[0], self.max_length))
                    chunk_owner_index.append(i)

                    start = end
            else:
                # min len <audio len < max len
                batch_audio_lens.append(min(audio.shape[0], self.max_length))
                batch_audios.append(audio)
                chunk_owner_index.append(i)

        audio_lens_after_cnn = [len(audio) // 320 for audio in batch_audios]
        chunk_embeds, audio_token_num = self.forward(batch_audios, audio_lens_after_cnn)

        num_audios = len(audio_items)

        per_audio_embeds = [[] for _ in range(num_audios)]

        for chunk_idx, owner in enumerate(chunk_owner_index):
            token_len = int(audio_token_num[chunk_idx])
            if token_len <= 0:
                continue
            # 切割出有效 token，剔除 padding 的部分
            per_audio_embeds[owner].append(chunk_embeds[chunk_idx][:token_len])

        embed_status = self.cache_client.root.get_items_embed_v2(pickle.dumps(uuids))
        ready_audio = pickle.loads(embed_status)

        ids_to_set = []
        for i, ready in enumerate(ready_audio):
            if ready:
                continue

            uid = uuids[i]
            item = items[i]

            # 拼接该 audio 的所有 chunk embedding
            cur_embed = torch.cat(per_audio_embeds[i], dim=0)
            cpu_embed_cache_client.copy_to_cache(
                embed_tensor=cur_embed, start_index_in_cache=item.start_index_in_embed_cache
            )
            ids_to_set.append(uid)

        if ids_to_set:
            self.cache_client.root.set_items_embed_v2(pickle.dumps(ids_to_set))

            torch.cuda.current_stream().synchronize()
