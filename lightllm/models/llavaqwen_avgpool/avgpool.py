import torch
import torch.nn as nn
from einops import rearrange
import math
import numpy as np
from functools import lru_cache


def build_eos_tokens(num_eos_tokens: int, output_hidden_size: int):
    # think tokens

    if num_eos_tokens:
        eos_tokens = torch.nn.Parameter(torch.zeros(1, num_eos_tokens, output_hidden_size))
        nn.init.normal_(eos_tokens, mean=0.0, std=0.02)
    else:
        eos_tokens = None

    return eos_tokens


def build_mlp(depth, hidden_size, output_hidden_size):
    layers = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        layers.append(nn.SiLU())
        layers.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*layers)


class PoolPojector(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config
        self.pooling = nn.AdaptiveAvgPool2d((config["proj_output_size"]["height"], config["proj_output_size"]["width"]))
        self.mlp1 = build_mlp(2, config["mm_hidden_size"], config["mm_hidden_size"] * 4)
        self.mlp2 = build_mlp(2, config["mm_hidden_size"] * 4, config["hidden_size"])
        self.mlp3 = build_mlp(2, config["mm_hidden_size"], config["hidden_size"])
        self.eos_tokens = build_eos_tokens(1, config["hidden_size"])

    def forward(self, x):
        x = rearrange(
            x,
            "b (h w) d -> b d h w",
            h=self.config["mm_image_size"]["height"] // self.config["mm_patch_size"],
            w=self.config["mm_image_size"]["width"] // self.config["mm_patch_size"],
        )

        x = self.pooling(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.mlp3(x)
        B = x.size(0)
        x = torch.cat([x, self.eos_tokens.expand(B, -1, -1)], dim=1)
        return x


class AudioAvgPoolProjector(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config
        self.audio_avg_pooler = nn.AvgPool1d(
            self.config.audio_downsample_ratio,
            stride=self.config.audio_downsample_ratio,
        )
        self.linear1 = nn.Linear(
            in_features=self.config.audio_hidden_size,
            out_features=self.config.hidden_size,
            bias=True,
        )
        self.relu = nn.GELU()  # or Silu？ minicpm用的relu
        self.linear2 = nn.Linear(
            in_features=self.config.hidden_size,
            out_features=self.config.hidden_size,
            bias=True,
        )

    def forward(self, x):
        # x [bs, seq, hidden]

        x = self.audio_avg_pooler(x.transpose(1, 2)).transpose(1, 2)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


@lru_cache(maxsize=100)
def get_adaptive_pool_size(M, N, scale=20):

    r = 1 / math.sqrt(scale)
    Mh = max(1, int(np.round(M * r)))
    Nw = max(1, int(np.round(N * r)))
    return Mh, Nw


class Qwen25VLAvgPoolProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mm_downsample_ratio = self.config.mm_downsample_ratio
        self.hidden_size = config.mm_hidden_size
        self.merge_size = config.mm_merge_size
        in_hidden = self.hidden_size
        out_hidden = config.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(in_hidden, in_hidden),
            nn.GELU(),
            nn.Linear(in_hidden, out_hidden),
        )

    def forward(self, images_feature, images_thw, merge_size=None):
        """
        Args:
            images: Tensor of shape [N, hidden_size]
            images_thw: Tensor of shape [m, 3], each row is [t, h, w]
                       There are m sequences; each sequence corresponds to t*h*w vectors in `images`.
        Returns:
            Tensor of shape [m, out_hidden]
        """
        outputs = []
        start = 0
        seq_len = []

        if merge_size is None:
            merge_size = torch.tensor([self.merge_size] * images_thw.shape[0])

        for thw, each_merge_size in zip(images_thw, merge_size):
            t, h, w = thw.cpu().tolist()
            h, w = int(h / each_merge_size.item()), int(w / each_merge_size.item())
            length = int(t * h * w)
            img_seq = images_feature[start : start + length]  # [t*h*w, hidden]
            start += length

            # reshape to [t, h, w, hidden] -> permute to [hidden, t, h, w]
            img_feat = img_seq.view(t, h, w, -1).permute(3, 0, 1, 2)  # [hidden, t, h, w]

            Mh, Nw = get_adaptive_pool_size(h, w, scale=self.mm_downsample_ratio)
            pool = nn.AdaptiveAvgPool2d((Mh, Nw))  # pool on H, W

            pooled = pool(img_feat)  # [hidden, t, Mh, Nw]
            pooled = pooled.permute(1, 2, 3, 0).contiguous()  # [t, Mh, Nw, hidden]

            pooled = pooled.view(-1, self.hidden_size)  # flatten: [t * Mh * Nw, hidden]
            # print(pooled.shape[0], (Mh, Nw), thw)
            # TODO 是否需要 加特殊token
            tokens = [pooled.shape[0] // t] * t  # calculate each image
            outputs.append(pooled)
            seq_len.extend(tokens)

        outputs = torch.cat(outputs, dim=0)  # [m, hidden_size]
        projected = self.mlp(outputs)  # [m, out_hidden]

        return projected, seq_len
