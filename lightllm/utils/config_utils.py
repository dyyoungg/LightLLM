import json
import os
from typing import Optional
from functools import lru_cache
from .envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def get_config_json(model_path: str):
    with open(os.path.join(model_path, "config.json"), "r") as file:
        json_obj = json.load(file)
    return json_obj


def get_hidden_size(model_path: str) -> Optional[int]:
    # try to get hidden_size in config.json
    config_json = get_config_json(model_path)
    hidden_size = None
    for cfg in (
        config_json,
        config_json.get("llm_config", {}),
        config_json.get("text_config", {}),
        config_json.get("foundation_config", {}),
    ):
        if isinstance(cfg, dict) and cfg.get("hidden_size") is not None:
            hidden_size = cfg["hidden_size"]
            break

    if isinstance(hidden_size, int):
        return hidden_size

    logger.error("cannot get hidden size from config.json, return None instead")
    return None


@lru_cache(maxsize=None)
def get_num_key_value_heads(model_path: str) -> int:
    config_json = get_config_json(model_path)
    for cfg in (
        config_json,
        config_json.get("llm_config", {}),
        config_json.get("text_config", {}),
        config_json.get("foundation_config", {}),
    ):
        if isinstance(cfg, dict) and cfg.get("num_key_value_heads") is not None:
            return cfg["num_key_value_heads"]
    raise KeyError("cannot find num_key_value_heads in config.json")


def get_eos_token_ids(model_path: str):
    config_json = get_config_json(model_path)
    # Different multimodal checkpoints place the language model config at
    # different nesting levels.  VeOmni stores it under ``foundation_config``
    # and may keep a top-level ``eos_token_id`` with value ``null``.  Probe
    # each supported location explicitly so that a valid token id of 0 is not
    # treated as missing.
    eos_token_id = None
    for cfg in (
        config_json,
        config_json.get("llm_config", {}),
        config_json.get("text_config", {}),
        config_json.get("foundation_config", {}),
    ):
        if isinstance(cfg, dict) and cfg.get("eos_token_id") is not None:
            eos_token_id = cfg["eos_token_id"]
            break

    if eos_token_id is None:
        raise KeyError("cannot find eos_token_id in config.json")

    if isinstance(eos_token_id, int):
        return [eos_token_id]
    if isinstance(eos_token_id, list):
        return eos_token_id
    assert False, "error eos_token_id format in config.json"


def get_model_architectures(model_path: str):
    try:
        config_json = get_config_json(model_path)
        arch = config_json["architectures"][0]
        return arch
    except:
        logger.error("can not get architectures from config.json, return unknown_architecture")
        return "unknown_architecture"


def get_vocab_size(model_path: str):
    try:
        config_json = get_config_json(model_path)
        for cfg in (
            config_json,
            config_json.get("llm_config", {}),
            config_json.get("text_config", {}),
            config_json.get("foundation_config", {}),
        ):
            if isinstance(cfg, dict) and cfg.get("vocab_size") is not None:
                return int(cfg["vocab_size"])
        raise KeyError("cannot find vocab_size in config.json")
    except:
        logger.error("can not get vocab_size from config.json, return 0")
        return 0


def get_dtype(model_path: str):
    config_json = get_config_json(model_path)
    for cfg in (
        config_json,
        config_json.get("llm_config", {}),
        config_json.get("text_config", {}),
        config_json.get("foundation_config", {}),
    ):
        if not isinstance(cfg, dict):
            continue
        torch_dtype = cfg.get("torch_dtype", cfg.get("dtype"))
        if torch_dtype is not None:
            return torch_dtype
    logger.warning("torch_dtype not in config.json, use float16 as default")
    return "float16"


@lru_cache(maxsize=None)
def get_fixed_kv_len():
    start_args = get_env_start_args()
    model_cfg = get_config_json(start_args.model_dir)
    if "prompt_cache_token_ids" in model_cfg:
        return len(model_cfg["prompt_cache_token_ids"])
    else:
        return 0
