"""Compatibility helpers for VeOmni LlavaQwen2 checkpoints."""

from copy import deepcopy


def is_veomni_config(config):
    return config.get("model_type") == "llavaqwen2_omni" or (
        "foundation_config" in config and "encoder_config" in config
    )


def normalize_veomni_config(config):
    """Flatten a VeOmni config to the LlavaQwen2 config used by LightLLM."""
    if not is_veomni_config(config):
        return config

    config = deepcopy(config)
    foundation = config.get("foundation_config", {})
    encoder = config.get("encoder_config", {})
    image = encoder.get("image_config", {})
    audio = encoder.get("audio_config", {})

    normalized = dict(config)
    normalized.update(foundation)
    normalized["model_type"] = "llavaqwen2"
    normalized["architectures"] = ["LlavaQwen2ForCausalLM"]
    normalized["torch_dtype"] = config.get("dtype", foundation.get("torch_dtype", "bfloat16"))
    normalized["compute_dtype"] = normalized["torch_dtype"]

    rope_parameters = foundation.get("rope_parameters", {})
    if "rope_theta" not in normalized and "rope_theta" in rope_parameters:
        normalized["rope_theta"] = rope_parameters["rope_theta"]

    if image:
        normalized["mm_vision_tower"] = [image.get("_name_or_path", "/mnt/afs/share/qwen25_vl_encoder")]
        normalized["mm_hidden_size"] = image.get("output_size", normalized.get("hidden_size", 5120))
        normalized["mm_projector_type"] = image.get("image_projector_type", "dynamic_avgpool")
        normalized["mm_downsample_ratio"] = image.get("image_downsample_size", 16)
        normalized["mm_patch_size"] = image.get("patch_size", 14)
        normalized["mm_merge_size"] = image.get("spatial_merge_size", 2)
        normalized["mm_head_num"] = image.get("num_heads", 16)
        normalized["mm_vision_select_feature"] = "patch"
        normalized["mm_vision_select_layer"] = -2
        normalized["mm_use_im_patch_token"] = False
        normalized["mm_use_im_start_end"] = False

    if audio:
        normalized["audio_encoder"] = audio.get("_name_or_path")
        normalized["audio_hidden_size"] = audio.get("d_model", 1280)
        normalized["audio_downsample_ratio"] = audio.get("audio_downsample_size", 10)
        normalized["audio_projector_type"] = audio.get("audio_projector_type", "conv_channel_upscale")

    return normalized


def remap_veomni_weight_key(key):
    """Map one VeOmni state-dict key to the legacy LlavaQwen2 name."""
    if key.startswith("audio_encoder.audio_projector."):
        return "model.audio_projector." + key[len("audio_encoder.audio_projector.") :]
    if key.startswith("audio_encoder."):
        return "model.audio_encoder.model." + key[len("audio_encoder.") :]
    if key.startswith("image_encoder.mm_projector."):
        return "model.mm_projector." + key[len("image_encoder.mm_projector.") :]
    if key.startswith("image_encoder."):
        return "model.vision_tower.vision_tower." + key[len("image_encoder.") :]
    return key
