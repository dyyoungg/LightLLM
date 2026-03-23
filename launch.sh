
export PYTHONPATH="/mnt/afs/yangdeyu/dependency/lightllm-dev2:$PYTHONPATH"
export PATH="/opt/conda/bin:$PATH"
which gunicorn

/usr/bin/env /opt/conda/bin/python -m lightllm.server.api_server \
    --run_mode normal \
    --model_dir /mnt/afs/share/20260305_beebee_32B_no_caption_aug_v2_search_silence_system1 \
    --max_req_total_len 8192 \
    --max_total_token_num 20000 \
    --cache_capacity 8000 \
    --llm_decode_att_backend triton \
    --data_type bf16 \
    --port 18003 \
    --tokenizer_mode auto \
    --trust_remote_code \
    --host 0.0.0.0 \
    --use_dynamic_prompt_cache \
    --tp 1 \
    --nccl_port 28765 \
    --mem_fraction 0.9 \
    --graph_max_batch_size 32 \
    --graph_max_len_in_batch 8192 \
    --enable_multimodal \
    --enable_multimodal_audio \
    --visual_nccl_ports 29501 \
    --visual_infer_batch_size 16 \
    --sampling_backend triton_top_kp \
    --enable_concurrent_alloc \
    --quant_type triton-fp8w8a8g128 \

# /usr/bin/env /mnt/afs/yangdeyu/conda_env/llava_4090/bin/python -m lightllm.server.api_server \
#     --run_mode normal \
#     --model_dir /mnt/afs/lijiayi1/code/game_video/test/20250813_beebee \
#     --max_req_total_len 4096 \
#     --max_total_token_num 48000 \
#     --cache_capacity 20000 \
#     --mode ppl_int8kv_flashdecoding \
#     --data_type bf16 \
#     --port 18003 \
#     --tokenizer_mode auto \
#     --trust_remote_code \
#     --host 0.0.0.0 \
#     --use_dynamic_prompt_cache \
#     --tp 1 \
#     --nccl_port 28765 \
#     --mem_fraction 0.9 \
#     --quant_type  vllm-fp8w8a8 \
#     --visual_nccl_ports 29501 \
#     --visual_infer_batch_size 8 \
#     --sampling_backend triton_top_kp \
#     --enable_concurrent_alloc \
#     --enable_multimodal \
#     --graph_max_batch_size 4 \
#     --graph_max_len_in_batch 8192 \
#     --visual_gpu_ids 1 \
#     --audio_gpu_ids 1 