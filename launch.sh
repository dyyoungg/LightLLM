export INPUT_PENALTY=1
export INPUT_PENALTY=ON
export INPUT_PENALTY=TRUE

export PYTHONPATH="/mnt/afs/jiayi/code/lightllm-dev:$PYTHONPATH"
echo $INPUT_PENALTY 

model_path=$1
LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1 LOADWORKER=8 python -m lightllm.server.api_server \
    --run_mode normal \
    --model_dir $model_path \
    --max_req_total_len 32000 \
    --max_total_token_num 50000 \
    --cache_capacity 1200 \
    --mode triton_gqa_flashdecoding \
    --data_type bf16 \
    --port 18003 \
    --tokenizer_mode auto \
    --trust_remote_code \
    --host 0.0.0.0 \
    --use_dynamic_prompt_cache \
    --tp 8 \
    --nccl_port 28765 \
    --mem_fraction 0.9 \
    --graph_max_batch_size 32 \
    --graph_max_len_in_batch 8192 \
    --enable_multimodal \
    --visual_nccl_ports 29501 \
    --visual_infer_batch_size 16 \
    --sampling_backend triton_top_kp \
    --enable_concurrent_alloc \
    --enable_multimodal_audio \

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