export PYTHONPATH="/mnt/afs/yangdeyu/dependency/lightllm-dev:$PYTHONPATH"
set -x

export PATH="/opt/conda/bin:$PATH"
which gunicorn
export RMSNORM_WARPS=4

/usr/bin/env /opt/conda/bin/python -m lightllm.server.api_server \
    --run_mode normal \
    --model_dir /mnt/afs/share/20260305_beebee_32B_no_caption_aug_v2_search_silence_system1 \
    --max_req_total_len 8192 \
    --max_total_token_num 21000 \
    --cache_capacity 12000 \
    --mode triton_gqa_flashdecoding \
    --data_type bf16 \
    --port 18003 \
    --tokenizer_mode auto \
    --trust_remote_code \
    --host 0.0.0.0 \
    --use_dynamic_prompt_cache \
    --tp 1 \
    --nccl_port 28765 \
    --mem_fraction 0.9 \
    --quant_type  vllm-fp8w8a8 \
    --visual_nccl_ports 29501 \
    --visual_infer_batch_size 16 \
    --sampling_backend triton_top_kp \
    --enable_concurrent_alloc \
    --enable_multimodal \
    --enable_multimodal_audio \
    --graph_max_batch_size 4 \
    --graph_max_len_in_batch 4096 \
    --visual_gpu_ids 0 \
    --audio_gpu_ids 0 \
    --chunked_prefill_size 4096 \
 
