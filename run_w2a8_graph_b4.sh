#!/usr/bin/env bash
# /dev/shm 守门
shm_size_guard() {
    local avail
    avail=$(df -m /dev/shm 2>/dev/null | awk 'NR==2{print $4}')
    if [[ -n "$avail" && "$avail" -lt 1024 ]]; then
        mount -o remount,size=16G /dev/shm 2>/dev/null || echo "[shm-guard] remount /dev/shm 失败(权限?),继续" >&2
    fi
}
shm_size_guard

# E1: W2A8(VQ2A8) · DP8 单机 · 图模式(full_graph) 基线
# VQ2A8 硬约束: ① 图模式必须 GS32 full_graph 形态(mode=0+FULL_DECODE_ONLY+capture_sizes=[1]
#   +max_num_seqs=1(B1 图) → 与 batch 128 互斥;吞吐靠 DP8 并发 8×B1)
#   ② 同步调度(--no-async-scheduling) ③ additional-config 需带 vq2a8 块(gs32 产物/执行模式/eplb off)
#   ④ block-size 32 ⑤ DP8 需 gs32_graph_routed_rows=dp_size×topk=48
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_EXEC_TIMEOUT=3600
export ACL_DEVICE_SYNC_TIMEOUT=3600

REPO=/mnt/share/w00608002/work/eplb/vllm-ascend

vllm serve /mnt/share/w00608002/weight/DeepSeek-V4-Flash-0731-vq2a8-mixed-gs32-repack \
--max-model-len 135000 \
--safetensors-load-strategy 'prefetch' \
--max-num-batched-tokens 4096 \
--served-model-name dsv \
--gpu-memory-utilization 0.9 \
--enable-expert-parallel \
--no-async-scheduling \
--max-num-seqs 4 \
--port 9000 \
--block-size 32 \
--no-enable-prefix-caching \
--tokenizer-mode deepseek_v4 \
--tool-call-parser deepseek_v4 \
--enable-auto-tool-choice \
--reasoning-parser deepseek_v4 \
--data-parallel-size 8 \
--api-server-count 1 \
--compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4], "max_cudagraph_capture_size": 4, "cudagraph_num_of_warmups": 1}' \
--profiler-config '{"profiler": "torch", "torch_profiler_dir": "/home/w00608002/vllm_profiling", "torch_profiler_with_stack": false}' \
--additional-config '{"enable_cpu_binding": "True", "ascend_compilation_config": {"enable_npugraph_ex": false}, "multistream_dsv4_dsa_overlap": false, "vq2a8": {"gs32_library": "'$REPO'/build/vq2a8-gs32-runtime/libvq2a8_gs32_stage.so", "gs32_binary": "'$REPO'/build/vq2a8-gs32-runtime/mixed_w2w4_moe_ffn.o", "gs32_execution": "full_graph", "gs32_fusion": "none", "gs32_graph_routed_rows": 192, "eplb": "off", "eplb_options": {"K": 8, "R_vq2": 16, "R_mx4": 8, "n_stage": 2, "w_lo": 0.2, "w_hi": 5.0, "ewma_lambda": 0.05}}}'
