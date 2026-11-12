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

# E1b: W2A8 · DP8 · serial+eager · block128 · seqs=128
# 降级原因: 图模式 full_graph 在 DP8 捕获阶段报 "GS32 whole-graph execution supports at most 8 routed rows"
# (capture 输入与 B1 约束不匹配,算子适配缺口) → 按高并发诉求转 eager
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
--enforce-eager \
--profiler-config '{"profiler": "torch", "torch_profiler_dir": "/home/w00608002/vllm_profiling", "torch_profiler_with_stack": false}' \
--additional-config '{"enable_cpu_binding": "True", "ascend_compilation_config": {"enable_npugraph_ex": false}, "multistream_dsv4_dsa_overlap": false, "vq2a8": {"gs32_library": "'$REPO'/build/vq2a8-gs32-runtime/libvq2a8_gs32_stage.so", "gs32_binary": "'$REPO'/build/vq2a8-gs32-runtime/mixed_w2w4_moe_ffn.o", "gs32_execution": "serial", "gs32_fusion": "none", "eplb": "off", "eplb_options": {"K": 8, "R_vq2": 16, "R_mx4": 8, "n_stage": 2, "w_lo": 0.2, "w_hi": 5.0, "ewma_lambda": 0.05}}}'
