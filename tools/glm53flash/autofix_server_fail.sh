#!/bin/bash
# autofix_server_fail.sh — vllm 服务起失败的已知症状自动修复
# 输入: 无(自动取最新 server_single_*.log 的错误特征)
# 输出: 0=已应用补丁(或症状已不存在), 可重试; 1=未知症状, 无法自修复
# 补丁均为幂等(锚点不存在=已修过则跳过); 纯 Python 补丁, 服务重启自动生效
set +e
LOGS=/mnt/share/w00608002/work/vllm-ascend/logs
BASE=/mnt/share/w00608002/work/vllm-ascend

L=$(ls -t ${LOGS}/server_single_*.log 2>/dev/null | head -n 1)
[[ -z "$L" ]] && { echo "[autofix] 无服务日志"; exit 1; }

# ===== 补丁表: 症状特征 -> 修复 =====

# ---- 补丁1: kda.py UnboundLocalError saved_quant_config(2026-11-16 实修) ----
if grep -q "UnboundLocalError.*saved_quant_config" "$L" 2>/dev/null; then
    echo "[autofix] 症状匹配: kda.py saved_quant_config 未赋值(cherry-pick 合引入)"
    python3 - <<'PY' && exit 0
import re
p = "/mnt/share/w00608002/work/vllm-ascend/vllm_ascend/models/glm5next/kda.py"
src = open(p).read()
# 幂等锚点: 修复形态(saved_quant_config 赋值在 if kda_quantized 之前)已存在则跳过
if re.search(r"saved_quant_config = vllm_config\.quant_config\n        if kda_quantized:", src):
    print("[autofix] kda.py 已是修复版, 跳过"); raise SystemExit(0)
# 病灶形态: 赋值在 else 分支内
pat = re.compile(
    r"( *)if kda_quantized:\n"
    r" *super\(\)\.__init__\(config, vllm_config, prefix\)\n"
    r" *else:\n"
    r" *saved_quant_config = vllm_config\.quant_config\n")
new, n = pat.subn(r"\1saved_quant_config = vllm_config.quant_config\n\1if kda_quantized:\n"
                  r"\1    super().__init__(config, vllm_config, prefix)\n\1else:\n", src)
if n == 0:
    print("[autofix] kda.py 未匹配病灶形态(可能结构已变), 需人工"); raise SystemExit(1)
open(p, "w").write(new)
import ast; ast.parse(new)
print(f"[autofix] kda.py 补丁已应用({n} 处), 语法校验通过")
PY
    [[ $? -eq 0 ]] && exit 0
fi

# ---- 补丁2(占位): 后续已知症状在此追加, 形如:
# if grep -q "<症状特征>" "$L"; then <幂等修复>; exit 0; fi

echo "[autofix] 未知症状, 最新错误摘要:"
grep -E "Error|error|Traceback" "$L" | tail -3
exit 1
