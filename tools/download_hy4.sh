#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -uo pipefail

readonly REPO="tencent/Hy4-preview"
readonly BRANCH="main"
readonly BASE_URL="https://hf-mirror.com/${REPO}/resolve/${BRANCH}"
readonly SAVE_DIR="/home/g00872988/models"
readonly TOTAL_SHARDS=131

file_size() {
    if [[ -f "$1" ]]; then
        wc -c < "$1"
    else
        printf '0\n'
    fi
}

download_one() {
    local filename target previous_size previous_time current_size current_time pid status
    printf -v filename 'model-%05d-of-%05d.safetensors' "$1" "$TOTAL_SHARDS"
    target="${SAVE_DIR}/${filename}"
    printf '\n开始下载：%s\n保存位置：%s\n下载地址：%s/%s\n' \
        "$filename" "$target" "$BASE_URL" "$filename"

    previous_size=$(file_size "$target")
    previous_time=$(date +%s)
    curl --location --fail --continue-at - \
        --retry 5 --retry-delay 3 --connect-timeout 30 \
        --silent --show-error \
        --output "$target" "${BASE_URL}/${filename}" &
    pid=$!
    trap 'kill "$pid" 2>/dev/null || :; wait "$pid" 2>/dev/null || :; printf "\n下载中断，保留文件供断点续传。\n" >&2; exit 130' INT
    trap 'kill "$pid" 2>/dev/null || :; wait "$pid" 2>/dev/null || :; printf "\n下载中断，保留文件供断点续传。\n" >&2; exit 143' TERM

    while kill -0 "$pid" 2>/dev/null; do
        sleep 1
        current_size=$(file_size "$target")
        current_time=$(date +%s)
        awk -v size="$current_size" -v previous="$previous_size" \
            -v elapsed="$((current_time - previous_time))" '
            BEGIN {
                delta = size - previous;
                speed = (elapsed > 0 && delta > 0) ? delta / elapsed / 1000000 : 0;
                printf "\r已保存：%.2f MB | 下载速度：%.2f MB/s          ", size / 1000000, speed;
            }'
        previous_size=$current_size
        previous_time=$current_time
    done
    wait "$pid"
    status=$?
    trap - INT TERM
    printf '\n'
    if [[ "$status" -eq 0 ]]; then
        printf '[成功] %s 下载完成。\n' "$filename"
    else
        printf '[失败] %s，curl 退出码：%s；重新运行相同命令可断点续传。\n' "$filename" "$status" >&2
    fi
    return "$status"
}

main() {
    local value id failed=0
    local ids=()
    if [[ "$#" -eq 0 ]]; then
        printf '用法：bash %s <编号1> [编号2] ...\n' "$0"
        printf '单个：bash %s 1\n多个：bash %s 1 5 10\n' "$0" "$0"
        printf '范围：bash %s $(seq 1 10)\n' "$0"
        return 2
    fi
    for value in "$@"; do
        if [[ ! "$value" =~ ^[0-9]{1,5}$ ]]; then
            printf '编号必须是 1～%s 的整数：%s\n' "$TOTAL_SHARDS" "$value" >&2
            return 2
        fi
        id=$((10#$value))
        if ((id < 1 || id > TOTAL_SHARDS)); then
            printf '编号超出 1～%s：%s\n' "$TOTAL_SHARDS" "$value" >&2
            return 2
        fi
        ids+=("$id")
    done
    command -v curl >/dev/null || { printf '请先安装 curl。\n' >&2; return 1; }
    mkdir -p "$SAVE_DIR" || return 1
    for id in "${ids[@]}"; do
        download_one "$id" || failed=1
    done
    return "$failed"
}

main "$@"
