#!/bin/bash

# ==============================================================================
# ---      LESS 项目环境设置脚本 (v3.0 - 防错最终版)                     ---
# ---      (能自动检测执行方式，并提示用户)                              ---
# ==============================================================================

# --- 核心：检测脚本是否被 source ---
# 如果脚本是被 `bash setup_env.sh` 这样执行的，那么 ${BASH_SOURCE[0]} 和 $0 会相等。
# 如果是被 `source setup_env.sh` 执行的，它们通常会不相等。
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo ""
    echo "❌ 错误：此脚本需要使用 'source' 命令来执行，以修改当前终端的环境。"
    echo "         请不要使用 'bash setup_env.sh'。"
    echo ""
    echo "✅ 请运行以下命令："
    echo "   source ${0}"
    echo ""
    exit 1
fi

# --- 只有在被 source 时，才会执行以下逻辑 ---

# 1. 创建所有必需的缓存目录
echo "--> 步骤 1/3: 在数据盘 /root/autodl-tmp/ 下创建缓存目录..."
mkdir -p /root/autodl-tmp/huggingface_cache/transformers
mkdir -p /root/autodl-tmp/huggingface_cache/datasets
mkdir -p /root/autodl-tmp/modelscope_cache
echo "缓存目录创建/验证完成。"
echo ""

# 2. 检查并写入环境变量到 ~/.bashrc (避免重复)
echo "--> 步骤 2/3: 检查并设置环境变量..."

BASHRC_FILE=~/.bashrc

# 检查并添加 Hugging Face 环境变量
if ! grep -q "HF_HOME=/root/autodl-tmp/huggingface_cache" "$BASHRC_FILE"; then
    echo "正在将 Hugging Face 环境变量写入 $BASHRC_FILE..."
    echo '' >> "$BASHRC_FILE"
    echo '# --- Hugging Face Cache Settings for Data Disk ---' >> "$BASHRC_FILE"
    echo 'export HF_HOME=/root/autodl-tmp/huggingface_cache' >> "$BASHRC_FILE"
    echo 'export TRANSFORMERS_CACHE=/root/autodl-tmp/huggingface_cache/transformers' >> "$BASHRC_FILE"
    echo 'export HF_DATASETS_CACHE=/root/autodl-tmp/huggingface_cache/datasets' >> "$BASHRC_FILE"
else
    echo "Hugging Face 环境变量已存在，跳过写入。"
fi

# 检查并添加 ModelScope 环境变量
if ! grep -q "MODELSCOPE_CACHE=/root/autodl-tmp/modelscope_cache" "$BASHRC_FILE"; then
    echo "正在将 ModelScope 环境变量写入 $BASHRC_FILE..."
    echo '' >> "$BASHRC_FILE"
    echo '# --- ModelScope Cache Settings for Data Disk ---' >> "$BASHRC_FILE"
    echo 'export MODELSCOPE_CACHE=/root/autodl-tmp/modelscope_cache' >> "$BASHRC_FILE"
else
    echo "ModelScope 环境变量已存在，跳过写入。"
fi

# 3. 立即应用 (因为脚本是被source的，所以无需再次source)
echo ""
echo "--> 步骤 3/3: 环境变量已在当前会话中设置！"
echo ""
echo "=============================================================================="
echo "✅ 所有缓存路径设置完成！"
echo "Hugging Face 缓存路径 (HF_HOME): $HF_HOME"
echo "ModelScope 缓存路径 (MODELSCOPE_CACHE): $MODELSCOPE_CACHE"
echo "=============================================================================="
