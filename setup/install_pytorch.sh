#!/bin/bash
#
# PyTorchをpipでインストールするスクリプト
#
# 使用方法:
#   bash install_pytorch.sh
#
# 環境変数（事前に設定される必要があります）:
#   - PYTORCH_VERSION: PyTorchのバージョン（例: 2.8.0）
#   - TORCHVISION_VERSION: torchvisionのバージョン（例: 0.23.0）
#   - TORCHAUDIO_VERSION: torchaudioのバージョン（例: 2.8.0）
#   - PYTORCH_INDEX_URL: PyTorchのindex-url（例: https://download.pytorch.org/whl/cu128）
#

# --- ユーティリティと設定 ---
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
source "${SCRIPT_DIR}/logging_utils.sh"

trap 'handle_error $? $LINENO "$BASH_COMMAND"' ERR
set -eo pipefail

# --- 必須環境変数のチェック ---
if [ -z "${PYTORCH_VERSION}" ]; then
    log_error "PYTORCH_VERSION環境変数が設定されていません"
    exit 1
fi

if [ -z "${PYTORCH_INDEX_URL}" ]; then
    log_error "PYTORCH_INDEX_URL環境変数が設定されていません"
    exit 1
fi

# --- スクリプト本体 ---
print_header "PyTorchのインストール (pip)"

log_info "PyTorchバージョン: ${PYTORCH_VERSION}"
log_info "torchvisionバージョン: ${TORCHVISION_VERSION:-指定なし}"
log_info "torchaudioバージョン: ${TORCHAUDIO_VERSION:-指定なし}"
log_info "Index URL: ${PYTORCH_INDEX_URL}"

# インストールするパッケージのリストを構築
PACKAGES_TO_INSTALL=("torch==${PYTORCH_VERSION}")

if [ -n "${TORCHVISION_VERSION}" ]; then
    PACKAGES_TO_INSTALL+=("torchvision==${TORCHVISION_VERSION}")
fi

if [ -n "${TORCHAUDIO_VERSION}" ]; then
    PACKAGES_TO_INSTALL+=("torchaudio==${TORCHAUDIO_VERSION}")
fi

log_info "インストールするパッケージ: ${PACKAGES_TO_INSTALL[*]}"

# PyTorchをインストール
start_timer "PyTorchをpipでインストールしています..."

if pip install "${PACKAGES_TO_INSTALL[@]}" --index-url "${PYTORCH_INDEX_URL}"; then
    end_timer
    log_success "PyTorchのインストールが完了しました"
else
    log_error "PyTorchのインストールに失敗しました"
    exit 1
fi

# インストール確認
log_info "PyTorchのインストールを確認しています..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

log_success "PyTorchのインストールと確認が完了しました"

trap - ERR
