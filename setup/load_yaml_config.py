#!/usr/bin/env python3
"""
YAMLファイルを読み込み、bash環境変数として出力するスクリプト
"""

import sys
import os
import yaml


def convert_key_to_env_var(key):
    """
    YAMLのキー名を環境変数名に変換
    例: conda_env_name -> CONDA_ENV_NAME
    """
    return key.upper()


def expand_vars(value):
    """
    文字列内の環境変数を展開
    例: $HOME/path -> /home/user/path
    """
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def flatten_dict(d, parent_key='', sep='_'):
    """
    ネストされた辞書をフラットな辞書に変換
    例: {"apex": {"repo_url": "..."}} -> {"apex_repo_url": "..."}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def main():
    if len(sys.argv) != 2:
        print("Usage: python load_yaml_config.py <config.yaml>", file=sys.stderr)
        sys.exit(1)

    config_file = sys.argv[1]

    if not os.path.exists(config_file):
        print(f"Error: Config file not found: {config_file}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error: Failed to load YAML file: {e}", file=sys.stderr)
        sys.exit(1)

    # 辞書をフラット化
    flat_config = flatten_dict(config)

    # 自動設定される値を追加
    home = os.path.expandvars("$HOME")
    conda_env_name = flat_config.get('conda_env_name', 'default_env')

    # 自動設定値
    auto_config = {
        'tools_dir': home,
        'conda_root_path': f"{home}/miniconda3",
        'conda_env_full_path': f"{home}/.conda/envs/{conda_env_name}",
    }

    # build_max_jobsが空の場合はnprocの値を使用
    if 'build_max_jobs' in flat_config and not flat_config['build_max_jobs']:
        import subprocess
        try:
            nproc = subprocess.check_output(['nproc']).decode().strip()
            flat_config['build_max_jobs'] = nproc
        except:
            flat_config['build_max_jobs'] = '1'

    # cuda_toolkit_versionからpytorch_cuda_versionを自動生成
    if 'cuda_toolkit_version' in flat_config and 'pytorch_cuda_version' not in flat_config:
        cuda_toolkit_ver = flat_config['cuda_toolkit_version']
        # "12.8.1" -> "12.8" のようにメジャー.マイナーバージョンを抽出
        parts = cuda_toolkit_ver.split('.')
        if len(parts) >= 2:
            pytorch_cuda_ver = f"{parts[0]}.{parts[1]}"
            flat_config['pytorch_cuda_version'] = pytorch_cuda_ver

            # index-url用のCUDAバージョンを生成（"12.8" -> "cu128"）
            cuda_index_version = f"cu{parts[0]}{parts[1]}"
            flat_config['pytorch_cuda_index'] = cuda_index_version

    # PyTorchのindex-urlを生成
    if 'pytorch_cuda_index' in flat_config:
        pytorch_index_url = f"https://download.pytorch.org/whl/{flat_config['pytorch_cuda_index']}"
        flat_config['pytorch_index_url'] = pytorch_index_url

    # torch_cuda_arch_listからcmake_cuda_architecturesを生成
    if 'torch_cuda_arch_list' in flat_config:
        arch_list = flat_config['torch_cuda_arch_list']
        cmake_arch = arch_list.replace('.', '')
        flat_config['cmake_cuda_architectures'] = cmake_arch

    # 自動設定値をマージ
    flat_config.update(auto_config)

    # 互換性のための変数名エイリアスを追加
    aliases = {
        'transformer_engine_repo_url': 'TE_REPO_URL',
        'transformer_engine_commit': 'TE_COMMIT',
    }

    # bash用の環境変数定義を出力
    for key, value in flat_config.items():
        if value is None or value == '':
            continue

        env_var_name = convert_key_to_env_var(key)
        expanded_value = expand_vars(str(value))

        # bash export文を出力
        print(f"export {env_var_name}='{expanded_value}'")
        print(f"readonly {env_var_name}")

        # エイリアスがある場合は追加で出力
        if key in aliases:
            alias_name = aliases[key]
            print(f"export {alias_name}='{expanded_value}'")
            print(f"readonly {alias_name}")


if __name__ == '__main__':
    main()
