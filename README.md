# LLM開発環境 自動構築プロジェクト

## 1. 概要 (Overview)

本プロジェクトは、再現性の高い機械学習（特にLLM）向けGPU開発環境を自動で構築するための一連のスクリプト群です。

Condaをベースに環境を作成し、PyTorchやCUDAツールキットのインストール、さらにApex、TransformerEngine、FlashAttentionといったライブラリのソースコードからのビルドまでをワンストップで自動化します。事前学習 (`pretrain`) や評価 (`eval0`) といった複数のユースケースに対応しています。

**元のプロジェクト**: このプロジェクトは`/home/llm-user/team_suzuki/infra/HPC`からコピーし、ローカル環境（Slurm非依存）で動作するように改善されたものです。

-----

## 2. クイックスタート (Quick Start)

最小限の手順で環境を構築できます。

### 前提条件

- NVIDIA GPUとドライバがインストールされていること
- Miniconda/Anacondaがインストールされていること（`$HOME/miniconda3`）
- PyYAMLがインストールされていること：`pip install pyyaml`

### 実行手順

1. **設定ファイルの編集**（オプション）

   必要に応じて `config/env_config.yaml` を編集します：
   ```bash
   vi ~/setup_env/config/env_config.yaml
   ```

2. **環境構築の実行**

   ```bash
   cd ~/setup_env/
   bash submit_local.sh
   ```

3. **環境のアクティベート**

   ```bash
   conda activate env_culture
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
   ```

詳細な設定やカスタマイズについては、「5. 環境構築と使い方」を参照してください。

-----

## 3. 主な特徴 (Features)

* **YAML形式の設定ファイル**: 設定値をYAML形式で管理し、読みやすく編集しやすい構成になっています。
* **自動バージョン生成**: `cuda_toolkit_version`を指定するだけで、PyTorchのCUDAバージョンやindex-urlが自動生成されます。設定の重複を排除し、バージョン不整合を防ぎます。
* **モジュール化された設計**: 設定とロジックが分離されており、各スクリプトが単一の責務を持つため、メンテナンスやカスタマイズが容易です。
* **柔軟なモード切替**: `pretrain`、`eval0`、デフォルトの3つのモードを切り替えることで、用途に応じたライブラリセットを簡単に構築できます。
* **ローカル環境対応**: Slurm環境に依存せず、ローカルマシンで直接実行可能です。

-----

## 4. ディレクトリ構成

```
~/setup_env/
├── config/
│   ├── env_config.yaml           # デフォルト/pretrainモード用の設定ファイル (YAML)
│   ├── env_eval0.yaml            # eval0モード用の設定ファイル (YAML)
│   ├── requirements.txt          # デフォルト/pretrainモード用のpip要件リスト
│   └── requirements-eval0.txt    # eval0モード用のpip要件リスト
│
├── setup/
│   ├── load_yaml_config.py       # YAML設定ファイルを読み込むPythonスクリプト
│   ├── install_conda_libs.sh     # Condaパッケージをインストール
│   ├── install_pip_libs.sh       # Pipパッケージをインストール
│   ├── build_utils.sh            # カスタムライブラリビルド用の共通関数
│   ├── build_apex.sh             # Apexのビルドスクリプト
│   └── ...                       # 他のビルドスクリプト
│
├── test/
│   └── ...
│
└── submit_local.sh               # 全てのセットアップを開始するメインスクリプト
```

-----

## 5. 環境構築と使い方 (Setup and Usage)

以下の手順に従って、環境を構築・利用してください。

### Step 0: 前提条件

* **NVIDIA GPUとドライバ**: NVIDIA GPUと適切なドライバがインストールされていること。
* **Conda**: Miniconda/Anacondaがインストールされていること（`$HOME/miniconda3`を想定）。
* **Python 3**: システムにPython 3がインストールされていること。
* **PyYAML**: PythonでYAMLを読み込むため、PyYAMLパッケージが必要です。
  ```bash
  pip install pyyaml
  ```

### Step 1: 環境設定

構築したい環境に応じて、`config/` ディレクトリ内のYAML設定ファイルを編集します。

* **デフォルト/pretrainモード**: `config/env_config.yaml` を編集します。
* **eval0モード**: `config/env_eval0.yaml` を編集します。

**主な設定項目:**

```yaml
# Conda環境名
conda_env_name: "compe_pretrain"

# Pythonバージョン
python_version: "3.12"

# CUDAバージョン（pytorch_cuda_versionは自動生成されます）
# 例: "12.8.1" → pytorch_cuda_version="12.8" が自動生成
cuda_toolkit_version: "12.8.1"

# PyTorchバージョン（pipでインストール）
pytorch_version: "2.8.0"
torchvision_version: "0.23.0"
torchaudio_version: "2.8.0"
# index-urlは自動生成されます（例: https://download.pytorch.org/whl/cu128）

# ビルド設定
torch_cuda_arch_list: "9.0"  # A100 GPU用のアーキテクチャ

# ソースビルド対象ライブラリのバージョン
apex:
  repo_url: "https://github.com/NVIDIA/apex.git"
  commit: "25.04"

transformer_engine:
  repo_url: "https://github.com/NVIDIA/TransformerEngine.git"
  commit: "release_v2.4"

flash_attention:
  repo_url: "https://github.com/Dao-AILab/flash-attention.git"
  commit: "v2.7.4"
```

**自動設定される項目:**
- `CONDA_ENV_FULL_PATH`: `$HOME/.conda/envs/${conda_env_name}`に自動設定されます。
- `TOOLS_DIR`: `$HOME`に自動設定されます。
- `CONDA_ROOT_PATH`: `$HOME/miniconda3`に自動設定されます。
- `PYTORCH_CUDA_VERSION`: `cuda_toolkit_version`から自動生成されます（例: `"12.8.1"` → `"12.8"`）。
- `PYTORCH_CUDA_INDEX`: CUDAインデックス用の短縮形（例: `"12.8"` → `"cu128"`）。
- `PYTORCH_INDEX_URL`: PyTorchのindex-url（例: `https://download.pytorch.org/whl/cu128`）。
- `CMAKE_CUDA_ARCHITECTURES`: `torch_cuda_arch_list`から自動生成されます（例: `"9.0"` → `"90"`）。

### Step 2: セットアップ実行

設定が完了したら、`submit_local.sh`スクリプトでセットアップを実行します。
このコマンド一つで、全自動で環境が構築されます。

`submit_local.sh`は、オプション引数 `pretrain` または `eval0` を認識します。

* **引数なし (デフォルトモード)**: `config/env_config.yaml`に基づき、基本的なライブラリのみをインストールします。
* **`pretrain` (事前学習モード)**: デフォルトのライブラリに加え、Apex、TransformerEngine、FlashAttentionといった追加のカスタムライブラリをソースからビルド・インストールします。
* **`eval0` (評価モード)**: 評価に特化したライブラリ群をインストールします（ソースビルドは行いません）。デフォルトで `config/env_eval0.yaml` と `config/requirements-eval0.txt` が使用されます。

#### **実行コマンド例**

**例1: デフォルト設定で最小限の環境を構築**

```bash
cd ~/setup_env/
bash submit_local.sh
```

**例2: デフォルト設定で事前学習用の全ライブラリを構築 (`pretrain`モード)**

```bash
cd ~/setup_env/
bash submit_local.sh pretrain
```

**例3: 評価用の環境を構築 (`eval0`モード)**

```bash
cd ~/setup_env/
bash submit_local.sh eval0
```

**例4: カスタム設定ファイルで最小限の環境を構築**

```bash
cd ~/setup_env/
# config/my_env.yaml は事前に作成しておく
bash submit_local.sh config/my_env.yaml
```

**例5: カスタム設定ファイルで事前学習用の全ライブラリを構築**

```bash
cd ~/setup_env/
bash submit_local.sh pretrain config/my_env.yaml
```

### Step 3: 環境の有効化と利用

セットアップが完了すると、`$HOME/.conda/envs/`配下にConda環境が作成されています。

```bash
# 環境をアクティベート（フルパスで指定）
conda activate /home/llm-user/.conda/envs/compe_pretrain

# または環境名で指定（Condaが認識できる場合）
conda activate compe_pretrain

# 環境の確認
which python
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

-----

## 6. YAML設定ファイルについて

本プロジェクトでは、設定をYAML形式で管理しています。YAML設定ファイルは、`setup/load_yaml_config.py`によってbash環境変数に自動変換されます。

### YAML設定の利点

- **読みやすい**: インデントベースの階層構造で、設定内容が一目で理解できます
- **編集しやすい**: シンタックスハイライトやバリデーションが利用できます
- **言語非依存**: bashだけでなく、PythonやRubyなどでも読み込み可能です

### 変数名の変換ルール

YAMLのキー名は、以下のルールでbash環境変数名に変換されます：

- スネークケース → 大文字 + アンダースコア
  - `conda_env_name` → `CONDA_ENV_NAME`
  - `python_version` → `PYTHON_VERSION`

- ネストされた構造 → フラット化
  - `apex.repo_url` → `APEX_REPO_URL`
  - `transformer_engine.commit` → `TRANSFORMER_ENGINE_COMMIT`

### 互換性エイリアス

一部の変数は、既存のビルドスクリプトとの互換性のために短い名前のエイリアスも設定されます：

- `TRANSFORMER_ENGINE_REPO_URL` → `TE_REPO_URL`
- `TRANSFORMER_ENGINE_COMMIT` → `TE_COMMIT`

-----

## 7. トラブルシューティング (Troubleshooting)

### YAMLの読み込みエラー

**エラー**: `Error: Failed to load YAML file`

**対処法**:
- PyYAMLがインストールされているか確認:
  ```bash
  python3 -c "import yaml; print(yaml.__version__)"
  ```
- YAMLファイルの構文が正しいか確認（インデントやコロンの位置など）

### Conda環境の作成に失敗する

**エラー**: `Conda環境 'xxx' の作成に失敗しました`

**対処法**:
- Condaが正しくインストールされているか確認:
  ```bash
  conda --version
  ```
- Conda設定ファイルのパス（`CONDA_ROOT_PATH`）が正しいか確認

### ライブラリのビルドに失敗する (ログに `build` や `cmake` のエラーがある)

**対処法**:
- **CUDAとPyTorchのバージョン不整合**: YAML設定ファイルで指定した`cuda_toolkit_version`と、PyTorchが要求するCUDAバージョンが一致しているか確認してください。
- **ビルド用パッケージの不足**: gcc、g++、nvccなどが正しくインストールされているか確認してください。

### GPU認識エラー

**エラー**: `torch.cuda.is_available()` が `False` を返す

**対処法**:
- NVIDIA ドライバが正しくインストールされているか確認:
  ```bash
  nvidia-smi
  ```
- PyTorchがCUDAサポート付きでインストールされているか確認:
  ```bash
  python -c "import torch; print(torch.version.cuda)"
  ```

-----

## 8. ライセンスと貢献

本プロジェクトは、元のHPCプロジェクトから派生したものです。改善や修正のプルリクエストは歓迎します。
