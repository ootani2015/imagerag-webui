## 各ディレクトリ・ファイルの役割詳細
プロジェクトの主要なファイルとディレクトリの役割は以下の通りです。
### imageRAG_UI.py:
・システムのメインエントリポイント。Streamlitを使用したWebインターフェースを提供し、プロンプト入力、生成プロセスの制御、画像のプレビューとダウンロードを管理します。
### imageRAG_SDXL.py:
・画像生成のコアロジック。Stable Diffusion XL (SDXL) と IP-Adapter を使用した画像生成パイプラインを定義しています。LLMによるプロンプト拡張機能もここに実装されています。
### environment.yml:
・実行環境定義ファイル。Apple Silicon (M1/M2/M3 Mac) のGPU（MPS）を最大限活用するためのPyTorch設定や、必要なライブラリ一式が定義されています。
### retrieval.py:
・参照画像データセットから、プロンプトに最適な画像を検索・抽出するためのロジックを担います。
### utils.py:
・画像処理や判定ロジックなど、システム全体で使用される共通ユーティリティ関数が含まれています。
### datasets/:
・この中に作成したフォルダ名（例：Tokyo_dataset）が、Web UI上の「使用するデータセットを選択」というプルダウンメニューに自動的に反映されます。

・各フォルダ内には、参照したい画像（.jpg, .png, .jpeg）を自由に配置してください。
## プロジェクトのディレクトリ構成
システムを正しく動作させるためのファイル配置は以下の通りです。
```
project/
├── datasets/                # 参照画像データを格納するルートフォルダ
│   ├── Tokyo_dataset/       # 特定のテーマ（例：東京の風景）のデータセット
│   │   ├── bridge_01.jpg    # 個別の画像ファイル
│   │   ├── bridge_02.jpg
│   │   └── embeddings/      # (自動生成) 検索を高速化するためのベクトルデータ
│   └── Animal_dataset/      # 別のテーマのデータセット
├── results/                 # (自動生成) 生成された画像や一時ファイルが保存される
├── imageRAG_UI.py           # Web UI（Streamlit）のメインプログラム
├── imageRAG_SDXL.py         # 画像生成とプロンプト拡張のコアロジック
├── retrieval.py             # 画像検索（Retrieval）エンジン
├── utils.py                 # 共通ユーティリティ関数
└── environment.yml          # 環境構築用設定ファイル
```
## 環境構築とファイル取得の手順
以下の手順に従って、ローカル環境にシステムを構築します。
### [1] リポジトリのクローンと移動
まず、GitHubからプロジェクトを取得し、ディレクトリに移動します。
```
git clone https://github.com/ootani2015/imagerag-webui
cd imagerag-webui
```
### [2] 仮想環境の作成
environment.yml を使用して、依存関係がすべて解決された専用の仮想環境を作成します。
```
conda env create -f environment.yml
conda activate imagerag-webui
```
### [3] 参照用データセットの準備
datasets フォルダに、参照したい画像群（例: Tokyo_dataset）を配置してください。
## アプリケーションの起動方法
環境構築完了後、imagerag-webui環境をアクティブになっている状態で、以下のコマンドを実行してアプリを起動します。
```
streamlit run imageRAG_UI.py
```
ローカル上でSDXLを動かすのが難しい場合は、以下のコマンドを実行してアプリを起動してください。
```
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
streamlit run ImageRAG_UI.py
```
※ システムによる強制終了の閾値が緩和され、最後まで生成が走りやすくなります。
