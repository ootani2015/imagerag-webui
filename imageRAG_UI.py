import streamlit as st
import torch
import os
import base64
import gc
from io import BytesIO
from PIL import Image
import numpy as np
import openai
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
from transformers import CLIPVisionModelWithProjection

# オリジナルコードからインポート
from imageRAG_SDXL import * 
from utils import *
from retrieval import *

# --- 設定とユーティリティ ---
def get_image_download_link(img, filename="generated.png", text="画像をダウンロード"):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

def clear_vram():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- UI 構成 ---
st.set_page_config(page_title="ImageRAG ウェブUI", layout="wide")
st.title("🖼️ ImageRAG WebUI")

# --- セッション状態の初期化 ---
if "generating" not in st.session_state:
    st.session_state.generating = False

# --- サイドバー設定 ---
st.sidebar.header("基本設定")
openai_api_key = st.sidebar.text_input("OpenAI API キー", type="password")

st.sidebar.markdown("---")
st.sidebar.subheader("参照画像の選択")

source_choice = st.sidebar.radio(
    "どちらの方法で画像を生成しますか？",
    ("データセットから自動検索", "アップロードした画像を参照"),
    disabled=st.session_state.generating
)

dataset_name = None
user_uploaded_file = None

if source_choice == "データセットから自動検索":
    dataset_root = "datasets"
    if os.path.exists(dataset_root):
        available_datasets = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))]
    else:
        available_datasets = []
    
    if available_datasets:
        dataset_name = st.sidebar.selectbox("使用するデータセットを選択", available_datasets, disabled=st.session_state.generating)
        dataset_path = f"datasets/{dataset_name}"
        
        with st.sidebar.expander("データセットの画像を確認する", expanded=False):
            preview_images = [f for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if preview_images:
                st.write(f"合計: {len(preview_images)} 枚")
                for img_file in preview_images[:10]: 
                    img_ptr = Image.open(os.path.join(dataset_path, img_file))
                    st.image(img_ptr, caption=img_file, use_container_width=True)
else:
    st.sidebar.info("アップロードした画像を元に不足要素を補完します。")
    user_uploaded_file = st.sidebar.file_uploader("参照したい画像をアップロードしてください", type=["png", "jpg", "jpeg"], disabled=st.session_state.generating)

st.sidebar.markdown("---")
out_name = st.sidebar.text_input("出力ファイル名", value="generated_image", disabled=st.session_state.generating)
ip_scale = st.sidebar.slider("IP-Adapter 強度", 0.0, 1.0, 0.4, disabled=st.session_state.generating)

# --- メインエリア ---
st.info("💡 **Tips:** 日本語でも入力可能ですが、英語の方がより正確な画像が生成されやすくなります。")
prompt = st.text_area(
    "プロンプトを入力", 
    placeholder="例：A golden retriever and a cradle. (日本語での入力も自動翻訳されます)", 
    disabled=st.session_state.generating
)

# ボタン制御
def start_generation():
    if not openai_api_key:
        st.error("OpenAI API キーを入力してください。")
    else:
        st.session_state.generating = True

def stop_generation():
    st.session_state.generating = False

col_run, col_stop = st.columns([1, 4])
with col_run:
    st.button("画像生成を開始", on_click=start_generation, disabled=st.session_state.generating)

with col_stop:
    if st.session_state.generating:
        st.button("画像生成を中止", on_click=stop_generation)

# --- 生成プロセス実行 ---
if st.session_state.generating:
    client = openai.OpenAI(api_key=openai_api_key)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, monochrome"

    os.makedirs("results", exist_ok=True)
    col1, col2 = st.columns(2)
    
    try:
        def latents_callback(pipe, step, timestep, callback_kwargs):
            if not st.session_state.generating:
                raise RuntimeError("Manual Stop")
            return callback_kwargs

        with st.status("ImageRAG 実行中...", expanded=True) as status:
            # Step 1: プロンプト翻訳と初期生成
            status.update(label="ステップ 1: プロンプトを解析中...")
            rephrased_prompt = get_enhanced_rephrased_prompt(prompt, client)
            st.write(f"**生成用プロンプト:** {rephrased_prompt}")

            status.update(label="ステップ 1: 初期の画像を生成しています...")
            pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", 
                torch_dtype=torch.float16, variant="fp16", use_safetensors=True
            ).to(device)
            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()
            
            generator = torch.Generator(device="cpu").manual_seed(42)
            
            init_image = pipe(
                prompt=rephrased_prompt, 
                negative_prompt=negative_prompt,
                num_inference_steps=35, 
                generator=generator,
                callback_on_step_end=latents_callback
            ).images[0]
            
            col1.image(init_image, caption="初期生成画像")
            del pipe
            clear_vram()
            
            if not st.session_state.generating: raise RuntimeError("Manual Stop")

            # Step 2: 判定
            status.update(label="ステップ 2: AIが内容を確認中...")
            temp_path = "results/temp_init.png"
            init_image.save(temp_path)
            decision, _ = decision_making(prompt, [temp_path], client)
            
            if not st.session_state.generating: raise RuntimeError("Manual Stop")

            if "YES" in decision.upper():
                st.success("初期画像で完成です！")
                final_image = init_image
            else:
                # Step 3: 参照画像とキャプションの準備
                status.update(label="ステップ 3: 不足要素を補完するための情報を収集中...")
                
                # caption の定義
                caption = "selected reference detail" # デフォルト値

                ref_image_final = None
                
                if source_choice == "アップロードした画像を参照" and user_uploaded_file:
                    status.update(label="ステップ 3: アップロードされた画像を使用します...")
                    ref_image_final = Image.open(user_uploaded_file).convert("RGB")
                    # アップロード時はプロンプトの主要な概念をキャプションとする
                    caption_res = retrieval_caption_generation(prompt, [temp_path], client, k_captions_per_concept=1)
                    caption = convert_res_to_captions(caption_res)[0]
                
                elif source_choice == "データセットから自動検索" and dataset_name:
                    status.update(label="ステップ 3: データセットから参照画像を検索中...")
                    dataset_path = f"datasets/{dataset_name}"
                    retrieval_image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
                    embeddings_path = f"{dataset_path}/embeddings"
                    
                    caption_res = retrieval_caption_generation(prompt, [temp_path], client, k_captions_per_concept=1)
                    caption = convert_res_to_captions(caption_res)[0]
                    paths = retrieve_img_per_caption([caption], retrieval_image_paths, embeddings_path=embeddings_path, k=1, device=device)
                    ref_image_path = np.array(paths).flatten()[0]
                    ref_image_final = Image.open(ref_image_path)
                    st.image(ref_image_final, caption=f"検索された画像: {caption}", width=300)

                if not st.session_state.generating: raise RuntimeError("Manual Stop")

                if ref_image_final is None:
                    st.warning("参照画像が指定されていないため、初期画像を最終結果とします。")
                    final_image = init_image
                else:
                    # Step 4: 再生成
                    status.update(label="ステップ 4: 参照画像を適用して再生成中...")
                    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", torch_dtype=torch.float16
                    ).to(device)
                    
                    pipe_ip = DiffusionPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0", image_encoder=image_encoder,
                        torch_dtype=torch.float16, variant="fp16", use_safetensors=True
                    ).to(device)
                    
                    pipe_ip.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
                    pipe_ip.enable_vae_slicing()
                    pipe_ip.enable_vae_tiling()
                    pipe_ip.set_ip_adapter_scale(ip_scale)
                    
                    # 再生成用プロンプトの組み立て
                    new_prompt = f"According to this image of {caption}, improve the following scene: {rephrased_prompt}"
                    
                    final_image = pipe_ip(
                        prompt=new_prompt,
                        ip_adapter_image=ref_image_final,
                        negative_prompt=negative_prompt,
                        num_inference_steps=35,
                        generator=generator,
                        callback_on_step_end=latents_callback
                    ).images[0]
                    
                    del pipe_ip, image_encoder
                    clear_vram()

            col2.image(final_image, caption="最終画像出力")
            status.update(label="完了！", state="complete")
            st.session_state.generating = False

        st.markdown("---")
        st.markdown(get_image_download_link(final_image, f"{out_name}.png", "生成された画像をダウンロード"), unsafe_allow_html=True)

    except RuntimeError as e:
        if str(e) == "Manual Stop":
            st.warning("生成がユーザーによって中止されました。")
        else:
            st.error(f"実行中にエラーが発生しました: {e}")
        st.session_state.generating = False
        clear_vram()
        st.rerun()
    except Exception as e:
        st.error(f"予期せぬエラーが発生しました: {e}")
        st.session_state.generating = False
        clear_vram()