import argparse
import os
from PIL import Image
import numpy as np
import openai
import torch
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
from transformers import CLIPVisionModelWithProjection

from utils import *
from retrieval import *

# デバイスの判定
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

# --- プロンプト翻訳・強化ロジックの極大化 ---
def get_enhanced_rephrased_prompt(user_prompt, client):
    """
    日本語を含むあらゆる言語を、SDXLが高画質に出力できる詳細な英語プロンプトに変換します。
    """
    system_instruction = (
        "You are an expert prompt engineer for Stable Diffusion XL. "
        "Your task is to convert the user's input into a highly descriptive, professional English prompt. "
        "Rules:\n"
        "1. If the input is Japanese, translate it accurately but expand it with rich visual details.\n"
        "2. Always include high-quality tags like 'masterpiece, highly detailed, 8k, extremely aesthetic, cinematic lighting, sharp focus'.\n"
        "3. Describe the environment, texture, and atmosphere specifically.\n"
        "4. Output ONLY the final English prompt text. No chatter."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # 高精度な翻訳と拡張のためにgpt-4oを推奨
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Create a masterpiece prompt based on: {user_prompt}"}
            ],
            temperature=0.6
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Prompt enhancement error: {e}")
        return user_prompt

def run_imagerag_sdxl_pipeline(args):
    device = get_device()
    print(f"Using device: {device}")

    client = openai.OpenAI(api_key=args.openai_api_key)

    # 乱数シードの設定
    generator1 = torch.Generator(device="cpu").manual_seed(args.seed)
    generator2 = torch.Generator(device="cpu").manual_seed(args.seed)

    # パスの設定
    retrieval_image_paths = [os.path.join(f"datasets/{args.dataset}", f) for f in os.listdir(f"datasets/{args.dataset}") if f.endswith(('.png', '.jpg', '.jpeg'))]
    embeddings_path = args.embeddings_path if args.embeddings_path else f"datasets/{args.dataset}/embeddings"
    os.makedirs(args.out_path, exist_ok=True)

    # 共通の高品質ネガティブプロンプト
    negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, monochrome"

    # 1. 初期生成用のパイプライン
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        cache_dir=args.hf_cache_dir
    ).to(device)

    # 2. IP-Adapter パイプライン
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter", 
        subfolder="models/image_encoder",
        torch_dtype=torch.float16
    ).to(device)

    pipe_ip = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        cache_dir=args.hf_cache_dir
    ).to(device)

    pipe_ip.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    pipe_ip.set_ip_adapter_scale(args.ip_scale)

    cur_out_path = os.path.join(args.out_path, f"{args.out_name}.png")
    
    if args.mode == 'sd_first':
        # 強化されたリフレーズ関数を使用
        rephrased_prompt = get_enhanced_rephrased_prompt(args.prompt, client)
        print(f"Original: {args.prompt}")
        print(f"Enhanced Prompt: {rephrased_prompt}")

        # 初回生成（ネガティブプロンプトとステップ数を最適化）
        init_image = pipe(
            prompt=rephrased_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=35,
            generator=generator1,
        ).images[0]
        
        temp_init_path = os.path.join(args.out_path, f"temp_{args.out_name}_init.png")
        init_image.save(temp_init_path)

        decision, ans = decision_making(args.prompt, [temp_init_path], client)
        
        if "NO" in decision:
            caption_res = retrieval_caption_generation(args.prompt, [temp_init_path], gpt_client=client)
            if caption_res == "YES":
                init_image.save(cur_out_path)
                return init_image
            
            caption = convert_res_to_captions(caption_res)[0]
        else:
            init_image.save(cur_out_path)
            return init_image
    else:
        rephrased_prompt = get_enhanced_rephrased_prompt(args.prompt, client)
        caption_res = retrieval_caption_generation(args.prompt, [], gpt_client=client, k_captions_per_concept=1, decision=False)
        caption = convert_res_to_captions(caption_res)[0]

    del pipe
    if device == "mps":
        torch.mps.empty_cache()

    paths = retrieve_img_per_caption([caption], retrieval_image_paths, embeddings_path=embeddings_path,
                                        k=1, device=device, method=args.retrieval_method)
    image_path = np.array(paths).flatten()[0]
    ref_image = Image.open(image_path)

    # ImageRAG による最終生成
    final_image = pipe_ip(
        prompt=rephrased_prompt,
        ip_adapter_image=ref_image,
        negative_prompt=negative_prompt,
        num_inference_steps=35,
        generator=generator2,
    ).images[0]

    final_image.save(cur_out_path)
    return final_image