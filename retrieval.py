import os
import torch
import clip
from open_clip import create_model_from_pretrained, get_tokenizer
import torch.nn.functional as F
from PIL import Image
import numpy as np

# デバイスの自動判定 (M1 Mac対応)
def get_default_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def get_clip_similarities(prompts, image_paths, embeddings_path="", bs=1024, k=50, device=None):
    if device is None:
        device = get_default_device()
    
    # clip.load で指定のデバイスを使用
    model, preprocess = clip.load("ViT-B/32", device=device)
    text = clip.tokenize(prompts).to(device)

    top_text_im_paths = []
    top_text_im_scores = []

    with torch.no_grad():
        text_features = model.encode_text(text)
        normalized_text_vectors = torch.nn.functional.normalize(text_features, p=2, dim=1)

        end = len(image_paths) if bs == len(image_paths) else len(image_paths) - bs
        # ステップが0にならないよう調整
        step = max(1, bs)

        for bi in range(0, len(image_paths), step):
            batch_end = min(bi + step, len(image_paths))
            emb_file = os.path.join(embeddings_path, f"clip_embeddings_b{bi}.pt")
            
            if os.path.exists(emb_file):
                # map_location を現在のデバイスに指定
                normalized_ims = torch.load(emb_file, map_location=device)
                normalized_im_vectors = normalized_ims['normalized_clip_embeddings'].to(device)
                final_bi_paths = normalized_ims['paths']
            else:
                images = []
                valid_paths = []
                for i in range(bi, batch_end):
                    try:
                        img = preprocess(Image.open(image_paths[i])).unsqueeze(0)
                        images.append(img)
                        valid_paths.append(image_paths[i])
                    except Exception as e:
                        print(f"Error loading {image_paths[i]}: {e}")
                        continue
                
                if not images: continue
                
                ims = torch.cat(images).to(device)
                im_features = model.encode_image(ims)
                normalized_im_vectors = torch.nn.functional.normalize(im_features, p=2, dim=1)
                final_bi_paths = valid_paths

            # 類似度計算
            scores = (normalized_text_vectors @ normalized_im_vectors.T)
            
            for i in range(len(prompts)):
                top_k_scores, top_k_indices = torch.topk(scores[i], min(k, len(final_bi_paths)))
                top_text_im_paths.extend([final_bi_paths[idx] for idx in top_k_indices.cpu().numpy()])
                top_text_im_scores.extend(top_k_scores.cpu().numpy())

    # 上位k個を最終抽出
    top_indices = np.argsort(top_text_im_scores)[::-1][:k]
    return [top_text_im_paths[i] for i in top_indices]

def gpt_rerank(caption, image_paths, embeddings_path="", bs=1024, k=3, device=None):
    # CLIPで候補を絞り込む
    initial_k = 50 
    candidate_paths = get_clip_similarities(caption, image_paths, embeddings_path, bs, initial_k, device)
    
    # 論文の実装に基づき、上位候補を返す（本来はここでVLMによる再ランクが入るが簡易化）
    return candidate_paths[:k]

def retrieve_img_per_caption(captions, image_paths, embeddings_path="", k=3, device=None, method='CLIP'):
    if device is None:
        device = get_default_device()
        
    paths = []
    for caption in captions:
        if method == 'CLIP':
            res = get_clip_similarities([caption], image_paths,
                                        embeddings_path=embeddings_path,
                                        bs=min(2048, len(image_paths)), k=k, device=device)
        elif method == 'gpt_rerank':
            res = gpt_rerank(caption, image_paths,
                             embeddings_path=embeddings_path,
                             bs=min(2048, len(image_paths)), k=k, device=device)
        else:
            # SigLIPやMoEも同様にdeviceを渡すように修正が必要（リポジトリ依存）
            res = get_clip_similarities([caption], image_paths, embeddings_path=embeddings_path, k=k, device=device)
        
        paths.append(res)
    return paths