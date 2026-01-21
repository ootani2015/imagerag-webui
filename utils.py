import base64
import torch

def convert_res_to_captions(res):
    captions = [c.strip() for c in res.split("\n") if c != ""]
    for i in range(len(captions)):
        if len(captions[i]) > 1:
            if captions[i][0].isnumeric() and captions[i][1] == ".":
                captions[i] = captions[i][2:]
            elif captions[i][0] == "-":
                captions[i] = captions[i][1:]
            elif f"{i+1}." in captions[i]:
                captions[i] = captions[i][captions[i].find(f"{i+1}.")+len(f"{i+1}.")+1:]

        captions[i] = captions[i].strip().replace("'", "").replace('"', '')
    return captions

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def message_gpt(msg, client, image_paths=[], context_msgs=[], images_idx=-1, temperature=0):
    messages = [{"role": "user",
                 "content": [{"type": "text", "text": msg}]
                 }]
    if context_msgs:
        messages = context_msgs + messages

    if image_paths:
        base_64_images = [encode_image(image_path) for image_path in image_paths]
        # images_idxが指定されていない場合、最新のメッセージに画像を追加
        target_idx = images_idx if images_idx != -1 else len(messages) - 1
        
        for i, img in enumerate(base_64_images):
            messages[target_idx]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img}",
                },
            })

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content

# --- 追加した判定ロジック ---

def decision_making(prompt, image_paths, gpt_client):
    """
    VLM (GPT-4o) を使って生成画像がプロンプトを満たしているか判定する
    """
    msg = (
        f"Does the image match the following prompt: '{prompt}'? "
        f"If it does, answer 'YES'. If it does not, answer 'NO' and specify the missing concepts "
        f"in a bulleted list. Answer in the following format: \n"
        f"Decision: <YES/NO>\n"
        f"Missing Concepts: <list>"
    )
    
    ans = message_gpt(msg, gpt_client, image_paths, images_idx=0)
    
    # Decisionの抽出
    decision = "YES" if "Decision: YES" in ans.upper() else "NO"
    
    return decision, ans

def retrieval_caption_generation(prompt, image_paths, gpt_client, k_captions_per_concept=1, decision=True):
    """
    不足しているコンセプトに基づき、検索用のキャプションを生成する
    """
    if decision:
        # 判定ステップ
        res, ans = decision_making(prompt, image_paths, gpt_client)
        if res == "YES":
            return "YES"
        msg2 = f"Based on the prompt: '{prompt}' and the generated image, " \
               f"list the missing concepts that need to be retrieved."
    else:
        msg2 = f"List the main visual concepts in the prompt: '{prompt}'."

    concepts = message_gpt(msg2, gpt_client, image_paths, images_idx=0)
    
    msg3 = f"Generate {k_captions_per_concept} short and simple search query for each concept: {concepts}. " \
           f"Provide only the queries, one per line."
    
    captions = message_gpt(msg3, gpt_client, image_paths, images_idx=0)
    return captions

def get_rephrased_prompt(prompt, gpt_client, image_paths=[], context_msgs=[], images_idx=-1):
    msg = f"Rephrase the following prompt for better image generation, keeping the original meaning: '{prompt}'"
    ans = message_gpt(msg, gpt_client, image_paths, context_msgs=context_msgs, images_idx=images_idx)
    return ans.strip().replace('"', '').replace("'", "")