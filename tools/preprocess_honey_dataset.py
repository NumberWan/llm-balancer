import os
import sys
SAFE_CACHE_DIR = "/home/hf_cache_w00917303" 

# 自動建立目錄
os.makedirs(SAFE_CACHE_DIR, exist_ok=True)

print(f"[System] Redirecting all cache and temp files to: {SAFE_CACHE_DIR}")

# 1. 轉移 Hugging Face (Datasets & Transformers) 的緩存
os.environ["HF_DATASETS_CACHE"] = SAFE_CACHE_DIR
os.environ["HF_HOME"] = SAFE_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = SAFE_CACHE_DIR

# 2. 轉移 Python/OS 的系統暫存 (這是導致 Docker 膨脹的主因)
# 當 datasets 處理 map 函數時，pyarrow 會大量寫入這裡
os.environ["TMPDIR"] = SAFE_CACHE_DIR
os.environ["TEMP"] = SAFE_CACHE_DIR
os.environ["TMP"] = SAFE_CACHE_DIR
# =================================================

# =================================================
# 全局變量說明
# =================================================
# 以下函數依賴於在 if __name__ == '__main__' 中定義的全局變量：
# 
# - task_type: 任務類型 (0=回歸, 1=二分類, 2=多分類)
# - FLAG_VICUNA_DATA_ONLY: 是否只處理單一模型數據
# - FLAG_FIRST_ROUND_ONLY: 是否只處理第一輪對話
# - FLAG_HEAD_TAIL: 是否使用 head-tail 截斷策略
# - FLAG_USE_LOG_LOSS: 是否使用 log-space labels
# - multi_cls_thresholds: 分類任務的閾值列表
# - vicuna_tokenizer: tokenizer（用於計算 token 數量）
# - bert_tokenizer: tokenizer（用於文本 tokenization）
# - model_name_to_idx: 模型名稱到索引的映射字典
# - num_models: 模型總數
# - percentiles: 每個模型的分位數列表（用於分類任務）
#
# 這些變量必須在調用 preprocess_dataset 之前定義。
# =================================================
import datasets
from datasets import load_dataset, Dataset, Features, Value
import transformers
from transformers import BlipProcessor
from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import torch
import glob
import io
import numpy as np
import multiprocessing

def exact_multi_round_prompt(dataset):
    """
    處理多輪對話數據集。
    
    全局變量依賴：
    - vicuna_tokenizer: tokenizer 用於計算 token 數量
    - task_type: 任務類型 (0=回歸, 1=二分類, 2=多分類)
    - multi_cls_thresholds: 分類任務的閾值列表
    - FLAG_USE_LOG_LOSS: 是否使用 log-space labels
    """
    df = dataset.to_pandas()
    ans_df = pd.DataFrame(columns=['prompt', 'labels', 'num_tokens', 'conversation_id', 'turn_id', 'images'])

    n_illegal_samples = 0
    for conversation_id in range(len(df)):
        if conversation_id % 10000 == 0:
            print('Processing conversation ' + str(conversation_id))
        sample = df.iloc[conversation_id]
        conversations = sample['conversations']
        images_data = sample.get('images', [])
        dialogue_so_far = ''

        new_samples = {'prompt': [], 
                       'labels': [], 
                       'num_tokens': [], 
                       'conversation_id': [], 
                       'turn_id': [],
                       'images': []}

        turn_id = 0
        for i, sentence in enumerate(conversations):
            if sentence.get('from') == 'human' or sentence.get('role') == 'user':
                dialogue_so_far += '[USER]: ' + sentence.get('value', sentence.get('content', '')) + '\n'
            elif sentence.get('from') == 'gpt' or sentence.get('role') == 'assistant':
                assistant_content = sentence.get('value', sentence.get('content', ''))

                encoded_response = vicuna_tokenizer(assistant_content, truncation=False)
                num_tokens = len(encoded_response['input_ids'])
                
                # Drop abnormal samples that have empty responses or might have been truncated.
                if num_tokens <= 1 or num_tokens >= 512:
                    n_illegal_samples += 1
                    break

                # Add a new prediction sample
                new_samples['prompt'].append(dialogue_so_far)
                new_samples['conversation_id'].append(conversation_id)
                new_samples['images'].append(images_data)  # 保留圖像數據
                new_samples['turn_id'].append(turn_id)
                new_samples['num_tokens'].append(num_tokens)
                
                if task_type == 0:
                    # 回歸任務：根據 FLAG_USE_LOG_LOSS 決定是否對標籤取 log
                    if FLAG_USE_LOG_LOSS:
                        import math
                        new_samples['labels'].append(math.log(num_tokens + 1.0))
                    else:
                        new_samples['labels'].append(num_tokens)
                else:
                    # 分類任務：使用閾值分類
                    # 確保所有樣本都會被分配標籤（最後一個閾值應該是一個很大的數）
                    label_assigned = False
                    for j, thresh in enumerate(multi_cls_thresholds):
                        if num_tokens < thresh:
                            new_samples['labels'].append(j)
                            label_assigned = True
                            break
                    # 如果沒有被分配標籤（理論上不應該發生，因為最後一個閾值很大），分配最後一個類別
                    if not label_assigned:
                        new_samples['labels'].append(len(multi_cls_thresholds) - 1)
                dialogue_so_far += '[ASSISTANT]: ' + assistant_content + '\n'
                turn_id += 1

        new_samples = pd.DataFrame(new_samples)
        ans_df = pd.concat([ans_df, new_samples], ignore_index=True)

    ans_dataset = Dataset.from_pandas(ans_df)
    print('Number of illegal samples: ', n_illegal_samples)
    return ans_dataset


def extract_first_round_prompt(example):
    """
    提取第一輪對話的用戶提示和助手回應。
    
    全局變量依賴：
    - vicuna_tokenizer: tokenizer 用於計算 token 數量
    - task_type: 任務類型 (0=回歸, 1=二分類, 2=多分類)
    - multi_cls_thresholds: 分類任務的閾值列表
    - FLAG_USE_LOG_LOSS: 是否使用 log-space labels
    """
    conversations = example['conversations']
    user_content = ''
    
    # Combining the sentences from the first-round of the user prompt
    i = 0
    for sentence in conversations:
        if sentence.get('from') == 'human' or sentence.get('role') == 'user':
            if i > 0:
                user_content += '\n'
            user_content += sentence.get('value', sentence.get('content', ''))
            i += 1
        else:
            break
    
    # 如果對話以助手消息開始（沒有用戶消息），跳過這個樣本
    if i == 0:
        example['prompt'] = ''
        example['num_tokens'] = 0
        example['labels'] = 0
        return example
    
    # Combining the sentences from the first-round of the assistant response
    assistant_content = ''
    for j, sentence in enumerate(conversations):
        if sentence.get('from') == 'gpt' or sentence.get('role') == 'assistant':
            if j > 0:
                assistant_content += '\n'
            assistant_content += sentence.get('value', sentence.get('content', ''))
        elif j > i:
            break

    example['prompt'] = user_content
    encoded_response = vicuna_tokenizer(assistant_content, truncation=False)
    num_tokens = len(encoded_response['input_ids'])
    example['num_tokens'] = num_tokens
    
    if task_type == 0:
        # 回歸任務：根據 FLAG_USE_LOG_LOSS 決定是否對標籤取 log
        if FLAG_USE_LOG_LOSS:
            import math
            example['labels'] = math.log(num_tokens + 1.0)  # log(tokens + 1)
        else:
            example['labels'] = num_tokens
    else:
        # 分類任務：使用閾值分類
        # 確保所有樣本都會被分配標籤（最後一個閾值應該是一個很大的數）
        label_assigned = False
        for i, thresh in enumerate(multi_cls_thresholds):
            if num_tokens < thresh:
                example['labels'] = i
                label_assigned = True
                break
        # 如果沒有被分配標籤（理論上不應該發生，因為最後一個閾值很大），分配最後一個類別
        if not label_assigned:
            example['labels'] = len(multi_cls_thresholds) - 1
    
    return example


def process_images(example):
    """
    處理圖像數據，統一調整大小並轉換為 JPEG bytes 格式。
    返回的 'image' 欄位將是純 bytes，配合 Features(image=Value("binary")) 使用。
    關鍵修復：使用 .load() 強制加載圖像到內存，斷開文件關聯，解決 _idat/fileno 問題。
    """
    # 預設為空 bytes
    image_bytes = b''
    
    images_data = example.get('images', [])
    
    # 取第一張圖
    current_image = None
    if isinstance(images_data, list) and len(images_data) > 0:
        current_image = images_data[0]
    elif images_data:
        current_image = images_data
        
    if current_image:
        try:
            img_obj = None
            # 1. 統一讀取為 PIL Image 對象
            if isinstance(current_image, bytes):
                img_obj = Image.open(io.BytesIO(current_image))
            elif isinstance(current_image, str) and os.path.exists(current_image):
                img_obj = Image.open(current_image)
            elif isinstance(current_image, Image.Image):
                img_obj = current_image
            elif isinstance(current_image, (np.ndarray, np.generic)):
                img_obj = Image.fromarray(current_image)
            
            if img_obj:
                # 關鍵修復：強制加載數據到內存，斷開文件關聯，解決 _idat/fileno 問題
                img_obj.load()
                
                # 2. 轉換模式
                if img_obj.mode != 'RGB':
                    img_obj = img_obj.convert('RGB')
                
                # 3. 調整大小 (Resize)
                max_size = (384, 384)
                if img_obj.size[0] > max_size[0] or img_obj.size[1] > max_size[1]:
                    img_obj.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # 4. 保存為 Bytes (JPEG)
                buffer = io.BytesIO()
                # optimize=True 會稍微慢一點點，但能顯著減少文件大小
                img_obj.save(buffer, format='JPEG', quality=85, optimize=True)
                image_bytes = buffer.getvalue()
                
        except Exception:
            # 處理失敗，保持為空 bytes（不打印，避免輸出過多）
            pass

    # 必須返回字典，且 key 要對應 features
    return {'image': image_bytes}


def tokenize_function(example):
    """
    對文本進行 tokenization。
    
    全局變量依賴：
    - bert_tokenizer: tokenizer（實際上是 blip_processor.tokenizer）
    - FLAG_HEAD_TAIL: 是否使用 head-tail 截斷策略
    """
    # 使用 bert_tokenizer (實際上是 blip_processor.tokenizer) 進行 tokenization
    encoded = bert_tokenizer(example["prompt"], truncation=False)
    
    # 獲取 input_ids 和 attention_mask (這些通常都有)
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    
    # 安全獲取 token_type_ids (如果沒有則設為 None)
    token_type_ids = encoded.get('token_type_ids', None)
    
    if len(input_ids) >= 512:
        if FLAG_HEAD_TAIL:
            # Head-Tail 截斷策略
            example['input_ids'] = input_ids[:128] + input_ids[-384:]
            example['attention_mask'] = attention_mask[:128] + attention_mask[-384:]
            if token_type_ids is not None:
                example['token_type_ids'] = token_type_ids[:128] + token_type_ids[-384:]
        else:
            # 僅保留尾部策略 (Tail only)
            example['input_ids'] = input_ids[-512:]
            example['attention_mask'] = attention_mask[-512:]
            if token_type_ids is not None:
                example['token_type_ids'] = token_type_ids[-512:]
    else:
        # 如果沒有超過長度，直接賦值
        example['input_ids'] = input_ids
        example['attention_mask'] = attention_mask
        if token_type_ids is not None:
            example['token_type_ids'] = token_type_ids
    
    return example


def replace_model_name_by_idx(example):
    """
    將模型名稱替換為索引，並進行 one-hot 編碼（如果是回歸任務）。
    
    全局變量依賴：
    - model_name_to_idx: 模型名稱到索引的映射字典
    - num_models: 模型總數
    - task_type: 任務類型 (0=回歸, 1=二分類, 2=多分類)
    """
    example['model'] = model_name_to_idx[example['model']]
    if task_type == 0:
        # update the model idx with one hot encoding
        # only for task_type == 0 because in other task types, the model idx will be one-hot coded
        # in the recalc_labels_and_one_hot_model_name function
        arr = [0 for _ in range(num_models)]
        arr[example['model']] = 1
        example['model'] = arr
    return example


def recalc_labels_and_one_hot_model_name(example):
    """
    根據分位數重新計算標籤，並進行模型名稱的 one-hot 編碼。
    
    全局變量依賴：
    - percentiles: 每個模型的分位數列表
    - num_models: 模型總數
    """
    for i, thresh in enumerate(percentiles[example['model']]):
        if example['num_tokens'] < thresh:
            example['labels'] = i
            break
    arr = [0 for _ in range(num_models)]
    arr[example['model']] = 1
    example['model'] = arr
    return example


def calc_percentile(dataset):
    """
    計算數據集的分位數，並根據分位數重新計算標籤。
    
    全局變量依賴：
    - FLAG_VICUNA_DATA_ONLY: 是否只處理單一模型數據
    - num_models: 模型總數
    - percentiles: 每個模型的分位數列表（會被修改）
    """
    if FLAG_VICUNA_DATA_ONLY:
        output_token_lengths = []
        for sample in dataset:
            output_token_lengths.append(sample['num_tokens'])
        s = pd.Series(output_token_lengths)
        print(s.describe(percentiles=[.25, .5, .75, .99]))
        # s = s[s < 2048]
        # sns.histplot(s,
        #          kde=False, 
        #          bins=100, color = 'blue')
        # plt.xlabel('Output Token Length')
        # plt.ylabel('User Requests')
        # plt.savefig('dist.png')
    else:
        output_token_lengths = [[] for _ in range(num_models)]
        for sample in dataset:
            output_token_lengths[sample['model']].append(sample['num_tokens'])
        for model_id in range(num_models):
            s = pd.Series(output_token_lengths[model_id])
            desc = s.describe(percentiles=[.25, .5, .75, .99])
            percentiles[model_id].extend([desc['25%'], desc['50%'], desc['75%'], desc['99%'], 1000000])
        # print(percentiles)
        dataset = dataset.map(recalc_labels_and_one_hot_model_name)
    return dataset


def preprocess_dataset(dataset, original_dataset=None, use_cache=True):
    """
    預處理數據集：提取對話、處理圖像、tokenization 等。
    
    全局變量依賴：
    - FLAG_VICUNA_DATA_ONLY: 是否只處理單一模型數據
    - FLAG_FIRST_ROUND_ONLY: 是否只處理第一輪對話
    - task_type: 任務類型 (0=回歸, 1=二分類, 2=多分類)
    - vicuna_tokenizer: tokenizer（用於 extract_first_round_prompt 和 exact_multi_round_prompt）
    - bert_tokenizer: tokenizer（用於 tokenize_function）
    - model_name_to_idx: 模型名稱到索引的映射（用於 replace_model_name_by_idx）
    - num_models: 模型總數（用於 replace_model_name_by_idx 和 calc_percentile）
    - percentiles: 每個模型的分位數列表（用於 calc_percentile）
    """
    # 移除 Honey-Data-1M 中不需要的欄位（保留 id, conversations, images）
    columns_to_remove = ['source', 'img_phash', 'img_size']
    existing_columns = dataset.column_names
    columns_to_remove = [col for col in columns_to_remove if col in existing_columns]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)
    
    new_sentence_column = [''] * len(dataset)
    dataset = dataset.add_column('prompt', new_sentence_column)
    new_label_column = [0] * len(dataset)
    dataset = dataset.add_column('labels', new_label_column)
    if task_type != 0:
        new_length_column = [0] * len(dataset)
        dataset = dataset.add_column('num_tokens', new_length_column)

    # Extract the user prompt(s) and the corresponding response length
    if FLAG_FIRST_ROUND_ONLY:
        # 先提取 prompt 和 labels，同時移除 conversations 欄位
        # 注意：這裡我們暫時保留 images 欄位，但會在 filter 之前移除它以避免觸發圖像解碼
        dataset = dataset.map(extract_first_round_prompt, remove_columns=['conversations'])
        
        # 在 filter 之前，保存包含 images 的數據集副本（用於後續圖像處理）
        # 然後移除 images 欄位以避免 filter 時觸發圖像解碼（這會導致性能問題和可能的錯誤）
        has_images = 'images' in dataset.column_names
        if has_images:
            # 保存包含 images 的數據集副本（如果沒有提供原始數據集）
            if original_dataset is None:
                # 只保存必要的欄位以節省內存（id 和 images）
                if 'id' in dataset.column_names:
                    original_dataset = dataset.select_columns(['id', 'images'])
                else:
                    # 如果沒有 id，創建一個臨時索引映射
                    original_dataset = dataset.select_columns(['images'])
                    # 添加臨時索引欄位
                    original_dataset = original_dataset.add_column('temp_idx', list(range(len(original_dataset))))
            # 移除 images 欄位以避免 filter 時觸發圖像解碼
            dataset = dataset.remove_columns(['images'])
            print('Temporarily removed images column to avoid decoding during filter')
        
        print('Num samples before filtering: ', len(dataset))
        if task_type == 0:
            dataset = dataset.filter(lambda example: example["labels"] > 1 and example["labels"] < 512)
        else:
            dataset = dataset.filter(lambda example: example["num_tokens"] > 1 and example["num_tokens"] < 512)
        print('Num samples after filtering: ', len(dataset))
    else:
        # 多輪對話處理（需要修改 exact_multi_round_prompt 來支援 Honey-Data-1M）
        dataset = exact_multi_round_prompt(dataset)
    
    # --- 處理圖像數據（修復 Schema 不匹配問題）---
    print('Processing images with explicit schema...')
    
    # 1. 獲取當前所有的欄位名稱
    current_columns = dataset.column_names
    
    # 2. 定義我們要移除的欄位（舊的圖像數據）
    # 確保 'images' 在移除列表中（如果存在）
    columns_to_remove = ['images'] if 'images' in current_columns else []
    
    # 3. 構建新的 Features Schema
    # 關鍵：不能直接 copy() 舊的 features，因為舊的 features 裡可能包含 'images'
    # 我們需要建立一個不包含 'images' 但包含新 'image' (binary) 的 Schema
    new_features = dataset.features.copy()
    
    # 如果舊 Schema 裡有 'images'，先刪除它（這是解決 Schema 不匹配的關鍵）
    if 'images' in new_features:
        del new_features['images']
    
    # 添加我們新的二進制圖像欄位
    new_features['image'] = Value("binary")
    
    # 為了安全起見，明確告訴 datasets 我們保留哪些欄位
    # 這能防止 "Schema and number of arrays unequal" 錯誤
    # 因為 map 返回的字典 key 必須與 new_features 的 key 一一對應
    print(f"Columns to remove: {columns_to_remove}")
    print(f"Target features: {list(new_features.keys())}")
    
    # 如果沒有 images 欄位，但有 original_dataset，嘗試從原始數據集獲取圖像
    if 'images' not in dataset.column_names:
        if original_dataset is not None and 'images' in original_dataset.column_names:
            # 從原始數據集獲取圖像
            print('Attempting to retrieve images from original dataset...')
            
            # 建立索引映射：filter 後的數據集索引 -> 原始數據集索引
            # 由於 filter 會改變索引，我們需要通過 id 或其他方式匹配
            if 'id' in dataset.column_names and 'id' in original_dataset.column_names:
                # 通過 id 匹配
                id_to_original_idx = {sample_id: idx for idx, sample_id in enumerate(original_dataset['id'])}
                
                def get_image_from_original(example):
                    try:
                        if 'id' in example:
                            original_idx = id_to_original_idx.get(example['id'])
                            if original_idx is not None and original_idx < len(original_dataset):
                                # 使用 try-except 捕獲圖像解碼錯誤（損壞的圖像文件）
                                try:
                                    # 訪問原始數據集時可能會觸發圖像自動解碼
                                    # 如果圖像損壞（如 broken PNG），會拋出 SyntaxError 或其他異常
                                    sample = original_dataset[original_idx]
                                    images_data = sample.get('images', [])
                                    # 使用 process_images 的邏輯處理圖像
                                    # process_images 內部也有異常處理，但這裡也加上以防萬一
                                    return process_images({'images': images_data})
                                except (SyntaxError, OSError, IOError, ValueError, 
                                        TypeError, AttributeError, KeyError, IndexError) as e:
                                    # 捕獲圖像解碼錯誤（損壞的 PNG、JPEG 等）
                                    # SyntaxError: broken PNG file
                                    # OSError/IOError: 文件讀取錯誤
                                    # ValueError: 圖像格式錯誤
                                    # 返回空圖像，不中斷處理流程
                                    return {'image': b''}
                                except Exception as e:
                                    # 捕獲任何其他未預期的錯誤，返回空圖像
                                    return {'image': b''}
                    except Exception as e:
                        # 捕獲任何其他錯誤（如索引錯誤等），返回空圖像
                        return {'image': b''}
                    return {'image': b''}
            elif 'temp_idx' in original_dataset.column_names:
                # 使用臨時索引（如果原始數據集有 temp_idx）
                # 注意：這需要 filter 後的數據集也保留相同的索引順序
                # 但實際上 filter 會改變順序，所以這個方法可能不適用
                print('Warning: Cannot reliably map indices after filter, using empty images')
                dataset = dataset.map(lambda x: {'image': b''}, features=new_features)
            else:
                # 無法建立映射，使用空圖像
                print('Warning: Cannot map to original dataset, using empty images')
                dataset = dataset.map(lambda x: {'image': b''}, features=new_features)
                original_dataset = None  # 標記為無法使用
            
            if original_dataset is not None:
                num_procs = min(64, max(1, multiprocessing.cpu_count() - 2))
                print(f"Using {num_procs} processes for image processing from original dataset.")
                dataset = dataset.map(
                    get_image_from_original,
                    batched=False,
                    writer_batch_size=1000,
                    num_proc=num_procs,
                    features=new_features,
                    desc="Processing images from original dataset",
                    load_from_cache_file=use_cache  # 使用緩存可以大幅加速（默認 True）
                )
        else:
            print('Warning: No images field found in dataset and cannot retrieve from original dataset')
            dataset = dataset.map(lambda x: {'image': b''}, features=new_features)
    else:
        # 計算適合的進程數（考慮到圖像已限制為 384x384，可以使用更多進程）
        # 但為了避免內存問題，建議不要超過 64
        num_procs = min(64, max(1, multiprocessing.cpu_count() - 2))
        print(f"Using {num_procs} processes for image processing.")
        
        dataset = dataset.map(
            process_images,
            batched=False,          # process_images 是一次處理單張
            writer_batch_size=1000, # 提高寫入緩衝區大小
            num_proc=num_procs,     # 啟用多進程加速（限制為 64 以避免內存問題）
            remove_columns=columns_to_remove,  # 在 map 結束後移除舊欄位
            features=new_features,  # 強制使用新的 Schema（無 'images'，有 'image'）
            desc="Processing images",
            load_from_cache_file=use_cache  # 使用緩存可以大幅加速（默認 True）
        )
    
    # 統計圖片狀況（因為數據量大，使用簡單的計數方式）
    try:
        has_image_count = sum(1 for i in range(min(1000, len(dataset))) if len(dataset[i]['image']) > 0)
        if len(dataset) > 1000:
            print(f'Number of samples with valid images (sampled from first 1000): {has_image_count}/1000')
        else:
            print(f'Number of samples with valid images: {has_image_count}/{len(dataset)}')
    except:
        pass  # 統計失敗不影響處理
    
    # 過濾掉沒有有效圖像的樣本（在預處理階段就移除，避免訓練時出錯）
    print('Filtering out samples without valid images...')
    num_before_filter = len(dataset)
    
    def has_valid_image(example):
        """檢查樣本是否有有效的圖像"""
        if 'image' not in example:
            return False
        image_data = example['image']
        
        # 檢查是否為空 bytes
        if isinstance(image_data, bytes):
            if len(image_data) == 0:
                return False
            # 嘗試驗證圖像是否有效
            try:
                import io
                img = Image.open(io.BytesIO(image_data))
                img.load()  # 強制加載，驗證圖像是否損壞
                return True
            except Exception:
                return False
        elif image_data is None:
            return False
        else:
            # 其他格式（PIL Image, numpy array 等）
            return True
    
    dataset = dataset.filter(has_valid_image)
    num_after_filter = len(dataset)
    num_removed = num_before_filter - num_after_filter
    print(f'Filtered out {num_removed} samples without valid images ({num_before_filter} -> {num_after_filter})')
    
    if num_after_filter == 0:
        raise ValueError("No samples with valid images found after filtering! Please check your dataset.")
    
    if FLAG_VICUNA_DATA_ONLY:
        # Honey-Data-1M 可能沒有 model 欄位，需要處理
        if 'model' in dataset.column_names:
            dataset = dataset.remove_columns(['model'])
    else:
        # 如果資料集有 model 欄位，進行處理
        if 'model' in dataset.column_names:
            dataset = dataset.map(replace_model_name_by_idx)
    
    # Tokenize the user prompt
    dataset = dataset.map(tokenize_function, batched=False, remove_columns=['prompt'])
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_models', action='store_true', default=False)
    parser.add_argument('--multi_round', action='store_true', default=False)
    parser.add_argument('--head_tail', action='store_true', default=False)
    parser.add_argument('--task_type', type=int, help='0 for regression, 1 for binary cls, 2 for multi-cls', default=2)
    parser.add_argument('--data_size', type=int, help='Size of the dataset to use (in thousands). Use 0 to process entire dataset', default=1000)
    parser.add_argument('--model_name', type=str, help='Name of the LLM to predict for (not used for Honey-Data-1M)', default='vicuna-13b')
    parser.add_argument('--local_dataset_path', type=str, help='Path to local dataset folder (for offline use)', default='/data2/models/Honey-Data-1M')
    parser.add_argument('--use_log_loss', action='store_true', help='Use log-space labels for regression', default=False)
    parser.add_argument('--blip_local_dir', type=str, help='Local path to BLIP model for offline use', default='/data2/w00917303/decode-router/blip-image-captioning-base/')
    parser.add_argument('--output_dir', type=str, help='Directory to save processed dataset', default='/tmp')
    parser.add_argument('--batch_size', type=int, help='Process dataset in batches to save memory (0 = process all at once)', default=0)
    parser.add_argument('--force_reprocess', action='store_true', help='Force reprocess all data, ignore cache (default: use cache to speed up)', default=False)
    args = parser.parse_args()

    # 0: regression; 1: binary classification; 2: multi-class classification;
    task_type = args.task_type
    FLAG_VICUNA_DATA_ONLY = not args.all_models
    FLAG_FIRST_ROUND_ONLY = not args.multi_round
    FLAG_HEAD_TAIL = args.head_tail
    FLAG_USE_LOG_LOSS = args.use_log_loss
    # cls_threshold = 328
    # 使用分位數作為閾值（20%, 40%, 60%, 80%）以平衡類別分布
    # 這些值需要根據實際資料集調整，可以執行 check_dataset_distribution.py 來獲取
    if task_type == 1:
        multi_cls_thresholds = [141, 503, 1000000]  # Binary classification (3 classes)
    else:
        # 5-class balanced thresholds based on percentiles (adjust if needed)
        # 估計值：20%≈80, 40%≈130, 60%≈170, 80%≈230
        # 實際建議：先執行 check_dataset_distribution.py 查看你的資料分布
        multi_cls_thresholds = [80, 130, 180, 250, 1000000] if FLAG_FIRST_ROUND_ONLY else [58, 147, 280, 499, 100000]
    
    dataset_name = 'Open-Bee/Honey-Data-1M'
    
    # Initialize BLIP processor and use其 tokenizer 作為文字 tokenizer（離線、本地路徑）
    print(f'Loading BLIP processor from local dir: {args.blip_local_dir} ...')
    blip_processor = BlipProcessor.from_pretrained(args.blip_local_dir, local_files_only=True)
    # BLIP 內部使用的 tokenizer（通常是 BertTokenizerFast）
    vicuna_tokenizer = blip_processor.tokenizer
    bert_tokenizer = blip_processor.tokenizer
    
    # 如果 data_size 為 0，表示處理整個數據集（不限制大小）
    selected_data_size = 0 if args.data_size == 0 else 1000 * args.data_size

    model_names = ['vicuna-13b', 'wizardlm-13b', 'palm-2', 'llama-2-13b-chat', 'koala-13b',
                   'claude-instant-1', 'oasst-pythia-12b', 'alpaca-13b', 'mpt-7b-chat',
                   'vicuna-7b', 'dolly-v2-12b', 'mpt-30b-chat', 'fastchat-t5-3b', 'chatglm-6b',
                   'claude-1', 'gpt-4', 'vicuna-33b', 'guanaco-33b', 'RWKV-4-Raven-14B',
                   'stablelm-tuned-alpha-7b', 'llama-13b', 'gpt-3.5-turbo', 'llama-2-7b-chat',
                   'claude-2', 'gpt4all-13b-snoozy']
    model_name_to_idx = {model_names[i]: i for i in range(len(model_names))}
    num_models = len(model_names)

    # 構建資料集路徑（Honey-Data-1M 專用）
    dataset_path = 'honey_' if FLAG_VICUNA_DATA_ONLY else 'honey_all_'
    dataset_path = dataset_path if task_type == 0 else dataset_path + 'cls_' if task_type == 1 else dataset_path + 'multi_cls_'
    if FLAG_FIRST_ROUND_ONLY:
        dataset_path = 'first_round_data_' + dataset_path
    elif FLAG_HEAD_TAIL:
        dataset_path = 'headtail_' + dataset_path
    else:
        dataset_path = 'tail_' + dataset_path
    
    # 如果使用 log loss，在路徑名稱中標記
    if FLAG_USE_LOG_LOSS and task_type == 0:
        dataset_path = dataset_path + 'log_'
    
    # 構建完整保存路徑
    if selected_data_size == 0:
        dataset_path = dataset_path + 'full'
    else:
        dataset_path = dataset_path + f'{int(selected_data_size / 1000)}K'
    full_dataset_path = os.path.join(args.output_dir, dataset_path)

    # Load dataset from local path if provided, otherwise download from HuggingFace
    if args.local_dataset_path:
        print(f'Loading dataset from local path: {args.local_dataset_path}')
        if os.path.isfile(args.local_dataset_path):
            # 單一檔案（.arrow 或 .parquet）
            if args.local_dataset_path.endswith('.parquet'):
                dataset = load_dataset('parquet', data_files=args.local_dataset_path, split='train')
            else:
                dataset = Dataset.from_file(args.local_dataset_path)
        elif os.path.isdir(args.local_dataset_path):
            # 檢查是否為 datasets 保存的格式（有 dataset_info.json）
            dataset_info_path = os.path.join(args.local_dataset_path, 'dataset_info.json')
            if os.path.exists(dataset_info_path):
                # 使用 load_from_disk 載入 datasets 保存的格式
                dataset = datasets.load_from_disk(args.local_dataset_path)
            else:
                # 嘗試找 .parquet 檔案（優先）
                parquet_files = sorted(glob.glob(os.path.join(args.local_dataset_path, '*.parquet')))
                if parquet_files:
                    print(f'Found {len(parquet_files)} parquet files, loading...')
                    dataset = load_dataset('parquet', data_files=parquet_files, split='train')
                else:
                    # 嘗試找 .arrow 檔案
                    arrow_files = sorted(glob.glob(os.path.join(args.local_dataset_path, '*.arrow')))
                    if arrow_files:
                        print(f'Found {len(arrow_files)} arrow files, loading...')
                        dataset = load_dataset('arrow', data_files=arrow_files, split='train')
                    else:
                        raise ValueError(f"No valid dataset files found in {args.local_dataset_path}. "
                                       f"Expected either a .parquet/.arrow file, a directory with dataset_info.json, "
                                       f"or a directory with .parquet/.arrow files.")
        else:
            raise ValueError(f"Path {args.local_dataset_path} does not exist.")
        
        print(f'Loaded {len(dataset)} samples from local dataset')
        if selected_data_size > 0 and len(dataset) > selected_data_size:
            dataset = dataset.select(range(selected_data_size))
            print(f'Selected first {selected_data_size} samples')
        elif selected_data_size == 0:
            print(f'Processing entire dataset ({len(dataset)} samples)')
    else:
        print(f'Downloading dataset from HuggingFace: {dataset_name}')
        dataset = load_dataset(dataset_name, split='train')
        if selected_data_size > 0 and len(dataset) > selected_data_size:
            dataset = dataset.select(range(selected_data_size))
        elif selected_data_size == 0:
            print(f'Processing entire dataset ({len(dataset)} samples)')
    
    # Honey-Data-1M 沒有 model 欄位，所以不需要過濾
    dataset = dataset.shuffle(seed=1)
    
    # 分批處理或一次性處理
    if args.batch_size > 0 and len(dataset) > args.batch_size:
        print(f'Processing dataset in batches of {args.batch_size} samples...')
        total_samples = len(dataset)
        processed_datasets = []
        
        for start_idx in range(0, total_samples, args.batch_size):
            end_idx = min(start_idx + args.batch_size, total_samples)
            print(f'Processing batch {start_idx}-{end_idx} / {total_samples}...')
            
            # 選取當前批次
            batch_dataset = dataset.select(range(start_idx, end_idx))
            
            # 處理當前批次
            # use_cache: 如果 force_reprocess=True，則不使用緩存（重新計算）
            batch_dataset = preprocess_dataset(batch_dataset, use_cache=not args.force_reprocess)
            
            # 保存臨時批次
            temp_path = os.path.join(args.output_dir, f'{dataset_path}_temp_batch_{start_idx}')
            os.makedirs(temp_path, exist_ok=True)
            batch_dataset.save_to_disk(temp_path)
            processed_datasets.append(temp_path)
            print(f'  Saved batch to {temp_path}')
        
        # 合併所有批次
        if not processed_datasets:
            print("No batches were successfully processed!")
            import sys
            sys.exit(1)
            
        print(f'Merging {len(processed_datasets)} batches...')
        from datasets import load_from_disk, concatenate_datasets
        datasets_list = []
        for path in processed_datasets:
            try:
                loaded_dataset = load_from_disk(path)
                datasets_list.append(loaded_dataset)
                print(f'  Loaded batch from {path}')
            except Exception as e:
                print(f"Error loading batch from {path}: {e}")
        
        if datasets_list:
            dataset = concatenate_datasets(datasets_list)
            print(f'Successfully merged {len(datasets_list)} batches, total samples: {len(dataset)}')
        else:
            print("No batches could be loaded!")
            import sys
            sys.exit(1)
        
        # 清理臨時批次檔案
        print('Cleaning up temporary batch files...')
        for temp_path in processed_datasets:
            import shutil
            try:
                shutil.rmtree(temp_path)
            except:
                pass
        
        # 重新計算分位數（如果需要）
        percentiles = [[] for _ in range(num_models)]
        if task_type != 0:
            dataset = calc_percentile(dataset)
    else:
        # 一次性處理所有資料
        print('Processing entire dataset at once...')
        try:
            # use_cache: 如果 force_reprocess=True，則不使用緩存（重新計算）
            dataset = preprocess_dataset(dataset, use_cache=not args.force_reprocess)
            
            percentiles = [[] for _ in range(num_models)]
            if task_type != 0:
                dataset = calc_percentile(dataset)
        except Exception as e:
            print(f"Error processing dataset: {e}")
            print("Consider using --batch_size to process in smaller batches.")
            import sys
            sys.exit(1)
    
    # 設置格式為 torch，但保留 pixel_values 為 tensor
    dataset.set_format("torch")

    # 保存最終資料集
    os.makedirs(args.output_dir, exist_ok=True)
    dataset.save_to_disk(full_dataset_path)
    print(f'Saved dataset to {full_dataset_path}')]