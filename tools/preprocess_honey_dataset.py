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

import datasets
from datasets import load_dataset, Dataset, Features, Value
from transformers import BlipProcessor
from PIL import Image
import argparse
import glob
import io
import numpy as np
import multiprocessing
import math


def extract_first_round_prompt(example):
    """
    提取第一輪對話的用戶提示和助手回應。
    使用 log-space labels (log(tokens + 1)) 用於回歸任務。
    
    全局變量依賴：
    - blip_tokenizer: tokenizer（blip_processor.tokenizer）用於計算 token 數量
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
    encoded_response = blip_tokenizer(assistant_content, truncation=False)
    num_tokens = len(encoded_response['input_ids'])
    
    # 回歸任務：使用 log-space labels (log(tokens + 1))
    example['labels'] = math.log(num_tokens + 1.0)
    
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
    如果用戶輸入超過 512 個 token，標記為無效樣本（將被過濾）。
    
    全局變量依賴：
    - blip_tokenizer: tokenizer（blip_processor.tokenizer）
    """
    # 使用 blip_tokenizer (blip_processor.tokenizer) 進行 tokenization
    encoded = blip_tokenizer(example["prompt"], truncation=False)
    
    # 獲取 input_ids 和 attention_mask (這些通常都有)
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    
    # 安全獲取 token_type_ids (如果沒有則設為 None)
    token_type_ids = encoded.get('token_type_ids', None)
    
    # 如果長度超過 512，標記為無效樣本
    if len(input_ids) > 512:
        example['is_valid'] = False
        # 仍然賦值（雖然會被過濾），避免錯誤
        example['input_ids'] = input_ids[:512]
        example['attention_mask'] = attention_mask[:512]
        if token_type_ids is not None:
            example['token_type_ids'] = token_type_ids[:512]
    else:
        # 如果沒有超過長度，直接賦值
        example['is_valid'] = True
        example['input_ids'] = input_ids
        example['attention_mask'] = attention_mask
        if token_type_ids is not None:
            example['token_type_ids'] = token_type_ids
    
    return example


def preprocess_dataset(dataset, original_dataset=None, use_cache=True):
    """
    預處理數據集：提取對話、處理圖像、tokenization 等。
    
    全局變量依賴：
    - blip_tokenizer: tokenizer（blip_processor.tokenizer，用於 extract_first_round_prompt 和 tokenize_function）
    """
    # 移除 Honey-Data-1M 中不需要的欄位（保留 id, conversations, images）
    columns_to_remove = ['source', 'img_phash', 'img_size']
    existing_columns = dataset.column_names
    columns_to_remove = [col for col in columns_to_remove if col in existing_columns]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)
    
    # 添加 prompt 和 labels 欄位
    new_sentence_column = [''] * len(dataset)
    dataset = dataset.add_column('prompt', new_sentence_column)
    new_label_column = [0] * len(dataset)
    dataset = dataset.add_column('labels', new_label_column)

    # 提取第一輪對話的 prompt 和 labels
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
    
    # 過濾：只保留 labels 在合理範圍內的樣本（log space: log(2) 到 log(512)）
    print('Num samples before filtering: ', len(dataset))
    dataset = dataset.filter(lambda example: example["labels"] > math.log(2) and example["labels"] < math.log(512))
    print('Num samples after filtering: ', len(dataset))
    
    # --- 處理圖像數據（修復 Schema 不匹配問題）---
    print('Processing images with explicit schema...')
    
    # 1. 獲取當前所有的欄位名稱
    current_columns = dataset.column_names
    
    # 2. 定義我們要移除的欄位（舊的圖像數據）
    columns_to_remove = ['images'] if 'images' in current_columns else []
    
    # 3. 構建新的 Features Schema
    new_features = dataset.features.copy()
    
    # 如果舊 Schema 裡有 'images'，先刪除它（這是解決 Schema 不匹配的關鍵）
    if 'images' in new_features:
        del new_features['images']
    
    # 添加我們新的二進制圖像欄位
    new_features['image'] = Value("binary")
    
    print(f"Columns to remove: {columns_to_remove}")
    print(f"Target features: {list(new_features.keys())}")
    
    # 如果沒有 images 欄位，但有 original_dataset，嘗試從原始數據集獲取圖像
    if 'images' not in dataset.column_names:
        if original_dataset is not None and 'images' in original_dataset.column_names:
            # 從原始數據集獲取圖像
            print('Attempting to retrieve images from original dataset...')
            
            # 建立索引映射：filter 後的數據集索引 -> 原始數據集索引
            if 'id' in dataset.column_names and 'id' in original_dataset.column_names:
                # 通過 id 匹配
                id_to_original_idx = {sample_id: idx for idx, sample_id in enumerate(original_dataset['id'])}
                
                def get_image_from_original(example):
                    try:
                        if 'id' in example:
                            original_idx = id_to_original_idx.get(example['id'])
                            if original_idx is not None and original_idx < len(original_dataset):
                                try:
                                    sample = original_dataset[original_idx]
                                    images_data = sample.get('images', [])
                                    return process_images({'images': images_data})
                                except (SyntaxError, OSError, IOError, ValueError, 
                                        TypeError, AttributeError, KeyError, IndexError) as e:
                                    return {'image': b''}
                                except Exception as e:
                                    return {'image': b''}
                    except Exception as e:
                        return {'image': b''}
                    return {'image': b''}
            elif 'temp_idx' in original_dataset.column_names:
                print('Warning: Cannot reliably map indices after filter, using empty images')
                dataset = dataset.map(lambda x: {'image': b''}, features=new_features)
            else:
                print('Warning: Cannot map to original dataset, using empty images')
                dataset = dataset.map(lambda x: {'image': b''}, features=new_features)
                original_dataset = None
            
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
                    load_from_cache_file=use_cache
                )
        else:
            print('Warning: No images field found in dataset and cannot retrieve from original dataset')
            dataset = dataset.map(lambda x: {'image': b''}, features=new_features)
    else:
        # 計算適合的進程數（考慮到圖像已限制為 384x384，可以使用更多進程）
        num_procs = min(64, max(1, multiprocessing.cpu_count() - 2))
        print(f"Using {num_procs} processes for image processing.")
        
        dataset = dataset.map(
            process_images,
            batched=False,
            writer_batch_size=1000,
            num_proc=num_procs,
            remove_columns=columns_to_remove,
            features=new_features,
            desc="Processing images",
            load_from_cache_file=use_cache
        )
    
    # 統計圖片狀況
    try:
        has_image_count = sum(1 for i in range(min(1000, len(dataset))) if len(dataset[i]['image']) > 0)
        if len(dataset) > 1000:
            print(f'Number of samples with valid images (sampled from first 1000): {has_image_count}/1000')
        else:
            print(f'Number of samples with valid images: {has_image_count}/{len(dataset)}')
    except:
        pass
    
    # 過濾掉沒有有效圖像的樣本
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
                img = Image.open(io.BytesIO(image_data))
                img.load()  # 強制加載，驗證圖像是否損壞
                return True
            except Exception:
                return False
        elif image_data is None:
            return False
        else:
            return True
    
    dataset = dataset.filter(has_valid_image)
    num_after_filter = len(dataset)
    num_removed = num_before_filter - num_after_filter
    print(f'Filtered out {num_removed} samples without valid images ({num_before_filter} -> {num_after_filter})')
    
    if num_after_filter == 0:
        raise ValueError("No samples with valid images found after filtering! Please check your dataset.")
    
    # 移除 model 欄位（如果存在）
    if 'model' in dataset.column_names:
        dataset = dataset.remove_columns(['model'])
    
    # Tokenize the user prompt
    dataset = dataset.map(tokenize_function, batched=False, remove_columns=['prompt'])
    
    # 過濾掉用戶輸入超過 512 個 token 的樣本
    print('Num samples before filtering long prompts: ', len(dataset))
    dataset = dataset.filter(lambda example: example.get('is_valid', True))
    print('Num samples after filtering long prompts: ', len(dataset))
    
    # 移除 is_valid 標記欄位
    if 'is_valid' in dataset.column_names:
        dataset = dataset.remove_columns(['is_valid'])
    
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Honey-Data-1M for regression with log-space labels (first round only)')
    parser.add_argument('--data_size', type=int, help='Size of the dataset to use (in thousands). Use 0 to process entire dataset', default=1000)
    parser.add_argument('--local_dataset_path', type=str, help='Path to local dataset folder (for offline use)', default='/data2/models/Honey-Data-1M')
    parser.add_argument('--blip_local_dir', type=str, help='Local path to BLIP model for offline use', default='/data2/w00917303/decode-router/blip-image-captioning-base/')
    parser.add_argument('--output_dir', type=str, help='Directory to save processed dataset', default='/tmp')
    parser.add_argument('--batch_size', type=int, help='Process dataset in batches to save memory (0 = process all at once)', default=0)
    parser.add_argument('--force_reprocess', action='store_true', help='Force reprocess all data, ignore cache (default: use cache to speed up)', default=False)
    args = parser.parse_args()

    dataset_name = 'Open-Bee/Honey-Data-1M'
    
    # Initialize BLIP processor and use its tokenizer as text tokenizer (offline, local path)
    print(f'Loading BLIP processor from local dir: {args.blip_local_dir} ...')
    blip_processor = BlipProcessor.from_pretrained(args.blip_local_dir, local_files_only=True)
    # BLIP 內部使用的 tokenizer（通常是 BertTokenizerFast）
    blip_tokenizer = blip_processor.tokenizer
    
    # 如果 data_size 為 0，表示處理整個數據集（不限制大小）
    selected_data_size = 0 if args.data_size == 0 else 1000 * args.data_size
    
    # 構建資料集路徑（固定為 first_round_data_honey_log_）
    dataset_path = 'first_round_data_honey_log_'
    
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
    
    # 打亂數據集
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
        import shutil
        for temp_path in processed_datasets:
            try:
                shutil.rmtree(temp_path)
            except:
                pass
    else:
        # 一次性處理所有資料
        print('Processing entire dataset at once...')
        try:
            dataset = preprocess_dataset(dataset, use_cache=not args.force_reprocess)
        except Exception as e:
            print(f"Error processing dataset: {e}")
            print("Consider using --batch_size to process in smaller batches.")
            import sys
            sys.exit(1)
    
    # 設置格式為 torch
    dataset.set_format("torch")

    # 保存最終資料集
    os.makedirs(args.output_dir, exist_ok=True)
    dataset.save_to_disk(full_dataset_path)
    print(f'Saved dataset to {full_dataset_path}')
