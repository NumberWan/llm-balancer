import datasets
from datasets import load_dataset
import argparse
import transformers
from transformers import AutoConfig, DataCollatorWithPadding, BlipModel, BlipProcessor
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os
import numpy as np
from datetime import datetime
import time
import io


class BlipRegressionModel(nn.Module):
    """
    BLIP 回歸模型：用於預測輸出 token 長度（log space）
    """
    def __init__(self, model_name, hidden_dim=128):
        super().__init__()
        # Load BLIP model from local path (offline use, no download)
        self.blip = BlipModel.from_pretrained(model_name, local_files_only=True)
        
        # 訓練策略：前 3 個 epoch 訓練 BLIP encoder，之後只訓練 MLP head
        # 這裡先不凍結，在訓練循環中動態處理
        self.feature_dim = self.blip.config.text_config.hidden_size
        
        # MLP head for regression
        self.cls = nn.Linear(self.feature_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def _extract_fused_features(self, input_ids=None, attention_mask=None, pixel_values=None):
        """
        從 BLIP 提取融合特徵的最終修正版。
        針對 Debug Info: keys=['text_model_output', 'vision_model_output', ...]
        """
        outputs = self.blip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=False,
        )

        # === 針對您的 Debug Info 進行修正 ===
        # 1. 檢查是否有 text_model_output (這是 Debug Info 顯示存在的)
        if hasattr(outputs, "text_model_output") and outputs.text_model_output is not None:
            # 融合後的特徵通常在 text_model 的 last_hidden_state 中
            # (前提是 input_ids 和 pixel_values 都傳入了，BLIP 會自動做 Cross-Attention)
            if hasattr(outputs.text_model_output, "last_hidden_state"):
                return outputs.text_model_output.last_hidden_state[:, 0, :]
        
        # 2. 嘗試直接獲取 last_hidden_state (標準結構)
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state[:, 0, :]

        # 3. 如果是 text_outputs 結構 (舊版)
        if getattr(outputs, "text_outputs", None) is not None:
            if hasattr(outputs.text_outputs, "last_hidden_state"):
                return outputs.text_outputs.last_hidden_state[:, 0, :]

        # 4. 如果都失敗，回退到分別提取並相加 (防止崩潰的最後手段)
        # 注意：這不是真正的融合，但能讓程式跑下去
        print("Warning: extract_fused_features fallback to simple addition.")
        
        if input_ids is not None:
            batch_size = input_ids.size(0)
            device = input_ids.device
        elif pixel_values is not None:
            batch_size = pixel_values.size(0)
            device = pixel_values.device
        else:
            raise ValueError("BLIP forward pass requires at least text or image inputs.")
        
        pool_text = torch.zeros(batch_size, self.feature_dim, device=device)
        pool_image = torch.zeros(batch_size, self.feature_dim, device=device)

        if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
            pool_text = outputs.text_embeds  # 已經是 pooled
        elif hasattr(outputs, "text_model_output") and outputs.text_model_output is not None:
            if hasattr(outputs.text_model_output, "last_hidden_state"):
                pool_text = outputs.text_model_output.last_hidden_state[:, 0, :]

        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            pool_image = outputs.image_embeds
        elif hasattr(outputs, "vision_model_output") and outputs.vision_model_output is not None:
            if hasattr(outputs.vision_model_output, "last_hidden_state"):
                pool_image = outputs.vision_model_output.last_hidden_state[:, 0, :]
        
        # 簡單融合 (雖然效果不如 Cross-Attn，但比報錯好)
        return pool_text + pool_image

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None):
        """前向傳播：預測 log(token_length)"""
        fused_features = self._extract_fused_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
        output = self.relu(self.cls(fused_features))
        output = self.relu(self.fc1(output))
        output = self.fc2(output).squeeze(-1)
        return output


def blip_collate_fn(batch, tokenizer, blip_processor):
    """自定義 collate function，處理文字和圖像"""
    text_data = []
    images = []
    other_data = {}
    
    for item in batch:
        # 提取文字相關的數據
        text_item = {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
        }
        if 'token_type_ids' in item:
            text_item['token_type_ids'] = item['token_type_ids']
        text_data.append(text_item)
        
        # 提取圖像
        if 'image' in item and item['image'] is not None:
            images.append(item['image'])
        else:
            images.append(None)
        
        # 提取其他數據（labels, num_tokens）
        for key in ['labels', 'num_tokens']:
            if key in item:
                if key not in other_data:
                    other_data[key] = []
                other_data[key].append(item[key])
    
    # 使用 DataCollatorWithPadding 處理文字
    text_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    batch_result = text_collator(text_data)
    
    # 處理圖像：使用 BLIP processor 動態轉換
    # 創建一個 dummy 黑色圖像作為回退（BLIP 標準輸入尺寸）
    dummy_image = Image.new('RGB', (384, 384), color=(0, 0, 0))
    dummy_processed = blip_processor(images=dummy_image, return_tensors="pt")
    dummy_pixel_values = dummy_processed['pixel_values'].squeeze(0)
    
    pixel_values_list = []
    for img in images:
        if img is not None:
            try:
                if isinstance(img, bytes):
                    if len(img) == 0:
                        # 使用 dummy image
                        pixel_values_list.append(dummy_pixel_values.clone())
                        continue
                    img_buffer = io.BytesIO(img)
                    img = Image.open(img_buffer)
                    img.load()
                elif isinstance(img, str):
                    img = Image.open(img)
                    img.load()
                elif isinstance(img, Image.Image):
                    img.load()
                else:
                    # 強制轉換為 numpy array 再轉回 PIL，確保格式正確
                    if isinstance(img, np.ndarray):
                        # 確保是 uint8 格式，避免 processor 崩潰
                        if img.dtype != np.uint8:
                            img = img.astype(np.uint8)
                        img = Image.fromarray(img)
                    else:
                        img = Image.fromarray(np.array(img, dtype=np.uint8))
                
                # 確保是 PIL Image 格式
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(np.uint8(img))
                
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                processed = blip_processor(images=img, return_tensors="pt")
                pixel_values_list.append(processed['pixel_values'].squeeze(0))
            except Exception as e:
                # 圖像解碼失敗時使用 dummy image，而不是 None
                pixel_values_list.append(dummy_pixel_values.clone())
        else:
            # 沒有圖像時使用 dummy image
            pixel_values_list.append(dummy_pixel_values.clone())
    
    # 處理圖像：確保 batch size 一致（現在所有圖像都有效，可以直接 stack）
    batch_result['pixel_values'] = torch.stack(pixel_values_list)
    
    # 合併其他數據
    for key, values in other_data.items():
        if len(values) > 0:
            if isinstance(values[0], torch.Tensor):
                batch_result[key] = torch.stack(values) if values[0].dim() > 0 else torch.tensor(values)
            elif isinstance(values[0], (int, float)):
                batch_result[key] = torch.tensor(values)
            else:
                batch_result[key] = values
    
    return batch_result


def generate_dataloaders(dataset, train_batch_size, test_batch_size, tokenizer, blip_processor=None):
    """生成訓練、驗證和測試數據加載器"""
    n_total_samples = len(dataset)
    # 使用簡單的 60/20/20 分割
    train_validationtest = dataset.train_test_split(test_size=0.4, shuffle=False)
    validation_test = train_validationtest['test'].train_test_split(test_size=0.5, shuffle=False)
    train_dataset = train_validationtest['train']
    validation_dataset = validation_test['train']
    test_dataset = validation_test['test']
    
    print(f'Total training samples: {len(train_dataset)}')
    print(f'Total validation samples: {len(validation_dataset)}')
    print(f'Total test samples: {len(test_dataset)}')

    # 自動計算 worker 數量（關鍵性能優化）
    num_workers = min(8, max(1, (os.cpu_count() or 1) // 2))
    print(f'Using {num_workers} workers for DataLoader (critical for performance)')

    # 如果有 BLIP processor，使用自定義 collate function 處理圖像
    if blip_processor is not None:
        def collate_fn(batch):
            return blip_collate_fn(batch, tokenizer, blip_processor)
        train_dataloader = DataLoader(
            train_dataset, 
            shuffle=False, 
            batch_size=train_batch_size, 
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True  # 加速數據傳輸到 GPU
        )
        validation_dataloader = DataLoader(
            validation_dataset, 
            shuffle=True, 
            batch_size=train_batch_size, 
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        train_dataloader = DataLoader(
            train_dataset, 
            shuffle=False, 
            batch_size=train_batch_size, 
            collate_fn=data_collator,
            num_workers=num_workers,
            pin_memory=True
        )
        validation_dataloader = DataLoader(
            validation_dataset, 
            shuffle=True, 
            batch_size=train_batch_size, 
            collate_fn=data_collator,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_dataloader, validation_dataloader, test_dataset


def train(model, criterion, optimizer, train_dataloader, validation_dataloader, num_epochs, device, patience=0, use_blip_tuning=True):
    """訓練模型：前 3 個 epoch 訓練 BLIP encoder，之後只訓練 MLP head"""
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = transformers.get_scheduler(
        'linear',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    training_loss_list = []
    validation_loss_list = []
    
    # Early stopping 變數
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    early_stopped = False

    # 打印訓練策略
    if use_blip_tuning:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        blip_trainable = sum(p.numel() for p in model.blip.parameters() if p.requires_grad)
        print(f"\n[Training Strategy] BLIP Tuning = True")
        print(f"  - Epoch 0-2: Training Vision encoder + Text encoder + Cross-attention + MLP head")
        print(f"  - Epoch 3+: Freezing all BLIP (Vision + Text + Cross-attention), training MLP head only")
        print(f"  - Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"  - BLIP trainable parameters: {blip_trainable:,}")
    else:
        print(f"\n[Training Strategy] BLIP Tuning = False: All BLIP parameters frozen, only MLP head trainable")

    for epoch in tqdm(range(num_epochs)):
        training_loss = 0
        model.train()
        
        # 訓練策略：Epoch 3 後凍結 BLIP，只訓練 MLP head
        if use_blip_tuning and epoch == 3:
            print(f"\n[Training Strategy] Epoch {epoch}: Freezing all BLIP parameters (Vision + Text + Cross-attention), only training MLP head...")
            for param in model.blip.parameters():
                param.requires_grad = False
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4
            print(f"[Training Strategy] Learning rate adjusted to 1e-4 for MLP head only")

        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch.get('pixel_values')
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)
            
            # 前向傳播
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )
            
            # 使用 log-space labels（與預處理一致）
            labels = batch['labels'].to(device)
            loss = criterion(output, labels.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            training_loss += loss.item()

        print(f"Training loss for epoch {epoch}: {training_loss / len(train_dataloader)}")
        training_loss_list.append(training_loss / len(train_dataloader))
        
        # 驗證
        if epoch % 1 == 0:
            # 僅在第一個 epoch 打印詳細統計
            validation_metrics = eval_regression(model, validation_dataloader, device, print_stats=(epoch == 0))
            print(f'Validation metrics after epoch {epoch}:')
            for k, v in validation_metrics.items():
                print(f'  {k}: {v:.4f}', end='\t')
            print()
            
            # Early stopping 邏輯
            if patience > 0:
                current_val_loss = validation_metrics['L1 error (log)']
                
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    print(f'  [*] New best validation loss: {best_val_loss:.4f}')
                else:
                    patience_counter += 1
                    print(f'  [!] No improvement for {patience_counter}/{patience} epochs')
                    
                    if patience_counter >= patience:
                        print(f'\n[Early Stopping] No improvement for {patience} epochs. Stopping training...')
                        early_stopped = True
                        break
        
        if early_stopped:
            break
    
    # 恢復最佳模型權重
    if patience > 0 and best_model_state is not None:
        print(f'\n[Early Stopping] Restoring best model weights (val_loss={best_val_loss:.4f})')
        model.load_state_dict(best_model_state)
    
    return training_loss_list, validation_loss_list


def eval_regression(model, dataloader, device, print_stats=False):
    """評估回歸模型：計算 L1 和 MSE 誤差（在 log space，與訓練一致）"""
    l1loss = nn.L1Loss()
    mseloss = nn.MSELoss()
    model.eval()

    total_l1err = 0.0
    total_mse = 0.0
    total_samples = 0
    
    # 用於統計的變量
    all_predictions_log = []
    all_labels_log = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch.get('pixel_values')
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)
            
            # 模型預測的是 log(token_length)
            prediction_log = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )
            
            # 使用 log-space labels（與訓練一致）
            # labels 已經是 log(tokens + 1) 格式（從預處理階段）
            labels_log = batch['labels'].to(device)
            
            # 累積總誤差和總樣本數（更精確的計算方式）
            batch_size = labels_log.size(0)
            total_samples += batch_size
            
            # 在 log space 計算誤差（與訓練時的 loss 一致）
            l1err_batch = torch.abs(prediction_log - labels_log.type_as(prediction_log)).sum().item()
            mse_batch = ((prediction_log - labels_log.type_as(prediction_log)) ** 2).sum().item()
            
            total_l1err += l1err_batch
            total_mse += mse_batch
            
            # 收集統計信息（僅前幾個 batch）
            if print_stats and len(all_predictions_log) < 100:
                all_predictions_log.extend(prediction_log.cpu().numpy().tolist())
                all_labels_log.extend(labels_log.cpu().numpy().tolist())

    # 計算整體平均誤差（log space）
    metric = {
        'L1 error (log)': total_l1err / total_samples if total_samples > 0 else 0.0,
        'MSE (log)': total_mse / total_samples if total_samples > 0 else 0.0
    }
    
    # 打印統計信息（僅在第一次調用時）
    if print_stats and len(all_predictions_log) > 0:
        import numpy as np
        all_predictions_log = np.array(all_predictions_log)
        all_labels_log = np.array(all_labels_log)
        print(f"\n[Evaluation Stats] (sample of {len(all_predictions_log)} predictions, all in log space)")
        print(f"  Prediction log range: [{all_predictions_log.min():.2f}, {all_predictions_log.max():.2f}], mean: {all_predictions_log.mean():.2f}")
        print(f"  Label log range: [{all_labels_log.min():.2f}, {all_labels_log.max():.2f}], mean: {all_labels_log.mean():.2f}")
        # 也顯示轉換回原始空間的值（僅供參考）
        all_predictions_orig = np.exp(all_predictions_log) - 1.0
        all_labels_orig = np.exp(all_labels_log) - 1.0
        print(f"  Prediction (original space) range: [{all_predictions_orig.min():.1f}, {all_predictions_orig.max():.1f}], mean: {all_predictions_orig.mean():.1f}")
        print(f"  Label (original space) range: [{all_labels_orig.min():.1f}, {all_labels_orig.max():.1f}], mean: {all_labels_orig.mean():.1f}")
    
    return metric


def predict(model, dataloader, device):
    """預測：返回預測的 token 長度和實際長度"""
    model.eval()
    predicted_lengths = []
    actual_lengths = []
    latencies = []
    
    with torch.no_grad():
        for batch in dataloader:
            start_time = time.time()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch.get('pixel_values')
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)
            
            # 模型預測的是 log(token_length)
            predictions_log = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )
            
            # 轉換回原始空間
            # 預處理時使用 log(tokens + 1)，所以這裡需要 exp(prediction) - 1
            predictions = (torch.exp(predictions_log) - 1.0).cpu().numpy()
            lengths = batch['num_tokens'].numpy()
            
            end_time = time.time()
            
            predicted_lengths.extend(predictions)
            actual_lengths.extend(lengths)
            latencies.append(end_time - start_time)

    df = pd.DataFrame({
        'actual_length': actual_lengths,
        'predicted_length': predicted_lengths,
        'latency': latencies
    })
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BLIP model for log-space token length prediction')
    parser.add_argument('--data_size', type=int, help='Size of the dataset (in thousands)', default=100)
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs', default=6)
    parser.add_argument('--patience', type=int, help='Early stopping patience (0 to disable)', default=5)
    parser.add_argument('--dataset_path', type=str, help='Path to preprocessed dataset', required=True)
    parser.add_argument('--backbone', type=str, help='BLIP model local path', default='/data2/w00917303/decode-router/blip-image-captioning-base/')
    parser.add_argument('--output_dir', type=str, help='Output directory for results', default='./results')
    parser.add_argument('--model_dir', type=str, help='Directory to save model weights', default='./models')
    parser.add_argument('--metrics_dir', type=str, help='Directory to save metrics', default='./metrics')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize BLIP processor from local path (offline use)
    print(f'Loading BLIP processor from local dir: {args.backbone} ...')
    blip_processor = BlipProcessor.from_pretrained(args.backbone, local_files_only=True)
    # Use BLIP processor's tokenizer instead of BERT tokenizer (offline, no download needed)
    text_tokenizer = blip_processor.tokenizer
    text_tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    print(f'Using BLIP tokenizer (offline mode, no download)')

    # Load dataset
    dataset = datasets.load_from_disk(args.dataset_path)
    print(f'Loaded dataset from {args.dataset_path}')
    print(f'Dataset size: {len(dataset)} samples')
    
    # 檢查資料集是否有圖像欄位
    has_images = 'image' in dataset.column_names
    print(f'Dataset has images: {has_images}')

    # 訓練參數
    num_epochs = args.num_epochs
    train_batch_size = 16
    test_batch_size = 1
    lr = 1e-5  # 初始學習率（BLIP tuning 時使用）
    use_blip_tuning = True  # 前 3 個 epoch 訓練 BLIP encoder

    # 生成數據加載器
    train_dataloader, validation_dataloader, test_dataset = generate_dataloaders(
        dataset, train_batch_size, test_batch_size, text_tokenizer, 
        blip_processor=blip_processor if has_images else None
    )
    
    # 為 test_dataloader 也使用相同的 collate function（同樣需要 num_workers）
    num_workers = min(8, max(1, (os.cpu_count() or 1) // 2))
    if has_images:
        def test_collate_fn(batch):
            return blip_collate_fn(batch, text_tokenizer, blip_processor)
        test_dataloader = DataLoader(
            test_dataset, 
            shuffle=False, 
            batch_size=test_batch_size, 
            collate_fn=test_collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        data_collator = DataCollatorWithPadding(tokenizer=text_tokenizer)
        test_dataloader = DataLoader(
            test_dataset, 
            shuffle=False, 
            batch_size=test_batch_size, 
            collate_fn=data_collator,
            num_workers=num_workers,
            pin_memory=True
        )
    
    # 創建模型
    config = AutoConfig.from_pretrained(args.backbone, local_files_only=True)
    model = BlipRegressionModel(args.backbone, hidden_dim=128).to(device)
    criterion = nn.MSELoss()  # 使用 MSE loss（預測 log space）
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)

    # Training
    print("Start training...")
    if args.patience > 0:
        print(f"Early stopping enabled with patience={args.patience}")
    
    training_loss_list, validation_loss_list = train(
        model, 
        criterion, 
        optimizer, 
        train_dataloader, 
        validation_dataloader, 
        num_epochs, 
        device,
        patience=args.patience,
        use_blip_tuning=use_blip_tuning
    )

    # 最終驗證
    validation_metrics = eval_regression(model, validation_dataloader, device)
    print(f'\nFinal validation metrics:')
    for k, v in validation_metrics.items():
        print(f'{k}: {v:.4f}')

    # 保存模型權重
    os.makedirs(args.model_dir, exist_ok=True)
    model_filename = f'blip_regression_log_{args.data_size}K.pth'
    torch.save(model.state_dict(), os.path.join(args.model_dir, model_filename))
    print(f'Saved model weights to {os.path.join(args.model_dir, model_filename)}')

    # Inference
    print("\nStart inference...")
    df = predict(model, test_dataloader, device)
    
    # 保存結果
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = f'predictions_blip_regression_log_{args.data_size}K.csv'
    df.to_csv(os.path.join(args.output_dir, output_filename), index=False)
    print(f'Saved results to {os.path.join(args.output_dir, output_filename)}')

    # 測試集評估
    test_metrics = eval_regression(model, test_dataloader, device)
    print(f'\nTest set metrics:')
    os.makedirs(args.metrics_dir, exist_ok=True)
    metrics_filename = f'blip_regression_log_{args.data_size}K.txt'
    with open(os.path.join(args.metrics_dir, metrics_filename), 'w') as f:
        for k, v in test_metrics.items():
            f.write(f'{k}: {v:.4f}\n')
            print(f'{k}: {v:.4f}')
    print(f'Saved metrics to {os.path.join(args.metrics_dir, metrics_filename)}')