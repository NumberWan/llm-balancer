import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt

import matplotlib.ticker as ticker
import seaborn as sns
import sys
import os

# ==================== 配置區域 ====================
if len(sys.argv) > 1:
    csv_file = sys.argv[1]
else:
    # 預設檔案（可修改）
    csv_file = '/data2/w00917303/decode-router/Multimodal-serving-with-proxy-models/results/predictions_blip_regression_log_100K_logspace.csv'
    #csv_file = '/data2/w00917303/decode-router/LLM-serving-with-proxy-models/results/predictions_vicuna-13b_warmup_reg_mse_10000K.csv'
    #csv_file ='/data2/w00917303/decode-router/LLM-serving-with-proxy-models/results/predictions_vicuna-13b_warmup_ordinal_multi_cls_mse_10000K.csv'
# 從檔名判斷任務類型
if 'ordinal_multi_cls' in csv_file:
    task_type = 3
    task_name = 'TASK_TYPE=3 (Ordinal Multi-class)'
    is_classification = True
elif 'multi_cls' in csv_file:
    task_type = 2
    task_name = 'TASK_TYPE=2 (Multi-class Classification)'
    is_classification = True
elif '_cls_' in csv_file and 'multi' not in csv_file:
    task_type = 1
    task_name = 'TASK_TYPE=1 (3-Class Classification)'
    is_classification = True
else:
    task_type = 0
    task_name = 'TASK_TYPE=0 (Regression)'
    is_classification = False

print("=" * 70)
print(f"Analyzing: {os.path.basename(csv_file)}")
print(f"Task: {task_name}")
print("=" * 70)

# 載入 CSV
try:
    df = pd.read_csv(csv_file)
    print(f"✓ Loaded {len(df)} samples")
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit(1)

# ==================== 定義閾值（用於所有任務類型）====================
if task_type == 1:
    thresholds = [100, 330]  # 3 分類
    num_classes = 3
else:
    thresholds = [36, 113, 242, 395]  # 5 分類
    #thresholds = [3.583519, 4.7361984  , 5.497168, 5.986452]
    num_classes = 5

# ==================== 回歸任務分析 ====================
if task_type == 0:
    print("\n" + "=" * 70)
    print("Regression Results")
    print("=" * 70)
    
    # 計算誤差
    errors = df['predicted_label'] - df['actual_length']
    abs_errors = np.abs(errors)
    relative_errors = abs_errors / (df['actual_length'] + 1e-8) * 100  # 避免除以 0
    
    # 統計指標
    mae = abs_errors.mean()
    rmse = np.sqrt((errors ** 2).mean())
    median_ae = np.median(abs_errors)
    mape = relative_errors.mean()
    
    print(f"Mean Absolute Error (MAE):     {mae:.2f} tokens")
    print(f"Root Mean Square Error (RMSE): {rmse:.2f} tokens")
    print(f"Median Absolute Error:         {median_ae:.2f} tokens")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    print(f"Min Error: {errors.min():.2f} tokens")
    print(f"Max Error: {errors.max():.2f} tokens")
    print(f"Std Dev:   {errors.std():.2f} tokens")
    
    # 視覺化（簡化版：只有散點圖）
    plt.figure(figsize=(10, 8))
    plt.scatter(df['actual_length'], df['predicted_label'], alpha=0.5, s=10, c='blue', edgecolors='none')
    max_val = max(df['actual_length'].max(), df['predicted_label'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2.5, label='Perfect Prediction')
    plt.xlabel('Actual Length (tokens)', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted Length (tokens)', fontsize=14, fontweight='bold')
    plt.title(f'Predicted vs Actual', 
              fontsize=15, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    #plt.xlim(0, 3000)
    #plt.ylim(0, 3000)
    #==================================
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    #==================================
    plt.tight_layout()
    
    output_png = csv_file.replace('.csv', '_scatter.png')
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"\n✓ Scatter plot saved to: {output_png}")
    
    # ==================== 轉換成分類並產生混淆矩陣 ====================
    print("\n" + "=" * 70)
    print("Converting to Classification (for comparison)")
    print("=" * 70)
    print(f"Using thresholds: {thresholds}")
    print()
    
    # 定義類別名稱
    if num_classes == 3:
        class_names = [
            f'Short\n(<{thresholds[0]})', 
            f'Medium\n({thresholds[0]}-{thresholds[1]-1})', 
            f'Long\n(≥{thresholds[1]})'
        ]
        class_names_simple = [
            f'Short (<{thresholds[0]})', 
            f'Medium ({thresholds[0]}-{thresholds[1]-1})', 
            f'Long (≥{thresholds[1]})'
        ]
    else:
        class_names = [
            f'Very Short\n(<{thresholds[0]})', 
            f'Short\n({thresholds[0]}-{thresholds[1]-1})', 
            f'Medium\n({thresholds[1]}-{thresholds[2]-1})', 
            f'Long\n({thresholds[2]}-{thresholds[3]-1})', 
            f'Very Long\n(≥{thresholds[3]})'
        ]
        class_names_simple = [
            f'Very Short (<{thresholds[0]})', 
            f'Short ({thresholds[0]}-{thresholds[1]-1})', 
            f'Medium ({thresholds[1]}-{thresholds[2]-1})', 
            f'Long ({thresholds[2]}-{thresholds[3]-1})', 
            f'Very Long (≥{thresholds[3]})'
        ]
    
    def classify(length):
        """將長度轉換為類別"""
        if pd.isna(length):
            return -1
        for i, threshold in enumerate(thresholds):
            if length < threshold:
                return i
        return len(thresholds)
    
    # 轉換為類別
    df['actual_class'] = df['actual_length'].apply(classify)
    df['predicted_class'] = df['predicted_label'].apply(classify)
    
    # 移除無效樣本
    df = df[(df['actual_class'] >= 0) & (df['predicted_class'] >= 0)]
    
    # 計算分類指標
    cls_accuracy = accuracy_score(df['actual_class'], df['predicted_class'])
    cls_f1 = f1_score(df['actual_class'], df['predicted_class'], average='macro', zero_division=0)
    cls_precision = precision_score(df['actual_class'], df['predicted_class'], average='macro', zero_division=0)
    cls_recall = recall_score(df['actual_class'], df['predicted_class'], average='macro', zero_division=0)
    
    # 更新散點圖的標題（加入分類準確率）
    for fig_num in matplotlib.pyplot.get_fignums():
        if fig_num == 1:  # 第一個圖（散點圖）
            fig = matplotlib.pyplot.figure(fig_num)
            ax = fig.axes[0]
            ax.set_title(f'Predicted vs Actual', 
                        fontsize=15, fontweight='bold')
            fig.savefig(csv_file.replace('.csv', '_scatter.png'), dpi=150, bbox_inches='tight')
    
    print("Classification Metrics (converted from regression):")
    print(f"Accuracy:  {cls_accuracy:.4f} ({cls_accuracy*100:.2f}%)")
    print(f"F1 Score:  {cls_f1:.4f}")
    print(f"Precision: {cls_precision:.4f}")
    print(f"Recall:    {cls_recall:.4f}")
    print()
    
    # 各類別分布
    print("Class Distribution:")
    class_counts = df['actual_class'].value_counts().sort_index()
    for i in range(num_classes):
        if i in class_counts.index:
            count = class_counts[i]
            percentage = count / len(df) * 100
            print(f"  Class {i} ({class_names_simple[i]}): {count} samples ({percentage:.1f}%)")
    print()
    
    # 混淆矩陣
    cm = confusion_matrix(df['actual_class'], df['predicted_class'], labels=range(num_classes))
    print("Confusion Matrix:")
    print(cm)
    print()
    
    # 各類別準確率
    print("Per-Class Accuracy:")
    for i in range(num_classes):
        class_mask = df['actual_class'] == i
        if class_mask.sum() > 0:
            class_acc = (df.loc[class_mask, 'actual_class'] == df.loc[class_mask, 'predicted_class']).mean()
            class_count = class_mask.sum()
            print(f"  Class {i} ({class_names_simple[i]}): {class_acc:.4f} ({class_acc*100:.2f}%) - {class_count} samples")
    
    # 產生混淆矩陣圖表
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names,
                cbar_kws={'label': 'Number of Samples'})
    plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
    plt.ylabel('Actual Class', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix - {task_name} (Converted to Classification)\nThresholds: {thresholds}\nAccuracy: {cls_accuracy:.2%}', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_cm_png = csv_file.replace('.csv', '_confusion_matrix.png')
    plt.savefig(output_cm_png, dpi=150, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to: {output_cm_png}")
    
    # 儲存詳細結果
    df['error'] = errors
    df['abs_error'] = abs_errors
    df['relative_error_%'] = relative_errors
    output_csv = csv_file.replace('.csv', '_with_classes.csv')
    df.to_csv(output_csv, index=False)
    print(f"✓ Detailed results saved to: {output_csv}")

# ==================== 分類任務分析 ====================
else:
    # 定義閾值
    if task_type == 1:
        thresholds = [100, 330]  # 3 分類
        num_classes = 3
    else:
        thresholds = [36, 113, 242, 395]  # 5 分類
        num_classes = 5
    
    print("\n" + "=" * 70)
    print("Threshold Configuration")
    print("=" * 70)
    print(f"Thresholds: {thresholds}")
    print(f"Number of classes: {num_classes}")
    print("=" * 70)
    print()
    
    # 定義類別名稱
    if num_classes == 3:
        class_names = [
            f'Short\n(<{thresholds[0]})', 
            f'Medium\n({thresholds[0]}-{thresholds[1]-1})', 
            f'Long\n(≥{thresholds[1]})'
        ]
        class_names_simple = [
            f'Short (<{thresholds[0]})', 
            f'Medium ({thresholds[0]}-{thresholds[1]-1})', 
            f'Long (≥{thresholds[1]})'
        ]
    else:
        class_names = [
            f'Very Short\n(<{thresholds[0]})', 
            f'Short\n({thresholds[0]}-{thresholds[1]-1})', 
            f'Medium\n({thresholds[1]}-{thresholds[2]-1})', 
            f'Long\n({thresholds[2]}-{thresholds[3]-1})', 
            f'Very Long\n(≥{thresholds[3]})'
        ]
        class_names_simple = [
            f'Very Short (<{thresholds[0]})', 
            f'Short ({thresholds[0]}-{thresholds[1]-1})', 
            f'Medium ({thresholds[1]}-{thresholds[2]-1})', 
            f'Long ({thresholds[2]}-{thresholds[3]-1})', 
            f'Very Long (≥{thresholds[3]})'
        ]
    
    def classify(length):
        """將長度轉換為類別"""
        if pd.isna(length):
            return -1
        for i, threshold in enumerate(thresholds):
            if length < threshold:
                return i
        return len(thresholds)
    
    # 轉換為類別
    df['actual_class'] = df['actual_length'].apply(classify)
    
    # 對於 TASK_TYPE=3（序數多分類），predicted_label 已經是類別（0-4）
    if task_type == 3:
        df['predicted_class'] = df['predicted_label'].round().astype(int).clip(0, num_classes-1)
    else:
        # 對於 TASK_TYPE=1, 2，也從長度轉換
        df['predicted_class'] = df['predicted_label'].apply(classify)
    
    # 移除無效樣本
    df = df[(df['actual_class'] >= 0) & (df['predicted_class'] >= 0)]
    
    # 計算分類指標
    accuracy = accuracy_score(df['actual_class'], df['predicted_class'])
    f1 = f1_score(df['actual_class'], df['predicted_class'], average='macro', zero_division=0)
    precision = precision_score(df['actual_class'], df['predicted_class'], average='macro', zero_division=0)
    recall = recall_score(df['actual_class'], df['predicted_class'], average='macro', zero_division=0)
    
    print("=" * 70)
    print(f"Classification Results - {task_name}")
    print("=" * 70)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print()
    
    # 各類別的樣本數
    print("Class Distribution:")
    class_counts = df['actual_class'].value_counts().sort_index()
    print(class_counts)
    total = len(df)
    for i in range(num_classes):
        if i in class_counts.index:
            count = class_counts[i]
            percentage = count / total * 100
            print(f"  Class {i} ({class_names_simple[i]}): {count} samples ({percentage:.1f}%)")
        else:
            print(f"  Class {i} ({class_names_simple[i]}): 0 samples (0.0%)")
    print()
    
    # 混淆矩陣
    cm = confusion_matrix(df['actual_class'], df['predicted_class'], labels=range(num_classes))
    print("Confusion Matrix:")
    print(cm)
    print()
    
    # 各類別準確率
    print("Per-Class Accuracy:")
    for i in range(num_classes):
        class_mask = df['actual_class'] == i
        if class_mask.sum() > 0:
            class_acc = (df.loc[class_mask, 'actual_class'] == df.loc[class_mask, 'predicted_class']).mean()
            class_count = class_mask.sum()
            print(f"  Class {i} ({class_names_simple[i]}): {class_acc:.4f} ({class_acc*100:.2f}%) - {class_count} samples")
        else:
            print(f"  Class {i} ({class_names_simple[i]}): N/A (0 samples)")
    
    # 視覺化混淆矩陣
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names,
                cbar_kws={'label': 'Number of Samples'})
    plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
    plt.ylabel('Actual Class', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix - {task_name}\nThresholds: {thresholds}\nAccuracy: {accuracy:.2%}', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_png = csv_file.replace('.csv', '_confusion_matrix.png')
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to: {output_png}")
    
    # 儲存結果
    output_csv = csv_file.replace('.csv', '_with_classes.csv')
    df.to_csv(output_csv, index=False)
    print(f"✓ Results saved to: {output_csv}")
    
    # 建議
    print()
    print("=" * 70)
    print("Suggested Code Modification (if needed)")
    print("=" * 70)
    print(f"Modify preprocess_dataset.py line 217:")
    if num_classes == 3:
        print(f"  multi_cls_thresholds = {thresholds + [1000000]}  # for TASK_TYPE=1")
    else:
        print(f"  multi_cls_thresholds = {thresholds + [1000000]}  # for TASK_TYPE=2,3")
    print("=" * 70)