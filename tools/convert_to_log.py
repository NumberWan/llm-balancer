"""
將預測結果 CSV 轉換為 log 空間版本
保持相同的格式（actual_length, predicted_length, latency），
但 actual_length 和 predicted_length 轉換為 log 空間的值
"""
import pandas as pd
import numpy as np
import argparse
import os


def convert_to_log_csv(input_csv_path, output_csv_path):
    """
    讀取預測結果 CSV，將 actual_length 和 predicted_length 轉換為 log 空間
    
    Args:
        input_csv_path: 輸入 CSV 文件路徑（包含原始空間的 actual_length 和 predicted_length）
        output_csv_path: 輸出 CSV 文件路徑（包含 log 空間的值）
    """
    # 讀取預測結果
    df = pd.read_csv(input_csv_path)
    
    # 檢查必要的欄位是否存在
    required_columns = ['actual_length', 'predicted_length']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # 轉換到 log 空間（與訓練時一致：log(tokens + 1)）
    df_log = pd.DataFrame({
        'actual_length': np.log(df['actual_length'] + 1.0),
        'predicted_length': np.log(df['predicted_length'] + 1.0),
        'latency': df['latency'] if 'latency' in df.columns else [0.0] * len(df)
    })
    
    # 保存為新的 CSV 文件
    df_log.to_csv(output_csv_path, index=False)
    
    print(f"Converted CSV file saved to: {output_csv_path}")
    print(f"Total samples: {len(df_log)}")
    print(f"\nLog space statistics:")
    print(f"  actual_length (log): mean={df_log['actual_length'].mean():.6f}, "
          f"std={df_log['actual_length'].std():.6f}, "
          f"min={df_log['actual_length'].min():.6f}, "
          f"max={df_log['actual_length'].max():.6f}")
    print(f"  predicted_length (log): mean={df_log['predicted_length'].mean():.6f}, "
          f"std={df_log['predicted_length'].std():.6f}, "
          f"min={df_log['predicted_length'].min():.6f}, "
          f"max={df_log['predicted_length'].max():.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert predictions CSV to log space')
    parser.add_argument('--input_csv', type=str,
                       default='./results/predictions_blip_regression_log_100K.csv',
                       help='Input CSV file path (original space)')
    parser.add_argument('--output_csv', type=str,
                       default='./results/predictions_blip_regression_log_100K_logspace.csv',
                       help='Output CSV file path (log space)')
    parser.add_argument('--data_size', type=int, default=100,
                       help='Dataset size in thousands (for auto-generating paths)')
    
    args = parser.parse_args()
    
    # 如果使用 data_size，自動生成路徑
    if args.input_csv == './results/predictions_blip_regression_log_100K.csv':
        args.input_csv = f'./results/predictions_blip_regression_log_{args.data_size}K.csv'
    if args.output_csv == './results/predictions_blip_regression_log_100K_logspace.csv':
        args.output_csv = f'./results/predictions_blip_regression_log_{args.data_size}K_logspace.csv'
    
    # 檢查輸入文件是否存在
    if not os.path.exists(args.input_csv):
        print(f"Error: Input CSV file not found: {args.input_csv}")
        print("Please provide the correct path to the predictions CSV file.")
        exit(1)
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    # 轉換 CSV
    convert_to_log_csv(args.input_csv, args.output_csv)