#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
唇拡大前処理スクリプト

機能:
- CSVファイルから動画（pt）ファイルのパスを読み込み
- 各動画の唇領域をcrop ratio 0.7で拡大
- 新しいptファイルとして保存
- 新しいCSVファイルを生成
"""

import torch
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

# ========================================
# 設定（ここを編集）
# ========================================

# 入力CSVファイル
INPUT_CSV = "/home/bv20049/dataset/npz/zundadata/processed/final_valid_clean.csv"

# 出力先
OUTPUT_CSV = "/home/bv20049/dataset/npz/zundadata/processed/final_valid_cropp.csv"
OUTPUT_VIDEO_DIR = "/home/bv20049/dataset/npz/zundadata/processed/videos_enlarged/valid"

# クロップ設定
CROP_RATIO = 0.7  # クロップ比率（0.7 = 1.43倍拡大）
DETECTION_METHOD = 'center'  # 'center', 'bottom', 'auto'

# その他設定
OVERWRITE = False  # すでに処理済みのファイルを上書きするか


# ========================================
# 唇位置検出
# ========================================

def detect_lip_center(frame):
    """画像中心を唇の位置と仮定"""
    H, W = frame.shape
    return H // 2, W // 2


def detect_lip_bottom(frame):
    """画像下部を唇の位置と仮定"""
    H, W = frame.shape
    return int(H * 0.65), W // 2


def detect_lip_auto(frame):
    """自動で唇領域を検出"""
    H, W = frame.shape
    
    # 画像の下半分に注目
    lower_half = frame[H//3:, :]
    
    # エッジ検出
    edges = cv2.Canny((lower_half * 255).astype(np.uint8), 50, 150)
    
    # 水平方向のエッジが多い場所を探す
    horizontal_edges = np.sum(edges, axis=1)
    
    if horizontal_edges.max() > 0:
        lip_y_in_lower = np.argmax(horizontal_edges)
        lip_y = H//3 + lip_y_in_lower
        lip_x = W // 2
    else:
        lip_y = int(H * 0.7)
        lip_x = W // 2
    
    return lip_y, lip_x


def detect_lip_region(frame, method='center'):
    """唇領域を検出"""
    if method == 'center':
        return detect_lip_center(frame)
    elif method == 'bottom':
        return detect_lip_bottom(frame)
    elif method == 'auto':
        return detect_lip_auto(frame)
    else:
        raise ValueError(f"Unknown method: {method}")


# ========================================
# クロップ処理
# ========================================

def crop_and_enlarge_frame(frame, crop_ratio, detection_method='center'):
    """
    1フレームをクロップして拡大
    
    Args:
        frame: (H, W) のグレースケール画像
        crop_ratio: クロップ比率
        detection_method: 検出方法
    
    Returns:
        enlarged_frame: 拡大後の画像（元のサイズ）
    """
    H, W = frame.shape
    
    # 唇の位置を検出
    lip_y, lip_x = detect_lip_region(frame, detection_method)
    
    # クロップサイズを計算
    crop_H = int(H * crop_ratio)
    crop_W = int(W * crop_ratio)
    
    # クロップ範囲を計算
    y1 = max(0, lip_y - crop_H // 2)
    y2 = min(H, lip_y + crop_H // 2)
    x1 = max(0, lip_x - crop_W // 2)
    x2 = min(W, lip_x + crop_W // 2)
    
    # 範囲調整
    if y2 - y1 < crop_H:
        if y1 == 0:
            y2 = min(H, y1 + crop_H)
        else:
            y1 = max(0, y2 - crop_H)
    
    if x2 - x1 < crop_W:
        if x1 == 0:
            x2 = min(W, x1 + crop_W)
        else:
            x1 = max(0, x2 - crop_W)
    
    # クロップ
    cropped = frame[y1:y2, x1:x2]
    
    # リサイズ（元のサイズに戻す = 拡大）
    enlarged = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_CUBIC)
    
    return enlarged


def process_video(video_tensor, crop_ratio=0.7, detection_method='center'):
    """
    動画全体を処理
    
    Args:
        video_tensor: 動画テンソル [T, H, W] or [T, C, H, W] or [T, 1, H, W]
        crop_ratio: クロップ比率
        detection_method: 検出方法
    
    Returns:
        processed_video: 処理後の動画テンソル（元と同じ形状）
    """
    original_shape = video_tensor.shape
    
    # 形状を調整
    video = video_tensor.clone()
    
    # [T, 1, H, W] → [T, H, W]
    if video.dim() == 4 and video.shape[1] == 1:
        video = video.squeeze(1)
    
    # [1, T, H, W] → [T, H, W]
    elif video.dim() == 4 and video.shape[0] == 1:
        video = video.squeeze(0)
    
    # [1, T, 1, H, W] → [T, H, W]
    elif video.dim() == 5:
        if video.shape[0] == 1:
            video = video.squeeze(0)
        if video.dim() == 4 and video.shape[1] == 1:
            video = video.squeeze(1)
    
    # 期待される形状: [T, H, W]
    if video.dim() != 3:
        raise ValueError(f"Unexpected video shape after adjustment: {video.shape}")
    
    T, H, W = video.shape
    
    # 各フレームを処理
    processed_frames = []
    
    for t in range(T):
        frame = video[t].numpy()
        
        # クロップ＆拡大
        enlarged_frame = crop_and_enlarge_frame(frame, crop_ratio, detection_method)
        
        processed_frames.append(enlarged_frame)
    
    # テンソルに戻す
    processed_video = torch.from_numpy(np.array(processed_frames))
    
    # 元の形状に戻す
    if len(original_shape) == 4 and original_shape[1] == 1:
        # [T, H, W] → [T, 1, H, W]
        processed_video = processed_video.unsqueeze(1)
    elif len(original_shape) == 4 and original_shape[0] == 1:
        # [T, H, W] → [1, T, H, W]
        processed_video = processed_video.unsqueeze(0)
    elif len(original_shape) == 5:
        # [T, H, W] → [1, T, 1, H, W]
        processed_video = processed_video.unsqueeze(1).unsqueeze(0)
    
    return processed_video


# ========================================
# メイン処理
# ========================================

def process_dataset(input_csv, output_csv, output_video_dir, 
                   crop_ratio=0.7, detection_method='center', overwrite=False):
    """
    データセット全体を処理
    
    Args:
        input_csv: 入力CSVファイルパス
        output_csv: 出力CSVファイルパス
        output_video_dir: 出力動画ディレクトリ
        crop_ratio: クロップ比率
        detection_method: 検出方法
        overwrite: 上書きするか
    """
    print("="*70)
    print("唇拡大前処理")
    print("="*70)
    print(f"入力CSV: {input_csv}")
    print(f"出力CSV: {output_csv}")
    print(f"出力先: {output_video_dir}")
    print(f"クロップ比率: {crop_ratio} (拡大率: {1/crop_ratio:.2f}x)")
    print(f"検出方法: {detection_method}")
    print("="*70)
    
    # CSVを読み込み
    try:
        df = pd.read_csv(input_csv)
        print(f"\n✓ CSV読み込み完了: {len(df)}行")
    except Exception as e:
        print(f"❌ CSV読み込みエラー: {e}")
        return
    
    # 必要なカラムを確認
    if 'video_path' not in df.columns:
        print("❌ CSVに'video_path'カラムがありません")
        return
    
    # 出力ディレクトリを作成
    output_dir = Path(output_video_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 処理結果を記録
    results = []
    success_count = 0
    skip_count = 0
    error_count = 0
    
    # 各動画を処理
    print("\n動画処理中...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        video_path = row['video_path']
        
        # ファイル名を生成
        video_file = Path(video_path)
        output_filename = f"enlarged_{video_file.name}"
        output_path = output_dir / output_filename
        
        # すでに存在する場合
        if output_path.exists() and not overwrite:
            skip_count += 1
            results.append({
                **row.to_dict(),
                'video_path': str(output_path),
                'status': 'skipped'
            })
            continue
        
        try:
            # 動画を読み込み
            data = torch.load(video_path)
            
            if isinstance(data, dict) and 'video' in data:
                video = data['video']
                other_data = {k: v for k, v in data.items() if k != 'video'}
            elif isinstance(data, torch.Tensor):
                video = data
                other_data = {}
            else:
                raise ValueError("Unsupported data structure")
            
            # 処理
            processed_video = process_video(video, crop_ratio, detection_method)
            
            # 保存
            if other_data:
                # 他のデータも含める
                save_data = {'video': processed_video, **other_data}
            else:
                save_data = processed_video
            
            torch.save(save_data, output_path)
            
            success_count += 1
            results.append({
                **row.to_dict(),
                'video_path': str(output_path),
                'status': 'success'
            })
            
        except Exception as e:
            error_count += 1
            print(f"\n  ❌ エラー [{video_path}]: {e}")
            results.append({
                **row.to_dict(),
                'video_path': video_path,  # 元のパスのまま
                'status': 'error',
                'error_message': str(e)
            })
            continue
    
    # 新しいCSVを保存
    result_df = pd.DataFrame(results)
    
    # status列を削除（オプション）
    if 'status' in result_df.columns:
        result_df = result_df.drop(columns=['status'])
    if 'error_message' in result_df.columns:
        result_df = result_df.drop(columns=['error_message'])
    
    result_df.to_csv(output_csv, index=False)
    
    # サマリー
    print("\n" + "="*70)
    print("処理完了")
    print("="*70)
    print(f"総数: {len(df)}")
    print(f"✓ 成功: {success_count}")
    print(f"⊖ スキップ: {skip_count}")
    print(f"✗ エラー: {error_count}")
    print(f"\n出力CSV: {output_csv}")
    print(f"出力動画: {output_video_dir}")
    print("="*70)
    
    print(f"\n次のステップ:")
    print(f"  python train.py --input {output_csv}")


# ========================================
# サンプル確認用
# ========================================

def preview_sample(input_csv, num_samples=3, crop_ratio=0.7, detection_method='center'):
    """
    処理前後を比較表示（サンプル確認用）
    
    Args:
        input_csv: 入力CSVファイル
        num_samples: 確認するサンプル数
        crop_ratio: クロップ比率
        detection_method: 検出方法
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    print("="*70)
    print("サンプル確認")
    print("="*70)
    
    # CSVを読み込み
    df = pd.read_csv(input_csv)
    
    # ランダムにサンプルを選択
    samples = df.sample(min(num_samples, len(df)))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (_, row) in enumerate(samples.iterrows()):
        video_path = row['video_path']
        
        try:
            # 動画を読み込み
            data = torch.load(video_path)
            
            if isinstance(data, dict) and 'video' in data:
                video = data['video']
            elif isinstance(data, torch.Tensor):
                video = data
            else:
                continue
            
            # 形状調整
            if video.dim() == 4 and video.shape[1] == 1:
                video = video.squeeze(1)
            elif video.dim() == 4 and video.shape[0] == 1:
                video = video.squeeze(0)
            elif video.dim() == 5:
                if video.shape[0] == 1:
                    video = video.squeeze(0)
                if video.dim() == 4 and video.shape[1] == 1:
                    video = video.squeeze(1)
            
            # 最初のフレーム
            frame = video[0].numpy()
            
            # 処理
            enlarged = crop_and_enlarge_frame(frame, crop_ratio, detection_method)
            
            # 差分
            diff = np.abs(enlarged - frame)
            
            # 表示
            axes[idx, 0].imshow(frame, cmap='gray')
            axes[idx, 0].set_title('Original')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(enlarged, cmap='gray')
            axes[idx, 1].set_title(f'Enlarged ({1/crop_ratio:.2f}x)')
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(diff, cmap='hot')
            axes[idx, 2].set_title('Difference')
            axes[idx, 2].axis('off')
            
            print(f"✓ サンプル {idx+1}: {Path(video_path).name}")
            
        except Exception as e:
            print(f"✗ サンプル {idx+1}: エラー - {e}")
    
    plt.tight_layout()
    plt.savefig('preview_lip_enlargement.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ プレビュー保存: preview_lip_enlargement.png")
    print("="*70)


# ========================================
# メイン
# ========================================

def main():
    parser = argparse.ArgumentParser(description='唇拡大前処理')
    parser.add_argument('--input', type=str, default=INPUT_CSV,
                       help='入力CSVファイル')
    parser.add_argument('--output', type=str, default=OUTPUT_CSV,
                       help='出力CSVファイル')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_VIDEO_DIR,
                       help='出力動画ディレクトリ')
    parser.add_argument('--crop-ratio', type=float, default=CROP_RATIO,
                       help='クロップ比率（デフォルト: 0.7）')
    parser.add_argument('--method', type=str, default=DETECTION_METHOD,
                       choices=['center', 'bottom', 'auto'],
                       help='検出方法（デフォルト: center）')
    parser.add_argument('--overwrite', action='store_true',
                       help='既存ファイルを上書き')
    parser.add_argument('--preview', action='store_true',
                       help='サンプル確認のみ（処理しない）')
    
    args = parser.parse_args()
    
    # プレビューモード
    if args.preview:
        preview_sample(args.input, num_samples=3, 
                      crop_ratio=args.crop_ratio, 
                      detection_method=args.method)
        return
    
    # メイン処理
    process_dataset(
        input_csv=args.input,
        output_csv=args.output,
        output_video_dir=args.output_dir,
        crop_ratio=args.crop_ratio,
        detection_method=args.method,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()