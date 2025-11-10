#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
クロップ比率比較可視化スクリプト（形状対応版）

機能:
- 1つのサンプル動画に対して
- 複数のcrop_ratioを試す
- 結果を一括で比較表示

修正内容:
- torch.Size([1, 40, 1, 64, 64]) のような形状に対応
- 形状を自動で調整
- GUI不要のバックエンド（Agg）を使用
"""

import torch
import numpy as np
import cv2

# ========================================
# ★ matplotlibのバックエンドをAggに設定（GUI不要）
# ========================================
import matplotlib
matplotlib.use('Agg')  # この行を追加
import matplotlib.pyplot as plt

from pathlib import Path

# ========================================
# 設定（ここを編集）
# ========================================

# サンプルptファイル
SAMPLE_PT_FILE = "/home/bv20049/dataset/npz/zundadata/processed/video_segments/LFROI_emoNormal001_emoNormal001_0001.pt"  # サンプルファイルのパス

# 試すクロップ比率のリスト
CROP_RATIOS = [0.8, 0.7, 0.6, 0.5, 0.4]  # 複数の比率を試す

# 唇検出方法
LIP_DETECTION_METHOD = 'center'  # 'auto', 'center', 'bottom'

# 表示設定
SHOW_FRAME_INDEX = 0  # 表示するフレーム（0 = 最初のフレーム）
OUTPUT_IMAGE = "/home/bv20049/dataset/npz/zundadata/processed/video_segments/crop_ratio_comparison.png"  # 出力画像ファイル名


# ========================================
# 唇位置検出（前のスクリプトと同じ）
# ========================================

def detect_lip_region_auto(frame):
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


def detect_lip_region_center(frame):
    """画像中心を唇の位置と仮定"""
    H, W = frame.shape
    return H // 2, W // 2


def detect_lip_region_bottom(frame):
    """画像下部を唇の位置と仮定"""
    H, W = frame.shape
    return int(H * 0.65), W // 2


def detect_lip_region(frame, method='auto'):
    """唇領域を検出"""
    if method == 'auto':
        return detect_lip_region_auto(frame)
    elif method == 'center':
        return detect_lip_region_center(frame)
    elif method == 'bottom':
        return detect_lip_region_bottom(frame)
    else:
        raise ValueError(f"Unknown method: {method}")


# ========================================
# クロップ処理
# ========================================

def crop_single_frame(frame, crop_ratio, detection_method='auto'):
    """
    1フレームをクロップ
    
    Args:
        frame: (H, W) のグレースケール画像
        crop_ratio: クロップ比率
        detection_method: 検出方法
    
    Returns:
        cropped_frame: クロップ後の画像
        lip_y, lip_x: 検出された唇の位置
        crop_box: クロップ範囲 (y1, y2, x1, x2)
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
    
    # リサイズ（元のサイズに戻す）
    resized = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_CUBIC)
    
    return resized, (lip_y, lip_x), (y1, y2, x1, x2)


# ========================================
# 可視化
# ========================================

def visualize_crop_comparison(pt_file, crop_ratios, detection_method='auto', 
                              frame_index=0, output_file='crop_ratio_comparison.png'):
    """
    複数のクロップ比率を比較表示
    
    Args:
        pt_file: ptファイルのパス
        crop_ratios: 試すクロップ比率のリスト
        detection_method: 検出方法
        frame_index: 表示するフレーム
        output_file: 出力画像ファイル名
    """
    print("="*70)
    print("クロップ比率比較可視化")
    print("="*70)
    print(f"サンプルファイル: {pt_file}")
    print(f"フレームインデックス: {frame_index}")
    print(f"検出方法: {detection_method}")
    print(f"試すクロップ比率: {crop_ratios}")
    print("="*70)
    
    # データを読み込み
    try:
        data = torch.load(pt_file)
        
        if isinstance(data, dict) and 'video' in data:
            video = data['video']
        elif isinstance(data, torch.Tensor):
            video = data
        else:
            raise ValueError("Unsupported data structure")
        
        print(f"\n動画サイズ: {video.shape}")
        
        # ========================================
        # ★ 形状を調整
        # ========================================
        # [T, 1, H, W] → [T, H, W]
        if video.dim() == 4 and video.shape[1] == 1:
            video = video.squeeze(1)
            print(f"調整: [T, 1, H, W] → [T, H, W]")
        
        # [1, T, H, W] → [T, H, W]
        elif video.dim() == 4 and video.shape[0] == 1:
            video = video.squeeze(0)
            print(f"調整: [1, T, H, W] → [T, H, W]")
        
        # [1, T, 1, H, W] → [T, H, W]
        elif video.dim() == 5:
            if video.shape[0] == 1:
                video = video.squeeze(0)
            if video.dim() == 4 and video.shape[1] == 1:
                video = video.squeeze(1)
            print(f"調整: 5次元 → [T, H, W]")
        
        print(f"最終サイズ: {video.shape}")
        
        # フレームを取得
        if video.dim() == 3:
            # (T, H, W)
            frame = video[frame_index].numpy()
        elif video.dim() == 4:
            # (T, C, H, W)
            frame = video[frame_index, 0].numpy()
        else:
            raise ValueError(f"Unsupported tensor shape: {video.shape}")
        
        H, W = frame.shape
        print(f"フレームサイズ: {H}x{W}")
        
    except Exception as e:
        print(f"❌ ファイル読み込みエラー: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 各クロップ比率で処理
    results = []
    
    print(f"\nクロップ処理中...")
    for ratio in crop_ratios:
        cropped, (lip_y, lip_x), (y1, y2, x1, x2) = crop_single_frame(
            frame, ratio, detection_method
        )
        
        enlarge_factor = 1 / ratio
        
        results.append({
            'ratio': ratio,
            'cropped': cropped,
            'lip_pos': (lip_y, lip_x),
            'crop_box': (y1, y2, x1, x2),
            'enlarge_factor': enlarge_factor
        })
        
        print(f"  ✓ ratio={ratio:.1f}: 唇が{enlarge_factor:.2f}倍に拡大")
    
    # 可視化
    print(f"\n可視化作成中...")
    
    num_ratios = len(crop_ratios)
    fig, axes = plt.subplots(2, num_ratios + 1, figsize=(4*(num_ratios+1), 8))
    
    # 元の画像（左端）
    ax_orig_top = axes[0, 0]
    ax_orig_top.imshow(frame, cmap='gray')
    ax_orig_top.set_title(f'Original\n{H}x{W}', fontsize=12, fontweight='bold')
    ax_orig_top.axis('off')
    
    # 唇の位置を表示
    lip_y, lip_x = results[0]['lip_pos']
    ax_orig_top.plot(lip_x, lip_y, 'r*', markersize=20, label='Lip Center')
    
    # 下段は空白
    axes[1, 0].axis('off')
    
    # 各クロップ比率の結果
    for i, result in enumerate(results, 1):
        ratio = result['ratio']
        cropped = result['cropped']
        lip_y, lip_x = result['lip_pos']
        y1, y2, x1, x2 = result['crop_box']
        enlarge = result['enlarge_factor']
        
        # 上段: 元画像 + クロップ範囲
        ax_top = axes[0, i]
        ax_top.imshow(frame, cmap='gray')
        ax_top.set_title(f'Crop Ratio: {ratio:.1f}\nLips: {enlarge:.2f}x', 
                        fontsize=11, fontweight='bold')
        
        # クロップ範囲を表示
        from matplotlib.patches import Rectangle
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                        fill=False, edgecolor='red', linewidth=2)
        ax_top.add_patch(rect)
        ax_top.plot(lip_x, lip_y, 'r*', markersize=15)
        ax_top.axis('off')
        
        # 下段: クロップ後
        ax_bottom = axes[1, i]
        ax_bottom.imshow(cropped, cmap='gray')
        ax_bottom.set_title(f'After Crop\n{H}x{W}', fontsize=10)
        ax_bottom.axis('off')
    
    plt.suptitle(f'Crop Ratio Comparison (Frame {frame_index})', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ 可視化を保存: {output_file}")
    print("="*70)
    print("比較完了")
    print("="*70)
    print(f"\n推奨:")
    print(f"  - 唇がはっきり見える")
    print(f"  - 背景が少ない")
    print(f"  - でも唇が切れていない")
    print(f"\nこの条件を満たすクロップ比率を選んでください")
    print("="*70)


# ========================================
# 複数サンプル対応版
# ========================================

def visualize_multiple_samples(pt_files, crop_ratios, detection_method='auto',
                               frame_index=0, output_file='crop_comparison_multi.png'):
    """
    複数サンプルで比較
    
    Args:
        pt_files: ptファイルのリスト
        crop_ratios: クロップ比率のリスト
        detection_method: 検出方法
        frame_index: フレームインデックス
        output_file: 出力ファイル
    """
    print("="*70)
    print("複数サンプル クロップ比較")
    print("="*70)
    print(f"サンプル数: {len(pt_files)}")
    print(f"クロップ比率: {crop_ratios}")
    print("="*70)
    
    num_samples = len(pt_files)
    num_ratios = len(crop_ratios)
    
    # 図を作成（サンプル数 × (元画像 + クロップ比率数)）
    fig, axes = plt.subplots(num_samples, num_ratios + 1, 
                            figsize=(4*(num_ratios+1), 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for sample_idx, pt_file in enumerate(pt_files):
        print(f"\nサンプル {sample_idx+1}/{num_samples}: {pt_file}")
        
        try:
            # データ読み込み
            data = torch.load(pt_file)
            
            if isinstance(data, dict) and 'video' in data:
                video = data['video']
            elif isinstance(data, torch.Tensor):
                video = data
            else:
                print(f"  ⚠️ スキップ: 非対応形式")
                continue
            
            # ★ 形状調整
            if video.dim() == 4 and video.shape[1] == 1:
                video = video.squeeze(1)
            elif video.dim() == 4 and video.shape[0] == 1:
                video = video.squeeze(0)
            elif video.dim() == 5:
                if video.shape[0] == 1:
                    video = video.squeeze(0)
                if video.dim() == 4 and video.shape[1] == 1:
                    video = video.squeeze(1)
            
            # フレーム取得
            if video.dim() == 3:
                frame = video[frame_index].numpy()
            elif video.dim() == 4:
                frame = video[frame_index, 0].numpy()
            else:
                print(f"  ⚠️ スキップ: 非対応形状 {video.shape}")
                continue
            
            H, W = frame.shape
            
            # 元画像（左端）
            ax = axes[sample_idx, 0]
            ax.imshow(frame, cmap='gray')
            ax.set_title(f'Sample {sample_idx+1}\nOriginal {H}x{W}', fontsize=10)
            ax.axis('off')
            
            # 各クロップ比率
            for ratio_idx, ratio in enumerate(crop_ratios, 1):
                cropped, (lip_y, lip_x), (y1, y2, x1, x2) = crop_single_frame(
                    frame, ratio, detection_method
                )
                
                ax = axes[sample_idx, ratio_idx]
                ax.imshow(cropped, cmap='gray')
                
                if sample_idx == 0:
                    ax.set_title(f'Ratio {ratio:.1f}\n({1/ratio:.2f}x)', fontsize=10)
                else:
                    ax.set_title('')
                
                ax.axis('off')
            
            print(f"  ✓ 処理完了")
            
        except Exception as e:
            print(f"  ❌ エラー: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    plt.suptitle('Crop Ratio Comparison (Multiple Samples)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ 保存: {output_file}")
    print("="*70)


# ========================================
# メイン処理
# ========================================

def main():
    """メイン処理"""
    
    # ファイル存在チェック
    if not Path(SAMPLE_PT_FILE).exists():
        print(f"❌ サンプルファイルが見つかりません: {SAMPLE_PT_FILE}")
        print("\nスクリプト内の SAMPLE_PT_FILE を正しいパスに変更してください")
        print("\n使用例:")
        print("  SAMPLE_PT_FILE = 'data/videos/sample001.pt'")
        return
    
    # 単一サンプル比較
    visualize_crop_comparison(
        pt_file=SAMPLE_PT_FILE,
        crop_ratios=CROP_RATIOS,
        detection_method=LIP_DETECTION_METHOD,
        frame_index=SHOW_FRAME_INDEX,
        output_file=OUTPUT_IMAGE
    )
    
    print(f"\n画像を確認してください: {OUTPUT_IMAGE}")
    print("\n推奨されるクロップ比率を選んで、")
    print("lip_center_crop_preprocessing.py の CROP_RATIO に設定してください")


# ========================================
# 複数サンプル比較用の便利関数
# ========================================

def compare_multiple_samples():
    """複数サンプルを比較する場合はこの関数を使用"""
    
    # 複数のサンプルファイル
    sample_files = [
        "data/videos/sample001.pt",
        "data/videos/sample002.pt",
        "data/videos/sample003.pt",
    ]
    
    # 試すクロップ比率
    crop_ratios = [0.8, 0.6, 0.5, 0.4]
    
    visualize_multiple_samples(
        pt_files=sample_files,
        crop_ratios=crop_ratios,
        detection_method='auto',
        frame_index=0,
        output_file='crop_comparison_multi.png'
    )


if __name__ == "__main__":
    main()