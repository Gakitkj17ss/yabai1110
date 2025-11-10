#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attention可視化モジュール（修正版）

【修正内容】
- ConsonantOnlyPhonemeEncoderの互換性修正
- encode_text → encode_phonemes への変更
- text_to_phonemes メソッドの追加
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
import os

# ===== PER helper (Levenshtein-based) =====
def _levenshtein_sdi(ref, hyp):
    """
    ref, hyp: list[str]（音素列）
    return (S, D, I)
    """
    n, m = len(ref), len(hyp)
    dp = [[0]*(m+1) for _ in range(n+1)]
    bt = [[0]*(m+1) for _ in range(n+1)]  # 0:diag, 1:up(del), 2:left(ins)
    for i in range(1, n+1):
        dp[i][0] = i; bt[i][0] = 1
    for j in range(1, m+1):
        dp[0][j] = j; bt[0][j] = 2
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            a = dp[i-1][j-1] + cost
            b = dp[i-1][j] + 1
            c = dp[i][j-1] + 1
            if a <= b and a <= c:
                dp[i][j] = a; bt[i][j] = 0
            elif b <= c:
                dp[i][j] = b; bt[i][j] = 1
            else:
                dp[i][j] = c; bt[i][j] = 2
    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        code = bt[i][j]
        if i > 0 and j > 0 and code == 0:
            if ref[i-1] != hyp[j-1]:
                S += 1
            i -= 1; j -= 1
        elif i > 0 and (j == 0 or code == 1):
            D += 1; i -= 1
        else:
            I += 1; j -= 1
    return S, D, I

def _sequence_per_percent(ref, hyp):
    """PER[%] = (S+D+I)/N * 100"""
    S, D, I = _levenshtein_sdi(ref, hyp)
    N = max(1, len(ref))
    return 100.0 * (S + D + I) / N
# ===== end helper =====


class AttentionVisualizer:
    """Attention重み可視化クラス"""
    
    def __init__(self, model, phoneme_encoder, device='cuda'):
        """
        Args:
            model: 訓練済みモデル
            phoneme_encoder: 音素エンコーダー
            device: 使用デバイス
        """
        self.model = model
        self.phoneme_encoder = phoneme_encoder
        self.device = device
        
        # モデルを評価モードに
        self.model.eval()
    
    def _text_to_phonemes(self, text) -> List[str]:
        """
        テキストを音素列に変換
        
        Args:
            text: 入力テキスト（文字列またはリスト）
        
        Returns:
            音素のリスト
        """
        # 0. すでに音素列（リスト）の場合はそのまま返す
        if isinstance(text, list):
            return text
        
        # phoneme_encoderの種類に応じて処理を分岐
        
        # 1. encode_textメソッドがある場合
        if hasattr(self.phoneme_encoder, 'encode_text'):
            return self.phoneme_encoder.encode_text(text)
        
        # 2. text_to_phonemesメソッドがある場合（文字列のみ受け付ける）
        if hasattr(self.phoneme_encoder, 'text_to_phonemes'):
            if isinstance(text, str):
                return self.phoneme_encoder.text_to_phonemes(text)
        
        # 3. デフォルト: 文字を分割して音素として扱う
        if isinstance(text, str):
            print(f"Warning: phoneme_encoderに音素変換メソッドがありません。文字列を分割します: {text}")
            return list(text)
        
        # 4. その他の型の場合はエラー
        raise TypeError(f"text must be str or list, got {type(text)}")
    
    def _phonemes_to_ids(self, phonemes: List[str]) -> List[int]:
        """
        音素列をIDに変換
        
        Args:
            phonemes: 音素のリスト
        
        Returns:
            音素IDのリスト
        """
        # encode_phonemesメソッドがある場合
        if hasattr(self.phoneme_encoder, 'encode_phonemes'):
            return self.phoneme_encoder.encode_phonemes(phonemes)
        
        # phoneme_to_idがある場合
        if hasattr(self.phoneme_encoder, 'phoneme_to_id'):
            return [self.phoneme_encoder.phoneme_to_id.get(p, 0) for p in phonemes]
        
        # デフォルト: id2phonemeの逆引き
        if hasattr(self.phoneme_encoder, 'id2phoneme'):
            phoneme_to_id = {v: k for k, v in self.phoneme_encoder.id2phoneme.items()}
            return [phoneme_to_id.get(p, 0) for p in phonemes]
        
        raise AttributeError("phoneme_encoderに音素→ID変換メソッドがありません")
    
    def visualize_attention_with_evaluation(
        self,
        video: torch.Tensor,
        text,  # str or List[str]
        save_path: Optional[str] = None,
        evaluator=None
    ) -> Dict:
        """
        Attention重みを可視化し、予測を評価
        
        Args:
            video: 動画テンソル [T, C, H, W]
            text: 正解テキスト（文字列）または音素列（リスト）
            save_path: 保存先パス
            evaluator: 評価器（オプション）
        
        Returns:
            結果辞書
        """
        self.model.eval()
        
        with torch.no_grad():
            # バッチ次元を追加
            if video.dim() == 4:
                video = video.unsqueeze(0)  # [1, T, C, H, W]
            
            video = video.to(self.device)
            
            # 順伝播（Attention weightsを取得）
            # モデルがreturn_attentionをサポートしているか確認
            try:
                outputs = self.model(video, return_attention=True)
                
                # outputsがタプルの場合（outputs, attention_weights）
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    outputs, attention_weights = outputs
                    attention_weights = attention_weights.cpu().numpy()
                    print(f"  ✓ Attention weights from return_attention=True: {attention_weights.shape}")
                else:
                    # return_attentionをサポートしていない場合
                    attention_weights = None
                    print(f"  ⚠ Model returned unexpected format with return_attention=True")
                    
            except TypeError:
                # return_attentionパラメータをサポートしていない
                print(f"  ⚠ Model does not support return_attention parameter")
                outputs = self.model(video)
                attention_weights = None
            
            # 追加のAttention weights取得方法
            if attention_weights is None:
                # 方法1: model.attention_weights属性
                if hasattr(self.model, 'attention_weights') and self.model.attention_weights is not None:
                    attention_weights = self.model.attention_weights.cpu().numpy()
                    print(f"  ✓ Attention weights from model.attention_weights: {attention_weights.shape}")
                
                # 方法2: outputs辞書にattention_weightsが含まれている場合
                elif isinstance(outputs, dict) and 'attention_weights' in outputs:
                    attention_weights = outputs['attention_weights'].cpu().numpy()
                    print(f"  ✓ Attention weights from outputs dict: {attention_weights.shape}")
                    outputs = outputs['logits']  # logitsを取り出す
                
                # 方法3: model.get_attention_weights()メソッド
                elif hasattr(self.model, 'get_attention_weights'):
                    attention_weights = self.model.get_attention_weights().cpu().numpy()
                    print(f"  ✓ Attention weights from get_attention_weights(): {attention_weights.shape}")
                
                # 方法4: 最後のforward passの結果を保存している場合
                elif hasattr(self.model, 'last_attention_weights'):
                    attention_weights = self.model.last_attention_weights.cpu().numpy()
                    print(f"  ✓ Attention weights from last_attention_weights: {attention_weights.shape}")
                
                else:
                    print(f"  ⚠ Attention weights not found in model")
                    print(f"     Available model methods: {[m for m in dir(self.model) if not m.startswith('_') and 'forward' in m.lower()]}")
            
            # outputsがタプルの場合
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # 予測を取得
            log_probs = torch.nn.functional.log_softmax(outputs, dim=-1)
            log_probs = log_probs.permute(1, 0, 2)  # [T, 1, num_classes]
            
            _, max_indices = torch.max(log_probs, dim=2)  # [T, 1]
            max_indices = max_indices.squeeze(1).cpu().numpy()  # [T]
            
            # CTC collapse
            pred_ids = []
            prev_id = None
            for idx in max_indices:
                if idx != self.phoneme_encoder.blank_id and idx != prev_id:
                    pred_ids.append(int(idx))
                prev_id = idx
            
            # デコード
            pred_phonemes = self.phoneme_encoder.decode_phonemes(pred_ids)
            
            # 正解音素を取得
            target_phonemes = self._text_to_phonemes(text)
            
            # 評価
            is_correct = (pred_phonemes == target_phonemes)
            
            result = {
                'predicted': pred_phonemes,
                'target': target_phonemes,
                'is_correct': is_correct,
                'attention_weights': attention_weights
            }
            
            # 評価器がある場合は詳細評価
            if evaluator is not None:
                try:
                    eval_result = evaluator.evaluate_single(pred_phonemes, target_phonemes)
                    result.update(eval_result)
                except Exception as e:
                    print(f"⚠ 評価エラー: {e}")
            
            # 可視化
            if save_path:
                self._plot_attention(
                    video=video.squeeze(0).cpu().numpy(),
                    attention_weights=attention_weights,
                    pred_phonemes=pred_phonemes,
                    target_phonemes=target_phonemes,
                    is_correct=is_correct,
                    save_path=save_path
                )
        
        return result
    
    def _plot_attention(
        self,
        video: np.ndarray,
        attention_weights: Optional[np.ndarray],
        pred_phonemes: List[str],
        target_phonemes: List[str],
        is_correct: bool,
        save_path: str
    ):
        """
        Attention重みをプロット（改善版）
        
        Args:
            video: 動画データ [T, C, H, W]
            attention_weights: Attention重み [1, T] or [T] or None
            pred_phonemes: 予測音素
            target_phonemes: 正解音素
            is_correct: 正解かどうか
            save_path: 保存先パス
        """
        num_frames = video.shape[0]
        
        # Attention weightsの確認とデバッグ情報
        has_attention = attention_weights is not None
        if has_attention:
            weights = attention_weights.squeeze()
            print(f"  📊 Attention統計:")
            print(f"     Shape: {attention_weights.shape} → {weights.shape}")
            print(f"     Min: {weights.min():.6f}, Max: {weights.max():.6f}")
            print(f"     Mean: {weights.mean():.6f}, Std: {weights.std():.6f}")
            print(f"     Range: {weights.max() - weights.min():.6f}")
            
            # ピーク情報
            peak_idx = np.argmax(weights)
            print(f"     Peak: Frame {peak_idx} (weight={weights[peak_idx]:.6f})")
            
            # 注目度の分布
            top_5_indices = np.argsort(weights)[-5:][::-1]
            print(f"     Top 5 frames: {top_5_indices.tolist()}")
            
            # 範囲が狭い場合は警告
            weight_range = weights.max() - weights.min()
            if weight_range < 0.1:
                print(f"     ⚠️  注意: Attention重みの範囲が狭い ({weight_range:.6f})")
                print(f"         → Attentionがほぼ均一で、選択的に注目できていない可能性")
                print(f"         → Temperature を下げる (例: 0.5) か、softmax に変更を検討")
        else:
            print(f"  ⚠ Attention weights not available")
        
        # 図の作成
        if has_attention:
            fig = plt.figure(figsize=(14, 10))
            gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
            ax_video = fig.add_subplot(gs[0])
            ax_attention = fig.add_subplot(gs[1])
            ax_heatmap = fig.add_subplot(gs[2])
        else:
            fig, ax_video = plt.subplots(1, 1, figsize=(14, 4))
            ax_attention = None
            ax_heatmap = None
        
        # フレームのサムネイル表示
        num_display = min(10, num_frames)
        indices = np.linspace(0, num_frames - 1, num_display, dtype=int)
        
        thumbnails = []
        for idx in indices:
            frame = video[idx]
            # チャンネルが最初の場合は最後に移動 [C, H, W] -> [H, W, C]
            if frame.shape[0] in [1, 3]:
                frame = np.transpose(frame, (1, 2, 0))
            # グレースケールの場合
            if frame.shape[-1] == 1:
                frame = frame.squeeze(-1)
            thumbnails.append(frame)
        
        # サムネイル結合
        thumbnail_strip = np.concatenate(thumbnails, axis=1)
        
        # 正規化
        if thumbnail_strip.max() > 1.0:
            thumbnail_strip = thumbnail_strip / 255.0
        
        ax_video.imshow(thumbnail_strip, cmap='gray' if len(thumbnail_strip.shape) == 2 else None)
        ax_video.set_title(
            f"{'✓ 正解' if is_correct else '✗ 不正解'}\n"
            f"予測: {' '.join(pred_phonemes)}\n"
            f"正解: {' '.join(target_phonemes)}",
            fontsize=12,
            fontweight='bold',
            color='green' if is_correct else 'red'
        )
        ax_video.axis('off')
        
        # Attention重みのプロット
        if ax_attention is not None and has_attention:
            weights = attention_weights.squeeze()  # [T]
            
            # 折れ線グラフ
            frames_idx = np.arange(len(weights))
            ax_attention.plot(frames_idx, weights, linewidth=2.5, color='#2E86DE', marker='o', markersize=4)
            ax_attention.fill_between(frames_idx, weights, alpha=0.3, color='#54A0FF')
            
            # ピーク位置に印
            peak_idx = np.argmax(weights)
            ax_attention.scatter([peak_idx], [weights[peak_idx]], 
                               color='red', s=100, zorder=5, marker='*', 
                               label=f'Peak at frame {peak_idx}')
            
            ax_attention.set_xlabel('Frame Index', fontsize=11, fontweight='bold')
            ax_attention.set_ylabel('Attention Weight', fontsize=11, fontweight='bold')
            ax_attention.set_title('Attention Weights over Time (Line Plot)', fontsize=11, fontweight='bold')
            ax_attention.grid(True, alpha=0.3, linestyle='--')
            ax_attention.legend(loc='upper right')
            ax_attention.set_xlim(-0.5, len(weights) - 0.5)
            
            # Y軸の範囲を調整
            y_min, y_max = weights.min(), weights.max()
            y_range = y_max - y_min
            ax_attention.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        
        # Attention重みのヒートマップ
        if ax_heatmap is not None and has_attention:
            weights = attention_weights.squeeze()  # [T]
            weights_2d = weights.reshape(1, -1)  # [1, T]
            
            im = ax_heatmap.imshow(weights_2d, cmap='hot', aspect='auto', interpolation='nearest')
            ax_heatmap.set_xlabel('Frame Index', fontsize=11, fontweight='bold')
            ax_heatmap.set_ylabel('Attention', fontsize=11, fontweight='bold')
            ax_heatmap.set_title('Attention Weights Heatmap', fontsize=11, fontweight='bold')
            ax_heatmap.set_yticks([])
            
            # カラーバー
            cbar = plt.colorbar(im, ax=ax_heatmap, orientation='horizontal', pad=0.1, fraction=0.05)
            cbar.set_label('Weight', fontsize=10)
        
        plt.tight_layout()
        
        # 保存
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {save_path}")


def visualize_attention_with_samples(
    model,
    data_loader,
    phoneme_encoder,
    device='cuda',
    num_samples=5,
    save_dir='results/attention_visualization',
    evaluator=None
) -> Dict:
    """
    複数サンプルでAttention可視化（不正解サンプルに PER[%] を付与し、ワーストTOP10を表示/保存）
    """

    # --- PER helpers (ローカル定義：外部依存なし) ---
    def _levenshtein_sdi(ref, hyp):
        """ref/hyp: list[str] -> (S,D,I)"""
        n, m = len(ref), len(hyp)
        dp = [[0]*(m+1) for _ in range(n+1)]
        bt = [[0]*(m+1) for _ in range(n+1)]  # 0:diag, 1:up(del), 2:left(ins)
        for i in range(1, n+1):
            dp[i][0] = i; bt[i][0] = 1
        for j in range(1, m+1):
            dp[0][j] = j; bt[0][j] = 2
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = 0 if ref[i-1] == hyp[j-1] else 1
                a = dp[i-1][j-1] + cost
                b = dp[i-1][j] + 1
                c = dp[i][j-1] + 1
                if a <= b and a <= c:
                    dp[i][j] = a; bt[i][j] = 0
                elif b <= c:
                    dp[i][j] = b; bt[i][j] = 1
                else:
                    dp[i][j] = c; bt[i][j] = 2
        i, j = n, m
        S = D = I = 0
        while i > 0 or j > 0:
            code = bt[i][j]
            if i > 0 and j > 0 and code == 0:
                if ref[i-1] != hyp[j-1]:
                    S += 1
                i -= 1; j -= 1
            elif i > 0 and (j == 0 or code == 1):
                D += 1; i -= 1
            else:
                I += 1; j -= 1
        return S, D, I

    def _per_percent(ref, hyp):
        """PER[%] = (S+D+I)/len(ref)*100"""
        S, D, I = _levenshtein_sdi(ref, hyp)
        N = max(1, len(ref))
        return 100.0 * (S + D + I) / N, (S, D, I)
    # --- end helpers ---

    os.makedirs(save_dir, exist_ok=True)
    visualizer = AttentionVisualizer(model, phoneme_encoder, device)

    # 評価器
    if evaluator is None:
        try:
            from matrics_undefined import CTCAwareEvaluator
            evaluator = CTCAwareEvaluator()
        except ImportError:
            print("⚠ CTCAwareEvaluator not found. Using simple evaluation.")
            evaluator = None

    correct_samples = []
    incorrect_samples = []

    total_samples = 0
    correct_count = 0

    # サンプル収集
    for batch in data_loader:
        videos = batch['video']
        targets = batch['target']
        target_lengths = batch['target_length']

        batch_size = videos.size(0)
        target_offset = 0

        for i in range(batch_size):
            if total_samples >= num_samples * 2:  # 正解・不正解それぞれ num_samples 目標
                break

            video = videos[i]
            target_len = int(target_lengths[i].item())
            target_ids = targets[target_offset:target_offset + target_len].cpu().numpy()
            target_phonemes = phoneme_encoder.decode_phonemes(target_ids)

            save_path = os.path.join(save_dir, f'sample_{total_samples:03d}.png')
            result = visualizer.visualize_attention_with_evaluation(
                video=video,
                text=target_phonemes,  # 音素列を直接渡す
                save_path=save_path,
                evaluator=evaluator
            )

            # PER 計算
            per, (S, D, I) = _per_percent(result['target'], result['predicted'])

            # 結果保存
            sample_info = {
                'sample_id': total_samples,
                'predicted': result['predicted'],
                'target': result['target'],
                'is_correct': result['is_correct'],
                'save_path': save_path,
                'per': round(per, 2),
                'S': int(S), 'D': int(D), 'I': int(I),
            }

            if result['is_correct']:
                correct_samples.append(sample_info)
                correct_count += 1
            else:
                incorrect_samples.append(sample_info)

            total_samples += 1
            target_offset += target_len

        if total_samples >= num_samples * 2:
            break

    # 不正解を PER 降順（悪い順）でソート
    incorrect_samples.sort(key=lambda s: -s['per'])

    # サマリー
    accuracy = correct_count / total_samples if total_samples > 0 else 0.0
    result = {
        'total_samples': total_samples,
        'correct_count': correct_count,
        'accuracy': accuracy,
        'correct_samples': correct_samples,
        'incorrect_samples': incorrect_samples,
        'save_dir': save_dir
    }

    # ===== レポート出力 =====
    print(f"\n{'='*70}")
    print(f"Attention可視化 + サンプル評価レポート")
    print(f"{'='*70}")

    print(f"\n【全体統計】")
    print(f"  総サンプル数: {total_samples}")
    print(f"  正解数: {correct_count}")
    print(f"  不正解数: {total_samples - correct_count}")
    print(f"  精度: {accuracy*100:.1f}%")
    print(f"  保存先: {save_dir}")

    # 正解サンプル（最大5件）
    if correct_samples:
        print(f"\n【正解サンプル】 ({min(5, len(correct_samples))}件)")
        for i, s in enumerate(correct_samples[:5], 1):
            pred_str = ' '.join(s['predicted'])
            tgt_str  = ' '.join(s['target'])
            print(f"  {i}. ✓ PER={s['per']:.2f}%  予測={pred_str}, 正解={tgt_str}")
            print(f"     ファイル: {os.path.basename(s['save_path'])}")

    # 不正解サンプル（TOP10）
    if incorrect_samples:
        topn = min(10, len(incorrect_samples))
        print(f"\n【不正解サンプル】 (TOP {topn})")
        for i, s in enumerate(incorrect_samples[:topn], 1):
            pred_str = ' '.join(s['predicted'])
            tgt_str  = ' '.join(s['target'])
            print(f"  {i}. ✗ PER={s['per']:.2f}%  予測={pred_str}, 正解={tgt_str}")
            print(f"     ファイル: {os.path.basename(s['save_path'])}")
            # エラー分析（集合差）
            missing = set(s['target']) - set(s['predicted'])
            extra   = set(s['predicted']) - set(s['target'])
            if missing:
                print(f"     欠落音素: {missing}")
            if extra:
                print(f"     余分音素: {extra}")

    # Attention統計（案内）
    if total_samples > 0:
        print(f"\n【Attention統計】")
        print(f"  可視化画像を確認してください: {save_dir}")
        print(f"  注目度が高いフレームを確認できます")
    print(f"{'='*70}\n")

    # ===== JSON 保存（PERとSDIを含む）=====
    import json
    to_json = {
        'summary': {
            'total_samples': total_samples,
            'correct_count': correct_count,
            'incorrect_count': total_samples - correct_count,
            'accuracy': accuracy
        },
        'correct_samples': [
            {
                'sample_id': s['sample_id'],
                'predicted': s['predicted'],
                'target': s['target'],
                'file': os.path.basename(s['save_path']),
                'per': s['per'], 'S': s['S'], 'D': s['D'], 'I': s['I'],
            } for s in correct_samples
        ],
        'incorrect_samples': [
            {
                'sample_id': s['sample_id'],
                'predicted': s['predicted'],
                'target': s['target'],
                'file': os.path.basename(s['save_path']),
                'per': s['per'], 'S': s['S'], 'D': s['D'], 'I': s['I'],
                'missing_phonemes': list(set(s['target']) - set(s['predicted'])),
                'extra_phonemes': list(set(s['predicted']) - set(s['target'])),
            } for s in incorrect_samples
        ]
    }
    json_path = os.path.join(save_dir, 'evaluation_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(to_json, f, indent=2, ensure_ascii=False)
    print(f"✓ 評価結果をJSON保存: {json_path}\n")

    return result


if __name__ == "__main__":
    print("Attention可視化モジュール（修正版）")