#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
モデル各コンポーネント診断ツール
CNNの特徴抽出、LSTMの系列処理、CTCの音韻予測を個別に診断
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from simple_model import SimpleLipReadingModel
from phoneme_encoder import JapanesePhonemeEncoder
from config import Config

class ComponentDiagnostics:
    """各コンポーネントの診断クラス"""
    
    def __init__(self, model, phoneme_encoder):
        self.model = model
        self.phoneme_encoder = phoneme_encoder
        
    def diagnose_cnn_features(self, video_data, visualize=True):
        """CNN特徴抽出の診断"""
        print("🔍 CNN特徴抽出診断")
        
        self.model.eval()
        with torch.no_grad():
            # CNN特徴のみ抽出
            cnn_features = self.model.cnn(video_data)  # (batch, time, feature_dim)
            
            batch_size, time_steps, feature_dim = cnn_features.shape
            
            print(f"  入力形状: {video_data.shape}")
            print(f"  CNN出力形状: {cnn_features.shape}")
            print(f"  特徴範囲: {cnn_features.min():.3f} ~ {cnn_features.max():.3f}")
            print(f"  特徴平均: {cnn_features.mean():.3f}")
            print(f"  特徴分散: {cnn_features.var():.3f}")
            
            # 時間軸での特徴変化
            temporal_variance = cnn_features.var(dim=1).mean()  # 時間軸の分散
            print(f"  時間変化: {temporal_variance:.3f} (高いほど動的)")
            
            # 特徴の相関（最初のフレームと他の比較）
            if time_steps > 1:
                first_frame = cnn_features[:, 0:1, :]  # (batch, 1, feature_dim)
                correlations = []
                for t in range(1, min(10, time_steps)):
                    frame_t = cnn_features[:, t:t+1, :]
                    corr = F.cosine_similarity(first_frame, frame_t, dim=-1).mean()
                    correlations.append(corr.item())
                
                avg_correlation = np.mean(correlations)
                print(f"  フレーム相関: {avg_correlation:.3f} (低いほど変化大)")
            
            if visualize and time_steps >= 10:
                # 特徴の可視化（最初の10フレーム、最初の16次元）
                plt.figure(figsize=(12, 8))
                
                plt.subplot(2, 2, 1)
                features_to_plot = cnn_features[0, :10, :16].cpu().numpy()
                plt.imshow(features_to_plot.T, aspect='auto', cmap='viridis')
                plt.title('CNN Features (first 16 dims, 10 frames)')
                plt.xlabel('Time')
                plt.ylabel('Feature Dimension')
                plt.colorbar()
                
                plt.subplot(2, 2, 2)
                feature_norms = torch.norm(cnn_features[0], dim=-1).cpu().numpy()
                plt.plot(feature_norms)
                plt.title('Feature Magnitude over Time')
                plt.xlabel('Time')
                plt.ylabel('L2 Norm')
                
                plt.subplot(2, 2, 3)
                plt.hist(cnn_features[0].flatten().cpu().numpy(), bins=50, alpha=0.7)
                plt.title('Feature Value Distribution')
                plt.xlabel('Feature Value')
                plt.ylabel('Count')
                
                plt.subplot(2, 2, 4)
                if len(correlations) > 0:
                    plt.plot(correlations)
                    plt.title('Frame-to-Frame Correlation')
                    plt.xlabel('Frame Offset')
                    plt.ylabel('Cosine Similarity')
                
                plt.tight_layout()
                plt.savefig('cnn_diagnosis.png', dpi=150)
                plt.show()
            
            return {
                'feature_shape': cnn_features.shape,
                'feature_range': (cnn_features.min().item(), cnn_features.max().item()),
                'feature_mean': cnn_features.mean().item(),
                'feature_var': cnn_features.var().item(),
                'temporal_variance': temporal_variance.item(),
                'avg_correlation': avg_correlation if time_steps > 1 else 1.0
            }
    
    def diagnose_lstm_processing(self, video_data):
        """LSTM系列処理の診断"""
        print("🔍 LSTM系列処理診断")
        
        self.model.eval()
        with torch.no_grad():
            # CNN → LSTM
            cnn_features = self.model.cnn(video_data)
            lstm_features = self.model.rnn(cnn_features)
            
            print(f"  CNN → LSTM: {cnn_features.shape} → {lstm_features.shape}")
            
            # LSTM出力の分析
            print(f"  LSTM出力範囲: {lstm_features.min():.3f} ~ {lstm_features.max():.3f}")
            print(f"  LSTM出力平均: {lstm_features.mean():.3f}")
            print(f"  LSTM出力分散: {lstm_features.var():.3f}")
            
            # 系列の滑らかさ（隣接フレーム間の差）
            if lstm_features.size(1) > 1:
                frame_diffs = torch.diff(lstm_features, dim=1)  # (batch, time-1, feature)
                avg_diff = frame_diffs.abs().mean()
                print(f"  系列滑らかさ: {avg_diff:.3f} (低いほど滑らか)")
            
            # 双方向性の確認（最初と最後のフレーム）
            if lstm_features.size(1) >= 10:
                first_frames = lstm_features[:, :5, :].mean(dim=1)  # 最初の5フレーム平均
                last_frames = lstm_features[:, -5:, :].mean(dim=1)  # 最後の5フレーム平均
                bidirectional_similarity = F.cosine_similarity(first_frames, last_frames, dim=-1).mean()
                print(f"  双方向情報統合: {bidirectional_similarity:.3f}")
            
            return {
                'lstm_shape': lstm_features.shape,
                'lstm_range': (lstm_features.min().item(), lstm_features.max().item()),
                'lstm_mean': lstm_features.mean().item(),
                'lstm_var': lstm_features.var().item(),
                'sequence_smoothness': avg_diff.item() if lstm_features.size(1) > 1 else 0.0
            }
    
    def diagnose_ctc_output(self, video_data, target_text=None):
        """CTC出力層の診断"""
        print("🔍 CTC出力層診断")
        
        self.model.eval()
        with torch.no_grad():
            # 完全な前向き計算
            outputs = self.model(video_data)  # (batch, time, num_classes)
            
            batch_size, time_steps, num_classes = outputs.shape
            print(f"  CTC出力形状: {outputs.shape}")
            
            # Log probabilities → probabilities
            probs = torch.exp(outputs)
            
            # 各クラスの平均確率
            avg_class_probs = probs.mean(dim=(0, 1))
            print(f"  各クラス平均確率:")
            for i, prob in enumerate(avg_class_probs):
                phoneme = self.phoneme_encoder.id_to_phoneme.get(i, f'ID{i}')
                print(f"    {phoneme}: {prob:.3f}")
            
            # BLANK確率の分析
            blank_prob = avg_class_probs[0].item()
            non_blank_prob = avg_class_probs[1:].sum().item()
            print(f"  BLANK vs 非BLANK: {blank_prob:.3f} vs {non_blank_prob:.3f}")
            
            # エントロピー分析
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # (batch, time)
            avg_entropy = entropy.mean()
            print(f"  平均エントロピー: {avg_entropy:.3f} (高いほど不確実)")
            
            # 最も確信度の高いクラス
            max_classes = torch.argmax(probs, dim=-1)  # (batch, time)
            class_counts = torch.bincount(max_classes.flatten(), minlength=num_classes)
            most_frequent_class = torch.argmax(class_counts).item()
            most_frequent_phoneme = self.phoneme_encoder.id_to_phoneme.get(most_frequent_class, f'ID{most_frequent_class}')
            print(f"  最頻出予測: {most_frequent_phoneme} ({class_counts[most_frequent_class].item()}回)")
            
            # CTC decoding結果
            pred_sequence = torch.argmax(outputs[0], dim=-1).cpu().numpy()
            decoded_sequence = []
            prev_token = -1
            for token in pred_sequence:
                if token != prev_token and token != 0:
                    decoded_sequence.append(token)
                prev_token = token
            
            pred_phonemes = self.phoneme_encoder.decode_phonemes(decoded_sequence)
            pred_text = ''.join(pred_phonemes)
            print(f"  CTC予測結果: '{pred_text}'")
            
            if target_text:
                target_phonemes = self.phoneme_encoder.text_to_phonemes(target_text)
                target_text_converted = ''.join(target_phonemes)
                print(f"  正解テキスト: '{target_text}' → '{target_text_converted}'")
            
            return {
                'ctc_shape': outputs.shape,
                'blank_prob': blank_prob,
                'non_blank_prob': non_blank_prob,
                'avg_entropy': avg_entropy.item(),
                'predicted_text': pred_text,
                'class_probabilities': avg_class_probs.cpu().numpy().tolist()
            }
    
    def full_diagnosis(self, video_data, target_text=None):
        """全コンポーネントの診断"""
        print("="*60)
        print("🔬 フルモデル診断")
        print("="*60)
        
        cnn_results = self.diagnose_cnn_features(video_data, visualize=True)
        lstm_results = self.diagnose_lstm_processing(video_data)
        ctc_results = self.diagnose_ctc_output(video_data, target_text)
        
        # 問題の特定
        print("\n🎯 問題診断:")
        
        # CNN問題
        if cnn_results['temporal_variance'] < 0.01:
            print("  ⚠️  CNN: 時間変化が少なすぎる（特徴抽出が不十分）")
        if cnn_results['avg_correlation'] > 0.95:
            print("  ⚠️  CNN: フレーム間の変化が少なすぎる")
        
        # LSTM問題  
        if lstm_results['sequence_smoothness'] > 1.0:
            print("  ⚠️  LSTM: 系列が不安定（過度な変化）")
        if lstm_results['lstm_var'] < 0.01:
            print("  ⚠️  LSTM: 出力の分散が小さすぎる")
        
        # CTC問題
        if ctc_results['blank_prob'] > 0.5:
            print("  🚨 CTC: BLANK偏重問題！")
        if ctc_results['avg_entropy'] < 0.5:
            print("  ⚠️  CTC: 過度に確信的（多様性不足）")
        if ctc_results['avg_entropy'] > 2.0:
            print("  ⚠️  CTC: 過度に不確実（学習不足）")
        
        # 推奨対策
        print("\n💡 推奨対策:")
        if ctc_results['blank_prob'] > 0.5:
            print("  1. CTCバイアスをさらに調整")
            print("  2. BLANK penaltyを強化")
            print("  3. 学習率を上げる")
        if cnn_results['temporal_variance'] < 0.01:
            print("  4. CNN学習率を個別に上げる")
            print("  5. データ拡張を追加")
        
        return {
            'cnn': cnn_results,
            'lstm': lstm_results, 
            'ctc': ctc_results
        }

def run_diagnosis_on_sample():
    """サンプルデータで診断実行"""
    # 音韻エンコーダー
    phoneme_encoder = JapanesePhonemeEncoder(vowel_only=True)
    
    # モデル読み込み
    model = SimpleLipReadingModel(phoneme_encoder.vocab_size)
    model = model.to(Config.DEVICE)
    
    # ダミーデータで診断
    dummy_video = torch.randn(1, 30, 1, 96, 96).to(Config.DEVICE)
    dummy_text = "こんにちは"
    
    # 診断実行
    diagnostics = ComponentDiagnostics(model, phoneme_encoder)
    results = diagnostics.full_diagnosis(dummy_video, dummy_text)
    
    return results

if __name__ == "__main__":
    results = run_diagnosis_on_sample()
    print("\n診断完了！")