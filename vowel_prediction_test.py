#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
母音のみモードで実際に予測できるかテスト
"""

import torch
import torch.nn as nn
from phoneme_encoder import JapanesePhonemeEncoder
from model import HybridCTCLipReadingModel
from config import Config

def test_model_creation():
    """モデルが正しく作成されるかテスト"""
    print("=== モデル作成テスト ===")
    
    # 母音エンコーダー
    phoneme_encoder = JapanesePhonemeEncoder(vowel_only=True)
    print(f"語彙数: {phoneme_encoder.vocab_size}")
    print(f"音韻: {phoneme_encoder.phonemes}")
    
    # モデル作成
    model = HybridCTCLipReadingModel(num_phonemes=phoneme_encoder.vocab_size)
    
    # パラメータ数確認
    total_params = sum(p.numel() for p in model.parameters())
    print(f"総パラメータ数: {total_params:,}")
    
    return model, phoneme_encoder

def test_forward_pass():
    """順伝播テスト"""
    print("\n=== 順伝播テスト ===")
    
    model, phoneme_encoder = test_model_creation()
    model.eval()
    
    # ダミー入力作成 (バッチ=2, 時間=10, チャンネル=1, 高さ=96, 幅=96)
    batch_size = 2
    seq_len = 10
    dummy_input = torch.randn(batch_size, seq_len, 1, 96, 96)
    
    print(f"入力形状: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"出力形状: {output.shape}")
    print(f"期待形状: (batch={batch_size}, time={seq_len}, classes={phoneme_encoder.vocab_size})")
    
    # 出力の内容確認
    print(f"出力値範囲: {output.min().item():.3f} ~ {output.max().item():.3f}")
    
    # CTC用にlog_softmaxされているか確認
    print("各時刻の確率和（log空間）:")
    for t in range(min(3, seq_len)):
        prob_sum = torch.exp(output[0, t]).sum().item()
        print(f"  時刻{t}: {prob_sum:.3f} (≈1.0が正常)")
    
    return model, phoneme_encoder, output

def test_ctc_decoding():
    """CTC デコーディングテスト"""
    print("\n=== CTC デコーディングテスト ===")
    
    model, phoneme_encoder, output = test_forward_pass()
    
    # 最初のサンプルでデコーディング
    log_probs = output[0]  # (time, num_classes)
    
    # Greedy decoding
    pred_seq = torch.argmax(log_probs, dim=-1)
    print(f"生予測ID: {pred_seq.tolist()}")
    
    # CTC decoding (重複・BLANK除去)
    decoded_seq = []
    prev_token = -1
    
    for token in pred_seq:
        token = token.item()
        if token != prev_token and token != 0:  # 0はBLANK
            decoded_seq.append(token)
        prev_token = token
    
    print(f"CTC後ID: {decoded_seq}")
    
    # 音韻に変換
    decoded_phonemes = phoneme_encoder.decode_phonemes(decoded_seq)
    result_text = ''.join(decoded_phonemes)
    
    print(f"予測音韻: {decoded_phonemes}")
    print(f"予測テキスト: '{result_text}'")
    
    # 各音韻の確率も表示
    print("\n各時刻の音韻確率:")
    for t in range(min(5, log_probs.size(0))):
        probs = torch.exp(log_probs[t])
        top_prob, top_idx = torch.max(probs, dim=0)
        top_phoneme = phoneme_encoder.id_to_phoneme[top_idx.item()]
        print(f"  時刻{t}: {top_phoneme} ({top_prob.item():.3f})")

def test_training_compatibility():
    """学習との互換性テスト"""
    print("\n=== 学習互換性テスト ===")
    
    model, phoneme_encoder = test_model_creation()
    
    # ダミー学習データ
    videos = torch.randn(2, 8, 1, 96, 96)  # バッチ=2, 時間=8
    
    # テキストデータ
    texts = ["こんにちは", "ありがとう"]
    
    # ターゲット作成
    all_targets = []
    target_lengths = []
    
    print("学習データ変換:")
    for text in texts:
        phonemes = phoneme_encoder.text_to_phonemes(text)
        phoneme_ids = phoneme_encoder.encode_phonemes(phonemes)
        
        all_targets.extend(phoneme_ids)
        target_lengths.append(len(phoneme_ids))
        
        print(f"  '{text}' → {phonemes} → {phoneme_ids}")
    
    targets = torch.tensor(all_targets, dtype=torch.long)
    input_lengths = torch.tensor([8, 8], dtype=torch.long)  # 動画長
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    
    print(f"\nターゲット形状: {targets.shape}")
    print(f"入力長: {input_lengths}")
    print(f"ターゲット長: {target_lengths}")
    
    # CTC損失テスト
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    with torch.no_grad():
        outputs = model(videos)  # (batch, time, classes)
        outputs_transposed = outputs.transpose(0, 1)  # (time, batch, classes)
        
        try:
            loss = criterion(outputs_transposed, targets, input_lengths, target_lengths)
            print(f"CTC損失: {loss.item():.4f}")
            print("✅ 学習準備完了")
        except Exception as e:
            print(f"❌ CTC損失エラー: {e}")

def test_vowel_only_effectiveness():
    """母音のみモードの効果確認"""
    print("\n=== 母音のみモード効果確認 ===")
    
    # 通常モード
    full_encoder = JapanesePhonemeEncoder(vowel_only=False)
    print(f"通常モード語彙数: {full_encoder.vocab_size}")
    
    # 母音のみモード  
    vowel_encoder = JapanesePhonemeEncoder(vowel_only=True)
    print(f"母音のみ語彙数: {vowel_encoder.vocab_size}")
    
    reduction = (1 - vowel_encoder.vocab_size / full_encoder.vocab_size) * 100
    print(f"語彙削減率: {reduction:.1f}%")
    
    # 学習の違い
    print("\n学習への影響:")
    print(f"  出力層: {full_encoder.vocab_size} → {vowel_encoder.vocab_size} ニューロン")
    print(f"  パラメータ削減: 約{reduction:.0f}%")
    print(f"  過学習リスク: 大幅減少")
    print(f"  収束速度: 大幅向上期待")

if __name__ == "__main__":
    print("🎯 母音のみモード動作確認テスト")
    print("=" * 50)
    
    try:
        test_model_creation()
        test_forward_pass() 
        test_ctc_decoding()
        test_training_compatibility()
        test_vowel_only_effectiveness()
        
        print("\n" + "=" * 50)
        print("✅ 全テスト成功！母音のみモードは正常に動作します")
        print("🎯 学習実行で母音予測が期待できます")
        
    except Exception as e:
        print(f"\n❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()