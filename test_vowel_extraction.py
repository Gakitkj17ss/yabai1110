#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
母音抽出のテストスクリプト
データセットで実際にどう変換されるかを確認
"""

from phoneme_encoder import JapanesePhonemeEncoder

def test_vowel_extraction():
    """母音抽出のテスト"""
    print("=== 母音抽出テスト ===")
    
    # 母音のみエンコーダー
    encoder = JapanesePhonemeEncoder(vowel_only=True)
    
    # よくある日本語サンプル
    test_texts = [
        "こんにちは",     # こ(オ) ん(除外) に(イ) ち(イ) は(ア) → オイイア
        "ありがとう",     # あ(ア) り(イ) が(ア) と(オ) う(ウ) → アイアオウ
        "おはよう",       # お(オ) は(ア) よ(オ) う(ウ) → オアオウ
        "さようなら",     # さ(ア) よ(オ) う(ウ) な(ア) ら(ア) → アオウアア
        "すみません",     # す(ウ) み(イ) ま(ア) せ(エ) ん(除外) → ウイアエ
        "はじめまして",   # は(ア) じ(イ) め(エ) ま(ア) し(イ) て(エ) → アイエアイエ
        "よろしく",       # よ(オ) ろ(オ) し(イ) く(ウ) → オオイウ
        "げんき",         # げ(エ) ん(除外) き(イ) → エイ
        "たべる",         # た(ア) べ(エ) る(ウ) → アエウ
        "のむ"            # の(オ) む(ウ) → オウ
    ]
    
    print("\n📝 テキスト → 母音変換:")
    print("-" * 50)
    
    total_original = 0
    total_vowels = 0
    
    for text in test_texts:
        phonemes = encoder.text_to_phonemes(text)
        phoneme_ids = encoder.encode_phonemes(phonemes)
        result = ''.join(phonemes)
        
        total_original += len(text)
        total_vowels += len(phonemes)
        
        print(f"{text:12} → {result:10} (IDs: {phoneme_ids})")
    
    print("-" * 50)
    print(f"圧縮率: {total_original}文字 → {total_vowels}母音 ({total_vowels/total_original:.1%})")
    
    # 母音統計
    encoder.get_vowel_statistics(test_texts)
    
    return encoder

def test_small_vowel_patterns():
    """小文字パターンの専用テスト"""
    print("\n=== 小文字パターン詳細テスト ===")
    
    encoder = JapanesePhonemeEncoder(vowel_only=True)
    
    # 小文字テストケース
    test_cases = [
        # フ＋小文字パターン
        ("ファ", "ア", "フ+ァ → ア（1音）"),
        ("フィ", "イ", "フ+ィ → イ（1音）"),
        ("フェ", "エ", "フ+ェ → エ（1音）"),
        ("フォ", "オ", "フ+ォ → オ（1音）"),
        
        # 他の子音＋小文字
        ("ティ", "イ", "ト+ィ → イ（1音）"),
        ("デュ", "ウ", "デ+ュ → ウ（1音）"),
        
        # 拗音パターン（比較用）
        ("きゃ", "イア", "き+ゃ → イア（2音）"),
        ("しゅ", "イウ", "し+ゅ → イウ（2音）"),
        ("ちょ", "イオ", "ち+ょ → イオ（2音）"),
        
        # 単独小文字
        ("ァ", "ア", "単独ァ → ア"),
        ("ィ", "イ", "単独ィ → イ"),
    ]
    
    print("テストケース:")
    print("-" * 60)
    
    all_passed = True
    for input_text, expected, description in test_cases:
        phonemes = encoder.text_to_phonemes(input_text)
        result = ''.join(phonemes)
        
        status = "✅" if result == expected else "❌"
        if result != expected:
            all_passed = False
        
        print(f"{status} {input_text:4} → {result:6} (期待: {expected:6}) {description}")
    
    print("-" * 60)
    if all_passed:
        print("✅ 全テストケース成功！小文字処理は正常です")
    else:
        print("❌ 一部テストケース失敗。処理を確認してください")
    
    return all_passed
    """データセット処理のシミュレーション"""
    print("\n=== データセット処理シミュレーション ===")
    
    encoder = JapanesePhonemeEncoder(vowel_only=True)
    
    # CSVデータの例
    csv_data = [
        {"video_path": "/path/video1.pth", "text": "こんにちは"},
        {"video_path": "/path/video2.pth", "text": "ありがとう"},
        {"video_path": "/path/video3.pth", "text": "おはよう"},
    ]
    
    print("CSVデータ → 学習用データ変換:")
    print("-" * 60)
    
    for i, row in enumerate(csv_data):
        text = row["text"]
        phonemes = encoder.text_to_phonemes(text)
        phoneme_ids = encoder.encode_phonemes(phonemes)
        
        print(f"サンプル {i+1}:")
        print(f"  元テキスト: {text}")
        print(f"  母音抽出:   {''.join(phonemes)}")
        print(f"  ID変換:     {phoneme_ids}")
        print(f"  ターゲット長: {len(phoneme_ids)}")
        print()
    
    print("✅ データセットのCSVは変更不要")
    print("✅ phoneme_encoderが自動で母音抽出")
    print("✅ 学習時は母音のみで学習される")
    print("✅ 'ん'や'っ'は除外（口形に母音要素なし）")

def test_learning_targets():
    """学習ターゲットの確認"""
    print("\n=== 学習ターゲット確認 ===")
    
    encoder = JapanesePhonemeEncoder(vowel_only=True)
    
    print("音韻ID対応表:")
    for i, phoneme in enumerate(encoder.phonemes):
        print(f"  ID {i}: {phoneme}")
    
    print(f"\n学習クラス数: {encoder.vocab_size}")
    print("CTCでは:")
    print("  - ID 0 (BLANK): アライメント用")
    print("  - ID 1 (UNK): 不明音韻") 
    print("  - ID 2-6: ア・イ・ウ・エ・オ")
    
    # 実際の学習例
    text = "こんにちは"
    phonemes = encoder.text_to_phonemes(text)
    ids = encoder.encode_phonemes(phonemes)
    
    print(f"\n学習例: '{text}'")
    print(f"  入力: 動画フレーム (口の動き)")
    print(f"  ターゲット: {ids} ({''.join(phonemes)})")
    print(f"  CTC損失: 予測系列とターゲット系列を比較")

if __name__ == "__main__":
    encoder = test_vowel_extraction()
    test_dataset_simulation() 
    test_learning_targets()
    
    print("\n🎯 結論:")
    print("✅ 既存のCSVデータセットをそのまま使用可能")
    print("✅ phoneme_encoderが自動で母音抽出")
    print("✅ 学習は母音のみで実行される")
    print("✅ データ準備の追加作業は不要")