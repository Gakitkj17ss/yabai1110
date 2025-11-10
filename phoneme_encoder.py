#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日本語音韻エンコーダー（母音のみ・段階的学習版）
"""

class JapanesePhonemeEncoder:
    def __init__(self, vowel_only=True):
        """
        vowel_only: Trueなら母音のみ、Falseなら全音韻
        """
        self.vowel_only = vowel_only
        
        if vowel_only:
            # 段階1: 母音+ン（読唇術の基本）
            self.phonemes = [
                'BLANK',  # ID=0: CTCのblank token
                'UNK',    # ID=1: 不明
                'ア',     # ID=2: 口を大きく開く
                'イ',     # ID=3: 口を横に引く  
                'ウ',     # ID=4: 口をすぼめる
                'エ',     # ID=5: 口を中程度に開く
                'オ',     # ID=6: 口を丸くする
                'ン',     # ID=7: 口を閉じる（重要な音韻）
            ]
            print("🎯 母音+ンモード: ア・イ・ウ・エ・オ・ン（6音韻）")
            print("口形の特徴:")
            print("  ア: 口を大きく縦に開く")
            print("  イ: 口を横に引く（笑顔）")
            print("  ウ: 口をすぼめる（キス）")
            print("  エ: 口を中程度に開く")
            print("  オ: 口を丸くする")
            print("  ン: 口を閉じる（鼻音）")
        else:
            # 段階2以降: 全音韻
            self.phonemes = [
                'BLANK', 'UNK',
                # 基本母音
                'ア', 'イ', 'ウ', 'エ', 'オ',
                # 小文字母音
                'ァ', 'ィ', 'ゥ', 'ェ', 'ォ',
                # 主要子音
                'カ', 'キ', 'ク', 'ケ', 'コ',
                'サ', 'シ', 'ス', 'セ', 'ソ',
                'タ', 'チ', 'ツ', 'テ', 'ト',
                'ナ', 'ニ', 'ヌ', 'ネ', 'ノ',
                'ハ', 'ヒ', 'フ', 'ヘ', 'ホ',
                'マ', 'ミ', 'ム', 'メ', 'モ',
                'ヤ', 'ユ', 'ヨ',
                'ャ', 'ュ', 'ョ',
                'ラ', 'リ', 'ル', 'レ', 'ロ',
                'ワ', 'ヲ', 'ン',
                # 濁音
                'ガ', 'ギ', 'グ', 'ゲ', 'ゴ',
                'ザ', 'ジ', 'ズ', 'ゼ', 'ゾ',
                'ダ', 'ヂ', 'ヅ', 'デ', 'ド',
                'バ', 'ビ', 'ブ', 'ベ', 'ボ',
                'パ', 'ピ', 'プ', 'ペ', 'ポ',
                # 特殊
                'ッ', 'ー'
            ]
            print(f"🎯 全音韻モード: {len(self.phonemes)}音韻")
        
        self.phoneme_to_id = {p: i for i, p in enumerate(self.phonemes)}
        self.id_to_phoneme = {i: p for i, p in enumerate(self.phonemes)}
        self.vocab_size = len(self.phonemes)
        
        print(f"音韻語彙数: {self.vocab_size} (BLANK含む)")
    
    def text_to_phonemes(self, text):
        """テキストを音韻リストに変換（デバッグ改良版）"""
        if self.vowel_only:
            phonemes = []
            i = 0
            
            # デバッグ用
            # print(f"処理テキスト: '{text}'")
            
            while i < len(text):
                char = text[i]
                # print(f"  位置{i}: '{char}'")
                
                # 直接的な母音・ン
                if char in ['ア', 'イ', 'ウ', 'エ', 'オ', 'ん', 'ン']:
                    if char == 'ん':
                        phonemes.append('ン')
                    else:
                        phonemes.append(char)
                    # print(f"    → 直接母音: {char}")
                    i += 1
                
                # 子音＋小文字の組み合わせ（ファ、フィ、フェ、フォ等）
                elif i < len(text) - 1 and text[i + 1] in ['ァ', 'ィ', 'ゥ', 'ェ', 'ォ']:
                    small_vowel_map = {'ァ': 'ア', 'ィ': 'イ', 'ゥ': 'ウ', 'ェ': 'エ', 'ォ': 'オ'}
                    phonemes.append(small_vowel_map[text[i + 1]])
                    # print(f"    → 子音+小文字: {char}+{text[i + 1]} = {small_vowel_map[text[i + 1]]}")
                    i += 2
                
                # 子音＋拗音の組み合わせ（きゃ、しゅ、ちょ等）
                elif i < len(text) - 1 and text[i + 1] in ['ャ', 'ュ', 'ョ']:
                    # 子音の母音部分を抽出
                    consonant_vowel = self._extract_vowel_from_consonant(char)
                    if consonant_vowel:
                        phonemes.append(consonant_vowel)
                    
                    # 拗音の母音部分も追加
                    small_vowel_map = {'ャ': 'ア', 'ュ': 'ウ', 'ョ': 'オ'}
                    phonemes.append(small_vowel_map[text[i + 1]])
                    # print(f"    → 拗音: {char}+{text[i + 1]} = {consonant_vowel}+{small_vowel_map[text[i + 1]]}")
                    i += 2
                
                # 単独の小文字（前に子音がない場合）
                elif char in ['ァ', 'ィ', 'ゥ', 'ェ', 'ォ']:
                    vowel_map = {'ァ': 'ア', 'ィ': 'イ', 'ゥ': 'ウ', 'ェ': 'エ', 'ォ': 'オ'}
                    phonemes.append(vowel_map[char])
                    # print(f"    → 単独小文字: {char} = {vowel_map[char]}")
                    i += 1
                
                # 通常の子音から母音抽出
                else:
                    vowel = self._extract_vowel_from_consonant(char)
                    if vowel:
                        phonemes.append(vowel)
                        # print(f"    → 子音から母音: {char} = {vowel}")
                    else:
                        # print(f"    → 無視: {char}")
                        pass
                    i += 1
            
            # print(f"  結果: {phonemes}")
            return phonemes if phonemes else ['UNK']
        else:
            # 通常処理
            phonemes = []
            for char in text:
                if char in self.phoneme_to_id:
                    phonemes.append(char)
                else:
                    phonemes.append('UNK')
            return phonemes
    
    def _extract_vowel_from_consonant(self, consonant):
        """子音から母音部分を抽出（改良版）"""
        vowel_map = {
            # か行
            'カ': 'ア', 'キ': 'イ', 'ク': 'ウ', 'ケ': 'エ', 'コ': 'オ',
            # さ行  
            'サ': 'ア', 'シ': 'イ', 'ス': 'ウ', 'セ': 'エ', 'ソ': 'オ',
            # た行
            'タ': 'ア', 'チ': 'イ', 'ツ': 'ウ', 'テ': 'エ', 'ト': 'オ',
            # な行
            'ナ': 'ア', 'ニ': 'イ', 'ヌ': 'ウ', 'ネ': 'エ', 'ノ': 'オ',
            # は行
            'ハ': 'ア', 'ヒ': 'イ', 'フ': 'ウ', 'ヘ': 'エ', 'ホ': 'オ',
            # ま行
            'マ': 'ア', 'ミ': 'イ', 'ム': 'ウ', 'メ': 'エ', 'モ': 'オ',
            # や行
            'ヤ': 'ア', 'ユ': 'ウ', 'ヨ': 'オ',
            # ら行
            'ラ': 'ア', 'リ': 'イ', 'ル': 'ウ', 'レ': 'エ', 'ロ': 'オ',
            # わ行
            'ワ': 'ア', 'ヲ': 'オ', 'ン': 'ン',
            # 濁音
            'ガ': 'ア', 'ギ': 'イ', 'グ': 'ウ', 'ゲ': 'エ', 'ゴ': 'オ',
            'ザ': 'ア', 'ジ': 'イ', 'ズ': 'ウ', 'ゼ': 'エ', 'ゾ': 'オ',
            'ダ': 'ア', 'ヂ': 'イ', 'ヅ': 'ウ', 'デ': 'エ', 'ド': 'オ',
            'バ': 'ア', 'ビ': 'イ', 'ブ': 'ウ', 'ベ': 'エ', 'ボ': 'オ',
            'パ': 'ア', 'ピ': 'イ', 'プ': 'ウ', 'ペ': 'エ', 'ポ': 'オ',
            # 特殊音韻
            'ッ': None,  # 促音は除外
            'ー': None,  # 長音は除外
        }
        return vowel_map.get(consonant, None)
    
    def encode_phonemes(self, phonemes):
        """音韻リストをIDリストに変換"""
        return [self.phoneme_to_id.get(p, self.phoneme_to_id['UNK']) for p in phonemes]
    
    def decode_phonemes(self, ids):
        """IDリストを音韻リストに変換"""
        return [self.id_to_phoneme.get(id, 'UNK') for id in ids]
    
    def encode_text(self, text):
        """テキスト → 音韻 → IDの一括変換"""
        phonemes = self.text_to_phonemes(text)
        return self.encode_phonemes(phonemes)
    
    def decode_ids(self, ids):
        """ID → 音韻 → テキストの一括変換"""
        phonemes = self.decode_phonemes(ids)
        return ''.join(phonemes)
    
    def print_sample_conversions(self):
        """サンプル変換例を表示"""
        if self.vowel_only:
            samples = ["こんにちは", "ありがとう", "おはよう", "さようなら"]
            print("\n📝 母音抽出例:")
            for text in samples:
                phonemes = self.text_to_phonemes(text)
                result = ''.join(phonemes)
                print(f"  {text} → {result}")
        
    def get_vowel_statistics(self, texts):
        """母音の出現統計（ン追加版）"""
        vowel_counts = {'ア': 0, 'イ': 0, 'ウ': 0, 'エ': 0, 'オ': 0, 'ン': 0}
        total = 0
        
        for text in texts:
            phonemes = self.text_to_phonemes(text)
            for p in phonemes:
                if p in vowel_counts:
                    vowel_counts[p] += 1
                    total += 1
        
        print("\n📊 母音+ン出現統計:")
        for vowel, count in vowel_counts.items():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {vowel}: {count}回 ({percentage:.1f}%)")
        
        return vowel_counts