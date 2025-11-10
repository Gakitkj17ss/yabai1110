#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音声文字起こし処理（デバッグ版）
- コード内で入力を指定
- 結果をターミナルに表示
- JSONファイルも出力
"""

import os
import json
import re
from pathlib import Path
from tqdm import tqdm
import logging
import argparse
import whisper
import torch
import MeCab

class AudioTranscriber:
    """音声文字起こしクラス"""
    
    def __init__(self, config):
        self.whisper_model_name = config['whisper_model']
        self.batch_size = config.get('batch_size', 1)
        self.output_format = config.get('output_format', 'json')
        self.debug_mode = config.get('debug_mode', False)
        
        self.setup_logging()
        self.load_whisper_model()
        self.setup_katakana_converter()
    
    def setup_logging(self):
        """ログ設定"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_whisper_model(self):
        """Whisperモデル読み込み"""
        try:
            self.logger.info(f"Whisperモデル読み込み中: {self.whisper_model_name}")
            self.whisper_model = whisper.load_model(self.whisper_model_name)
            self.logger.info("✅ Whisperモデル読み込み完了")
        except Exception as e:
            self.logger.error(f"❌ Whisperモデル読み込みエラー: {e}")
            self.whisper_model = None
    
    def setup_katakana_converter(self):
        """カタカナ変換設定（MeCab + 読み仮名変換）"""
        try:
            dict_paths = [
                '-d /var/lib/mecab/dic/debian',
                '-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd',
                ''
            ]
            
            self.mecab = None
            for dict_path in dict_paths:
                try:
                    self.mecab = MeCab.Tagger(dict_path)
                    self.logger.info(f"✅ MeCab初期化完了: {dict_path if dict_path else 'default'}")
                    break
                except:
                    continue
            
            if self.mecab is None:
                self.mecab = MeCab.Tagger()
                self.logger.info("✅ MeCab初期化完了: default")
                
        except Exception as e:
            self.logger.error(f"❌ MeCab初期化エラー: {e}")
            self.mecab = None
        
        self.hiragana_to_katakana = str.maketrans(
            'あいうえおかきくけこがきぐげござしすせそざじずぜぞたちつてとだぢづでどなにぬねのはひふへほばびぶべぼぱぴぷぺぽまみむめもやゆよらりるれろわをんゃゅょっー',
            'アイウエオカキクケコガキグゲゴザシスセソザジズゼゾタチツテトダヂヅデドナニヌネノハヒフヘホバビブベボパピプペポマミムメモヤユヨラリルレロワヲンャュョッー'
        )
    
    def word_to_katakana(self, word: str) -> str:
        """単語をカタカナに変換（MeCab使用）"""
        if not word or self.mecab is None:
            return word.translate(self.hiragana_to_katakana)
        
        try:
            node = self.mecab.parseToNode(word)
            katakana_word = ""
            
            while node:
                surface = node.surface
                features = node.feature.split(',')
                
                if surface == "を":
                    katakana_word += "ヲ"
                elif surface == "は" and len(features) > 0 and features[0] == "助詞":
                    katakana_word += "ハ"
                elif len(features) > 7 and features[7] != '*':
                    katakana_word += features[7]
                elif len(features) > 6 and features[6] != '*':
                    katakana_word += features[6]
                else:
                    katakana_word += surface.translate(self.hiragana_to_katakana)
                
                node = node.next
            
            return katakana_word
            
        except Exception as e:
            if self.debug_mode:
                self.logger.warning(f"単語変換エラー {word}: {e}")
            return word.translate(self.hiragana_to_katakana)
    
    def convert_to_katakana_with_mecab(self, text: str) -> str:
        """MeCabを使って漢字を読み仮名に変換してからカタカナ化"""
        if not text or self.mecab is None:
            return text.translate(self.hiragana_to_katakana)
        
        try:
            words = text.split()
            katakana_words = [self.word_to_katakana(word) for word in words]
            return ''.join(katakana_words)
                
        except Exception as e:
            if self.debug_mode:
                self.logger.warning(f"MeCab変換エラー: {e}")
            return text.translate(self.hiragana_to_katakana)
    
    def clean_and_convert_to_katakana(self, text: str) -> str:
        """テキストをクリーニングしてカタカナに変換（MeCab使用）"""
        if not text:
            return ""
        
        text = text.strip()
        text = re.sub(r'[\n\t\r]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[。、！？．，!?.,]', '', text)
        
        text = self.convert_to_katakana_with_mecab(text)
        text = text.translate(self.hiragana_to_katakana)
        
        text = text.translate(str.maketrans(
            '０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ',
            '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        ))
        
        text = re.sub(r'[^\u30A0-\u30FFー・]', '', text)
        text = re.sub(r'[ーー]+', 'ー', text)
        text = re.sub(r'[・]', '', text)
        
        return text.strip()
    
    def transcribe_single_audio(self, audio_path: str) -> dict:
        """単一音声ファイルの文字起こし（詳細情報付き）"""
        if self.whisper_model is None:
            self.logger.error("❌ Whisperモデルが利用できません")
            return {
                'raw_text': '',
                'clean_text': '',
                'error': 'Whisperモデルが利用できません'
            }
        
        try:
            result = self.whisper_model.transcribe(
                audio_path,
                language='ja',
                verbose=False,
                fp16=torch.cuda.is_available()
            )
            
            raw_text = result.get('text', '').strip()
            clean_text = self.clean_and_convert_to_katakana(raw_text)
            
            return {
                'raw_text': raw_text,
                'clean_text': clean_text,
                'text_length': len(clean_text),
                'error': None
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️  文字起こしエラー {audio_path}: {e}")
            return {
                'raw_text': '',
                'clean_text': '',
                'text_length': 0,
                'error': str(e)
            }
    
    def process_audio_directory(self, audio_dir: str, output_file: str = None):
        """音声ディレクトリを一括処理（デバッグモード）"""
        audio_path = Path(audio_dir)
        
        # パスチェック
        print(f"\n🔍 パス確認: {audio_path}")
        print(f"  絶対パス: {audio_path.absolute()}")
        print(f"  存在確認: {audio_path.exists()}")
        print(f"  ディレクトリ: {audio_path.is_dir() if audio_path.exists() else 'N/A'}")
        print(f"  ファイル: {audio_path.is_file() if audio_path.exists() else 'N/A'}")
        
        # ディレクトリかファイルかを判定
        if audio_path.is_file():
            # 単一ファイルの場合
            print(f"⚠️  単一ファイルが指定されました: {audio_path.name}")
            audio_files = [audio_path]
        elif audio_path.is_dir():
            # ディレクトリの場合
            audio_files = list(audio_path.glob("*.wav"))
            audio_files.extend(audio_path.glob("*.mp3"))
            audio_files.extend(audio_path.glob("*.m4a"))
            audio_files.sort()
        else:
            # 存在しない、または親ディレクトリで検索
            self.logger.error(f"❌ パスが見つかりません: {audio_dir}")
            
            # 親ディレクトリで検索してみる
            parent = audio_path.parent
            if parent.exists():
                print(f"\n🔍 親ディレクトリを探索: {parent}")
                audio_files = list(parent.rglob("*.wav"))
                audio_files.extend(parent.rglob("*.mp3"))
                audio_files.extend(parent.rglob("*.m4a"))
                audio_files.sort()
                
                if audio_files:
                    print(f"✅ {len(audio_files)}個の音声ファイルを発見")
                else:
                    print("❌ 音声ファイルが見つかりません")
                    return
            else:
                return
        
        if not audio_files:
            self.logger.warning(f"⚠️  音声ファイルが見つかりません: {audio_dir}")
            return
        
        print("\n" + "=" * 80)
        print(f"📂 音声ファイル発見: {len(audio_files)}個")
        print("=" * 80)
        
        # 最初の数ファイルを表示
        print("\n📄 処理対象ファイル（最初の5件）:")
        for i, f in enumerate(audio_files[:5], 1):
            print(f"  {i}. {f.name}")
        if len(audio_files) > 5:
            print(f"  ... 他 {len(audio_files) - 5}件")
        
        print("\n" + "=" * 80)
        print("🔄 文字起こし開始")
        print("=" * 80)
        
        transcription_results = {}
        successful_count = 0
        error_count = 0
        
        for i, audio_file in enumerate(tqdm(audio_files, desc="処理中"), 1):
            file_key = audio_file.stem
            result = self.transcribe_single_audio(str(audio_file))
            
            transcription_results[file_key] = {
                'audio_file': str(audio_file),
                'transcribed_text': result['clean_text'],
                'text_length': result['text_length'],
                'has_text': len(result['clean_text']) > 0
            }
            
            if result['error']:
                error_count += 1
            elif result['clean_text']:
                successful_count += 1
            
            # 最初の10件を詳細表示
            if i <= 10:
                print(f"\n{'='*80}")
                print(f"📝 ファイル {i}: {audio_file.name}")
                print(f"{'='*80}")
                print(f"元のテキスト: {result['raw_text']}")
                print(f"カタカナ変換: {result['clean_text']}")
                print(f"文字数: {result['text_length']}")
                if result['error']:
                    print(f"エラー: {result['error']}")
        
        # 統計情報
        print("\n" + "=" * 80)
        print("📊 処理結果サマリー")
        print("=" * 80)
        print(f"総ファイル数: {len(audio_files)}")
        print(f"✅ 成功: {successful_count}")
        print(f"❌ エラー: {error_count}")
        print(f"⚪ 空（無音声）: {len(audio_files) - successful_count - error_count}")
        print(f"成功率: {successful_count/len(audio_files)*100:.1f}%")
        
        text_lengths = [r['text_length'] for r in transcription_results.values() if r['has_text']]
        if text_lengths:
            print(f"\n📏 テキスト長統計:")
            print(f"  平均: {sum(text_lengths)/len(text_lengths):.1f}文字")
            print(f"  最大: {max(text_lengths)}文字")
            print(f"  最小: {min(text_lengths)}文字")
        
        # 全結果表示
        print("\n" + "=" * 80)
        print("📋 全文字起こし結果")
        print("=" * 80)
        for key, result in transcription_results.items():
            status = "✅" if result['has_text'] else "⚪"
            print(f"{status} {key}: {result['transcribed_text']} ({result['text_length']}文字)")
        
        # JSONファイル出力
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(transcription_results, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 JSONファイル保存: {output_path}")
        
        print("=" * 80)
    
    def check_mecab_installation(self):
        """MeCabインストール状況確認"""
        print("\n" + "=" * 80)
        print("🔍 MeCab診断")
        print("=" * 80)
        
        if self.mecab is None:
            print("❌ MeCabが利用できません")
            print("\nインストール方法:")
            print("  Ubuntu/Debian: sudo apt-get install mecab mecab-ipadic-utf8 python3-mecab")
            print("  pip install mecab-python3")
            return False
        else:
            print("✅ MeCab利用可能")
            try:
                test_result = self.mecab.parse("テスト")
                print(f"テスト結果: {test_result.strip()}")
                return True
            except Exception as e:
                print(f"⚠️  MeCab動作に問題: {e}")
                return False
    
    def test_mecab_conversion(self):
        """MeCab変換のテスト"""
        test_texts = [
            "こんにちは",
            "今日は良い天気ですね",
            "機械学習について話しましょう",
            "東京駅に行きます",
            "ありがとうございました"
        ]
        
        print("\n" + "=" * 80)
        print("🧪 MeCab変換テスト")
        print("=" * 80)
        for text in test_texts:
            converted = self.clean_and_convert_to_katakana(text)
            print(f"'{text}' -> '{converted}'")
        print("=" * 80)

def main():
    """メイン実行関数"""
    
    # ========================================
    # 🔧 デバッグ設定（ここを編集）
    # ========================================
    
    # 入力音声ディレクトリ
    INPUT_AUDIO_DIR = '/home/bv20049/dataset/npz/zundadata/ROHAN4600_split/ROHAN4600_0002.wav'
    
    # 出力JSONファイル（Noneの場合はターミナル表示のみ）
    OUTPUT_JSON_FILE = None
    
    # Whisperモデル
    WHISPER_MODEL = 'medium'  # tiny, base, small, medium, large
    
    # デバッグモード（詳細ログ表示）
    DEBUG_MODE = True
    
    # ========================================
    
    print("\n" + "=" * 80)
    print("🎤 音声文字起こしデバッグモード")
    print("=" * 80)
    print(f"入力: {INPUT_AUDIO_DIR}")
    print(f"出力: {OUTPUT_JSON_FILE if OUTPUT_JSON_FILE else 'ターミナルのみ'}")
    print(f"モデル: {WHISPER_MODEL}")
    print(f"GPU: {'✅ 使用可能' if torch.cuda.is_available() else '❌ 使用不可'}")
    print("=" * 80)
    
    config = {
        'whisper_model': WHISPER_MODEL,
        'output_format': 'json',
        'debug_mode': DEBUG_MODE
    }
    
    transcriber = AudioTranscriber(config)
    
    # MeCab確認
    if not transcriber.check_mecab_installation():
        print("\n⚠️  MeCabが正しく動作しない可能性があります")
        print("それでも継続しますか？ (y/N): ", end='')
        response = input()
        if response.lower() != 'y':
            print("中断しました")
            return
    
    # MeCab変換テスト
    transcriber.test_mecab_conversion()
    
    # 実行確認
    print("\n文字起こし処理を開始しますか？ (y/N): ", end='')
    response = input()
    if response.lower() != 'y':
        print("中断しました")
        return
    
    # 処理実行
    transcriber.process_audio_directory(INPUT_AUDIO_DIR, OUTPUT_JSON_FILE)
    
    print("\n✅ デバッグ完了")

if __name__ == "__main__":
    main()
#/home/bv20049/dataset/npz/zundadata/ROHAN4600_split/ROHAN4600_0060.wav
#python3 /home/bv20049/dataset/npz/zundadata/test_transcription.py --audio /home/bv20049/dataset/npz/zundadata/ROHAN4600_split/ROHAN4600_0002.wav
#python3 /home/bv20049/dataset/npz/zundadata/test_transcription.py --mecab-only
#python3 /home/bv20049/dataset/npz/zundadata/test_transcription.py --check-mecab --input dummy --output dummy
#python3 /home/bv20049/dataset/npz/zundadata/test_transcription.py --test-mecab --input dummy --output dummy
#python3 /home/bv20049/dataset/npz/zundadata/test_transcription.py --input /home/bv20049/dataset/npz/zundadata/ROHAN4600_split/ROHAN4600_0002.wav --output /home/bv20049/dataset/npz/zundadata/ROHAN4600_splittranscriptions.json --model base