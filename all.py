#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
読唇術データセット完全自動処理パイプライン（音声分割機能追加版）
全ての処理をこのファイル1つで完結:
1. ペアCSVを読み込み
2. GPU 0で動画分割（2秒ウィンドウ、1秒スライド、FPS 25→20、PT形式、(4, 150, 1, 64, 64)）
3. GPU 1で音声分割（2秒ウィンドウ、1秒スライド）
4. GPU 1でWhisper文字起こし（カタカナ変換）
5. 最終CSVを生成（train/valid分割）
"""

import os
import subprocess
import time
import sys
import json
import re
import cv2
import numpy as np
from pathlib import Path
from threading import Thread
import logging
import argparse
import pandas as pd
from tqdm import tqdm

# PyTorch関連
try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorchがインストールされていません")

# Whisper関連（文字起こし用）
try:
    import whisper
    import MeCab
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️  Whisper/MeCabがインストールされていません")

# ========================================
# 動画分割クラス（統合）
# ========================================

class VideoSegmenter:
    """動画分割クラス"""
    
    def __init__(self, config):
        self.window_sec = config.get('window_sec', 2.0)  # 2秒ウィンドウ
        self.slide_sec = config.get('slide_sec', 1.0)   # 1秒スライド
        self.target_fps = config.get('target_fps', 15)   # 目標FPS
        self.target_size = config.get('target_size', (64, 64))  # リサイズサイズ
        
        self.setup_logging()
    
    def setup_logging(self):
        """ログ設定"""
        self.logger = logging.getLogger(__name__)
    
    def process_single_video(self, video_path: str, output_dir: str, video_name: str):
        """単一動画を処理"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                self.logger.error(f"動画を開けません: {video_path}")
                return []
            
            # 動画情報取得
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / original_fps
            
            # FPS変換の計算
            fps_ratio = self.target_fps / original_fps
            
            # ウィンドウとスライドのフレーム数
            window_frames = int(self.window_sec * self.target_fps)  # 2秒 * 20fps = 40フレーム
            slide_frames = int(self.slide_sec * self.target_fps)    # 1秒 * 20fps = 20フレーム
            
            segments = []
            segment_idx = 0
            
            # スライディングウィンドウで処理
            start_frame_target = 0
            
            while True:
                # 元の動画での開始フレーム位置
                start_frame_original = int(start_frame_target / fps_ratio)
                
                if start_frame_original >= total_frames:
                    break
                
                # フレーム収集
                frames = []
                
                for i in range(window_frames):
                    frame_idx_original = int((start_frame_target + i) / fps_ratio)
                    
                    if frame_idx_original >= total_frames:
                        break
                    
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_original)
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                    
                    # グレースケール変換
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # リサイズ
                    resized = cv2.resize(gray_frame, self.target_size)
                    
                    frames.append(resized)
                
                # フレーム数チェック
                if len(frames) < window_frames * 0.8:  # 80%未満なら破棄
                    break
                
                # 不足分をゼロパディング
                while len(frames) < window_frames:
                    frames.append(np.zeros(self.target_size, dtype=np.uint8))
                
                # テンソル変換: (T, H, W) → (B, T, C, H, W)
                # (40, 64, 64) → (1, 40, 1, 64, 64)
                frames_array = np.array(frames)  # (40, 64, 64)
                
                # (40, 64, 64) → (40, 1, 64, 64) チャンネル次元追加
                frames_array = frames_array[:, np.newaxis, :, :]  # (40, 1, 64, 64)
                
                # バッチ次元追加: (40, 1, 64, 64) → (1, 40, 1, 64, 64)
                frames_array = frames_array[np.newaxis, :, :, :, :]  # (1, 40, 1, 64, 64)
                
                # PyTorchテンソルに変換
                tensor = torch.from_numpy(frames_array).float()
                
                # 正規化 [0, 255] → [0, 1]
                tensor = tensor / 255.0
                
                # 保存
                output_path = Path(output_dir) / f"{video_name}_{segment_idx:04d}.pt"
                torch.save(tensor, output_path)
                
                segments.append({
                    'video_name': video_name,
                    'segment_id': segment_idx,
                    'start_time': start_frame_target / self.target_fps,
                    'tensor_path': str(output_path),
                    'tensor_shape': tuple(tensor.shape)
                })
                
                segment_idx += 1
                start_frame_target += slide_frames
            
            cap.release()
            
            return segments
            
        except Exception as e:
            self.logger.error(f"動画処理エラー {video_path}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []
    
    def process_from_csv(self, csv_path: str, output_dir: str):
        """CSVから動画を一括処理"""
        df = pd.read_csv(csv_path)
        
        if 'video_path' not in df.columns:
            self.logger.error("CSVに'video_path'列がありません")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_segments = []
        
        print(f"\n動画分割開始: {len(df)}個の動画")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="動画分割中"):
            video_path = row['video_path']
            video_name = row.get('name', Path(video_path).stem)
            
            if not Path(video_path).exists():
                self.logger.warning(f"動画が見つかりません: {video_path}")
                continue
            
            segments = self.process_single_video(video_path, output_dir, video_name)
            all_segments.extend(segments)
        
        # セグメント情報をCSVに保存
        if all_segments:
            segments_df = pd.DataFrame(all_segments)
            segments_csv = output_path / 'segments_info.csv'
            segments_df.to_csv(segments_csv, index=False)
            
            self.logger.info(f"動画分割完了: {len(all_segments)}セグメント生成")
            self.logger.info(f"セグメント情報: {segments_csv}")
        
        return all_segments

# ========================================
# 音声分割クラス（新規追加）
# ========================================

class AudioSegmenter:
    """音声分割クラス"""
    
    def __init__(self, config):
        self.window_sec = config.get('window_sec', 2.0)  # 2秒ウィンドウ
        self.slide_sec = config.get('slide_sec', 1.0)   # 1秒スライド
        self.target_sample_rate = config.get('target_sample_rate', 16000)  # 16kHz
        
        self.setup_logging()
    
    def setup_logging(self):
        """ログ設定"""
        self.logger = logging.getLogger(__name__)
    
    def process_single_audio(self, audio_path: str, output_dir: str, audio_name: str):
        """単一音声ファイルを処理"""
        try:
            # 音声ファイル読み込み
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # モノラル変換
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # リサンプリング
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.target_sample_rate
                )
                waveform = resampler(waveform)
            
            # 音声の長さ（秒）
            total_samples = waveform.shape[1]
            duration = total_samples / self.target_sample_rate
            
            # ウィンドウとスライドのサンプル数
            window_samples = int(self.window_sec * self.target_sample_rate)
            slide_samples = int(self.slide_sec * self.target_sample_rate)
            
            segments = []
            segment_idx = 0
            
            # スライディングウィンドウで処理
            start_sample = 0
            
            while start_sample < total_samples:
                end_sample = start_sample + window_samples
                
                # セグメント切り出し
                if end_sample <= total_samples:
                    segment_waveform = waveform[:, start_sample:end_sample]
                else:
                    # 最後のセグメント：ゼロパディング
                    segment_waveform = waveform[:, start_sample:]
                    padding_size = window_samples - segment_waveform.shape[1]
                    
                    if padding_size > 0:
                        padding = torch.zeros(1, padding_size)
                        segment_waveform = torch.cat([segment_waveform, padding], dim=1)
                
                # 80%未満の長さならスキップ
                actual_samples = min(end_sample, total_samples) - start_sample
                if actual_samples < window_samples * 0.8:
                    break
                
                # WAVファイルとして保存
                output_path = Path(output_dir) / f"{audio_name}_{segment_idx:04d}.wav"
                torchaudio.save(
                    str(output_path),
                    segment_waveform,
                    self.target_sample_rate
                )
                
                segments.append({
                    'audio_name': audio_name,
                    'segment_id': segment_idx,
                    'start_time': start_sample / self.target_sample_rate,
                    'end_time': min(end_sample, total_samples) / self.target_sample_rate,
                    'audio_path': str(output_path),
                    'duration': segment_waveform.shape[1] / self.target_sample_rate
                })
                
                segment_idx += 1
                start_sample += slide_samples
            
            return segments
            
        except Exception as e:
            self.logger.error(f"音声処理エラー {audio_path}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []
    
    def process_from_csv(self, csv_path: str, output_dir: str):
        """CSVから音声を一括処理"""
        df = pd.read_csv(csv_path)
        
        if 'audio_path' not in df.columns:
            self.logger.error("CSVに'audio_path'列がありません")
            return []
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_segments = []
        
        print(f"\n音声分割開始: {len(df)}個の音声")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="音声分割中"):
            audio_path = row['audio_path']
            audio_name = row.get('name', Path(audio_path).stem)
            
            if not Path(audio_path).exists():
                self.logger.warning(f"音声が見つかりません: {audio_path}")
                continue
            
            segments = self.process_single_audio(audio_path, output_dir, audio_name)
            all_segments.extend(segments)
        
        # セグメント情報をCSVに保存
        if all_segments:
            segments_df = pd.DataFrame(all_segments)
            segments_csv = output_path / 'audio_segments_info.csv'
            segments_df.to_csv(segments_csv, index=False)
            
            self.logger.info(f"音声分割完了: {len(all_segments)}セグメント生成")
            self.logger.info(f"セグメント情報: {segments_csv}")
        
        return all_segments

# ========================================
# 文字起こしクラス（統合）
# ========================================

class AudioTranscriber:
    """音声文字起こしクラス"""
    
    def __init__(self, config):
        self.whisper_model_name = config['whisper_model']
        self.output_format = config.get('output_format', 'json')
        self.debug_mode = config.get('debug_mode', False)
        
        self.setup_logging()
        self.load_whisper_model()
        self.setup_katakana_converter()
    
    def setup_logging(self):
        """ログ設定"""
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
        """カタカナ変換設定"""
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
        """単語をカタカナに変換"""
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
            return word.translate(self.hiragana_to_katakana)
    
    def convert_to_katakana_with_mecab(self, text: str) -> str:
        """MeCabでカタカナ変換"""
        if not text or self.mecab is None:
            return text.translate(self.hiragana_to_katakana)
        
        try:
            words = text.split()
            katakana_words = [self.word_to_katakana(word) for word in words]
            return ''.join(katakana_words)
        except Exception as e:
            return text.translate(self.hiragana_to_katakana)
    
    def clean_and_convert_to_katakana(self, text: str) -> str:
        """テキストをクリーニングしてカタカナに変換"""
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
    
    def transcribe_single_audio(self, audio_path: str) -> str:
        """単一音声ファイルの文字起こし"""
        if self.whisper_model is None:
            return ""
        
        try:
            result = self.whisper_model.transcribe(
                audio_path,
                language='ja',
                verbose=False,
                fp16=torch.cuda.is_available()
            )
            
            raw_text = result.get('text', '').strip()
            clean_text = self.clean_and_convert_to_katakana(raw_text)
            
            if self.debug_mode and not hasattr(self, 'debug_count'):
                self.debug_count = 0
            
            if self.debug_mode:
                self.debug_count += 1
                if self.debug_count <= 3:
                    self.logger.info(f"変換例 {self.debug_count}: '{raw_text}' -> '{clean_text}'")
            
            return clean_text
        except Exception as e:
            self.logger.warning(f"文字起こしエラー {audio_path}: {e}")
            return ""
    
    def find_audio_files(self, audio_dir: str) -> list:
        """音声ファイルを探索"""
        audio_path = Path(audio_dir)
        audio_files = []
        
        if audio_path.is_file():
            audio_files = [audio_path]
        elif audio_path.is_dir():
            audio_files = list(audio_path.glob("*.wav"))
            audio_files.extend(audio_path.glob("*.mp3"))
            audio_files.extend(audio_path.glob("*.m4a"))
            audio_files.sort()
        else:
            parent = audio_path.parent
            if parent.exists():
                audio_files = list(parent.rglob("*.wav"))
                audio_files.extend(parent.rglob("*.mp3"))
                audio_files.extend(parent.rglob("*.m4a"))
                audio_files.sort()
        
        return audio_files
    
    def process_audio_directory(self, audio_dir: str, output_file: str):
        """音声ディレクトリを一括処理"""
        audio_files = self.find_audio_files(audio_dir)
        
        if not audio_files:
            self.logger.warning(f"音声ファイルが見つかりません: {audio_dir}")
            return
        
        self.logger.info(f"文字起こし開始: {len(audio_files)}ファイル")
        
        transcription_results = {}
        successful_count = 0
        
        for audio_file in tqdm(audio_files, desc="文字起こし中"):
            file_key = audio_file.stem
            transcribed_text = self.transcribe_single_audio(str(audio_file))
            
            transcription_results[file_key] = {
                'audio_file': str(audio_file),
                'transcribed_text': transcribed_text,
                'text_length': len(transcribed_text),
                'has_text': len(transcribed_text) > 0
            }
            
            if transcribed_text:
                successful_count += 1
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcription_results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"文字起こし完了: {successful_count}/{len(audio_files)}ファイル")

# ========================================
# パイプライン統合クラス
# ========================================

class CompletePipeline:
    """完全統合パイプラインクラス"""
    
    def __init__(self, config):
        self.config = config
        self.input_csv = config['input_csv']
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.setup_logging()
        
        # 処理状態
        self.video_done = False
        self.audio_done = False
        self.transcribe_done = False
        self.video_error = None
        self.audio_error = None
        self.transcribe_error = None
    
    def setup_logging(self):
        """ログ設定"""
        log_file = self.output_dir / 'process.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def verify_input_csv(self):
        """入力CSV検証"""
        if not Path(self.input_csv).exists():
            self.logger.error(f"入力CSVが見つかりません: {self.input_csv}")
            return False
        
        try:
            df = pd.read_csv(self.input_csv)
            required_cols = ['video_path', 'audio_path']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                self.logger.error(f"必須列が不足: {missing_cols}")
                return False
            
            self.logger.info(f"入力CSV検証OK: {len(df)}ペア")
            
            print("\n入力データサンプル（最初の3件）:")
            for i, row in df.head(3).iterrows():
                print(f"  {i+1}. 動画: {Path(row['video_path']).name}")
                print(f"      音声: {Path(row['audio_path']).name}")
            
            return True
        except Exception as e:
            self.logger.error(f"CSV読み込みエラー: {e}")
            return False
    
    def gpu0_video_processing(self):
        """GPU 0: 動画分割（統合実行）"""
        self.logger.info("[GPU 0] 動画分割処理開始")
        
        try:
            # GPU 0を指定
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            
            video_output = self.output_dir / 'video_segments'
            video_output.mkdir(parents=True, exist_ok=True)
            
            # VideoSegmenterで処理
            segmenter_config = {
                'window_sec': self.config.get('window_sec', 2.0),
                'slide_sec': self.config.get('slide_sec', 1.0),
                'target_fps': self.config.get('target_fps', 20),
                'target_size': (64, 64)
            }
            
            segmenter = VideoSegmenter(segmenter_config)
            segments = segmenter.process_from_csv(self.input_csv, str(video_output))
            
            self.logger.info(f"[GPU 0] 動画分割完了: {len(segments) if segments else 0}セグメント")
            self.video_done = True
            
        except Exception as e:
            self.logger.error(f"[GPU 0] エラー: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.video_error = str(e)
    
    def gpu1_audio_and_transcribe(self):
        """GPU 1: 音声分割 + 文字起こし（統合実行）"""
        self.logger.info("[GPU 1] 音声処理 + 文字起こし開始")
        
        # GPU 1を指定
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        
        try:
            # ステップ1: 音声分割
            self.logger.info("[GPU 1] 音声分割処理開始")
            audio_output = self.output_dir / 'audio_segments'
            audio_output.mkdir(parents=True, exist_ok=True)
            
            # AudioSegmenterで処理
            audio_segmenter_config = {
                'window_sec': self.config.get('window_sec', 2.0),
                'slide_sec': self.config.get('slide_sec', 1.0),
                'target_sample_rate': 16000
            }
            
            audio_segmenter = AudioSegmenter(audio_segmenter_config)
            audio_segments = audio_segmenter.process_from_csv(self.input_csv, str(audio_output))
            
            self.logger.info(f"[GPU 1] 音声分割完了: {len(audio_segments) if audio_segments else 0}セグメント")
            self.audio_done = True
            
            # ステップ2: 文字起こし（統合実行）
            self.logger.info("[GPU 1] 文字起こし処理開始")
            transcription_file = self.output_dir / 'transcriptions.json'
            
            transcriber_config = {
                'whisper_model': self.config.get('whisper_model', 'base'),
                'output_format': 'json',
                'debug_mode': self.config.get('debug_mode', False)
            }
            
            transcriber = AudioTranscriber(transcriber_config)
            transcriber.process_audio_directory(str(audio_output), str(transcription_file))
            
            self.logger.info("[GPU 1] 文字起こし完了")
            self.transcribe_done = True
            
        except Exception as e:
            self.logger.error(f"[GPU 1] 予期しないエラー: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            if not self.audio_done:
                self.audio_error = str(e)
            else:
                self.transcribe_error = str(e)
    
    def combine_to_final_csv(self):
        """最終CSV生成（統合実行）"""
        self.logger.info("最終CSV生成開始")
        
        try:
            video_dir = self.output_dir / 'video_segments'
            transcription_file = self.output_dir / 'transcriptions.json'
            
            # 文字起こし結果を読み込み
            if not transcription_file.exists():
                self.logger.error(f"文字起こし結果が見つかりません: {transcription_file}")
                return False
            
            with open(transcription_file, 'r', encoding='utf-8') as f:
                transcriptions = json.load(f)
            
            # 動画セグメントファイルを取得
            video_files = list(video_dir.glob("*.pt"))
            
            if not video_files:
                self.logger.error(f"動画セグメントが見つかりません: {video_dir}")
                return False
            
            self.logger.info(f"動画セグメント: {len(video_files)}個")
            self.logger.info(f"文字起こし結果: {len(transcriptions)}個")
            
            # データを結合
            dataset = []
            
            for video_file in tqdm(video_files, desc="データ結合中"):
                video_name = video_file.stem  # 拡張子なしのファイル名
                
                # 対応する文字起こしを探す
                # video_name形式: "video1_0001" → 元の名前 "video1" を抽出
                base_name = '_'.join(video_name.split('_')[:-1])  # 最後のセグメント番号を除去
                
                # 完全一致を優先
                transcription = transcriptions.get(video_name, None)
                
                # 完全一致がない場合、ベース名で検索
                if transcription is None:
                    transcription = transcriptions.get(base_name, None)
                
                if transcription and transcription.get('has_text', False):
                    text = transcription['transcribed_text']
                    text_len = transcription['text_length']
                    
                    # フィルタリング
                    min_len = self.config.get('min_text_len', 3)
                    max_len = self.config.get('max_text_len', 50)
                    
                    if min_len <= text_len <= max_len:
                        dataset.append({
                            'video_path': str(video_file),
                            'text': text,
                            'text_length': text_len
                        })
            
            if not dataset:
                self.logger.error("有効なデータが1件もありません")
                return False
            
            self.logger.info(f"有効なデータ: {len(dataset)}件")
            
            # DataFrameに変換
            df = pd.DataFrame(dataset)
            
            # シャッフル
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # train/valid分割
            valid_ratio = self.config.get('valid_ratio', 0.2)
            valid_size = int(len(df) * valid_ratio)
            train_size = len(df) - valid_size
            
            df_train = df.iloc[:train_size]
            df_valid = df.iloc[train_size:]
            
            # CSV保存
            train_csv = self.output_dir / 'final_train.csv'
            valid_csv = self.output_dir / 'final_valid.csv'
            
            df_train.to_csv(train_csv, index=False, encoding='utf-8')
            df_valid.to_csv(valid_csv, index=False, encoding='utf-8')
            
            self.logger.info(f"学習データ: {len(df_train)}件 → {train_csv}")
            self.logger.info(f"検証データ: {len(df_valid)}件 → {valid_csv}")
            
            # 統計情報
            print("\n" + "="*70)
            print("=== 最終データセット統計 ===")
            print("="*70)
            print(f"総データ数: {len(df)}件")
            print(f"学習データ: {len(df_train)}件 ({len(df_train)/len(df)*100:.1f}%)")
            print(f"検証データ: {len(df_valid)}件 ({len(df_valid)/len(df)*100:.1f}%)")
            print(f"\nテキスト長統計:")
            print(f"  平均: {df['text_length'].mean():.1f}文字")
            print(f"  最小: {df['text_length'].min()}文字")
            print(f"  最大: {df['text_length'].max()}文字")
            print(f"  中央値: {df['text_length'].median():.1f}文字")
            print("="*70)
            
            # サンプル表示
            print("\n=== データサンプル（学習データから5件）===")
            for idx, row in df_train.head(5).iterrows():
                print(f"{idx+1}. {Path(row['video_path']).name}")
                print(f"   テキスト: {row['text']} ({row['text_length']}文字)")
            print("="*70)
            
            return True
            
        except Exception as e:
            self.logger.error(f"CSV生成エラー: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def monitor_progress(self):
        """進捗モニタリング"""
        print("\n" + "="*70)
        print("=== 処理進捗 ===")
        print("="*70)
        print("GPU 0: 動画分割（2秒ウィンドウ、1秒スライド、FPS20、PT形式）")
        print("GPU 1: 音声分割（2秒ウィンドウ、1秒スライド）+ Whisper文字起こし")
        print("="*70)
        print()
        
        start_time = time.time()
        
        while not (self.video_done and self.audio_done and self.transcribe_done):
            elapsed = time.time() - start_time
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            
            video_status = "✅ 完了" if self.video_done else ("❌ エラー" if self.video_error else "🔄 処理中")
            audio_status = "✅ 完了" if self.audio_done else ("❌ エラー" if self.audio_error else "🔄 処理中")
            transcribe_status = "✅ 完了" if self.transcribe_done else ("❌ エラー" if self.transcribe_error else ("⏸️  待機中" if not self.audio_done else "🔄 処理中"))
            
            print(f"\r経過時間: {mins:02d}:{secs:02d} | "
                  f"動画: {video_status:12s} | "
                  f"音声: {audio_status:12s} | "
                  f"文字起こし: {transcribe_status:12s}",
                  end='', flush=True)
            
            if self.video_error or self.audio_error or self.transcribe_error:
                print("\n\n❌ エラーが発生しました")
                return False
            
            time.sleep(1)
        
        elapsed = time.time() - start_time
        print(f"\n\n✅ 並列処理完了！ 処理時間: {elapsed/60:.1f}分\n")
        return True
    
    def run(self):
        """メイン実行"""
        print("\n" + "="*70)
        print("=== 読唇術データセット完全自動処理 ===")
        print("="*70)
        print(f"入力CSV: {self.input_csv}")
        print(f"出力先: {self.output_dir}")
        print(f"動画設定: {self.config.get('window_sec', 2.0)}秒ウィンドウ、{self.config.get('slide_sec', 1.0)}秒スライド、FPS{self.config.get('target_fps', 20)}")
        print(f"音声設定: {self.config.get('window_sec', 2.0)}秒ウィンドウ、{self.config.get('slide_sec', 1.0)}秒スライド、16kHz")
        print(f"テンソル形状: (1, 40, 1, 64, 64)")
        print(f"Whisperモデル: {self.config.get('whisper_model', 'base')}")
        print("="*70)
        
        overall_start = time.time()
        
        if not self.verify_input_csv():
            return False
        
        print("\n処理を開始します...")
        time.sleep(2)
        
        video_thread = Thread(target=self.gpu0_video_processing, name="GPU-0")
        audio_thread = Thread(target=self.gpu1_audio_and_transcribe, name="GPU-1")
        
        video_thread.start()
        audio_thread.start()
        
        if not self.monitor_progress():
            video_thread.join()
            audio_thread.join()
            return False
        
        video_thread.join()
        audio_thread.join()
        
        print("="*70)
        print("=== 最終CSV生成 ===")
        print("="*70)
        
        if not self.combine_to_final_csv():
            return False
        
        overall_elapsed = time.time() - overall_start
        
        print("\n" + "="*70)
        print("=== 🎉 全処理完了！ ===")
        print("="*70)
        print(f"総処理時間: {overall_elapsed/60:.1f}分")
        print(f"\n📁 出力ファイル:")
        print(f"   動画セグメント: {self.output_dir / 'video_segments'}")
        print(f"   音声セグメント: {self.output_dir / 'audio_segments'}")
        print(f"   文字起こし結果: {self.output_dir / 'transcriptions.json'}")
        print(f"   最終データセット: {self.output_dir / 'final_train.csv'}")
        print(f"                     {self.output_dir / 'final_valid.csv'}")
        print(f"\n📊 ログファイル:")
        print(f"   {self.output_dir / 'process.log'}")
        print("="*70)
        
        return True

# ========================================
# メイン実行
# ========================================

def main():
    # ========================================
    # 🔧 設定（ここを編集）
    # ========================================
    
    # 入力CSV（create_dataset_csv.pyで作成したファイル）
    INPUT_CSV = '/home/bv20049/dataset/npz/zundadata/dataset_be.csv'
    
    # 出力ディレクトリ
    OUTPUT_DIR = '/home/bv20049/dataset/npz/zundadata/processed'
    
    # Whisperモデル（tiny, base, small, medium, large）
    WHISPER_MODEL = 'medium'
    
    # 動画・音声処理設定
    WINDOW_SEC = 2.0        # ウィンドウサイズ（秒）
    SLIDE_SEC = 1.0         # スライド幅（秒）
    TARGET_FPS = 20         # 目標FPS（動画のみ）
    
    # フィルタリング設定
    MIN_TEXT_LENGTH = 3     # 最小テキスト長
    MAX_TEXT_LENGTH = 50    # 最大テキスト長
    VALID_RATIO = 0.2       # 検証データ割合（20%）
    
    # デバッグモード
    DEBUG_MODE = False
    
    # ========================================
    
    parser = argparse.ArgumentParser(description='読唇術データセット完全自動処理')
    parser.add_argument('--input', help='入力CSVファイル')
    parser.add_argument('--output', help='出力ディレクトリ')
    parser.add_argument('--whisper-model', choices=['tiny', 'base', 'small', 'medium', 'large'])
    parser.add_argument('--window-sec', type=float, help='ウィンドウサイズ（秒）')
    parser.add_argument('--slide-sec', type=float, help='スライド幅（秒）')
    parser.add_argument('--target-fps', type=int, help='目標FPS')
    parser.add_argument('--min-text-len', type=int)
    parser.add_argument('--max-text-len', type=int)
    parser.add_argument('--valid-ratio', type=float)
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    
    config = {
        'input_csv': args.input if args.input else INPUT_CSV,
        'output_dir': args.output if args.output else OUTPUT_DIR,
        'whisper_model': args.whisper_model if args.whisper_model else WHISPER_MODEL,
        'window_sec': args.window_sec if args.window_sec else WINDOW_SEC,
        'slide_sec': args.slide_sec if args.slide_sec else SLIDE_SEC,
        'target_fps': args.target_fps if args.target_fps else TARGET_FPS,
        'min_text_len': args.min_text_len if args.min_text_len else MIN_TEXT_LENGTH,
        'max_text_len': args.max_text_len if args.max_text_len else MAX_TEXT_LENGTH,
        'valid_ratio': args.valid_ratio if args.valid_ratio else VALID_RATIO,
        'debug_mode': args.debug or DEBUG_MODE
    }
    
    print("\n" + "="*70)
    print("=== 設定確認 ===")
    print("="*70)
    print(f"入力CSV: {config['input_csv']}")
    print(f"出力ディレクトリ: {config['output_dir']}")
    print(f"\n動画処理設定:")
    print(f"  ウィンドウサイズ: {config['window_sec']}秒")
    print(f"  スライド幅: {config['slide_sec']}秒")
    print(f"  目標FPS: {config['target_fps']}")
    print(f"  出力形状: (1, 40, 1, 64, 64)")
    print(f"\n音声処理設定:")
    print(f"  ウィンドウサイズ: {config['window_sec']}秒")
    print(f"  スライド幅: {config['slide_sec']}秒")
    print(f"  サンプルレート: 16kHz")
    print(f"  Whisperモデル: {config['whisper_model']}")
    print(f"\nフィルタリング設定:")
    print(f"  テキスト長: {config['min_text_len']}～{config['max_text_len']}文字")
    print(f"  検証データ割合: {config['valid_ratio']*100:.0f}%")
    print(f"\nその他:")
    print(f"  デバッグモード: {'ON' if config['debug_mode'] else 'OFF'}")
    print("="*70)
    
    # 確認プロンプト
    response = input("\nこの設定で実行しますか？ (y/N): ")
    if response.lower() != 'y':
        print("キャンセルしました")
        return
    
    # 依存関係チェック
    if not TORCH_AVAILABLE:
        print("❌ PyTorch/torchaudioがインストールされていません")
        print("インストール: pip install torch torchaudio")
        return
    
    if not WHISPER_AVAILABLE:
        print("❌ Whisper/MeCabがインストールされていません")
        print("インストール: pip install openai-whisper mecab-python3")
        return
    
    # 実行
    pipeline = CompletePipeline(config)
    success = pipeline.run()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()