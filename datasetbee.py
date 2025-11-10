#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データセットCSV作成スクリプト
- 動画ファイルと音声ファイルを自動探索
- ファイル名で対応付け
- CSVファイルを生成
"""

import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
from typing import Dict, List, Optional

class DatasetCSVCreator:
    """データセットCSV作成クラス"""
    
    def __init__(self):
        self.setup_logging()
        
        # 対応する拡張子
        self.video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
        self.audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.aac']
    
    def setup_logging(self):
        """ログ設定"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def find_files(self, root_dir: str, extensions: List[str]) -> Dict[str, str]:
        """指定拡張子のファイルを探索"""
        root_path = Path(root_dir)
        
        if not root_path.exists():
            self.logger.error(f"ディレクトリが存在しません: {root_dir}")
            return {}
        
        files_dict = {}
        
        self.logger.info(f"探索中: {root_dir}")
        self.logger.info(f"対象拡張子: {extensions}")
        
        for ext in extensions:
            for file_path in root_path.rglob(f'*{ext}'):
                # ファイル名（拡張子なし）をキーにする
                stem = file_path.stem
                
                # すでに同じ名前のファイルがある場合は警告
                if stem in files_dict:
                    self.logger.warning(f"重複ファイル名: {stem}")
                    self.logger.warning(f"  既存: {files_dict[stem]}")
                    self.logger.warning(f"  新規: {file_path}")
                
                files_dict[stem] = str(file_path)
        
        self.logger.info(f"発見したファイル数: {len(files_dict)}")
        
        # デバッグ: 最初の5件を表示
        if files_dict:
            print("  発見したファイル例（最初の5件）:")
            for i, (stem, path) in enumerate(list(files_dict.items())[:5]):
                print(f"    {i+1}. {stem} -> {Path(path).name}")
        
        return files_dict
    
    def match_video_audio_pairs(self, video_files: Dict[str, str], 
                                audio_files: Dict[str, str],
                                audio_dir: Optional[str] = None) -> List[Dict]:
        """動画と音声ファイルを対応付け（順番ベース）"""
        
        matched_pairs = []
        unmatched_videos = []
        unmatched_audios = []
        
        print("\n動画と音声の対応付け中...")
        print("方式: ソート順で対応付け（1番目の動画 ↔ 1番目の音声）")
        
        # ファイル名でソート
        sorted_video_items = sorted(video_files.items())
        sorted_audio_items = sorted(audio_files.items())
        
        print(f"ソート後の動画ファイル数: {len(sorted_video_items)}")
        print(f"ソート後の音声ファイル数: {len(sorted_audio_items)}")
        
        # 最初の数件を表示（確認用）
        if sorted_video_items:
            print("\n動画ファイル例（最初の3件）:")
            for i, (name, path) in enumerate(sorted_video_items[:3]):
                print(f"  {i+1}. {Path(path).name}")
        
        if sorted_audio_items:
            print("\n音声ファイル例（最初の3件）:")
            for i, (name, path) in enumerate(sorted_audio_items[:3]):
                print(f"  {i+1}. {Path(path).name}")
        
        # 順番で対応付け
        min_count = min(len(sorted_video_items), len(sorted_audio_items))
        
        for i in tqdm(range(min_count), desc="対応付け"):
            video_name, video_path = sorted_video_items[i]
            audio_name, audio_path = sorted_audio_items[i]
            
            # ペア名を生成（両方のファイル名を含める）
            pair_name = f"{video_name}_{audio_name}"
            
            matched_pairs.append({
                'video_name': video_name,
                'audio_name': audio_name,
                'pair_name': pair_name,
                'video_path': video_path,
                'audio_path': audio_path
            })
        
        # 余った動画
        for i in range(min_count, len(sorted_video_items)):
            video_name, video_path = sorted_video_items[i]
            unmatched_videos.append({
                'video_name': video_name,
                'video_path': video_path,
                'audio_path': ''
            })
        
        # 余った音声
        for i in range(min_count, len(sorted_audio_items)):
            audio_name, audio_path = sorted_audio_items[i]
            unmatched_audios.append({
                'audio_name': audio_name,
                'audio_path': audio_path
            })
        
        return matched_pairs, unmatched_videos, unmatched_audios
    
    def create_csv(self, root_dir: str, output_csv: str, 
                  audio_dir: Optional[str] = None,
                  include_unmatched: bool = False,
                  append_mode: bool = False) -> pd.DataFrame:
        """CSVファイル作成"""
        
        print(f"ファイル探索中: {root_dir}")
        
        # 動画ファイル探索
        print("\n動画ファイル探索中...")
        video_files = self.find_files(root_dir, self.video_extensions)
        print(f"発見した動画ファイル: {len(video_files)}個")
        
        # 音声ファイル探索
        print("\n音声ファイル探索中...")
        search_dir = audio_dir if audio_dir else root_dir
        audio_files = self.find_files(search_dir, self.audio_extensions)
        print(f"発見した音声ファイル: {len(audio_files)}個")
        
        if not video_files:
            raise ValueError("動画ファイルが見つかりません")
        
        # 追記モードの場合、既存データを読み込み
        existing_df = None
        existing_names = set()
        
        if append_mode and Path(output_csv).exists():
            print(f"\n追記モード: 既存CSV読み込み中...")
            existing_df = pd.read_csv(output_csv)
            existing_names = set(existing_df['name'].tolist())
            print(f"既存データ: {len(existing_df)}行")
            print(f"既存ファイル名: {len(existing_names)}個")
        
        # 対応付け
        matched_pairs, unmatched_videos, unmatched_audios = self.match_video_audio_pairs(
            video_files, audio_files, audio_dir
        )
        
        # データフレーム作成
        data_list = []
        skipped_count = 0
        
        # マッチしたペア
        for pair in matched_pairs:
            # 追記モードで既に存在する場合はスキップ
            if append_mode and pair['pair_name'] in existing_names:
                skipped_count += 1
                continue
            
            data_list.append({
                'video_path': pair['video_path'],
                'audio_path': pair['audio_path'],
                'name': pair['pair_name'],
                'video_name': pair['video_name'],
                'audio_name': pair['audio_name'],
                'status': 'matched'
            })
        
        # マッチしなかった動画
        if include_unmatched:
            for item in unmatched_videos:
                if append_mode and item['video_name'] in existing_names:
                    skipped_count += 1
                    continue
                
                data_list.append({
                    'video_path': item['video_path'],
                    'audio_path': '',
                    'name': item['video_name'],
                    'status': 'video_only'
                })
        
        new_df = pd.DataFrame(data_list)
        
        # データが空の場合の警告
        if len(new_df) == 0:
            self.logger.warning("新規追加データが0件です")
            if not append_mode:
                self.logger.error("マッチしたペアが1つもありません。ファイル名が一致しているか確認してください")
        
        # 追記モードの場合は既存データと結合
        if append_mode and existing_df is not None:
            df = pd.concat([existing_df, new_df], ignore_index=True)
            print(f"\n追記結果:")
            print(f"  既存データ: {len(existing_df)}行")
            print(f"  新規追加: {len(new_df)}行")
            print(f"  スキップ（重複）: {skipped_count}個")
            print(f"  合計: {len(df)}行")
        else:
            df = new_df
        
        # CSV保存
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        # サマリー表示
        self.print_summary(matched_pairs, unmatched_videos, unmatched_audios, output_csv, 
                          append_mode, skipped_count if append_mode else 0)
        
        return df
    
    def print_summary(self, matched_pairs: List, unmatched_videos: List, 
                     unmatched_audios: List, output_csv: str, 
                     append_mode: bool = False, skipped_count: int = 0):
        """サマリー表示"""
        print(f"\n{'='*60}")
        print("=== 対応付けサマリー ===")
        print(f"{'='*60}")
        
        if append_mode:
            print(f"モード: 追記")
            print(f"🔄 スキップ（既存）: {skipped_count}個")
            print(f"✅ 新規追加: {len(matched_pairs) - skipped_count}ペア")
        else:
            print(f"モード: 新規作成")
            print(f"✅ 対応付け成功: {len(matched_pairs)}ペア")
        
        print(f"⚠️  音声なし動画: {len(unmatched_videos)}個")
        print(f"⚠️  動画なし音声: {len(unmatched_audios)}個")
        
        if unmatched_videos:
            print(f"\n音声が見つからなかった動画（最初の5件）:")
            for item in unmatched_videos[:5]:
                print(f"  - {item['video_name']}")
            if len(unmatched_videos) > 5:
                print(f"  ... 他 {len(unmatched_videos)-5}件")
        
        if unmatched_audios:
            print(f"\n動画が見つからなかった音声（最初の5件）:")
            for item in unmatched_audios[:5]:
                print(f"  - {item['audio_name']}")
            if len(unmatched_audios) > 5:
                print(f"  ... 他 {len(unmatched_audios)-5}件")
        
        print(f"\n出力ファイル: {output_csv}")
        if append_mode:
            print(f"既存CSVに新規データを追記しました")
        else:
            print(f"対応付けされたペアのみCSVに保存されました")
    
    def validate_paths(self, csv_file: str) -> Dict:
        """CSVファイルのパス検証"""
        # CSVファイルが空でないかチェック
        csv_path = Path(csv_file)
        
        if not csv_path.exists():
            self.logger.error(f"CSVファイルが存在しません: {csv_file}")
            return {}
        
        # ファイルサイズチェック
        file_size = csv_path.stat().st_size
        if file_size == 0:
            self.logger.error(f"CSVファイルが空です: {csv_file}")
            return {}
        
        try:
            df = pd.read_csv(csv_file)
        except pd.errors.EmptyDataError:
            self.logger.error(f"CSVファイルにデータがありません: {csv_file}")
            return {}
        
        if len(df) == 0:
            self.logger.warning(f"CSVファイルに行がありません: {csv_file}")
            return {'total': 0, 'video_exists': 0, 'audio_exists': 0, 'both_exist': 0}
        
        print(f"\nCSVファイル検証中: {csv_file}")
        print(f"ファイルサイズ: {file_size} bytes")
        print(f"行数: {len(df)}")
        
        video_exists = 0
        audio_exists = 0
        both_exist = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="検証中"):
            v_exists = Path(row['video_path']).exists()
            a_exists = Path(row['audio_path']).exists() if row['audio_path'] else False
            
            if v_exists:
                video_exists += 1
            if a_exists:
                audio_exists += 1
            if v_exists and a_exists:
                both_exist += 1
        
        result = {
            'total': len(df),
            'video_exists': video_exists,
            'audio_exists': audio_exists,
            'both_exist': both_exist
        }
        
        print(f"\n検証結果:")
        print(f"  総行数: {result['total']}")
        print(f"  動画ファイル存在: {result['video_exists']}/{result['total']}")
        print(f"  音声ファイル存在: {result['audio_exists']}/{result['total']}")
        print(f"  両方存在: {result['both_exist']}/{result['total']}")
        
        return result

def main():
    """メイン実行"""
    
    # ========================================
    # ここに処理したいフォルダのパスを書く
    # ========================================
    
    FOLDER_CONFIGS = [
        {
            'video_dir': '/home/bv20049/dataset/npz/zundadata/ROHAN4600_zumndamon_normal_picture/ROHAN4600_0001-0400_LFROI',
            'audio_dir': '/home/bv20049/dataset/npz/zundadata/ROHAN4600_split',
        },
        {
            'video_dir': '/home/bv20049/dataset/npz/zundadata/zunda/ITA_recitation_nomal_picture/LFROI_20230420',
            'audio_dir': '/home/bv20049/dataset/npz/zundadata/zunda/ITA_recitation_nomal_picture/ITA_recitation_nomal_synchronized_wav',
        },
        {
            'video_dir': '/home/bv20049/dataset/npz/zundadata/zunda/ITA_emotion_normal_picture/LFROI_20230420',
            'audio_dir': '/home/bv20049/dataset/npz/zundadata/zunda/ITA_emotion_normal_picture/ITA_emotion_normal_synchronized_wav',
        },
        {
            'video_dir': '/home/bv20049/dataset/npz/zundadata/usagi/ITA_recitation_normal_picture/LFROI_20230420',
            'audio_dir': '/home/bv20049/dataset/npz/zundadata/usagi/ITA_recitation_normal_picture/ITA_recitation_normal_synchronized_wav',
        },
        {
            'video_dir': '/home/bv20049/dataset/npz/zundadata/sora/ITA_recitation_normal_picture/LFROI_20230420',
            'audio_dir': '/home/bv20049/dataset/npz/zundadata/sora/ITA_recitation_normal_picture/ITA_recitation_normal_synchronized_wav',
        },
        {
            'video_dir': '/home/bv20049/dataset/npz/zundadata/sora/ITA_emotion_normal_picture/LFROI_20230420',
            'audio_dir': '/home/bv20049/dataset/npz/zundadata/sora/ITA_emotion_normal_picture/ITA_emotion_normal_synchronized_wav',
        },
        {
            'video_dir': '/home/bv20049/dataset/npz/zundadata/metan/ITA_recitation_normal_picture/LFROI_20230420',
            'audio_dir': '/home/bv20049/dataset/npz/zundadata/metan/ITA_recitation_normal_picture/ITA_recitation_normal_synchronized_wav',
        },
        {
            'video_dir': '/home/bv20049/dataset/npz/zundadata/metan/ITA_emotion_normal_picture/LFROI_20230420',
            'audio_dir': '/home/bv20049/dataset/npz/zundadata/metan/ITA_emotion_normal_picture/ITA_emotion_normal_synchronized_wav',
        },
        {
            'video_dir': '/home/bv20049/dataset/npz/zundadata/itako/ITA_recitation_normal_picture/LFROI_20230420',
            'audio_dir': '/home/bv20049/dataset/npz/zundadata/itako/ITA_recitation_normal_picture/ITA_recitation_normal_synchronized_wav',
        },
        {
            'video_dir': '/home/bv20049/dataset/npz/zundadata/itako/ITA_emotion_normal_picture/LFROI_20230420',
            'audio_dir': '/home/bv20049/dataset/npz/zundadata/itako/ITA_emotion_normal_picture/ITA_emotion_normal_synchronized_wav',
        },
    ]
    
    # 出力CSVファイル名
    OUTPUT_CSV = '/home/bv20049/dataset/npz/zundadata/dataset_be.csv'
    
    # その他のオプション
    INCLUDE_UNMATCHED = False  # 音声がない動画も含めるか
    VALIDATE = True            # 作成後に検証するか
    
    # ========================================
    # ここから下は変更不要
    # ========================================
    
    parser = argparse.ArgumentParser(description='データセットCSV作成')
    
    # コマンドライン引数（オプション）
    parser.add_argument('--video-dir', help='動画ディレクトリ（コマンドライン指定）')
    parser.add_argument('--audio-dir', help='音声ディレクトリ（コマンドライン指定）')
    parser.add_argument('--output', help='出力CSVファイル（コマンドライン指定）')
    parser.add_argument('--include-unmatched', action='store_true',
                       help='音声がない動画もCSVに含める')
    parser.add_argument('--append', action='store_true',
                       help='既存CSVに追記（重複はスキップ）')
    parser.add_argument('--validate', action='store_true',
                       help='作成したCSVファイルを検証')
    
    args = parser.parse_args()
    
    # コマンドライン引数が指定されている場合はそちらを優先
    if args.video_dir:
        # コマンドライン実行モード
        print("=== コマンドライン実行モード ===")
        creator = DatasetCSVCreator()
        
        df = creator.create_csv(
            root_dir=args.video_dir,
            output_csv=args.output or OUTPUT_CSV,
            audio_dir=args.audio_dir,
            include_unmatched=args.include_unmatched or INCLUDE_UNMATCHED,
            append_mode=args.append
        )
        
        if args.validate or VALIDATE:
            creator.validate_paths(args.output or OUTPUT_CSV)
        
        print(f"\n✅ CSV作成完了！")
        print(f"次のステップ: python batch_process.py --input {args.output or OUTPUT_CSV} --output ./processed_data")
        return
    
    # コード内設定での実行モード
    print("=== コード内設定実行モード ===")
    print(f"処理フォルダ数: {len(FOLDER_CONFIGS)}")
    print(f"出力ファイル: {OUTPUT_CSV}")
    
    creator = DatasetCSVCreator()
    
    # 各フォルダを処理
    for i, config in enumerate(FOLDER_CONFIGS, 1):
        video_dir = config.get('video_dir')
        audio_dir = config.get('audio_dir', None)
        
        if not video_dir:
            print(f"\n[{i}/{len(FOLDER_CONFIGS)}] スキップ: video_dirが指定されていません")
            continue
        
        print(f"\n{'='*60}")
        print(f"[{i}/{len(FOLDER_CONFIGS)}] 処理中: {video_dir}")
        print(f"{'='*60}")
        
        # 最初のフォルダは新規作成、以降は追記
        append_mode = (i > 1)
        
        try:
            df = creator.create_csv(
                root_dir=video_dir,
                output_csv=OUTPUT_CSV,
                audio_dir=audio_dir,
                include_unmatched=INCLUDE_UNMATCHED,
                append_mode=append_mode
            )
        except Exception as e:
            print(f"❌ エラー: {e}")
            continue
    
    # 検証
    if VALIDATE and Path(OUTPUT_CSV).exists():
        print(f"\n{'='*60}")
        print("最終検証")
        print(f"{'='*60}")
        creator.validate_paths(OUTPUT_CSV)
    
    print(f"\n{'='*60}")
    print("✅ 全フォルダの処理完了！")
    print(f"{'='*60}")
    print(f"出力ファイル: {OUTPUT_CSV}")
    print(f"\n次のステップ:")
    print(f"  python batch_process.py --input {OUTPUT_CSV} --output ./processed_data")

if __name__ == "__main__":
    main()

