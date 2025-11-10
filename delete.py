import pandas as pd
import os

def create_char_dict():
    katakana = 'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンッァィゥェォャュョガギグゲゴザジズゼゾダヂヅデドーバビブベボパピプペポヴ'
    digits = '0123456789'
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    additional_chars = ' ・%.,。、'  # 特殊文字
    full_char_set = katakana + digits + alphabet + additional_chars
    return {char: idx + 1 for idx, char in enumerate(full_char_set)}

def filter_dataset(input_csv, output_csv, removed_csv, min_length=10, max_length=35):
    """
    フィルタリング: 
    - 顔ランドマークが描画されていない動画
    - 文字辞書に登録されていない文字を含む文
    - 長すぎるまたは短すぎる文章を削除
    """
    char_dict = create_char_dict()
    data = pd.read_csv(input_csv)
    valid_data = []
    removed_data = []

    for _, row in data.iterrows():
        video_path = row['file_path']
        text = row['text']

        # 動画ファイルが存在するか確認
        if not os.path.exists(video_path):
            removed_data.append({**row, "reason": "Video file not found"})
            continue

        # 文字辞書に登録されていない文字が含まれているか確認
        if not all(char in char_dict for char in text):
            removed_data.append({**row, "reason": "Invalid characters"})
            continue

        # 文章の長さが指定範囲内か確認
        if len(text) < min_length:
            removed_data.append({**row, "reason": "Text too short"})
            continue
        if len(text) > max_length:
            removed_data.append({**row, "reason": "Text too long"})
            continue

        valid_data.append(row)

    # 結果を保存
    valid_df = pd.DataFrame(valid_data)
    removed_df = pd.DataFrame(removed_data)
    valid_df.to_csv(output_csv, index=False)
    removed_df.to_csv(removed_csv, index=False)
    print(f"Filtered dataset saved to {output_csv}")
    print(f"Removed entries logged to {removed_csv}")

# ファイル名をコード内で指定
input_csv = "datasetu.csv"
output_csv = "filtered_datasetnn.csv"
removed_csv = "removed_entriesnn.csv"

# フィルタリング条件を指定
filter_dataset(input_csv, output_csv, removed_csv, min_length=20, max_length=40)