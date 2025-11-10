#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データセットとDataLoader（子音専用・簡潔版）
- 20fps × 2秒 = 40フレーム想定（max_lengthは引数で変更可）
"""
import re
import os
import random
from collections import OrderedDict

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def build_phoneme_encoder(mode: str):
    """
    mode: 'consonant' or 'vowel'
    Returns:
        encoder  : 子音/母音エンコーダ
        labels   : 可視化や分析用のラベル配列（blank除く）
    """
    mode = (mode or "consonant").lower()
    if mode == "consonant":
        enc = ConsonantOnlyPhonemeEncoder()
        labels = getattr(enc, "consonants", None) or []
    elif mode == "vowel":
        enc = VowelOnlyPhonemeEncoder()
        labels = getattr(enc, "vowels", None) or []
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return enc, labels


# =========================
#  Data Augmentation
# =========================
class VideoAugmentation:
    """動画データ拡張（必要な時だけ有効化）"""

    def __init__(self, config=None):
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", False))

    def __call__(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_tensor: (T, C, H, W)
        Returns:
            (T, C, H, W)
        """
        if not self.enabled:
            return video_tensor

        if self.config.get("random_crop", False):
            video_tensor = self._random_spatial_crop(video_tensor)

        if self.config.get("random_brightness", False):
            video_tensor = self._random_brightness(video_tensor)

        if self.config.get("random_noise", False):
            video_tensor = self._random_noise(video_tensor)

        return video_tensor

    def _random_spatial_crop(self, video: torch.Tensor) -> torch.Tensor:
        """ランダムクロップ（全フレーム同じ位置）"""
        T, C, H, W = video.shape
        s0, s1 = self.config.get("crop_scale", [0.85, 1.0])
        scale = random.uniform(s0, s1)

        new_h, new_w = int(H * scale), int(W * scale)
        top = random.randint(0, H - new_h) if new_h < H else 0
        left = random.randint(0, W - new_w) if new_w < W else 0

        video = video[:, :, top : top + new_h, left : left + new_w]
        # 元サイズに戻す
        video = (
            F.interpolate(
                video.view(T * C, 1, new_h, new_w),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            .view(T, C, H, W)
            .contiguous()
        )
        return video

    def _random_brightness(self, video: torch.Tensor) -> torch.Tensor:
        """輝度変更"""
        factor = float(self.config.get("brightness_factor", 0.2))
        brightness = 1.0 + random.uniform(-factor, factor)
        return (video * brightness).clamp(0, 1)

    def _random_noise(self, video: torch.Tensor) -> torch.Tensor:
        """ノイズ追加"""
        std = float(self.config.get("noise_std", 0.03))
        noise = torch.randn_like(video) * std
        return (video + noise).clamp(0, 1)


# =========================
#  Consonant-only Encoder
# =========================
class ConsonantOnlyPhonemeEncoder:
    """カタカナ→子音→ID（子音のみ）"""

    def __init__(self):
        # 必要最低限の子音セット（blankは0）
        self.consonants = [
            "k", "s", "t", "n", "h", "m", "y", "r", "w",
            "g", "z", "d", "b", "p", "N"
        ]
        self.blank_token = "<blank>"

        self.phoneme_to_id = {self.blank_token: 0}
        for i, c in enumerate(self.consonants, start=1):
            self.phoneme_to_id[c] = i
        self.id_to_phoneme = {v: k for k, v in self.phoneme_to_id.items()}

        # 1文字マッピング（子音だけ取り出す）
        self.katakana_to_consonant = {
            # 清音
            "カ": "k", "キ": "k", "ク": "k", "ケ": "k", "コ": "k",
            "サ": "s", "シ": "s", "ス": "s", "セ": "s", "ソ": "s",
            "タ": "t", "チ": "t", "ツ": "t", "テ": "t", "ト": "t",
            "ナ": "n", "ニ": "n", "ヌ": "n", "ネ": "n", "ノ": "n",
            "ハ": "h", "ヒ": "h", "フ": "h", "ヘ": "h", "ホ": "h",
            "マ": "m", "ミ": "m", "ム": "m", "メ": "m", "モ": "m",
            "ヤ": "y", "ユ": "y", "ヨ": "y",
            "ラ": "r", "リ": "r", "ル": "r", "レ": "r", "ロ": "r",
            "ワ": "w", "ヲ": "w",
            # 濁音・半濁音
            "ガ": "g", "ギ": "g", "グ": "g", "ゲ": "g", "ゴ": "g",
            "ザ": "z", "ジ": "z", "ズ": "z", "ゼ": "z", "ゾ": "z",
            "ダ": "d", "ヂ": "d", "ヅ": "d", "デ": "d", "ド": "d",
            "バ": "b", "ビ": "b", "ブ": "b", "ベ": "b", "ボ": "b",
            "パ": "p", "ピ": "p", "プ": "p", "ペ": "p", "ポ": "p",
            # 撥音
            "ン": "N",
            # 子音なし（母音単体や長音）は無視
            "ア": None, "イ": None, "ウ": None, "エ": None, "オ": None, "ー": None,
        }

        # 拗音（2文字）→ 子音
        self.palatalized_map = {
            "キャ": "k", "キュ": "k", "キョ": "k",
            "シャ": "s", "シュ": "s", "ショ": "s",
            "チャ": "t", "チュ": "t", "チョ": "t",
            "ニャ": "n", "ニュ": "n", "ニョ": "n",
            "ヒャ": "h", "ヒュ": "h", "ヒョ": "h",
            "ミャ": "m", "ミュ": "m", "ミョ": "m",
            "リャ": "r", "リュ": "r", "リョ": "r",
            "ギャ": "g", "ギュ": "g", "ギョ": "g",
            "ジャ": "z", "ジュ": "z", "ジョ": "z",
            "ビャ": "b", "ビュ": "b", "ビョ": "b",
            "ピャ": "p", "ピュ": "p", "ピョ": "p",
        }

    @property
    def blank_id(self) -> int:
        return 0

    @property
    def num_classes(self) -> int:
        return len(self.phoneme_to_id)

    # ---- API ----
    def text_to_phonemes(self, katakana_text: str):
        """カタカナ文字列を子音列へ"""
        phonemes = []
        i = 0
        while i < len(katakana_text):
            # 2文字（拗音）
            if i < len(katakana_text) - 1:
                two = katakana_text[i : i + 2]
                if two in self.palatalized_map:
                    c = self.palatalized_map[two]
                    if c:
                        phonemes.append(c)
                    i += 2
                    continue
            # 1文字
            ch = katakana_text[i]
            cons = self.katakana_to_consonant.get(ch, None)
            if cons:
                phonemes.append(cons)
            i += 1
        return phonemes

    def encode_phonemes(self, phonemes):
        """子音列→ID列（blankを挟まない素直な符号化）"""
        if not phonemes:
            return []
        return [self.phoneme_to_id[p] for p in phonemes if p in self.phoneme_to_id]

    def decode_phonemes(self, ids):
        """ID列→子音列（blankは除外）"""
        return [self.id_to_phoneme.get(i, "<unk>") for i in ids if i != self.blank_id]


# =========================
#  Dataset
# =========================
class CachedAttentionCTCDataset:
    """LRUキャッシュ付きデータセット（ptファイルから[T,1,64,64]を得る）"""

    def __init__(
        self,
        csv_path: str,
        phoneme_encoder: ConsonantOnlyPhonemeEncoder,
        max_length: int = 40,
        cache_size: int = 100,
        augmentation_config=None,
        is_training: bool = True,
    ):
        self.df = pd.read_csv(csv_path)
        self.phoneme_encoder = phoneme_encoder
        self.max_length = int(max_length)
        self.cache_size = int(cache_size)
        self.is_training = bool(is_training)
        self.augmentation = VideoAugmentation(augmentation_config) if augmentation_config else None

        # LRU cache
        self._cache = OrderedDict()
        self._hits = 0
        self._miss = 0

        self._filter_existing_files()
        print(f"✓ Dataset loaded: {len(self.df)} samples (cache={cache_size}, max_len={max_length})")

    def _filter_existing_files(self):
        valid_idx = []
        for i in range(len(self.df)):
            try:
                vp = self.df.iloc[i]["video_path"]
                if isinstance(vp, str) and os.path.exists(vp):
                    valid_idx.append(i)
            except Exception:
                pass
        if len(valid_idx) < len(self.df):
            self.df = self.df.iloc[valid_idx].reset_index(drop=True)
            print(f"⚠ 無効ファイルを除外: {len(self.df) - len(valid_idx)}件")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        video_path = row["video_path"]

        # cache
        if self.cache_size > 0 and video_path in self._cache:
            self._hits += 1
            self._cache.move_to_end(video_path)
            video = self._cache[video_path].clone()
        else:
            self._miss += 1
            video = self._load_video(video_path)
            if self.cache_size > 0:
                self._cache[video_path] = video
                if len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)

        # augmentation
        if self.is_training and self.augmentation and self.augmentation.enabled:
            video = self.augmentation(video)

        # labels
        text = str(row.get("text", ""))
        # ---- 子音 or 母音で分岐 ----
        if hasattr(self.phoneme_encoder, "text_to_phonemes"):
            # 子音モード
            phs = self.phoneme_encoder.text_to_phonemes(text)
            ids = self.phoneme_encoder.encode_phonemes(phs)  # List[int]
        else:
            # 母音モード
            phs = self.phoneme_encoder.katakana_to_vowel_sequence(text)  # List[str]（a,i,u,e,o）
            ids = self.phoneme_encoder.encode(text)  # Tensor になってるのでListに統一
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()

        return {
            "video": video,                           # (T,1,64,64)
            "target": ids,                            # List[int]（←必ずリスト）
            "input_length": video.size(0),
            "target_length": len(ids),
            "text": text,
            "phonemes": phs,
        }

    def _load_video(self, video_path: str) -> torch.Tensor:
        """
        ptファイルから動画テンソルを読み込み、(T,1,64,64)へ整形
        許容入力: dict or Tensor
          - dictの場合: 'video'/'data'/'frames' いずれか
          - 形状: (T,1,64,64) / (1,T,1,64,64) / (T,64,64) / (T,H,W,C)
        """
        try:
            data = torch.load(video_path, map_location="cpu")

            # dict -> pick a likely key
            if isinstance(data, dict):
                for k in ("video", "data", "frames"):
                    if k in data:
                        data = data[k]
                        break
                else:
                    # 最初の値を取る
                    data = next(iter(data.values()))

            # (1,T,C,H,W) -> (T,C,H,W)
            if data.ndim == 5:
                if data.size(0) == 1:
                    data = data.squeeze(0)
                else:
                    raise ValueError(f"Unexpected 5D batch size: {tuple(data.shape)}")

            # (T,H,W) -> (T,1,H,W)
            if data.ndim == 3:
                data = data.unsqueeze(1)

            # (T,H,W,C) -> (T,C,H,W)
            if data.ndim == 4 and data.size(1) > 10:
                data = data.permute(0, 3, 1, 2)

            if data.ndim != 4:
                raise ValueError(f"Expect 4D tensor, got {tuple(data.shape)}")

            # トリム
            if data.size(0) > self.max_length:
                data = data[: self.max_length]

            # リサイズ to 64x64
            if data.size(-1) != 64 or data.size(-2) != 64:
                T, C, H, W = data.shape
                data = F.interpolate(
                    data.view(T * C, 1, H, W),
                    size=(64, 64),
                    mode="bilinear",
                    align_corners=False,
                ).view(T, C, 64, 64)

            # 最終形状確認
            assert data.shape[1:] == (1, 64, 64), f"Invalid shape: {tuple(data.shape)}"
            return data.float()

        except Exception as e:
            print(f"❌ 動画読み込みエラー: {video_path} -> {e}")
            return torch.zeros(self.max_length, 1, 64, 64, dtype=torch.float32)

    # 必要になれば参照できる程度に最小限だけ残す
    def get_cache_stats(self):
        total = self._hits + self._miss
        hit_rate = (self._hits / total * 100.0) if total > 0 else 0.0
        return {"cache": len(self._cache), "hits": self._hits, "miss": self._miss, "hit_rate": hit_rate}


# =========================
#  Collate
# =========================
def attention_ctc_collate_fn(batch):
    """可変長シーケンスをまとめてバッチ化（CTC用にtargetは連結）"""
    videos, targets, input_lengths, target_lengths, texts, phonemes = [], [], [], [], [], []

    # 1) 収集（targetsは連結）
    for item in batch:
        v = item["video"]
        videos.append(v)
        targets.extend(item["target"])
        input_lengths.append(item["input_length"])
        target_lengths.append(item["target_length"])
        texts.append(item["text"])
        phonemes.append(item["phonemes"])

    # 2) 動画を時間次元でパディング（右側ゼロ埋め）
    max_len = max(v.size(0) for v in videos)
    padded = []
    for v in videos:
        if v.size(0) < max_len:
            pad = torch.zeros(max_len - v.size(0), *v.shape[1:], dtype=v.dtype)
            v = torch.cat([v, pad], dim=0)
        padded.append(v)

    return {
        "video": torch.stack(padded, dim=0),                 # (B, T, 1, 64, 64)
        "target": torch.tensor(targets, dtype=torch.long),   # (sum_L,)
        "input_length": torch.tensor(input_lengths, dtype=torch.long),   # (B,)
        "target_length": torch.tensor(target_lengths, dtype=torch.long), # (B,)
        "text": texts,
        "phonemes": phonemes,
    }


# =========================
#  DataLoader factory
# =========================
# =========================
#  DataLoader factory
# =========================
def create_dataloaders(
    train_csv_path: str,
    valid_csv_path: str,
    batch_size: int = 16,
    num_workers: int = 0,
    cache_size: int = 100,
    augmentation_config=None,
    max_length: int = 40,
    mode: str = "consonant",   # ← 追加: 'consonant' or 'vowel'
):
    """
    DataLoader を作成（modeで子音/母音を切替）
    Returns:
        train_loader, valid_loader, phoneme_encoder
    """
    # --- ここでエンコーダを切替 ---
    if mode == "consonant":
        print("[DataLoader] mode=consonant")
        encoder = ConsonantOnlyPhonemeEncoder()
        labels = encoder.consonants
    elif mode == "vowel":
        print("[DataLoader] mode=vowel")
        encoder = VowelOnlyPhonemeEncoder()
        labels = encoder.vowels
    else:
        raise ValueError(f"Unknown mode: {mode}")

    train_dataset = CachedAttentionCTCDataset(
        csv_path=train_csv_path,
        phoneme_encoder=encoder,
        max_length=max_length,
        cache_size=cache_size,
        augmentation_config=augmentation_config,
        is_training=True,
    )

    valid_dataset = CachedAttentionCTCDataset(
        csv_path=valid_csv_path,
        phoneme_encoder=encoder,
        max_length=max_length,
        cache_size=cache_size,
        augmentation_config=None,
        is_training=False,
    )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=attention_ctc_collate_fn,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=attention_ctc_collate_fn,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, valid_loader, encoder, labels  # ← 4つ返す

class VowelOnlyPhonemeEncoder:
    """
    カタカナ列 → 母音列[a,i,u,e,o] に変換
    - 撥音「ン」、促音「ッ」、記号・空白は無視
    - 長音「ー」は直前母音に吸収（=重複として圧縮）
    - 連続同一母音は1個に圧縮（例:「オオイ」→「オイ」）
    """
    def __init__(self):
        # 出力語彙（blankはCTC用）
        self.vowels = ['a', 'i', 'u', 'e', 'o']
        self.blank_id = 0
        self.vocab = ['<blank>'] + self.vowels
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.itos = {i: s for s, i in self.stoi.items()}
    

        # --- カタカナ→母音核 辞書（主要音節＋拗音＋外来音の代表）
        #     ※母音だけが欲しいので、各仮名を最終母音に写像
        A = 'アァカガサザタダナハバパマヤャラワヮァァ'  # 最終母音 a
        I = 'イィキギシジチヂニヒビピミリヰ'            # i
        U = 'ウゥクグスズツヅヌフブプムユュルゥゥ'        # u
        E = 'エェケゲセゼテデネヘベペメレヱ'            # e
        O = 'オォコゴソゾトドノホボポモヨョロヲォ'        # o
        # 小文字拗音（ャュョ）は前子音と結合し最終母音が a/u/o になる想定だが、
        # ここでは単独出現時の保険で a/u/o に割り当て済み。

        self.kana2vowel = {ch: 'a' for ch in A}
        self.kana2vowel.update({ch: 'i' for ch in I})
        self.kana2vowel.update({ch: 'u' for ch in U})
        self.kana2vowel.update({ch: 'e' for ch in E})
        self.kana2vowel.update({ch: 'o' for ch in O})

        # 外来音や拡張（ヴ/ティ/ディ/ファ/フィ/フェ/フォ 等）
        self.digraph_map = {
            # ヴ + 母音
            'ヴァ':'a','ヴィ':'i','ヴ':'u','ヴェ':'e','ヴォ':'o',
            # 子音+小母音（CV拡張）
            'キャ':'a','キュ':'u','キョ':'o','ギャ':'a','ギュ':'u','ギョ':'o',
            'シャ':'a','シュ':'u','ショ':'o','ジャ':'a','ジュ':'u','ジョ':'o',
            'チャ':'a','チュ':'u','チョ':'o','ヂャ':'a','ヂュ':'u','ヂョ':'o',
            'ニャ':'a','ニュ':'u','ニョ':'o','ヒャ':'a','ヒュ':'u','ヒョ':'o',
            'ミャ':'a','ミュ':'u','ミョ':'o','リャ':'a','リュ':'u','リョ':'o',
            'ファ':'a','フィ':'i','フェ':'e','フォ':'o','フュ':'u',
            'ティ':'i','ディ':'i','トゥ':'u','ドゥ':'u',
            'チェ':'e','シェ':'e','ジェ':'e',
            'ツァ':'a','ツィ':'i','ツェ':'e','ツォ':'o',
            # 小母音単独（保険）
            'ァ':'a','ィ':'i','ゥ':'u','ェ':'e','ォ':'o',
        }

        # 無視する記号類
        self.ignore_pattern = re.compile(r'[^\u30A0-\u30FF]')  # カタカナ以外
        self.skip_chars = set('ンッ・，、。 「」『』（）()[]{}／-ー　 ')  # ーは特別扱い

    def num_classes(self) -> int:
        return len(self.vocab)
    
    def katakana_to_vowel_sequence(self, text: str):
        # カタカナ以外を除去
        s = self.ignore_pattern.sub('', text)

        res = []
        i = 0
        L = len(s)
        while i < L:
            # 2文字の拗音・外来音を優先
            if i+1 < L:
                pair = s[i:i+2]
                if pair in self.digraph_map:
                    v = self.digraph_map[pair]
                    res.append(v)
                    i += 2
                    continue
            ch = s[i]
            if ch in self.skip_chars:
                if ch == 'ー' and res:
                    # 長音：直前母音を繰り返し → 後段で圧縮される
                    res.append(res[-1])
                # それ以外は無視
                i += 1
                continue
            v = self.kana2vowel.get(ch, None)
            if v is not None:
                res.append(v)
            i += 1

        # 連続同一母音を圧縮（例: オオイ→オイ）
        res_comp = []
        for v in res:
            if not res_comp or res_comp[-1] != v:
                res_comp.append(v)
        return res_comp

    # ====== public API ======
    def encode(self, kata: str) -> torch.Tensor:
        vs = self.katakana_to_vowel_sequence(kata)
        ids = [self.stoi[v] for v in vs]  # blankは含めない
        return torch.tensor(ids, dtype=torch.long)

    def decode_phonemes(self, ids) -> list:
        return [self.itos[int(i)] for i in ids if int(i) in self.itos]