# model_compact_vowel.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 小バッチでBNが不安定なら True に ---
USE_GN = False

def _norm2d(ch: int):
    if USE_GN:
        return nn.GroupNorm(num_groups=min(32, ch), num_channels=ch)
    else:
        return nn.BatchNorm2d(ch)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            _norm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _norm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):  # (B*T,C,H,W)
        return self.net(x)


class CompactVowelLipReader_NoAttn(nn.Module):
    """
    CompactVowelLipReader_NoAttn
    - Frame-wise Attention を完全に除去
    - 単純な 2D-CNN ×3層 + 1〜2層 BiLSTM + CTC
    - 出力は log_softmax 済み（CTCロスに直結）
    """
    def __init__(self, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.returns_log_probs = True

        # ===== CNN部分（空間情報抽出のみ） =====
        self.cnn = nn.Sequential(
            DSConvBlock(1, 32),
            nn.AvgPool2d(2),      # 64→32
            DSConvBlock(32, 64),
            nn.AvgPool2d(2),      # 32→16
            DSConvBlock(64, 128),
            nn.AvgPool2d(2),      # 16→8
        )

        self.feat_dim = 128  # CNN最終出力次元

        # ===== LSTM部分 =====
        self.pre_ln = nn.LayerNorm(self.feat_dim)
        self.pre_drop = nn.Dropout(0.1)

        self.lstm = nn.LSTM(
            input_size=self.feat_dim,
            hidden_size=128,
            num_layers=1,            # 必要なら2にしてもOK
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )

        # ===== 出力層 =====
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def _extract_feats(self, x):  # (B,T,1,64,64) -> (B,T,feat_dim)
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.cnn(x)
        x = x.mean(dim=[2, 3])  # Global Average Pooling
        x = x.view(B, T, self.feat_dim)
        return x

    def forward(self, x, lengths=None, return_attention=False):
        # return_attention は互換性のため残す（Noneを返す）
        x = self._extract_feats(x)
        x = self.pre_ln(x)
        x = self.pre_drop(x)

        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        if lengths is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x = self.classifier(x)
        x = F.log_softmax(x, dim=-1)

        if return_attention:
            return x, None
        return x
