#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pattern A 改善版: CNN後にAttention（修正版）

【修正内容】
- forwardメソッドで常にattention_weightsを保存
- 可視化時に確実にAttention weightsを取得できるように
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameWiseAttention(nn.Module):
    """改善版 Frame-wise Attention"""
    
    def __init__(self, hidden_size, attention_type='softmax', temperature=0.1):
        super().__init__()
        
        self.attention_type = attention_type
        self.temperature = temperature
        
        # より深いAttention計算（学習能力を上げる）
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),  # 大きめに
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 1)
        )
        
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.attention_net.modules():
            if isinstance(module, nn.Linear):
                
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    
                    nn.init.normal_(module.bias, mean=0, std=0.1)
    
    def forward(self, x, return_weights=False):
        """
        Args:
            x: (B, T, hidden_size) - CNN出力
        """
        # Attention計算
        scores = self.attention_net(x) / self.temperature  # (B, T, 1)
        
        if self.attention_type == 'sigmoid':
            weights = torch.sigmoid(scores)
        else:
            weights = F.softmax(scores, dim=1)
        
        # 重み付き出力
        output = x * weights  # (B,T,H)
        
        if return_weights:
            return output, weights.squeeze(-1)
        return output


class ConsonantModelPatternA_Improved(nn.Module):
    """
    Pattern A 改善版: CNN → Frame Attention → LSTM
    
    【修正】
    - Attention weightsを常に保存（可視化用）
    """
    
    def __init__(self, num_classes=16, dropout_rate=0.3,
                 attention_type='sigmoid', temperature=1.0,
                 dual_attention: bool = False):
        super().__init__()
        
        print("\n" + "="*60)
        print(f"Pattern A (Improved): CNN → Frame Attention → LSTM")
        print("="*60)
        
        # CNN
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # ★ Frame-wise Attention (CNN直後)
        self.frame_attention = FrameWiseAttention(
            hidden_size=256,
            attention_type='sigmoid',
            temperature=temperature
    
        )
        
        # LayerNorm（Attention後）
        self.attention_norm = nn.LayerNorm(256)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=192,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate
        )
        

        # ★ Temporal Attention (LSTM後) — dual_attention=True のとき有効
        self.dual_attention = bool(dual_attention)
        if self.dual_attention:
            self.temporal_attention = FrameWiseAttention(
                hidden_size=384,              # BiLSTM出力次元
                attention_type='softmax',
                temperature=temperature
            )
            self.temporal_norm = nn.LayerNorm(384)
        # 分類層
        self.classifier = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        self.log_softmax = nn.LogSoftmax(dim=2)
        
        # Attention weightsを保存（可視化用）
        self.attention_weights = {'frame': None, 'temporal': None}
        
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"\n✓ Model created (Pattern A - 修正版)")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Attention type: {attention_type}")
        print(f"  - Temperature: {temperature}")
        print(f"  - Attention位置: CNN直後（Frame）"
              f"{' + LSTM後（Temporal）' if self.dual_attention else ''}")
        print(f"  - Attention weights保存: frame/temporal")
        print("="*60 + "\n")
    
    def forward(self, x, lengths=None, return_attention=False):
        B, T, C, H, W = x.size()
        
        # CNN
        x = x.view(-1, C, H, W)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = x.view(B, T, -1)
        
        x, frame_weights = self.frame_attention(x, return_weights=True)  # (B,T,256) / (B,T)
        self.attention_weights['frame'] = frame_weights.detach()
        
        x = self.attention_norm(x)
        
        # LSTM
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        x, _ = self.lstm(x)
        
        if lengths is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        
        # ★ Temporal Attention（必要時）
        temporal_weights = None
        if self.dual_attention:
            x, temporal_weights = self.temporal_attention(x, return_weights=True)  # (B,T,384), (B,T)
            self.attention_weights['temporal'] = temporal_weights.detach()
            x = self.temporal_norm(x)
        
        # 分類
        x = self.classifier(x)
        x = self.log_softmax(x)
        
        if return_attention:
            # 可視化互換性のため、Temporalがあればそれを返す（時間整合のため）
            att = temporal_weights if temporal_weights is not None else frame_weights
            return x, att
        return x


def create_improved_pattern_a_model(num_classes=16, dropout_rate=0.3,
                                    attention_type='sigmoid', temperature=1.0,
                                    dual_attention: bool = False):
    """
    Pattern A改善版を作成（修正版）
    
    【修正】
    - Attention weightsを常に保存して可視化できるように
    """
    return ConsonantModelPatternA_Improved(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        attention_type=attention_type,
        temperature=temperature,
        dual_attention=dual_attention
    )


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Pattern A (Improved) Model Test - 修正版")
    print("="*60 + "\n")
    
    batch_size = 2
    frames = 50
    dummy_input = torch.randn(batch_size, frames, 1, 64, 64)
    lengths = torch.tensor([50, 45])
    
    model = create_improved_pattern_a_model(
        num_classes=16,
        dropout_rate=0.3,
        attention_type='sigmoid',
        temperature=1.0
    )
    
    # テスト1: 通常のforward
    print("テスト1: 通常のforward")
    output = model(dummy_input, lengths, return_attention=False)
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Attention weights saved: {model.attention_weights is not None}")
    if model.attention_weights is not None:
        print(f"✓ Attention shape: {model.attention_weights.shape}")
        print(f"✓ Attention stats - Mean: {model.attention_weights[0].mean():.4f}, Std: {model.attention_weights[0].std():.4f}")
    
    # テスト2: return_attention=True
    print("\nテスト2: return_attention=True")
    output, attention_weights = model(dummy_input, lengths, return_attention=True)
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Attention weights shape: {attention_weights.shape}")
    print(f"✓ First sample weights: {attention_weights[0, :10]}")
    
    print("\n" + "="*60)
    print("✓ Attention weights可視化準備完了！")
    print("="*60)