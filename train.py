#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
読唇術モデル訓練モジュール（簡潔・統一版・NaN対策込み）
- Pattern B: CNN → LSTM → Temporal Attention を想定
- vowel/consonant 共通: 途中評価も最終評価も CTCデコード + UnifiedEvaluationMetrics を使用
- 数値安定:
  * モデルが log確率 or ロジットを返すかを自動判定（returns_log_probs）
  * 空ラベル/無効長サンプルの除外
  * CTCの長さ整合（input_length のクランプ、target_length の下限1）
  * logp の finite 化
  * 勾配クリップ
"""

import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CTCLoss
from torch.cuda.amp import autocast, GradScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matrics_undefined import UnifiedEvaluationMetrics
from utils_pattern_b import ctc_greedy_decode, ids_to_phonemes
try:
    from utils_pattern_b import ctc_beam_search_decode
except Exception:
    ctc_beam_search_decode = None


# ----------------------------
# PER (Levenshtein) helpers
# ----------------------------
def _levenshtein_sdi(ref, hyp):
    """ref/hyp: List[str] -> (S,D,I)"""
    n, m = len(ref), len(hyp)
    dp = [[0]*(m+1) for _ in range(n+1)]
    bt = [[0]*(m+1) for _ in range(n+1)]  # 0:diag, 1:up(del), 2:left(ins)
    for i in range(1, n+1):
        dp[i][0] = i; bt[i][0] = 1
    for j in range(1, m+1):
        dp[0][j] = j; bt[0][j] = 2
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            a = dp[i-1][j-1] + cost
            b = dp[i-1][j] + 1
            c = dp[i][j-1] + 1
            if a <= b and a <= c:
                dp[i][j] = a; bt[i][j] = 0
            elif b <= c:
                dp[i][j] = b; bt[i][j] = 1
            else:
                dp[i][j] = c; bt[i][j] = 2
    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        code = bt[i][j]
        if i > 0 and j > 0 and code == 0:
            if ref[i-1] != hyp[j-1]:
                S += 1
            i -= 1; j -= 1
        elif i > 0 and (j == 0 or code == 1):
            D += 1; i -= 1
        else:
            I += 1; j -= 1
    return S, D, I

def _lev_sdi_counts(ref, hyp):
    S, D, I = _levenshtein_sdi(ref, hyp)
    return S, D, I, len(ref), len(hyp)

def _sequence_per_percent(ref, hyp):
    """PER[%]（参照長基準）"""
    S, D, I, Nref, _ = _lev_sdi_counts(ref, hyp)
    return 100.0 * (S + D + I) / max(1, Nref)

def _sequence_per_percent_display(ref, hyp):
    """PER[%]（表示用・0–100%）：分母=max(len(ref),len(hyp))"""
    S, D, I, Nref, Nhyp = _lev_sdi_counts(ref, hyp)
    return 100.0 * (S + D + I) / max(1, max(Nref, Nhyp))


# ----------------------------
# Trainer
# ----------------------------
class LipReadingTrainer:
    """読唇術モデル訓練クラス（CTC + 統一評価 + NaN対策）"""

    def __init__(self, model, phoneme_encoder, device='cuda', save_dir='checkpoints',
                 early_stopping_metric='val_loss', min_delta=0.0,
                 decode_beam_width: int = 5):
        self.model = model.to(device)
        self.phoneme_encoder = phoneme_encoder
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # モデルが log確率を返すなら True（例: CompactVowelLipReader）
        self.model_returns_log_probs = getattr(self.model, "returns_log_probs", False)

        # 損失関数（CTC）
        self.criterion = CTCLoss(blank=phoneme_encoder.blank_id, zero_infinity=True)

        # 最適化・スケジューラ・AMP
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler()

        # 統一評価器
        self.evaluator = UnifiedEvaluationMetrics()

        # 訓練履歴
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'ctc_compressed_acc': [],       # consonant_accuracy(%) を 0-1 正規化
            'ctc_edit_distance': [],
            'ctc_normalized_distance': [],  # per_per(%) を 0-1 正規化
            'per_percent': [],
            'lr': [],
            'epoch_time': [],
            'cumulative_time': []
        }

        # 監視設定（EarlyStopping）
        self.min_delta = float(min_delta)
        self.early_stopping_metric = early_stopping_metric  # 'val_loss' / 'per_per' / 'consonant_accuracy' / 'normalized_distance'
        self._monitor_mode = 'min' if early_stopping_metric in ('val_loss', 'per_per', 'normalized_distance') else 'max'
        self._best_monitor = float('inf') if self._monitor_mode == 'min' else -float('inf')
        self.best_val_loss = float('inf')
        self._patience_counter = 0

        # デコード
        self.decode_beam_width = int(decode_beam_width)

    # ----------------------------
    # セットアップ
    # ----------------------------
    def setup_optimizer(self, optimizer_type='adamw', lr=3e-4, weight_decay=1e-2):
        ot = optimizer_type.lower()
        if ot == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif ot == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif ot == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        print(f"✓ Optimizer: {optimizer_type} (lr={lr})")

    def setup_scheduler(self, scheduler_type='cosine', **kwargs):
        st = scheduler_type.lower() if scheduler_type else ''
        if st == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min',
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 10),
                verbose=True
            )
        elif st == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=kwargs.get('T_max', 50),
                eta_min=kwargs.get('eta_min', 1e-6)
            )
        elif st == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1)
            )
        else:
            self.scheduler = None
        print(f"✓ Scheduler: {scheduler_type if scheduler_type else 'None'}")

    # ----------------------------
    # NaN/長さ対策: バッチ整形
    # ----------------------------
    def _filter_valid_batch(self, batch):
        """
        tlen==0 や ilen==0 を除外し、CTC用に target を再構成。
        返り値: (x, y, ilen, tlen) or None
        """
        x = batch['video'].to(self.device)        # (B,T,1,64,64)
        y_cat = batch['target']                   # (sum_L,)
        ilen = batch['input_length']              # (B,)
        tlen = batch['target_length']             # (B,)

        valid = (tlen > 0) & (ilen > 0)
        if valid.sum() == 0:
            return None

        x = x[valid]
        ilen = ilen[valid]

        # target連結を有効サンプルだけ再構成
        new_targets, new_tlens = [], []
        off = 0
        for i in range(len(tlen)):
            tl = int(tlen[i])
            seg = y_cat[off:off+tl]
            if valid[i] and tl > 0:
                new_targets.extend(seg.tolist())
                new_tlens.append(tl)
            off += tl

        y = torch.tensor(new_targets, dtype=torch.long, device=self.device)
        tlen = torch.tensor(new_tlens, dtype=torch.long, device=self.device)
        return x, y, ilen, tlen

    # ----------------------------
    # 学習・検証
    # ----------------------------
    def train_epoch(self, train_loader):
        """1エポック学習（NaN対策込み）"""
        self.model.train()
        total_loss, num_batches = 0.0, 0

        for batch in train_loader:
            flt = self._filter_valid_batch(batch)
            if flt is None:
                continue
            videos, targets, input_lengths, target_lengths = flt

            self.optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = self.model(videos)  # (B,T,C) or logp
                log_probs = (outputs if self.model_returns_log_probs
                             else torch.log_softmax(outputs, dim=-1)).permute(1, 0, 2)  # (T,B,C)
                # 数値安定
                log_probs = torch.where(torch.isfinite(log_probs), log_probs, torch.zeros_like(log_probs))

                # 長さ整合（CTC要件）
                Tcur = log_probs.size(0)
                input_lengths = torch.clamp(input_lengths, max=Tcur)
                target_lengths = torch.clamp(target_lengths, min=1)

                loss = self.criterion(log_probs, targets, input_lengths, target_lengths)

            # NaN/inf スキップ
            if not torch.isfinite(loss):
                continue

            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.item())
            num_batches += 1

        return total_loss / max(1, num_batches)

    def validate(self, val_loader):
        """検証（途中評価）— 統一評価器を使用"""
        self.model.eval()
        total_loss = 0.0
        all_predictions, all_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                flt = self._filter_valid_batch(batch)
                if flt is None:
                    continue
                x, y, ilen, tlen = flt

                out = self.model(x)  # (B,T,C) or logp
                logp = (out if self.model_returns_log_probs
                        else torch.log_softmax(out, dim=-1)).permute(1, 0, 2)
                logp = torch.where(torch.isfinite(logp), logp, torch.zeros_like(logp))

                Tcur = logp.size(0)
                ilen = torch.clamp(ilen, max=Tcur)
                tlen = torch.clamp(tlen, min=1)

                # CTC loss
                loss = self.criterion(logp, y, ilen, tlen)
                total_loss += float(loss.item())

                # ---- decode（ビーム幅で切替 & 安全フォールバック）----
                use_beam = (self.decode_beam_width is not None
                            and self.decode_beam_width >= 2
                            and ctc_beam_search_decode is not None)
                if use_beam:
                    pred_ids = ctc_beam_search_decode(
                        logp, self.phoneme_encoder.blank_id, ilen, beam_width=self.decode_beam_width
                    )
                else:
                    pred_ids = ctc_greedy_decode(logp, self.phoneme_encoder.blank_id, ilen)

                preds = ids_to_phonemes(pred_ids, self.phoneme_encoder)

                # split targets（文字列へ）
                off = 0
                for tl in tlen.tolist():
                    ids = y[off:off + tl].detach().cpu().tolist()
                    all_targets.append(self.phoneme_encoder.decode_phonemes(ids))
                    off += tl
                all_predictions.extend(preds)

        # ========= サンプル抽出 & グローバル要約 =========
        sample_results = []
        global_S = global_D = global_I = 0
        len_refs, len_hyps = [], []

        for p, t in zip(all_predictions, all_targets):
            S, D, I, Nref, Nhyp = _lev_sdi_counts(t, p)
            per_ref  = 100.0 * (S + D + I) / max(1, Nref)
            per_disp = 100.0 * (S + D + I) / max(1, max(Nref, Nhyp))
            if per_ref > 0.0:
                sample_results.append({
                    'per_ref': round(per_ref, 2),
                    'per_display': round(per_disp, 2),
                    'S': S, 'D': D, 'I': I,
                    'len_ref': Nref, 'len_hyp': Nhyp,
                    'predicted': p, 'target': t
                })
            global_S += S; global_D += D; global_I += I
            len_refs.append(Nref); len_hyps.append(Nhyp)

        sample_results.sort(key=lambda x: -x['per_ref'])

        # メトリクス
        metrics = self.evaluator.calculate_all_metrics(all_predictions, all_targets)
        metrics['sample_results'] = sample_results[:10]
        metrics['global_S'] = global_S
        metrics['global_D'] = global_D
        metrics['global_I'] = global_I
        metrics['avg_len_ref'] = float(np.mean(len_refs)) if len_refs else 0.0
        metrics['avg_len_hyp'] = float(np.mean(len_hyps)) if len_hyps else 0.0

        # 長さビン別 PER（ref 基準）
        bins = [(1,1),(2,3),(4,6),(7,10),(11,20),(21,999)]
        bin_map = {}
        for lo, hi in bins:
            S=D=I=N=0
            for p, t in zip(all_predictions, all_targets):
                if lo <= len(t) <= hi:
                    s,d,i = _levenshtein_sdi(t, p)
                    S += s; D += d; I += i; N += len(t)
            per_bin = 100.0*(S+D+I)/max(1,N)
            bin_map[f"{lo}-{hi}"] = per_bin
        metrics['length_bucket_per'] = bin_map

        avg_val_loss = total_loss / max(1, len(val_loader))
        return avg_val_loss, metrics

    # ----------------------------
    # ループ
    # ----------------------------
    def train(self, train_loader, val_loader, epochs=100, early_stopping_patience=20):
        """学習ループ（PERを含む統一メトリクスで進捗管理）"""
        print(f"\n{'=' * 70}\n訓練開始: {epochs}エポック\n{'=' * 70}")

        start_time = time.time()
        first_epoch = True

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.validate(val_loader)

            # スケジューラ更新
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # 履歴
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start
            cumulative_time = time.time() - start_time

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            consonant_acc = val_metrics.get('consonant_accuracy', 0.0) / 100.0
            self.history['ctc_compressed_acc'].append(consonant_acc)

            edit_dist = val_metrics.get('per_total_errors', val_metrics.get('edit_distance', 0.0))
            self.history['ctc_edit_distance'].append(edit_dist)

            per_norm = val_metrics.get('per_per', val_metrics.get('per', 0.0)) / 100.0
            self.history['ctc_normalized_distance'].append(per_norm)
            self.history['per_percent'].append(val_metrics.get('per_per', 0.0))

            self.history['lr'].append(current_lr)
            self.history['epoch_time'].append(epoch_time)
            self.history['cumulative_time'].append(cumulative_time)

            # ログ
            print(
                f"Epoch {epoch}/{epochs} | "
                f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                f"Acc: {val_metrics.get('consonant_accuracy', 0.0):.2f}% | "
                f"PER: {val_metrics.get('per_per', val_metrics.get('per', 0.0)):.2f}% | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s",
                end=""
            )

            # 詳細（5エポック毎）
            if epoch % 5 == 0 or epoch == 1:
                print("\n  詳細:")
                print(f"    Train Loss: {train_loss:.4f}")
                print(f"    Val   Loss: {val_loss:.4f}")
                print(f"    Consonant Acc: {val_metrics.get('consonant_accuracy', 0.0):.2f}%")
                print(f"    PER: {val_metrics.get('per_per', val_metrics.get('per', 0.0)):.2f}%")
                print(f"    Edit Distance: {edit_dist:.2f}")
                if first_epoch:
                    print(f"    [DEBUG] metrics keys: {list(val_metrics.keys())}")
                    first_epoch = False
            else:
                print()

            # ===== EarlyStopping: メトリクス選択 =====
            if self.early_stopping_metric == 'val_loss':
                monitor = val_loss
            elif self.early_stopping_metric == 'per_per':
                monitor = val_metrics.get('per_per', float('inf'))  # 低い方が良い
            elif self.early_stopping_metric == 'consonant_accuracy':
                monitor = val_metrics.get('consonant_accuracy', float('-inf'))  # 高い方が良い
            else:  # 'normalized_distance' など
                monitor = self.history['ctc_normalized_distance'][-1]

            # 改善判定（min_delta考慮）
            improved = (monitor < (self._best_monitor - self.min_delta)) if \
                (self._monitor_mode == 'min') else (monitor > (self._best_monitor + self.min_delta))

            if improved:
                self._best_monitor = monitor
                self._patience_counter = 0
                # 監視対象ベストのチェックポイント（分かりやすく別名で保存）
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'monitor': self.early_stopping_metric,
                    'best_monitor_value': self._best_monitor,
                    'history': self.history,
                    'phoneme_encoder': self.phoneme_encoder,
                }, os.path.join(self.save_dir, f'best_{self.early_stopping_metric}.pth'))
                print(f"  ← ✓ Best {self.early_stopping_metric} updated: {self._best_monitor:.4f}")
            else:
                self._patience_counter += 1

            # 参考として val_loss ベストも別枠で保存（patienceはリセットしない）
            if val_loss < self.best_val_loss - 0.0:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'best_val_loss': self.best_val_loss,
                    'history': self.history,
                    'phoneme_encoder': self.phoneme_encoder,
                }, os.path.join(self.save_dir, 'best_val_loss.pth'))
                print("  ← ✓ Best val_loss updated")

            # 定期保存
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)

            # 早期終了
            if self._patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement in {self.early_stopping_metric} for {early_stopping_patience} epochs)")
                break

        total_time = time.time() - start_time
        print(f"\n{'=' * 70}\n訓練完了\n  総時間: {total_time/3600:.2f}時間\n  最良Val Loss: {self.best_val_loss:.4f}\n{'=' * 70}")
        return self.history

    # ----------------------------
    # 保存・復帰・可視化
    # ----------------------------
    def save_checkpoint(self, epoch, is_best=False):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'phoneme_encoder': self.phoneme_encoder,
        }
        fname = 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch}.pth'
        torch.save(ckpt, os.path.join(self.save_dir, fname))

    def load_checkpoint(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if self.scheduler and ckpt.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        self.history = ckpt.get('history', self.history)
        print(f"✓ Checkpoint loaded: Epoch {ckpt.get('epoch', '?')}")
        return ckpt.get('epoch', 0)

    def plot_history(self, save_path=None):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        epochs = range(1, len(self.history['train_loss']) + 1)

        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Loss Curve', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

        # Accuracy (consonant_accuracy を 0-1 で)
        axes[0, 1].plot(epochs, self.history['ctc_compressed_acc'], linewidth=2)
        axes[0, 1].set_title('Consonant Accuracy'); axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_ylim([0, 1]); axes[0, 1].grid(True, alpha=0.3)

        # Edit Distance
        axes[0, 2].plot(epochs, self.history['ctc_edit_distance'], linewidth=2)
        axes[0, 2].set_title('Edit Distance'); axes[0, 2].set_xlabel('Epoch'); axes[0, 2].set_ylabel('Distance'); axes[0, 2].grid(True, alpha=0.3)

        # Normalized PER
        axes[1, 0].plot(epochs, self.history['ctc_normalized_distance'], linewidth=2)
        axes[1, 0].set_title('PER (Normalized)'); axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('PER')
        axes[1, 0].set_ylim([0, 1]); axes[1, 0].grid(True, alpha=0.3)

        # PER [%] 追加（論文図表向け）
        axes[1, 1].plot(epochs, self.history['per_percent'], linewidth=2)
        axes[1, 1].set_title('PER [%]'); axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('PER [%]')
        axes[1, 1].grid(True, alpha=0.3)

        # Learning Rate
        axes[1, 2].plot(epochs, self.history['lr'], linewidth=2)
        axes[1, 2].set_title('Learning Rate'); axes[1, 2].set_xlabel('Epoch'); axes[1, 2].set_ylabel('LR')
        axes[1, 2].set_yscale('log'); axes[1, 2].grid(True, alpha=0.3)

        # Epoch Time（右上に平均線）
        axes[0, 2].clear()
        axes[0, 2].plot(epochs, self.history['epoch_time'], linewidth=2)
        avg_t = np.mean(self.history['epoch_time']) if self.history['epoch_time'] else 0.0
        axes[0, 2].axhline(y=avg_t, color='red', linestyle='--', label=f'Avg: {avg_t:.1f}s')
        axes[0, 2].set_title('Time per Epoch'); axes[0, 2].set_xlabel('Epoch'); axes[0, 2].set_ylabel('Time (s)')
        axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)

        plt.suptitle('Training Progress', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"✓ Training history saved: {save_path}")
        plt.close()


# ----------------------------
# 最終評価（validate と同じ“正しい経路”）
# ----------------------------
def evaluate_model(model, data_loader, phoneme_encoder, device,
                   show_samples=False, num_samples=5, beam_width=5):
    """最終評価。途中評価と同一経路（CTCデコード＋統一メトリクス）"""
    model.eval()
    evaluator = UnifiedEvaluationMetrics()

    all_predictions, all_targets = [], []
    with torch.no_grad():
        for batch in data_loader:
            # data_loader 側で空ラベル除外済み前提
            x = batch['video'].to(device)
            y = batch['target']
            ilen = batch['input_length']
            tlen = batch['target_length']

            out = model(x)
            returns_log = getattr(model, "returns_log_probs", False)
            logp = (out if returns_log else torch.log_softmax(out, dim=-1)).permute(1, 0, 2)  # (T,B,C)
            logp = torch.where(torch.isfinite(logp), logp, torch.zeros_like(logp))

            # CTC要件の長さ整合
            Tcur = logp.size(0)
            ilen = torch.clamp(ilen, max=Tcur)
            tlen = torch.clamp(tlen, min=1)

            # ---- decode (beam/greeedy) ----
            if beam_width and (beam_width >= 2) and (ctc_beam_search_decode is not None):
                pred_ids = ctc_beam_search_decode(
                    logp, phoneme_encoder.blank_id, ilen, beam_width=beam_width
                )
            else:
                pred_ids = ctc_greedy_decode(logp, phoneme_encoder.blank_id, ilen)

            preds = ids_to_phonemes(pred_ids, phoneme_encoder)

            # targets（文字列化）
            off = 0
            for tl in tlen.tolist():
                ids = y[off:off + tl].cpu().tolist()
                all_targets.append(phoneme_encoder.decode_phonemes(ids))
                off += tl
            all_predictions.extend(preds)

    # ========= サンプル抽出 & グローバル要約 =========
    sample_results = []
    global_S = global_D = global_I = 0
    len_refs, len_hyps = [], []

    for p, t in zip(all_predictions, all_targets):
        S, D, I, Nref, Nhyp = _lev_sdi_counts(t, p)
        per_ref  = 100.0 * (S + D + I) / max(1, Nref)
        per_disp = 100.0 * (S + D + I) / max(1, max(Nref, Nhyp))
        if per_ref > 0.0:
            sample_results.append({
                'per_ref': round(per_ref, 2),
                'per_display': round(per_disp, 2),
                'S': S, 'D': D, 'I': I,
                'len_ref': Nref, 'len_hyp': Nhyp,
                'predicted': p, 'target': t
            })
        global_S += S; global_D += D; global_I += I
        len_refs.append(Nref); len_hyps.append(Nhyp)

    sample_results.sort(key=lambda x: -x['per_ref'])

    metrics = evaluator.calculate_all_metrics(all_predictions, all_targets)

    # 追加要約を metrics に格納
    metrics['sample_results'] = sample_results[:10]
    metrics['global_S'] = global_S
    metrics['global_D'] = global_D
    metrics['global_I'] = global_I
    metrics['avg_len_ref'] = float(np.mean(len_refs)) if len_refs else 0.0
    metrics['avg_len_hyp'] = float(np.mean(len_hyps)) if len_hyps else 0.0

    # 長さビン別 PER（ref 基準）
    bins = [(1,1),(2,3),(4,6),(7,10),(11,20),(21,999)]
    bin_map = {}
    for lo, hi in bins:
        S=D=I=N=0
        for p, t in zip(all_predictions, all_targets):
            if lo <= len(t) <= hi:
                s,d,i = _levenshtein_sdi(t, p)
                S += s; D += d; I += i; N += len(t)
        per_bin = 100.0*(S+D+I)/max(1,N)
        bin_map[f"{lo}-{hi}"] = per_bin
    metrics['length_bucket_per'] = bin_map

    # ====== 表示 ======
    if show_samples:
        print("\n[Top 10 Incorrect Samples by PER]")
        for i, s in enumerate(sample_results[:10], 1):
            print(
                f"{i:2d}) PER(ref)={s['per_ref']:>6.2f}%  PER(disp)={s['per_display']:>6.2f}%  "
                f"[S/D/I]={s['S']}/{s['D']}/{s['I']}  "
                f"len(ref/hyp)={s['len_ref']}/{s['len_hyp']} | "
                f"pred: {' '.join(s['predicted'][:20])} || tgt: {' '.join(s['target'][:20])}"
            )

        # 追加の要約表示
        total_ref = max(1, sum(len_refs))
        print("\n--- Global Error Breakdown ---")
        print(f"  S/D/I: {global_S}/{global_D}/{global_I}  "
              f"(rates: {100*global_S/total_ref:.2f}% / {100*global_D/total_ref:.2f}% / {100*global_I/total_ref:.2f}%)")
        print(f"  Avg length (ref/hyp): {np.mean(len_refs) if len_refs else 0.0:.2f} / {np.mean(len_hyps) if len_hyps else 0.0:.2f}")

        if bin_map:
            print("\n--- PER by Length (ref) ---")
            for k in ["1-1","2-3","4-6","7-10","11-20","21-999"]:
                if k in bin_map:
                    print(f"  L={k:>6}: PER={bin_map[k]:5.2f}%")

        # 母音別 正解率 & 混同行列（a,i,u,e,o）
        vowels = ['a','i','u','e','o']; idx = {v:i for i,v in enumerate(vowels)}
        cm = np.zeros((5,5), dtype=int)
        corr = np.zeros(5, dtype=int); total = np.zeros(5, dtype=int)
        for p, t in zip(all_predictions, all_targets):
            L = min(len(p), len(t))
            for k in range(L):
                if t[k] in idx and p[k] in idx:
                    i = idx[t[k]]; j = idx[p[k]]
                    cm[i,j] += 1
                    total[i] += 1
                    if i==j: corr[i] += 1
        if total.sum() > 0:
            print("\n--- Vowel-wise Acc & Confusion (diag=acc) ---")
            for i, v in enumerate(vowels):
                acc = 100.0*corr[i]/max(1,total[i])
                top_conf = np.argsort(-cm[i])[:3]
                conf_str = ", ".join([f"{vowels[j]}:{cm[i,j]}" for j in top_conf])
                print(f"  {v}: Acc={acc:5.2f}%  | conf={conf_str}")

    return {
        'ctc_compressed_acc': metrics.get('exact_match_consonant_exact_match_rate', 0.0),
        'edit_distance': metrics.get('per_total_errors', 0.0),
        'normalized_distance': (metrics.get('per_per', 0.0) / 100.0) if metrics.get('per_per') is not None else 0.0,
        'avg_per': metrics.get('per_per', 0.0),
        'raw': {
            **metrics,
            'predictions': all_predictions,
            'targets': all_targets,
            'sample_results': sample_results[:10],
        },
    }


if __name__ == "__main__":
    print("train.py (統一・NaN対策版) loaded")
