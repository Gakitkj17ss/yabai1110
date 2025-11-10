#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統一評価指標モジュール（サンプル表示機能付き）

【追加機能】
- evaluate_with_samples: 各5サンプル結果を表示
- 詳細なエラー分析
"""

import numpy as np
from typing import List, Dict, Tuple
import Levenshtein

def _safe_token(seq, idx, fallback="∅"):
    return seq[idx] if 0 <= idx < len(seq) else fallback

class UnifiedEvaluationMetrics:
    """統一評価指標クラス（サンプル表示機能付き）"""
    
    # 子音の定義
    CONSONANTS = {'k', 'g', 's', 'z', 't', 'd', 'n', 'h', 'b', 'p', 'm', 'y', 'r', 'w', 'N'}
    
    def __init__(self):
        pass
    
    @classmethod
    def is_consonant(cls, phoneme: str) -> bool:
        """音素が子音かどうか判定"""
        return phoneme in cls.CONSONANTS
    
    @staticmethod
    def ctc_collapse(sequence: List[str], blank_id: int = None) -> List[str]:
        """
        CTCのcollapse処理
        連続する同じ文字を1つにまとめる
        """
        if not sequence:
            return []
        
        collapsed = []
        prev = None
        
        for phoneme in sequence:
            if blank_id is not None and phoneme == blank_id:
                prev = None
                continue
            
            if phoneme != prev:
                collapsed.append(phoneme)
                prev = phoneme
        
        return collapsed
    
    @classmethod
    def preprocess_sequence(cls, sequence: List[str], 
                           apply_collapse: bool = True,
                           consonants_only: bool = True) -> List[str]:
        """評価前の前処理を統一"""
        if apply_collapse:
            sequence = cls.ctc_collapse(sequence)
        
        if consonants_only:
            sequence = [p for p in sequence if cls.is_consonant(p)]
        
        return sequence
    
    def calculate_per(self, predictions: List[List[str]], 
                     targets: List[List[str]],
                     apply_collapse: bool = True) -> Dict[str, float]:
        """Phoneme Error Rate (PER) 計算"""
        total_substitutions = 0
        total_deletions = 0
        total_insertions = 0
        total_phonemes = 0
        
        for pred, target in zip(predictions, targets):
            pred_processed = self.preprocess_sequence(
                pred, apply_collapse=apply_collapse, consonants_only=False
            )
            target_processed = self.preprocess_sequence(
                target, apply_collapse=apply_collapse, consonants_only=False
            )
            
            ops = Levenshtein.editops(pred_processed, target_processed)
            
            for op in ops:
                if op[0] == 'replace':
                    total_substitutions += 1
                elif op[0] == 'delete':
                    total_deletions += 1
                elif op[0] == 'insert':
                    total_insertions += 1
            
            total_phonemes += len(target_processed)
        
        total_errors = total_substitutions + total_deletions + total_insertions
        per = (total_errors / total_phonemes * 100) if total_phonemes > 0 else 0.0
        
        return {
            'per': per,
            'substitutions': total_substitutions,
            'deletions': total_deletions,
            'insertions': total_insertions,
            'total_phonemes': total_phonemes,
            'total_errors': total_errors
        }
    
    def calculate_consonant_accuracy(self, predictions: List[List[str]], 
                                    targets: List[List[str]],
                                    apply_collapse: bool = True) -> Dict[str, float]:
        """子音正解率計算（統一版）"""
        correct = 0
        total = 0
        total_errors = 0
        
        substitutions = 0
        deletions = 0
        insertions = 0
        
        for pred, target in zip(predictions, targets):
            pred_consonants = self.preprocess_sequence(
                pred, apply_collapse=apply_collapse, consonants_only=True
            )
            target_consonants = self.preprocess_sequence(
                target, apply_collapse=apply_collapse, consonants_only=True
            )
            
            if len(target_consonants) == 0:
                continue
            
            edit_dist = Levenshtein.distance(pred_consonants, target_consonants)
            
            ops = Levenshtein.editops(pred_consonants, target_consonants)
            for op in ops:
                if op[0] == 'replace':
                    substitutions += 1
                elif op[0] == 'delete':
                    deletions += 1
                elif op[0] == 'insert':
                    insertions += 1
            
            correct_in_seq = max(0, len(target_consonants) - edit_dist)
            
            correct += correct_in_seq
            total += len(target_consonants)
            total_errors += edit_dist
        
        accuracy = (correct / total * 100) if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'errors': total_errors,
            'substitutions': substitutions,
            'deletions': deletions,
            'insertions': insertions
        }
    
    def calculate_exact_match(self, predictions: List[List[str]], 
                             targets: List[List[str]],
                             apply_collapse: bool = True,
                             consonants_only: bool = False) -> Dict[str, float]:
        """完全一致率計算"""
        exact_matches = 0
        total_samples = len(predictions)
        
        for pred, target in zip(predictions, targets):
            pred_processed = self.preprocess_sequence(
                pred, apply_collapse=apply_collapse, consonants_only=consonants_only
            )
            target_processed = self.preprocess_sequence(
                target, apply_collapse=apply_collapse, consonants_only=consonants_only
            )
            
            if pred_processed == target_processed:
                exact_matches += 1
        
        exact_match_rate = (exact_matches / total_samples * 100) if total_samples > 0 else 0.0
        
        return {
            'exact_match_rate': exact_match_rate,
            'exact_matches': exact_matches,
            'total_samples': total_samples
        }
    
    def calculate_position_accuracy(self, predictions: List[List[str]], 
                                   targets: List[List[str]],
                                   apply_collapse: bool = True) -> Dict[str, float]:
        """位置別正解率計算（子音のみ）"""
        first_correct = 0
        first_total = 0
        middle_correct = 0
        middle_total = 0
        last_correct = 0
        last_total = 0
        
        for pred, target in zip(predictions, targets):
            pred_consonants = self.preprocess_sequence(
                pred, apply_collapse=apply_collapse, consonants_only=True
            )
            target_consonants = self.preprocess_sequence(
                target, apply_collapse=apply_collapse, consonants_only=True
            )
            
            if len(target_consonants) == 0:
                continue
            
            if len(pred_consonants) > 0:
                if pred_consonants[0] == target_consonants[0]:
                    first_correct += 1
            first_total += 1
            
            if len(target_consonants) >= 1 and len(pred_consonants) >= 1:
                if pred_consonants[-1] == target_consonants[-1]:
                    last_correct += 1
                last_total += 1
            
            if len(target_consonants) > 2:
                for i in range(1, len(target_consonants) - 1):
                    if i < len(pred_consonants) and pred_consonants[i] == target_consonants[i]:
                        middle_correct += 1
                    middle_total += 1
        
        first_accuracy = (first_correct / first_total * 100) if first_total > 0 else 0.0
        middle_accuracy = (middle_correct / middle_total * 100) if middle_total > 0 else 0.0
        last_accuracy = (last_correct / last_total * 100) if last_total > 0 else 0.0
        
        return {
            'first_accuracy': first_accuracy,
            'first_correct': first_correct,
            'first_total': first_total,
            'middle_accuracy': middle_accuracy,
            'middle_correct': middle_correct,
            'middle_total': middle_total,
            'last_accuracy': last_accuracy,
            'last_correct': last_correct,
            'last_total': last_total
        }
    
    def print_sample_results(self, predictions: List[List[str]], 
                       targets: List[List[str]],
                       texts: List[str] = None,
                       num_samples: int = 5,
                       apply_collapse: bool = True,
                       show_correct: bool = True,
                       show_incorrect: bool = True,
                       vowel_mode: bool = False):  # ← ★追加

        """
        サンプル結果を表示
        """
        print(f"\n{'='*70}")
        print(f"サンプル結果（最大{num_samples}件）")
        print(f"{'='*70}")
        
        correct_samples = []
        incorrect_samples = []
        
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            # ★子音フィルタをオフに（vowel_mode=True のとき）
            pred_all = self.preprocess_sequence(pred, apply_collapse, consonants_only=False)
            target_all = self.preprocess_sequence(target, apply_collapse, consonants_only=False)
            pred_cons = self.preprocess_sequence(pred, apply_collapse, consonants_only=not vowel_mode)
            target_cons = self.preprocess_sequence(target, apply_collapse, consonants_only=not vowel_mode)
            
            text = texts[i] if texts and i < len(texts) else f"Sample{i}"
            is_correct = (pred_cons == target_cons)
            
            sample_info = {
                'index': i,
                'text': text,
                'pred_all': pred_all,
                'target_all': target_all,
                'pred_cons': pred_cons,
                'target_cons': target_cons,
                'is_correct': is_correct
            }
            
            if is_correct:
                correct_samples.append(sample_info)
            else:
                incorrect_samples.append(sample_info)
        
        # 正解例を表示
        if show_correct and correct_samples:
            print(f"\n【✓ 正解例】")
            for i, sample in enumerate(correct_samples[:num_samples]):
                print(f"\n{i+1}. [{sample['text']}] (Sample {sample['index']})")
                print(f"   Target: {' '.join(sample['target_cons'])}")
                print(f"   Pred:   {' '.join(sample['pred_cons'])} ✓")
        
        # 不正解例を表示
        if show_incorrect and incorrect_samples:
            print(f"\n【✗ 不正解例】")
            for i, sample in enumerate(incorrect_samples[:num_samples]):
                pred_cons = sample['pred_cons']
                target_cons = sample['target_cons']
                
                print(f"\n{i+1}. [{sample['text']}] (Sample {sample['index']})")
                print(f"   Target: {' '.join(target_cons)}")
                print(f"   Pred:   {' '.join(pred_cons)} ✗")
                
                # エラー詳細
                if pred_cons != target_cons:
                    # editops(a, b): a -> b
                    ops = Levenshtein.editops(pred_cons, target_cons)
                    if ops:
                        print(f"   Errors:")
                        for op in ops:
                            op_type, i, j = op  # i: pred側index, j: target側index
                            if op_type == 'insert':
                                # b[j] が挿入される
                                print(f"     挿入 @{j}: '{_safe_token(target_cons, j)}'")
                            elif op_type == 'delete':
                                # a[i] が削除される
                                print(f"     削除 @{i}: '{_safe_token(pred_cons, i)}'")
                            elif op_type == 'replace':
                                # a[i] が b[j] に置換
                                print(f"     置換 @{i}: '{_safe_token(pred_cons, i)}' → '{_safe_token(target_cons, j)}'")
        
        # サマリー
        total = len(predictions)
        correct_count = len(correct_samples)
        incorrect_count = len(incorrect_samples)

        acc = (correct_count/total*100) if total > 0 else 0.0
        err = (incorrect_count/total*100) if total > 0 else 0.0

        print(f"\n{'='*70}")
        print(f"サマリー: 正解 {correct_count}/{total} ({acc:.1f}%), 不正解 {incorrect_count}/{total} ({err:.1f}%)")
        print(f"{'='*70}")

    
    def evaluate_with_samples(self, predictions: List[List[str]], 
                             targets: List[List[str]],
                             texts: List[str] = None,
                             apply_collapse: bool = True,
                             num_samples: int = 5,
                             prefix: str = "") -> Dict[str, float]:
        """
        評価指標計算 + サンプル表示
        
        Args:
            predictions: 予測音素列
            targets: 正解音素列
            texts: テキスト（カタカナなど）
            apply_collapse: collapseを適用するか
            num_samples: 表示サンプル数
            prefix: 表示プレフィックス
        
        Returns:
            dict: 全指標
        """
        # 全指標計算
        metrics = self.calculate_all_metrics(predictions, targets, apply_collapse)
        
        # メトリクス表示
        self.print_metrics(metrics, prefix)
        
        # サンプル表示
        self.print_sample_results(
            predictions, targets, texts,
            num_samples=num_samples,
            apply_collapse=apply_collapse
        )
        
        return metrics
    
    def calculate_all_metrics(self, predictions: List[List[str]], 
                             targets: List[List[str]],
                             apply_collapse: bool = True) -> Dict[str, float]:
        """全ての評価指標を統一的に計算"""
        metrics = {}
        
        # PER
        per_metrics = self.calculate_per(predictions, targets, apply_collapse)
        metrics.update({f'per_{k}': v for k, v in per_metrics.items()})
        
        # 子音正解率
        acc_metrics = self.calculate_consonant_accuracy(predictions, targets, apply_collapse)
        metrics.update({f'consonant_{k}': v for k, v in acc_metrics.items()})
        
        # 完全一致率（全音素）
        em_all_metrics = self.calculate_exact_match(
            predictions, targets, apply_collapse=apply_collapse, consonants_only=False
        )
        metrics.update({f'exact_match_all_{k}': v for k, v in em_all_metrics.items()})
        
        # 完全一致率（子音のみ）
        em_consonant_metrics = self.calculate_exact_match(
            predictions, targets, apply_collapse=apply_collapse, consonants_only=True
        )
        metrics.update({f'exact_match_consonant_{k}': v for k, v in em_consonant_metrics.items()})
        
        # デフォルト
        metrics['exact_match_exact_match_rate'] = em_all_metrics['exact_match_rate']
        metrics['exact_match_exact_matches'] = em_all_metrics['exact_matches']
        metrics['exact_match_total_samples'] = em_all_metrics['total_samples']
        
        # 位置別正解率
        pos_metrics = self.calculate_position_accuracy(predictions, targets, apply_collapse)
        metrics.update({f'position_{k}': v for k, v in pos_metrics.items()})
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """評価指標を見やすく表示"""
        print(f"\n{'='*70}")
        print(f"{prefix} 統一評価指標（CTC collapse適用）" if prefix else "統一評価指標（CTC collapse適用）")
        print(f"{'='*70}")
        
        # PER
        if 'per_per' in metrics:
            print(f"\n【音素誤り率 (PER)】")
            print(f"  PER: {metrics['per_per']:.2f}%")
            print(f"  置換: {metrics.get('per_substitutions', 0):,}")
            print(f"  削除: {metrics.get('per_deletions', 0):,}")
            print(f"  挿入: {metrics.get('per_insertions', 0):,}")
            print(f"  総音素数: {metrics.get('per_total_phonemes', 0):,}")
        
        # 子音正解率
        if 'consonant_accuracy' in metrics:
            print(f"\n【子音正解率（編集距離ベース）】")
            print(f"  正解率: {metrics['consonant_accuracy']:.2f}%")
            print(f"  正解: {metrics.get('consonant_correct', 0):,}/{metrics.get('consonant_total', 0):,}")
            print(f"  エラー: {metrics.get('consonant_errors', 0):,}")
            print(f"    - 置換: {metrics.get('consonant_substitutions', 0):,}")
            print(f"    - 削除: {metrics.get('consonant_deletions', 0):,}")
            print(f"    - 挿入: {metrics.get('consonant_insertions', 0):,}")
        
        # 完全一致率
        if 'exact_match_all_exact_match_rate' in metrics:
            print(f"\n【完全一致率】")
            print(f"  全音素: {metrics['exact_match_all_exact_match_rate']:.2f}%")
            print(f"    一致数: {metrics.get('exact_match_all_exact_matches', 0):,}/{metrics.get('exact_match_all_total_samples', 0):,}")
        
        if 'exact_match_consonant_exact_match_rate' in metrics:
            print(f"  子音のみ: {metrics['exact_match_consonant_exact_match_rate']:.2f}%")
            print(f"    一致数: {metrics.get('exact_match_consonant_exact_matches', 0):,}/{metrics.get('exact_match_consonant_total_samples', 0):,}")
        
        # 位置別正解率
        if 'position_first_accuracy' in metrics:
            print(f"\n【位置別正解率（子音のみ）】")
            print(f"  最初の子音: {metrics['position_first_accuracy']:.2f}%  ({metrics.get('position_first_correct', 0):,}/{metrics.get('position_first_total', 0):,})")
            print(f"  中間の子音: {metrics['position_middle_accuracy']:.2f}%  ({metrics.get('position_middle_correct', 0):,}/{metrics.get('position_middle_total', 0):,})")
            print(f"  最後の子音: {metrics['position_last_accuracy']:.2f}%  ({metrics.get('position_last_correct', 0):,}/{metrics.get('position_last_total', 0):,})")
        
        print(f"{'='*70}")


if __name__ == "__main__":
    # テスト
    evaluator = UnifiedEvaluationMetrics()
    
    predictions = [
        ['k', 'k', 'o', 'n', 'n'],
        ['s', 'u', 't'],
        ['h', 'a'],
        ['t', 't', 'a', 'k', 'k', 'a']
    ]
    
    targets = [
        ['k', 'o', 'n'],
        ['s', 'u', 't'],
        ['h', 'a'],
        ['t', 'a', 'k', 'a']
    ]
    
    texts = ['コン', 'スト', 'ハ', 'タカ']
    
    # サンプル表示付き評価
    metrics = evaluator.evaluate_with_samples(
        predictions, targets, texts,
        apply_collapse=True,
        num_samples=5,
        prefix="Test"
    )