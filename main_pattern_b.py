#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
読唇術プロジェクト メインスクリプト (Pattern B対応・評価指標統一版)
- Pattern B: CNN → LSTM → Temporal Attention
- Sigmoid/Softmax簡単切り替え
- vowelモデルは Attention なし(NoAttn)版に差し替え可能
"""

import os
import argparse
import torch
from pathlib import Path
import numpy as np

# モジュールインポート
print("モジュール読み込み中...")
from phoneme_analysis_unified import analyze_phonemes_unified
from matrics_undefined import UnifiedEvaluationMetrics
from dataset import create_dataloaders
from train import LipReadingTrainer, evaluate_model
from utils_pattern_b import (
    Config, set_seed, setup_logging, check_data_paths,
    check_gpu_availability, create_directories, save_results,
    print_model_info, MetricsCalculator, build_loaders_from_config, sync_num_classes_with_encoder
)
# ===== モード非依存の統一サマリー出力 =====
def _compute_first_last_accuracy(pred_seqs, tgt_seqs):
    n = 0; first_ok = 0; last_ok = 0
    for p, t in zip(pred_seqs, tgt_seqs):
        if len(t) == 0:
            continue
        n += 1
        if len(p) > 0 and p[0] == t[0]:
            first_ok += 1
        if len(p) > 0 and p[-1] == t[-1]:
            last_ok += 1
    if n == 0:
        return 0.0, 0.0
    return 100.0 * first_ok / n, 100.0 * last_ok / n

def print_unified_summary(final_raw, encoder):
    """
    final_raw: evaluate_modelの戻り値のraw部（predictions/targets必須）
    表示を '子音側のフォーマット' に統一して出力し、mmにも統一キーを追加して返す
    """
    from matrics_undefined import UnifiedEvaluationMetrics
    evalr = UnifiedEvaluationMetrics()

    preds = final_raw.get('predictions', [])
    tgts  = final_raw.get('targets', [])

    # 1) PER
    per = final_raw.get('per_per', final_raw.get('PER', None))
    if per is None:
        per = evalr.sequence_per(preds, tgts)  # ％を返す実装想定

    # 2) 完全一致率（系列）
    exact = final_raw.get('exact_match_consonant_exact_match_rate',
                          final_raw.get('exact_match_vowel_rate', None))
    if exact is None:
        # collapse済み前提のpreds/tgtsならそのまま、未collapseなら内部でcollapseする実装に依存
        exact = evalr.sequence_exact_match_rate(preds, tgts)  # ％を返す実装想定

    # 3) 最初/最後トークン正解率（モード非依存）
    # 文字列リストが前提のはずだが、もしintならencoderで変換
    if preds and preds[0] and isinstance(preds[0][0], int):
        preds_tok = [encoder.ids_to_symbols(x) for x in preds]
        tgts_tok  = [encoder.ids_to_symbols(x) for x in tgts]
    else:
        preds_tok, tgts_tok = preds, tgts

    first_acc, last_acc = _compute_first_last_accuracy(preds_tok, tgts_tok)

    # ---- 表示（子音側の体裁に統一）----
    print(f"PER (音素誤り率):     {per:.2f}%")
    print(f"完全一致率（系列）:    {exact:.2f}%")
    print(f"最初/最後のトークン:   {first_acc:.2f}% / {last_acc:.2f}%")

    # mmに統一キーも足して返す（保存jsonも揃う）
    final_raw.setdefault('per_per', per)
    final_raw['exact_match_sequence_rate'] = exact
    final_raw['position_first_accuracy'] = first_acc
    final_raw['position_last_accuracy']  = last_acc
    return final_raw

# =========================================================
# 引数と設定処理
# =========================================================
def parse_arguments():
    """コマンドライン引数パース"""
    parser = argparse.ArgumentParser(description='読唇術モデル訓練・評価 (Pattern B)')

    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'test'],
                        default='train', help='実行モード')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='設定ファイルパス')

    parser.add_argument('--train_csv', type=str, help='訓練用CSVパス')
    parser.add_argument('--valid_csv', type=str, help='検証用CSVパス')
    parser.add_argument('--test_csv', type=str, help='テスト用CSVパス')
    parser.add_argument('--checkpoint', type=str, help='チェックポイントパス')

    # Attention設定（子音モデルや注意ありモデルで有効）
    parser.add_argument('--attention-type', type=str, choices=['sigmoid', 'softmax'],
                        help='Attention type: sigmoid or softmax')
    parser.add_argument('--temperature', type=float, help='Attention temperature (0.1-1.0)')

    # 訓練パラメータ
    parser.add_argument('--epochs', type=int, help='エポック数')
    parser.add_argument('--batch_size', type=int, help='バッチサイズ')
    parser.add_argument('--lr', type=float, help='学習率')

    # その他
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'auto'],
                        default='auto', help='使用デバイス')
    parser.add_argument('--seed', type=int, default=42, help='乱数シード')
    parser.add_argument('--debug', action='store_true', help='デバッグモード')
    return parser.parse_args()


def setup_config(args):
    """設定セットアップ"""
    config = Config(args.config if os.path.exists(args.config) else None)

    # 引数上書き
    if args.train_csv: config['data']['train_csv'] = args.train_csv
    if args.valid_csv: config['data']['valid_csv'] = args.valid_csv
    if args.test_csv:  config['data']['test_csv']  = args.test_csv
    if args.epochs:    config['training']['epochs'] = args.epochs
    if args.batch_size:config['data']['batch_size']  = args.batch_size
    if args.lr:        config['training']['lr']      = args.lr

    if args.attention_type: config['model']['attention_type'] = args.attention_type
    if args.temperature:    config['model']['temperature']    = args.temperature

    if args.device == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        config['device'] = args.device

    config['debug'] = args.debug
    return config


# =========================================================
# モデル生成
# =========================================================
def create_model_from_config(config, num_classes_from_encoder=None):
    """
    mode=='vowel' の場合は Attentionなしの CompactVowelLipReader_NoAttn を使用
    それ以外（'consonant'）は Pattern B 系のモデルを使用
    """
    mode = config['model'].get('mode', 'consonant')

    # ★ num_classes は「値」を渡す（methodオブジェクトを渡さない）
    num_classes = int(num_classes_from_encoder or config['model']['num_classes'])

    if mode == 'vowel':
        # NoAttn版のみを使う（attention_type/temperatureは渡さない）
        try:
            from model_compact_vowel import CompactVowelLipReader_NoAttn
        except ImportError:
            # ファイル構成によっては model_compact_vowel.py に居る場合のフォールバック
            from model_compact_vowel import CompactVowelLipReader_NoAttn

        return CompactVowelLipReader_NoAttn(
            num_classes=num_classes,
            dropout=config['model'].get('dropout_rate', 0.2),
        )

    else:
        # 子音用 Pattern B（関数名の揺れに対応）
        try:
            from model_pattern_b import create_improved_pattern_b_model as _factory
        except ImportError:
            from model_pattern_b import create_improved_pattern_a_model as _factory

        return _factory(
            num_classes=num_classes,
            dropout_rate=config['model']['dropout_rate'],
            attention_type=config['model']['attention_type'],
            temperature=config['model']['temperature'],
            dual_attention=config['model'].get('dual_attention', False)
        )


# =========================================================
# 学習メイン
# =========================================================
def train_model(config, args):
    """モデル訓練"""
    # 子音モデル用のAttention設定表示（NoAttnのときは単に表示だけ）
    config.print_attention_config()

    print("\nデータローダー作成中...")
    train_loader, valid_loader, phoneme_encoder, labels = build_loaders_from_config(config.config)

    # エンコーダに合わせて num_classes を自動同期（手動変更不要）
    sync_num_classes_with_encoder(config.config, phoneme_encoder)

    print("\nPattern B モデル作成中...")
    model = create_model_from_config(
        config.config,
        num_classes_from_encoder=phoneme_encoder.num_classes()  # ★ () を忘れない
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ モデルパラメータ数: {total_params:,}")
    input_shape = (
        config['data']['batch_size'],
        config['data']['max_length'],
        config['model']['input_channels'],
        config['data']['input_size'],
        config['data']['input_size']
    )
    print_model_info(model, input_shape)

    trainer = LipReadingTrainer(
        model=model,
        phoneme_encoder=phoneme_encoder,
        device=config['device'],
        save_dir=config['save']['checkpoint_dir'],
        early_stopping_metric=config['training'].get('early_stopping_metric', 'val_loss')
    )
    trainer.setup_optimizer(
        optimizer_type=config['training']['optimizer'],
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler_params = config['training'].get('scheduler_params', {})
    trainer.setup_scheduler(
        scheduler_type=config['training']['scheduler'],
        **scheduler_params
    )

    # チェックポイント読み込み
    start_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\nチェックポイント読み込み: {args.checkpoint}")
        start_epoch = trainer.load_checkpoint(args.checkpoint)

    # 学習ループ
    print("\n" + "=" * 70)
    print("訓練開始")
    print("=" * 70)
    history = trainer.train(
        train_loader=train_loader,
        val_loader=valid_loader,
        epochs=config['training']['epochs'],
        early_stopping_patience=config['training'].get('early_stopping_patience', 200)
    )

    # ===============================================
    # ✅ 統一された最終評価（途中評価と同一Evaluatorを使用）
    # ===============================================
    print("\n" + "=" * 70)
    print("最終評価指標計算中（validateと同一ロジック）...")
    print("=" * 70)
    final = evaluate_model(
    model, valid_loader, phoneme_encoder, config['device'],
    show_samples=True, num_samples=10
)
    mm = print_unified_summary(final['raw'], phoneme_encoder)

    print("\nサンプル結果（母音/子音共通）:")
    evaluator = UnifiedEvaluationMetrics()
    evaluator.print_sample_results(
        final['raw']['predictions'],
        final['raw']['targets'],
        num_samples=10,
        apply_collapse=True,
        show_correct=True,
        show_incorrect=True,
        vowel_mode=(config['model'].get('mode', '') == 'vowel')
    )


    print("\n" + "=" * 70)
    print(" 最終評価サマリー")
    print("=" * 70)
    mode = config['model'].get('mode', 'consonant')

    # PERはキーの揺れに対応
    per_val = mm.get('per_per', mm.get('PER', 0.0))
    print(f"PER (音素誤り率):     {per_val:.2f}%")

    if mode == 'consonant':
        print(f"子音完全一致率:       {mm.get('exact_match_consonant_exact_match_rate', 0.0):.2f}%")
        print(f"最初/最後の子音正解率: {mm.get('position_first_accuracy', 0.0):.2f}% / {mm.get('position_last_accuracy', 0.0):.2f}%")
    else:
        # vowelのときは子音系メトリクスは出さない（混乱防止）
        if 'exact_match_vowel_rate' in mm:
            print(f"完全一致率（母音列）:  {mm['exact_match_vowel_rate']:.2f}%")

    import json
    metrics_path = os.path.join(config['save']['result_dir'], 'final_metrics.json')
    os.makedirs(config['save']['result_dir'], exist_ok=True)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(mm, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 評価指標を保存: {metrics_path}")

    # ===============================================
    # Attention可視化 + 音素分析
    # NoAttnモデル（vowel）では自動スキップ
    # ===============================================
    try:
        print("\n" + "=" * 70)
        print("Attention可視化 + サンプル評価")
        print("=" * 70)
        from attention_visualizer import visualize_attention_with_samples

        # NoAttn想定のスキップ判定
        has_attn_attr = hasattr(model, "attention_weights")
        if mode == 'vowel' and not has_attn_attr:
            print("（NoAttnモデルのためAttention可視化をスキップ）")
        else:
            attention_result = visualize_attention_with_samples(
                model=model,
                data_loader=valid_loader,
                phoneme_encoder=phoneme_encoder,
                device=config['device'],
                num_samples=5,
                save_dir=os.path.join(config['save']['result_dir'], 'attention_visualization')
            )
            print(f"\n✓ Attention可視化完了")
            print(f"  - 可視化画像: {len(attention_result['correct_samples']) + len(attention_result['incorrect_samples'])}枚")
            print(f"  - 正解率: {attention_result['accuracy']*100:.1f}%")
    except Exception as e:
        print(f"⚠ Attention可視化エラー: {e}")
        import traceback; traceback.print_exc()

    print("\n音素別詳細分析を実行中...")
    analysis_dir = os.path.join(config['save']['result_dir'], 'phoneme_analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    analyze_phonemes_unified(
        predictions=final['raw'].get('predictions', []),
        targets=final['raw'].get('targets', []),
        save_dir=analysis_dir,
        apply_collapse=True
    )

    print("\n" + "=" * 70)
    print("✓ すべての処理が完了しました")
    print("=" * 70)


# =========================================================
# 評価モード
# =========================================================
def evaluate_model_mode(config, args):
    """評価モード"""
    print("\n" + "=" * 70)
    print("評価モード")
    print("=" * 70)

    # --- DataLoader（testはvalid側に流し込み） ---
    print("\nテストデータローダー作成中...")
    mode = config['model'].get('mode', 'consonant')
    _, test_loader, phoneme_encoder, labels = create_dataloaders(
        train_csv_path=config['data'].get('train_csv'),  # エンコーダ確立用
        valid_csv_path=config['data']['test_csv'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        augmentation_config=None,
        max_length=config['data'].get('max_length', 40),
        mode=mode,
    )

    # --- クラス数をエンコーダに同期（安全のため評価時も1回だけ実施） ---
    sync_num_classes_with_encoder(config.config, phoneme_encoder)

    # --- Model（modeで自動切替 & 出力次元=encoderに同期） ---
    print(f"\nモデル作成中... (mode={mode}, num_classes={phoneme_encoder.num_classes()})")
    model = create_model_from_config(
        config.config,
        num_classes_from_encoder=phoneme_encoder.num_classes()  # ★ () を忘れない
    )
    model.to(config['device'])

    # --- Checkpoint ---
    if not args.checkpoint:
        raise ValueError("評価モードではチェックポイント(--checkpoint)が必須です")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"チェックポイントが見つかりません: {args.checkpoint}")

    print(f"\nチェックポイント読み込み: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=config['device'])

    # 出力次元不一致の早期検出（例：consonantモデルでvowelをロードしようとした等）
    if 'model_state_dict' in checkpoint:
        head_keys = [k for k in checkpoint['model_state_dict'].keys() if k.endswith('classifier.3.weight')]
        if head_keys:
            ckpt_out = checkpoint['model_state_dict'][head_keys[0]].shape[0]
            if ckpt_out != phoneme_encoder.num_classes():
                raise ValueError(
                    f"Checkpoint出力次元({ckpt_out})とエンコーダ({phoneme_encoder.num_classes()})が不一致です。"
                    " mode と checkpoint が対応しているか確認してください。"
                )
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    else:
        # 万一フォーマットが異なる場合のフォールバック
        model.load_state_dict(checkpoint, strict=False)

    model.eval()

    # --- Evaluate ---
    print("\n最終評価を実行中...")
    final = evaluate_model(
    model, test_loader, phoneme_encoder, config['device'],
    show_samples=True, num_samples=10
)
    mm = print_unified_summary(final['raw'], phoneme_encoder)

    print("\nサンプル結果（母音/子音共通）:")
    evaluator = UnifiedEvaluationMetrics()
    evaluator.print_sample_results(
        final['raw']['predictions'],
        final['raw']['targets'],
        num_samples=10,
        apply_collapse=True,
        show_correct=True,
        show_incorrect=True,
        vowel_mode=(config['model'].get('mode', '') == 'vowel')  # ← ★追加
    )

    # 表示ラベル（モードに応じて動的に切替）
    print("\n" + "=" * 70)
    print("テストセット評価結果")
    print("=" * 70)
    per_val = mm.get('per_per', mm.get('PER', 0.0))
    print(f"PER (音素誤り率):     {per_val:.2f}%")

    if mode == 'consonant':
        print(f"完全一致率（子音列）:  {mm.get('exact_match_consonant_exact_match_rate', 0.0):.2f}%")
    else:
        if 'exact_match_vowel_rate' in mm:
            print(f"完全一致率（母音列）:  {mm.get('exact_match_vowel_rate', 0.0):.2f}%")


    # --- 分析（labelsは loader から受け取った集合を使う） ---
    analysis_dir = os.path.join(config['save']['result_dir'], 'phoneme_analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    analysis = analyze_phonemes_unified(
        predictions=final['raw'].get('predictions', []),
        targets=final['raw'].get('targets', []),
        labels=labels,                 # ← 子音/母音どちらでもOK
        save_dir=analysis_dir,
        top_k=5,
        plot_confusion=True,
    )

    print("\n--- 音素別分析 ---")
    print(f"Overall Acc: {analysis.get('overall_accuracy', 0.0)*100:.2f}%")
    print(f"Macro   Acc: {analysis.get('macro_accuracy', 0.0)*100:.2f}%")

    print("\n✓ 評価完了")
    return mm


# =========================================================
# エントリポイント
# =========================================================
def main():
    args = parse_arguments()
    config = setup_config(args)
    set_seed(config['seed'])
    check_gpu_availability()
    create_directories(config)

    if not check_data_paths(config):
        print("エラー: データファイルが見つかりません")
        return

    if args.mode == 'train':
        train_model(config, args)
    elif args.mode in ['eval', 'test']:
        evaluate_model_mode(config, args)
    else:
        raise ValueError(f"未知のモード: {args.mode}")

    print("\n処理完了")


if __name__ == "__main__":
    main()
