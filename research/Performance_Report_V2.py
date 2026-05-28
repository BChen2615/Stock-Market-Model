"""
Performance Report V2 — Comprehensive model evaluation for all prediction horizons.

Key fixes vs Final_Performance_Report.py:
  - F1 calculation: handles division-by-zero and precision_recall_curve length mismatch
  - Consistent label filtering across evaluation and scan
  - Per-stock metric breakdown (not just aggregate)
  - Multi-horizon evaluation (1d, 2d, 3d, 7d, 14d)
  - Calibration analysis: predicted probability vs actual return
  - Market scan uses the model's own optimal threshold
"""
from __future__ import annotations  # enables X | Y union hints on Python 3.9

import os
import sys
import sqlite3
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_curve,
    average_precision_score, confusion_matrix, ConfusionMatrixDisplay
)

current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir    = os.path.join(os.path.dirname(current_dir), 'core')
sys.path.insert(0, core_dir)
from Feature_Engineering_V2 import prepare_training_data_v2

BASE_DIR    = os.path.dirname(current_dir)
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
DB_PATH     = os.path.join(BASE_DIR, 'data', 'twstock.db')
OUTPUT_PATH = os.path.join(BASE_DIR, 'market_scan_results.csv')

# ── Config ────────────────────────────────────────────────────────────────────
HORIZONS = [1, 2, 3, 7, 14]

# Ground-truth training set (must match Train_Universal_Model.py exactly)
_TRAIN_STOCKS = {
    '2330','2454','2303','3711','3034',   # Semi
    '2881','2882','2891','2886','2884',   # Finance
    '2317','2308','2382','2357','3231',   # Tech
    '2002','1301','1303','1101',          # Traditional
    '2603','2609','2615',                 # Shipping
}

# These stocks are NOT in the training set (generalisation test)
# Aligned with Backtest_V2.py so both files evaluate the same universe.
TEST_STOCKS = ['2344', '2834', '1402', '2618', '3045', '2912', '2207', '9910']

# Sanity-check: blow up early if any test stock leaked into training
_overlap = [s for s in TEST_STOCKS if s in _TRAIN_STOCKS]
if _overlap:
    raise ValueError(f"TEST_STOCKS contains training stocks: {_overlap}")

# Noise filter: same as used during training
# threshold = 0.005 * sqrt(days) per horizon
def sig_threshold(days: int) -> float:
    return 0.005 * np.sqrt(days)


# ── Data helpers ───────────────────────────────────────────────────────────────

def get_all_stock_ids() -> list[str]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT Stock_ID FROM tw_stock_prices")
    stocks = [row[0] for row in cursor.fetchall()]
    conn.close()
    return [s for s in stocks if len(s) == 4]


def load_features(stock_id: str, target_days: int, keep_raw: bool = False) -> pd.DataFrame | None:
    try:
        df = prepare_training_data_v2(
            stock_id, target_days=target_days, path=DB_PATH, keep_raw_prices=keep_raw
        )
    except Exception:
        return None
    return df if (df is not None and not df.empty) else None


def build_test_set(target_days: int) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build labelled test set for a given horizon.
    Applies the SAME noise filter used during training.
    """
    thr = sig_threshold(target_days)
    X_parts, y_parts = [], []

    for stock in TEST_STOCKS:
        df = load_features(stock, target_days)
        if df is None:
            continue
        mask = (df['Future_Return'] > thr) | (df['Future_Return'] < -thr)
        df   = df[mask].copy()
        df['Target'] = (df['Future_Return'] > 0).astype(int)

        drop_cols = ['Stock_ID', 'Type', 'Future_Return', 'Target']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)

        X_parts.append(X)
        y_parts.append(df['Target'])

    if not X_parts:
        return pd.DataFrame(), pd.Series(dtype=int)

    return pd.concat(X_parts), pd.concat(y_parts)


# ── Threshold optimisation (fixed) ────────────────────────────────────────────

def find_best_threshold(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, float, float, float]:
    """
    Returns (best_threshold, best_precision, best_recall, best_f1).

    Fix: precision_recall_curve returns arrays of length n+1 for precision/recall
    but n for thresholds. We use precision[:-1] and recall[:-1] to align.
    Also guards against division-by-zero.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # Align lengths: drop the last element (precision=1, recall=0 boundary point)
    p = precisions[:-1]
    r = recalls[:-1]

    denom     = p + r
    f1_scores = np.where(denom > 0, 2 * p * r / denom, 0.0)

    best_idx = int(np.argmax(f1_scores))
    return (
        float(thresholds[best_idx]),
        float(p[best_idx]),
        float(r[best_idx]),
        float(f1_scores[best_idx]),
    )


# ── Section 1: Per-horizon model evaluation ───────────────────────────────────

def section1_model_evaluation() -> dict:
    """
    Evaluate each horizon model on unseen test stocks.
    Returns dict: horizon -> {'auc', 'ap', 'acc', 'threshold', ...}
    """
    print("\n" + "═"*60)
    print("SECTION 1 — Model Performance (Unseen Test Stocks)")
    print("═"*60)

    horizon_results = {}

    for days in HORIZONS:
        model_path = os.path.join(MODELS_DIR, f'xgb_universal_{days}d.pkl')
        if not os.path.exists(model_path):
            print(f"  [{days}d] model not found, skipping.")
            continue

        model  = joblib.load(model_path)
        X_test, y_test = build_test_set(days)

        if X_test.empty:
            print(f"  [{days}d] no test data.")
            continue

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        auc = roc_auc_score(y_test, y_prob)
        ap  = average_precision_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        thr, prec, rec, f1 = find_best_threshold(y_test, y_prob)

        # Per-stock breakdown
        per_stock = {}
        for stock in TEST_STOCKS:
            df = load_features(stock, days)
            if df is None:
                continue
            thr_s = sig_threshold(days)
            mask  = (df['Future_Return'] > thr_s) | (df['Future_Return'] < -thr_s)
            df    = df[mask].copy()
            if len(df) < 10:
                continue
            df['Target'] = (df['Future_Return'] > 0).astype(int)
            drop_cols = ['Stock_ID', 'Type', 'Future_Return', 'Target']
            Xs = df.drop(columns=[c for c in drop_cols if c in df.columns])
            Xs = Xs.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)
            ys = df['Target']
            try:
                per_stock[stock] = {
                    'auc': roc_auc_score(ys, model.predict_proba(Xs)[:, 1]),
                    'acc': accuracy_score(ys, model.predict(Xs)),
                    'n':   len(ys),
                }
            except Exception:
                pass

        horizon_results[days] = {
            'auc':       auc,
            'ap':        ap,
            'acc':       acc,
            'threshold': thr,
            'precision': prec,
            'recall':    rec,
            'f1':        f1,
            'y_test':    y_test,
            'y_prob':    y_prob,
            'per_stock': per_stock,
        }

        print(f"\n  [{days}d model]  n={len(y_test):,}")
        print(f"    ROC-AUC:     {auc:.4f}")
        print(f"    PR-AUC:      {ap:.4f}")
        print(f"    Accuracy:    {acc:.4f}")
        print(f"    Best Threshold (max F1={f1:.4f}): {thr:.4f}  "
              f"[Prec={prec:.4f}, Rec={rec:.4f}]")
        print(f"\n    Per-stock AUC:")
        for s, m in per_stock.items():
            print(f"      {s}: AUC={m['auc']:.3f}  Acc={m['acc']:.3f}  (n={m['n']})")

    return horizon_results


# ── Section 2: Calibration — prob vs actual return ────────────────────────────

def section2_calibration(target_days: int = 1):
    """
    Show whether higher predicted probability actually corresponds to higher returns.
    Uses raw (unfiltered) data so all market conditions are represented.
    """
    print("\n" + "═"*60)
    print(f"SECTION 2 — Calibration Analysis ({target_days}d model)")
    print("═"*60)

    model_path = os.path.join(MODELS_DIR, f'xgb_universal_{target_days}d.pkl')
    if not os.path.exists(model_path):
        print("  Model not found.")
        return

    model = joblib.load(model_path)
    all_probs, all_returns = [], []

    for stock in TEST_STOCKS:
        df = load_features(stock, target_days)
        if df is None:
            continue
        df = df.dropna(subset=['Future_Return'])
        drop_cols = ['Stock_ID', 'Type', 'Future_Return']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)
        probs = model.predict_proba(X)[:, 1]
        all_probs.extend(probs.tolist())
        all_returns.extend(df['Future_Return'].values.tolist())

    if not all_probs:
        print("  No data collected.")
        return

    cal_df = pd.DataFrame({'prob': all_probs, 'ret': all_returns})

    # 10 equal-width bins across [0, 1]
    cal_df['bin'] = pd.cut(cal_df['prob'], bins=10, labels=False)
    stats = (
        cal_df.groupby('bin', observed=True)['ret']
        .agg(mean='mean', std='std', count='count')
        .reset_index()
    )
    stats['mean_pct'] = stats['mean'] * 100
    stats['std_pct']  = stats['std']  * 100
    stats['bin_label'] = stats['bin'].apply(lambda b: f"{b*0.1:.1f}–{(b+1)*0.1:.1f}")

    print("\n  Probability Bin   Mean Return (%)   Std (%)   Count")
    print("  " + "─"*56)
    for _, row in stats.iterrows():
        print(f"  {row['bin_label']:>14}   {row['mean_pct']:>+14.3f}   "
              f"{row['std_pct']:>8.3f}   {int(row['count']):>6}")

    # Monotonicity check: does mean return increase with probability?
    high_half = stats.loc[stats['bin'] >= 5, 'mean_pct'].mean()
    low_half  = stats.loc[stats['bin'] < 5,  'mean_pct'].mean()
    print(f"\n  High-confidence bins (0.5–1.0) avg return: {high_half:+.3f}%")
    print(f"  Low-confidence  bins (0.0–0.5) avg return: {low_half:+.3f}%")
    if high_half > low_half:
        print("  ✓ Model is calibrated: higher prob → higher return")
    else:
        print("  ✗ Model calibration issue: high prob does NOT predict higher return")

    return stats


# ── Section 3: Precision-Recall curves for all horizons ───────────────────────

def section3_pr_curves(horizon_results: dict):
    """Plot PR curves for all horizons in one figure."""
    print("\n" + "═"*60)
    print("SECTION 3 — Precision-Recall Curves (all horizons)")
    print("═"*60)

    n = len(horizon_results)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    colors = ['#7c3aed', '#2563eb', '#16a34a', '#d97706', '#dc2626']

    for ax, (days, res), color in zip(axes, horizon_results.items(), colors):
        precisions, recalls, _ = precision_recall_curve(res['y_test'], res['y_prob'])
        ap = res['ap']

        ax.plot(recalls, precisions, color=color, lw=2, label=f'AP={ap:.3f}')
        ax.scatter(
            [res['recall']], [res['precision']],
            marker='o', s=80, color='red', zorder=5,
            label=f"Best thr={res['threshold']:.2f}\nF1={res['f1']:.3f}"
        )
        ax.axhline(y=0.5, linestyle='--', color='gray', lw=0.8, alpha=0.6)
        ax.set_title(f"{days}d Model  (AUC={res['auc']:.3f})")
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    fig.suptitle('Precision-Recall Curves — All Horizons', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


# ── Section 4: Confusion matrix for primary model ────────────────────────────

def section4_confusion_matrix(horizon_results: dict, primary_days: int = 1):
    """Show confusion matrix for the primary (1d) model."""
    if primary_days not in horizon_results:
        return

    res    = horizon_results[primary_days]
    thr    = res['threshold']
    y_pred = (res['y_prob'] >= thr).astype(int)
    cm     = confusion_matrix(res['y_test'], y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Down', 'Up'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f"{primary_days}d Model — Confusion Matrix\n(threshold={thr:.4f})")
    plt.tight_layout()
    plt.show()


# ── Section 5: Market scan ────────────────────────────────────────────────────

def section5_market_scan(target_days: int = 1, threshold: float | None = None):
    """
    Scan all stocks. Uses the same noise filter for consistency.
    threshold: if None, uses 0.55 as default.
    """
    print("\n" + "═"*60)
    print(f"SECTION 5 — Market Scan ({target_days}d model)")
    print("═"*60)

    model_path = os.path.join(MODELS_DIR, f'xgb_universal_{target_days}d.pkl')
    if not os.path.exists(model_path):
        print("  Model not found.")
        return

    model     = joblib.load(model_path)
    all_stocks = get_all_stock_ids()
    threshold  = threshold if threshold is not None else 0.55

    print(f"  Scanning {len(all_stocks)} stocks  (threshold={threshold:.4f}) ...")
    results = []

    for i, stock in enumerate(all_stocks):
        if i % 200 == 0 and i > 0:
            print(f"  ... {i}/{len(all_stocks)}")
        try:
            df = load_features(stock, target_days, keep_raw=True)
            if df is None or df.empty:
                continue

            last_row   = df.iloc[[-1]]
            last_date  = last_row.index[0]
            last_close = last_row['Close'].values[0]
            last_vol   = last_row['Volume'].values[0] if 'Volume' in last_row.columns else np.nan

            drop_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                         'Type', 'Stock_ID', 'Future_Return']
            X = last_row.drop(columns=[c for c in drop_cols if c in last_row.columns])
            X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)

            prob = model.predict_proba(X)[0, 1]
            results.append({
                'Stock_ID': stock,
                'Date':     last_date,
                'Close':    last_close,
                'Volume':   last_vol,
                'Prob_Up':  prob,
            })
        except Exception:
            continue

    if not results:
        print("  No results.")
        return

    res_df   = pd.DataFrame(results)
    top_picks = res_df[res_df['Prob_Up'] > threshold].sort_values('Prob_Up', ascending=False)

    print(f"\n  Stocks above threshold: {len(top_picks)}")
    print(f"\n  Top 20:")
    print(top_picks.head(20).to_string(index=False, float_format='%.4f'))

    # Save
    top_picks.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Saved to: {OUTPUT_PATH}")

    # Plot top 10
    top10 = top_picks.head(10)
    if not top10.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#16a34a' if p > threshold else '#6b7280' for p in top10['Prob_Up']]
        bars = ax.barh(top10['Stock_ID'].astype(str), top10['Prob_Up'], color=colors)
        ax.axvline(threshold, color='red', linestyle='--', label=f'Threshold={threshold:.2f}')
        ax.set_xlabel('Probability of Up Move')
        ax.set_title(f'Top 10 Stocks — {target_days}d Model  '
                     f'(Date: {top10["Date"].iloc[0].date() if hasattr(top10["Date"].iloc[0], "date") else top10["Date"].iloc[0]})')
        ax.set_xlim([max(0, threshold - 0.1), 1.0])
        ax.legend()
        ax.invert_yaxis()
        for bar, val in zip(bars, top10['Prob_Up']):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9)
        plt.tight_layout()
        plt.show()

    return top_picks


# ── Section 2b: Calibration chart ────────────────────────────────────────────

def plot_calibration_chart(stats: pd.DataFrame, target_days: int = 1):
    if stats is None or stats.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(
        range(len(stats)), stats['mean_pct'],
        yerr=stats['std_pct'] / np.sqrt(stats['count'].clip(lower=1)),
        color=['#16a34a' if m > 0 else '#dc2626' for m in stats['mean_pct']],
        alpha=0.8, capsize=4
    )
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(stats['bin_label'], rotation=30, ha='right', fontsize=9)
    ax.set_xlabel('Predicted Probability Bin')
    ax.set_ylabel('Mean Actual Return (%)')
    ax.set_title(f'Calibration: Predicted Probability vs Actual Return ({target_days}d)')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary_table(horizon_results: dict):
    print("\n" + "═"*60)
    print("SUMMARY — All Horizons")
    print("═"*60)
    print(f"  {'Horizon':>8}  {'ROC-AUC':>8}  {'PR-AUC':>7}  {'Accuracy':>9}  "
          f"{'Best Thr':>9}  {'F1':>7}")
    print(f"  {'─'*58}")
    for days, res in horizon_results.items():
        print(f"  {str(days)+'d':>8}  {res['auc']:>8.4f}  {res['ap']:>7.4f}  "
              f"{res['acc']:>9.4f}  {res['threshold']:>9.4f}  {res['f1']:>7.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Performance Report V2")
    print(f"DB:     {DB_PATH}")
    print(f"Models: {MODELS_DIR}")
    print(f"Test stocks: {TEST_STOCKS}")

    # 1. Evaluate all horizon models
    horizon_results = section1_model_evaluation()

    if horizon_results:
        print_summary_table(horizon_results)

        # 3. PR curves
        section3_pr_curves(horizon_results)

        # 4. Confusion matrix for 1d model
        section4_confusion_matrix(horizon_results, primary_days=1)

    # 2. Calibration analysis (1d model, most actionable)
    cal_stats = section2_calibration(target_days=1)
    if cal_stats is not None:
        plot_calibration_chart(cal_stats, target_days=1)

    # 5. Market scan using best threshold from 1d model
    best_thr = None
    if 1 in horizon_results:
        best_thr = horizon_results[1]['threshold']
        print(f"\nUsing optimal threshold from 1d model: {best_thr:.4f}")
    
    section5_market_scan(target_days=1, threshold=best_thr)


if __name__ == "__main__":
    main()
