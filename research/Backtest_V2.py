"""
Backtest V2 — Fixed and comprehensive backtesting engine for Taiwan stock market.

Key fixes vs Backtest_Universal.py:
  - Correct equity model: track shares × price, not cash adjustments
  - Fee applied symmetrically on both buy and sell; tax only on sell
  - Reproducible: random seed
  - Full metrics: CAGR, Sharpe, max drawdown, win rate, avg hold, profit factor
  - Benchmark also pays entry/exit fees for fair comparison
  - Random out-of-sample stock selection (reproducible via RANDOM_SEED)
  - Time-based out-of-sample split (EVAL_START_DATE) to avoid regime overlap

⚠  Regime-overlap note
   The current model was trained on ALL data from 2020 onwards.
   Even though the test stocks are different from the training stocks,
   backtesting on the *same time period* means the model has already
   "seen" the same market regimes (COVID crash, rate hikes, etc.).
   EVAL_START_DATE mitigates this by limiting evaluation to a window
   the model has seen less of.  For a fully clean evaluation, retrain
   the model with data only up to EVAL_START_DATE.
"""
from __future__ import annotations  # enables X | Y union hints on Python 3.9

import os
import sys
import sqlite3
import random

import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(os.path.dirname(current_dir), 'core')
sys.path.insert(0, core_dir)
from Feature_Engineering_V2 import prepare_training_data_v2

BASE_DIR    = os.path.dirname(current_dir)
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
DB_PATH     = os.path.join(BASE_DIR, 'data', 'twstock.db')

# ── Trading constants ──────────────────────────────────────────────────────────
INITIAL_CAPITAL  = 1_000_000
FEE              = 0.001425   # 0.1425% on both buy and sell
TAX              = 0.003      # 0.3% on sell only
SLIPPAGE         = 0.001      # 0.1% per trade (bid-ask spread + market impact)
BUY_THRESHOLD    = 0.70
SELL_THRESHOLD   = 0.4
RISK_FREE_ANNUAL = 0.02       # Taiwan bank rate ~2%, used for Sharpe
RANDOM_SEED      = 33

# ── Backtest scope ─────────────────────────────────────────────────────────────
HORIZONS      = [1, 2, 3, 7, 14]   # prediction horizons to evaluate
N_TEST_STOCKS = 10                  # stocks to randomly sample per run

# Out-of-sample evaluation window.
# Only data strictly AFTER this date is used in the backtest simulation.
# This limits (but does not fully eliminate) regime overlap with the training
# period.  The model was trained on ALL dates; for a truly clean split you
# would also retrain the model with data only up to this cutoff.
# Set to None to use the entire history (not recommended).
EVAL_START_DATE = '2025-01-01'

# ── Training-stock registry (must match Train_Universal_Model.py exactly) ──────
_TRAIN_STOCKS = {
    '2330','2454','2303','3711','3034',  # Semi
    '2881','2882','2891','2886','2884',  # Finance
    '2317','2308','2382','2357','3231',  # Tech
    '2002','1301','1303','1101',         # Traditional
    '2603','2609','2615',               # Shipping
}


# ── Data helpers ───────────────────────────────────────────────────────────────

def get_all_stock_ids() -> list[str]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT Stock_ID FROM tw_stock_prices")
    stocks = [row[0] for row in cursor.fetchall()]
    conn.close()
    return [s for s in stocks if len(s) == 4]


def sample_test_stocks(n: int = N_TEST_STOCKS, seed: int = RANDOM_SEED) -> list[str]:
    """
    Randomly sample n stocks that are NOT in the training set.
    Uses a fixed seed so runs are reproducible; change RANDOM_SEED to
    get a different draw.
    """
    all_stocks = get_all_stock_ids()
    eligible   = [s for s in all_stocks if s not in _TRAIN_STOCKS]
    rng        = random.Random(seed)
    sampled    = rng.sample(eligible, min(n, len(eligible)))
    # Sanity-check (should never fire given the filter above)
    overlap = [s for s in sampled if s in _TRAIN_STOCKS]
    if overlap:
        raise ValueError(f"sample_test_stocks: training stocks leaked in: {overlap}")
    return sampled


# ── Core backtest engine ───────────────────────────────────────────────────────

def run_backtest(stock_id: str, model, target_days: int = 1) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    """
    Simulate the trading strategy on one stock.

    Returns
    -------
    (equity_df, trades_df)  or  (None, None) on failure.

    equity_df columns : Strategy, Benchmark, Prob_Up
    trades_df columns : entry_date, exit_date, entry_price, exit_price,
                        return_pct, holding_days
    """
    try:
        df = prepare_training_data_v2(
            stock_id, target_days=target_days, path=DB_PATH, keep_raw_prices=True
        )
    except Exception as e:
        print(f"  [ERROR] {stock_id}: {e}")
        return None, None

    if df is None or df.empty:
        print(f"  [SKIP] {stock_id}: no data")
        return None, None

    # Drop last row(s) where Future_Return is NaN (nothing to trade against)
    df = df.dropna(subset=['Future_Return'])

    # ── Time-based out-of-sample filter ──────────────────────────────────────
    # Restrict simulation to dates strictly AFTER EVAL_START_DATE so the
    # evaluated period has minimal overlap with the model's training regime.
    # Note: features are still computed from the FULL history (warmup intact);
    # we only restrict WHICH bars are actually traded.
    if EVAL_START_DATE is not None:
        cutoff = pd.Timestamp(EVAL_START_DATE)
        df = df[df.index > cutoff]

    if len(df) < 60:
        label = f"after {EVAL_START_DATE}" if EVAL_START_DATE else "total"
        print(f"  [SKIP] {stock_id}: only {len(df)} rows {label} (need ≥60)")
        return None, None

    # ── Prepare feature matrix ────────────────────────────────────────────────
    drop_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                 'Type', 'Stock_ID', 'Future_Return']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    closes = df['Close'].values.astype(float)
    probs  = model.predict_proba(X)[:, 1]
    dates  = df.index

    # ── Simulation ────────────────────────────────────────────────────────────
    # State
    cash   = float(INITIAL_CAPITAL)
    shares = 0.0
    entry_price: float | None = None
    entry_date = None

    # Benchmark: buy at first close, pay same FEE + SLIPPAGE as strategy.
    # Exit fee applied at end for a fair apples-to-apples comparison.
    bm_shares = INITIAL_CAPITAL / (closes[0] * (1 + FEE + SLIPPAGE))

    equity_curve = []
    bm_curve     = []
    trade_log    = []

    for i in range(len(df)):
        price = closes[i]
        prob  = probs[i]
        date  = dates[i]

        # ── Buy signal ───────────────────────────────────────────────────────
        # Model is trained on: Future_Return[i] = (close[i+1] - close[i]) / close[i]
        # Features at row i use data up to and including close[i].
        # Executing at close[i] is correct: we enter at the prediction baseline
        # and profit if the predicted up-move materialises at close[i+1].
        # (Trading at close[i+1] would mean entering AFTER the predicted move.)
        if shares == 0 and prob > BUY_THRESHOLD:
            shares      = cash / (price * (1 + FEE + SLIPPAGE))
            cash        = 0.0
            entry_price = price
            entry_date  = date

        # ── Sell signal ──────────────────────────────────────────────────────
        elif shares > 0 and prob < SELL_THRESHOLD:
            gross_proceeds = shares * price
            cash   = gross_proceeds * (1 - FEE - TAX - SLIPPAGE)
            shares = 0.0

            rt_return = (price * (1 - FEE - TAX - SLIPPAGE)) / (entry_price * (1 + FEE + SLIPPAGE)) - 1
            hold_days = (date - entry_date).days

            trade_log.append({
                'entry_date':   entry_date,
                'exit_date':    date,
                'entry_price':  entry_price,
                'exit_price':   price,
                'return_pct':   rt_return,
                'holding_days': hold_days,
            })
            entry_price = None
            entry_date  = None

        # ── Mark-to-market ───────────────────────────────────────────────────
        equity_curve.append(shares * price if shares > 0 else cash)
        bm_curve.append(bm_shares * price)

    # ── Liquidation at end: apply exit costs so both curves are comparable ───
    # Strategy: if still holding at end, mark final bar net of sell costs.
    if shares > 0:
        equity_curve[-1] = shares * closes[-1] * (1 - FEE - TAX - SLIPPAGE)
    # Benchmark always "sells" at last close, paying the same exit costs.
    bm_curve[-1] = bm_curve[-1] * (1 - FEE - TAX - SLIPPAGE)

    equity_df = pd.DataFrame({
        'Strategy':  equity_curve,
        'Benchmark': bm_curve,
        'Prob_Up':   probs,
    }, index=dates)

    trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame(
        columns=['entry_date', 'exit_date', 'entry_price', 'exit_price',
                 'return_pct', 'holding_days']
    )

    return equity_df, trades_df


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(equity: pd.Series, trades: pd.DataFrame, label='Strategy',
                    initial_capital: float = INITIAL_CAPITAL) -> dict:
    """
    Compute strategy-level metrics from an equity curve and trade log.

    initial_capital: the true starting cash before any fees are paid.
    Using equity.iloc[0] as the baseline would overstate returns slightly
    because the first bar already reflects entry fees on day 0 buys.
    """
    equity = equity.dropna()
    if len(equity) < 2:
        return {}

    daily_ret = equity.pct_change().dropna()
    total_return = (equity.iloc[-1] / initial_capital) - 1

    # Use actual calendar span for CAGR; fall back to trading-day estimate
    # if the index doesn't support date arithmetic.
    try:
        years = (equity.index[-1] - equity.index[0]).days / 365.25
    except Exception:
        years = len(equity) / 252
    if years < 0.01:
        years = len(equity) / 252

    cagr = (equity.iloc[-1] / initial_capital) ** (1 / years) - 1 if years > 0.01 else 0.0

    rf_daily = RISK_FREE_ANNUAL / 252
    excess   = daily_ret - rf_daily
    sharpe   = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 1e-8 else 0.0

    rolling_max  = equity.cummax()
    drawdown     = (equity - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    m = {
        'Total Return': f"{total_return:+.2%}",
        'CAGR':         f"{cagr:+.2%}",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Max Drawdown': f"{max_drawdown:.2%}",
        'Num Trades':   len(trades),
    }

    if len(trades) > 0 and 'return_pct' in trades.columns:
        wins     = trades['return_pct'] > 0
        gross_p  = trades.loc[wins, 'return_pct'].sum()
        gross_l  = abs(trades.loc[~wins, 'return_pct'].sum())
        pf       = (gross_p / gross_l) if gross_l > 0 else float('inf')

        m['Win Rate']        = f"{wins.mean():.2%}"
        m['Avg Trade Ret']   = f"{trades['return_pct'].mean():+.2%}"
        m['Avg Hold (days)'] = f"{trades['holding_days'].mean():.1f}"
        m['Profit Factor']   = f"{pf:.2f}"

    return m


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_results(stock_id: str, equity_df: pd.DataFrame, trades_df: pd.DataFrame,
                 strat_metrics: dict, bench_metrics: dict):
    """Interactive Plotly chart: equity + probability + trade markers."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.68, 0.32],
        subplot_titles=(
            f"{stock_id} — Equity Curve (normalized to 100)",
            "Model Confidence (Prob Up)"
        )
    )

    # ── Equity curves (normalised) ────────────────────────────────────────────
    strat_norm = equity_df['Strategy']  / equity_df['Strategy'].iloc[0]  * 100
    bench_norm = equity_df['Benchmark'] / equity_df['Benchmark'].iloc[0] * 100

    fig.add_trace(go.Scatter(
        x=equity_df.index, y=strat_norm,
        name='AI Strategy', line=dict(color='#7c3aed', width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=equity_df.index, y=bench_norm,
        name='Buy & Hold', line=dict(color='#6b7280', width=1.5, dash='dash')
    ), row=1, col=1)

    # ── Trade markers ─────────────────────────────────────────────────────────
    if not trades_df.empty:
        entries = equity_df.loc[equity_df.index.isin(trades_df['entry_date'])]
        exits   = equity_df.loc[equity_df.index.isin(trades_df['exit_date'])]

        entry_norm = entries['Strategy'] / equity_df['Strategy'].iloc[0] * 100
        exit_norm  = exits['Strategy']  / equity_df['Strategy'].iloc[0] * 100

        fig.add_trace(go.Scatter(
            x=entries.index, y=entry_norm,
            mode='markers', name='Buy',
            marker=dict(symbol='triangle-up', color='#16a34a', size=10)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=exits.index, y=exit_norm,
            mode='markers', name='Sell',
            marker=dict(symbol='triangle-down', color='#dc2626', size=10)
        ), row=1, col=1)

    # ── Probability ───────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=equity_df.index, y=equity_df['Prob_Up'],
        name='Prob Up', fill='tozeroy',
        line=dict(color='#2563eb', width=1),
        fillcolor='rgba(37,99,235,0.15)'
    ), row=2, col=1)

    fig.add_hline(y=BUY_THRESHOLD,  line_dash='dot', line_color='#16a34a',
                  annotation_text=f'Buy >{BUY_THRESHOLD}',  row=2, col=1)
    fig.add_hline(y=SELL_THRESHOLD, line_dash='dot', line_color='#dc2626',
                  annotation_text=f'Sell <{SELL_THRESHOLD}', row=2, col=1)

    # ── Annotation box ────────────────────────────────────────────────────────
    ann_text = (
        f"<b>Strategy</b><br>"
        + "<br>".join(f"{k}: {v}" for k, v in strat_metrics.items())
        + f"<br><br><b>Buy & Hold</b><br>"
        + "<br>".join(f"{k}: {v}" for k, v in bench_metrics.items())
    )

    fig.add_annotation(
        xref='paper', yref='paper', x=0.01, y=0.98,
        text=ann_text, align='left', showarrow=False,
        bgcolor='rgba(255,255,255,0.85)',
        bordercolor='#9ca3af', borderwidth=1, font=dict(size=11)
    )

    fig.update_layout(
        title=f"Backtest: {stock_id}  |  Buy >{BUY_THRESHOLD:.0%}  Sell <{SELL_THRESHOLD:.0%}",
        height=700, hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    fig.update_yaxes(title_text='Normalised Value', row=1, col=1)
    fig.update_yaxes(title_text='Prob Up', range=[0, 1], row=2, col=1)

    fig.show()


def plot_multi_horizon_summary(results: dict, horizons: list[int]):
    """
    Grouped bar chart: for each test stock, one bar per horizon + B&H.
    Makes it easy to see which horizon model performs best per stock.
    """
    stocks   = list(results.keys())
    colors   = ['#7c3aed', '#2563eb', '#16a34a', '#d97706', '#dc2626']
    bh_color = '#6b7280'

    def _pct(s: str) -> float:
        return float(s.rstrip('%').replace('+', ''))

    fig = go.Figure()

    for color, days in zip(colors, horizons):
        y_vals = []
        for stock in stocks:
            hr = results[stock].get(days)
            y_vals.append(_pct(hr['strat'].get('Total Return', '0%')) if hr else None)
        fig.add_trace(go.Bar(name=f'{days}d model', x=stocks, y=y_vals, marker_color=color))

    # Buy & Hold (same for all horizons; use the first available horizon per stock)
    bh_vals = []
    for stock in stocks:
        hr = next((results[stock][d] for d in horizons if d in results[stock]), None)
        bh_vals.append(_pct(hr['bench'].get('Total Return', '0%')) if hr else None)
    fig.add_trace(go.Bar(name='Buy & Hold', x=stocks, y=bh_vals, marker_color=bh_color))

    fig.add_hline(y=0, line_color='black', line_width=0.8)
    fig.update_layout(
        title=f'All Horizons vs Buy & Hold  (eval: {EVAL_START_DATE} → latest)',
        barmode='group',
        yaxis_title='Total Return (%)',
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.show()


def print_multi_horizon_summary(results: dict, horizons: list[int]):
    """Compact table: rows = stocks, columns = each horizon + B&H."""
    col_w  = 11
    h_hdrs = [f'{d}d Strat' for d in horizons]
    header = f"  {'Stock':>6}  {'B&H':>{col_w}}" + ''.join(f"  {h:>{col_w}}" for h in h_hdrs)

    print(f"\n{'═'*len(header)}")
    print("SUMMARY — Total Return by Horizon")
    print('═'*len(header))
    print(header)
    print('  ' + '─'*(len(header)-2))

    for stock, stock_res in results.items():
        bh = next((stock_res[d]['bench'].get('Total Return', 'N/A')
                   for d in horizons if d in stock_res), 'N/A')
        row = f"  {stock:>6}  {bh:>{col_w}}"
        for days in horizons:
            val = stock_res[days]['strat'].get('Total Return', '─') if days in stock_res else '─'
            row += f"  {val:>{col_w}}"
        print(row)

    # Average row
    print('  ' + '─'*(len(header)-2))
    def _avg(col_vals):
        nums = []
        for v in col_vals:
            try: nums.append(float(v.rstrip('%').replace('+','')))
            except: pass
        return f"{sum(nums)/len(nums):+.2f}%" if nums else 'N/A'

    bh_vals   = [next((r[d]['bench'].get('Total Return','N/A') for d in horizons if d in r), 'N/A')
                 for r in results.values()]
    avg_row   = f"  {'AVG':>6}  {_avg(bh_vals):>{col_w}}"
    for days in horizons:
        vals = [r[days]['strat'].get('Total Return','N/A') for r in results.values() if days in r]
        avg_row += f"  {_avg(vals):>{col_w}}"
    print(avg_row)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Load all horizon models ───────────────────────────────────────────────
    models = {}
    for days in HORIZONS:
        path = os.path.join(MODELS_DIR, f'xgb_universal_{days}d.pkl')
        if os.path.exists(path):
            models[days] = joblib.load(path)
            print(f"  Loaded {days}d model")
        else:
            print(f"  [{days}d] model not found, skipping")

    if not models:
        print("No models found. Run Train_Universal_Model.py first.")
        return

    # ── Randomly sample out-of-sample test stocks ─────────────────────────────
    stocks      = sample_test_stocks(n=N_TEST_STOCKS, seed=RANDOM_SEED)
    eval_period = f"{EVAL_START_DATE} → latest" if EVAL_START_DATE else "full history"

    print(f"\n{'═'*60}")
    print(f"  Horizons : {list(models.keys())}")
    print(f"  Stocks   : {stocks}  (seed={RANDOM_SEED})")
    print(f"  Eval     : {eval_period}")
    if EVAL_START_DATE:
        print(f"  ⚠  Regime-overlap caveat: retrain up to {EVAL_START_DATE} for a fully clean test.")
    print(f"  Buy >{BUY_THRESHOLD:.0%}  Sell <{SELL_THRESHOLD:.0%}  "
          f"Fee {FEE:.4%}×2 + Tax {TAX:.2%} on sell")
    print(f"{'═'*60}\n")

    # results[stock][days] = {'strat': metrics, 'bench': metrics, 'equity_df', 'trades_df'}
    results: dict[str, dict[int, dict]] = {}

    for days, model in models.items():
        print(f"\n{'─'*60}")
        print(f"  ── {days}d Model ──")
        print(f"{'─'*60}")

        for stock_id in stocks:
            if stock_id not in results:
                results[stock_id] = {}

            equity_df, trades_df = run_backtest(stock_id, model, target_days=days)
            if equity_df is None:
                continue

            strat_m = compute_metrics(equity_df['Strategy'],  trades_df,      'Strategy', INITIAL_CAPITAL)
            bench_m = compute_metrics(equity_df['Benchmark'], pd.DataFrame(), 'B&H',      INITIAL_CAPITAL)

            results[stock_id][days] = {
                'strat':     strat_m,
                'bench':     bench_m,
                'equity_df': equity_df,
                'trades_df': trades_df,
            }

            eval_start = str(equity_df.index[0].date())
            eval_end   = str(equity_df.index[-1].date())
            tr  = strat_m.get('Total Return', 'N/A')
            btr = bench_m.get('Total Return', 'N/A')
            shr = strat_m.get('Sharpe Ratio', 'N/A')
            nt  = strat_m.get('Num Trades', 0)
            print(f"  {stock_id}  {eval_start}→{eval_end}  "
                  f"Strat={tr:>8}  B&H={btr:>8}  Sharpe={shr}  Trades={nt}")

    if not results:
        print("No results.")
        return

    # ── Cross-horizon summary ─────────────────────────────────────────────────
    active_horizons = list(models.keys())
    print_multi_horizon_summary(results, active_horizons)
    plot_multi_horizon_summary(results, active_horizons)

    # ── Individual equity curves for the 1d model (most actionable) ──────────
    if 1 in models:
        print(f"\n{'═'*60}")
        print("  Equity curves — 1d model")
        print(f"{'═'*60}")
        for stock_id, stock_res in results.items():
            if 1 not in stock_res:
                continue
            r = stock_res[1]
            plot_results(stock_id, r['equity_df'], r['trades_df'], r['strat'], r['bench'])


if __name__ == "__main__":
    main()
