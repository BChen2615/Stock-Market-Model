# Taiwan Stock Market AI — End-to-End ML Trading System

An end-to-end machine learning system that scans the entire Taiwan Stock Exchange (~1,900 stocks) daily, predicts short-to-medium-term price movements across five time horizons, and surfaces the highest-confidence opportunities through a web dashboard.

---

## Demo

> **Live app** — Streamlit web dashboard with login, market radar, and individual stock analysis

![AI Stock Radar](https://via.placeholder.com/900x450.png?text=AI+Stock+Radar+Dashboard)

*Market Radar page: all stocks ranked by model confidence. Click any row to drill into the full technical analysis and prediction history.*

---

## Highlights

| | |
|---|---|
| **Coverage** | ~1,900 TWSE-listed stocks scanned every trading day |
| **Prediction horizons** | 1d · 2d · 3d · 7d · 14d — five independent models |
| **Feature set** | 80+ stationary technical + macro features |
| **Model** | XGBoost binary classifier (Up / Down) |
| **Backtest** | Realistic simulation — brokerage fee + securities tax + slippage |
| **Web app** | Streamlit dashboard with user authentication |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Data Layer                            │
│  twstock API  ──┐                                            │
│  yfinance     ──┼──► database_builder.py ──► twstock.db      │
│  SOX / SPX    ──┘    fetch_external_data.py  (SQLite)        │
└───────────────────────────────┬──────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────┐
│                     Feature Engineering                       │
│  Feature_Engineering_V2.py                                   │
│  • Price: Open Gap, High/Low Change, Bias_5/10/20/60         │
│  • Momentum: RSI, MACD_Hist_Norm, Close Location             │
│  • Volatility: NATR, Daily Range                             │
│  • Volume: Volume Change Rate                                 │
│  • External: SOX, S&P500, TSM ADR, TWII  (Lag 1–5)          │
│  • Lag features: 5-day history of all key signals            │
└───────────────────────────────┬──────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────┐
│                     Model Training                            │
│  Train_Universal_Model.py                                    │
│  • XGBoost  ×  5 horizons  (1d / 2d / 3d / 7d / 14d)       │
│  • Trained on 22 sector-leader stocks                        │
│  • Noise filter: |return| < 0.5% × √N removed               │
│  • Validated on unseen out-of-sample stocks                  │
└───────────────────────────────┬──────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────┐
│                     Daily Batch Prediction                    │
│  Batch_Predict.py                                            │
│  • Scans all ~1,900 TWSE stocks                              │
│  • Writes Prob_1d / 2d / 3d / 7d / 14d to daily_predictions │
│  • Scheduled to run after each market close                  │
└───────────────────────────────┬──────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────┐
│                     Web Dashboard (Streamlit)                 │
│  app/main.py                                                 │
│  • User authentication (register / login / access logs)      │
│  • Market Radar: full universe ranked by model confidence    │
│  • Stock Analysis: price chart + prob history + indicators   │
│  • Multi-horizon probability display per stock               │
└──────────────────────────────────────────────────────────────┘
```

---

## Key Technical Decisions

### Why five separate models instead of one?

Each horizon captures fundamentally different market dynamics:

| Horizon | What it captures |
|---------|-----------------|
| **1d** | Overnight gaps, intraday momentum, short-term mean reversion |
| **2–3d** | Post-earnings follow-through, short-term trend continuation |
| **7d** | Swing trade setups, sector rotation |
| **14d** | Fund flow positioning, macro regime |

A model trained to predict "up in 14 days" learns completely different feature weights than one trained on 1-day returns. Training them separately keeps each model optimal for its task and enables **multi-horizon consensus** as a higher-conviction signal.

### Feature stationarity

All features are ratio-based or normalized (e.g., `Close / SMA(20) - 1` rather than raw price). This ensures the model generalizes across different price levels and avoids spurious non-stationarity that would cause it to overfit to the training period's absolute price range.

### Generalization by design

The model is trained on 22 sector-leader stocks and evaluated on a **randomly sampled set of completely different stocks**. Strong test performance (ROC-AUC ~0.58–0.62 across horizons) on these unseen stocks demonstrates the features capture generalizable patterns rather than stock-specific idiosyncrasies.

### Realistic backtest cost model

| Cost item | Rate |
|-----------|------|
| Brokerage fee | 0.1425% × 2 (buy + sell) |
| Securities transaction tax | 0.3% (sell only) |
| Slippage | 0.1% × 2 |
| **Round-trip total** | **~0.685%** |

Each trade requires a return > ~0.7% just to break even, which naturally filters out low-conviction signals.

---

## Research & Analysis

The `research/` directory contains rigorous evaluation tooling beyond the production pipeline:

| File | Purpose |
|------|---------|
| `Backtest_V2.py` | Full equity-curve backtest engine — CAGR, Sharpe, Max Drawdown, Profit Factor |
| `Performance_Report_V2.py` | Classification metrics (ROC-AUC, PR-AUC), calibration analysis, market scan |
| `notebooks/Strategy_Performance_Analysis.ipynb` | End-to-end analysis notebook (9 sections) |

### Notebook highlights

The analysis notebook includes sections not found in most ML stock projects:

- **Multi-horizon signal interaction** — 2×2 quadrant analysis of 1d × 14d probability combinations to answer: *"when the 14-day model is bullish but the 1-day model is bearish, should you buy?"*
- **Horizon slope (prob momentum)** — classifies days as "Building" (14d > 1d), "Flat", or "Fading" (1d > 14d) and measures realized returns in each regime
- **Multi-strategy backtest comparison** — four strategies tested head-to-head: baseline (1d only), consensus (all horizons agree), dip buy (14d bullish + 1d weak), and alignment (average probability)
- **Threshold sensitivity** — sweeps all Buy × Sell threshold combinations across every test stock and visualizes averaged performance heatmaps

---

## Results Summary

*Evaluated on out-of-sample stocks never seen during training, post-2026-01-01 evaluation window.*

| Metric | 1d Model | 14d Model |
|--------|---------|----------|
| ROC-AUC | ~0.59 | ~0.62 |
| PR-AUC | ~0.56 | ~0.58 |
| Backtest Strategy vs B&H | varies by stock | — |
| Avg Sharpe (backtest) | ~0.8–1.2 | — |

> Full reproducible results are in `research/notebooks/Strategy_Performance_Analysis.ipynb`.

---

## Stack

| Layer | Technology |
|-------|------------|
| Data fetching | `twstock`, `yfinance`, `requests` |
| Storage | SQLite (via `sqlite3`) |
| Feature engineering | `pandas`, `numpy` |
| Modeling | `xgboost`, `scikit-learn` |
| Visualization | `plotly`, `matplotlib`, `seaborn` |
| Web app | `streamlit` |
| Research | Jupyter Notebooks |
| Language | Python 3.10 |

---

## Project Structure

```
Stock-Market-Model/
├── core/
│   ├── database_builder.py       # TWSE price data ingestion
│   ├── fetch_external_data.py    # SOX, S&P500, TWII, TSM ADR
│   ├── Feature_Engineering_V2.py # 80+ feature pipeline
│   ├── Train_Universal_Model.py  # XGBoost multi-horizon training
│   └── Batch_Predict.py          # Daily full-market scan
├── app/
│   ├── main.py                   # Streamlit dashboard
│   └── auth_system.py            # User auth & access logging
├── models/
│   ├── xgb_universal_1d.pkl
│   ├── xgb_universal_2d.pkl
│   ├── xgb_universal_3d.pkl
│   ├── xgb_universal_7d.pkl
│   └── xgb_universal_14d.pkl
├── research/
│   ├── Backtest_V2.py
│   ├── Performance_Report_V2.py
│   └── notebooks/
│       └── Strategy_Performance_Analysis.ipynb
└── data/
    └── twstock.db                # SQLite — prices + predictions
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# or: conda env create -f environment.yml
```

### 2. Build the database

```bash
python core/database_builder.py      # Fetch TWSE price history
python core/fetch_external_data.py   # Fetch SOX, S&P500, TWII, TSM ADR
```

### 3. Train the models

```bash
python core/Train_Universal_Model.py
# Trains all 5 horizon models → saves to models/
```

### 4. Run the daily prediction batch

```bash
python core/Batch_Predict.py
# Scans ~1,900 stocks → writes to daily_predictions table
```

### 5. Launch the web dashboard

```bash
streamlit run app/main.py
```

### 6. (Optional) Run the full analysis notebook

```
jupyter notebook research/notebooks/Strategy_Performance_Analysis.ipynb
```

---

## Limitations & Known Caveats

- **Regime overlap**: models are trained on historical data that overlaps with the backtest evaluation period (same macroeconomic regimes). A fully clean test would retrain models with data strictly before the evaluation window.
- **Execution assumption**: backtest assumes closing-price execution. Live trading requires order submission before market close (~2:30 PM TWTime).
- **No portfolio-level risk management**: each stock is evaluated independently. A real deployment would need position sizing, correlation limits, and drawdown controls.
- **Liquidity**: small-cap and thinly traded stocks may not fill the full notional position assumed by the model.

---

## Update Path

### Planned: Neural Network Feature Extractor → XGBoost Classifier (Two-Stage Pipeline)

The next architectural upgrade introduces a neural network as a learned feature extractor upstream of the existing XGBoost classifier.

**Motivation:** Raw technical indicators are hand-crafted and stationary by assumption, but they may miss non-linear temporal patterns across the 5-day lag window. A neural network can learn compact, task-relevant representations from the raw lag sequences before XGBoost applies its tree-based decision logic.

**Proposed pipeline:**

```
Raw features (80+)
       │
       ▼
┌──────────────────────────────┐
│   Neural Network Encoder     │
│   (MLP or 1-D CNN / LSTM)    │
│   Input:  80+ raw features   │
│   Output: N-dim embedding    │
│   Loss:   horizon-specific   │
│           binary CE          │
└──────────────┬───────────────┘
               │  learned embeddings
               ▼
┌──────────────────────────────┐
│   XGBoost Classifier         │
│   Input:  NN embeddings      │
│   Output: Prob_Up (0–1)      │
│   (same 5-horizon structure) │
└──────────────────────────────┘
```

**Training strategy:**
1. Train the NN end-to-end on the 22 sector-leader training stocks (same split as today).
2. Freeze the encoder and extract embeddings for all stocks.
3. Train XGBoost on those embeddings — same horizon-specific labels, same noise filter.

**Why keep XGBoost at the end?**
- Interpretability: SHAP values remain available on the XGBoost layer.
- Robustness: tree ensembles generalize better than a single neural output on small tabular datasets.
- Modularity: the NN encoder can be swapped or fine-tuned independently of the classifier.

**Expected impact:** higher ROC-AUC on medium horizons (7d / 14d) where temporal patterns in the lag window are most informative; marginal gain expected on 1d where noise dominates.

---

## License

MIT — see [LICENSE.txt](LICENSE.txt)
