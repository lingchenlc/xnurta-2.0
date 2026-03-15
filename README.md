# 🚀 Xnurta 2.0 — AI-Powered Ad Optimization Engine

Amazon Ads 智能优化引擎 · 4 大核心 ML 模块 + 交互式 Dashboard

## 🏗 Architecture

```
┌─────────────┬──────────────┬──────────────┬─────────────────┐
│  📈 AdTFT    │  🚫 Negation  │  🌾 Harvest   │  💰 Bid         │
│  时序预测     │  否定词引擎    │  关键词收割    │  竞价优化        │
├─────────────┴──────────────┴──────────────┴─────────────────┤
│              🤖 AI Impact Simulator (集成仿真)                │
├──────────────────────────────────────────────────────────────┤
│              📊 Streamlit Dashboard (11 pages)               │
└──────────────────────────────────────────────────────────────┘
```

## 📊 Results (60-day dataset: $2.86M spend, $28.8M sales)

| Module | Impact | Type |
|---|---|---|
| 📈 AdTFT Prediction | +$1,439,111 | Revenue uplift |
| 🌾 Keyword Harvest | +$846,274 | Revenue + savings |
| 💰 Bid Optimization | +$724,698 | Savings + revenue |
| 🚫 Negation Engine | +$308,729 | Pure cost savings |
| **Total** | **+$3,302,167** | **ACoS 9.95% → 7.02%** |

## 🧠 Core Algorithms

- **AdTFT**: Temporal Fusion Transformer (135K params) · 21-day lookback → 1/3/7-day forecasts · Quantile regression (P10-P90)
- **Semantic Engine**: Sentence Transformer (all-MiniLM-L6-v2) + HDBSCAN · 500K search terms → 903 clusters → 27,448 negation candidates
- **Keyword Harvest**: Cross-reference search terms vs keyword structure · 4,558 Broad→Exact promotion candidates
- **Bid Landscape**: Isotonic Regression + LightGBM · 28 market segments · 499K bid recommendations

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run dashboard/app.py
```

## 📁 Structure

```
xnurta-2.0/
├── models/
│   ├── ad_tft/           # Temporal Fusion Transformer
│   ├── semantic_engine/  # Search term clustering + negation
│   └── bid_landscape/    # Bid optimization curves
├── dashboard/
│   ├── app.py            # Streamlit dashboard (11 pages)
│   └── data/             # Pre-computed visualization data
├── pipeline/             # Unified inference pipeline
└── notebooks/            # EDA exploration
```

## 📐 Tech Stack

PyTorch · Sentence Transformers · HDBSCAN · LightGBM · Isotonic Regression · Plotly · Streamlit

---

*Built with Xnurta 2.0 AI Engine · 2026*
