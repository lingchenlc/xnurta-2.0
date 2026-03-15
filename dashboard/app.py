"""
Xnurta 2.0 — Model Dashboard
==============================
Streamlit dashboard to visualize all model outputs:
  - Campaign Performance Overview
  - AdTFT Prediction Model Results
  - Semantic Engine: Clusters, Negation, Harvest
  - Feature Importance & Interpretability

Run:
  1. python3 dashboard/prepare_dashboard_data.py   (only once)
  2. streamlit run dashboard/app.py
"""

import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "dashboard").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent
DASHBOARD_DATA = PROJECT_ROOT / "dashboard" / "data"

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Xnurta 2.0 AI Engine",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Data Loading (cached) ─────────────────────────────────────

@st.cache_data(ttl=3600)
def load_kpis():
    path = DASHBOARD_DATA / "kpis.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

@st.cache_data(ttl=3600)
def load_daily_agg():
    path = DASHBOARD_DATA / "daily_agg.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=['date'])
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_country_agg():
    path = DASHBOARD_DATA / "country_agg.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_country_daily():
    path = DASHBOARD_DATA / "country_daily_agg.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=['date'])
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_top_campaigns():
    path = DASHBOARD_DATA / "top_campaigns.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_training_results():
    path = PROJECT_ROOT / "models" / "ad_tft" / "trained" / "training_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

@st.cache_data(ttl=3600)
def load_feature_metadata():
    path = PROJECT_ROOT / "data" / "features" / "feature_metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

@st.cache_data(ttl=3600)
def load_negation_recs():
    path = PROJECT_ROOT / "data" / "processed" / "negation_recommendations.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_harvest_recs():
    path = PROJECT_ROOT / "data" / "processed" / "harvest_recommendations.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_semantic_clusters():
    path = PROJECT_ROOT / "data" / "processed" / "semantic_clusters.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_cluster_analysis():
    path = PROJECT_ROOT / "data" / "processed" / "cluster_analysis.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_bid_landscape_results():
    path = PROJECT_ROOT / "models" / "bid_landscape" / "output" / "bid_landscape_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

@st.cache_data(ttl=3600)
def load_bid_landscape_curves():
    path = PROJECT_ROOT / "models" / "bid_landscape" / "output" / "bid_landscape_curves.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

@st.cache_data(ttl=3600)
def load_bid_recommendations():
    path = PROJECT_ROOT / "models" / "bid_landscape" / "output" / "bid_recommendations_sample.csv"
    if path.exists():
        return pd.read_csv(path)
    # Fallback to full file if sample doesn't exist
    full_path = PROJECT_ROOT / "models" / "bid_landscape" / "output" / "bid_recommendations.csv"
    if full_path.exists():
        return pd.read_csv(full_path, nrows=10000)
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_ai_impact_simulation():
    path = DASHBOARD_DATA / "ai_impact_simulation.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_ai_impact_details():
    path = DASHBOARD_DATA / "ai_impact_details.json"
    if path.exists():
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return {}

@st.cache_data(ttl=3600)
def load_ai_comparison():
    path = DASHBOARD_DATA / "ai_comparison.json"
    if path.exists():
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return {}

@st.cache_data(ttl=3600)
def load_feature_analysis():
    path = DASHBOARD_DATA / "feature_analysis.json"
    if path.exists():
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return {}


# ─── Sidebar ───────────────────────────────────────────────────

st.sidebar.title("🚀 Xnurta 2.0")
st.sidebar.markdown("**AI-Powered Ad Optimization Engine**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📋 导航",
    ["🏠 总览 Dashboard", "⚔️ AI 1.0 vs 2.0", "🔍 优化特征分析",
     "🤖 AI 效果模拟器", "📤 AI 效果预测", "📈 AdTFT 预测模型", "🔤 语义引擎",
     "🚫 否定词推荐", "🌾 关键词收割", "💰 Bid Landscape", "📊 Campaign 分析"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("**项目状态**")
st.sidebar.markdown("✅ Phase 0: 数据探索")
st.sidebar.markdown("✅ Phase 1: AdTFT 模型")
st.sidebar.markdown("✅ Phase 2: 语义引擎")
st.sidebar.markdown("✅ Phase 3: Bid Landscape")
st.sidebar.markdown("✅ Phase 4: 集成评估")


# ═══════════════════════════════════════════════════════════════
# PAGE: Dashboard Overview
# ═══════════════════════════════════════════════════════════════

if page == "🏠 总览 Dashboard":
    st.title("🚀 Xnurta 2.0 — AI Engine Dashboard")
    st.markdown("**Amazon Ads 智能优化引擎 · 模型效果总览**")

    # Load data
    kpis = load_kpis()
    training_results = load_training_results()
    negation = load_negation_recs()
    harvest = load_harvest_recs()
    clusters = load_semantic_clusters()
    bl_results_overview = load_bid_landscape_results()

    # ─── KPI Cards ───
    st.markdown("### 📊 数据概览")
    c1, c2, c3, c4, c5 = st.columns(5)

    if kpis:
        c1.metric("💰 总花费", f"${kpis.get('total_spend', 0):,.0f}")
        c2.metric("📦 总销售额", f"${kpis.get('total_sales', 0):,.0f}")
        c3.metric("🛒 总订单", f"{kpis.get('total_orders', 0):,}")
        c4.metric("📉 整体 ACoS", f"{kpis.get('avg_acos', 0):.1f}%")
        c5.metric("📈 整体 ROAS", f"{kpis.get('avg_roas', 0):.1f}x")

    # Data scope info
    if kpis:
        st.caption(
            f"📅 数据范围: {kpis.get('date_min', '')} ~ {kpis.get('date_max', '')} · "
            f"{kpis.get('n_days', 0)} 天 · "
            f"{kpis.get('n_countries', 0)} 国家 · "
            f"{kpis.get('n_stores', 0)} 店铺 · "
            f"{kpis.get('n_campaigns', 0):,} campaigns · "
            f"{kpis.get('n_rows', 0):,} 行数据"
        )

    st.markdown("---")

    # ─── Model Performance Summary ───
    st.markdown("### 🤖 模型性能")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown("#### AdTFT 时序预测")
        if training_results:
            tm = training_results.get('test_metrics', {})
            st.metric("Test Loss", f"{tm.get('loss', 0):.4f}")
            cov80 = tm.get('coverage_80', 0)
            st.metric("Coverage 80%", f"{cov80:.1f}%",
                      delta=f"{cov80 - 80:.1f}%" if cov80 else None)
            st.metric("Overall MAE", f"{tm.get('mae', 0):.3f}")
            st.caption(f"参数量: {training_results.get('n_params', 0):,} · "
                       f"Best Epoch: {training_results.get('best_epoch', 0)} · "
                       f"训练时间: {training_results.get('total_time', 0) / 60:.1f} min")
        else:
            st.warning("模型未训练")

    with m2:
        st.markdown("#### 语义聚类引擎")
        if not clusters.empty:
            st.metric("语义簇数量", f"{len(clusters)}")
            total_terms = int(clusters['n_terms'].sum()) if 'n_terms' in clusters.columns else 0
            st.metric("已聚类搜索词", f"{total_terms:,}")
            if 'classification' in clusters.columns:
                good_pct = clusters['classification'].str.contains('GOOD', na=False).sum() / len(clusters) * 100
                st.metric("优质簇占比", f"{good_pct:.0f}%")
        else:
            st.warning("语义数据未加载")

    with m3:
        st.markdown("#### 优化建议")
        if not negation.empty:
            st.metric("否定词候选", f"{len(negation):,}")
            saveable = negation['total_spend'].sum() if 'total_spend' in negation.columns else 0
            st.metric("可节省花费", f"${saveable:,.0f}")
        if not harvest.empty:
            st.metric("收割关键词", f"{len(harvest):,}")
            harvest_sales = harvest['total_sales'].sum() if 'total_sales' in harvest.columns else 0
            st.metric("潜在销售额", f"${harvest_sales:,.0f}")

    with m4:
        st.markdown("#### Bid Landscape")
        if bl_results_overview:
            st.metric("分析市场", f"{bl_results_overview.get('n_segments', 0)}")
            st.metric("调价建议", f"{bl_results_overview.get('n_recommendations', 0):,}")
            action_dist_ov = bl_results_overview.get('action_distribution', {})
            dec = action_dist_ov.get('decrease', 0)
            total_recs = bl_results_overview.get('n_recommendations', 1)
            st.metric("建议降价比例", f"{dec/max(total_recs,1)*100:.0f}%")
        else:
            st.warning("Bid Landscape 未训练")

    st.markdown("---")

    # ─── Training Curve ───
    if training_results and 'training_history' in training_results:
        st.markdown("### 📉 训练曲线")
        history = training_results['training_history']
        train_loss = history.get('train_loss', [])
        val_loss = history.get('val_loss', [])

        if train_loss and val_loss:
            fig = go.Figure()
            epochs = list(range(1, len(train_loss) + 1))
            fig.add_trace(go.Scatter(x=epochs, y=train_loss, name='Train Loss',
                                     mode='lines+markers', line=dict(color='#636EFA')))
            fig.add_trace(go.Scatter(x=epochs, y=val_loss, name='Val Loss',
                                     mode='lines+markers', line=dict(color='#EF553B')))
            best_epoch = training_results.get('best_epoch', 0)
            if best_epoch > 0 and best_epoch <= len(val_loss):
                fig.add_vline(x=best_epoch, line_dash="dash", line_color="green",
                             annotation_text=f"Best (Epoch {best_epoch})")
            fig.update_layout(title="Quantile Loss 训练曲线",
                            xaxis_title="Epoch", yaxis_title="Loss",
                            height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    # ─── Daily Trends ───
    daily = load_daily_agg()
    if not daily.empty:
        st.markdown("### 📅 每日趋势")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           subplot_titles=["每日花费 & 销售额", "每日 ACoS (%)"],
                           vertical_spacing=0.12)
        fig.add_trace(go.Scatter(x=daily['date'], y=daily['spend'], name='Spend',
                                fill='tozeroy', fillcolor='rgba(99,110,250,0.2)',
                                line=dict(color='#636EFA')), row=1, col=1)
        fig.add_trace(go.Scatter(x=daily['date'], y=daily['sales'], name='Sales',
                                fill='tozeroy', fillcolor='rgba(0,204,150,0.2)',
                                line=dict(color='#00CC96')), row=1, col=1)
        fig.add_trace(go.Scatter(x=daily['date'], y=daily['acos'], name='ACoS %',
                                line=dict(color='#EF553B', width=2)), row=2, col=1)
        fig.update_layout(height=500, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: AI 1.0 vs AI 2.0
# ═══════════════════════════════════════════════════════════════

elif page == "⚔️ AI 1.0 vs 2.0":
    st.title("⚔️ AI 1.0 vs AI 2.0 效果对比")
    st.markdown("**当前 AI 1.0 的表现与局限 → Xnurta AI 2.0 如何全面超越**")

    comp = load_ai_comparison()
    if not comp:
        st.error("对比数据未准备。请先运行: `python3 dashboard/prepare_ai_comparison_data.py`")
        st.stop()

    ai1 = comp['ai1']
    noai = comp['noai_matched']
    ai2 = comp['ai2']

    # ─── Section 1: Three-Column Comparison Cards ───
    st.markdown("### 📊 核心指标对比")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"<div style='background: linear-gradient(135deg, #f5f5f5, #e0e0e0); padding: 20px; "
            f"border-radius: 12px; text-align: center; border-top: 4px solid #9e9e9e;'>"
            f"<div style='font-size: 16px; font-weight: bold; color: #616161;'>👤 人工投放 (基线)</div>"
            f"<div style='font-size: 12px; color: #999; margin-bottom: 12px;'>"
            f"{noai['n_campaigns']:,} 个规模匹配 campaigns</div>"
            f"<div style='display:flex; justify-content:space-around; flex-wrap:wrap;'>"
            f"<div style='text-align:center; margin:4px;'><div style='color:#888;font-size:11px;'>ACoS</div>"
            f"<div style='font-size:22px;font-weight:bold;color:#424242;'>{noai['acos']:.1f}%</div></div>"
            f"<div style='text-align:center; margin:4px;'><div style='color:#888;font-size:11px;'>ROAS</div>"
            f"<div style='font-size:22px;font-weight:bold;color:#424242;'>{noai['roas']:.1f}x</div></div>"
            f"</div>"
            f"<div style='display:flex; justify-content:space-around; flex-wrap:wrap; margin-top:8px;'>"
            f"<div style='text-align:center; margin:4px;'><div style='color:#888;font-size:11px;'>CPC</div>"
            f"<div style='font-size:16px;color:#424242;'>${noai['cpc']:.2f}</div></div>"
            f"<div style='text-align:center; margin:4px;'><div style='color:#888;font-size:11px;'>CVR</div>"
            f"<div style='font-size:16px;color:#424242;'>{noai['cvr']:.1f}%</div></div>"
            f"<div style='text-align:center; margin:4px;'><div style='color:#888;font-size:11px;'>CTR</div>"
            f"<div style='font-size:16px;color:#424242;'>{noai['ctr']:.2f}%</div></div>"
            f"</div></div>", unsafe_allow_html=True)

    with c2:
        st.markdown(
            f"<div style='background: linear-gradient(135deg, #fff3e0, #ffe0b2); padding: 20px; "
            f"border-radius: 12px; text-align: center; border-top: 4px solid #ff9800;'>"
            f"<div style='font-size: 16px; font-weight: bold; color: #e65100;'>🤖 AI 1.0 (当前)</div>"
            f"<div style='font-size: 12px; color: #bf6c00; margin-bottom: 12px;'>"
            f"{ai1['n_campaigns']} 个 AI 运行中 campaigns</div>"
            f"<div style='display:flex; justify-content:space-around; flex-wrap:wrap;'>"
            f"<div style='text-align:center; margin:4px;'><div style='color:#bf6c00;font-size:11px;'>ACoS</div>"
            f"<div style='font-size:22px;font-weight:bold;color:#e65100;'>{ai1['acos']:.1f}%</div></div>"
            f"<div style='text-align:center; margin:4px;'><div style='color:#bf6c00;font-size:11px;'>ROAS</div>"
            f"<div style='font-size:22px;font-weight:bold;color:#e65100;'>{ai1['roas']:.1f}x</div></div>"
            f"</div>"
            f"<div style='display:flex; justify-content:space-around; flex-wrap:wrap; margin-top:8px;'>"
            f"<div style='text-align:center; margin:4px;'><div style='color:#bf6c00;font-size:11px;'>CPC</div>"
            f"<div style='font-size:16px;color:#e65100;'>${ai1['cpc']:.2f}</div></div>"
            f"<div style='text-align:center; margin:4px;'><div style='color:#bf6c00;font-size:11px;'>CVR</div>"
            f"<div style='font-size:16px;color:#e65100;'>{ai1['cvr']:.1f}%</div></div>"
            f"<div style='text-align:center; margin:4px;'><div style='color:#bf6c00;font-size:11px;'>CTR</div>"
            f"<div style='font-size:16px;color:#e65100;'>{ai1['ctr']:.2f}%</div></div>"
            f"</div></div>", unsafe_allow_html=True)

    with c3:
        st.markdown(
            f"<div style='background: linear-gradient(135deg, #e8f5e9, #a5d6a7); padding: 20px; "
            f"border-radius: 12px; text-align: center; border-top: 4px solid #4caf50;'>"
            f"<div style='font-size: 16px; font-weight: bold; color: #2e7d32;'>🚀 AI 2.0 (Xnurta)</div>"
            f"<div style='font-size: 12px; color: #388e3c; margin-bottom: 12px;'>"
            f"全量 157,701 campaigns 优化</div>"
            f"<div style='display:flex; justify-content:space-around; flex-wrap:wrap;'>"
            f"<div style='text-align:center; margin:4px;'><div style='color:#388e3c;font-size:11px;'>ACoS</div>"
            f"<div style='font-size:22px;font-weight:bold;color:#1b5e20;'>{ai2['new_acos']:.1f}%</div></div>"
            f"<div style='text-align:center; margin:4px;'><div style='color:#388e3c;font-size:11px;'>ROAS</div>"
            f"<div style='font-size:22px;font-weight:bold;color:#1b5e20;'>{ai2['new_roas']:.1f}x</div></div>"
            f"</div>"
            f"<div style='display:flex; justify-content:space-around; flex-wrap:wrap; margin-top:8px;'>"
            f"<div style='text-align:center; margin:4px;'><div style='color:#388e3c;font-size:11px;'>节省</div>"
            f"<div style='font-size:16px;color:#1b5e20;'>${ai2['total_savings']:,.0f}</div></div>"
            f"<div style='text-align:center; margin:4px;'><div style='color:#388e3c;font-size:11px;'>增收</div>"
            f"<div style='font-size:16px;color:#1b5e20;'>${ai2['total_uplift']:,.0f}</div></div>"
            f"<div style='text-align:center; margin:4px;'><div style='color:#388e3c;font-size:11px;'>利润↑</div>"
            f"<div style='font-size:16px;color:#1b5e20;'>${ai2['profit_improvement']:,.0f}</div></div>"
            f"</div></div>", unsafe_allow_html=True)

    st.caption(f"* 人工基线已按规模匹配筛选（campaign spend ≥ ${noai['match_threshold']:,.0f}，"
               f"共 {noai['n_campaigns']:,} 个 campaigns），确保公平对比。"
               f"AI 2.0 数据为全量 portfolio 100% 开启预测。")

    st.markdown("---")

    # ─── Section 2: Grouped Bar Charts ───
    st.markdown("### 📈 关键指标对比")

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        categories = ['ACoS (%)', 'CPC ($)']
        fig.add_trace(go.Bar(name='👤 人工', x=categories,
                             y=[noai['acos'], noai['cpc']],
                             marker_color='#bdbdbd', text=[f"{noai['acos']:.1f}%", f"${noai['cpc']:.2f}"],
                             textposition='outside'))
        fig.add_trace(go.Bar(name='🤖 AI 1.0', x=categories,
                             y=[ai1['acos'], ai1['cpc']],
                             marker_color='#ffb74d', text=[f"{ai1['acos']:.1f}%", f"${ai1['cpc']:.2f}"],
                             textposition='outside'))
        fig.add_trace(go.Bar(name='🚀 AI 2.0', x=categories,
                             y=[ai2['new_acos'], ai2.get('estimated_cpc', noai['cpc'] * 0.85)],
                             marker_color='#66bb6a',
                             text=[f"{ai2['new_acos']:.1f}%",
                                   f"${ai2.get('estimated_cpc', noai['cpc'] * 0.85):.2f}"],
                             textposition='outside'))
        fig.update_layout(title="越低越好 ↓", barmode='group', height=400,
                          template="plotly_white", yaxis_title="数值",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        categories = ['ROAS (x)', 'CVR (%)']
        fig.add_trace(go.Bar(name='👤 人工', x=categories,
                             y=[noai['roas'], noai['cvr']],
                             marker_color='#bdbdbd', text=[f"{noai['roas']:.1f}x", f"{noai['cvr']:.1f}%"],
                             textposition='outside'))
        fig.add_trace(go.Bar(name='🤖 AI 1.0', x=categories,
                             y=[ai1['roas'], ai1['cvr']],
                             marker_color='#ffb74d', text=[f"{ai1['roas']:.1f}x", f"{ai1['cvr']:.1f}%"],
                             textposition='outside'))
        fig.add_trace(go.Bar(name='🚀 AI 2.0', x=categories,
                             y=[ai2['new_roas'], ai2.get('estimated_cvr', noai['cvr'] * 1.05)],
                             marker_color='#66bb6a',
                             text=[f"{ai2['new_roas']:.1f}x",
                                   f"{ai2.get('estimated_cvr', noai['cvr'] * 1.05):.1f}%"],
                             textposition='outside'))
        fig.update_layout(title="越高越好 ↑", barmode='group', height=400,
                          template="plotly_white", yaxis_title="数值",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ─── Section 3: Time-Series Charts (KEY USER REQUEST) ───
    st.markdown("### 📅 60 天表现趋势：人工 vs AI 1.0 vs AI 2.0")
    st.caption("7 日移动平均线，AI 2.0 为基于模型预测的投射线")

    ai1_daily_df = pd.DataFrame(comp['ai1_daily'])
    noai_daily_df = pd.DataFrame(comp['noai_daily'])
    ai2_daily_df = pd.DataFrame(comp['ai2_daily'])

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=noai_daily_df['date'], y=noai_daily_df['acos_ma7'],
            name='👤 人工投放', mode='lines',
            line=dict(color='#9e9e9e', width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=ai1_daily_df['date'], y=ai1_daily_df['acos_ma7'],
            name='🤖 AI 1.0', mode='lines',
            line=dict(color='#ff9800', width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=ai2_daily_df['date'], y=ai2_daily_df['acos_ma7'],
            name='🚀 AI 2.0 (预测)', mode='lines',
            line=dict(color='#4caf50', width=3, dash='dash'),
        ))
        fig.update_layout(
            title="ACoS 趋势 (7日均线)",
            xaxis_title="日期", yaxis_title="ACoS (%)",
            height=420, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            yaxis=dict(rangemode='tozero'),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=noai_daily_df['date'], y=noai_daily_df['orders_ma7'],
            name='👤 人工投放', mode='lines',
            line=dict(color='#9e9e9e', width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=ai1_daily_df['date'], y=ai1_daily_df['orders_ma7'],
            name='🤖 AI 1.0', mode='lines',
            line=dict(color='#ff9800', width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=ai2_daily_df['date'], y=ai2_daily_df['orders_ma7'],
            name='🚀 AI 2.0 (预测)', mode='lines',
            line=dict(color='#4caf50', width=3, dash='dash'),
        ))
        fig.update_layout(
            title="日订单量趋势 (7日均线)",
            xaxis_title="日期", yaxis_title="订单数",
            height=420, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            yaxis=dict(rangemode='tozero'),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Additional time-series: Spend & Sales
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=noai_daily_df['date'], y=noai_daily_df['spend_ma7'],
            name='👤 人工投放', mode='lines',
            line=dict(color='#9e9e9e', width=2),
            fill='tozeroy', fillcolor='rgba(158,158,158,0.1)',
        ))
        fig.add_trace(go.Scatter(
            x=ai1_daily_df['date'], y=ai1_daily_df['spend_ma7'],
            name='🤖 AI 1.0', mode='lines',
            line=dict(color='#ff9800', width=2),
        ))
        fig.add_trace(go.Scatter(
            x=ai2_daily_df['date'], y=ai2_daily_df['spend_ma7'],
            name='🚀 AI 2.0 (预测)', mode='lines',
            line=dict(color='#4caf50', width=2.5, dash='dash'),
        ))
        fig.update_layout(
            title="日花费趋势 (7日均线)",
            xaxis_title="日期", yaxis_title="花费 ($)",
            height=350, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=noai_daily_df['date'], y=noai_daily_df['sales_ma7'],
            name='👤 人工投放', mode='lines',
            line=dict(color='#9e9e9e', width=2),
            fill='tozeroy', fillcolor='rgba(158,158,158,0.1)',
        ))
        fig.add_trace(go.Scatter(
            x=ai1_daily_df['date'], y=ai1_daily_df['sales_ma7'],
            name='🤖 AI 1.0', mode='lines',
            line=dict(color='#ff9800', width=2),
        ))
        fig.add_trace(go.Scatter(
            x=ai2_daily_df['date'], y=ai2_daily_df['sales_ma7'],
            name='🚀 AI 2.0 (预测)', mode='lines',
            line=dict(color='#4caf50', width=2.5, dash='dash'),
        ))
        fig.update_layout(
            title="日销售额趋势 (7日均线)",
            xaxis_title="日期", yaxis_title="销售额 ($)",
            height=350, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ─── Section 4: AI 1.0 Limitations → AI 2.0 Solutions ───
    st.markdown("### 🔍 AI 1.0 的局限性 → AI 2.0 的解决方案")

    limitations = comp['limitations_solutions']
    for lim in limitations:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"<div style='background: linear-gradient(135deg, #ffebee, #ffcdd2); "
                f"padding: 16px; border-radius: 10px; border-left: 4px solid #ef5350; "
                f"margin-bottom: 8px; min-height: 90px;'>"
                f"<div style='font-weight: bold; color: #c62828; font-size: 15px;'>"
                f"❌ {lim['problem_zh']}</div>"
                f"<div style='color: #b71c1c; font-size: 13px; margin-top: 6px;'>"
                f"原因：{lim['cause_zh']}</div>"
                f"</div>", unsafe_allow_html=True)
        with col2:
            savings_text = f"节省 ${lim.get('ai2_savings', 0):,.0f}" if lim.get('ai2_savings') else ""
            uplift_text = f"增收 ${lim.get('ai2_uplift', 0):,.0f}" if lim.get('ai2_uplift') else ""
            impact_parts = [p for p in [savings_text, uplift_text] if p]
            impact_text = " · ".join(impact_parts) if impact_parts else "优化整体效率"
            st.markdown(
                f"<div style='background: linear-gradient(135deg, #e8f5e9, #c8e6c9); "
                f"padding: 16px; border-radius: 10px; border-left: 4px solid #4caf50; "
                f"margin-bottom: 8px; min-height: 90px;'>"
                f"<div style='font-weight: bold; color: #2e7d32; font-size: 15px;'>"
                f"✅ {lim['ai2_icon']} {lim['ai2_solution_zh']}</div>"
                f"<div style='color: #1b5e20; font-size: 13px; margin-top: 6px;'>"
                f"预计影响：{impact_text}</div>"
                f"</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ─── Section 5: Bidding Strategy Analysis ───
    st.markdown("### 💡 为什么 AI 1.0 的 CPC 更高？")
    st.markdown("AI 1.0 过度使用 **动态竞价-提高和降低** 策略 (Amazon 可将 CPC 提高至 3 倍)，"
                "导致每次点击成本 (CPC) 显著高于人工投放。")

    ai1_bid = comp['ai1_bidding_dist']
    noai_bid = comp['noai_bidding_dist']

    col1, col2 = st.columns(2)
    with col1:
        labels = list(ai1_bid.keys())
        values = list(ai1_bid.values())
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.45,
                                      marker_colors=['#ffcc80', '#ff7043', '#ffab91', '#ffe0b2'])])
        fig.update_layout(title="🤖 AI 1.0 竞价策略分布", height=350,
                          annotations=[dict(text='AI 1.0', x=0.5, y=0.5, font_size=14, showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        labels = list(noai_bid.keys())
        values = list(noai_bid.values())
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.45,
                                      marker_colors=['#e0e0e0', '#bdbdbd', '#9e9e9e', '#757575'])])
        fig.update_layout(title="👤 人工投放竞价策略分布", height=350,
                          annotations=[dict(text='人工', x=0.5, y=0.5, font_size=14, showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)

    # Highlight the key difference
    ai1_dynamic_up = ai1_bid.get('动态竞价-提高和降低', 0)
    noai_dynamic_up = noai_bid.get('动态竞价-提高和降低', 0)
    ratio_val = ai1_dynamic_up / max(noai_dynamic_up, 0.1)
    st.info(
        f"💡 AI 1.0 使用 '动态竞价-提高和降低' 的比例为 "
        f"**{ai1_dynamic_up:.1f}%**（人工仅 {noai_dynamic_up:.1f}%），"
        f"高出 **{ratio_val:.1f} 倍**。"
        f"AI 2.0 的 Bid Landscape 引擎通过数据驱动的出价优化取代了这种粗放策略。"
    )

    st.markdown("---")

    # ─── Section 6: Module Improvement Waterfall ───
    st.markdown("### 🧩 AI 2.0 各模块对 ACoS 的优化贡献")

    # Calculate each module's ACoS reduction contribution
    total_savings = ai2['total_savings']
    total_uplift = ai2['total_uplift']
    base_acos = ai2['baseline_acos']
    target_acos = ai2['new_acos']
    acos_gap = base_acos - target_acos

    # Approximate each module's contribution to ACoS reduction proportionally
    mod_impacts = comp['limitations_solutions']
    total_financial = sum(m.get('ai2_savings', 0) + m.get('ai2_uplift', 0) for m in mod_impacts)
    if total_financial > 0:
        mod_acos_impacts = []
        for m in mod_impacts:
            m_total = m.get('ai2_savings', 0) + m.get('ai2_uplift', 0)
            m_acos = -acos_gap * (m_total / total_financial)
            mod_acos_impacts.append(round(m_acos, 2))
    else:
        mod_acos_impacts = [-acos_gap / 4] * 4

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "relative", "total"],
        x=["当前 ACoS<br>(全局基线)",
           f"🚫 否定词引擎<br>-${mod_impacts[1].get('ai2_savings',0)/1000:.0f}K",
           f"💰 Bid 优化<br>-${mod_impacts[0].get('ai2_savings',0)/1000:.0f}K+${mod_impacts[0].get('ai2_uplift',0)/1000:.0f}K",
           f"🌾 关键词收割<br>+${mod_impacts[2].get('ai2_uplift',0)/1000:.0f}K",
           f"📈 AdTFT<br>+${mod_impacts[3].get('ai2_uplift',0)/1000:.0f}K",
           "AI 2.0 ACoS"],
        y=[base_acos, mod_acos_impacts[1], mod_acos_impacts[0],
           mod_acos_impacts[2], mod_acos_impacts[3], 0],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#4caf50"}},
        increasing={"marker": {"color": "#ef5350"}},
        totals={"marker": {"color": "#1b5e20"}},
        text=[f"{base_acos:.2f}%"] + [f"{v:+.2f}%" for v in [mod_acos_impacts[1], mod_acos_impacts[0],
              mod_acos_impacts[2], mod_acos_impacts[3]]] + [f"{target_acos:.2f}%"],
        textposition="outside",
    ))
    fig.update_layout(title="ACoS 优化瀑布图：各模块贡献", height=420,
                      template="plotly_white", yaxis_title="ACoS (%)",
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ─── Section 7: Summary Table ───
    st.markdown("### 📋 完整对比总结")

    summary_data = {
        '指标': ['ACoS', 'ROAS', 'CPC', 'CTR', 'CVR', '总花费', '总销售额', '总订单数'],
        '👤 人工投放': [
            f"{noai['acos']:.2f}%", f"{noai['roas']:.2f}x", f"${noai['cpc']:.2f}",
            f"{noai['ctr']:.2f}%", f"{noai['cvr']:.2f}%",
            f"${noai['total_spend']:,.0f}", f"${noai['total_sales']:,.0f}",
            f"{noai['total_orders']:,}",
        ],
        '🤖 AI 1.0': [
            f"{ai1['acos']:.2f}%", f"{ai1['roas']:.2f}x", f"${ai1['cpc']:.2f}",
            f"{ai1['ctr']:.2f}%", f"{ai1['cvr']:.2f}%",
            f"${ai1['total_spend']:,.0f}", f"${ai1['total_sales']:,.0f}",
            f"{ai1['total_orders']:,}",
        ],
        '🚀 AI 2.0': [
            f"{ai2['new_acos']:.2f}%", f"{ai2['new_roas']:.2f}x",
            f"优化中", f"优化中",
            f"优化中",
            f"${ai2['net_spend']:,.0f}", f"${ai2['net_sales']:,.0f}",
            f"—",
        ],
        'AI 2.0 vs 1.0': [
            f"↓ {ai1['acos'] - ai2['new_acos']:.2f}pp",
            f"↑ {ai2['new_roas'] - ai1['roas']:.2f}x",
            "Bid优化解决", "全面提升", "Harvest提升",
            f"省 ${ai2['total_savings']:,.0f}",
            f"+${ai2['total_uplift']:,.0f}",
            "增量预估中",
        ],
    }
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ─── Conclusion ───
    st.success(
        f"🎯 **结论**：AI 1.0 在规模化运营方面有价值（CVR {ai1['cvr']:.1f}% 高于人工），"
        f"但存在 **CPC 过高**（${ai1['cpc']:.2f} vs ${noai['cpc']:.2f}）、"
        f"**ACoS 偏高**（{ai1['acos']:.1f}% vs {noai['acos']:.1f}%）的核心问题。"
        f"AI 2.0 通过 4 大模块全面优化，预计 ACoS 从 {ai1['acos']:.1f}% 降至 **{ai2['new_acos']:.1f}%**，"
        f"ROAS 从 {ai1['roas']:.1f}x 提升至 **{ai2['new_roas']:.1f}x**，"
        f"综合利润提升 **${ai2['profit_improvement']:,.0f}**。"
    )

    with st.expander("📐 方法论说明"):
        st.markdown(f"""
        **数据基础：** {comp['n_days']} 天数据，2026-01-13 ~ 2026-03-13

        **人工基线匹配方法：** 从 {comp['noai_all']['n_campaigns']:,} 个非 AI campaigns 中，
        筛选 spend ≥ ${noai['match_threshold']:,.0f}（AI 1.0 campaign P25 水平）的 {noai['n_campaigns']:,} 个 campaigns 作为公平对比基线。

        **AI 2.0 预测方法：** 基于 4 大模型的量化输出（否定词、收割、Bid优化、AdTFT），
        分别计算节省花费和增量收入，采用保守实现率（30%-60%）。

        **时间序列 AI 2.0 线：** 将模型的 ACoS 改善比率（{comp['acos_improvement_ratio']:.1%}）
        和订单增长比率（{comp['orders_uplift_ratio']:.1%}）应用到人工基线的每日数据上。
        """)


# ═══════════════════════════════════════════════════════════════
# PAGE: Feature Analysis (优化特征分析)
# ═══════════════════════════════════════════════════════════════

elif page == "🔍 优化特征分析":
    st.title("🔍 优化特征分析")
    st.markdown("**三种投放管理方法的特征对比与深度分析 — 人工 vs AI 1.0 vs AI 2.0**")

    fa = load_feature_analysis()
    if not fa:
        st.warning("请先运行 `python3 dashboard/prepare_feature_analysis_data.py`")
        st.stop()

    m_data = fa['manual']
    a1_data = fa['ai1']
    a2_data = fa['ai2']
    radar = fa['radar']

    # ── Section 1: Three Profile Cards ──
    st.markdown("### 🎯 三种优化方式画像")
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        st.markdown(
            "<div style='background: linear-gradient(135deg, #f5f5f5, #e0e0e0); "
            "padding: 20px; border-radius: 12px; border-top: 4px solid #9e9e9e;'>"
            "<div style='font-size: 18px; font-weight: bold; color: #424242; text-align:center;'>👤 人工投放</div>"
            "<div style='font-size: 13px; color: #757575; text-align:center; margin-bottom:12px;'>规模最大但效率参差</div>"
            f"<div style='font-size: 13px; color: #424242;'>"
            f"• <b>{m_data['n_campaigns']:,}</b> campaigns, <b>{m_data['n_stores']}</b> stores, <b>{m_data['n_countries']}</b> countries<br>"
            f"• 竞价策略保守: 47% 动态降低<br>"
            f"• ACoS <b>{m_data['acos']}%</b> | ROAS <b>{m_data['roas']}x</b><br>"
            f"• 仅 <b>{m_data['pct_with_spend']}%</b> campaigns 有花费<br>"
            f"• 预算利用率 <b>{m_data['budget']['utilization_mean']}%</b>"
            "</div></div>", unsafe_allow_html=True)
    with pc2:
        st.markdown(
            "<div style='background: linear-gradient(135deg, #fff3e0, #ffe0b2); "
            "padding: 20px; border-radius: 12px; border-top: 4px solid #ff9800;'>"
            "<div style='font-size: 18px; font-weight: bold; color: #e65100; text-align:center;'>🤖 AI 1.0</div>"
            "<div style='font-size: 13px; color: #bf6c00; text-align:center; margin-bottom:12px;'>集中运营但成本偏高</div>"
            f"<div style='font-size: 13px; color: #424242;'>"
            f"• <b>{a1_data['n_campaigns']}</b> campaigns, US only, <b>{a1_data['n_stores']}</b> stores<br>"
            f"• Fixed 竞价 <b>60.4%</b> | 动态提高和降低 <b>27.7%</b><br>"
            f"• ACoS <b>{a1_data['acos']}%</b> | ROAS <b>{a1_data['roas']}x</b><br>"
            f"• <b>{a1_data['pct_with_spend']}%</b> campaigns 有花费<br>"
            f"• 预算利用率 <b>{a1_data['budget']['utilization_mean']}%</b>"
            "</div></div>", unsafe_allow_html=True)
    with pc3:
        st.markdown(
            "<div style='background: linear-gradient(135deg, #e8f5e9, #a5d6a7); "
            "padding: 20px; border-radius: 12px; border-top: 4px solid #4caf50;'>"
            "<div style='font-size: 18px; font-weight: bold; color: #1b5e20; text-align:center;'>🚀 AI 2.0</div>"
            "<div style='font-size: 13px; color: #2e7d32; text-align:center; margin-bottom:12px;'>数据驱动全面优化</div>"
            f"<div style='font-size: 13px; color: #424242;'>"
            f"• 全量 portfolio 覆盖<br>"
            f"• 4 大 ML 模块协同优化<br>"
            f"• ACoS <b>{a2_data['new_acos']}%</b> | ROAS <b>{a2_data['new_roas']}x</b><br>"
            f"• 花费节省 <b>${a2_data['total_savings']:,.0f}</b><br>"
            f"• 利润提升 <b>${a2_data['profit_improvement']:,.0f}</b>"
            "</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Section 2: Radar Chart ──
    st.markdown("### 📊 综合能力雷达图")
    cats = radar['categories'] + [radar['categories'][0]]  # close polygon

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=radar['manual'] + [radar['manual'][0]],
        theta=cats, name='👤 人工投放', fill='toself',
        fillcolor='rgba(158,158,158,0.15)',
        line=dict(color='#9e9e9e', width=2.5),
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=radar['ai1'] + [radar['ai1'][0]],
        theta=cats, name='🤖 AI 1.0', fill='toself',
        fillcolor='rgba(255,152,0,0.15)',
        line=dict(color='#ff9800', width=2.5),
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=radar['ai2'] + [radar['ai2'][0]],
        theta=cats, name='🚀 AI 2.0', fill='toself',
        fillcolor='rgba(76,175,80,0.15)',
        line=dict(color='#4caf50', width=2.5),
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10))),
        height=480, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(t=30),
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.info("💡 **AI 1.0** 在覆盖范围和预算利用方面表现优秀，但成本控制和投放效率明显不足。**AI 2.0** 通过 4 大模块协同优化，在几乎所有维度上达到最高水平。")

    st.markdown("---")

    # ── Section 3: Feature Comparison Table ──
    st.markdown("### 📋 特征维度对比表")

    def cell_color(val, best_is_low=False, vals=None):
        if vals is None:
            return ''
        sorted_v = sorted(vals, reverse=not best_is_low)
        if val == sorted_v[0]:
            return 'background-color: #e8f5e9; color: #1b5e20; font-weight:bold;'
        if val == sorted_v[-1]:
            return 'background-color: #ffebee; color: #c62828;'
        return ''

    table_rows = [
        ('规模', f"{m_data['n_campaigns']:,} campaigns", f"{a1_data['n_campaigns']} campaigns", '全量覆盖'),
        ('市场覆盖', f"{m_data['n_countries']} countries, {m_data['n_stores']} stores", f"1 country, {a1_data['n_stores']} stores", '全部市场'),
        ('竞价策略', '47% 动态降低, 42% Fixed', '60% Fixed, 28% 动态提高和降低', 'ML 优化出价 (Isotonic Regression)'),
        ('预算均值', f"${m_data['budget']['avg_budget']}", f"${a1_data['budget']['avg_budget']}", '动态分配'),
        ('预算利用率', f"{m_data['budget']['utilization_mean']}%", f"{a1_data['budget']['utilization_mean']}%", '自适应优化'),
        ('ACoS', f"{m_data['acos']}%", f"{a1_data['acos']}%", f"{a2_data['new_acos']}%"),
        ('ROAS', f"{m_data['roas']}x", f"{a1_data['roas']}x", f"{a2_data['new_roas']}x"),
        ('CPC', f"${m_data['cpc']}", f"${a1_data['cpc']}", f"~${a2_data['estimated_cpc']}"),
        ('CVR', f"{m_data['cvr']}%", f"{a1_data['cvr']}%", f"~{a2_data['estimated_cvr']}%"),
        ('匹配策略', 'Auto 29%, Broad 17%, ASIN 15%', 'Other 51%, Auto 31%', '+ Harvest Broad->Exact 升级'),
        ('店铺集中度', '分散 (41 stores)', '97.4% 集中在 2 stores', '全量覆盖'),
        ('方法论', '人工经验判断', '规则+预算自动化', '4 模块 ML 系统'),
    ]
    rows_html = ''
    for i, (dim, man, ai1, ai2) in enumerate(table_rows):
        bg = '#fafafa' if i % 2 == 0 else '#ffffff'
        rows_html += f"<tr style='background:{bg};'><td style='padding:8px 12px; font-weight:bold; color:#424242;'>{dim}</td>"
        rows_html += f"<td style='padding:8px 12px; color:#616161;'>{man}</td>"
        rows_html += f"<td style='padding:8px 12px; color:#e65100;'>{ai1}</td>"
        rows_html += f"<td style='padding:8px 12px; color:#1b5e20; font-weight:bold;'>{ai2}</td></tr>"

    st.markdown(f"""
    <table style='width:100%; border-collapse:collapse; border:1px solid #e0e0e0; border-radius:8px; overflow:hidden;'>
    <thead><tr style='background: linear-gradient(135deg, #37474f, #455a64); color:white;'>
        <th style='padding:10px 12px; text-align:left;'>维度</th>
        <th style='padding:10px 12px; text-align:left;'>👤 人工投放</th>
        <th style='padding:10px 12px; text-align:left;'>🤖 AI 1.0</th>
        <th style='padding:10px 12px; text-align:left;'>🚀 AI 2.0</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Section 4: Bidding & Targeting ──
    st.markdown("### 💡 竞价策略与匹配类型深度分析")
    bt1, bt2 = st.columns(2)

    with bt1:
        strategies = ['固定竞价', '动态竞价-仅降低', '动态竞价-提高和降低']
        approaches = ['🚀 AI 2.0', '🤖 AI 1.0', '👤 人工']
        colors = {'固定竞价': '#78909c', '动态竞价-仅降低': '#4db6ac', '动态竞价-提高和降低': '#ef5350'}
        fig_bid = go.Figure()
        for strat in strategies:
            fig_bid.add_trace(go.Bar(
                y=approaches,
                x=[
                    0,
                    a1_data['bidding_strategy'].get(strat, 0),
                    m_data['bidding_strategy'].get(strat, 0),
                ],
                name=strat, orientation='h',
                marker_color=colors.get(strat, '#90a4ae'),
                text=[
                    'ML',
                    f"{a1_data['bidding_strategy'].get(strat, 0)}%",
                    f"{m_data['bidding_strategy'].get(strat, 0)}%",
                ],
                textposition='inside',
            ))
        fig_bid.update_layout(barmode='stack', height=350, template='plotly_white',
                              title='竞价策略分布 (按花费占比)',
                              xaxis_title='占比 (%)',
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                              margin=dict(l=10))
        st.plotly_chart(fig_bid, use_container_width=True)

    with bt2:
        targeting_types = ['Auto', 'Broad', 'Exact', 'ASIN', 'Phrase', 'Other']
        t_colors = {'Auto': '#42a5f5', 'Broad': '#66bb6a', 'Exact': '#ab47bc',
                    'ASIN': '#ffa726', 'Phrase': '#26c6da', 'Other': '#bdbdbd'}
        fig_tgt = go.Figure()
        for ttype in targeting_types:
            fig_tgt.add_trace(go.Bar(
                y=['🚀 AI 2.0', '🤖 AI 1.0', '👤 人工'],
                x=[
                    a2_data['targeting_mix'].get(ttype, 0),
                    a1_data['targeting_mix'].get(ttype, 0),
                    m_data['targeting_mix'].get(ttype, 0),
                ],
                name=ttype, orientation='h',
                marker_color=t_colors.get(ttype, '#90a4ae'),
            ))
        fig_tgt.update_layout(barmode='stack', height=350, template='plotly_white',
                              title='匹配类型分布 (按花费占比)',
                              xaxis_title='占比 (%)',
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                              margin=dict(l=10))
        st.plotly_chart(fig_tgt, use_container_width=True)

    st.warning("⚠️ **AI 1.0 竞价问题**: 60.4% 使用 Fixed 竞价 (限制 Amazon 算法优化空间) + 27.7% 使用动态提高和降低 (CPC 被推高)。人工投放 47.2% 使用动态降低策略，反而更保守、更安全。")
    st.info("📌 **AI 1.0 匹配问题**: 51.2% 花费在 '其他/未分类' 匹配类型上，缺乏结构化关键词策略。人工投放有更多样化的匹配组合 (Auto 29% + Broad 17% + ASIN 15%)。")
    st.success("✅ **AI 2.0 方案**: Bid Landscape 用 Isotonic Regression 为每个 targeting 找最优出价; Harvest 将 4,558 个高转化 Broad/Phrase 词升级到 Exact 匹配。")

    st.markdown("---")

    # ── Section 5: ACoS Distribution Box Plot ──
    st.markdown("### 📊 Campaign 级 ACoS 分布对比")
    st.caption("基于有花费且有销售的 campaign 级聚合数据")

    m_dist = m_data.get('acos_distribution', {})
    a1_dist = a1_data.get('acos_distribution', {})

    if m_dist and a1_dist:
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            q1=[m_dist['p25']], median=[m_dist['median']], q3=[m_dist['p75']],
            lowerfence=[m_dist['p10']], upperfence=[m_dist['p90']],
            mean=[m_dist['mean']],
            name=f"👤 人工 ({m_dist['n_campaigns']:,} campaigns)",
            marker_color='#9e9e9e', fillcolor='rgba(158,158,158,0.3)',
            boxpoints=False,
        ))
        fig_box.add_trace(go.Box(
            q1=[a1_dist['p25']], median=[a1_dist['median']], q3=[a1_dist['p75']],
            lowerfence=[a1_dist['p10']], upperfence=[a1_dist['p90']],
            mean=[a1_dist['mean']],
            name=f"🤖 AI 1.0 ({a1_dist['n_campaigns']} campaigns)",
            marker_color='#ff9800', fillcolor='rgba(255,152,0,0.3)',
            boxpoints=False,
        ))
        fig_box.update_layout(
            title="Campaign 级 ACoS 分布 (P10-P25-Median-P75-P90)",
            yaxis_title="ACoS (%)", height=420, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_box, use_container_width=True)

        bx1, bx2 = st.columns(2)
        iqr_m = m_dist['p75'] - m_dist['p25']
        iqr_a = a1_dist['p75'] - a1_dist['p25']
        with bx1:
            st.markdown(f"**👤 人工投放**: 中位数 **{m_dist['median']}%**, IQR = {iqr_m:.1f}pp — 分布较宽，表现参差不齐")
        with bx2:
            st.markdown(f"**🤖 AI 1.0**: 中位数 **{a1_dist['median']}%**, IQR = {iqr_a:.1f}pp — 分布更集中但中心偏高")

    st.markdown("---")

    # ── Section 6: Budget & Scale ──
    st.markdown("### 💰 预算与规模分析")
    bg1, bg2, bg3 = st.columns(3)

    with bg1:
        m_tiers = m_data['budget']['tier_distribution']
        a1_tiers = a1_data['budget']['tier_distribution']
        tier_labels = ['$0-5', '$5-10', '$10-20', '$20-50', '$50+']
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Bar(name='👤 人工', x=tier_labels,
                                y=[m_tiers.get(t, 0) for t in tier_labels],
                                marker_color='#bdbdbd'))
        fig_bt.add_trace(go.Bar(name='🤖 AI 1.0', x=tier_labels,
                                y=[a1_tiers.get(t, 0) for t in tier_labels],
                                marker_color='#ffb74d'))
        fig_bt.update_layout(barmode='group', title='预算分层分布', height=380,
                             template='plotly_white', yaxis_title='占比 (%)',
                             legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_bt, use_container_width=True)

    with bg2:
        fig_util = go.Figure()
        fig_util.add_trace(go.Bar(
            x=['👤 人工', '🤖 AI 1.0'],
            y=[m_data['budget']['utilization_mean'], a1_data['budget']['utilization_mean']],
            marker_color=['#9e9e9e', '#ff9800'],
            text=[f"{m_data['budget']['utilization_mean']}%", f"{a1_data['budget']['utilization_mean']}%"],
            textposition='outside',
        ))
        fig_util.add_hline(y=95, line_dash='dot', line_color='red',
                           annotation_text='触顶线 (95%)', annotation_position='top right')
        fig_util.update_layout(title='预算利用率', height=380,
                               template='plotly_white', yaxis_title='利用率 (%)',
                               yaxis_range=[0, 110])
        st.plotly_chart(fig_util, use_container_width=True)

    with bg3:
        m_del = m_data['delivery_status']
        a1_del = a1_data['delivery_status']
        status_list = ['投放中', '超出预算', '已暂停', '广告活动未完成']
        s_colors = {'投放中': '#66bb6a', '超出预算': '#ef5350', '已暂停': '#ffa726', '广告活动未完成': '#78909c'}
        fig_status = go.Figure()
        for s in status_list:
            fig_status.add_trace(go.Bar(
                x=['👤 人工', '🤖 AI 1.0'],
                y=[m_del.get(s, 0), a1_del.get(s, 0)],
                name=s, marker_color=s_colors.get(s, '#bdbdbd'),
            ))
        fig_status.update_layout(barmode='stack', title='投放状态分布', height=380,
                                 template='plotly_white', yaxis_title='占比 (%)',
                                 legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_status, use_container_width=True)

    cap_pct = a1_data['budget']['pct_at_budget_cap']
    st.warning(f"⚠️ AI 1.0 有 **{cap_pct}%** 的天数预算利用率达到 95%+ — campaign 提前耗尽预算，错失后续转化机会。AI 2.0 的 AdTFT 模块预测未来趋势，提前优化预算分配，避免触顶。")

    st.markdown("---")

    # ── Section 7: AI 2.0 Module Summary ──
    st.markdown("### 🚀 AI 2.0 四大模块：逐项攻克 AI 1.0 弱点")

    problems = [
        ('零转化搜索词持续浪费广告花费', '20,883 个零转化高花费词未被否定'),
        ('CPC 过高，竞价缺乏数据支撑', '60% Fixed 竞价 + 28% 激进策略推高点击成本'),
        ('高转化词停留在 Broad/Phrase 匹配', '仅 3.4% 花费在 Exact 匹配，错失低 CPC 机会'),
        ('缺乏前瞻性，只能事后调整', '基于昨天数据做决策，无法预判趋势变化'),
    ]

    for i, mod in enumerate(a2_data.get('modules', [])):
        prob_title, prob_detail = problems[i] if i < len(problems) else ('', '')
        savings = mod.get('savings', 0)
        uplift = mod.get('uplift', 0)
        impact_parts = []
        if savings > 0:
            impact_parts.append(f"省 ${savings:,.0f}")
        if uplift > 0:
            impact_parts.append(f"+${uplift:,.0f}")
        impact_str = ' | '.join(impact_parts) if impact_parts else '--'

        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown(
                f"<div style='background: linear-gradient(135deg, #ffebee, #ffcdd2); "
                f"padding: 16px; border-radius: 10px; border-left: 4px solid #ef5350; height:100%;'>"
                f"<div style='font-size: 14px; font-weight: bold; color: #c62828;'>❌ {prob_title}</div>"
                f"<div style='font-size: 12px; color: #616161; margin-top: 6px;'>{prob_detail}</div>"
                f"</div>", unsafe_allow_html=True)
        with mc2:
            st.markdown(
                f"<div style='background: linear-gradient(135deg, #e8f5e9, #c8e6c9); "
                f"padding: 16px; border-radius: 10px; border-left: 4px solid #4caf50; height:100%;'>"
                f"<div style='font-size: 14px; font-weight: bold; color: #1b5e20;'>"
                f"{mod.get('icon', '')} {mod.get('name', '')} — {impact_str}</div>"
                f"<div style='font-size: 12px; color: #424242; margin-top: 6px;'>"
                f"{'<br>'.join(mod.get('why_better', []))}</div>"
                f"</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    total_save = a2_data.get('total_savings', 0)
    total_up = a2_data.get('total_uplift', 0)
    total_profit = a2_data.get('profit_improvement', 0)
    st.success(
        f"🎯 **AI 2.0 综合影响**: "
        f"节省花费 **${total_save:,.0f}** + 增量收入 **${total_up:,.0f}** = "
        f"利润提升 **${total_profit:,.0f}**  |  "
        f"ACoS: 9.95% → **{a2_data['new_acos']}%** (-2.93pp)  |  "
        f"ROAS: 10.05x → **{a2_data['new_roas']}x** (+4.19x)"
    )

    with st.expander("📄 方法论说明"):
        st.markdown("""
        **数据来源**: `campaign_daily_clean.csv` — 60 天, 157,701 campaigns

        **人工投放**: ai_status = 'AI未开启' 的 156,794 campaigns (全量，包含不活跃 campaigns)

        **AI 1.0**: ai_status = 'AI运行中' 的 361 campaigns (Xnurta AI 当前版本)

        **AI 2.0**: 基于四大模块 (否定词引擎、Bid Landscape、关键词收割、AdTFT) 的模拟预测值，
        应用于全量数据的 100% adoption 场景。

        **竞价/匹配分布**: 按花费金额加权计算，而非按 campaign 数量。

        **ACoS 分布**: 仅包含有花费且有销售的 campaigns，避免无销售 campaigns 导致的 ACoS=Inf 噪声。

        **雷达图归一化**: 使用绝对锚点 (如 ACoS 5%-15%, ROAS 5x-15x, CPC $0.3-$1.0) 而非相对排名。
        """)


# ═══════════════════════════════════════════════════════════════
# PAGE: AI Impact Simulator
# ═══════════════════════════════════════════════════════════════

elif page == "🤖 AI 效果模拟器":
    st.title("🤖 AI 效果模拟器")
    st.markdown("**开启不同比例的 AI 优化能力，对整体广告效果的影响预估**")

    sim_df = load_ai_impact_simulation()
    details = load_ai_impact_details()

    if sim_df.empty or not details:
        st.error("AI 效果数据未准备。请先运行: python3 dashboard/prepare_ai_impact_data.py")
        st.stop()

    baseline = details.get('baseline', {})
    modules = details.get('modules', {})

    # ─── AI Adoption Slider ───
    st.markdown("### 🎛️ AI 开启率")
    adoption_pct = st.slider(
        "拖动调整 AI 优化开启比例",
        min_value=0, max_value=100, value=50, step=10,
        format="%d%%",
        help="模拟将 AI 优化应用到不同比例的 campaign/targeting 上的效果"
    )
    adoption_rate = adoption_pct / 100

    # Get the matching row from simulation data
    sim_row = sim_df.iloc[(sim_df['adoption_rate'] - adoption_rate).abs().idxmin()]

    st.markdown("---")

    # ─── Impact KPI Cards ───
    st.markdown("### 📊 效果预估")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div style='background: linear-gradient(135deg, #e8f5e9, #c8e6c9); padding: 20px; "
            f"border-radius: 12px; text-align: center;'>"
            f"<div style='color: #2e7d32; font-size: 14px;'>💰 节省花费</div>"
            f"<div style='color: #1b5e20; font-size: 32px; font-weight: bold;'>"
            f"${sim_row['total_savings']:,.0f}</div>"
            f"<div style='color: #388e3c; font-size: 13px;'>占总花费 "
            f"{sim_row['total_savings']/max(baseline['total_spend'],1)*100:.1f}%</div>"
            f"</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(
            f"<div style='background: linear-gradient(135deg, #e3f2fd, #bbdefb); padding: 20px; "
            f"border-radius: 12px; text-align: center;'>"
            f"<div style='color: #1565c0; font-size: 14px;'>📈 增加收入</div>"
            f"<div style='color: #0d47a1; font-size: 32px; font-weight: bold;'>"
            f"${sim_row['total_uplift']:,.0f}</div>"
            f"<div style='color: #1976d2; font-size: 13px;'>占总销售额 "
            f"{sim_row['total_uplift']/max(baseline['total_sales'],1)*100:.1f}%</div>"
            f"</div>", unsafe_allow_html=True)

    with c3:
        st.markdown(
            f"<div style='background: linear-gradient(135deg, #fff3e0, #ffe0b2); padding: 20px; "
            f"border-radius: 12px; text-align: center;'>"
            f"<div style='color: #e65100; font-size: 14px;'>🎯 综合利润提升</div>"
            f"<div style='color: #bf360c; font-size: 32px; font-weight: bold;'>"
            f"${sim_row['profit_improvement']:,.0f}</div>"
            f"<div style='color: #d84315; font-size: 13px;'>= 节省 + 增收 - 额外支出</div>"
            f"</div>", unsafe_allow_html=True)

    st.markdown("")

    # ACoS / ROAS improvement
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("当前 ACoS", f"{baseline['acos']:.2f}%")
    c2.metric("优化后 ACoS", f"{sim_row['new_acos']:.2f}%",
              delta=f"{sim_row['acos_improvement']:+.2f}pp", delta_color="inverse")
    c3.metric("当前 ROAS", f"{baseline['roas']:.1f}x")
    c4.metric("优化后 ROAS", f"{sim_row['new_roas']:.1f}x",
              delta=f"{sim_row['roas_improvement']:+.1f}x")

    st.markdown("---")

    # ─── Impact by Adoption Rate Chart ───
    st.markdown("### 📈 AI 开启率 vs 效果提升")

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sim_df['adoption_rate'] * 100, y=sim_df['total_savings'],
            name='节省花费', marker_color='#4caf50', opacity=0.85
        ))
        fig.add_trace(go.Bar(
            x=sim_df['adoption_rate'] * 100, y=sim_df['total_uplift'],
            name='增加收入', marker_color='#2196f3', opacity=0.85
        ))
        # Highlight current selection
        fig.add_vline(x=adoption_pct, line_dash="dash", line_color="red",
                     annotation_text=f"当前: {adoption_pct}%")
        fig.update_layout(
            title="各开启率下的节省 & 增收",
            xaxis_title="AI 开启率 (%)", yaxis_title="金额 ($)",
            height=400, template="plotly_white",
            barmode='group',
            xaxis=dict(tickvals=list(range(0, 110, 10)), ticksuffix="%"),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=sim_df['adoption_rate'] * 100, y=sim_df['new_acos'],
            name='ACoS (%)', mode='lines+markers',
            line=dict(color='#ef5350', width=3),
            fill='tozeroy', fillcolor='rgba(239,83,80,0.1)'
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=sim_df['adoption_rate'] * 100, y=sim_df['new_roas'],
            name='ROAS (x)', mode='lines+markers',
            line=dict(color='#42a5f5', width=3)
        ), secondary_y=True)
        fig.add_hline(y=baseline['acos'], line_dash="dot", line_color="gray",
                     annotation_text=f"当前 ACoS {baseline['acos']:.1f}%", secondary_y=False)
        fig.update_layout(
            title="ACoS & ROAS 随 AI 开启率变化",
            xaxis_title="AI 开启率 (%)",
            height=400, template="plotly_white",
            xaxis=dict(tickvals=list(range(0, 110, 10)), ticksuffix="%"),
        )
        fig.update_yaxes(title_text="ACoS (%)", secondary_y=False)
        fig.update_yaxes(title_text="ROAS (x)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ─── Module Breakdown ───
    st.markdown("### 🧩 各 AI 模块贡献分解")
    st.caption(f"以下按当前 {adoption_pct}% 开启率计算")

    # Savings breakdown
    neg_save = sim_row['negation_savings']
    bid_save = sim_row['bid_savings']
    har_save = sim_row['harvest_savings']
    har_uplift = sim_row['harvest_uplift']
    bid_uplift = sim_row['bid_uplift']
    adtft_uplift = sim_row['adtft_uplift']

    col1, col2 = st.columns(2)

    with col1:
        # Savings waterfall
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=["🚫 否定词<br>节省花费", "💰 Bid优化<br>节省花费", "🌾 收割<br>ACoS优化", "合计节省"],
            y=[neg_save, bid_save, har_save, 0],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#4caf50"}},
            totals={"marker": {"color": "#2e7d32"}},
            text=[f"${neg_save:,.0f}", f"${bid_save:,.0f}", f"${har_save:,.0f}",
                  f"${neg_save+bid_save+har_save:,.0f}"],
            textposition="outside",
        ))
        fig.update_layout(title="💰 节省花费分解", height=400, template="plotly_white",
                         yaxis_title="节省金额 ($)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Revenue uplift waterfall
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=["🌾 收割<br>增量销售", "💰 Bid加价<br>增量销售", "📈 AdTFT<br>预算优化", "合计增收"],
            y=[har_uplift, bid_uplift, adtft_uplift, 0],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#2196f3"}},
            totals={"marker": {"color": "#0d47a1"}},
            text=[f"${har_uplift:,.0f}", f"${bid_uplift:,.0f}", f"${adtft_uplift:,.0f}",
                  f"${har_uplift+bid_uplift+adtft_uplift:,.0f}"],
            textposition="outside",
        ))
        fig.update_layout(title="📈 增加收入分解", height=400, template="plotly_white",
                         yaxis_title="增收金额 ($)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ─── Why AI is Better: Module by Module ───
    st.markdown("### 🧠 为什么 AI 比人工优化更好？")

    combined_reasons = details.get('combined_why_better', [])
    if combined_reasons:
        cols = st.columns(len(combined_reasons))
        for i, reason in enumerate(combined_reasons):
            with cols[i]:
                st.markdown(f"<div style='background: #f5f5f5; padding: 12px; border-radius: 8px; "
                           f"font-size: 13px; min-height: 100px; text-align: center;'>{reason}</div>",
                           unsafe_allow_html=True)

    st.markdown("")

    # Module detail cards
    for mod_key, mod in modules.items():
        with st.expander(f"{mod['icon']} {mod['name']} — 为什么 AI 更强？", expanded=False):
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("**核心数据**")
                if mod_key == 'negation':
                    st.metric("可安全否定", f"{mod['n_green']:,} 词")
                    st.metric("需审核", f"{mod['n_yellow']:,} 词")
                    st.metric("节省花费", f"${mod['savings_at_100pct']:,.0f}")
                elif mod_key == 'harvest':
                    st.metric("收割候选", f"{mod['n_candidates']:,} 词")
                    st.metric("新增 Exact", f"{mod['n_new']:,} 词")
                    st.metric("增量销售", f"${mod['incremental_sales_at_100pct']:,.0f}")
                elif mod_key == 'bid_landscape':
                    st.metric("建议降价", f"{mod['n_decrease']:,} 条")
                    st.metric("建议加价", f"{mod['n_increase']:,} 条")
                    st.metric("平均多出价", f"${mod['avg_overbid']:.3f}")
                elif mod_key == 'adtft':
                    st.metric("ROAS 提升", f"{mod['roas_improvement_pct']:.0f}%")
                    st.metric("增量销售", f"${mod['incremental_sales_at_100pct']:,.0f}")

            with col2:
                st.markdown("**AI 优势 vs 人工**")
                for reason in mod['why_better']:
                    st.markdown(f"✅ {reason}")
                st.markdown(f"\n**方法论:** {mod.get('methodology', '')}")

    st.markdown("---")

    # ─── Summary Table ───
    st.markdown("### 📋 完整效果对比表")
    display_df = sim_df[['adoption_pct', 'total_savings', 'total_uplift',
                          'profit_improvement', 'new_acos', 'new_roas',
                          'acos_improvement', 'roas_improvement']].copy()
    display_df.columns = ['AI 开启率', '节省花费 ($)', '增加收入 ($)',
                          '利润提升 ($)', '优化后 ACoS (%)', '优化后 ROAS',
                          'ACoS 改善 (pp)', 'ROAS 改善']
    st.dataframe(
        display_df.style.format({
            '节省花费 ($)': '{:,.0f}',
            '增加收入 ($)': '{:,.0f}',
            '利润提升 ($)': '{:,.0f}',
            '优化后 ACoS (%)': '{:.2f}',
            '优化后 ROAS': '{:.2f}',
            'ACoS 改善 (pp)': '{:+.2f}',
            'ROAS 改善': '{:+.2f}',
        }).background_gradient(subset=['利润提升 ($)'], cmap='Greens'),
        use_container_width=True, hide_index=True
    )

    st.markdown("---")

    # ─── Methodology Note ───
    with st.expander("📐 计算方法说明", expanded=False):
        st.markdown(f"""
        **基准数据**: {baseline.get('n_days', 60)} 天 · 总花费 ${baseline['total_spend']:,.0f} · 总销售 ${baseline['total_sales']:,.0f}

        **各模块计算逻辑：**

        | 模块 | 节省/增收来源 | 计算方法 | 保守系数 |
        |------|-------------|---------|---------|
        | 🚫 否定词 | 消除零转化/极端 ACoS 词的浪费花费 | 绿灯 100% + 黄灯 60% 实现率 | 黄灯 60% |
        | 🌾 收割 | 新 Exact 匹配带来增量销售 + 现有词 ACoS 改善 | 新词 20% 增量 + 现有词 15% ACoS 优化 | 20% / 15% |
        | 💰 Bid优化 | 降低过度出价节省花费 + 提高不足出价增加销售 | 花费占比 × 相对过度出价 × 30% 实现率 | 30% |
        | 📈 AdTFT | 预测性预算分配提升整体 ROAS | 保守估计 5% ROAS 提升（行业基准 5-12%） | 5% |

        **说明：**
        - 所有估算均取保守值，实际效果可能更好
        - "AI 开启率"模拟将 AI 建议应用到对应比例的 targeting/campaign 上
        - 各模块独立计算，总效果为线性叠加（实际上模块间可能有正交互效应）
        """)


# ═══════════════════════════════════════════════════════════════
# PAGE: AI Effect Prediction (AI 效果预测)
# ═══════════════════════════════════════════════════════════════

elif page == "📤 AI 效果预测":
    st.title("📤 AI 效果预测")
    st.markdown("**上传新的 Campaign 数据，预测 AI 2.0 托管 30 天的效果提升**")

    # Load improvement ratios from simulation data
    sim_df = load_ai_impact_simulation()
    if sim_df.empty:
        st.warning("缺少 AI 模拟数据，请先生成 ai_impact_simulation.csv")
        st.stop()

    row100 = sim_df[sim_df['adoption_rate'] == 1.0].iloc[0]
    baseline_spend = row100['net_spend'] + row100['total_savings']  # original spend
    savings_ratio = row100['total_savings'] / max(baseline_spend, 1)
    uplift_ratio = row100['total_uplift'] / max(row100['net_sales'] - row100['total_uplift'], 1)

    # Module-level ratios
    neg_save_ratio = row100['negation_savings'] / max(baseline_spend, 1)
    bid_save_ratio = row100['bid_savings'] / max(baseline_spend, 1)
    harv_save_ratio = row100['harvest_savings'] / max(baseline_spend, 1)
    bid_uplift_ratio = row100['bid_uplift'] / max(row100['net_sales'] - row100['total_uplift'], 1)
    harv_uplift_ratio = row100['harvest_uplift'] / max(row100['net_sales'] - row100['total_uplift'], 1)
    adtft_uplift_ratio = row100['adtft_uplift'] / max(row100['net_sales'] - row100['total_uplift'], 1)

    st.markdown("---")

    # ── Section 1: Upload ──
    st.markdown("### 📁 上传 Campaign 数据")
    st.markdown("""
    CSV 文件需要包含以下列（与系统导出格式一致）：
    `date`, `campaign`, `spend`, `sales`, `orders`, `clicks`, `impressions`

    可选列：`budget`, `acos`, `roas`, `cpc`, `cvr`
    """)

    uploaded_file = st.file_uploader("选择 CSV 文件", type=['csv'], key='prediction_upload')

    if uploaded_file is not None:
        try:
            udf = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"CSV 读取错误: {e}")
            st.stop()

        # Validate required columns
        required_cols = ['date', 'campaign', 'spend', 'sales']
        missing = [c for c in required_cols if c not in udf.columns]
        if missing:
            st.error(f"缺少必要列: {missing}. 请确保 CSV 包含: {required_cols}")
            st.stop()

        # Fill optional numeric columns
        for col in ['orders', 'clicks', 'impressions']:
            if col not in udf.columns:
                udf[col] = 0

        udf['date'] = pd.to_datetime(udf['date'], errors='coerce')
        udf = udf.dropna(subset=['date'])

        st.success(f"✅ 数据加载成功!")

        # ── Section 2: Current Performance ──
        st.markdown("### 📊 当前数据概览")

        n_camps = udf['campaign'].nunique()
        date_range = f"{udf['date'].min().strftime('%Y-%m-%d')} ~ {udf['date'].max().strftime('%Y-%m-%d')}"
        n_days = udf['date'].nunique()
        total_spend = udf['spend'].sum()
        total_sales = udf['sales'].sum()
        total_orders = udf['orders'].sum()
        current_acos = total_spend / max(total_sales, 1) * 100
        current_roas = total_sales / max(total_spend, 1)

        ov1, ov2, ov3, ov4 = st.columns(4)
        ov1.metric("Campaigns", f"{n_camps:,}")
        ov2.metric("日期范围", f"{n_days} 天")
        ov3.metric("总花费", f"${total_spend:,.0f}")
        ov4.metric("总销售", f"${total_sales:,.0f}")

        km1, km2, km3, km4 = st.columns(4)
        km1.metric("当前 ACoS", f"{current_acos:.2f}%")
        km2.metric("当前 ROAS", f"{current_roas:.2f}x")
        km3.metric("总订单", f"{total_orders:,.0f}")
        km4.metric("日均花费", f"${total_spend/max(n_days,1):,.0f}")

        with st.expander("查看数据预览"):
            st.dataframe(udf.head(20), use_container_width=True)

        st.markdown("---")

        # ── Section 3: Prediction ──
        st.markdown("### 🚀 AI 2.0 托管 30 天预测")
        st.markdown("基于 Xnurta 2.0 四大模块的改进比率，预测 AI 托管后的效果提升。")

        # Compute daily averages from uploaded data
        daily_avg = udf.groupby('date').agg(
            spend=('spend', 'sum'),
            sales=('sales', 'sum'),
            orders=('orders', 'sum'),
            clicks=('clicks', 'sum'),
        ).reset_index()
        daily_avg = daily_avg.sort_values('date')

        avg_daily_spend = daily_avg['spend'].mean()
        avg_daily_sales = daily_avg['sales'].mean()
        avg_daily_orders = daily_avg['orders'].mean()

        # 30-day projection
        pred_days = 30
        pred_daily_spend = avg_daily_spend * (1 - savings_ratio)
        pred_daily_sales = avg_daily_sales * (1 + uplift_ratio)
        pred_daily_orders = avg_daily_orders * (1 + uplift_ratio * 0.5)  # orders uplift is more conservative

        pred_total_spend = pred_daily_spend * pred_days
        pred_total_sales = pred_daily_sales * pred_days
        pred_total_orders = pred_daily_orders * pred_days
        pred_acos = pred_total_spend / max(pred_total_sales, 1) * 100
        pred_roas = pred_total_sales / max(pred_total_spend, 1)

        curr_30d_spend = avg_daily_spend * pred_days
        curr_30d_sales = avg_daily_sales * pred_days
        curr_30d_orders = avg_daily_orders * pred_days

        # KPI cards with delta
        st.markdown("#### 30 天预测 vs 当前趋势")
        pk1, pk2, pk3, pk4 = st.columns(4)
        pk1.metric("预测月花费", f"${pred_total_spend:,.0f}",
                   delta=f"-${curr_30d_spend - pred_total_spend:,.0f}", delta_color="inverse")
        pk2.metric("预测月销售", f"${pred_total_sales:,.0f}",
                   delta=f"+${pred_total_sales - curr_30d_sales:,.0f}")
        pk3.metric("预测 ACoS", f"{pred_acos:.2f}%",
                   delta=f"{pred_acos - current_acos:.2f}pp", delta_color="inverse")
        pk4.metric("预测 ROAS", f"{pred_roas:.2f}x",
                   delta=f"+{pred_roas - current_roas:.2f}x")

        profit_delta = (pred_total_sales - pred_total_spend) - (curr_30d_sales - curr_30d_spend)
        st.success(f"💰 **30 天预估利润提升: ${profit_delta:,.0f}** (花费节省 ${curr_30d_spend - pred_total_spend:,.0f} + 收入增长 ${pred_total_sales - curr_30d_sales:,.0f})")

        # ── 30-day time series chart ──
        st.markdown("#### 📈 30 天趋势预测")

        # Generate 30-day forecast dates
        last_date = daily_avg['date'].max()
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=pred_days, freq='D')

        # Add day-of-week pattern variation
        np.random.seed(42)
        dow_pattern = [1.0, 1.02, 1.05, 1.03, 0.98, 0.92, 0.95]  # Mon-Sun
        noise = np.random.normal(1.0, 0.03, pred_days)

        curr_forecast_spend = []
        curr_forecast_sales = []
        ai_forecast_spend = []
        ai_forecast_sales = []

        for i, d in enumerate(forecast_dates):
            dow_mult = dow_pattern[d.weekday()] * noise[i]
            curr_forecast_spend.append(avg_daily_spend * dow_mult)
            curr_forecast_sales.append(avg_daily_sales * dow_mult)
            ai_forecast_spend.append(pred_daily_spend * dow_mult)
            ai_forecast_sales.append(pred_daily_sales * dow_mult)

        tc1, tc2 = st.columns(2)
        with tc1:
            fig_spend = go.Figure()
            fig_spend.add_trace(go.Scatter(
                x=forecast_dates, y=curr_forecast_spend,
                name='当前趋势', mode='lines',
                line=dict(color='#9e9e9e', width=2.5),
            ))
            fig_spend.add_trace(go.Scatter(
                x=forecast_dates, y=ai_forecast_spend,
                name='AI 2.0 预测', mode='lines',
                line=dict(color='#4caf50', width=2.5, dash='dash'),
            ))
            fig_spend.update_layout(
                title='日花费趋势 (30天)', height=380, template='plotly_white',
                yaxis_title='花费 ($)', xaxis_title='日期',
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                yaxis=dict(rangemode='tozero'),
            )
            st.plotly_chart(fig_spend, use_container_width=True)

        with tc2:
            fig_sales = go.Figure()
            fig_sales.add_trace(go.Scatter(
                x=forecast_dates, y=curr_forecast_sales,
                name='当前趋势', mode='lines',
                line=dict(color='#9e9e9e', width=2.5),
            ))
            fig_sales.add_trace(go.Scatter(
                x=forecast_dates, y=ai_forecast_sales,
                name='AI 2.0 预测', mode='lines',
                line=dict(color='#4caf50', width=2.5, dash='dash'),
            ))
            fig_sales.update_layout(
                title='日销售趋势 (30天)', height=380, template='plotly_white',
                yaxis_title='销售 ($)', xaxis_title='日期',
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                yaxis=dict(rangemode='tozero'),
            )
            st.plotly_chart(fig_sales, use_container_width=True)

        st.markdown("---")

        # ── Section 4: Module Waterfall ──
        st.markdown("### 🧩 模块级优化分解")

        neg_save = curr_30d_spend * neg_save_ratio
        bid_save = curr_30d_spend * bid_save_ratio
        harv_save = curr_30d_spend * harv_save_ratio
        bid_up = curr_30d_sales * bid_uplift_ratio
        harv_up = curr_30d_sales * harv_uplift_ratio
        adtft_up = curr_30d_sales * adtft_uplift_ratio

        wf1, wf2 = st.columns(2)
        with wf1:
            fig_wf = go.Figure(go.Waterfall(
                name="ACoS", orientation="v",
                measure=["absolute", "relative", "relative", "relative", "relative", "total"],
                x=["当前 ACoS", "🚫 否定词引擎", "💰 Bid 优化", "🌾 关键词收割", "📈 AdTFT", "AI 2.0 ACoS"],
                y=[current_acos,
                   -(current_acos * neg_save_ratio * 0.8),
                   -(current_acos * bid_save_ratio * 0.7),
                   -(current_acos * harv_save_ratio * 0.5),
                   -(current_acos * adtft_uplift_ratio * 0.3),
                   0],
                connector={"line": {"color": "#e0e0e0"}},
                decreasing={"marker": {"color": "#66bb6a"}},
                increasing={"marker": {"color": "#ef5350"}},
                totals={"marker": {"color": "#1b5e20"}},
                text=[f"{current_acos:.1f}%", "", "", "", "", f"{pred_acos:.1f}%"],
                textposition="outside",
            ))
            fig_wf.update_layout(title="ACoS 优化瀑布图", height=400, template="plotly_white",
                                 yaxis_title="ACoS (%)")
            st.plotly_chart(fig_wf, use_container_width=True)

        with wf2:
            module_table = pd.DataFrame({
                '模块': ['🚫 否定词引擎', '💰 Bid 竞价优化', '🌾 关键词收割', '📈 AdTFT 预测', '合计'],
                '花费节省': [f"${neg_save:,.0f}", f"${bid_save:,.0f}", f"${harv_save:,.0f}", '--',
                           f"${neg_save + bid_save + harv_save:,.0f}"],
                '收入提升': ['--', f"${bid_up:,.0f}", f"${harv_up:,.0f}", f"${adtft_up:,.0f}",
                           f"${bid_up + harv_up + adtft_up:,.0f}"],
                '利润贡献': [f"${neg_save:,.0f}", f"${bid_save + bid_up:,.0f}",
                           f"${harv_save + harv_up:,.0f}", f"${adtft_up:,.0f}",
                           f"${profit_delta:,.0f}"],
            })
            st.markdown("**各模块 30 天预估贡献**")
            st.dataframe(module_table, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Section 5: Download ──
        st.markdown("### 📥 下载预测报告")

        report_df = pd.DataFrame({
            'date': forecast_dates,
            'current_spend': curr_forecast_spend,
            'ai_predicted_spend': ai_forecast_spend,
            'spend_savings': [c - a for c, a in zip(curr_forecast_spend, ai_forecast_spend)],
            'current_sales': curr_forecast_sales,
            'ai_predicted_sales': ai_forecast_sales,
            'sales_uplift': [a - c for c, a in zip(curr_forecast_sales, ai_forecast_sales)],
        })
        report_df['current_acos'] = report_df['current_spend'] / report_df['current_sales'].clip(lower=0.01) * 100
        report_df['ai_predicted_acos'] = report_df['ai_predicted_spend'] / report_df['ai_predicted_sales'].clip(lower=0.01) * 100

        csv_data = report_df.to_csv(index=False)
        st.download_button(
            label="📥 下载 30 天预测 CSV",
            data=csv_data,
            file_name="ai2_prediction_30day.csv",
            mime="text/csv",
        )

        st.info(f"""
        **预测方法说明**:
        基于上传数据的日均表现，应用 Xnurta 2.0 四大模块的改进比率:
        - 花费节省比率: {savings_ratio:.1%} (否定词 {neg_save_ratio:.1%} + Bid {bid_save_ratio:.1%} + Harvest {harv_save_ratio:.1%})
        - 收入提升比率: {uplift_ratio:.1%} (Bid {bid_uplift_ratio:.1%} + Harvest {harv_uplift_ratio:.1%} + AdTFT {adtft_uplift_ratio:.1%})
        - 30天加入星期效应波动模拟
        """)

    else:
        st.markdown("---")
        st.markdown("""
        <div style='background: linear-gradient(135deg, #e3f2fd, #bbdefb); padding: 40px; border-radius: 16px;
                    text-align: center; border: 2px dashed #42a5f5;'>
            <div style='font-size: 48px;'>📤</div>
            <div style='font-size: 20px; font-weight: bold; color: #1565c0; margin-top: 12px;'>上传 Campaign CSV 开始预测</div>
            <div style='font-size: 14px; color: #616161; margin-top: 8px;'>
                支持格式: campaign_daily_clean.csv 或类似格式<br>
                必须包含: date, campaign, spend, sales 列
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: AdTFT Model
# ═══════════════════════════════════════════════════════════════

elif page == "📈 AdTFT 预测模型":
    st.title("📈 AdTFT 时序预测模型")
    st.markdown("**Temporal Fusion Transformer · 分位数预测 · 可解释特征重要性**")

    training_results = load_training_results()
    meta = load_feature_metadata()

    if not training_results:
        st.error("训练结果文件未找到")
        st.stop()

    tm = training_results.get('test_metrics', {})
    config = training_results.get('config', {})

    # ─── Model Architecture ───
    st.markdown("### 🏗️ 模型架构")
    st.markdown("""
    ```
    Static Features ──→ Linear ──→ Context Vector ──┐
                                                     │
    Observed Features → FeatureAttention → Projection ─┤
                                                       ├→ GRU Encoder → Temporal Attention → Pool → GRN → Quantile Heads
    Known Features ──→ FeatureAttention → Projection ──┘
    ```
    """)

    # ─── Model Config ───
    with st.expander("⚙️ 模型配置", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.json({
            'hidden_dim': config.get('hidden_dim'),
            'n_heads': config.get('n_heads'),
            'dropout': config.get('dropout'),
            'parameters': f"{training_results.get('n_params', 0):,}",
        })
        c2.json({
            'lookback_window': config.get('lookback_window'),
            'max_horizon': config.get('max_horizon'),
            'quantiles': config.get('quantiles'),
            'batch_size': config.get('batch_size'),
        })
        c3.json({
            'learning_rate': config.get('learning_rate'),
            'max_epochs': config.get('max_epochs'),
            'patience': config.get('patience'),
            'best_epoch': training_results.get('best_epoch'),
            'total_time_min': round(training_results.get('total_time', 0) / 60, 1),
        })

    # ─── Test Metrics ───
    st.markdown("### 🎯 测试集评估")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test Quantile Loss", f"{tm.get('loss', 0):.4f}")
    c2.metric("Overall MAE", f"{tm.get('mae', 0):.3f}")
    c3.metric("P10-P90 Coverage", f"{tm.get('coverage_80', 0):.1f}%",
              help="分位数校准：80%预测区间应覆盖约80%真实值")
    c4.metric("P25-P75 Coverage", f"{tm.get('coverage_50', 0):.1f}%",
              help="50%预测区间应覆盖约50%真实值")

    # Calibration assessment
    cov80 = tm.get('coverage_80', 0)
    cov50 = tm.get('coverage_50', 0)
    if cov80 > 75 and cov80 < 85:
        st.success(f"✅ 量化校准优秀: P10-P90 覆盖率 {cov80:.1f}% (目标 80%), P25-P75 覆盖率 {cov50:.1f}% (目标 50%)")
    elif cov80 > 70:
        st.info(f"ℹ️ 量化校准良好: P10-P90 覆盖率 {cov80:.1f}%")
    else:
        st.warning(f"⚠️ 量化校准偏差: P10-P90 覆盖率 {cov80:.1f}%")

    # ─── Per-target Metrics Table ───
    st.markdown("### 📊 各预测目标精度")

    target_data = []
    for horizon in ['1d', '3d', '7d']:
        for metric_name in ['spend', 'sales', 'orders', 'acos']:
            name = f'{metric_name}_{horizon}'
            target_data.append({
                '指标': metric_name.upper(),
                '预测步长': horizon,
                'MAE': round(tm.get(f'{name}_mae', 0), 4),
                'RMSE': round(tm.get(f'{name}_rmse', 0), 4),
                'MAPE (%)': round(tm.get(f'{name}_mape', 0), 1),
            })

    target_df = pd.DataFrame(target_data)

    st.dataframe(
        target_df.style.background_gradient(subset=['MAPE (%)'], cmap='RdYlGn_r', vmin=0, vmax=100),
        use_container_width=True, hide_index=True
    )

    # ─── MAPE Heatmap ───
    st.markdown("### 🗺️ MAPE 热力图")
    pivot = target_df.pivot(index='指标', columns='预测步长', values='MAPE (%)')
    pivot = pivot[['1d', '3d', '7d']]

    fig = px.imshow(pivot, text_auto='.1f', color_continuous_scale='RdYlGn_r',
                    labels=dict(color="MAPE %"), aspect="auto")
    fig.update_layout(height=300, template="plotly_white",
                     title="预测误差热力图 (MAPE % — 越低越好)")
    st.plotly_chart(fig, use_container_width=True)

    # ─── MAPE by Horizon ───
    st.markdown("### 📏 MAPE 随预测步长变化")
    fig = px.bar(target_df, x='指标', y='MAPE (%)', color='预测步长',
                 barmode='group', color_discrete_sequence=['#636EFA', '#EF553B', '#00CC96'],
                 title="各指标在不同预测步长下的 MAPE")
    fig.update_layout(height=350, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # ─── Training Curve ───
    st.markdown("### 📉 训练曲线")
    history = training_results.get('training_history', {})
    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])

    if train_loss:
        col1, col2 = st.columns([2, 1])

        with col1:
            fig = go.Figure()
            epochs = list(range(1, len(train_loss) + 1))
            fig.add_trace(go.Scatter(x=epochs, y=train_loss, name='Train Loss',
                                     mode='lines+markers'))
            fig.add_trace(go.Scatter(x=epochs, y=val_loss, name='Val Loss',
                                     mode='lines+markers'))
            best_epoch = training_results.get('best_epoch', 0)
            fig.add_vline(x=best_epoch, line_dash="dash", line_color="green",
                         annotation_text=f"Best Epoch {best_epoch}")
            fig.update_layout(height=400, template="plotly_white",
                             xaxis_title="Epoch", yaxis_title="Quantile Loss")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**训练指标**")
            st.write(f"- 总 Epoch: {len(train_loss)}")
            st.write(f"- Best Epoch: {best_epoch}")
            st.write(f"- Best Val Loss: {training_results.get('best_val_loss', 0):.6f}")
            st.write(f"- Final Train Loss: {train_loss[-1]:.6f}")
            st.write(f"- Final Val Loss: {val_loss[-1]:.6f}")
            overfitting = (val_loss[-1] - train_loss[-1]) / train_loss[-1] * 100
            st.write(f"- 过拟合指标: {overfitting:.1f}%")

    # ─── Feature Importance ───
    st.markdown("### 🔍 特征重要性")
    if meta:
        c1, c2, c3 = st.columns(3)
        c1.metric("观察特征", f"{len(meta.get('observed_features', []))}")
        c2.metric("已知未来特征", f"{len(meta.get('known_future_features', []))}")
        c3.metric("静态特征", f"{len(meta.get('static_features', []))}")

        st.info("💡 **Top 特征 (Feature Attention 权重):**\n"
                "- `orders_ma7d` (12.1%) — 近期订单移动平均\n"
                "- `spend_lag1` (3.7%) — 前日花费\n"
                "- `impressions_momentum14d` (3.0%) — 曝光动量\n"
                "- 模型最关注近期订单趋势和花费惯性")

        with st.expander("📋 完整特征列表", expanded=False):
            obs = meta.get('observed_features', [])
            known = meta.get('known_future_features', [])
            static = meta.get('static_features', [])
            st.markdown(f"**观察特征 ({len(obs)}):**")
            st.text(", ".join(obs[:30]) + (f" ... (+{len(obs)-30} more)" if len(obs) > 30 else ""))
            st.markdown(f"**已知未来特征 ({len(known)}):**")
            st.text(", ".join(known))
            st.markdown(f"**静态特征 ({len(static)}):**")
            st.text(", ".join(static))


# ═══════════════════════════════════════════════════════════════
# PAGE: Semantic Engine
# ═══════════════════════════════════════════════════════════════

elif page == "🔤 语义引擎":
    st.title("🔤 搜索词语义智能引擎")
    st.markdown("**Sentence Transformer Embedding + HDBSCAN 密度聚类 + 语义分析**")

    clusters = load_semantic_clusters()
    cluster_analysis = load_cluster_analysis()

    if clusters.empty:
        st.error("语义聚类数据未找到")
        st.stop()

    # ─── Overview ───
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("语义簇", f"{len(clusters)}")
    total_terms = int(clusters['n_terms'].sum()) if 'n_terms' in clusters.columns else 0
    c2.metric("聚类搜索词", f"{total_terms:,}")
    total_spend = clusters['total_spend'].sum() if 'total_spend' in clusters.columns else 0
    c3.metric("覆盖花费", f"${total_spend:,.0f}")

    if 'classification' in clusters.columns:
        good = clusters['classification'].str.contains('GOOD', na=False).sum()
        c4.metric("优质簇 (ACoS<15%)", f"{good}")

    # ─── Classification Distribution ───
    if 'classification' in clusters.columns:
        st.markdown("### 📊 簇分类分布")
        col1, col2 = st.columns(2)

        with col1:
            class_dist = clusters['classification'].value_counts().reset_index()
            class_dist.columns = ['classification', 'count']
            fig = px.pie(class_dist, values='count', names='classification',
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         title="语义簇 ACoS 分类 (按数量)")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            class_spend = clusters.groupby('classification')['total_spend'].sum().reset_index()
            class_spend.columns = ['classification', 'total_spend']
            fig = px.pie(class_spend, values='total_spend', names='classification',
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         title="语义簇 ACoS 分类 (按花费)")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # ─── Spend vs ACoS Scatter ───
    if 'acos' in clusters.columns and 'total_spend' in clusters.columns:
        st.markdown("### 🗺️ 簇表现地图 (花费 vs ACoS)")
        plot_df = clusters[clusters['total_spend'] > 0].copy()
        plot_df['acos_pct'] = plot_df['acos'] * 100

        fig = px.scatter(plot_df, x='total_spend', y='acos_pct',
                         size='n_terms', color='classification',
                         hover_data=['representative_terms', 'total_orders'],
                         log_x=True,
                         labels={'total_spend': '总花费 ($)', 'acos_pct': 'ACoS (%)'},
                         title="语义簇地图: 花费 vs ACoS (气泡大小 = 词数)")
        fig.add_hline(y=15, line_dash="dash", line_color="green",
                     annotation_text="ACoS 15% 目标线")
        fig.add_hline(y=30, line_dash="dash", line_color="orange",
                     annotation_text="ACoS 30%")
        fig.update_layout(height=500, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # ─── Top Clusters by Spend ───
    st.markdown("### 💰 Top 簇（按花费排序）")
    display_cols = ['semantic_cluster', 'n_terms', 'total_spend', 'total_sales',
                    'total_orders', 'acos', 'cvr', 'classification', 'representative_terms']
    available_cols = [c for c in display_cols if c in clusters.columns]
    top_clusters = clusters.sort_values('total_spend', ascending=False).head(30)
    st.dataframe(top_clusters[available_cols], use_container_width=True, hide_index=True)

    # ─── TF-IDF Clusters ───
    if not cluster_analysis.empty:
        with st.expander("📝 TF-IDF 文本聚类详情", expanded=False):
            st.dataframe(cluster_analysis.head(30), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: Negation Recommendations
# ═══════════════════════════════════════════════════════════════

elif page == "🚫 否定词推荐":
    st.title("🚫 否定词推荐引擎")
    st.markdown("**自动识别低效/无效搜索词，推荐安全否定**")

    negation = load_negation_recs()

    if negation.empty:
        st.error("否定词推荐数据未找到")
        st.stop()

    # ─── Summary ───
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("否定候选总数", f"{len(negation):,}")
    if 'safety_level' in negation.columns:
        green = int((negation['safety_level'] == 'GREEN_SAFE_TO_NEGATE').sum())
        yellow = int((negation['safety_level'] == 'YELLOW_NEEDS_REVIEW').sum())
        c2.metric("🟢 绿灯（可安全否定）", f"{green:,}")
        c3.metric("🟡 黄灯（需审核）", f"{yellow:,}")
    total_save = negation['total_spend'].sum() if 'total_spend' in negation.columns else 0
    c4.metric("💰 可节省花费", f"${total_save:,.0f}")

    st.markdown("---")

    # ─── Reason Distribution (show first) ───
    if 'negate_reason' in negation.columns:
        st.markdown("### 📊 否定原因分布")
        col1, col2 = st.columns(2)

        with col1:
            reason_dist = negation.groupby('negate_reason').agg(
                count=('search_term_clean', 'count'),
                total_spend=('total_spend', 'sum')
            ).sort_values('total_spend', ascending=False).reset_index()

            fig = px.bar(reason_dist, x='negate_reason', y='total_spend',
                         color='count', text='count',
                         labels={'negate_reason': '否定原因', 'total_spend': '涉及花费 ($)'},
                         title="各否定原因的花费分布")
            fig.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if 'safety_level' in negation.columns:
                safety_reason = negation.groupby(['negate_reason', 'safety_level']).size().reset_index(name='count')
                fig = px.bar(safety_reason, x='negate_reason', y='count',
                             color='safety_level',
                             color_discrete_map={
                                 'GREEN_SAFE_TO_NEGATE': '#2ecc71',
                                 'YELLOW_NEEDS_REVIEW': '#f39c12'
                             },
                             title="安全等级 × 否定原因",
                             labels={'negate_reason': '否定原因', 'count': '数量'})
                fig.update_layout(height=400, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

    # ─── Filters ───
    st.markdown("### 🔍 筛选否定词")
    col1, col2 = st.columns(2)

    with col1:
        safety_options = ["全部"]
        if 'safety_level' in negation.columns:
            safety_options += sorted(negation['safety_level'].unique().tolist())
        safety_filter = st.selectbox("安全等级", safety_options)

    with col2:
        reason_options = ["全部"]
        if 'negate_reason' in negation.columns:
            reason_options += sorted(negation['negate_reason'].unique().tolist())
        reason_filter = st.selectbox("否定原因", reason_options)

    filtered = negation.copy()
    if safety_filter != "全部" and 'safety_level' in filtered.columns:
        filtered = filtered[filtered['safety_level'] == safety_filter]
    if reason_filter != "全部" and 'negate_reason' in filtered.columns:
        filtered = filtered[filtered['negate_reason'] == reason_filter]

    st.markdown(f"**显示 {len(filtered):,} / {len(negation):,} 条结果**")

    # ─── Table ───
    display_cols = ['search_term_clean', 'total_spend', 'total_sales', 'total_orders',
                    'acos', 'cvr', 'negate_reason', 'confidence', 'safety_level']
    available = [c for c in display_cols if c in filtered.columns]
    sorted_df = filtered[available].sort_values('total_spend', ascending=False).head(200)
    st.dataframe(sorted_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: Harvest Recommendations
# ═══════════════════════════════════════════════════════════════

elif page == "🌾 关键词收割":
    st.title("🌾 关键词收割推荐")
    st.markdown("**发现高转化搜索词，推荐精确匹配投放**")

    harvest = load_harvest_recs()

    if harvest.empty:
        st.error("收割推荐数据未找到")
        st.stop()

    # ─── Summary ───
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("收割候选", f"{len(harvest):,}")
    total_sales = harvest['total_sales'].sum() if 'total_sales' in harvest.columns else 0
    c2.metric("潜在销售额", f"${total_sales:,.0f}")
    total_orders = int(harvest['total_orders'].sum()) if 'total_orders' in harvest.columns else 0
    c3.metric("潜在订单", f"{total_orders:,}")
    if 'has_exact' in harvest.columns:
        new_exact = int((~harvest['has_exact']).sum())
        c4.metric("新增精确匹配", f"{new_exact:,}")

    st.markdown("---")

    # ─── ACoS Distribution ───
    if 'acos' in harvest.columns:
        st.markdown("### 📊 收割词 ACoS 分布")
        col1, col2 = st.columns(2)

        with col1:
            plot_df = harvest[harvest['acos'] > 0].copy()
            plot_df['acos_pct'] = plot_df['acos'] * 100
            fig = px.histogram(plot_df, x='acos_pct', nbins=50,
                              labels={'acos_pct': 'ACoS (%)'},
                              title="收割候选词 ACoS 分布")
            fig.add_vline(x=15, line_dash="dash", line_color="green",
                         annotation_text="ACoS 15%")
            fig.add_vline(x=30, line_dash="dash", line_color="orange",
                         annotation_text="ACoS 30%")
            fig.update_layout(height=350, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if 'total_sales' in harvest.columns and 'total_spend' in harvest.columns:
                top_h = harvest.nlargest(50, 'total_sales')
                top_h_plot = top_h.copy()
                top_h_plot['acos_pct'] = top_h_plot['acos'] * 100
                fig = px.scatter(top_h_plot, x='total_spend', y='total_sales',
                                size='total_orders', color='acos_pct',
                                color_continuous_scale='RdYlGn_r',
                                hover_data=['search_term_clean'],
                                labels={'total_spend': '花费', 'total_sales': '销售额',
                                        'acos_pct': 'ACoS %'},
                                title="Top 50 收割词 (花费 vs 销售额)")
                fig.update_layout(height=350, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

    # ─── Filter ───
    st.markdown("### 🔍 筛选收割词")
    exact_filter = st.selectbox("匹配状态",
                                 ["全部", "新词（未有精确匹配）", "已有精确匹配"])

    filtered = harvest.copy()
    if exact_filter == "新词（未有精确匹配）" and 'has_exact' in filtered.columns:
        filtered = filtered[~filtered['has_exact']]
    elif exact_filter == "已有精确匹配" and 'has_exact' in filtered.columns:
        filtered = filtered[filtered['has_exact']]

    st.markdown(f"**显示 {len(filtered):,} / {len(harvest):,} 条**")

    # ─── Table ───
    display_cols = ['search_term_clean', 'total_spend', 'total_sales', 'total_orders',
                    'acos', 'cvr', 'cpc', 'has_exact']
    available = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[available].sort_values('total_sales', ascending=False).head(200),
        use_container_width=True, hide_index=True
    )


# ═══════════════════════════════════════════════════════════════
# PAGE: Bid Landscape
# ═══════════════════════════════════════════════════════════════

elif page == "💰 Bid Landscape":
    st.title("💰 Bid Landscape 竞价分析")
    st.markdown("**Cross-Sectional 竞价响应曲线 · 最优出价区间 · 调价建议**")

    bl_results = load_bid_landscape_results()
    bl_curves = load_bid_landscape_curves()
    bl_recs = load_bid_recommendations()

    if not bl_results:
        st.error("Bid Landscape 模型未训练。请先运行: python3 models/bid_landscape/bid_landscape_model.py")
        st.stop()

    # ─── Overview KPIs ───
    st.markdown("### 📊 模型概览")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("分析记录数", f"{bl_results.get('n_records', 0):,}")
    c2.metric("细分市场", f"{bl_results.get('n_segments', 0)}")
    c3.metric("调价建议", f"{bl_results.get('n_recommendations', 0):,}")

    action_dist = bl_results.get('action_distribution', {})
    decrease_pct = action_dist.get('decrease', 0) / max(bl_results.get('n_recommendations', 1), 1) * 100
    c4.metric("建议降价", f"{decrease_pct:.0f}%", delta=f"{action_dist.get('decrease', 0):,} 条")
    increase_pct = action_dist.get('increase', 0) / max(bl_results.get('n_recommendations', 1), 1) * 100
    c5.metric("建议加价", f"{increase_pct:.0f}%", delta=f"{action_dist.get('increase', 0):,} 条")

    st.markdown("---")

    # ─── Global Bid Response Curves ───
    st.markdown("### 📈 全局竞价响应曲线")
    st.caption("基于 500K 投放记录的跨切面分析 — 不同出价水平对应的平均表现")

    global_curve = bl_results.get('global_curve', {})

    if global_curve and 'bid' in global_curve:
        gc_df = pd.DataFrame(global_curve)

        col1, col2 = st.columns(2)

        with col1:
            # Bid → Impressions & Clicks
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(
                x=gc_df['bid'], y=gc_df['impressions'],
                name='平均曝光量', mode='lines+markers',
                line=dict(color='#636EFA', width=3),
                marker=dict(size=gc_df['n_samples'] / gc_df['n_samples'].max() * 15 + 3)
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=gc_df['bid'], y=gc_df['clicks'],
                name='平均点击量', mode='lines+markers',
                line=dict(color='#00CC96', width=3)
            ), secondary_y=True)
            fig.update_layout(
                title="出价 → 曝光量 & 点击量",
                xaxis_title="出价 ($)",
                height=420, template="plotly_white",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            fig.update_yaxes(title_text="平均曝光量", secondary_y=False)
            fig.update_yaxes(title_text="平均点击量", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Bid → Orders & CVR
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(
                x=gc_df['bid'], y=gc_df['orders'],
                name='平均订单数', mode='lines+markers',
                line=dict(color='#EF553B', width=3)
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=gc_df['bid'], y=[v * 100 for v in gc_df['cvr']],
                name='CVR (%)', mode='lines+markers',
                line=dict(color='#AB63FA', width=3)
            ), secondary_y=True)
            fig.update_layout(
                title="出价 → 订单量 & 转化率",
                xaxis_title="出价 ($)",
                height=420, template="plotly_white",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            fig.update_yaxes(title_text="平均订单数", secondary_y=False)
            fig.update_yaxes(title_text="CVR (%)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

        # Efficiency curve: Impressions per bid dollar
        st.markdown("### ⚡ 出价效率曲线")
        col1, col2 = st.columns(2)

        with col1:
            gc_df['imp_per_bid'] = gc_df['impressions'] / gc_df['bid'].clip(lower=0.01)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=gc_df['bid'], y=gc_df['imp_per_bid'],
                name='曝光/出价', mode='lines+markers',
                fill='tozeroy', fillcolor='rgba(99,110,250,0.15)',
                line=dict(color='#636EFA', width=3)
            ))
            # Mark the peak efficiency
            peak_idx = gc_df['imp_per_bid'].idxmax()
            peak_bid = gc_df.loc[peak_idx, 'bid']
            peak_val = gc_df.loc[peak_idx, 'imp_per_bid']
            fig.add_annotation(x=peak_bid, y=peak_val,
                              text=f"最优效率 ${peak_bid:.2f}",
                              showarrow=True, arrowhead=2, bgcolor="#FFE066")
            fig.update_layout(
                title="曝光效率 (Impressions per Bid Dollar)",
                xaxis_title="出价 ($)", yaxis_title="曝光/出价 ($)",
                height=400, template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            gc_df['ctr_pct'] = [v * 100 for v in gc_df['ctr']]
            gc_df['conv_pct'] = [v * 100 for v in gc_df['conv_rate']]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=gc_df['bid'], y=gc_df['ctr_pct'],
                name='CTR (%)', mode='lines+markers',
                line=dict(color='#00CC96', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=gc_df['bid'], y=gc_df['conv_pct'],
                name='转化率 (%)', mode='lines+markers',
                line=dict(color='#EF553B', width=3)
            ))
            fig.update_layout(
                title="出价 → CTR & 转化率",
                xaxis_title="出价 ($)", yaxis_title="百分比 (%)",
                height=400, template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ─── Segment Analysis ───
    st.markdown("### 🌍 细分市场竞价分析")

    segment_summary = bl_results.get('segment_summary', {})
    optimal_bids = bl_results.get('optimal_bids', {})

    if segment_summary:
        seg_df = pd.DataFrame([
            {
                '市场': k,
                '国家': v['country'],
                '匹配类型': v['match_type'],
                '记录数': v['n_records'],
                '出价中位数': v['bid_median'],
                '平均曝光': v['avg_impressions'],
                '平均订单': v['avg_orders'],
                '平均CTR': round(v['avg_ctr'] * 100, 2),
                '平均CVR': round(v['avg_cvr'] * 100, 2),
            }
            for k, v in segment_summary.items()
        ]).sort_values('记录数', ascending=False)

        # Country filter
        seg_countries = sorted(seg_df['国家'].unique())
        sel_country = st.selectbox("🌍 选择国家 (Bid Landscape)", ["全部"] + seg_countries)

        if sel_country != "全部":
            seg_df = seg_df[seg_df['国家'] == sel_country]

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(seg_df, x='市场', y='出价中位数',
                         color='匹配类型',
                         color_discrete_map={'BROAD': '#636EFA', 'PHRASE': '#EF553B', 'EXACT': '#00CC96'},
                         title="各市场出价中位数",
                         labels={'出价中位数': '出价 ($)'})
            fig.update_layout(height=400, template="plotly_white",
                             xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(seg_df, x='出价中位数', y='平均曝光',
                             size='记录数', color='匹配类型',
                             hover_data=['市场', '平均订单', '平均CTR'],
                             color_discrete_map={'BROAD': '#636EFA', 'PHRASE': '#EF553B', 'EXACT': '#00CC96'},
                             title="出价 vs 曝光 (各市场)",
                             labels={'出价中位数': '出价中位数 ($)', '平均曝光': '平均曝光量'})
            fig.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(seg_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ─── Segment Bid Response Curves ───
    st.markdown("### 📉 细分市场竞价响应曲线")

    seg_curves = bl_curves.get('segment_curves', {}) if bl_curves else {}

    if seg_curves:
        seg_options = sorted(seg_curves.keys())
        selected_seg = st.selectbox("选择市场", seg_options,
                                     index=seg_options.index('US_EXACT') if 'US_EXACT' in seg_options else 0)

        seg_c = seg_curves[selected_seg]
        cdata = seg_c['curve_data']

        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cdata['bid'], y=cdata['impressions'],
                name='曝光量 (isotonic)', mode='lines+markers',
                line=dict(color='#636EFA', width=3),
                fill='tozeroy', fillcolor='rgba(99,110,250,0.1)'
            ))

            # Mark current median
            median_bid = seg_c['bid_median']
            fig.add_vline(x=median_bid, line_dash="dash", line_color="red",
                         annotation_text=f"当前中位数 ${median_bid:.2f}")

            # Mark optimal zone if available
            if selected_seg in optimal_bids:
                opt = optimal_bids[selected_seg]
                rec_range = opt.get('recommended_range', [])
                if len(rec_range) == 2:
                    fig.add_vrect(x0=rec_range[0], x1=rec_range[1],
                                  fillcolor="rgba(0,204,150,0.15)", line_width=0,
                                  annotation_text="推荐区间")

            fig.update_layout(
                title=f"{selected_seg} — 出价 → 曝光量",
                xaxis_title="出价 ($)", yaxis_title="平均曝光量",
                height=400, template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cdata['bid'], y=cdata['orders'],
                name='订单量', mode='lines+markers',
                line=dict(color='#EF553B', width=3),
                fill='tozeroy', fillcolor='rgba(239,85,59,0.1)'
            ))
            fig.add_vline(x=median_bid, line_dash="dash", line_color="red",
                         annotation_text=f"当前中位数 ${median_bid:.2f}")
            fig.update_layout(
                title=f"{selected_seg} — 出价 → 订单量",
                xaxis_title="出价 ($)", yaxis_title="平均订单数",
                height=400, template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Segment metrics
        st.caption(f"📊 {selected_seg}: {seg_c['n_records']:,} 条记录 | "
                   f"出价范围: ${seg_c['bid_range'][0]:.2f} - ${seg_c['bid_range'][1]:.2f} | "
                   f"中位数: ${seg_c['bid_median']:.2f}")

    st.markdown("---")

    # ─── Bid Adjustment Recommendations ───
    st.markdown("### 💡 调价建议")

    if not bl_recs.empty:
        col1, col2 = st.columns(2)

        with col1:
            # Action distribution pie chart
            action_counts = bl_recs['action'].value_counts().reset_index()
            action_counts.columns = ['action', 'count']
            fig = px.pie(action_counts, values='count', names='action',
                         color='action',
                         color_discrete_map={
                             'decrease': '#EF553B', 'maintain': '#636EFA', 'increase': '#00CC96'
                         },
                         title="调价建议分布")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Bid change distribution
            bl_recs['bid_change'] = bl_recs['recommended_bid'] - bl_recs['current_bid']
            change_df = bl_recs[bl_recs['bid_change'] != 0].copy()
            if len(change_df) > 0:
                fig = px.histogram(change_df, x='bid_change', nbins=50,
                                   color='action',
                                   color_discrete_map={
                                       'decrease': '#EF553B', 'increase': '#00CC96'
                                   },
                                   title="建议调价幅度分布",
                                   labels={'bid_change': '建议调价 ($)'})
                fig.add_vline(x=0, line_dash="solid", line_color="black")
                fig.update_layout(height=350, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

        # Filter recommendations
        st.markdown("### 🔍 筛选调价建议")
        fc1, fc2, fc3 = st.columns(3)

        with fc1:
            action_filter = st.selectbox("建议动作", ["全部", "decrease", "maintain", "increase"])
        with fc2:
            country_filter_bl = st.selectbox("国家 (BL)", ["全部"] + sorted(bl_recs['country'].unique().tolist()))
        with fc3:
            match_filter_bl = st.selectbox("匹配类型 (BL)", ["全部"] + sorted(bl_recs['match_type'].unique().tolist()))

        filtered_recs = bl_recs.copy()
        if action_filter != "全部":
            filtered_recs = filtered_recs[filtered_recs['action'] == action_filter]
        if country_filter_bl != "全部":
            filtered_recs = filtered_recs[filtered_recs['country'] == country_filter_bl]
        if match_filter_bl != "全部":
            filtered_recs = filtered_recs[filtered_recs['match_type'] == match_filter_bl]

        st.markdown(f"**显示 {len(filtered_recs):,} / {len(bl_recs):,} 条结果**")

        display_cols_bl = ['keyword', 'country', 'match_type', 'current_bid',
                           'recommended_bid', 'action', 'impressions', 'orders', 'acos']
        available_bl = [c for c in display_cols_bl if c in filtered_recs.columns]
        st.dataframe(
            filtered_recs[available_bl].sort_values('impressions', ascending=False).head(200),
            use_container_width=True, hide_index=True
        )

    st.markdown("---")

    # ─── Model Details ───
    with st.expander("⚙️ 模型详情 & 方法论", expanded=False):
        st.markdown("""
        **建模方法**: Cross-Sectional Bid Landscape

        由于缺少 targeting 级别的每日时序数据，采用横截面分析方法：
        - 利用 **500K 投放记录**中不同出价水平的表现差异来估计竞价响应曲线
        - 按 **国家 × 匹配类型** 分段建模，共 28 个细分市场
        - 使用 **Isotonic Regression** 保证单调性约束（出价越高，曝光应越多）
        - 使用 **参数化曲线拟合**（Log / Power / Hill 函数）建立全局响应函数
        - 训练 **GradientBoosting 模型**进行细粒度预测

        **局限性**:
        - 横截面数据无法完全控制 keyword 异质性（不同关键词天然有不同表现）
        - R² 偏低（0.05-0.10）说明 bid 只是影响表现的众多因素之一
        - 待有 targeting 每日数据后，可升级为时序 bid landscape 模型

        **调价建议逻辑**:
        1. 计算每个细分市场的"效率甜蜜点"（曝光/出价 最大化的 bid 水平）
        2. 计算边际回报递减点（marginal impressions 下降到平均的 30%）
        3. 推荐区间 = [甜蜜点 - 1 bin, 递减点 + 1 bin]
        4. 当前出价低于推荐区间 → 建议加价；高于 → 建议降价
        """)

        # Curve fitting results
        curve_fit = bl_results.get('curve_fitting', {})
        if curve_fit:
            st.markdown("**参数化曲线拟合 R² 值:**")
            fit_data = []
            for metric, info in curve_fit.items():
                param = info.get('parametric', {})
                row = {
                    '指标': info.get('label', metric),
                    '最佳函数': param.get('function', 'N/A'),
                    'R²': param.get('r2', 0),
                }
                if 'isotonic_r2' in info:
                    row['Isotonic R²'] = info['isotonic_r2']
                fit_data.append(row)
            st.dataframe(pd.DataFrame(fit_data), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: Campaign Analysis
# ═══════════════════════════════════════════════════════════════

elif page == "📊 Campaign 分析":
    st.title("📊 Campaign 表现分析")

    country_agg = load_country_agg()
    country_daily = load_country_daily()
    top_campaigns = load_top_campaigns()

    if country_agg.empty:
        st.error("Campaign 数据未加载。请先运行: python3 dashboard/prepare_dashboard_data.py")
        st.stop()

    # ─── Country Filter ───
    countries = sorted(country_agg['country'].unique())
    selected_country = st.selectbox("🌍 选择国家", ["全部"] + countries)

    # ─── Country Breakdown ───
    st.markdown("### 🌍 各国家表现")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(country_agg, x='country', y=['spend', 'sales'],
                     barmode='group', title="各国家花费 vs 销售额",
                     labels={'value': '金额 ($)', 'country': '国家'},
                     color_discrete_sequence=['#636EFA', '#00CC96'])
        fig.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(country_agg, x='country', y='acos',
                     title="各国家 ACoS (%)",
                     labels={'acos': 'ACoS (%)', 'country': '国家'},
                     color='acos', color_continuous_scale='RdYlGn_r')
        fig.add_hline(y=20, line_dash="dash", line_color="green",
                     annotation_text="ACoS 20%")
        fig.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(country_agg, use_container_width=True, hide_index=True)

    # ─── Daily Trend for Selected Country ───
    st.markdown(f"### 📅 每日趋势 ({selected_country})")

    if not country_daily.empty:
        if selected_country != "全部":
            daily_filtered = country_daily[country_daily['country'] == selected_country]
        else:
            daily_filtered = country_daily.groupby('date').agg(
                spend=('spend', 'sum'),
                sales=('sales', 'sum'),
                orders=('orders', 'sum'),
                impressions=('impressions', 'sum'),
                clicks=('clicks', 'sum'),
            ).reset_index()
            daily_filtered['acos'] = daily_filtered['spend'] / daily_filtered['sales'].clip(lower=0.01) * 100
            daily_filtered['ctr'] = daily_filtered['clicks'] / daily_filtered['impressions'].clip(lower=1) * 100
            daily_filtered['cvr'] = daily_filtered['orders'] / daily_filtered['clicks'].clip(lower=1) * 100

        metric_choice = st.selectbox("选择指标", ['spend', 'sales', 'orders', 'acos', 'ctr', 'cvr'])

        fig = px.line(daily_filtered, x='date', y=metric_choice,
                      title=f"{metric_choice.upper()} 每日趋势",
                      labels={'date': '日期', metric_choice: metric_choice.upper()})
        fig.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # ─── Top Campaigns ───
    st.markdown(f"### 🏆 Top Campaigns ({selected_country})")
    sort_metric = st.selectbox("排序方式", ['spend', 'sales', 'orders', 'roas'])

    if not top_campaigns.empty:
        camp_filtered = top_campaigns.copy()
        if selected_country != "全部":
            camp_filtered = camp_filtered[camp_filtered['country'] == selected_country]

        camp_filtered = camp_filtered.sort_values(sort_metric, ascending=False).head(50)
        st.dataframe(camp_filtered, use_container_width=True, hide_index=True)


# ─── Footer ───
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "🚀 Xnurta 2.0 AI Engine · "
    "Temporal Fusion Transformer + Sentence Transformer + HDBSCAN · "
    "Built for Amazon Ads Optimization"
    "</div>",
    unsafe_allow_html=True
)
