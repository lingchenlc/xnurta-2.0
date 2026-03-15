#!/usr/bin/env python3
"""
Prepare AI 1.0 vs AI 2.0 comparison data for the dashboard.
Computes:
  - AI 1.0 aggregate metrics (campaigns with ai_status='AI运行中')
  - Controlled non-AI baseline (scale-matched campaigns)
  - AI 2.0 projected metrics (from ai_impact_simulation)
  - Daily time-series for AI 1.0, non-AI, and AI 2.0 projected
  - Bidding strategy distributions
  - Limitation-to-solution mappings
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
DASHBOARD_DATA = Path(__file__).resolve().parent / "data"

print("=" * 60)
print("AI 1.0 vs AI 2.0 Comparison Data Preparation")
print("=" * 60)

# ─── Load source data ───
print("\n[1/7] Loading campaign_daily_clean.csv ...")
cols = ['date', 'country', 'store', 'campaign', 'bidding_strategy',
        'ai_status', 'impressions', 'clicks', 'spend', 'orders', 'sales']
df = pd.read_csv(DATA_DIR / "campaign_daily_clean.csv", usecols=cols)
df['date'] = pd.to_datetime(df['date'])
print(f"  Loaded {len(df):,} rows, {df['campaign'].nunique():,} unique campaigns")
print(f"  Date range: {df['date'].min().date()} ~ {df['date'].max().date()}")
n_days = (df['date'].max() - df['date'].min()).days + 1

# ─── AI 1.0 Metrics (campaigns with AI running) ───
print("\n[2/7] Computing AI 1.0 metrics ...")
ai_running = df[df['ai_status'] == 'AI运行中'].copy()
ai_cancelled = df[df['ai_status'] == 'AI已取消'].copy()

# Campaign-level aggregation
ai_camp = ai_running.groupby('campaign').agg(
    spend=('spend', 'sum'), sales=('sales', 'sum'), orders=('orders', 'sum'),
    clicks=('clicks', 'sum'), impressions=('impressions', 'sum'),
    days=('date', 'nunique'),
    country=('country', 'first'),
    bidding_strategy=('bidding_strategy', 'first'),
).reset_index()

# Filter to campaigns with actual spend
ai_camp_active = ai_camp[ai_camp['spend'] > 0]

ai1_agg = {
    'n_campaigns': int(ai_camp['campaign'].nunique()),
    'n_campaigns_active': int(len(ai_camp_active)),
    'total_spend': float(ai_camp['spend'].sum()),
    'total_sales': float(ai_camp['sales'].sum()),
    'total_orders': int(ai_camp['orders'].sum()),
    'total_clicks': int(ai_camp['clicks'].sum()),
    'total_impressions': int(ai_camp['impressions'].sum()),
}
ai1_agg['acos'] = round(ai1_agg['total_spend'] / max(ai1_agg['total_sales'], 1) * 100, 2)
ai1_agg['roas'] = round(ai1_agg['total_sales'] / max(ai1_agg['total_spend'], 1), 2)
ai1_agg['ctr'] = round(ai1_agg['total_clicks'] / max(ai1_agg['total_impressions'], 1) * 100, 2)
ai1_agg['cvr'] = round(ai1_agg['total_orders'] / max(ai1_agg['total_clicks'], 1) * 100, 2)
ai1_agg['cpc'] = round(ai1_agg['total_spend'] / max(ai1_agg['total_clicks'], 1), 2)
ai1_agg['avg_spend_per_camp'] = round(ai1_agg['total_spend'] / max(ai1_agg['n_campaigns_active'], 1), 2)

print(f"  AI 1.0: {ai1_agg['n_campaigns']} campaigns ({ai1_agg['n_campaigns_active']} active)")
print(f"  Spend: ${ai1_agg['total_spend']:,.0f}  Sales: ${ai1_agg['total_sales']:,.0f}")
print(f"  ACoS: {ai1_agg['acos']:.2f}%  ROAS: {ai1_agg['roas']:.2f}x")
print(f"  CPC: ${ai1_agg['cpc']:.2f}  CTR: {ai1_agg['ctr']:.2f}%  CVR: {ai1_agg['cvr']:.2f}%")

# ─── Controlled Non-AI Baseline ───
print("\n[3/7] Computing controlled non-AI baseline ...")
no_ai = df[df['ai_status'] == 'AI未开启'].copy()

noai_camp = no_ai.groupby('campaign').agg(
    spend=('spend', 'sum'), sales=('sales', 'sum'), orders=('orders', 'sum'),
    clicks=('clicks', 'sum'), impressions=('impressions', 'sum'),
    days=('date', 'nunique'),
    bidding_strategy=('bidding_strategy', 'first'),
).reset_index()

# Scale-matched: filter to campaigns with spend >= AI 1.0 P25
ai_p25_spend = float(ai_camp_active['spend'].quantile(0.25))
noai_matched = noai_camp[noai_camp['spend'] >= ai_p25_spend].copy()

noai_agg = {
    'n_campaigns': int(len(noai_matched)),
    'match_threshold': round(ai_p25_spend, 2),
    'total_spend': float(noai_matched['spend'].sum()),
    'total_sales': float(noai_matched['sales'].sum()),
    'total_orders': int(noai_matched['orders'].sum()),
    'total_clicks': int(noai_matched['clicks'].sum()),
    'total_impressions': int(noai_matched['impressions'].sum()),
}
noai_agg['acos'] = round(noai_agg['total_spend'] / max(noai_agg['total_sales'], 1) * 100, 2)
noai_agg['roas'] = round(noai_agg['total_sales'] / max(noai_agg['total_spend'], 1), 2)
noai_agg['ctr'] = round(noai_agg['total_clicks'] / max(noai_agg['total_impressions'], 1) * 100, 2)
noai_agg['cvr'] = round(noai_agg['total_orders'] / max(noai_agg['total_clicks'], 1) * 100, 2)
noai_agg['cpc'] = round(noai_agg['total_spend'] / max(noai_agg['total_clicks'], 1), 2)

print(f"  Match threshold: spend >= ${ai_p25_spend:,.2f}")
print(f"  Matched: {noai_agg['n_campaigns']:,} campaigns (from {len(noai_camp):,} total non-AI)")
print(f"  Spend: ${noai_agg['total_spend']:,.0f}  Sales: ${noai_agg['total_sales']:,.0f}")
print(f"  ACoS: {noai_agg['acos']:.2f}%  ROAS: {noai_agg['roas']:.2f}x")
print(f"  CPC: ${noai_agg['cpc']:.2f}  CTR: {noai_agg['ctr']:.2f}%  CVR: {noai_agg['cvr']:.2f}%")

# Also compute unmatched (all non-AI) for context
noai_all_agg = {
    'n_campaigns': int(len(noai_camp)),
    'total_spend': float(noai_camp['spend'].sum()),
    'total_sales': float(noai_camp['sales'].sum()),
}
noai_all_agg['acos'] = round(noai_all_agg['total_spend'] / max(noai_all_agg['total_sales'], 1) * 100, 2)
noai_all_agg['roas'] = round(noai_all_agg['total_sales'] / max(noai_all_agg['total_spend'], 1), 2)

# ─── AI 2.0 Projected Metrics ───
print("\n[4/7] Loading AI 2.0 projected metrics ...")
sim_df = pd.read_csv(DASHBOARD_DATA / "ai_impact_simulation.csv")
full_row = sim_df[sim_df['adoption_rate'] == 1.0].iloc[0]

with open(DASHBOARD_DATA / "ai_impact_details.json", encoding='utf-8') as f:
    impact_details = json.load(f)

baseline = impact_details['baseline']

ai2_agg = {
    'baseline_spend': baseline['total_spend'],
    'baseline_sales': baseline['total_sales'],
    'baseline_acos': baseline['acos'],
    'baseline_roas': baseline['roas'],
    'new_acos': float(full_row['new_acos']),
    'new_roas': float(full_row['new_roas']),
    'total_savings': float(full_row['total_savings']),
    'total_uplift': float(full_row['total_uplift']),
    'profit_improvement': float(full_row['profit_improvement']),
    'net_spend': float(full_row['net_spend']),
    'net_sales': float(full_row['net_sales']),
    # Estimate AI 2.0 CPC: bid optimization reduces overbidding
    # Baseline CPC ≈ total_spend / total_clicks, with bid savings reducing spend
    'estimated_cpc': round(
        (baseline['total_spend'] - float(full_row['total_savings'])) /
        max(baseline.get('total_clicks', baseline['total_spend'] / 0.54), 1), 2
    ),
    # CVR improvement from harvest (better targeting)
    'estimated_cvr': round(baseline.get('cvr', 4.92) * 1.05, 2),  # ~5% CVR lift from harvest
}

print(f"  AI 2.0 projected ACoS: {ai2_agg['new_acos']:.2f}%  ROAS: {ai2_agg['new_roas']:.2f}x")
print(f"  Savings: ${ai2_agg['total_savings']:,.0f}  Uplift: ${ai2_agg['total_uplift']:,.0f}")

# ─── Bidding Strategy Distributions ───
print("\n[5/7] Computing bidding strategy distributions ...")
ai1_bid = ai_camp_active['bidding_strategy'].value_counts(normalize=True)
ai1_bid_dist = {k: round(v * 100, 1) for k, v in ai1_bid.items()}

noai_bid = noai_matched['bidding_strategy'].value_counts(normalize=True)
noai_bid_dist = {k: round(v * 100, 1) for k, v in noai_bid.items()}

print(f"  AI 1.0 bidding: {ai1_bid_dist}")
print(f"  Non-AI bidding: {noai_bid_dist}")

# ─── Daily Time-Series ───
print("\n[6/7] Computing daily time-series ...")

# AI 1.0 daily
ai_daily = ai_running.groupby('date').agg(
    spend=('spend', 'sum'), sales=('sales', 'sum'),
    clicks=('clicks', 'sum'), impressions=('impressions', 'sum'),
    orders=('orders', 'sum'),
).reset_index().sort_values('date')

ai_daily['acos'] = (ai_daily['spend'] / ai_daily['sales'].clip(lower=0.01) * 100).round(2)
ai_daily['roas'] = (ai_daily['sales'] / ai_daily['spend'].clip(lower=0.01)).round(2)
ai_daily['cpc'] = (ai_daily['spend'] / ai_daily['clicks'].clip(lower=1)).round(3)

# 7-day rolling averages for smoothness
for col in ['acos', 'roas', 'orders', 'spend', 'sales', 'cpc']:
    ai_daily[f'{col}_ma7'] = ai_daily[col].rolling(7, min_periods=1).mean().round(2)

# Non-AI matched daily
matched_campaigns = set(noai_matched['campaign'].tolist())
noai_matched_rows = no_ai[no_ai['campaign'].isin(matched_campaigns)]

noai_daily = noai_matched_rows.groupby('date').agg(
    spend=('spend', 'sum'), sales=('sales', 'sum'),
    clicks=('clicks', 'sum'), impressions=('impressions', 'sum'),
    orders=('orders', 'sum'),
).reset_index().sort_values('date')

noai_daily['acos'] = (noai_daily['spend'] / noai_daily['sales'].clip(lower=0.01) * 100).round(2)
noai_daily['roas'] = (noai_daily['sales'] / noai_daily['spend'].clip(lower=0.01)).round(2)
noai_daily['cpc'] = (noai_daily['spend'] / noai_daily['clicks'].clip(lower=1)).round(3)

for col in ['acos', 'roas', 'orders', 'spend', 'sales', 'cpc']:
    noai_daily[f'{col}_ma7'] = noai_daily[col].rolling(7, min_periods=1).mean().round(2)

# AI 2.0 projected daily: apply improvement ratios to NON-AI baseline daily
# This shows "if you had AI 2.0 running on these manual campaigns"
acos_improvement_ratio = ai2_agg['new_acos'] / baseline['acos']  # e.g., 7.02 / 9.95 = 0.706
roas_improvement_ratio = ai2_agg['new_roas'] / baseline['roas']
# For orders: AI 2.0 uplift adds incremental orders
orders_uplift_ratio = 1 + (ai2_agg['total_uplift'] / max(baseline['total_sales'], 1))

ai2_daily = noai_daily[['date', 'acos_ma7', 'roas_ma7', 'orders_ma7', 'spend_ma7', 'sales_ma7']].copy()
ai2_daily['acos_ma7'] = (ai2_daily['acos_ma7'] * acos_improvement_ratio).round(2)
ai2_daily['roas_ma7'] = (ai2_daily['roas_ma7'] * roas_improvement_ratio).round(2)
ai2_daily['orders_ma7'] = (ai2_daily['orders_ma7'] * orders_uplift_ratio).round(0)
ai2_daily['sales_ma7'] = (ai2_daily['sales_ma7'] * orders_uplift_ratio).round(0)
# Spend decreases due to savings
spend_savings_ratio = 1 - (ai2_agg['total_savings'] / max(baseline['total_spend'], 1))
ai2_daily['spend_ma7'] = (ai2_daily['spend_ma7'] * spend_savings_ratio).round(0)

# Normalize orders for per-campaign comparison (scale AI 1.0 and non-AI differently)
# For the chart, we'll use per-campaign-per-day metrics to make fair comparison
ai1_n_active = max(ai1_agg['n_campaigns_active'], 1)
noai_n_active = max(noai_agg['n_campaigns'], 1)

# Convert daily to records
ai_daily_records = ai_daily[['date', 'acos', 'roas', 'orders', 'spend', 'sales', 'cpc',
                              'acos_ma7', 'roas_ma7', 'orders_ma7', 'spend_ma7', 'sales_ma7', 'cpc_ma7']].copy()
ai_daily_records['date'] = ai_daily_records['date'].dt.strftime('%Y-%m-%d')

noai_daily_records = noai_daily[['date', 'acos', 'roas', 'orders', 'spend', 'sales', 'cpc',
                                  'acos_ma7', 'roas_ma7', 'orders_ma7', 'spend_ma7', 'sales_ma7', 'cpc_ma7']].copy()
noai_daily_records['date'] = noai_daily_records['date'].dt.strftime('%Y-%m-%d')

ai2_daily_records = ai2_daily.copy()
ai2_daily_records['date'] = ai2_daily_records['date'].dt.strftime('%Y-%m-%d')

print(f"  AI 1.0 daily: {len(ai_daily_records)} days")
print(f"  Non-AI daily: {len(noai_daily_records)} days")
print(f"  AI 2.0 projected daily: {len(ai2_daily_records)} days")
print(f"  ACoS improvement ratio: {acos_improvement_ratio:.3f}")
print(f"  Orders uplift ratio: {orders_uplift_ratio:.3f}")

# ─── Limitation → Solution Mapping ───
print("\n[7/7] Building limitation-solution pairs ...")

modules = impact_details.get('modules', {})

limitations = [
    {
        'problem_zh': f"CPC 过高（${ai1_agg['cpc']:.2f} vs 非AI ${noai_agg['cpc']:.2f}）",
        'cause_zh': '过度依赖"动态竞价-提高和降低"策略，缺乏精细化出价控制',
        'ai2_module': 'bid_landscape',
        'ai2_icon': '💰',
        'ai2_solution_zh': 'Bid 竞价优化引擎：基于 50 万条投放记录的统计模型，找到每个 targeting 的最优出价区间',
        'ai2_savings': modules.get('bid_landscape', {}).get('savings_at_100pct', 312413),
        'ai2_uplift': modules.get('bid_landscape', {}).get('incremental_sales_at_100pct', 412285),
    },
    {
        'problem_zh': f"ACoS 偏高（{ai1_agg['acos']:.1f}% vs 非AI {noai_agg['acos']:.1f}%）",
        'cause_zh': '没有系统性识别和否定低效搜索词，浪费花费在零转化词上',
        'ai2_module': 'negation',
        'ai2_icon': '🚫',
        'ai2_solution_zh': '否定词引擎：绿灯/黄灯分级，识别 27,448 个无效/低效搜索词，安全否定',
        'ai2_savings': modules.get('negation', {}).get('savings_at_100pct', 308729),
        'ai2_uplift': 0,
    },
    {
        'problem_zh': '缺少精确匹配收割机制',
        'cause_zh': '高转化词停留在 Broad/Phrase 匹配，错失低 CPC 的 Exact 匹配机会',
        'ai2_module': 'harvest',
        'ai2_icon': '🌾',
        'ai2_solution_zh': '关键词收割引擎：识别 4,558 个可升级为 Exact 匹配的高转化词',
        'ai2_savings': modules.get('harvest', {}).get('acos_savings_at_100pct', 46391),
        'ai2_uplift': modules.get('harvest', {}).get('incremental_sales_at_100pct', 799883),
    },
    {
        'problem_zh': '缺乏前瞻性预算分配能力',
        'cause_zh': 'AI 1.0 基于历史规则调整，无法预测未来趋势和季节性变化',
        'ai2_module': 'adtft',
        'ai2_icon': '📈',
        'ai2_solution_zh': 'AdTFT 预测引擎：预测未来 1-7 天 Spend/Sales/ACoS，提前优化预算分配',
        'ai2_savings': 0,
        'ai2_uplift': modules.get('adtft', {}).get('incremental_sales_at_100pct', 1439111),
    },
]

# ─── Assemble output ───
output = {
    'ai1': ai1_agg,
    'noai_matched': noai_agg,
    'noai_all': noai_all_agg,
    'ai2': ai2_agg,
    'n_days': n_days,
    'ai1_bidding_dist': ai1_bid_dist,
    'noai_bidding_dist': noai_bid_dist,
    'limitations_solutions': limitations,
    'ai1_daily': ai_daily_records.to_dict(orient='records'),
    'noai_daily': noai_daily_records.to_dict(orient='records'),
    'ai2_daily': ai2_daily_records.to_dict(orient='records'),
    'acos_improvement_ratio': round(acos_improvement_ratio, 4),
    'orders_uplift_ratio': round(orders_uplift_ratio, 4),
}

out_path = DASHBOARD_DATA / "ai_comparison.json"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)

print(f"\n{'=' * 60}")
print(f"Output saved to {out_path}")
print(f"File size: {out_path.stat().st_size / 1024:.1f} KB")
print(f"{'=' * 60}")

# ─── Summary ───
print(f"\n📊 Summary Comparison:")
print(f"{'':>25} {'人工(匹配)':>12} {'AI 1.0':>12} {'AI 2.0':>12}")
print(f"{'ACoS':>25} {noai_agg['acos']:>11.2f}% {ai1_agg['acos']:>11.2f}% {ai2_agg['new_acos']:>11.2f}%")
print(f"{'ROAS':>25} {noai_agg['roas']:>11.2f}x {ai1_agg['roas']:>11.2f}x {ai2_agg['new_roas']:>11.2f}x")
print(f"{'CPC':>25} ${noai_agg['cpc']:>10.2f} ${ai1_agg['cpc']:>10.2f} {'(优化后)':>12}")
print(f"{'CVR':>25} {noai_agg['cvr']:>11.2f}% {ai1_agg['cvr']:>11.2f}% {'(提升)':>12}")
print(f"{'Campaigns':>25} {noai_agg['n_campaigns']:>12,} {ai1_agg['n_campaigns']:>12,} {'全部':>12}")
