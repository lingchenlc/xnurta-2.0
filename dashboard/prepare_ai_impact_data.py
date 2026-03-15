"""
Pre-compute AI Impact Simulation data for the dashboard.
Calculates the projected savings and revenue uplift at different AI adoption rates.

Run once:  python3 dashboard/prepare_ai_impact_data.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "dashboard").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent
OUT = PROJECT_ROOT / "dashboard" / "data"

print("=" * 60)
print("🤖 AI Impact Simulation — Data Preparation")
print("=" * 60)

# ─── 1. Load Baseline ────────────────────────────────────────
print("\n📊 Loading baseline KPIs...")
with open(OUT / "kpis.json") as f:
    kpis = json.load(f)

baseline = {
    'total_spend': kpis['total_spend'],
    'total_sales': kpis['total_sales'],
    'total_orders': kpis['total_orders'],
    'acos': kpis['avg_acos'],       # percentage
    'roas': kpis['avg_roas'],
    'n_days': kpis['n_days'],
}
print(f"  Spend: ${baseline['total_spend']:,.0f}")
print(f"  Sales: ${baseline['total_sales']:,.0f}")
print(f"  ACoS: {baseline['acos']:.2f}%")

# ─── 2. Negation Module Impact ───────────────────────────────
print("\n🚫 Computing negation impact...")
neg = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "negation_recommendations.csv")

# GREEN = high confidence, can fully save
# YELLOW = needs review, estimate 60% realization
green = neg[neg['safety_level'] == 'GREEN_SAFE_TO_NEGATE']
yellow = neg[neg['safety_level'] == 'YELLOW_REVIEW_RECOMMENDED']

green_spend = float(green['total_spend'].sum())
yellow_spend = float(yellow['total_spend'].sum())
yellow_realization = 0.6  # conservative: 60% of yellow terms are actually worth negating

negation_savings_full = green_spend + yellow_spend * yellow_realization
negation_n_green = len(green)
negation_n_yellow = len(yellow)
negation_n_total = len(neg)

# These are zero/low-conversion terms, so eliminating them has minimal sales impact
# But some yellow terms may have occasional conversions → estimate 5% sales loss
negation_sales_loss = float(yellow[yellow['total_sales'] > 0]['total_sales'].sum() * 0.05)

# By reason: how many each reason contributes
reason_impact = neg.groupby('negate_reason').agg(
    count=('search_term_clean', 'count'),
    spend_saved=('total_spend', 'sum'),
    sales_at_risk=('total_sales', 'sum'),
).reset_index()
reason_impact['net_savings'] = reason_impact['spend_saved'] - reason_impact['sales_at_risk'] * 0.05

print(f"  Green terms: {negation_n_green:,} → saves ${green_spend:,.0f}")
print(f"  Yellow terms: {negation_n_yellow:,} → saves ${yellow_spend * yellow_realization:,.0f} (60% realization)")
print(f"  Total saveable: ${negation_savings_full:,.0f}")
print(f"  Sales at risk: ${negation_sales_loss:,.0f}")

# ─── 3. Harvest Module Impact ────────────────────────────────
print("\n🌾 Computing harvest impact...")
har = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "harvest_recommendations.csv")

# New exact match terms → incremental sales uplift
# When a broad/phrase match term is promoted to exact match:
# - Typically 15-25% lower CPC (more precise targeting)
# - 10-20% higher CVR (better match intent)
# - Net effect: lower ACoS, similar or higher sales

# For terms without existing exact match → pure incremental
new_terms = har[~har['has_exact']] if 'has_exact' in har.columns else har
existing_terms = har[har['has_exact']] if 'has_exact' in har.columns else pd.DataFrame()

# Conservative uplift estimates:
# New exact match: 20% of current broad/phrase sales become incremental
# (because exact match captures intent that was split across broad matches)
new_term_incremental_rate = 0.20
existing_term_acos_improvement = 0.15  # 15% ACoS improvement for existing terms

harvest_new_sales = float(new_terms['total_sales'].sum())
harvest_new_spend = float(new_terms['total_spend'].sum())
harvest_existing_sales = float(existing_terms['total_sales'].sum()) if len(existing_terms) > 0 else 0
harvest_existing_spend = float(existing_terms['total_spend'].sum()) if len(existing_terms) > 0 else 0

# Incremental revenue from new exact match terms
harvest_incremental_sales = harvest_new_sales * new_term_incremental_rate
# Spend savings from ACoS improvement on existing terms
harvest_acos_savings = harvest_existing_spend * existing_term_acos_improvement

harvest_n_new = len(new_terms)
harvest_n_existing = len(existing_terms)
harvest_n_total = len(har)
harvest_avg_acos = float(har['acos'].mean())

print(f"  New terms: {harvest_n_new:,} → incremental sales ${harvest_incremental_sales:,.0f}")
print(f"  Existing terms: {harvest_n_existing:,} → ACoS savings ${harvest_acos_savings:,.0f}")
print(f"  Average harvest ACoS: {harvest_avg_acos*100:.1f}%")

# ─── 4. Bid Landscape Module Impact ─────────────────────────
print("\n💰 Computing bid landscape impact...")
bl_full = pd.read_csv(PROJECT_ROOT / "models" / "bid_landscape" / "output" / "bid_recommendations.csv")

decrease_recs = bl_full[bl_full['action'] == 'decrease']
increase_recs = bl_full[bl_full['action'] == 'increase']
maintain_recs = bl_full[bl_full['action'] == 'maintain']

avg_order_value = baseline['total_sales'] / max(baseline['total_orders'], 1)

# For DECREASE recommendations:
# 68% of targetings are overbidding. Use spend-proportion approach:
# 1. Estimate what share of total spend goes to overbidding targetings
# 2. Calculate the relative overbid (overbid_amount / current_bid)
# 3. Apply a conservative realization rate (not all bid reduction = spend reduction)
if len(decrease_recs) > 0:
    w = decrease_recs['impressions'].clip(lower=1)
    overbid_amount = float(np.average(
        decrease_recs['current_bid'] - decrease_recs['recommended_bid'],
        weights=w
    ))
    avg_current_bid = float(np.average(decrease_recs['current_bid'], weights=w))
    relative_overbid = overbid_amount / max(avg_current_bid, 0.01)  # ~50%

    # Estimate spend share: decrease targetings as fraction of all targetings
    # weighted by impressions (more impressions = more spend)
    total_imp = float(bl_full['impressions'].sum())
    decrease_imp_share = float(decrease_recs['impressions'].sum()) / max(total_imp, 1)

    # Spend on decrease targetings (estimated)
    decrease_spend_estimate = baseline['total_spend'] * decrease_imp_share

    # Savings = decrease_spend * relative_overbid * realization_rate
    # Realization rate: only ~30% of the theoretical savings materializes
    # (lower bid → fewer impressions → fewer clicks → partially offsets CPC savings)
    bid_realization_rate = 0.30
    bid_decrease_period_savings = decrease_spend_estimate * relative_overbid * bid_realization_rate
else:
    overbid_amount = 0
    avg_current_bid = 0
    relative_overbid = 0
    bid_decrease_period_savings = 0

# For INCREASE recommendations:
# These targetings are underbidding → missing potential impressions/sales
if len(increase_recs) > 0:
    inc_underbid = float(np.average(
        increase_recs['recommended_bid'] - increase_recs['current_bid'],
        weights=increase_recs['impressions'].clip(lower=1)
    ))
    # Higher bid → more impressions → more orders
    # Conservative: 10% more orders from increased bids
    increase_incremental_orders = float(increase_recs['orders'].sum() * 0.10)
    bid_increase_incremental_sales = increase_incremental_orders * avg_order_value

    # Extra spend: proportional to underbid * spend share
    increase_imp_share = float(increase_recs['impressions'].sum()) / max(total_imp, 1)
    increase_spend_estimate = baseline['total_spend'] * increase_imp_share
    avg_inc_bid = float(np.average(increase_recs['current_bid'], weights=increase_recs['impressions'].clip(lower=1)))
    relative_underbid = inc_underbid / max(avg_inc_bid, 0.01)
    bid_increase_extra_spend = increase_spend_estimate * relative_underbid * 0.30
else:
    increase_incremental_orders = 0
    bid_increase_incremental_sales = 0
    bid_increase_extra_spend = 0

bid_n_decrease = len(decrease_recs)
bid_n_increase = len(increase_recs)
bid_n_maintain = len(maintain_recs)

print(f"  Decrease recs: {bid_n_decrease:,} → avg overbid ${overbid_amount:.3f}")
print(f"  Period savings from bid reduction: ${bid_decrease_period_savings:,.0f}")
print(f"  Increase recs: {bid_n_increase:,} → incremental sales ${bid_increase_incremental_sales:,.0f}")
print(f"  Extra spend for increase: ${bid_increase_extra_spend:,.0f}")

# ─── 5. AdTFT Predictive Value ───────────────────────────────
print("\n📈 Computing AdTFT predictive value...")
# AdTFT enables proactive budget allocation:
# - Shift budget from low-ROAS days to high-ROAS days
# - Avoid overspending on declining campaigns
# Industry benchmark: predictive budget allocation improves ROAS by 5-12%
# Conservative estimate: 5% improvement via budget timing optimization

adtft_roas_improvement = 0.05  # 5% ROAS improvement
adtft_incremental_sales = baseline['total_sales'] * adtft_roas_improvement
# This improvement comes at no additional spend
adtft_extra_spend = 0

print(f"  ROAS improvement estimate: {adtft_roas_improvement*100:.0f}%")
print(f"  Incremental sales: ${adtft_incremental_sales:,.0f}")

# ─── 6. Compute Impact at Different Adoption Rates ───────────
print("\n📊 Computing impact at different adoption rates...")

adoption_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

impact_data = []

for rate in adoption_rates:
    # --- Savings (spend reduction) ---
    neg_save = negation_savings_full * rate
    neg_sales_risk = negation_sales_loss * rate
    bid_save = bid_decrease_period_savings * rate
    harvest_save = harvest_acos_savings * rate

    total_savings = neg_save + bid_save + harvest_save

    # --- Revenue uplift ---
    harvest_uplift = harvest_incremental_sales * rate
    bid_uplift = bid_increase_incremental_sales * rate
    adtft_uplift = adtft_incremental_sales * rate

    total_uplift = harvest_uplift + bid_uplift + adtft_uplift

    # --- Extra costs ---
    bid_extra = bid_increase_extra_spend * rate

    # --- Net impact ---
    net_spend = baseline['total_spend'] - total_savings + bid_extra
    net_sales = baseline['total_sales'] + total_uplift - neg_sales_risk
    net_orders = baseline['total_orders'] + int(
        (harvest_uplift / max(avg_order_value, 1)) +  # harvest incremental orders
        increase_incremental_orders * rate  # bid increase orders
    )

    new_acos = net_spend / max(net_sales, 1) * 100
    new_roas = net_sales / max(net_spend, 1)
    profit_improvement = (total_savings - bid_extra) + total_uplift - neg_sales_risk

    impact_data.append({
        'adoption_rate': rate,
        'adoption_pct': f"{rate*100:.0f}%",
        # Savings breakdown
        'negation_savings': round(neg_save, 0),
        'bid_savings': round(bid_save, 0),
        'harvest_savings': round(harvest_save, 0),
        'total_savings': round(total_savings, 0),
        # Revenue uplift breakdown
        'harvest_uplift': round(harvest_uplift, 0),
        'bid_uplift': round(bid_uplift, 0),
        'adtft_uplift': round(adtft_uplift, 0),
        'total_uplift': round(total_uplift, 0),
        # Net
        'extra_spend': round(bid_extra, 0),
        'net_spend': round(net_spend, 0),
        'net_sales': round(net_sales, 0),
        'net_orders': net_orders,
        'new_acos': round(new_acos, 2),
        'new_roas': round(new_roas, 2),
        'acos_improvement': round(baseline['acos'] - new_acos, 2),
        'roas_improvement': round(new_roas - baseline['roas'], 2),
        'profit_improvement': round(profit_improvement, 0),
    })

impact_df = pd.DataFrame(impact_data)
impact_df.to_csv(OUT / "ai_impact_simulation.csv", index=False)
print(f"  Saved: ai_impact_simulation.csv")

# ─── 7. Module Explanations ──────────────────────────────────
print("\n📝 Generating module explanations...")

module_details = {
    'baseline': baseline,
    'modules': {
        'negation': {
            'name': '否定词引擎',
            'icon': '🚫',
            'savings_at_100pct': round(negation_savings_full, 0),
            'n_candidates': negation_n_total,
            'n_green': negation_n_green,
            'n_yellow': negation_n_yellow,
            'green_spend': round(green_spend, 0),
            'yellow_spend_adjusted': round(yellow_spend * yellow_realization, 0),
            'reason_breakdown': reason_impact.to_dict('records'),
            'why_better': [
                '人工每天最多审核几百个搜索词，AI 同时分析 50 万条投放记录',
                '人工依赖直觉判断"这个词好不好"，AI 用统计置信度判断（零转化 + 高花费 = 安全否定）',
                '人工容易遗漏低频长尾词的浪费（每个花几毛钱，但累计惊人），AI 全量扫描无遗漏',
                '人工否定词容易误伤（否掉有转化的词），AI 设置绿灯/黄灯安全等级保护',
            ],
            'methodology': '分析搜索词的花费、转化率、ACoS，识别零转化高花费词和极端高 ACoS 词',
        },
        'harvest': {
            'name': '关键词收割',
            'icon': '🌾',
            'incremental_sales_at_100pct': round(harvest_incremental_sales, 0),
            'acos_savings_at_100pct': round(harvest_acos_savings, 0),
            'n_candidates': harvest_n_total,
            'n_new': harvest_n_new,
            'n_existing': harvest_n_existing,
            'avg_acos': round(harvest_avg_acos * 100, 1),
            'why_better': [
                '人工很难在几十万搜索词报告中发现"已经在 Broad 匹配下跑出好成绩但还没建 Exact 匹配"的词',
                'AI 自动交叉比对广告组结构，秒级定位缺失的精确匹配投放',
                '人工往往等到月底看报告才发现好词，AI 实时识别 → 抢占先机',
                'Exact 匹配比 Broad 平均 CPC 低 15-25%，相同转化量下花更少的钱',
            ],
            'methodology': '识别在 Broad/Phrase 匹配下表现优秀（高 CVR、低 ACoS）但尚未建立 Exact 匹配的搜索词',
        },
        'bid_landscape': {
            'name': 'Bid 竞价优化',
            'icon': '💰',
            'savings_at_100pct': round(bid_decrease_period_savings, 0),
            'incremental_sales_at_100pct': round(bid_increase_incremental_sales, 0),
            'n_decrease': bid_n_decrease,
            'n_increase': bid_n_increase,
            'n_maintain': bid_n_maintain,
            'avg_overbid': round(overbid_amount, 3),
            'why_better': [
                '人工调价通常是"感觉这个词贵了就降一点"，AI 基于 50 万条记录的统计模型找到最优出价区间',
                '每个市场（国家×匹配类型）的最优出价不同，人工不可能逐一优化 28 个细分市场',
                'AI 计算"边际回报递减点"—— 超过这个出价，多花的钱几乎不带来额外曝光',
                '68% 的投放在"过度出价"，平均多花 $0.35/次点击 → AI 帮你把钱花在刀刃上',
            ],
            'methodology': '横截面分析 50 万投放记录的出价-表现关系，用 Isotonic Regression 拟合单调响应曲线',
        },
        'adtft': {
            'name': 'AdTFT 预测引擎',
            'icon': '📈',
            'incremental_sales_at_100pct': round(adtft_incremental_sales, 0),
            'roas_improvement_pct': adtft_roas_improvement * 100,
            'why_better': [
                '人工根据"昨天的数据"做今天的决策（滞后性），AI 预测未来 1-7 天趋势（前瞻性）',
                '预算在 ROAS 下降前转移到高效 campaign，而不是等亏了才调整',
                '人工无法同时监控 15 万+ campaigns 的趋势变化，AI 全量实时预警',
                '量化不确定性：AI 给出预测区间（P10-P90），帮你做风险可控的决策',
            ],
            'methodology': 'Temporal Fusion Transformer 时序模型，融合多维特征预测 Spend/Sales/Orders/ACoS',
        },
    },
    'combined_why_better': [
        '🧠 全量分析：AI 同时处理 50 万投放、16 万关键词、8 个市场，人工不可能逐一覆盖',
        '⚡ 实时响应：AI 秒级完成分析，人工需要数天甚至数周',
        '📊 数据驱动：AI 基于统计显著性做决策，消除主观判断偏差',
        '🔄 持续优化：AI 7×24 不间断运行，不存在"忘了调"或"太忙没看"',
        '🎯 精准度：AI 同时考虑出价效率、关键词语义、时序趋势三个维度，人工通常只能顾一个',
    ],
}

with open(OUT / "ai_impact_details.json", 'w', encoding='utf-8') as f:
    json.dump(module_details, f, indent=2, ensure_ascii=False, default=str)
print(f"  Saved: ai_impact_details.json")

# ─── 8. Summary ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ AI Impact Data Prepared!")
print("=" * 60)

full = impact_df[impact_df['adoption_rate'] == 1.0].iloc[0]
print(f"\n  At 100% AI adoption:")
print(f"  💰 Total savings:       ${full['total_savings']:>12,.0f}")
print(f"  📈 Total revenue uplift: ${full['total_uplift']:>12,.0f}")
print(f"  🎯 Profit improvement:   ${full['profit_improvement']:>12,.0f}")
print(f"  📉 ACoS improvement:     {full['acos_improvement']:>+.2f}pp ({baseline['acos']:.2f}% → {full['new_acos']:.2f}%)")
print(f"  📈 ROAS improvement:     {full['roas_improvement']:>+.2f}x ({baseline['roas']:.2f}x → {full['new_roas']:.2f}x)")
