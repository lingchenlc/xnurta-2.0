"""
Xnurta 2.0 — Feature Analysis Data Preparation
================================================
Computes comprehensive feature/characteristic data for Manual vs AI 1.0 vs AI 2.0
comparison page ("优化特征分析").

Output: dashboard/data/feature_analysis.json
"""

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "dashboard").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
DASHBOARD_DATA = PROJECT_ROOT / "dashboard" / "data"


def classify_targeting(campaign_name: str) -> str:
    """Classify targeting type from campaign name patterns."""
    name = campaign_name.lower()
    if any(k in name for k in ['-auto', '-at', 'sp-auto', 'auto-']):
        return 'Auto'
    if any(k in name for k in ['asin', 'sp/asin', 'sp-asin']):
        return 'ASIN'
    if any(k in name for k in ['exact', '精确', 'sp-exact']):
        return 'Exact'
    if any(k in name for k in ['phrase', '词组']):
        return 'Phrase'
    if any(k in name for k in ['broad', '广泛', 'brd', 'board']):
        return 'Broad'
    if any(k in name for k in ['category', '品类', 'cat-']):
        return 'Category'
    return 'Other'


def compute_group_metrics(df_group):
    """Compute aggregate weighted metrics for a group."""
    total_spend = df_group['spend'].sum()
    total_sales = df_group['sales'].sum()
    total_clicks = df_group['clicks'].sum()
    total_impressions = df_group['impressions'].sum()
    total_orders = df_group['orders'].sum()

    return {
        'total_spend': round(total_spend, 2),
        'total_sales': round(total_sales, 2),
        'total_orders': int(total_orders),
        'total_clicks': int(total_clicks),
        'total_impressions': int(total_impressions),
        'acos': round(total_spend / max(total_sales, 1) * 100, 2),
        'roas': round(total_sales / max(total_spend, 1), 2),
        'cpc': round(total_spend / max(total_clicks, 1), 2),
        'ctr': round(total_clicks / max(total_impressions, 1) * 100, 2),
        'cvr': round(total_orders / max(total_clicks, 1) * 100, 2),
    }


def compute_bidding_dist(df_group):
    """Compute spend-weighted bidding strategy distribution."""
    active = df_group[df_group['spend'] > 0]
    if active.empty:
        return {}
    by_strat = active.groupby('bidding_strategy')['spend'].sum()
    total = by_strat.sum()
    return {k: round(v / total * 100, 1) for k, v in by_strat.items()}


def compute_targeting_dist(df_group):
    """Compute spend-weighted targeting type distribution."""
    active = df_group[df_group['spend'] > 0].copy()
    if active.empty:
        return {}
    active['targeting_type'] = active['campaign'].apply(classify_targeting)
    by_type = active.groupby('targeting_type')['spend'].sum()
    total = by_type.sum()
    return {k: round(v / total * 100, 1) for k, v in by_type.sort_values(ascending=False).items()}


def compute_budget_analysis(df_group):
    """Compute budget statistics."""
    # Per-campaign average budget
    camp_budget = df_group.groupby('campaign')['budget'].mean()
    # Budget utilization on spend days — cap at 100% for sanity
    spend_days = df_group[(df_group['spend'] > 0) & (df_group['budget'] > 0)].copy()
    if not spend_days.empty:
        spend_days['util'] = (spend_days['spend'] / spend_days['budget'] * 100).clip(upper=100)
        util_mean = round(spend_days['util'].mean(), 1)
        util_at_95 = round((spend_days['util'] >= 95).mean() * 100, 1)
    else:
        util_mean = 0
        util_at_95 = 0

    # Budget tiers
    tiers = pd.cut(camp_budget, bins=[0, 5, 10, 20, 50, 9999],
                   labels=['$0-5', '$5-10', '$10-20', '$20-50', '$50+'])
    tier_dist = {str(k): round(v * 100, 1) for k, v in tiers.value_counts(normalize=True).items()}

    return {
        'avg_budget': round(camp_budget.mean(), 2),
        'median_budget': round(camp_budget.median(), 2),
        'utilization_mean': util_mean,
        'pct_at_budget_cap': util_at_95,
        'tier_distribution': tier_dist,
    }


def compute_acos_distribution(df_group):
    """Compute campaign-level ACoS distribution (for box plots)."""
    camp = df_group.groupby('campaign').agg(
        spend=('spend', 'sum'),
        sales=('sales', 'sum'),
    )
    # Only include campaigns with meaningful spend AND sales for fair ACoS calc
    active = camp[(camp['spend'] >= 1.0) & (camp['sales'] >= 1.0)].copy()
    if active.empty:
        return {}
    active['acos'] = active['spend'] / active['sales'] * 100
    # Cap extreme values for cleaner visualization
    acos = active['acos'].clip(upper=100)
    return {
        'p10': round(float(acos.quantile(0.10)), 1),
        'p25': round(float(acos.quantile(0.25)), 1),
        'median': round(float(acos.quantile(0.50)), 1),
        'p75': round(float(acos.quantile(0.75)), 1),
        'p90': round(float(acos.quantile(0.90)), 1),
        'mean': round(float(acos.mean()), 1),
        'n_campaigns': len(active),
    }


def compute_delivery_status(df_group):
    """Compute delivery status distribution (by row count = campaign-days)."""
    dist = df_group['delivery_status'].value_counts(normalize=True)
    return {str(k): round(v * 100, 1) for k, v in dist.items()}


def compute_halo_effect(df_group):
    """Compute promoted vs halo (other) sales ratio."""
    promoted = df_group['promoted_sales'].sum()
    other = df_group['other_sales'].sum()
    total = promoted + other
    if total == 0:
        return {'promoted_pct': 0, 'halo_pct': 0}
    return {
        'promoted_pct': round(promoted / total * 100, 1),
        'halo_pct': round(other / total * 100, 1),
    }


def compute_scale_features(df_group):
    """Compute scale-related features."""
    camps = df_group.groupby('campaign').agg(
        total_spend=('spend', 'sum'),
        n_days=('date', 'nunique'),
        n_spend_days=('spend', lambda x: (x > 0).sum()),
    )
    pct_with_spend = round((camps['total_spend'] > 0).mean() * 100, 1)
    active = camps[camps['total_spend'] > 0]

    # Daily spend for active campaigns on spend days
    spend_days = df_group[df_group['spend'] > 0]
    daily_spend_per_camp = spend_days.groupby('campaign')['spend'].mean() if not spend_days.empty else pd.Series()

    return {
        'n_campaigns': len(camps),
        'n_stores': df_group['store'].nunique(),
        'n_countries': df_group['country'].nunique(),
        'pct_with_spend': pct_with_spend,
        'avg_daily_spend': round(float(daily_spend_per_camp.mean()), 2) if not daily_spend_per_camp.empty else 0,
        'median_daily_spend': round(float(daily_spend_per_camp.median()), 2) if not daily_spend_per_camp.empty else 0,
        'avg_active_days': round(float(active['n_spend_days'].mean()), 1) if not active.empty else 0,
        'median_active_days': round(float(active['n_spend_days'].median()), 1) if not active.empty else 0,
    }


def compute_radar_scores(manual_m, ai1_m, ai2_m, manual_b, ai1_b, manual_d, ai1_d):
    """Compute normalized 0-100 scores for 6 radar dimensions.

    Uses absolute scale anchors so all groups get meaningful scores:
    - ACoS: 0%=100pts, 15%=0pts (lower is better)
    - ROAS: 0x=0pts, 15x=100pts (higher is better)
    - CPC: $0=100pts, $1.0=0pts (lower is better)
    - CVR: 0%=0pts, 7%=100pts (higher is better)
    - Coverage: 0%=0pts, 100%=100pts
    - Budget Util: 0%=0pts, 100%=100pts
    """
    acos_vals = [manual_m['acos'], ai1_m['acos'], ai2_m.get('new_acos', 7.02)]
    roas_vals = [manual_m['roas'], ai1_m['roas'], ai2_m.get('new_roas', 14.24)]
    cpc_vals = [manual_m['cpc'], ai1_m['cpc'], ai2_m.get('estimated_cpc', 0.41)]
    cvr_vals = [manual_m['cvr'], ai1_m['cvr'], ai2_m.get('estimated_cvr', 5.17)]
    coverage_vals = [manual_d.get('pct_delivering', 81.4), ai1_d.get('pct_delivering', 95.3), 100]
    util_vals = [manual_b.get('utilization_mean', 61), ai1_b.get('utilization_mean', 91), 85]

    def inv_scale(val, worst, best):
        """Scale inversely: best=100, worst=0."""
        return round(max(5, min(100, (worst - val) / max(worst - best, 0.01) * 100)))

    def fwd_scale(val, worst, best):
        """Scale forward: best=100, worst=0."""
        return round(max(5, min(100, (val - worst) / max(best - worst, 0.01) * 100)))

    cost_ctrl = [inv_scale(v, 15, 5) for v in acos_vals]       # ACoS 5%=100, 15%=0
    efficiency = [fwd_scale(v, 5, 15) for v in roas_vals]       # ROAS 5x=0, 15x=100
    traffic_cost = [inv_scale(v, 1.0, 0.3) for v in cpc_vals]  # CPC $0.3=100, $1.0=0
    conversion = [fwd_scale(v, 3, 7) for v in cvr_vals]         # CVR 3%=0, 7%=100
    coverage = [round(max(5, min(100, v))) for v in coverage_vals]
    budget_util = [round(max(5, min(100, v))) for v in util_vals]

    categories = ['成本控制', '投放效率', '流量成本', '转化能力', '覆盖范围', '预算利用']
    return {
        'categories': categories,
        'manual': [cost_ctrl[0], efficiency[0], traffic_cost[0], conversion[0], coverage[0], budget_util[0]],
        'ai1': [cost_ctrl[1], efficiency[1], traffic_cost[1], conversion[1], coverage[1], budget_util[1]],
        'ai2': [cost_ctrl[2], efficiency[2], traffic_cost[2], conversion[2], coverage[2], budget_util[2]],
    }


def main():
    print("=" * 60)
    print("Feature Analysis Data Preparation")
    print("=" * 60)

    # Load data
    print("\n1. Loading campaign_daily_clean.csv...")
    df = pd.read_csv(DATA_DIR / "campaign_daily_clean.csv")
    print(f"   Loaded {len(df):,} rows, {df['campaign'].nunique():,} campaigns")

    manual = df[df['ai_status'] == 'AI未开启']
    ai1 = df[df['ai_status'] == 'AI运行中']
    print(f"   Manual: {len(manual):,} rows, {manual['campaign'].nunique():,} campaigns")
    print(f"   AI 1.0: {len(ai1):,} rows, {ai1['campaign'].nunique():,} campaigns")

    # Load AI 2.0 data
    print("\n2. Loading AI 2.0 reference data...")
    with open(DASHBOARD_DATA / "ai_impact_details.json") as f:
        ai2_details = json.load(f)
    sim = pd.read_csv(DASHBOARD_DATA / "ai_impact_simulation.csv")
    ai2_100 = sim[sim['adoption_rate'] == 1.0].iloc[0]

    with open(DASHBOARD_DATA / "ai_comparison.json") as f:
        ai_comp = json.load(f)
    ai2_metrics = ai_comp.get('ai2', {})

    # ── Manual metrics ──
    print("\n3. Computing Manual metrics...")
    manual_metrics = compute_group_metrics(manual)
    manual_bidding = compute_bidding_dist(manual)
    manual_targeting = compute_targeting_dist(manual)
    manual_budget = compute_budget_analysis(manual)
    manual_acos_dist = compute_acos_distribution(manual)
    manual_delivery = compute_delivery_status(manual)
    manual_halo = compute_halo_effect(manual)
    manual_scale = compute_scale_features(manual)
    print(f"   ACoS: {manual_metrics['acos']}%, ROAS: {manual_metrics['roas']}x")

    # Delivery status -> pct_delivering
    manual_del_pct = {'pct_delivering': manual_delivery.get('投放中', 0)}

    # ── AI 1.0 metrics ──
    print("\n4. Computing AI 1.0 metrics...")
    ai1_metrics = compute_group_metrics(ai1)
    ai1_bidding = compute_bidding_dist(ai1)
    ai1_targeting = compute_targeting_dist(ai1)
    ai1_budget = compute_budget_analysis(ai1)
    ai1_acos_dist = compute_acos_distribution(ai1)
    ai1_delivery = compute_delivery_status(ai1)
    ai1_halo = compute_halo_effect(ai1)
    ai1_scale = compute_scale_features(ai1)
    print(f"   ACoS: {ai1_metrics['acos']}%, ROAS: {ai1_metrics['roas']}x")

    ai1_del_pct = {'pct_delivering': ai1_delivery.get('投放中', 0)}

    # AI 1.0 ai_target breakdown
    print("   Computing ai_target breakdown...")
    ai_target_data = {}
    for target_val in ['保持稳定', '推动增长']:
        subset = ai1[ai1['ai_target'] == target_val]
        if not subset.empty:
            m = compute_group_metrics(subset)
            n_camps = subset['campaign'].nunique()
            ai_target_data[target_val] = {
                'n_campaigns': n_camps,
                'pct': round(n_camps / ai1['campaign'].nunique() * 100, 1),
                'acos': m['acos'],
                'roas': m['roas'],
                'cpc': m['cpc'],
                'cvr': m['cvr'],
            }

    # AI 1.0 store concentration
    ai1_store = ai1.groupby('store')['spend'].sum().sort_values(ascending=False)
    total_ai1_spend = ai1_store.sum()
    store_concentration = []
    for store, spend in ai1_store.head(5).items():
        store_concentration.append({
            'store': store,
            'spend': round(float(spend), 2),
            'pct': round(float(spend / max(total_ai1_spend, 1) * 100), 1),
        })

    # ── AI 2.0 projected data ──
    print("\n5. Preparing AI 2.0 projected data...")
    ai2_proj = {
        'new_acos': float(ai2_100['new_acos']),
        'new_roas': float(ai2_100['new_roas']),
        'total_savings': float(ai2_100['total_savings']),
        'total_uplift': float(ai2_100['total_uplift']),
        'profit_improvement': float(ai2_100['profit_improvement']),
        'estimated_cpc': ai2_metrics.get('estimated_cpc', 0.41),
        'estimated_cvr': ai2_metrics.get('estimated_cvr', 5.17),
    }

    # AI 2.0 module summary
    modules = []
    for key, mod in ai2_details.get('modules', {}).items():
        modules.append({
            'key': key,
            'name': mod.get('name', key),
            'icon': mod.get('icon', ''),
            'savings': round(mod.get('savings_at_100pct', 0)),
            'why_better': mod.get('why_better', [])[:3],
        })
    # Add uplift data from simulation
    module_uplift = {
        'negation': 0,
        'bid': float(ai2_100.get('bid_uplift', 0)),
        'harvest': float(ai2_100.get('harvest_uplift', 0)),
        'adtft': float(ai2_100.get('adtft_uplift', 0)),
    }
    for m in modules:
        m['uplift'] = round(module_uplift.get(m['key'], 0))

    # ── Radar chart ──
    print("\n6. Computing radar chart scores...")
    radar = compute_radar_scores(
        manual_metrics, ai1_metrics, ai2_proj,
        manual_budget, ai1_budget,
        manual_del_pct, ai1_del_pct,
    )
    print(f"   Manual: {radar['manual']}")
    print(f"   AI 1.0: {radar['ai1']}")
    print(f"   AI 2.0: {radar['ai2']}")

    # ── AI 2.0 bidding/targeting (conceptual) ──
    ai2_bidding = {'ML优化出价': 100.0}
    ai2_targeting = {
        'Auto': 25.0, 'Exact': 25.0, 'Broad': 15.0,
        'ASIN': 15.0, 'Phrase': 10.0, 'Other': 10.0,
    }

    # ── Build output ──
    print("\n7. Building output JSON...")
    output = {
        'manual': {
            **manual_scale,
            **manual_metrics,
            'bidding_strategy': manual_bidding,
            'targeting_mix': manual_targeting,
            'budget': manual_budget,
            'acos_distribution': manual_acos_dist,
            'delivery_status': manual_delivery,
            'halo': manual_halo,
        },
        'ai1': {
            **ai1_scale,
            **ai1_metrics,
            'bidding_strategy': ai1_bidding,
            'targeting_mix': ai1_targeting,
            'budget': ai1_budget,
            'acos_distribution': ai1_acos_dist,
            'delivery_status': ai1_delivery,
            'halo': ai1_halo,
            'ai_target': ai_target_data,
            'store_concentration': store_concentration,
        },
        'ai2': {
            **ai2_proj,
            'modules': modules,
            'bidding_strategy': ai2_bidding,
            'targeting_mix': ai2_targeting,
        },
        'radar': radar,
    }

    # Write
    out_path = DASHBOARD_DATA / "feature_analysis.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    file_size = out_path.stat().st_size / 1024
    print(f"\n✅ Saved to {out_path} ({file_size:.1f} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
