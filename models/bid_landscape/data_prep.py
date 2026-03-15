"""
Xnurta 2.0 — Bid Landscape Data Preparation
==============================================
Clean 500K targeting records + operation logs for bid landscape modeling.

Strategy (cross-sectional):
  - Use bid variation ACROSS different targetings to estimate bid-response curves
  - Group targetings by keyword cluster / match type / competition level
  - Build features: bid level, keyword properties, historical performance

Output:
  - bid_landscape_features.csv: clean feature matrix for modeling
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "dashboard").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

print("=" * 60)
print("🎯 Bid Landscape — Data Preparation")
print("=" * 60)

# ─── 1. Load & Clean Keywords/Targeting Data ──────────────

print("\n📊 Loading keywords.csv (500K targeting records)...")
kw = pd.read_csv(RAW_DIR / "keywords.csv", encoding='utf-8-sig', low_memory=False)
print(f"  Raw: {len(kw):,} rows × {len(kw.columns)} cols")

# Column name mapping (Chinese → English)
col_map = {
    '匹配类型': 'match_type',
    '开关': 'toggle',
    '投放状态': 'delivery_status',
    '国家': 'country',
    '店铺': 'store',
    '广告活动名称': 'campaign',
    '广告组名称': 'ad_group',
    '建议竞价': 'suggested_bid',
    '出价': 'bid',
    '当前出价': 'current_bid',
    '关键词': 'keyword',
    '标签': 'label',
    # Current period metrics
    '曝光量(当前周期)': 'impressions',
    '点击量(当前周期)': 'clicks',
    '花费(当前周期)': 'spend',
    '订单数(当前周期)': 'orders',
    '销售额(当前周期)': 'sales',
    '点击率(当前周期)': 'ctr_raw',
    '转化率(当前周期)': 'cvr_raw',
    '点击成本(当前周期)': 'cpc_raw',
    'ACOS(当前周期)': 'acos_raw',
    'ROAS(当前周期)': 'roas_raw',
    '搜索结果顶部展示份额(当前周期)': 'top_of_search_share',
    '销量(当前周期)': 'units',
    '推广商品销售额(当前周期)': 'promoted_sales',
    '其他商品销售额(当前周期)': 'other_sales',
    '推广商品订单数(当前周期)': 'promoted_orders',
    '其他商品订单数(当前周期)': 'other_orders',
    '可见曝光量(当前周期)': 'viewable_impressions',
    '可见曝光率(当前周期)': 'viewable_rate',
    # Comparison period
    '曝光量(对比周期)': 'impressions_prev',
    '点击量(对比周期)': 'clicks_prev',
    '花费(对比周期)': 'spend_prev',
    '订单数(对比周期)': 'orders_prev',
    '销售额(对比周期)': 'sales_prev',
    '点击率(对比周期)': 'ctr_prev',
    '转化率(对比周期)': 'cvr_prev',
    # Growth rates
    '曝光量(环比增长率)': 'impressions_growth',
    '点击量(环比增长率)': 'clicks_growth',
    '花费(环比增长率)': 'spend_growth',
    '订单数(环比增长率)': 'orders_growth',
    '销售额(环比增长率)': 'sales_growth',
    'ACOS(环比增长率)': 'acos_growth',
}

kw = kw.rename(columns={k: v for k, v in col_map.items() if k in kw.columns})

# ─── 2. Parse Numeric Columns ─────────────────────────────

print("\n🔧 Parsing numeric columns...")

def parse_numeric(series):
    """Convert Chinese number format to float."""
    return pd.to_numeric(
        series.astype(str)
        .str.replace('--', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.replace('$', '', regex=False)
        .str.replace('%', '', regex=False)
        .str.strip(),
        errors='coerce'
    )

numeric_cols = ['bid', 'current_bid', 'impressions', 'clicks', 'spend',
                'orders', 'sales', 'units', 'cpc_raw', 'ctr_raw', 'cvr_raw',
                'acos_raw', 'roas_raw', 'top_of_search_share',
                'viewable_impressions', 'viewable_rate',
                'promoted_sales', 'other_sales', 'promoted_orders', 'other_orders',
                'impressions_prev', 'clicks_prev', 'spend_prev', 'orders_prev', 'sales_prev',
                'impressions_growth', 'clicks_growth', 'spend_growth',
                'orders_growth', 'sales_growth', 'acos_growth']

for col in numeric_cols:
    if col in kw.columns:
        kw[col] = parse_numeric(kw[col])

# ─── 3. Filter Active Targetings ──────────────────────────

print("\n🔍 Filtering active targetings...")
print(f"  Total: {len(kw):,}")

# Must have valid bid and impressions
kw = kw[kw['bid'].notna() & (kw['bid'] > 0)]
print(f"  With valid bid > 0: {len(kw):,}")

kw = kw[kw['impressions'].notna() & (kw['impressions'] > 0)]
print(f"  With impressions > 0: {len(kw):,}")

# Only standard match types
kw = kw[kw['match_type'].isin(['BROAD', 'PHRASE', 'EXACT'])]
print(f"  Standard match types (B/P/E): {len(kw):,}")

# ─── 4. Compute Derived Features ──────────────────────────

print("\n📐 Computing derived features...")

# Core metrics
kw['ctr'] = kw['clicks'] / kw['impressions'].clip(lower=1)
kw['cvr'] = kw['orders'] / kw['clicks'].clip(lower=1)
kw['cpc'] = kw['spend'] / kw['clicks'].clip(lower=1)
kw['acos'] = kw['spend'] / kw['sales'].clip(lower=0.01)
kw['roas'] = kw['sales'] / kw['spend'].clip(lower=0.01)
kw['order_value'] = kw['sales'] / kw['orders'].clip(lower=1)

# Log-transformed features (for modeling)
kw['log_bid'] = np.log1p(kw['bid'])
kw['log_impressions'] = np.log1p(kw['impressions'])
kw['log_clicks'] = np.log1p(kw['clicks'])
kw['log_spend'] = np.log1p(kw['spend'])

# Bid relative to average (by match type × country)
bid_avg = kw.groupby(['country', 'match_type'])['bid'].transform('median')
kw['bid_relative'] = kw['bid'] / bid_avg.clip(lower=0.01)
kw['bid_relative_log'] = np.log1p(kw['bid_relative'])

# Impressions per dollar of bid (efficiency proxy)
kw['imp_per_bid'] = kw['impressions'] / kw['bid']

# Has conversion flag
kw['has_orders'] = (kw['orders'] > 0).astype(int)

# Keyword length (proxy for specificity/long-tail)
kw['keyword_length'] = kw['keyword'].astype(str).str.len()
kw['keyword_word_count'] = kw['keyword'].astype(str).str.split().str.len()

# Growth features (previous period comparison)
for metric in ['impressions', 'clicks', 'spend', 'orders', 'sales']:
    prev_col = f'{metric}_prev'
    growth_col = f'{metric}_growth_rate'
    if prev_col in kw.columns:
        kw[growth_col] = (kw[metric] - kw[prev_col].fillna(0)) / kw[prev_col].clip(lower=0.01).fillna(1)

# Top of search share (indicator of ad rank)
kw['top_of_search_share'] = kw['top_of_search_share'].fillna(0)

# Viewable rate
kw['viewable_rate'] = kw['viewable_rate'].fillna(0)

# ─── 5. Bid Buckets for Analysis ──────────────────────────

print("\n📊 Creating bid buckets...")
kw['bid_bucket'] = pd.cut(kw['bid'],
    bins=[0, 0.15, 0.30, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 100],
    labels=['$0-0.15', '$0.15-0.30', '$0.30-0.50', '$0.50-0.75',
            '$0.75-1.00', '$1.00-1.50', '$1.50-2.00', '$2.00-3.00',
            '$3.00-5.00', '$5.00+']
)

# Print bid bucket summary
print("\n  Bid bucket summary:")
bucket_summary = kw.groupby('bid_bucket', observed=True).agg(
    n_targets=('bid', 'count'),
    avg_bid=('bid', 'mean'),
    avg_impressions=('impressions', 'mean'),
    avg_clicks=('clicks', 'mean'),
    avg_ctr=('ctr', 'mean'),
    avg_orders=('orders', 'mean'),
    avg_cvr=('cvr', 'mean'),
    avg_cpc=('cpc', 'mean'),
    conv_rate=('has_orders', 'mean'),
).round(4)
print(bucket_summary.to_string())

# ─── 6. Encode Categoricals ──────────────────────────────

print("\n🏷️ Encoding categoricals...")

# Country encoding
country_map = {c: i for i, c in enumerate(sorted(kw['country'].unique()))}
kw['country_enc'] = kw['country'].map(country_map)

# Match type encoding
match_map = {'BROAD': 0, 'PHRASE': 1, 'EXACT': 2}
kw['match_type_enc'] = kw['match_type'].map(match_map)

print(f"  Countries: {country_map}")
print(f"  Match types: {match_map}")

# ─── 7. Save Processed Data ──────────────────────────────

print("\n💾 Saving processed data...")

# Feature columns for modeling
feature_cols = [
    # IDs
    'country', 'store', 'campaign', 'ad_group', 'match_type', 'keyword',
    # Bid
    'bid', 'current_bid', 'log_bid', 'bid_relative', 'bid_bucket',
    # Performance (current)
    'impressions', 'clicks', 'spend', 'orders', 'sales', 'units',
    'ctr', 'cvr', 'cpc', 'acos', 'roas', 'order_value',
    'log_impressions', 'log_clicks', 'log_spend',
    'imp_per_bid', 'has_orders',
    'top_of_search_share', 'viewable_rate',
    'promoted_sales', 'other_sales', 'promoted_orders', 'other_orders',
    # Performance (prev period)
    'impressions_prev', 'clicks_prev', 'spend_prev', 'orders_prev', 'sales_prev',
    # Growth
    'impressions_growth_rate', 'clicks_growth_rate', 'spend_growth_rate',
    # Keyword properties
    'keyword_length', 'keyword_word_count',
    # Encoded
    'country_enc', 'match_type_enc',
]

available_cols = [c for c in feature_cols if c in kw.columns]
output = kw[available_cols].copy()

output.to_csv(OUT_DIR / "bid_landscape_features.csv", index=False)
print(f"  Saved: bid_landscape_features.csv ({len(output):,} rows × {len(available_cols)} cols)")

# Save metadata
metadata = {
    'n_records': len(output),
    'n_features': len(available_cols),
    'feature_columns': available_cols,
    'country_encoding': country_map,
    'match_type_encoding': match_map,
    'bid_stats': {
        'min': float(kw['bid'].min()),
        'mean': float(kw['bid'].mean()),
        'median': float(kw['bid'].median()),
        'max': float(kw['bid'].max()),
        'std': float(kw['bid'].std()),
    },
    'data_source': 'keywords.csv (90-day aggregated targeting data)',
    'model_approach': 'cross-sectional bid landscape using bid variation across targetings',
}

with open(OUT_DIR / "bid_landscape_metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2, default=str)

# ─── 8. Summary Stats ────────────────────────────────────

print("\n" + "=" * 60)
print("✅ Data Preparation Complete!")
print("=" * 60)
print(f"  Records: {len(output):,}")
print(f"  Features: {len(available_cols)}")
print(f"  Countries: {kw['country'].nunique()}")
print(f"  Match types: {kw['match_type'].nunique()}")
print(f"  Unique keywords: {kw['keyword'].nunique():,}")
print(f"  Campaigns: {kw['campaign'].nunique():,}")
print(f"  Conversion rate: {kw['has_orders'].mean()*100:.1f}%")
print(f"  Bid range: ${kw['bid'].min():.2f} - ${kw['bid'].max():.2f}")
