"""
Xnurta 2.0 — Phase 2: Search Term Semantic Engine
====================================================
Step 1: 数据清洗 + 特征工程
Step 2: TF-IDF Embedding + 降维 (快速，覆盖全量50万)
Step 3: HDBSCAN 聚类
Step 4: 簇级分析 + 表现评估
Step 5: 浪费花费识别 + 否定建议
Step 6: 关键词-搜索词匹配质量分析
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
import hdbscan
import warnings
import time
import os
warnings.filterwarnings('ignore')

DATA_DIR = str(Path(__file__).resolve().parent.parent.parent / 'data' / 'raw')
OUTPUT_DIR = str(Path(__file__).resolve().parent.parent.parent / 'data' / 'processed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def safe_numeric(series):
    return pd.to_numeric(series.replace('--', np.nan).replace('', np.nan), errors='coerce')

# ============================================================
# STEP 1: DATA LOADING & CLEANING
# ============================================================
print("=" * 80)
print("STEP 1: 数据加载与清洗")
print("=" * 80)

t0 = time.time()

# Load search term data
df_st = pd.read_csv(DATA_DIR + 'search_term.csv', encoding='utf-8-sig')
print(f"Loaded search terms: {len(df_st):,} rows")

# Load keywords data for cross-reference
df_kw = pd.read_csv(DATA_DIR + 'keywords.csv', encoding='utf-8-sig')
print(f"Loaded keywords: {len(df_kw):,} rows")

# Clean numeric columns
numeric_cols_map = {
    '花费(当前周期)': 'spend',
    '花费(对比周期)': 'spend_prev',
    '销售额(当前周期)': 'sales',
    '销售额(对比周期)': 'sales_prev',
    '订单数(当前周期)': 'orders',
    '订单数(对比周期)': 'orders_prev',
    '点击量(当前周期)': 'clicks',
    '点击量(对比周期)': 'clicks_prev',
    '曝光量(当前周期)': 'impressions',
    '曝光量(对比周期)': 'impressions_prev',
    'ACOS(当前周期)': 'acos',
    'ROAS(当前周期)': 'roas',
    '转化率(当前周期)': 'cvr',
    '点击率(当前周期)': 'ctr',
    '点击成本(当前周期)': 'cpc',
}

for cn_col, en_col in numeric_cols_map.items():
    if cn_col in df_st.columns:
        df_st[en_col] = safe_numeric(df_st[cn_col])

# Rename key text columns
df_st = df_st.rename(columns={
    '搜索词': 'search_term',
    '投放': 'targeting',
    '匹配类型': 'match_type',
    '国家': 'country',
    '店铺': 'store',
    '广告活动': 'campaign',
    '广告组': 'ad_group',
    '广告类型': 'ad_type',
})

# Clean search terms
df_st['search_term_clean'] = df_st['search_term'].astype(str).str.strip().str.lower()
df_st = df_st[df_st['search_term_clean'].str.len() > 0].copy()

# Compute derived metrics
df_st['has_orders'] = (df_st['orders'] > 0).astype(int)
df_st['is_wasted'] = ((df_st['spend'] > 0) & (df_st['orders'] == 0)).astype(int)
df_st['word_count'] = df_st['search_term_clean'].str.split().str.len()

print(f"After cleaning: {len(df_st):,} rows")
print(f"Time: {time.time()-t0:.1f}s")

# ============================================================
# STEP 2: TF-IDF EMBEDDING + DIMENSIONALITY REDUCTION
# ============================================================
print("\n" + "=" * 80)
print("STEP 2: TF-IDF Embedding + SVD 降维")
print("=" * 80)

t1 = time.time()

# Get unique search terms for embedding
unique_terms = df_st['search_term_clean'].unique()
print(f"Unique search terms: {len(unique_terms):,}")

# TF-IDF with character n-grams (captures typos and partial matches)
print("Building TF-IDF matrix...")
tfidf = TfidfVectorizer(
    analyzer='char_wb',      # character n-grams at word boundaries
    ngram_range=(3, 5),      # 3-5 char grams
    max_features=50000,      # limit features for memory
    min_df=2,                # must appear in at least 2 terms
    max_df=0.5,              # ignore terms appearing in >50% of docs
    sublinear_tf=True,       # apply log normalization
)
tfidf_matrix = tfidf.fit_transform(unique_terms)
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

# SVD dimensionality reduction (128 dimensions)
print("Applying SVD dimensionality reduction...")
svd = TruncatedSVD(n_components=128, random_state=42)
embeddings_svd = svd.fit_transform(tfidf_matrix)
embeddings_norm = normalize(embeddings_svd)  # L2 normalize

explained_var = svd.explained_variance_ratio_.sum()
print(f"SVD 128d explained variance: {explained_var:.2%}")
print(f"Embedding shape: {embeddings_norm.shape}")
print(f"Time: {time.time()-t1:.1f}s")

# Create term -> embedding mapping
term_to_idx = {term: i for i, term in enumerate(unique_terms)}

# ============================================================
# STEP 3: CLUSTERING
# ============================================================
print("\n" + "=" * 80)
print("STEP 3: 搜索词聚类")
print("=" * 80)

t2 = time.time()

# Strategy: 2-level clustering
# Level 1: MiniBatchKMeans for broad groups (fast, scalable)
# Level 2: Finer analysis within each cluster

N_CLUSTERS = 500  # Target ~500 clusters for 344K unique terms
print(f"Running MiniBatchKMeans with K={N_CLUSTERS}...")

kmeans = MiniBatchKMeans(
    n_clusters=N_CLUSTERS,
    batch_size=5000,
    random_state=42,
    n_init=3,
    max_iter=100,
)
cluster_labels = kmeans.fit_predict(embeddings_norm)
print(f"Clustering complete. {N_CLUSTERS} clusters created.")

# Map clusters back to search terms
term_cluster_map = {term: cluster_labels[i] for i, term in enumerate(unique_terms)}
df_st['cluster_id'] = df_st['search_term_clean'].map(term_cluster_map)

print(f"Time: {time.time()-t2:.1f}s")

# ============================================================
# STEP 4: CLUSTER ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("STEP 4: 簇级分析")
print("=" * 80)

# Aggregate metrics per cluster
cluster_stats = df_st.groupby('cluster_id').agg(
    n_search_terms=('search_term_clean', 'nunique'),
    n_rows=('search_term_clean', 'count'),
    total_spend=('spend', 'sum'),
    total_sales=('sales', 'sum'),
    total_orders=('orders', 'sum'),
    total_clicks=('clicks', 'sum'),
    total_impressions=('impressions', 'sum'),
    pct_with_orders=('has_orders', 'mean'),
    pct_wasted=('is_wasted', 'mean'),
    avg_word_count=('word_count', 'mean'),
).reset_index()

# Derived metrics
cluster_stats['acos'] = cluster_stats['total_spend'] / cluster_stats['total_sales'].replace(0, np.nan)
cluster_stats['cvr'] = cluster_stats['total_orders'] / cluster_stats['total_clicks'].replace(0, np.nan)
cluster_stats['ctr'] = cluster_stats['total_clicks'] / cluster_stats['total_impressions'].replace(0, np.nan)
cluster_stats['cpc'] = cluster_stats['total_spend'] / cluster_stats['total_clicks'].replace(0, np.nan)
cluster_stats['wasted_spend'] = df_st[df_st['is_wasted'] == 1].groupby('cluster_id')['spend'].sum().reindex(cluster_stats['cluster_id']).values

# Get representative terms for each cluster (top 5 by spend)
def get_top_terms(group, n=5):
    return ' | '.join(group.nlargest(n, 'spend')['search_term_clean'].tolist())

cluster_top_terms = df_st.groupby('cluster_id').apply(get_top_terms).reset_index()
cluster_top_terms.columns = ['cluster_id', 'top_terms']
cluster_stats = cluster_stats.merge(cluster_top_terms, on='cluster_id')

# Classify clusters
def classify_cluster(row):
    if row['total_orders'] == 0:
        return '❌ 零转化 (建议否定)'
    elif row['acos'] is not None and row['acos'] > 0.50:
        return '⚠️ 高ACoS (>50%, 需优化)'
    elif row['acos'] is not None and row['acos'] > 0.30:
        return '🟡 中ACoS (30-50%, 观察)'
    elif row['acos'] is not None and row['acos'] <= 0.15:
        return '🟢 优质 (ACoS<15%)'
    else:
        return '🔵 正常 (ACoS 15-30%)'

cluster_stats['classification'] = cluster_stats.apply(classify_cluster, axis=1)

# Print summary
print("\n--- 簇分类统计 ---")
class_summary = cluster_stats.groupby('classification').agg(
    n_clusters=('cluster_id', 'count'),
    total_terms=('n_search_terms', 'sum'),
    total_spend=('total_spend', 'sum'),
    total_sales=('total_sales', 'sum'),
    total_wasted=('wasted_spend', 'sum'),
).sort_values('total_spend', ascending=False)
class_summary['pct_of_spend'] = class_summary['total_spend'] / class_summary['total_spend'].sum()
print(class_summary.round(2).to_string())

# Top 20 clusters by spend
print("\n--- Top 20 花费最高的簇 ---")
top20 = cluster_stats.nlargest(20, 'total_spend')
for _, row in top20.iterrows():
    acos_str = f"{row['acos']:.1%}" if pd.notna(row['acos']) else 'N/A'
    print(f"\n  Cluster {int(row['cluster_id']):3d} | {row['classification']}")
    print(f"    搜索词数: {int(row['n_search_terms']):,} | 花费: ${row['total_spend']:,.0f} | 销售: ${row['total_sales']:,.0f} | ACoS: {acos_str}")
    print(f"    订单: {int(row['total_orders']):,} | 浪费花费: ${row['wasted_spend']:,.0f} | 转化率: {row['pct_with_orders']:.1%}")
    print(f"    代表词: {row['top_terms'][:120]}")

# Top 20 clusters by wasted spend
print("\n--- Top 20 浪费花费最高的簇 (优化重点) ---")
top20_wasted = cluster_stats.nlargest(20, 'wasted_spend')
for _, row in top20_wasted.iterrows():
    acos_str = f"{row['acos']:.1%}" if pd.notna(row['acos']) else 'N/A'
    print(f"\n  Cluster {int(row['cluster_id']):3d} | {row['classification']}")
    print(f"    浪费花费: ${row['wasted_spend']:,.0f} | 总花费: ${row['total_spend']:,.0f} | 浪费占比: {row['wasted_spend']/max(row['total_spend'],1):.1%}")
    print(f"    搜索词数: {int(row['n_search_terms']):,} | 有转化比例: {row['pct_with_orders']:.1%}")
    print(f"    代表词: {row['top_terms'][:120]}")

# ============================================================
# STEP 5: NEGATION ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("STEP 5: 否定词建议分析")
print("=" * 80)

# Zero-order search terms (high spend, zero orders)
print("\n--- 零转化高花费搜索词 (Top 50) ---")
wasted_df = df_st[df_st['is_wasted'] == 1].copy()
wasted_top50 = wasted_df.nlargest(50, 'spend')[['search_term_clean', 'targeting', 'store', 'campaign', 'match_type', 'spend', 'clicks', 'impressions']]
for _, row in wasted_top50.iterrows():
    print(f"  ${row['spend']:>8.2f} | {row['clicks']:>4.0f} clicks | {row['match_type']:>12s} | {row['search_term_clean'][:50]:50s} | → {str(row['targeting'])[:30]}")

# Aggregate wasted spend by search term (across all campaigns)
print("\n--- 跨Campaign汇总: 最浪费的搜索词 (Top 30) ---")
wasted_by_term = wasted_df.groupby('search_term_clean').agg(
    total_wasted_spend=('spend', 'sum'),
    total_clicks=('clicks', 'sum'),
    n_campaigns=('campaign', 'nunique'),
    n_stores=('store', 'nunique'),
    match_types=('match_type', lambda x: ','.join(x.unique())),
).nlargest(30, 'total_wasted_spend')

for term, row in wasted_by_term.iterrows():
    print(f"  ${row['total_wasted_spend']:>10.2f} | {row['total_clicks']:>6.0f} clicks | {row['n_campaigns']:>3d} campaigns | {row['n_stores']:>2d} stores | {term[:50]}")

# ============================================================
# STEP 6: KEYWORD - SEARCH TERM QUALITY ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("STEP 6: 关键词-搜索词匹配质量分析")
print("=" * 80)

# For each targeting keyword, analyze its search term quality
targeting_quality = df_st.groupby('targeting').agg(
    n_search_terms=('search_term_clean', 'nunique'),
    total_spend=('spend', 'sum'),
    total_sales=('sales', 'sum'),
    total_orders=('orders', 'sum'),
    total_clicks=('clicks', 'sum'),
    wasted_spend=('spend', lambda x: x[df_st.loc[x.index, 'is_wasted'] == 1].sum()),
    pct_wasted_terms=('is_wasted', 'mean'),
).reset_index()

targeting_quality['acos'] = targeting_quality['total_spend'] / targeting_quality['total_sales'].replace(0, np.nan)
targeting_quality['waste_ratio'] = targeting_quality['wasted_spend'] / targeting_quality['total_spend'].replace(0, np.nan)

# Keywords with highest waste ratio (money going to wrong search terms)
print("\n--- 匹配质量最差的投放词 (浪费占比最高, 花费>$100) ---")
high_spend_targeting = targeting_quality[targeting_quality['total_spend'] > 100]
worst_targeting = high_spend_targeting.nlargest(30, 'waste_ratio')
for _, row in worst_targeting.iterrows():
    acos_str = f"{row['acos']:.1%}" if pd.notna(row['acos']) else 'N/A'
    print(f"  {str(row['targeting'])[:40]:40s} | 花费: ${row['total_spend']:>8.0f} | 浪费: {row['waste_ratio']:.1%} | 搜索词数: {int(row['n_search_terms']):>4d} | ACoS: {acos_str}")

# ============================================================
# STEP 7: SAVE RESULTS
# ============================================================
print("\n" + "=" * 80)
print("STEP 7: 保存结果")
print("=" * 80)

# Save cluster stats
cluster_stats.to_csv(OUTPUT_DIR + 'cluster_analysis.csv', index=False, encoding='utf-8-sig')
print(f"Saved: cluster_analysis.csv ({len(cluster_stats)} clusters)")

# Save wasted spend analysis
wasted_by_term_df = wasted_by_term.reset_index()
wasted_by_term_df.to_csv(OUTPUT_DIR + 'wasted_search_terms.csv', index=False, encoding='utf-8-sig')
print(f"Saved: wasted_search_terms.csv ({len(wasted_by_term_df)} terms)")

# Save targeting quality
targeting_quality.to_csv(OUTPUT_DIR + 'targeting_quality.csv', index=False, encoding='utf-8-sig')
print(f"Saved: targeting_quality.csv ({len(targeting_quality)} targeting keywords)")

# Save full search term data with cluster assignments
df_st_export = df_st[['search_term_clean', 'targeting', 'store', 'campaign', 'match_type',
                       'spend', 'sales', 'orders', 'clicks', 'impressions',
                       'acos', 'cvr', 'cpc', 'cluster_id', 'has_orders', 'is_wasted', 'word_count']].copy()
df_st_export.to_csv(OUTPUT_DIR + 'search_terms_clustered.csv', index=False, encoding='utf-8-sig')
print(f"Saved: search_terms_clustered.csv ({len(df_st_export)} rows)")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("PHASE 2 STEP 1-6 完成! 总结")
print("=" * 80)

total_wasted = wasted_df['spend'].sum()
total_spend = df_st['spend'].sum()
zero_order_clusters = cluster_stats[cluster_stats['classification'].str.contains('零转化')]
high_acos_clusters = cluster_stats[cluster_stats['classification'].str.contains('高ACoS')]

print(f"""
📊 分析结果总结:

🔢 规模:
  - 分析搜索词: {len(df_st):,} 条
  - 唯一搜索词: {len(unique_terms):,} 个
  - 生成簇: {N_CLUSTERS} 个
  - 涉及投放词: {df_st['targeting'].nunique():,} 个

💰 花费分析:
  - 总花费: ${total_spend:,.0f}
  - 浪费花费 (零订单): ${total_wasted:,.0f} ({total_wasted/total_spend:.1%})
  - 零转化簇数: {len(zero_order_clusters)} 个 (含 {zero_order_clusters['n_search_terms'].sum():,} 个搜索词)
  - 高ACoS(>50%)簇数: {len(high_acos_clusters)} 个

🎯 优化建议:
  - 零转化簇建议批量否定 → 预估可省: ${zero_order_clusters['total_spend'].sum():,.0f}
  - 高ACoS簇建议降价或否定 → 涉及花费: ${high_acos_clusters['total_spend'].sum():,.0f}
  - 匹配质量差的投放词建议调整匹配方式

📁 输出文件:
  - {OUTPUT_DIR}cluster_analysis.csv
  - {OUTPUT_DIR}wasted_search_terms.csv
  - {OUTPUT_DIR}targeting_quality.csv
  - {OUTPUT_DIR}search_terms_clustered.csv

⏱️ 总耗时: {time.time()-t0:.1f}s
""")
