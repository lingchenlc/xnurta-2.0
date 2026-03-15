"""
Xnurta 2.0 — Phase 2 Step 2: 深度语义分析
============================================
使用 Sentence Transformer 对高价值搜索词做神经网络 Embedding
然后做更精准的语义聚类和分析
"""

import pandas as pd
import numpy as np
import time
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = str(Path(__file__).resolve().parent.parent.parent / 'data' / 'processed')

# ============================================================
# STEP 1: 筛选高价值搜索词 (降低计算量)
# ============================================================
print("=" * 80)
print("STEP 1: 筛选高价值搜索词用于深度分析")
print("=" * 80)

df = pd.read_csv(OUTPUT_DIR + 'search_terms_clustered.csv')
print(f"Loaded: {len(df):,} rows")

# 筛选标准: 花费 > $1 的搜索词 (过滤长尾噪声)
# 按搜索词聚合
term_agg = df.groupby('search_term_clean').agg(
    total_spend=('spend', 'sum'),
    total_sales=('sales', 'sum'),
    total_orders=('orders', 'sum'),
    total_clicks=('clicks', 'sum'),
    total_impressions=('impressions', 'sum'),
    n_campaigns=('campaign', 'nunique'),
    n_stores=('store', 'nunique'),
    match_types=('match_type', lambda x: list(x.unique())),
    targeting_words=('targeting', lambda x: list(x.unique())[:5]),
).reset_index()

term_agg['acos'] = term_agg['total_spend'] / term_agg['total_sales'].replace(0, np.nan)
term_agg['cvr'] = term_agg['total_orders'] / term_agg['total_clicks'].replace(0, np.nan)
term_agg['is_wasted'] = (term_agg['total_orders'] == 0).astype(int)
term_agg['word_count'] = term_agg['search_term_clean'].str.split().str.len()

# Filter: spend > $1 (covers ~95% of total spend)
high_value = term_agg[term_agg['total_spend'] >= 1.0].copy()
print(f"High-value search terms (spend >= $1): {len(high_value):,}")
print(f"Covers {high_value['total_spend'].sum() / term_agg['total_spend'].sum():.1%} of total spend")

# Separate ASIN-like terms from text terms
is_asin = high_value['search_term_clean'].str.match(r'^b0[a-z0-9]{8}$', case=False)
asin_terms = high_value[is_asin].copy()
text_terms = high_value[~is_asin].copy()
print(f"ASIN-based terms: {len(asin_terms):,}")
print(f"Text-based terms: {len(text_terms):,} → these go to Sentence Transformer")

# ============================================================
# STEP 2: Sentence Transformer Embedding
# ============================================================
print("\n" + "=" * 80)
print("STEP 2: Sentence Transformer Embedding")
print("=" * 80)

from sentence_transformers import SentenceTransformer

# Use all-MiniLM-L6-v2: fast, good quality, 384d
print("Loading model: all-MiniLM-L6-v2...")
t1 = time.time()
model = SentenceTransformer('all-MiniLM-L6-v2')

terms_list = text_terms['search_term_clean'].tolist()
print(f"Computing embeddings for {len(terms_list):,} text terms...")

# Batch encode
embeddings = model.encode(
    terms_list,
    batch_size=512,
    show_progress_bar=True,
    normalize_embeddings=True,
)
print(f"Embedding shape: {embeddings.shape}")
print(f"Time: {time.time()-t1:.1f}s")

# Save embeddings
np.save(OUTPUT_DIR + 'search_term_embeddings.npy', embeddings)
pd.Series(terms_list).to_csv(OUTPUT_DIR + 'search_term_embedding_index.csv', index=False)
print("Embeddings saved.")

# ============================================================
# STEP 3: HDBSCAN 语义聚类
# ============================================================
print("\n" + "=" * 80)
print("STEP 3: HDBSCAN 语义聚类")
print("=" * 80)

import hdbscan

t2 = time.time()

# HDBSCAN: density-based, auto-determines number of clusters
# min_cluster_size=20: minimum 20 terms to form a cluster
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=15,
    min_samples=5,
    metric='euclidean',  # on normalized embeddings = cosine
    cluster_selection_method='eom',
    prediction_data=True,
)
semantic_labels = clusterer.fit_predict(embeddings)

n_clusters = len(set(semantic_labels)) - (1 if -1 in semantic_labels else 0)
n_noise = (semantic_labels == -1).sum()
print(f"Semantic clusters found: {n_clusters}")
print(f"Noise points (unclustered): {n_noise} ({n_noise/len(semantic_labels):.1%})")
print(f"Time: {time.time()-t2:.1f}s")

text_terms = text_terms.copy()
text_terms['semantic_cluster'] = semantic_labels

# ============================================================
# STEP 4: 语义簇分析
# ============================================================
print("\n" + "=" * 80)
print("STEP 4: 语义簇深度分析")
print("=" * 80)

# Aggregate by semantic cluster
clustered = text_terms[text_terms['semantic_cluster'] >= 0].copy()

sem_stats = clustered.groupby('semantic_cluster').agg(
    n_terms=('search_term_clean', 'count'),
    total_spend=('total_spend', 'sum'),
    total_sales=('total_sales', 'sum'),
    total_orders=('total_orders', 'sum'),
    total_clicks=('total_clicks', 'sum'),
    wasted_terms=('is_wasted', 'sum'),
    wasted_spend=('total_spend', lambda x: x[clustered.loc[x.index, 'is_wasted'] == 1].sum()),
    avg_word_count=('word_count', 'mean'),
).reset_index()

sem_stats['acos'] = sem_stats['total_spend'] / sem_stats['total_sales'].replace(0, np.nan)
sem_stats['waste_ratio'] = sem_stats['wasted_spend'] / sem_stats['total_spend'].replace(0, np.nan)
sem_stats['cvr'] = sem_stats['total_orders'] / sem_stats['total_clicks'].replace(0, np.nan)

# Get top terms per cluster
def get_cluster_terms(cluster_id, n=8):
    mask = text_terms['semantic_cluster'] == cluster_id
    cluster_terms = text_terms[mask].nlargest(n, 'total_spend')
    return ' | '.join(cluster_terms['search_term_clean'].tolist())

sem_stats['representative_terms'] = sem_stats['semantic_cluster'].apply(get_cluster_terms)

# Classify
def classify_semantic_cluster(row):
    if row['total_orders'] == 0:
        return '❌ 零转化'
    elif row['acos'] is not None and row['acos'] > 0.50:
        return '⚠️ 高ACoS(>50%)'
    elif row['acos'] is not None and row['acos'] > 0.30:
        return '🟡 中ACoS(30-50%)'
    elif row['acos'] is not None and row['acos'] <= 0.15:
        return '🟢 优质(<15%)'
    else:
        return '🔵 正常(15-30%)'

sem_stats['classification'] = sem_stats.apply(classify_semantic_cluster, axis=1)

# Summary
print("\n--- 语义簇分类统计 ---")
class_sum = sem_stats.groupby('classification').agg(
    n_clusters=('semantic_cluster', 'count'),
    total_terms=('n_terms', 'sum'),
    total_spend=('total_spend', 'sum'),
    total_sales=('total_sales', 'sum'),
    wasted_spend=('wasted_spend', 'sum'),
).sort_values('total_spend', ascending=False)
class_sum['pct_spend'] = class_sum['total_spend'] / class_sum['total_spend'].sum()
class_sum['avg_acos'] = class_sum['total_spend'] / class_sum['total_sales'].replace(0, np.nan)
print(class_sum.round(2).to_string())

# Top clusters by spend
print("\n--- Top 30 语义簇 (按花费排序) ---")
top30 = sem_stats.nlargest(30, 'total_spend')
for _, row in top30.iterrows():
    acos_str = f"{row['acos']:.1%}" if pd.notna(row['acos']) else 'N/A'
    print(f"\n  Cluster {int(row['semantic_cluster']):4d} | {row['classification']}")
    print(f"    词数: {int(row['n_terms']):>5d} | 花费: ${row['total_spend']:>10,.0f} | 销售: ${row['total_sales']:>10,.0f} | ACoS: {acos_str}")
    print(f"    订单: {int(row['total_orders']):>5d} | 浪费: ${row['wasted_spend']:>8,.0f} ({row['waste_ratio']:.0%}) | CVR: {row['cvr']:.2%}" if pd.notna(row['cvr']) else f"    订单: 0 | 浪费: ${row['wasted_spend']:>8,.0f}")
    terms = row['representative_terms'][:150]
    print(f"    代表词: {terms}")

# Zero-conversion semantic clusters
print("\n--- 零转化语义簇 (全部建议否定) ---")
zero_clusters = sem_stats[sem_stats['classification'] == '❌ 零转化'].nlargest(30, 'total_spend')
if len(zero_clusters) > 0:
    for _, row in zero_clusters.iterrows():
        print(f"  Cluster {int(row['semantic_cluster']):4d} | 词数: {int(row['n_terms']):>4d} | 浪费: ${row['total_spend']:>8,.0f} | {row['representative_terms'][:100]}")
    print(f"\n  零转化簇总浪费: ${zero_clusters['total_spend'].sum():,.0f}")
else:
    print("  没有整个簇零转化的情况 (各簇内都混有转化词和非转化词)")

# High waste-ratio clusters
print("\n--- 高浪费率语义簇 (浪费占比>70%, 花费>$500) ---")
high_waste = sem_stats[(sem_stats['waste_ratio'] > 0.7) & (sem_stats['total_spend'] > 500)].nlargest(20, 'wasted_spend')
for _, row in high_waste.iterrows():
    acos_str = f"{row['acos']:.1%}" if pd.notna(row['acos']) else 'N/A'
    print(f"  Cluster {int(row['semantic_cluster']):4d} | 浪费率: {row['waste_ratio']:.0%} | 浪费: ${row['wasted_spend']:>6,.0f} / ${row['total_spend']:>6,.0f} | ACoS: {acos_str}")
    print(f"    → {row['representative_terms'][:120]}")

# ============================================================
# STEP 5: 跨簇语义相似度 (发现蚕食)
# ============================================================
print("\n" + "=" * 80)
print("STEP 5: 簇间相似度分析 (蚕食检测)")
print("=" * 80)

# Compute cluster centroids
cluster_ids = sorted(sem_stats['semantic_cluster'].unique())
centroids = []
for cid in cluster_ids:
    mask = semantic_labels == cid
    if mask.sum() > 0:
        centroid = embeddings[mask].mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        centroids.append(centroid)
    else:
        centroids.append(np.zeros(embeddings.shape[1]))

centroids = np.array(centroids)

# Find most similar cluster pairs
sim_matrix = centroids @ centroids.T
np.fill_diagonal(sim_matrix, 0)  # ignore self-similarity

# Top similar pairs
print("\n--- 最相似的语义簇对 (可能存在蚕食) ---")
n_show = 20
flat_idx = np.argsort(sim_matrix.flatten())[::-1][:n_show * 2]  # *2 because matrix is symmetric
shown = set()
count = 0
for idx in flat_idx:
    i, j = divmod(idx, len(cluster_ids))
    if i >= j:
        continue
    pair = (cluster_ids[i], cluster_ids[j])
    if pair in shown:
        continue
    shown.add(pair)

    ci, cj = cluster_ids[i], cluster_ids[j]
    si = sem_stats[sem_stats['semantic_cluster'] == ci].iloc[0]
    sj = sem_stats[sem_stats['semantic_cluster'] == cj].iloc[0]

    print(f"\n  相似度: {sim_matrix[i,j]:.3f}")
    print(f"    簇A ({int(ci):4d}): ${si['total_spend']:>8,.0f} | ACoS: {si['acos']:.1%}" if pd.notna(si['acos']) else f"    簇A ({int(ci):4d}): ${si['total_spend']:>8,.0f} | ACoS: N/A")
    print(f"      → {si['representative_terms'][:100]}")
    print(f"    簇B ({int(cj):4d}): ${sj['total_spend']:>8,.0f} | ACoS: {sj['acos']:.1%}" if pd.notna(sj['acos']) else f"    簇B ({int(cj):4d}): ${sj['total_spend']:>8,.0f} | ACoS: N/A")
    print(f"      → {sj['representative_terms'][:100]}")

    count += 1
    if count >= 15:
        break

# ============================================================
# STEP 6: 保存结果
# ============================================================
print("\n" + "=" * 80)
print("STEP 6: 保存分析结果")
print("=" * 80)

sem_stats.to_csv(OUTPUT_DIR + 'semantic_clusters.csv', index=False, encoding='utf-8-sig')
print(f"Saved: semantic_clusters.csv ({len(sem_stats)} clusters)")

text_terms.to_csv(OUTPUT_DIR + 'text_terms_with_semantic_clusters.csv', index=False, encoding='utf-8-sig')
print(f"Saved: text_terms_with_semantic_clusters.csv ({len(text_terms)} terms)")

# ============================================================
# FINAL SUMMARY
# ============================================================
total_clustered_spend = clustered['total_spend'].sum()
total_wasted_in_clusters = sem_stats['wasted_spend'].sum()

print(f"""

{'='*80}
Phase 2 深度语义分析完成!
{'='*80}

📊 分析结果:
  - 高价值文本搜索词: {len(text_terms):,} 个
  - 神经网络语义簇: {n_clusters} 个
  - 未聚类(噪声): {n_noise:,} 个 ({n_noise/len(text_terms):.1%})

💰 花费洞察:
  - 已聚类词总花费: ${total_clustered_spend:,.0f}
  - 其中浪费花费: ${total_wasted_in_clusters:,.0f} ({total_wasted_in_clusters/max(total_clustered_spend,1):.1%})

🎯 可执行建议:
  - 零转化簇: {len(zero_clusters)} 个 → 整簇否定
  - 高浪费率簇: {len(high_waste)} 个 → 逐词审查后否定
  - 相似簇对: 已识别蚕食风险 → 合并或差异化投放

📁 输出文件:
  - semantic_clusters.csv (簇级分析)
  - text_terms_with_semantic_clusters.csv (词级标注)
  - search_term_embeddings.npy (Embedding向量，供后续模型使用)
""")
