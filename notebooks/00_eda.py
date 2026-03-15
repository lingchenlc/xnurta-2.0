"""
Xnurta 2.0 — Phase 0: 数据探索与质量分析
==========================================
4个数据集全面分析:
1. Campaign (103K rows)
2. Keywords (500K rows)
3. Search Terms (500K rows)
4. Operation Logs (20K rows)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 120)
pd.set_option('display.float_format', '{:.4f}'.format)

DATA_DIR = str(Path(__file__).resolve().parent.parent / 'data' / 'raw')

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 80)
print("PHASE 0: 数据加载")
print("=" * 80)

print("\n[1/4] Loading Campaign data...")
df_camp = pd.read_csv(DATA_DIR + 'campaign.csv', encoding='utf-8-sig')
print(f"  Shape: {df_camp.shape}")

print("[2/4] Loading Keywords data...")
df_kw = pd.read_csv(DATA_DIR + 'keywords.csv', encoding='utf-8-sig')
print(f"  Shape: {df_kw.shape}")

print("[3/4] Loading Search Term data...")
df_st = pd.read_csv(DATA_DIR + 'search_term.csv', encoding='utf-8-sig', nrows=500001)
print(f"  Shape: {df_st.shape}")

print("[4/4] Loading Operation Log data...")
df_ops = pd.read_csv(DATA_DIR + 'operation_log.csv', encoding='utf-8-sig')
print(f"  Shape: {df_ops.shape}")

# ============================================================
# 2. CAMPAIGN DATA ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("CAMPAIGN DATA ANALYSIS (103K campaigns)")
print("=" * 80)

# Clean numeric columns helper
def safe_numeric(series):
    return pd.to_numeric(series.replace('--', np.nan).replace('', np.nan), errors='coerce')

# Basic stats
print(f"\n--- 店铺分布 ---")
store_counts = df_camp['店铺'].value_counts()
print(f"Total unique stores: {len(store_counts)}")
print(store_counts.head(15).to_string())

print(f"\n--- 广告类型分布 ---")
print(df_camp['广告类型'].value_counts().to_string())

print(f"\n--- Campaign 状态分布 ---")
print(df_camp['投放状态'].value_counts().head(10).to_string())

print(f"\n--- 开关状态 ---")
print(df_camp['开关'].value_counts().to_string())

# Core metrics
print(f"\n--- 核心指标统计 (当前周期, 90天汇总) ---")
camp_metrics = pd.DataFrame()
for col_cn, col_name in [
    ('花费(当前周期)', 'spend'),
    ('销售额(当前周期)', 'sales'),
    ('订单数(当前周期)', 'orders'),
    ('点击量(当前周期)', 'clicks'),
    ('ACOS(当前周期)', 'acos'),
    ('ROAS(当前周期)', 'roas'),
    ('日预算', 'daily_budget'),
    ('目标ACOS', 'target_acos'),
]:
    if col_cn in df_camp.columns:
        camp_metrics[col_name] = safe_numeric(df_camp[col_cn])

print(camp_metrics.describe().round(2).to_string())

# ACoS distribution
print(f"\n--- ACoS 分布 ---")
acos = camp_metrics['acos'].dropna()
print(f"  有ACoS数据的Campaign数: {len(acos)} / {len(df_camp)}")
print(f"  ACoS < 10%: {(acos < 0.10).sum()} ({(acos < 0.10).mean()*100:.1f}%)")
print(f"  ACoS 10-20%: {((acos >= 0.10) & (acos < 0.20)).sum()} ({((acos >= 0.10) & (acos < 0.20)).mean()*100:.1f}%)")
print(f"  ACoS 20-30%: {((acos >= 0.20) & (acos < 0.30)).sum()} ({((acos >= 0.20) & (acos < 0.30)).mean()*100:.1f}%)")
print(f"  ACoS 30-50%: {((acos >= 0.30) & (acos < 0.50)).sum()} ({((acos >= 0.30) & (acos < 0.50)).mean()*100:.1f}%)")
print(f"  ACoS > 50%: {(acos >= 0.50).sum()} ({(acos >= 0.50).mean()*100:.1f}%)")

# Target ACoS distribution
print(f"\n--- Target ACoS 分布 ---")
target_acos = camp_metrics['target_acos'].dropna()
print(f"  有Target ACoS数据的Campaign数: {len(target_acos)}")
if len(target_acos) > 0:
    print(f"  均值: {target_acos.mean():.2%}")
    print(f"  中位数: {target_acos.median():.2%}")
    print(f"  分布: {target_acos.describe().to_string()}")

# Total spend & sales
print(f"\n--- 总量汇总 (90天) ---")
total_spend = camp_metrics['spend'].sum()
total_sales = camp_metrics['sales'].sum()
total_orders = camp_metrics['orders'].sum()
total_clicks = camp_metrics['clicks'].sum()
print(f"  总花费: ${total_spend:,.0f}")
print(f"  总销售额: ${total_sales:,.0f}")
print(f"  总订单数: {total_orders:,.0f}")
print(f"  总点击数: {total_clicks:,.0f}")
print(f"  整体ACoS: {total_spend/max(total_sales,1):.2%}")
print(f"  整体ROAS: {total_sales/max(total_spend,1):.2f}")

# Per-store summary
print(f"\n--- 各店铺汇总 ---")
df_camp['_spend'] = safe_numeric(df_camp['花费(当前周期)'])
df_camp['_sales'] = safe_numeric(df_camp['销售额(当前周期)'])
df_camp['_orders'] = safe_numeric(df_camp['订单数(当前周期)'])
df_camp['_clicks'] = safe_numeric(df_camp['点击量(当前周期)'])

store_summary = df_camp.groupby('店铺').agg(
    campaigns=('广告活动', 'count'),
    spend=('_spend', 'sum'),
    sales=('_sales', 'sum'),
    orders=('_orders', 'sum'),
    clicks=('_clicks', 'sum'),
).sort_values('spend', ascending=False)
store_summary['acos'] = store_summary['spend'] / store_summary['sales'].replace(0, np.nan)
store_summary['cpc'] = store_summary['spend'] / store_summary['clicks'].replace(0, np.nan)
print(store_summary.round(2).to_string())

# ============================================================
# 3. KEYWORDS DATA ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("KEYWORDS DATA ANALYSIS (500K keywords)")
print("=" * 80)

print(f"\n--- 匹配类型分布 ---")
print(df_kw['匹配类型'].value_counts().to_string())

print(f"\n--- 开关状态 ---")
print(df_kw['开关'].value_counts().to_string())

print(f"\n--- 投放状态 ---")
print(df_kw['投放状态'].value_counts().head(10).to_string())

# Bid analysis
print(f"\n--- 出价分析 ---")
bid = safe_numeric(df_kw['出价'])
print(f"  有出价数据: {bid.notna().sum()} / {len(df_kw)}")
print(f"  出价统计:")
print(f"    均值: ${bid.mean():.2f}")
print(f"    中位数: ${bid.median():.2f}")
print(f"    P25: ${bid.quantile(0.25):.2f}")
print(f"    P75: ${bid.quantile(0.75):.2f}")
print(f"    P95: ${bid.quantile(0.95):.2f}")
print(f"    最大: ${bid.max():.2f}")

# Keyword performance
print(f"\n--- 关键词表现统计 ---")
kw_spend = safe_numeric(df_kw['花费(当前周期)'])
kw_clicks = safe_numeric(df_kw['点击量(当前周期)'])
kw_orders = safe_numeric(df_kw['订单数(当前周期)'])
kw_impressions = safe_numeric(df_kw['曝光量(当前周期)'])
kw_sales = safe_numeric(df_kw['销售额(当前周期)'])

print(f"  有花费的关键词: {(kw_spend > 0).sum()} / {len(df_kw)} ({(kw_spend > 0).mean()*100:.1f}%)")
print(f"  有点击的关键词: {(kw_clicks > 0).sum()} / {len(df_kw)} ({(kw_clicks > 0).mean()*100:.1f}%)")
print(f"  有订单的关键词: {(kw_orders > 0).sum()} / {len(df_kw)} ({(kw_orders > 0).mean()*100:.1f}%)")
print(f"  有曝光的关键词: {(kw_impressions > 0).sum()} / {len(df_kw)} ({(kw_impressions > 0).mean()*100:.1f}%)")

# Spend concentration
print(f"\n--- 花费集中度 (帕累托分析) ---")
kw_spend_sorted = kw_spend.dropna().sort_values(ascending=False)
total_kw_spend = kw_spend_sorted.sum()
if total_kw_spend > 0:
    cumsum = kw_spend_sorted.cumsum()
    top_1pct = int(len(kw_spend_sorted) * 0.01)
    top_5pct = int(len(kw_spend_sorted) * 0.05)
    top_10pct = int(len(kw_spend_sorted) * 0.10)
    top_20pct = int(len(kw_spend_sorted) * 0.20)
    print(f"  Top 1% 关键词占总花费: {cumsum.iloc[top_1pct]/total_kw_spend:.1%}")
    print(f"  Top 5% 关键词占总花费: {cumsum.iloc[top_5pct]/total_kw_spend:.1%}")
    print(f"  Top 10% 关键词占总花费: {cumsum.iloc[top_10pct]/total_kw_spend:.1%}")
    print(f"  Top 20% 关键词占总花费: {cumsum.iloc[top_20pct]/total_kw_spend:.1%}")

# Keyword text analysis
print(f"\n--- 关键词文本分析 ---")
kw_text = df_kw['关键词'].dropna()
print(f"  唯一关键词数: {kw_text.nunique()}")
kw_word_count = kw_text.str.split().str.len()
print(f"  平均词数: {kw_word_count.mean():.1f}")
print(f"  1 个词: {(kw_word_count == 1).sum()} ({(kw_word_count == 1).mean()*100:.1f}%)")
print(f"  2 个词: {(kw_word_count == 2).sum()} ({(kw_word_count == 2).mean()*100:.1f}%)")
print(f"  3 个词: {(kw_word_count == 3).sum()} ({(kw_word_count == 3).mean()*100:.1f}%)")
print(f"  4+ 个词: {(kw_word_count >= 4).sum()} ({(kw_word_count >= 4).mean()*100:.1f}%)")

# Top of search impression share
print(f"\n--- 搜索结果顶部展示份额 ---")
tos = safe_numeric(df_kw['搜索结果顶部展示份额(当前周期)'])
tos_valid = tos.dropna()
print(f"  有数据的关键词: {len(tos_valid)}")
if len(tos_valid) > 0:
    print(f"  均值: {tos_valid.mean():.2%}")
    print(f"  中位数: {tos_valid.median():.2%}")

# ============================================================
# 4. SEARCH TERM DATA ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("SEARCH TERM DATA ANALYSIS (500K search terms)")
print("=" * 80)

print(f"\n--- 匹配类型分布 ---")
print(df_st['匹配类型'].value_counts().to_string())

print(f"\n--- 基本统计 ---")
st_unique = df_st['搜索词'].nunique()
print(f"  唯一搜索词数: {st_unique}")
print(f"  唯一投放词数: {df_st['投放'].nunique()}")
print(f"  唯一店铺数: {df_st['店铺'].nunique()}")
print(f"  唯一广告活动数: {df_st['广告活动'].nunique()}")

# Search term performance
print(f"\n--- 搜索词表现统计 ---")
st_spend = safe_numeric(df_st['花费(当前周期)'])
st_clicks = safe_numeric(df_st['点击量(当前周期)'])
st_orders = safe_numeric(df_st['订单数(当前周期)'])
st_impressions = safe_numeric(df_st['曝光量(当前周期)'])
st_sales = safe_numeric(df_st['销售额(当前周期)'])

print(f"  有花费: {(st_spend > 0).sum()} ({(st_spend > 0).mean()*100:.1f}%)")
print(f"  有点击: {(st_clicks > 0).sum()} ({(st_clicks > 0).mean()*100:.1f}%)")
print(f"  有订单: {(st_orders > 0).sum()} ({(st_orders > 0).mean()*100:.1f}%)")
print(f"  零花费零点击: {((st_spend == 0) & (st_clicks == 0)).sum()} ({((st_spend == 0) & (st_clicks == 0)).mean()*100:.1f}%)")

# Search term word count
print(f"\n--- 搜索词长度分析 ---")
st_text = df_st['搜索词'].dropna().astype(str)
st_word_count = st_text.str.split().str.len()
print(f"  平均词数: {st_word_count.mean():.1f}")
print(f"  1 词: {(st_word_count == 1).sum()} ({(st_word_count == 1).mean()*100:.1f}%)")
print(f"  2 词: {(st_word_count == 2).sum()} ({(st_word_count == 2).mean()*100:.1f}%)")
print(f"  3 词: {(st_word_count == 3).sum()} ({(st_word_count == 3).mean()*100:.1f}%)")
print(f"  4-5 词: {((st_word_count >= 4) & (st_word_count <= 5)).sum()} ({((st_word_count >= 4) & (st_word_count <= 5)).mean()*100:.1f}%)")
print(f"  6+ 词: {(st_word_count >= 6).sum()} ({(st_word_count >= 6).mean()*100:.1f}%)")

# ACoS distribution of search terms
print(f"\n--- 搜索词 ACoS 分布 ---")
st_acos = safe_numeric(df_st['ACOS(当前周期)'])
st_acos_valid = st_acos.dropna()
print(f"  有ACoS数据: {len(st_acos_valid)}")
if len(st_acos_valid) > 0:
    print(f"  ACoS = 0 (有销售无花费或无数据): {(st_acos_valid == 0).sum()}")
    print(f"  ACoS < 10%: {(st_acos_valid < 0.10).sum()} ({(st_acos_valid < 0.10).mean()*100:.1f}%)")
    print(f"  ACoS 10-25%: {((st_acos_valid >= 0.10) & (st_acos_valid < 0.25)).sum()} ({((st_acos_valid >= 0.10) & (st_acos_valid < 0.25)).mean()*100:.1f}%)")
    print(f"  ACoS 25-50%: {((st_acos_valid >= 0.25) & (st_acos_valid < 0.50)).sum()} ({((st_acos_valid >= 0.25) & (st_acos_valid < 0.50)).mean()*100:.1f}%)")
    print(f"  ACoS > 50%: {(st_acos_valid >= 0.50).sum()} ({(st_acos_valid >= 0.50).mean()*100:.1f}%)")

# "Wasted spend" analysis — search terms with spend but no orders
print(f"\n--- 浪费花费分析 (有花费无订单的搜索词) ---")
mask_wasted = (st_spend > 0) & (st_orders == 0)
wasted_count = mask_wasted.sum()
wasted_spend = st_spend[mask_wasted].sum()
print(f"  有花费无订单的搜索词数: {wasted_count} ({wasted_count/len(df_st)*100:.1f}%)")
print(f"  浪费的花费总额: ${wasted_spend:,.0f}")
print(f"  占总花费的比例: {wasted_spend/max(st_spend.sum(),1):.1%}")

# ============================================================
# 5. OPERATION LOG ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("OPERATION LOG ANALYSIS (20K records)")
print("=" * 80)

print(f"\n--- 操作类型分布 ---")
print(df_ops['Operation Type'].value_counts().to_string())

print(f"\n--- 操作来源 ---")
print(df_ops['Operation'].value_counts().to_string())

print(f"\n--- 资源类型 ---")
print(df_ops['Resource Type'].value_counts().head(10).to_string())

print(f"\n--- 操作属性 ---")
print(df_ops['Attribute'].value_counts().head(10).to_string())

# Bid change analysis
bid_changes = df_ops[df_ops['Attribute'] == 'bid'].copy()
if len(bid_changes) > 0:
    bid_changes['old_bid'] = safe_numeric(bid_changes['Old Value'])
    bid_changes['new_bid'] = safe_numeric(bid_changes['New Value'])
    bid_changes['change_pct'] = (bid_changes['new_bid'] - bid_changes['old_bid']) / bid_changes['old_bid']

    print(f"\n--- 出价调整分析 ---")
    print(f"  出价调整次数: {len(bid_changes)}")
    print(f"  降价次数: {(bid_changes['change_pct'] < 0).sum()}")
    print(f"  加价次数: {(bid_changes['change_pct'] > 0).sum()}")
    print(f"  平均调整幅度: {bid_changes['change_pct'].mean():.2%}")
    print(f"  调整幅度分布:")
    print(f"    P5:  {bid_changes['change_pct'].quantile(0.05):.2%}")
    print(f"    P25: {bid_changes['change_pct'].quantile(0.25):.2%}")
    print(f"    P50: {bid_changes['change_pct'].quantile(0.50):.2%}")
    print(f"    P75: {bid_changes['change_pct'].quantile(0.75):.2%}")
    print(f"    P95: {bid_changes['change_pct'].quantile(0.95):.2%}")

# ============================================================
# 6. CROSS-DATASET ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("CROSS-DATASET ANALYSIS")
print("=" * 80)

# Match search terms to keywords
print(f"\n--- 搜索词与关键词的关联 ---")
st_targeting_set = set(df_st['投放'].dropna().unique())
kw_text_set = set(df_kw['关键词'].dropna().unique())
overlap = st_targeting_set & kw_text_set
print(f"  Search Term 报告中的投放词数: {len(st_targeting_set)}")
print(f"  Keywords 表中的关键词数: {len(kw_text_set)}")
print(f"  交集: {len(overlap)}")

# Campaign overlap
st_camps = set(df_st['广告活动'].dropna().unique())
camp_camps = set(df_camp['广告活动'].dropna().unique())
print(f"\n--- Campaign 关联 ---")
print(f"  Search Term 中的Campaign数: {len(st_camps)}")
print(f"  Campaign 表中的Campaign数: {len(camp_camps)}")
print(f"  交集: {len(st_camps & camp_camps)}")

# ============================================================
# 7. SUMMARY & RECOMMENDATIONS
# ============================================================
print("\n" + "=" * 80)
print("PHASE 0 总结与建模建议")
print("=" * 80)

print(f"""
📊 数据概览:
  - {len(df_camp):,} campaigns | {len(df_kw):,} keywords | {len(df_st):,} search terms | {len(df_ops):,} operations
  - {df_camp['店铺'].nunique()} 个店铺, 全部 US 站点
  - 90天总花费: ${total_spend:,.0f} | 总销售: ${total_sales:,.0f} | 整体ACoS: {total_spend/max(total_sales,1):.1%}

🔍 关键发现:
  1. 花费高度集中: Top 5% 关键词贡献大部分花费 (帕累托效应)
  2. 大量零表现关键词: 约 {((kw_spend==0)|(kw_spend.isna())).sum()/len(df_kw)*100:.0f}% 关键词 90 天零花费
  3. 浪费花费: ${wasted_spend:,.0f} 花在了无转化的搜索词上
  4. 操作日志以降价为主 ({(df_ops['Operation Type'].str.contains('Decreased', na=False)).sum()/len(df_ops)*100:.0f}%)

✅ Phase 2 可立即开始:
  - 50 万搜索词 → Embedding 聚类 + 意图分析
  - 50 万关键词 → 出价分析 + 关键词-搜索词匹配分析
  - 10 万 Campaign → 结构分析 + 店铺对比

⚠️ 仍需日粒度数据用于 Phase 1 (AdTFT):
  - Campaign daily report (每 campaign 每天一行)
  - Targeting daily report (每关键词每天一行)
""")

print("Phase 0 EDA 完成! ✅")
