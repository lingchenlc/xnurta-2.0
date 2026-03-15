"""
Pre-compute dashboard summary data from the large campaign CSV.
This avoids loading 1.5GB CSV every time the dashboard starts.

Run once:  python3 dashboard/prepare_dashboard_data.py
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "dashboard").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent
SRC = PROJECT_ROOT / "data" / "processed" / "campaign_daily_clean.csv"
OUT = PROJECT_ROOT / "dashboard" / "data"
OUT.mkdir(parents=True, exist_ok=True)

print("Loading campaign_daily_clean.csv …")
cols = ['date', 'country', 'store', 'campaign', 'impressions',
        'clicks', 'spend', 'orders', 'sales']
df = pd.read_csv(SRC, usecols=cols, parse_dates=['date'])
print(f"  Loaded {len(df):,} rows")

# ─── 1. Daily aggregate ──────────────────────────────────────
print("Computing daily aggregates …")
daily = df.groupby('date').agg(
    spend=('spend', 'sum'),
    sales=('sales', 'sum'),
    orders=('orders', 'sum'),
    clicks=('clicks', 'sum'),
    impressions=('impressions', 'sum'),
    active_campaigns=('campaign', 'nunique'),
).reset_index()
daily['acos'] = daily['spend'] / daily['sales'].clip(lower=0.01) * 100
daily['roas'] = daily['sales'] / daily['spend'].clip(lower=0.01)
daily['ctr'] = daily['clicks'] / daily['impressions'].clip(lower=1) * 100
daily['cvr'] = daily['orders'] / daily['clicks'].clip(lower=1) * 100
daily['cpc'] = daily['spend'] / daily['clicks'].clip(lower=1)
daily.to_csv(OUT / "daily_agg.csv", index=False)
print(f"  → daily_agg.csv ({len(daily)} rows)")

# ─── 2. Country aggregate ────────────────────────────────────
print("Computing country aggregates …")
country = df.groupby('country').agg(
    spend=('spend', 'sum'),
    sales=('sales', 'sum'),
    orders=('orders', 'sum'),
    clicks=('clicks', 'sum'),
    impressions=('impressions', 'sum'),
    campaigns=('campaign', 'nunique'),
    stores=('store', 'nunique'),
).reset_index()
country['acos'] = country['spend'] / country['sales'].clip(lower=0.01) * 100
country['roas'] = country['sales'] / country['spend'].clip(lower=0.01)
country = country.sort_values('spend', ascending=False)
country.to_csv(OUT / "country_agg.csv", index=False)
print(f"  → country_agg.csv ({len(country)} rows)")

# ─── 3. Country x Date aggregate ─────────────────────────────
print("Computing country x date aggregates …")
country_daily = df.groupby(['country', 'date']).agg(
    spend=('spend', 'sum'),
    sales=('sales', 'sum'),
    orders=('orders', 'sum'),
    clicks=('clicks', 'sum'),
    impressions=('impressions', 'sum'),
).reset_index()
country_daily['acos'] = country_daily['spend'] / country_daily['sales'].clip(lower=0.01) * 100
country_daily['ctr'] = country_daily['clicks'] / country_daily['impressions'].clip(lower=1) * 100
country_daily['cvr'] = country_daily['orders'] / country_daily['clicks'].clip(lower=1) * 100
country_daily.to_csv(OUT / "country_daily_agg.csv", index=False)
print(f"  → country_daily_agg.csv ({len(country_daily)} rows)")

# ─── 4. Top campaigns per country ────────────────────────────
print("Computing top campaigns …")
camp = df.groupby(['country', 'store', 'campaign']).agg(
    spend=('spend', 'sum'),
    sales=('sales', 'sum'),
    orders=('orders', 'sum'),
    clicks=('clicks', 'sum'),
    impressions=('impressions', 'sum'),
    days=('date', 'nunique'),
).reset_index()
camp['acos'] = (camp['spend'] / camp['sales'].clip(lower=0.01) * 100).round(1)
camp['roas'] = (camp['sales'] / camp['spend'].clip(lower=0.01)).round(1)
camp = camp.sort_values('spend', ascending=False)
# Keep top 200 per country
top_camps = camp.groupby('country').head(200)
top_camps.to_csv(OUT / "top_campaigns.csv", index=False)
print(f"  → top_campaigns.csv ({len(top_camps)} rows)")

# ─── 5. Overall KPIs ─────────────────────────────────────────
import json
kpis = {
    'total_spend': float(df['spend'].sum()),
    'total_sales': float(df['sales'].sum()),
    'total_orders': int(df['orders'].sum()),
    'total_clicks': int(df['clicks'].sum()),
    'total_impressions': int(df['impressions'].sum()),
    'avg_acos': float(df['spend'].sum() / max(df['sales'].sum(), 1) * 100),
    'avg_roas': float(df['sales'].sum() / max(df['spend'].sum(), 1)),
    'n_campaigns': int(df['campaign'].nunique()),
    'n_countries': int(df['country'].nunique()),
    'n_stores': int(df['store'].nunique()),
    'date_min': str(df['date'].min().date()),
    'date_max': str(df['date'].max().date()),
    'n_days': int(df['date'].nunique()),
    'n_rows': len(df),
}
with open(OUT / "kpis.json", 'w') as f:
    json.dump(kpis, f, indent=2)
print(f"  → kpis.json")

print("\n✅ Dashboard data prepared!")
for p in sorted(OUT.glob("*")):
    print(f"  {p.name:30s} {p.stat().st_size / 1024:.0f} KB")
