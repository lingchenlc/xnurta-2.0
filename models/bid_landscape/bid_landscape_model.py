"""
Xnurta 2.0 — Bid Landscape Model (Cross-Sectional)
=====================================================
Build bid-response curves from 500K targeting records.

Strategy:
  - Segment targetings by match_type × country
  - Fit parametric bid-response curves: bid → impressions, clicks, orders
  - Use isotonic regression for monotonic constraints (impressions should increase with bid)
  - Compute marginal ROI curves and optimal bid zones
  - Train LightGBM for fine-grained bid → performance prediction

Output:
  - bid_landscape_curves.json: fitted curve parameters per segment
  - bid_landscape_lgb.pkl: LightGBM models for bid-performance prediction
  - bid_landscape_results.json: summary statistics & optimal ranges
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import optimize
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score, mean_absolute_error
import pickle

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "dashboard").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "models" / "bid_landscape" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("🎯 Bid Landscape — Model Training")
print("=" * 60)

# ─── 1. Load Data ──────────────────────────────────────────
print("\n📊 Loading bid_landscape_features.csv...")
df = pd.read_csv(DATA_DIR / "bid_landscape_features.csv")
print(f"  Loaded: {len(df):,} rows × {len(df.columns)} cols")

# ─── 2. Aggregate Bid Response Curves ─────────────────────
print("\n📈 Building bid response curves...")

# Create fine-grained bid bins for curve fitting
df['bid_bin'] = pd.cut(df['bid'], bins=50, labels=False)
df['bid_bin_center'] = pd.cut(df['bid'], bins=50).apply(lambda x: x.mid if pd.notna(x) else np.nan)

# Global curve: aggregate by bid bin
global_curve = df.groupby('bid_bin_center').agg(
    n=('bid', 'count'),
    avg_bid=('bid', 'mean'),
    avg_impressions=('impressions', 'mean'),
    avg_clicks=('clicks', 'mean'),
    avg_orders=('orders', 'mean'),
    avg_spend=('spend', 'mean'),
    avg_sales=('sales', 'mean'),
    avg_ctr=('ctr', 'mean'),
    avg_cvr=('cvr', 'mean'),
    avg_cpc=('cpc', 'mean'),
    conv_rate=('has_orders', 'mean'),
    median_impressions=('impressions', 'median'),
    median_clicks=('clicks', 'median'),
).reset_index()

global_curve = global_curve[global_curve['n'] >= 20]  # min sample size
print(f"  Global curve: {len(global_curve)} bid bins")

# ─── 3. Parametric Curve Fitting ──────────────────────────
print("\n🔧 Fitting parametric bid-response curves...")

def log_response(x, a, b, c):
    """Log response: y = a * log(1 + b*x) + c"""
    return a * np.log1p(b * x) + c

def power_response(x, a, b, c):
    """Power response: y = a * x^b + c"""
    return a * np.power(x + 0.001, b) + c

def saturation_response(x, a, b, c):
    """Saturation (Hill function): y = a * x^c / (b^c + x^c)"""
    return a * np.power(x, c) / (np.power(b, c) + np.power(x, c))

curve_results = {}

# Fit curves for each metric
metrics_to_fit = {
    'impressions': ('avg_impressions', 'Impressions'),
    'clicks': ('avg_clicks', 'Clicks'),
    'orders': ('avg_orders', 'Orders'),
    'ctr': ('avg_ctr', 'CTR'),
    'cvr': ('avg_cvr', 'CVR'),
    'conv_rate': ('conv_rate', 'Conversion Rate'),
}

for metric_key, (col, label) in metrics_to_fit.items():
    x = global_curve['avg_bid'].values
    y = global_curve[col].values

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[mask], y[mask]

    best_fit = None
    best_r2 = -np.inf
    best_name = None

    for func, name in [(log_response, 'log'), (power_response, 'power'), (saturation_response, 'saturation')]:
        try:
            popt, _ = optimize.curve_fit(func, x, y, p0=[1, 1, 0.5], maxfev=10000,
                                          bounds=(-np.inf, np.inf))
            y_pred = func(x, *popt)
            r2 = r2_score(y, y_pred)
            if r2 > best_r2:
                best_r2 = r2
                best_fit = popt.tolist()
                best_name = name
        except Exception:
            continue

    # Isotonic regression as non-parametric alternative
    if metric_key in ['impressions', 'clicks']:
        iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
        y_iso = iso.fit_transform(x, y)
        r2_iso = r2_score(y, y_iso)

        curve_results[metric_key] = {
            'parametric': {'function': best_name, 'params': best_fit, 'r2': round(best_r2, 4)},
            'isotonic_r2': round(r2_iso, 4),
            'label': label,
        }
        print(f"  {label:20s} | parametric ({best_name}): R²={best_r2:.4f} | isotonic: R²={r2_iso:.4f}")
    else:
        curve_results[metric_key] = {
            'parametric': {'function': best_name, 'params': best_fit, 'r2': round(best_r2, 4)},
            'label': label,
        }
        print(f"  {label:20s} | parametric ({best_name}): R²={best_r2:.4f}")


# ─── 4. Segment-Level Curves ─────────────────────────────
print("\n📊 Building segment-level curves...")

segment_curves = {}

for (country, match_type), grp in df.groupby(['country', 'match_type']):
    if len(grp) < 100:
        continue

    seg_key = f"{country}_{match_type}"

    # Create bid bins for this segment
    try:
        bins = min(20, max(5, len(grp) // 50))
        grp_binned = grp.copy()
        grp_binned['seg_bid_bin'] = pd.qcut(grp_binned['bid'], q=bins, duplicates='drop')
        grp_binned['seg_bid_center'] = grp_binned['seg_bid_bin'].apply(lambda x: x.mid if pd.notna(x) else np.nan)

        seg_agg = grp_binned.groupby('seg_bid_center').agg(
            n=('bid', 'count'),
            avg_bid=('bid', 'mean'),
            avg_impressions=('impressions', 'mean'),
            avg_clicks=('clicks', 'mean'),
            avg_orders=('orders', 'mean'),
            avg_spend=('spend', 'mean'),
            avg_sales=('sales', 'mean'),
            avg_ctr=('ctr', 'mean'),
            avg_cvr=('cvr', 'mean'),
            conv_rate=('has_orders', 'mean'),
        ).reset_index()

        seg_agg = seg_agg[seg_agg['n'] >= 10]

        if len(seg_agg) < 3:
            continue

        # Fit isotonic for impressions
        x_seg = seg_agg['avg_bid'].values
        y_imp = seg_agg['avg_impressions'].values

        mask = np.isfinite(x_seg) & np.isfinite(y_imp)
        x_seg, y_imp = x_seg[mask], y_imp[mask]

        iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
        y_imp_iso = iso.fit_transform(x_seg, y_imp)

        segment_curves[seg_key] = {
            'country': country,
            'match_type': match_type,
            'n_records': len(grp),
            'n_bins': len(seg_agg),
            'bid_range': [float(grp['bid'].min()), float(grp['bid'].max())],
            'bid_median': float(grp['bid'].median()),
            'avg_impressions': float(grp['impressions'].mean()),
            'avg_orders': float(grp['orders'].mean()),
            'avg_ctr': float(grp['ctr'].mean()),
            'avg_cvr': float(grp['cvr'].mean()),
            'curve_data': {
                'bid': x_seg.tolist(),
                'impressions': y_imp_iso.tolist(),
                'clicks': seg_agg['avg_clicks'].values[mask].tolist(),
                'orders': seg_agg['avg_orders'].values[mask].tolist(),
                'ctr': seg_agg['avg_ctr'].values[mask].tolist(),
                'cvr': seg_agg['avg_cvr'].values[mask].tolist(),
            }
        }
    except Exception:
        continue

print(f"  Built curves for {len(segment_curves)} segments")

# ─── 5. LightGBM Bid-Performance Model ───────────────────
print("\n🌲 Training LightGBM bid-performance models...")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    print("  ⚠️ LightGBM not installed, using sklearn GradientBoosting")
    HAS_LGB = False
    from sklearn.ensemble import GradientBoostingRegressor

# Features for LightGBM
feature_cols = [
    'bid', 'log_bid', 'bid_relative',
    'match_type_enc', 'country_enc',
    'keyword_length', 'keyword_word_count',
]

targets = {
    'log_impressions': 'Log Impressions',
    'log_clicks': 'Log Clicks',
    'has_orders': 'Has Orders (classification proxy)',
}

available_features = [c for c in feature_cols if c in df.columns]
X = df[available_features].copy()

# Handle missing values
X = X.fillna(0)

lgb_models = {}
lgb_metrics = {}

for target_col, target_label in targets.items():
    if target_col not in df.columns:
        continue

    y = df[target_col].values

    # Simple train/test split (80/20)
    np.random.seed(42)
    idx = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if HAS_LGB:
        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=50,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=50,
            random_state=42,
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    lgb_models[target_col] = model
    lgb_metrics[target_col] = {
        'r2': round(r2, 4),
        'mae': round(mae, 4),
        'label': target_label,
    }

    print(f"  {target_label:35s} | R²={r2:.4f} | MAE={mae:.4f}")

# Feature importance
if lgb_models:
    print("\n  Feature importance (log_impressions):")
    imp_model = lgb_models.get('log_impressions')
    if imp_model:
        if HAS_LGB:
            importances = imp_model.feature_importances_
        else:
            importances = imp_model.feature_importances_

        for feat, imp in sorted(zip(available_features, importances), key=lambda x: -x[1]):
            print(f"    {feat:25s} {imp:8.1f}")

# Save models
with open(OUT_DIR / "bid_landscape_lgb.pkl", 'wb') as f:
    pickle.dump({
        'models': lgb_models,
        'features': available_features,
        'metrics': lgb_metrics,
    }, f)

# ─── 6. Optimal Bid Analysis ─────────────────────────────
print("\n🎯 Computing optimal bid zones...")

# For each segment, find the bid range that maximizes efficiency
optimal_bids = {}

for seg_key, seg_data in segment_curves.items():
    curve = seg_data['curve_data']
    bids = np.array(curve['bid'])
    impressions = np.array(curve['impressions'])
    orders = np.array(curve['orders'])

    if len(bids) < 3:
        continue

    # Compute marginal impressions per bid dollar
    marginal_imp = np.diff(impressions) / np.diff(bids).clip(min=0.001)

    # Compute expected orders per impression (conversion efficiency)
    ctr_arr = np.array(curve['ctr'])
    cvr_arr = np.array(curve['cvr'])

    # Expected value per impression at each bid level
    # Higher bid → more impressions but potentially diminishing returns
    imp_efficiency = impressions / bids  # impressions per dollar of bid

    # Find the "sweet spot" — maximum efficiency point
    best_idx = np.argmax(imp_efficiency)

    # Find the point where marginal returns drop below 50% of average
    avg_marginal = np.mean(marginal_imp[marginal_imp > 0]) if np.any(marginal_imp > 0) else 0

    diminishing_idx = len(bids) - 1
    for i, m in enumerate(marginal_imp):
        if m < avg_marginal * 0.3 and i > 0:
            diminishing_idx = i
            break

    optimal_bids[seg_key] = {
        'country': seg_data['country'],
        'match_type': seg_data['match_type'],
        'n_records': seg_data['n_records'],
        'current_median_bid': seg_data['bid_median'],
        'bid_range': seg_data['bid_range'],
        'sweet_spot_bid': float(bids[best_idx]),
        'diminishing_returns_bid': float(bids[min(diminishing_idx, len(bids)-1)]),
        'recommended_range': [
            float(bids[max(0, best_idx - 1)]),
            float(bids[min(diminishing_idx + 1, len(bids)-1)])
        ],
        'efficiency_at_sweet_spot': float(imp_efficiency[best_idx]),
    }

print(f"  Computed optimal bids for {len(optimal_bids)} segments")

# Print top segments
print("\n  Top 10 segments by record count:")
print(f"  {'Segment':20s} {'Records':>8s} {'Median Bid':>10s} {'Sweet Spot':>10s} {'Diminishing':>12s}")
print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10} {'-'*12}")
for seg_key, info in sorted(optimal_bids.items(), key=lambda x: -x[1]['n_records'])[:10]:
    print(f"  {seg_key:20s} {info['n_records']:>8,d} ${info['current_median_bid']:>8.2f} "
          f"${info['sweet_spot_bid']:>8.2f} ${info['diminishing_returns_bid']:>8.2f}")

# ─── 7. Bid Adjustment Recommendations ───────────────────
print("\n💡 Generating bid adjustment recommendations...")

# Merge optimal bids back to individual records
recs = []
for _, row in df.iterrows():
    seg_key = f"{row['country']}_{row['match_type']}"
    if seg_key in optimal_bids:
        opt = optimal_bids[seg_key]
        current = row['bid']
        sweet = opt['sweet_spot_bid']
        rec_low, rec_high = opt['recommended_range']

        if current < rec_low:
            action = 'increase'
            suggested = rec_low
        elif current > rec_high:
            action = 'decrease'
            suggested = rec_high
        else:
            action = 'maintain'
            suggested = current

        recs.append({
            'country': row['country'],
            'match_type': row['match_type'],
            'keyword': row.get('keyword', ''),
            'campaign': row.get('campaign', ''),
            'current_bid': current,
            'sweet_spot_bid': sweet,
            'recommended_bid': suggested,
            'action': action,
            'impressions': row.get('impressions', 0),
            'orders': row.get('orders', 0),
            'acos': row.get('acos', 0),
        })

recs_df = pd.DataFrame(recs)
if len(recs_df) > 0:
    action_summary = recs_df['action'].value_counts()
    print(f"  Total recommendations: {len(recs_df):,}")
    for action, count in action_summary.items():
        pct = count / len(recs_df) * 100
        print(f"    {action:10s}: {count:>7,d} ({pct:.1f}%)")

    # Save recommendations
    recs_df.to_csv(OUT_DIR / "bid_recommendations.csv", index=False)
    print(f"\n  Saved: bid_recommendations.csv")

# ─── 8. Save All Results ─────────────────────────────────
print("\n💾 Saving all results...")

# Global curve data for dashboard
global_curve_data = {
    'bid': global_curve['avg_bid'].tolist(),
    'impressions': global_curve['avg_impressions'].tolist(),
    'clicks': global_curve['avg_clicks'].tolist(),
    'orders': global_curve['avg_orders'].tolist(),
    'ctr': global_curve['avg_ctr'].tolist(),
    'cvr': global_curve['avg_cvr'].tolist(),
    'cpc': global_curve['avg_cpc'].tolist(),
    'conv_rate': global_curve['conv_rate'].tolist(),
    'n_samples': global_curve['n'].tolist(),
}

results = {
    'model_type': 'cross_sectional_bid_landscape',
    'data_source': 'keywords.csv (500K targeting records, 90-day aggregated)',
    'n_records': len(df),
    'n_segments': len(segment_curves),
    'n_recommendations': len(recs_df) if len(recs_df) > 0 else 0,
    'curve_fitting': curve_results,
    'lgb_metrics': lgb_metrics,
    'global_curve': global_curve_data,
    'segment_summary': {
        k: {kk: vv for kk, vv in v.items() if kk != 'curve_data'}
        for k, v in segment_curves.items()
    },
    'optimal_bids': optimal_bids,
    'action_distribution': action_summary.to_dict() if len(recs_df) > 0 else {},
}

with open(OUT_DIR / "bid_landscape_results.json", 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Save curve data separately for dashboard visualization
with open(OUT_DIR / "bid_landscape_curves.json", 'w') as f:
    json.dump({
        'global_curve': global_curve_data,
        'segment_curves': segment_curves,
    }, f, indent=2, default=str)

print(f"  → bid_landscape_results.json")
print(f"  → bid_landscape_curves.json")
print(f"  → bid_landscape_lgb.pkl")
print(f"  → bid_recommendations.csv")

# ─── 9. Summary ──────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ Bid Landscape Model Complete!")
print("=" * 60)
print(f"  Records analyzed: {len(df):,}")
print(f"  Segments modeled: {len(segment_curves)}")
if lgb_metrics:
    for k, v in lgb_metrics.items():
        print(f"  LGB {v['label']}: R²={v['r2']:.4f}")
if len(recs_df) > 0:
    print(f"  Bid recommendations: {len(recs_df):,}")
    avg_change = (recs_df['recommended_bid'] - recs_df['current_bid']).mean()
    print(f"  Average bid change: ${avg_change:+.3f}")
print(f"\n  Output: {OUT_DIR}")
