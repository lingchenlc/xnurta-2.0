"""
Xnurta 2.0 — Phase 1: AdTFT Feature Engineering Pipeline
=========================================================
Transforms cleaned campaign daily data into TFT-ready features:
  1. Filter to eligible campaigns (>= 30 days, >= 14 active, >= $10 spend)
  2. Fill missing dates with zeros
  3. Compute rolling window features (7d/14d/30d MA, std, momentum)
  4. Add calendar features (day_of_week, is_weekend, month, Fourier)
  5. Add promo day markers
  6. Construct forward-looking labels (future 1d/3d/7d)
  7. Encode categorical features
  8. Save feature-engineered dataset
"""

import pandas as pd
import numpy as np
import os
import time
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "dashboard").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

# TFT model parameters
LOOKBACK_WINDOW = 21      # Use 21 days of history as input
PREDICTION_HORIZONS = [1, 3, 7]  # Predict 1d, 3d, 7d ahead
TARGET_METRICS = ['spend', 'sales', 'orders', 'acos']

# Rolling window sizes
ROLLING_WINDOWS = [7, 14]

# Eligibility criteria
MIN_TOTAL_DAYS = 30
MIN_ACTIVE_DAYS = 14
MIN_TOTAL_SPEND = 10.0

# ─── Promo Calendar ──────────────────────────────────────────────
# Major Amazon promo events (approximate dates for 2026)
PROMO_DATES = {
    # US promos
    'presidents_day': ['2026-02-16'],
    'valentines': ['2026-02-14'],
    'spring_sale': ['2026-03-10', '2026-03-11', '2026-03-12'],
    # Weekly patterns captured by day_of_week
}

# Country-specific shopping days
COUNTRY_PROMOS = {
    '墨西哥': {'buen_fin': []},  # November event
    '澳大利亚': {'click_frenzy': []},
}


def load_and_filter(verbose=True):
    """Load campaign data and filter to eligible campaigns."""
    t0 = time.time()

    if verbose:
        print("=" * 70)
        print("Step 1: Load & Filter Eligible Campaigns")
        print("=" * 70)

    df = pd.read_csv(DATA_DIR / "campaign_daily_clean.csv", parse_dates=['date'])

    # Create unique campaign_id
    df['campaign_id'] = df['country'] + '|' + df['store'] + '|' + df['campaign']

    if verbose:
        print(f"  Loaded {len(df):,} rows, {df['campaign_id'].nunique():,} campaigns")

    # Compute eligibility
    campaign_stats = df.groupby('campaign_id').agg(
        total_days=('date', 'nunique'),
        active_days=('impressions', lambda x: (x > 0).sum()),
        total_spend=('spend', 'sum'),
    )

    eligible_ids = campaign_stats[
        (campaign_stats['total_days'] >= MIN_TOTAL_DAYS) &
        (campaign_stats['active_days'] >= MIN_ACTIVE_DAYS) &
        (campaign_stats['total_spend'] >= MIN_TOTAL_SPEND)
    ].index

    df = df[df['campaign_id'].isin(eligible_ids)].copy()

    if verbose:
        print(f"  After filtering: {len(df):,} rows, {df['campaign_id'].nunique():,} campaigns")
        print(f"  Spend coverage: ${df['spend'].sum():,.0f}")
        print(f"  Time: {time.time()-t0:.1f}s")

    return df


def fill_missing_dates(df, verbose=True):
    """Fill missing dates for each campaign with zeros."""
    t0 = time.time()

    if verbose:
        print("\n" + "=" * 70)
        print("Step 2: Fill Missing Dates")
        print("=" * 70)

    date_range = pd.date_range(df['date'].min(), df['date'].max(), freq='D')

    # Get static features for each campaign
    static_cols = ['country', 'store', 'campaign', 'campaign_type',
                   'bidding_strategy', 'ai_target', 'ai_status', 'campaign_id']

    # Use last known values for static features
    campaign_static = df.groupby('campaign_id')[static_cols].last().reset_index(drop=True)
    campaign_ids = campaign_static['campaign_id'].unique()

    # Build complete index
    full_idx = pd.MultiIndex.from_product(
        [campaign_ids, date_range],
        names=['campaign_id', 'date']
    )

    # Numeric columns to fill with 0
    numeric_cols = ['budget', 'impressions', 'clicks', 'spend', 'orders',
                    'sales', 'units', 'promoted_sales', 'other_sales',
                    'promoted_orders', 'new_customer_orders', 'new_customer_sales']

    # Ratio columns to fill with NaN (will be recomputed)
    ratio_cols = ['ctr', 'cpc', 'cvr', 'acos', 'roas']

    df_indexed = df.set_index(['campaign_id', 'date'])
    df_full = df_indexed.reindex(full_idx)

    # Fill numeric columns with 0
    for col in numeric_cols:
        if col in df_full.columns:
            df_full[col] = df_full[col].fillna(0)

    # Forward-fill static features within each campaign
    for col in static_cols:
        if col in df_full.columns and col != 'campaign_id':
            df_full[col] = df_full.groupby(level=0)[col].ffill().bfill()

    # Recompute ratio metrics (avoid division by zero)
    df_full['ctr'] = np.where(df_full['impressions'] > 0,
                               df_full['clicks'] / df_full['impressions'], 0)
    df_full['cpc'] = np.where(df_full['clicks'] > 0,
                               df_full['spend'] / df_full['clicks'], 0)
    df_full['cvr'] = np.where(df_full['clicks'] > 0,
                               df_full['orders'] / df_full['clicks'], 0)
    df_full['acos'] = np.where(df_full['sales'] > 0,
                                df_full['spend'] / df_full['sales'],
                                np.where(df_full['spend'] > 0, 2.0, 0))  # 2.0 as penalty for spend with no sales
    df_full['roas'] = np.where(df_full['spend'] > 0,
                                df_full['sales'] / df_full['spend'], 0)

    df_full = df_full.reset_index()

    if verbose:
        print(f"  Before fill: {len(df):,} rows")
        print(f"  After fill:  {len(df_full):,} rows")
        print(f"  Date range: {date_range[0].date()} to {date_range[-1].date()} ({len(date_range)} days)")
        print(f"  Time: {time.time()-t0:.1f}s")

    return df_full


def add_rolling_features(df, verbose=True):
    """Add rolling window features per campaign."""
    t0 = time.time()

    if verbose:
        print("\n" + "=" * 70)
        print("Step 3: Rolling Window Features")
        print("=" * 70)

    df = df.sort_values(['campaign_id', 'date']).reset_index(drop=True)

    # Metrics to compute rolling features for
    metrics = ['impressions', 'clicks', 'spend', 'orders', 'sales',
               'ctr', 'cvr', 'cpc', 'acos', 'roas']

    new_cols = {}

    for window in ROLLING_WINDOWS:
        if verbose:
            print(f"  Computing {window}d rolling features...")

        for metric in metrics:
            # Moving average
            col_ma = f'{metric}_ma{window}d'
            new_cols[col_ma] = df.groupby('campaign_id')[metric].transform(
                lambda x: x.rolling(window, min_periods=max(1, window//2)).mean()
            )

            # Moving standard deviation (volatility)
            col_std = f'{metric}_std{window}d'
            new_cols[col_std] = df.groupby('campaign_id')[metric].transform(
                lambda x: x.rolling(window, min_periods=max(1, window//2)).std()
            )

        # Momentum (current vs MA)
        for metric in ['spend', 'sales', 'orders', 'impressions']:
            col_mom = f'{metric}_momentum{window}d'
            ma = new_cols[f'{metric}_ma{window}d']
            new_cols[col_mom] = np.where(ma > 0, (df[metric] - ma) / ma, 0)

    # WoW change (7d lag difference)
    if verbose:
        print("  Computing week-over-week changes...")

    for metric in ['spend', 'sales', 'orders', 'impressions', 'clicks']:
        lag7 = df.groupby('campaign_id')[metric].shift(7)
        new_cols[f'{metric}_wow_change'] = np.where(
            lag7 > 0, (df[metric] - lag7) / lag7, 0
        )
        # Also add raw lag values
        new_cols[f'{metric}_lag1'] = df.groupby('campaign_id')[metric].shift(1).fillna(0)
        new_cols[f'{metric}_lag7'] = lag7.fillna(0)

    # Cumulative metrics (running totals within the window)
    if verbose:
        print("  Computing cumulative features...")

    for metric in ['spend', 'orders', 'sales']:
        new_cols[f'{metric}_cumsum'] = df.groupby('campaign_id')[metric].cumsum()
        new_cols[f'{metric}_expanding_mean'] = df.groupby('campaign_id')[metric].transform(
            lambda x: x.expanding().mean()
        )

    # Assign all new columns at once
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    if verbose:
        n_new = len(new_cols)
        print(f"  Added {n_new} rolling features")
        print(f"  Total columns: {df.shape[1]}")
        print(f"  Time: {time.time()-t0:.1f}s")

    return df


def add_calendar_features(df, verbose=True):
    """Add calendar and seasonality features."""
    t0 = time.time()

    if verbose:
        print("\n" + "=" * 70)
        print("Step 4: Calendar & Seasonality Features")
        print("=" * 70)

    # Basic calendar
    df['day_of_week'] = df['date'].dt.dayofweek  # 0=Mon, 6=Sun
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)

    # Fourier seasonal encoding
    # Weekly cycle (period=7)
    day_in_year = df['date'].dt.dayofyear
    for k in [1, 2]:  # 2 Fourier pairs for weekly
        df[f'weekly_sin_{k}'] = np.sin(2 * np.pi * k * df['day_of_week'] / 7)
        df[f'weekly_cos_{k}'] = np.cos(2 * np.pi * k * df['day_of_week'] / 7)

    # Monthly cycle (period=30.44)
    for k in [1, 2]:
        df[f'monthly_sin_{k}'] = np.sin(2 * np.pi * k * df['day_of_month'] / 30.44)
        df[f'monthly_cos_{k}'] = np.cos(2 * np.pi * k * df['day_of_month'] / 30.44)

    # Day-in-window position (normalized 0-1)
    date_min = df['date'].min()
    date_max = df['date'].max()
    date_span = (date_max - date_min).days
    df['time_position'] = (df['date'] - date_min).dt.days / max(date_span, 1)

    # Promo day markers
    df['is_promo'] = 0
    for promo_name, dates in PROMO_DATES.items():
        for d in dates:
            df.loc[df['date'] == d, 'is_promo'] = 1

    # Days since campaign first activity
    first_active = df[df['impressions'] > 0].groupby('campaign_id')['date'].min()
    df['campaign_first_active'] = df['campaign_id'].map(first_active)
    df['days_since_first_active'] = (df['date'] - df['campaign_first_active']).dt.days
    df['days_since_first_active'] = df['days_since_first_active'].fillna(0).clip(lower=0)
    df.drop('campaign_first_active', axis=1, inplace=True)

    # Is this the first week of campaign? (cold start indicator)
    df['is_cold_start'] = (df['days_since_first_active'] < 7).astype(int)

    if verbose:
        cal_cols = [c for c in df.columns if any(x in c for x in ['day_of', 'is_weekend', 'month', 'week_of',
                                                                    'sin_', 'cos_', 'time_position', 'is_promo',
                                                                    'days_since', 'is_cold'])]
        print(f"  Added {len(cal_cols)} calendar features: {cal_cols}")
        print(f"  Time: {time.time()-t0:.1f}s")

    return df


def add_prediction_labels(df, verbose=True):
    """Construct forward-looking labels for each prediction horizon."""
    t0 = time.time()

    if verbose:
        print("\n" + "=" * 70)
        print("Step 5: Prediction Labels (Future Targets)")
        print("=" * 70)

    df = df.sort_values(['campaign_id', 'date']).reset_index(drop=True)

    for horizon in PREDICTION_HORIZONS:
        if verbose:
            print(f"  Building {horizon}d-ahead labels...")

        if horizon == 1:
            # Next day value
            for metric in TARGET_METRICS:
                df[f'target_{metric}_{horizon}d'] = df.groupby('campaign_id')[metric].shift(-1)
        else:
            # Average over next N days
            for metric in TARGET_METRICS:
                # Use forward rolling window
                reversed_col = df.groupby('campaign_id')[metric].transform(
                    lambda x: x[::-1].rolling(horizon, min_periods=horizon).mean()[::-1]
                )
                # Shift by -1 to exclude current day
                df[f'target_{metric}_{horizon}d'] = df.groupby('campaign_id')[
                    metric].transform(
                    lambda x: x.shift(-1).rolling(horizon, min_periods=horizon).mean()
                )

    # Drop rows where any target is NaN (end of series)
    target_cols = [c for c in df.columns if c.startswith('target_')]
    n_before = len(df)

    # Mark rows with valid targets
    df['has_valid_targets'] = df[target_cols].notna().all(axis=1)

    if verbose:
        n_valid = df['has_valid_targets'].sum()
        print(f"  Target columns: {target_cols}")
        print(f"  Rows with valid targets: {n_valid:,} / {n_before:,} ({n_valid/n_before*100:.1f}%)")
        print(f"  Time: {time.time()-t0:.1f}s")

    return df


def encode_categoricals(df, verbose=True):
    """Encode categorical features for the model."""
    t0 = time.time()

    if verbose:
        print("\n" + "=" * 70)
        print("Step 6: Categorical Feature Encoding")
        print("=" * 70)

    # Country encoding
    country_map = {c: i for i, c in enumerate(sorted(df['country'].unique()))}
    df['country_encoded'] = df['country'].map(country_map)

    # Bidding strategy encoding
    bid_strat_map = {s: i for i, s in enumerate(sorted(df['bidding_strategy'].dropna().unique()))}
    df['bidding_strategy_encoded'] = df['bidding_strategy'].map(bid_strat_map).fillna(-1).astype(int)

    # AI status encoding (binary: on/off)
    df['ai_enabled'] = (df['ai_status'] != 'AI未开启').astype(int)

    # Active status encoding
    df['is_active'] = (df['active_status'] == '已启用').astype(int)

    # Store encoding (hash to reduce cardinality)
    store_map = {s: i for i, s in enumerate(sorted(df['store'].unique()))}
    df['store_encoded'] = df['store'].map(store_map)

    if verbose:
        print(f"  Countries: {len(country_map)} → {country_map}")
        print(f"  Bidding strategies: {len(bid_strat_map)}")
        print(f"  Stores: {len(store_map)}")
        print(f"  AI enabled: {df['ai_enabled'].mean()*100:.1f}%")
        print(f"  Time: {time.time()-t0:.1f}s")

    # Save encoding maps for inference
    encoding_maps = {
        'country_map': country_map,
        'bid_strat_map': bid_strat_map,
        'store_map': store_map,
    }

    return df, encoding_maps


def build_time_split(df, verbose=True):
    """Split data by time into train/val/test sets."""
    t0 = time.time()

    if verbose:
        print("\n" + "=" * 70)
        print("Step 7: Time-based Train/Val/Test Split")
        print("=" * 70)

    dates = sorted(df['date'].unique())
    n_dates = len(dates)

    # With 60 days and lookback=21, prediction_horizon=7:
    # - Need 21 days of history before first prediction
    # - Last prediction at day 53 (day 60 - 7)
    # - Usable window: days 22-53 (32 days of targets)
    # Split: ~70% train, ~15% val, ~15% test

    # Training: days where rolling features are stable
    # We use the date-based split for the TARGET dates
    train_end_idx = int(n_dates * 0.70)  # ~day 42
    val_end_idx = int(n_dates * 0.85)    # ~day 51

    train_end_date = dates[train_end_idx]
    val_end_date = dates[val_end_idx]

    df['split'] = 'train'
    df.loc[df['date'] > train_end_date, 'split'] = 'val'
    df.loc[df['date'] > val_end_date, 'split'] = 'test'

    if verbose:
        for split in ['train', 'val', 'test']:
            mask = df['split'] == split
            n = mask.sum()
            n_valid = (mask & df['has_valid_targets']).sum()
            d_min = df.loc[mask, 'date'].min()
            d_max = df.loc[mask, 'date'].max()
            print(f"  {split:5s}: {n:>8,} rows, {n_valid:>8,} with targets, "
                  f"dates {d_min.date()} to {d_max.date()}")
        print(f"  Time: {time.time()-t0:.1f}s")

    return df


def save_features(df, encoding_maps, verbose=True):
    """Save the feature-engineered dataset."""
    t0 = time.time()

    if verbose:
        print("\n" + "=" * 70)
        print("Step 8: Save Feature Dataset")
        print("=" * 70)

    # Save full feature dataset
    output_path = FEATURE_DIR / "tft_features.parquet"
    try:
        df.to_parquet(output_path, index=False, engine='pyarrow')
    except ImportError:
        output_path = FEATURE_DIR / "tft_features.csv"
        df.to_csv(output_path, index=False)
        print("  (Saved as CSV — pyarrow not available)")

    # Save encoding maps
    import json
    maps_path = FEATURE_DIR / "encoding_maps.json"
    with open(maps_path, 'w') as f:
        json.dump(encoding_maps, f, ensure_ascii=False, indent=2)

    # Save feature metadata
    # Categorize columns by type
    static_features = ['country_encoded', 'store_encoded', 'bidding_strategy_encoded',
                       'ai_enabled']

    known_future_features = ['day_of_week', 'is_weekend', 'day_of_month', 'month',
                              'week_of_year', 'time_position', 'is_promo',
                              'weekly_sin_1', 'weekly_cos_1', 'weekly_sin_2', 'weekly_cos_2',
                              'monthly_sin_1', 'monthly_cos_1', 'monthly_sin_2', 'monthly_cos_2']

    observed_features = ['impressions', 'clicks', 'spend', 'orders', 'sales',
                         'ctr', 'cvr', 'cpc', 'acos', 'roas', 'budget',
                         'units', 'promoted_sales', 'other_sales',
                         'is_active', 'days_since_first_active', 'is_cold_start']

    # Add rolling features
    rolling_features = [c for c in df.columns if any(
        x in c for x in ['_ma7d', '_ma14d', '_std7d', '_std14d',
                          '_momentum', '_wow_change', '_lag1', '_lag7',
                          '_cumsum', '_expanding_mean']
    )]
    observed_features.extend(rolling_features)

    target_features = [c for c in df.columns if c.startswith('target_')]

    metadata = {
        'static_features': static_features,
        'known_future_features': known_future_features,
        'observed_features': [f for f in observed_features if f in df.columns],
        'target_features': target_features,
        'lookback_window': LOOKBACK_WINDOW,
        'prediction_horizons': PREDICTION_HORIZONS,
        'n_campaigns': df['campaign_id'].nunique(),
        'n_dates': df['date'].nunique(),
        'date_range': [str(df['date'].min().date()), str(df['date'].max().date())],
        'total_rows': len(df),
    }

    meta_path = FEATURE_DIR / "feature_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        file_size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"  Saved: {output_path} ({file_size_mb:.0f} MB)")
        print(f"  Total columns: {df.shape[1]}")
        print(f"  Static features ({len(static_features)}): {static_features}")
        print(f"  Known future features ({len(known_future_features)}): {known_future_features[:5]}...")
        print(f"  Observed features ({len(metadata['observed_features'])}): first 10 = {metadata['observed_features'][:10]}...")
        print(f"  Target features ({len(target_features)}): {target_features}")
        print(f"  Encoding maps: {maps_path}")
        print(f"  Feature metadata: {meta_path}")
        print(f"  Time: {time.time()-t0:.1f}s")

    return metadata


def main():
    """Run the full feature engineering pipeline."""
    t_start = time.time()

    print("🚀 Xnurta 2.0 — AdTFT Feature Engineering Pipeline")
    print("=" * 70)

    # Step 1: Load & filter
    df = load_and_filter()

    # Step 2: Fill missing dates
    df = fill_missing_dates(df)

    # Step 3: Rolling window features
    df = add_rolling_features(df)

    # Step 4: Calendar & seasonality
    df = add_calendar_features(df)

    # Step 5: Prediction labels
    df = add_prediction_labels(df)

    # Step 6: Categorical encoding
    df, encoding_maps = encode_categoricals(df)

    # Step 7: Time split
    df = build_time_split(df)

    # Step 8: Save
    metadata = save_features(df, encoding_maps)

    total_time = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"✅ Feature engineering complete! Total time: {total_time:.1f}s")
    print(f"   Dataset: {metadata['total_rows']:,} rows × {len(metadata['observed_features']) + len(metadata['static_features']) + len(metadata['known_future_features']) + len(metadata['target_features'])} features")
    print(f"   Campaigns: {metadata['n_campaigns']:,}")
    print(f"   Ready for AdTFT training!")
    print("=" * 70)


if __name__ == "__main__":
    main()
