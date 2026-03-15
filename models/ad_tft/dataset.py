"""
Xnurta 2.0 — AdTFT Dataset & DataLoader
==========================================
Creates sliding-window time series samples from the feature-engineered data.
Each sample contains:
  - static_features: campaign-level categorical features [n_static]
  - past_observed: lookback window of observed metrics [lookback, n_observed]
  - past_known: lookback window of known features [lookback, n_known]
  - future_known: future window of known features [max_horizon, n_known]
  - targets: future values to predict [n_targets]

Key design: windows use data from ANY time period for lookback,
but SPLIT is determined by the TARGET date (when the prediction is for).
"""

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class AdTFTDataset(Dataset):
    """PyTorch Dataset for AdTFT time series prediction."""

    def __init__(self, campaign_data_dict, samples, normalize_stats=None):
        """
        Args:
            campaign_data_dict: dict of cid → {static, observed, known, targets, ...}
            samples: list of (campaign_id, time_index) tuples
            normalize_stats: dict of (mean, std) for normalization
        """
        self.campaign_data = campaign_data_dict
        self.samples = samples
        self.normalize_stats = normalize_stats

        # Get lookback and horizon from the first sample (set by build_samples)
        # These are stored as class-level attributes during build
        self.lookback = self._lookback
        self.max_horizon = self._max_horizon

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cid, t = self.samples[idx]
        data = self.campaign_data[cid]

        # Extract windows
        past_obs = data['observed'][t - self.lookback:t].copy()
        past_known = data['known'][t - self.lookback:t].copy()
        future_known = data['known'][t:t + self.max_horizon].copy()
        static = data['static'].copy()
        targets = data['targets'][t].copy()

        # Handle NaN
        past_obs = np.nan_to_num(past_obs, 0.0)
        past_known = np.nan_to_num(past_known, 0.0)
        future_known = np.nan_to_num(future_known, 0.0)
        targets = np.nan_to_num(targets, 0.0)

        # Normalize
        if self.normalize_stats:
            obs_mean, obs_std = self.normalize_stats['observed']
            past_obs = (past_obs - obs_mean) / obs_std

            known_mean, known_std = self.normalize_stats['known']
            past_known = (past_known - known_mean) / known_std
            future_known = (future_known - known_mean) / known_std

        # Clip extreme values
        past_obs = np.clip(past_obs, -10, 10)
        past_known = np.clip(past_known, -10, 10)
        future_known = np.clip(future_known, -10, 10)

        return {
            'static': torch.tensor(static, dtype=torch.float32),
            'past_observed': torch.tensor(past_obs, dtype=torch.float32),
            'past_known': torch.tensor(past_known, dtype=torch.float32),
            'future_known': torch.tensor(future_known, dtype=torch.float32),
            'targets': torch.tensor(targets, dtype=torch.float32),
        }


def _compute_stats(df, observed_cols, known_cols, target_cols):
    """Compute normalization statistics from training data."""
    stats = {}
    obs_data = df[observed_cols].values.astype(np.float32)
    obs_data = np.nan_to_num(obs_data, 0.0)
    stats['observed'] = (
        np.nanmean(obs_data, axis=0),
        np.nanstd(obs_data, axis=0) + 1e-8
    )

    known_data = df[known_cols].values.astype(np.float32)
    known_data = np.nan_to_num(known_data, 0.0)
    stats['known'] = (
        np.nanmean(known_data, axis=0),
        np.nanstd(known_data, axis=0) + 1e-8
    )

    target_data = df[target_cols].values.astype(np.float32)
    target_data = np.nan_to_num(target_data, 0.0)
    stats['targets'] = (
        np.nanmean(target_data, axis=0),
        np.nanstd(target_data, axis=0) + 1e-8
    )

    return stats


def create_dataloaders(feature_dir, batch_size=256, num_workers=0):
    """
    Create train/val/test DataLoaders.

    Key: All campaign data is loaded into memory. Sliding windows are created
    from the FULL time series. Split is determined by which date the TARGET
    prediction corresponds to (not the lookback window).
    """
    feature_dir = Path(feature_dir)

    with open(feature_dir / 'feature_metadata.json', 'r') as f:
        metadata = json.load(f)

    print("Loading feature data...")
    parquet_path = feature_dir / 'tft_features.parquet'
    csv_path = feature_dir / 'tft_features.csv'

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        df = pd.read_csv(csv_path, parse_dates=['date'])

    print(f"  Loaded {len(df):,} rows")

    # Feature column names
    static_cols = metadata['static_features']
    observed_cols = [c for c in metadata['observed_features'] if c in df.columns]
    known_cols = [c for c in metadata['known_future_features'] if c in df.columns]
    target_cols = metadata['target_features']

    lookback = metadata['lookback_window']
    max_horizon = max(metadata['prediction_horizons'])

    # Sort all data by campaign and date
    df = df.sort_values(['campaign_id', 'date']).reset_index(drop=True)

    # Get split dates from data
    train_dates = df[df['split'] == 'train']['date']
    val_dates = df[df['split'] == 'val']['date']
    test_dates = df[df['split'] == 'test']['date']

    train_date_range = set(train_dates.dt.strftime('%Y-%m-%d'))
    val_date_range = set(val_dates.dt.strftime('%Y-%m-%d'))
    test_date_range = set(test_dates.dt.strftime('%Y-%m-%d'))

    print(f"  Train dates: {min(train_date_range)} to {max(train_date_range)}")
    print(f"  Val dates:   {min(val_date_range)} to {max(val_date_range)}")
    print(f"  Test dates:  {min(test_date_range)} to {max(test_date_range)}")

    # Build campaign data dict (ALL data, not split)
    campaigns = df['campaign_id'].unique()
    campaign_data = {}
    campaign_dates = {}

    for cid in campaigns:
        cdf = df[df['campaign_id'] == cid].reset_index(drop=True)
        n_days = len(cdf)

        if n_days < lookback + max_horizon:
            continue

        campaign_data[cid] = {
            'static': cdf[static_cols].iloc[0].values.astype(np.float32),
            'observed': cdf[observed_cols].values.astype(np.float32),
            'known': cdf[known_cols].values.astype(np.float32),
            'targets': cdf[target_cols].values.astype(np.float32),
            'has_valid': cdf['has_valid_targets'].values,
        }
        campaign_dates[cid] = cdf['date'].dt.strftime('%Y-%m-%d').values

    print(f"  Campaigns with enough data: {len(campaign_data):,}")

    # Build samples for each split
    # Split is determined by the date at time index t (the prediction point)
    train_samples = []
    val_samples = []
    test_samples = []

    for cid in campaign_data:
        dates = campaign_dates[cid]
        data = campaign_data[cid]
        n_days = len(dates)

        for t in range(lookback, n_days - max_horizon + 1):
            if not data['has_valid'][t]:
                continue

            date_str = dates[t]
            if date_str in train_date_range:
                train_samples.append((cid, t))
            elif date_str in val_date_range:
                val_samples.append((cid, t))
            elif date_str in test_date_range:
                test_samples.append((cid, t))

    print(f"  Train samples: {len(train_samples):,}")
    print(f"  Val samples:   {len(val_samples):,}")
    print(f"  Test samples:  {len(test_samples):,}")

    # Compute normalization stats from training data only
    train_df = df[df['split'] == 'train']
    normalize_stats = _compute_stats(train_df, observed_cols, known_cols, target_cols)

    # Clean up stats
    for key in normalize_stats:
        mean, std = normalize_stats[key]
        std = np.where(std < 1e-8, 1.0, std)
        mean = np.nan_to_num(mean, 0.0)
        normalize_stats[key] = (mean, std)

    # Create datasets
    # Monkey-patch class attributes for lookback/horizon
    AdTFTDataset._lookback = lookback
    AdTFTDataset._max_horizon = max_horizon

    train_dataset = AdTFTDataset(campaign_data, train_samples, normalize_stats)
    val_dataset = AdTFTDataset(campaign_data, val_samples, normalize_stats)
    test_dataset = AdTFTDataset(campaign_data, test_samples, normalize_stats)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )

    # Feature dimensions
    dims = {
        'n_static': len(static_cols),
        'n_observed': len(observed_cols),
        'n_known': len(known_cols),
        'n_targets': len(target_cols),
        'lookback': lookback,
        'max_horizon': max_horizon,
    }
    print(f"  Dimensions: {dims}")

    return train_loader, val_loader, test_loader, normalize_stats, dims, metadata
