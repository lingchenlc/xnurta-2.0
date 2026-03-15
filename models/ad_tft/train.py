"""
Xnurta 2.0 — AdTFT Training Pipeline
=======================================
End-to-end training with:
  - MPS (Apple Silicon) or CUDA acceleration
  - OneCycleLR scheduler
  - Early stopping with patience
  - Quantile loss (pinball loss)
  - Training metrics tracking
  - Model checkpoint saving
  - Evaluation & interpretability analysis
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "dashboard").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "models" / "ad_tft"))

from dataset import create_dataloaders
from model import AdTFT, QuantileLoss, get_default_config

# ─── Paths ────────────────────────────────────────────────────────
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
MODEL_DIR = PROJECT_ROOT / "models" / "ad_tft" / "trained"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def get_device():
    """Select best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        static = batch['static'].to(device)
        past_obs = batch['past_observed'].to(device)
        past_known = batch['past_known'].to(device)
        future_known = batch['future_known'].to(device)
        targets = batch['targets'].to(device)

        optimizer.zero_grad()
        predictions, _, _ = model(static, past_obs, past_known, future_known)
        loss = criterion(predictions, targets)

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds = []
    all_targets = []

    for batch in loader:
        static = batch['static'].to(device)
        past_obs = batch['past_observed'].to(device)
        past_known = batch['past_known'].to(device)
        future_known = batch['future_known'].to(device)
        targets = batch['targets'].to(device)

        predictions, _, _ = model(static, past_obs, past_known, future_known)
        loss = criterion(predictions, targets)

        total_loss += loss.item()
        n_batches += 1

        all_preds.append(predictions.cpu())
        all_targets.append(targets.cpu())

    avg_loss = total_loss / max(n_batches, 1)

    # Compute detailed metrics
    all_preds = torch.cat(all_preds, dim=0)    # [N, n_targets, n_quantiles]
    all_targets = torch.cat(all_targets, dim=0)  # [N, n_targets]

    metrics = compute_metrics(all_preds, all_targets)
    metrics['loss'] = avg_loss

    return metrics


def compute_metrics(predictions, targets):
    """Compute regression metrics for each target."""
    # predictions: [N, n_targets, n_quantiles]
    # targets: [N, n_targets]

    n_targets = targets.shape[1]
    median_idx = 2  # quantile index for P50 (0.5)

    metrics = {}

    # Median predictions
    median_preds = predictions[:, :, median_idx]  # [N, n_targets]

    # Overall MAE, RMSE
    mae = (median_preds - targets).abs().mean().item()
    rmse = ((median_preds - targets) ** 2).mean().sqrt().item()
    metrics['mae'] = mae
    metrics['rmse'] = rmse

    # Per-target metrics
    target_names = [
        'spend_1d', 'sales_1d', 'orders_1d', 'acos_1d',
        'spend_3d', 'sales_3d', 'orders_3d', 'acos_3d',
        'spend_7d', 'sales_7d', 'orders_7d', 'acos_7d',
    ]

    for i in range(min(n_targets, len(target_names))):
        name = target_names[i]
        pred = median_preds[:, i]
        actual = targets[:, i]

        mae_i = (pred - actual).abs().mean().item()
        rmse_i = ((pred - actual) ** 2).mean().sqrt().item()

        # MAPE (avoid div by zero)
        mask = actual.abs() > 0.01
        if mask.sum() > 0:
            mape_i = ((pred[mask] - actual[mask]).abs() / actual[mask].abs()).mean().item() * 100
        else:
            mape_i = float('nan')

        metrics[f'{name}_mae'] = mae_i
        metrics[f'{name}_rmse'] = rmse_i
        metrics[f'{name}_mape'] = mape_i

    # Calibration: P10-P90 coverage
    if predictions.shape[2] >= 5:
        p10 = predictions[:, :, 0]  # quantile 0.1
        p90 = predictions[:, :, 4]  # quantile 0.9
        in_range = ((targets >= p10) & (targets <= p90)).float()
        metrics['coverage_80'] = in_range.mean().item() * 100  # Should be ~80%

        p25 = predictions[:, :, 1]
        p75 = predictions[:, :, 3]
        in_50 = ((targets >= p25) & (targets <= p75)).float()
        metrics['coverage_50'] = in_50.mean().item() * 100  # Should be ~50%

    return metrics


def train(config=None):
    """Main training function."""
    t_start = time.time()

    print("=" * 70)
    print("🚀 Xnurta 2.0 — AdTFT Training")
    print("=" * 70)

    device = get_device()
    print(f"  Device: {device}")

    # ─── Data ─────────────────────────────────────────────────────
    print("\n📊 Loading data...")
    train_loader, val_loader, test_loader, norm_stats, dims, metadata = \
        create_dataloaders(FEATURE_DIR, batch_size=256, num_workers=0)

    # ─── Model ────────────────────────────────────────────────────
    if config is None:
        config = get_default_config(dims)

    print(f"\n🏗️  Model Configuration:")
    for k, v in config.items():
        if not isinstance(v, (list, dict)):
            print(f"    {k}: {v}")
        elif k == 'quantiles':
            print(f"    {k}: {v}")

    model = AdTFT(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Trainable parameters: {n_params:,}")

    # ─── Training Setup ───────────────────────────────────────────
    criterion = QuantileLoss(config['quantiles']).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # OneCycleLR scheduler
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        epochs=config['max_epochs'],
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy='cos'
    )

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}

    # ─── Training Loop ────────────────────────────────────────────
    print(f"\n🎯 Training ({config['max_epochs']} max epochs, patience={config['patience']})...")
    print(f"   Steps/epoch: {steps_per_epoch}")
    print("-" * 70)

    for epoch in range(1, config['max_epochs'] + 1):
        t_epoch = time.time()

        # Train
        train_loss = 0
        model.train()
        n_batches = 0

        for batch in train_loader:
            static = batch['static'].to(device)
            past_obs = batch['past_observed'].to(device)
            past_known = batch['past_known'].to(device)
            future_known = batch['future_known'].to(device)
            targets = batch['targets'].to(device)

            optimizer.zero_grad()
            predictions, _, _ = model(static, past_obs, past_known, future_known)
            loss = criterion(predictions, targets)

            loss.backward()
            if config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append({k: v for k, v in val_metrics.items()
                                        if isinstance(v, (int, float))})

        # Progress
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t_epoch

        coverage_str = f"Cov80={val_metrics.get('coverage_80', 0):.1f}%"
        print(f"  Epoch {epoch:3d}/{config['max_epochs']} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"MAE: {val_metrics['mae']:.4f} | "
              f"{coverage_str} | "
              f"LR: {lr:.2e} | "
              f"{elapsed:.1f}s", end="")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'norm_stats': {k: (v[0].tolist(), v[1].tolist())
                              for k, v in norm_stats.items()},
            }, MODEL_DIR / "best_model.pt")
            print(" ★ best", end="")

        else:
            patience_counter += 1

        print()

        if patience_counter >= config['patience']:
            print(f"\n  ⏹  Early stopping at epoch {epoch} (best={best_epoch})")
            break

    # ─── Test Evaluation ──────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("📈 Final Evaluation on Test Set")
    print("=" * 70)

    # Load best model
    checkpoint = torch.load(MODEL_DIR / "best_model.pt", map_location=device,
                             weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\n  Test Loss: {test_metrics['loss']:.6f}")
    print(f"  Overall MAE: {test_metrics['mae']:.4f}")
    print(f"  Overall RMSE: {test_metrics['rmse']:.4f}")
    print(f"  Coverage 80% (P10-P90): {test_metrics.get('coverage_80', 0):.1f}%")
    print(f"  Coverage 50% (P25-P75): {test_metrics.get('coverage_50', 0):.1f}%")

    print(f"\n  Per-target metrics (Test):")
    print(f"  {'Target':<15s} {'MAE':>10s} {'RMSE':>10s} {'MAPE%':>10s}")
    print(f"  {'-'*45}")
    target_names = [
        'spend_1d', 'sales_1d', 'orders_1d', 'acos_1d',
        'spend_3d', 'sales_3d', 'orders_3d', 'acos_3d',
        'spend_7d', 'sales_7d', 'orders_7d', 'acos_7d',
    ]
    for name in target_names:
        mae = test_metrics.get(f'{name}_mae', float('nan'))
        rmse = test_metrics.get(f'{name}_rmse', float('nan'))
        mape = test_metrics.get(f'{name}_mape', float('nan'))
        print(f"  {name:<15s} {mae:>10.4f} {rmse:>10.4f} {mape:>9.1f}%")

    # ─── Feature Importance ───────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("🔍 Feature Importance Analysis")
    print("=" * 70)

    model.eval()
    all_obs_weights = []
    all_static_weights = []
    all_known_weights = []

    with torch.no_grad():
        for batch in test_loader:
            static = batch['static'].to(device)
            past_obs = batch['past_observed'].to(device)
            past_known = batch['past_known'].to(device)
            future_known = batch['future_known'].to(device)

            _, _, feat_weights = model(static, past_obs, past_known, future_known)
            all_obs_weights.append(feat_weights['observed'].cpu())
            all_static_weights.append(feat_weights['static'].cpu())
            all_known_weights.append(feat_weights['known'].cpu())

    # Average weights
    obs_w = torch.cat(all_obs_weights).mean(dim=0).numpy()
    static_w = torch.cat(all_static_weights).mean(dim=0).numpy()
    known_w = torch.cat(all_known_weights).mean(dim=0).numpy()

    # Load feature names
    with open(FEATURE_DIR / 'feature_metadata.json', 'r') as f:
        meta = json.load(f)

    observed_names = [c for c in meta['observed_features']
                      if c in ['impressions', 'clicks', 'spend', 'orders', 'sales',
                               'ctr', 'cvr', 'cpc', 'acos', 'roas', 'budget',
                               'units', 'promoted_sales', 'other_sales',
                               'is_active', 'days_since_first_active', 'is_cold_start'] or
                      any(x in c for x in ['_ma7d', '_ma14d', '_std7d', '_std14d',
                                            '_momentum', '_wow_change', '_lag1', '_lag7',
                                            '_cumsum', '_expanding_mean'])]

    print(f"\n  Top 15 Observed Feature Weights:")
    if len(obs_w) == len(observed_names):
        sorted_idx = np.argsort(obs_w)[::-1][:15]
        for i, idx in enumerate(sorted_idx):
            print(f"    {i+1:2d}. {observed_names[idx]:<30s} {obs_w[idx]:.4f}")
    else:
        sorted_idx = np.argsort(obs_w)[::-1][:15]
        for i, idx in enumerate(sorted_idx):
            print(f"    {i+1:2d}. Feature_{idx:<25d} {obs_w[idx]:.4f}")

    print(f"\n  Static Feature Weights:")
    static_names = meta['static_features']
    for i, (name, w) in enumerate(zip(static_names, static_w)):
        print(f"    {name:<30s} {w:.4f}")

    print(f"\n  Known Future Feature Weights:")
    known_names = meta['known_future_features']
    sorted_kidx = np.argsort(known_w)[::-1]
    for idx in sorted_kidx[:10]:
        if idx < len(known_names):
            print(f"    {known_names[idx]:<30s} {known_w[idx]:.4f}")

    # ─── Save Results ─────────────────────────────────────────────
    results = {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'test_metrics': {k: v for k, v in test_metrics.items()
                          if isinstance(v, (int, float))},
        'training_history': history,
        'config': {k: v for k, v in config.items()},
        'n_params': n_params,
        'total_time': time.time() - t_start,
    }

    with open(MODEL_DIR / "training_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    total_time = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"✅ Training complete! Total time: {total_time/60:.1f} minutes")
    print(f"   Best model saved at epoch {best_epoch}")
    print(f"   Model checkpoint: {MODEL_DIR / 'best_model.pt'}")
    print(f"   Results: {MODEL_DIR / 'training_results.json'}")
    print("=" * 70)

    return model, test_metrics, history


if __name__ == "__main__":
    model, metrics, history = train()
