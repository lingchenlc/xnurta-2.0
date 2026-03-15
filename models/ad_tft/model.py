"""
Xnurta 2.0 — AdTFT Model Architecture (Optimized)
=====================================================
Streamlined Temporal Fusion Transformer for ad performance prediction.
Optimized for speed while maintaining interpretability.

Architecture:
  1. Feature Projection: Linear → BatchNorm for each feature group
  2. Feature Attention: Lightweight attention-based feature selection
  3. GRU Encoder: Captures temporal patterns (faster than LSTM)
  4. Temporal Attention: Interpretable multi-head attention
  5. Quantile Output: Multi-quantile predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FeatureAttention(nn.Module):
    """Lightweight feature selection via attention weights."""

    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(n_features, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.Softmax(dim=-1)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        x: [batch, (seq,) n_features]
        returns: projected [batch, (seq,) hidden_dim], weights [batch, (seq,) n_features]
        """
        weights = self.gate(x)  # soft feature selection
        weighted = x * weights
        projected = self.proj(weighted)
        projected = self.norm(projected)
        return projected, weights


class GatedResidualBlock(nn.Module):
    """Simplified GRN: Linear → ELU → Linear → Gate → Add&Norm."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        g = torch.sigmoid(self.gate(h))
        return self.norm(x + h * g)


class TemporalAttention(nn.Module):
    """Multi-head attention over time steps with interpretable weights."""

    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = dim // n_heads

        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, dk]
        Q, K, V = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.out(out)
        return self.norm(x + self.dropout(out)), attn


class AdTFT(nn.Module):
    """
    Optimized Advertising Temporal Fusion Transformer.

    Flow:
    1. Static → Linear → context vector
    2. Observed → FeatureAttention → projection
    3. Known → FeatureAttention → projection
    4. Concat projections → GRU encoder
    5. Temporal self-attention
    6. Pool → GRN → Quantile heads
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        H = config['hidden_dim']
        self.n_quantiles = len(config['quantiles'])
        self.quantiles = config['quantiles']

        # ─── Static features ───
        self.static_proj = nn.Sequential(
            nn.Linear(config['n_static'], H),
            nn.ELU(),
            nn.Linear(H, H),
        )

        # ─── Observed features: attention-based selection ───
        self.obs_attn = FeatureAttention(config['n_observed'], H)

        # ─── Known features ───
        self.known_attn = FeatureAttention(config['n_known'], H)

        # ─── Temporal encoder (GRU for speed) ───
        self.encoder = nn.GRU(
            input_size=H * 2,  # observed + known projected
            hidden_size=H,
            num_layers=2,
            batch_first=True,
            dropout=config['dropout'],
        )

        # ─── Temporal attention ───
        self.temporal_attn = TemporalAttention(H, n_heads=config['n_heads'],
                                                dropout=config['dropout'])

        # ─── Post-processing ───
        self.post_grn = GatedResidualBlock(H, dropout=config['dropout'])

        # Context enrichment (add static info to temporal output)
        self.context_gate = nn.Sequential(
            nn.Linear(H * 2, H),
            nn.Sigmoid()
        )
        self.context_proj = nn.Linear(H * 2, H)

        # ─── Quantile output heads ───
        # Shared base + per-target heads for efficiency
        self.output_base = nn.Sequential(
            nn.Linear(H, H),
            nn.ELU(),
            nn.Dropout(config['dropout']),
        )
        self.output_heads = nn.ModuleList([
            nn.Linear(H, self.n_quantiles)
            for _ in range(config['n_targets'])
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, static, past_observed, past_known, future_known=None):
        """
        Args:
            static: [B, n_static]
            past_observed: [B, T, n_observed]
            past_known: [B, T, n_known]
            future_known: [B, H, n_known] (unused in simplified version)
        Returns:
            predictions: [B, n_targets, n_quantiles]
            attn_weights: [B, n_heads, T, T]
            feature_weights: dict
        """
        B, T, _ = past_observed.shape

        # 1. Static embedding
        static_emb = self.static_proj(static)  # [B, H]

        # 2. Feature attention + projection
        obs_proj, obs_weights = self.obs_attn(past_observed)    # [B, T, H], [B, T, F]
        known_proj, known_weights = self.known_attn(past_known)  # [B, T, H], [B, T, F]

        # 3. Combine and encode
        temporal_input = torch.cat([obs_proj, known_proj], dim=-1)  # [B, T, 2H]

        # Initialize GRU hidden with static embedding
        h0 = static_emb.unsqueeze(0).expand(2, -1, -1).contiguous()  # [2, B, H]
        encoded, _ = self.encoder(temporal_input, h0)  # [B, T, H]

        # 4. Temporal attention
        attended, attn_weights = self.temporal_attn(encoded)  # [B, T, H]

        # 5. Pool over time (attention-weighted)
        last_attn_avg = attn_weights.mean(dim=1)[:, -1, :]  # [B, T]
        last_attn_avg = last_attn_avg.unsqueeze(-1)  # [B, T, 1]
        pooled = (attended * last_attn_avg).sum(dim=1)  # [B, H]

        # 6. Enrich with static context
        combined = torch.cat([pooled, static_emb], dim=-1)  # [B, 2H]
        gate = self.context_gate(combined)
        enriched = self.context_proj(combined) * gate + pooled * (1 - gate)

        # 7. Post-processing
        enriched = self.post_grn(enriched)  # [B, H]

        # 8. Quantile predictions
        base = self.output_base(enriched)  # [B, H]
        predictions = torch.stack([head(base) for head in self.output_heads], dim=1)
        # [B, n_targets, n_quantiles]

        feature_weights = {
            'observed': obs_weights.mean(dim=1),  # [B, n_obs]
            'known': known_weights.mean(dim=1),    # [B, n_known]
            'static': static_emb,                  # placeholder
        }

        return predictions, attn_weights, feature_weights


class QuantileLoss(nn.Module):
    """Pinball loss for quantile regression."""

    def __init__(self, quantiles):
        super().__init__()
        self.register_buffer('quantiles',
                             torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, predictions, targets):
        """
        predictions: [B, n_targets, n_quantiles]
        targets: [B, n_targets]
        """
        targets = targets.unsqueeze(-1)  # [B, T, 1]
        errors = targets - predictions    # [B, T, Q]
        loss = torch.max(
            self.quantiles * errors,
            (self.quantiles - 1) * errors
        )
        return loss.mean()


def get_default_config(dims):
    """Default model config."""
    return {
        'n_static': dims['n_static'],
        'n_observed': dims['n_observed'],
        'n_known': dims['n_known'],
        'n_targets': dims['n_targets'],
        'hidden_dim': 64,             # Smaller for speed
        'n_heads': 4,
        'dropout': 0.1,
        'quantiles': [0.1, 0.25, 0.5, 0.75, 0.9],
        'lookback_window': dims['lookback'],
        'max_horizon': dims['max_horizon'],
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'batch_size': 512,            # Larger batch for speed
        'max_epochs': 30,
        'patience': 6,
        'grad_clip': 1.0,
    }
