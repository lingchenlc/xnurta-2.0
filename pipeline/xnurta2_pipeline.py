"""
Xnurta 2.0 — Unified Pipeline
================================
Integrates AdTFT predictions + Semantic Engine results into a single
optimization recommendation system.

Usage:
    pipeline = Xnurta2Pipeline()
    report = pipeline.generate_report(campaign_id=...)
    recommendations = pipeline.get_recommendations(country='美国', top_n=50)
"""

import json
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "dashboard").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "models" / "ad_tft"))

from model import AdTFT, get_default_config


class Xnurta2Pipeline:
    """Unified pipeline combining all Xnurta 2.0 models."""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.data_dir = PROJECT_ROOT / "data" / "processed"
        self.feature_dir = PROJECT_ROOT / "data" / "features"
        self.model_dir = PROJECT_ROOT / "models" / "ad_tft" / "trained"

        self._load_adtft()
        self._load_semantic_data()
        self._load_campaign_data()

    # ─── Model Loading ─────────────────────────────────────────────

    def _load_adtft(self):
        """Load trained AdTFT model."""
        checkpoint_path = self.model_dir / "best_model.pt"
        if not checkpoint_path.exists():
            print("⚠️  AdTFT model not found, skipping")
            self.adtft_model = None
            self.adtft_config = None
            self.training_results = None
            return

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.adtft_config = checkpoint['config']
        self.adtft_model = AdTFT(self.adtft_config)
        self.adtft_model.load_state_dict(checkpoint['model_state_dict'])
        self.adtft_model.eval()
        self.norm_stats = checkpoint.get('norm_stats', {})

        # Load training results
        results_path = self.model_dir / "training_results.json"
        if results_path.exists():
            with open(results_path) as f:
                self.training_results = json.load(f)
        else:
            self.training_results = None

        # Load feature metadata
        meta_path = self.feature_dir / "feature_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.feature_metadata = json.load(f)

        print(f"✅ AdTFT loaded (epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f})")

    def _load_semantic_data(self):
        """Load semantic engine outputs."""
        # Semantic clusters
        path = self.data_dir / "semantic_clusters.csv"
        if path.exists():
            self.semantic_clusters = pd.read_csv(path)
            print(f"✅ Semantic clusters: {len(self.semantic_clusters)} clusters")
        else:
            self.semantic_clusters = pd.DataFrame()

        # Negation recommendations
        path = self.data_dir / "negation_recommendations.csv"
        if path.exists():
            self.negation_recs = pd.read_csv(path)
            print(f"✅ Negation recommendations: {len(self.negation_recs)} terms")
        else:
            self.negation_recs = pd.DataFrame()

        # Harvest recommendations
        path = self.data_dir / "harvest_recommendations.csv"
        if path.exists():
            self.harvest_recs = pd.read_csv(path)
            print(f"✅ Harvest recommendations: {len(self.harvest_recs)} terms")
        else:
            self.harvest_recs = pd.DataFrame()

        # Clustered search terms
        path = self.data_dir / "text_terms_with_semantic_clusters.csv"
        if path.exists():
            self.clustered_terms = pd.read_csv(path)
            print(f"✅ Clustered search terms: {len(self.clustered_terms)} terms")
        else:
            self.clustered_terms = pd.DataFrame()

        # TF-IDF cluster analysis
        path = self.data_dir / "cluster_analysis.csv"
        if path.exists():
            self.cluster_analysis = pd.read_csv(path)
        else:
            self.cluster_analysis = pd.DataFrame()

    def _load_campaign_data(self):
        """Load campaign performance data summary."""
        path = self.data_dir / "campaign_daily_clean.csv"
        if path.exists():
            # Load only a summary to keep memory low
            print("  Loading campaign data summary...")
            cols = ['date', 'country', 'store', 'campaign', 'impressions',
                    'clicks', 'spend', 'orders', 'sales', 'acos', 'roas']
            self.campaign_daily = pd.read_csv(path, usecols=cols, parse_dates=['date'])
            print(f"✅ Campaign data: {len(self.campaign_daily):,} rows")
        else:
            self.campaign_daily = pd.DataFrame()

    # ─── AdTFT Analysis ───────────────────────────────────────────

    def get_adtft_summary(self):
        """Get AdTFT model performance summary."""
        if self.training_results is None:
            return {}

        tm = self.training_results.get('test_metrics', {})
        return {
            'model_params': self.training_results.get('n_params', 0),
            'best_epoch': self.training_results.get('best_epoch', 0),
            'training_time_min': self.training_results.get('total_time', 0) / 60,
            'test_loss': tm.get('loss', 0),
            'overall_mae': tm.get('mae', 0),
            'overall_rmse': tm.get('rmse', 0),
            'coverage_80': tm.get('coverage_80', 0),
            'coverage_50': tm.get('coverage_50', 0),
            'per_target': {
                name: {
                    'mae': tm.get(f'{name}_mae', 0),
                    'rmse': tm.get(f'{name}_rmse', 0),
                    'mape': tm.get(f'{name}_mape', 0),
                }
                for name in [
                    'spend_1d', 'sales_1d', 'orders_1d', 'acos_1d',
                    'spend_3d', 'sales_3d', 'orders_3d', 'acos_3d',
                    'spend_7d', 'sales_7d', 'orders_7d', 'acos_7d',
                ]
            },
            'training_history': {
                'train_loss': self.training_results.get('training_history', {}).get('train_loss', []),
                'val_loss': self.training_results.get('training_history', {}).get('val_loss', []),
            }
        }

    # ─── Semantic Engine Analysis ──────────────────────────────────

    def get_semantic_summary(self):
        """Get semantic engine summary statistics."""
        summary = {}

        if not self.negation_recs.empty:
            summary['negation'] = {
                'total_candidates': len(self.negation_recs),
                'green_safe': int((self.negation_recs.get('safety_level', pd.Series()) == 'GREEN_SAFE_TO_NEGATE').sum()),
                'yellow_review': int((self.negation_recs.get('safety_level', pd.Series()) == 'YELLOW_NEEDS_REVIEW').sum()),
                'total_saveable_spend': float(self.negation_recs['total_spend'].sum()) if 'total_spend' in self.negation_recs.columns else 0,
            }

        if not self.harvest_recs.empty:
            summary['harvest'] = {
                'total_candidates': len(self.harvest_recs),
                'total_potential_sales': float(self.harvest_recs['total_sales'].sum()) if 'total_sales' in self.harvest_recs.columns else 0,
            }

        if not self.semantic_clusters.empty:
            summary['clusters'] = {
                'total_clusters': len(self.semantic_clusters),
                'total_terms_clustered': self.clustered_terms['semantic_cluster'].notna().sum() if 'semantic_cluster' in self.clustered_terms.columns else 0,
            }

        return summary

    def get_negation_recommendations(self, safety_level=None, top_n=100):
        """Get negation recommendations with optional filtering."""
        df = self.negation_recs.copy()
        if safety_level and 'safety_level' in df.columns:
            df = df[df['safety_level'] == safety_level]
        if 'spend' in df.columns:
            df = df.sort_values('spend', ascending=False)
        return df.head(top_n)

    def get_harvest_recommendations(self, top_n=100):
        """Get harvest recommendations."""
        df = self.harvest_recs.copy()
        if 'sales' in df.columns:
            df = df.sort_values('sales', ascending=False)
        return df.head(top_n)

    # ─── Campaign Performance ─────────────────────────────────────

    def get_campaign_overview(self):
        """Get high-level campaign performance metrics."""
        if self.campaign_daily.empty:
            return {}

        df = self.campaign_daily
        total = {
            'total_spend': df['spend'].sum(),
            'total_sales': df['sales'].sum(),
            'total_orders': df['orders'].sum(),
            'total_clicks': df['clicks'].sum(),
            'total_impressions': df['impressions'].sum(),
            'avg_acos': df['spend'].sum() / max(df['sales'].sum(), 1),
            'avg_roas': df['sales'].sum() / max(df['spend'].sum(), 1),
            'n_campaigns': df['campaign'].nunique(),
            'n_countries': df['country'].nunique(),
            'n_stores': df['store'].nunique(),
            'date_range': [str(df['date'].min().date()), str(df['date'].max().date())],
            'n_days': df['date'].nunique(),
        }
        return total

    def get_daily_trends(self):
        """Get daily aggregated trends."""
        if self.campaign_daily.empty:
            return pd.DataFrame()

        daily = self.campaign_daily.groupby('date').agg(
            spend=('spend', 'sum'),
            sales=('sales', 'sum'),
            orders=('orders', 'sum'),
            clicks=('clicks', 'sum'),
            impressions=('impressions', 'sum'),
            active_campaigns=('campaign', 'nunique'),
        ).reset_index()

        daily['acos'] = daily['spend'] / daily['sales'].clip(lower=0.01)
        daily['roas'] = daily['sales'] / daily['spend'].clip(lower=0.01)
        daily['ctr'] = daily['clicks'] / daily['impressions'].clip(lower=1)
        daily['cvr'] = daily['orders'] / daily['clicks'].clip(lower=1)
        daily['cpc'] = daily['spend'] / daily['clicks'].clip(lower=1)

        return daily

    def get_country_breakdown(self):
        """Get performance by country."""
        if self.campaign_daily.empty:
            return pd.DataFrame()

        country = self.campaign_daily.groupby('country').agg(
            spend=('spend', 'sum'),
            sales=('sales', 'sum'),
            orders=('orders', 'sum'),
            clicks=('clicks', 'sum'),
            impressions=('impressions', 'sum'),
            campaigns=('campaign', 'nunique'),
        ).reset_index()

        country['acos'] = country['spend'] / country['sales'].clip(lower=0.01)
        country['roas'] = country['sales'] / country['spend'].clip(lower=0.01)
        country = country.sort_values('spend', ascending=False)

        return country

    def get_top_campaigns(self, country=None, metric='spend', top_n=20):
        """Get top performing campaigns."""
        df = self.campaign_daily.copy()
        if country:
            df = df[df['country'] == country]

        campaign_agg = df.groupby(['country', 'store', 'campaign']).agg(
            spend=('spend', 'sum'),
            sales=('sales', 'sum'),
            orders=('orders', 'sum'),
            clicks=('clicks', 'sum'),
            impressions=('impressions', 'sum'),
        ).reset_index()

        campaign_agg['acos'] = campaign_agg['spend'] / campaign_agg['sales'].clip(lower=0.01)
        campaign_agg['roas'] = campaign_agg['sales'] / campaign_agg['spend'].clip(lower=0.01)

        return campaign_agg.sort_values(metric, ascending=False).head(top_n)


if __name__ == "__main__":
    print("Loading Xnurta 2.0 Pipeline...")
    pipeline = Xnurta2Pipeline()

    print("\n=== Campaign Overview ===")
    overview = pipeline.get_campaign_overview()
    for k, v in overview.items():
        print(f"  {k}: {v}")

    print("\n=== AdTFT Summary ===")
    adtft = pipeline.get_adtft_summary()
    print(f"  Test Loss: {adtft.get('test_loss', 'N/A')}")
    print(f"  Coverage 80%: {adtft.get('coverage_80', 'N/A')}%")
    print(f"  MAE: {adtft.get('overall_mae', 'N/A')}")

    print("\n=== Semantic Summary ===")
    sem = pipeline.get_semantic_summary()
    for k, v in sem.items():
        print(f"  {k}: {v}")
