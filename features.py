"""
Feature engineering pipeline for equity prediction.

This module builds the feature panel from the complete dataset by:
1. Creating industry dummies (SIC2 codes)
2. Winsorizing characteristics (1st and 99th percentiles by month)
3. Scaling characteristics to [-1, 1] (QuantileTransformer by month)
4. Creating macro interactions (characteristics × macro variables)
5. Building the final feature panel

Supports three modes via DataRegimeConfig:
- "full": All 94 characteristics, all 8 macro interactions
- "coding": All characteristics, but only top N macro interactions (based on coding_keep_macro_count)
- "coding_reduced": Only 15 candidate characteristics, only "d/p" macro interactions
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from typing import List, Tuple, Optional

from config import DataRegimeConfig


def build_feature_panel(
    df: pd.DataFrame,
    regime_config: DataRegimeConfig,
    include_macro_interactions: bool = True,
    reduced_macro_vars: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build feature panel from complete dataset.
    
    Args:
        df: Complete dataset with characteristics and macro variables
        regime_config: DataRegimeConfig with mode ("full", "coding", or "coding_reduced")
        include_macro_interactions: Whether to create macro interactions
        reduced_macro_vars: For coding_reduced mode, which macro vars to use (e.g., ["d/p"])
    
    Returns:
        feature_panel: DataFrame with features (permno, date, excess_ret_lead, features...)
        feature_cols: List of feature column names (excludes permno, date, excess_ret_lead)
    """
    df = df.copy()
    
    # Identify column types
    id_cols = ["permno", "date"]
    target_col = "excess_ret_lead"
    
    # Macro columns (from Goyal & Welch)
    macro_cols = ["d/p", "e/p", "b/m", "ntis", "tbl", "tms", "dfy", "svar"]
    
    # Characteristic columns (all other numeric columns except id, target, macro)
    exclude_cols = id_cols + [target_col] + macro_cols
    all_char_cols = [c for c in df.columns if c not in exclude_cols]
    
    # ------------------------------------------------------------------
    # Apply coding_reduced mode: filter characteristics and macro vars
    # ------------------------------------------------------------------
    if regime_config.mode == "coding_reduced":
        # Get candidate characteristic lists
        candidate_cols = (
            regime_config.monthly_candidate_cols +
            regime_config.quarterly_candidate_cols +
            regime_config.annual_candidate_cols
        )
        
        # Filter characteristics to only candidate cols
        all_char_cols = [c for c in all_char_cols if c in candidate_cols]
        logger.info(f"coding_reduced mode: filtered to {len(all_char_cols)} characteristics: {all_char_cols}")
        
        # Filter macro vars if specified
        if reduced_macro_vars is not None:
            macro_cols = [c for c in macro_cols if c in reduced_macro_vars]
            logger.info(f"coding_reduced mode: filtered macro vars to {macro_cols}")
    
    # ------------------------------------------------------------------
    # 1. Create industry dummies
    # ------------------------------------------------------------------
    if "sic2" in df.columns:
        # Get top industries by count
        industry_counts = df["sic2"].value_counts()
        top_industries = industry_counts.head(74).index.tolist()  # Top 74 industries
        
        # Create dummy variables
        industry_dummies = pd.get_dummies(df["sic2"], prefix="ind", columns=None)
        
        # Keep only top industries
        keep_cols = [c for c in industry_dummies.columns if c.split("_")[-1] in top_industries]
        industry_dummies = industry_dummies[keep_cols]
        
        # Add to dataframe
        df = pd.concat([df, industry_dummies], axis=1)
        
        # Add industry dummy columns to feature list
        industry_col_names = keep_cols
    else:
        industry_col_names = []
    
    # ------------------------------------------------------------------
    # 2. Winsorize characteristics (1st and 99th percentiles by month)
    # ------------------------------------------------------------------
    for char in all_char_cols:
        # Winsorize by month (cross-sectional)
        df[char] = df.groupby("yyyymm")[char].transform(
            lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99))
        )
    
    # ------------------------------------------------------------------
    # 3. Scale characteristics to [-1, 1] (QuantileTransformer by month)
    # ------------------------------------------------------------------
    for char in all_char_cols:
        # Scale by month (cross-sectional)
        def scale_to_unit(x):
            if len(x) < 10 or x.nunique() < 5:
                # Not enough data or variance - just normalize to [-1, 1]
                if x.min() == x.max():
                    return pd.Series(0, index=x.index)
                return (x - x.mean()) / (x.max() - x.min() + 1e-8) * 2
            qt = QuantileTransformer(output_range=(-1, 1), random_state=42)
            return pd.Series(qt.fit_transform(x.values.reshape(-1, 1)).flatten(), index=x.index)
        
        df[char] = df.groupby("yyyymm")[char].transform(scale_to_unit)
    
    # ------------------------------------------------------------------
    # 4. Create macro interactions
    # ------------------------------------------------------------------
    interaction_cols = []
    if include_macro_interactions:
        for char in all_char_cols:
            for macro in macro_cols:
                if macro in df.columns:
                    col_name = f"{char}_x_{macro.replace('/', '_')}"
                    df[col_name] = df[char] * df[macro]
                    interaction_cols.append(col_name)
        
        logger.info(f"Created {len(interaction_cols)} macro interactions")
    
    # ------------------------------------------------------------------
    # 5. Build final feature panel
    # ------------------------------------------------------------------
    # Feature columns = characteristics + industry dummies + interactions
    feature_cols = all_char_cols + industry_col_names + interaction_cols
    
    # Select only id, target, and feature columns
    keep_cols = id_cols + [target_col] + feature_cols
    feature_panel = df[keep_cols].copy()
    
    logger.info(f"Built feature panel with {len(feature_cols)} features:")
    logger.info(f"  - {len(all_char_cols)} characteristics")
    logger.info(f"  - {len(industry_col_names)} industry dummies")
    logger.info(f"  - {len(interaction_cols)} macro interactions")
    
    return feature_panel, feature_cols


def get_feature_stats(feature_panel: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Get statistics for each feature column.
    
    Args:
        feature_panel: Feature panel DataFrame
        feature_cols: List of feature column names
    
    Returns:
        DataFrame with mean, std, min, max, null_count for each feature
    """
    stats = []
    for col in feature_cols:
        stats.append({
            "feature": col,
            "mean": feature_panel[col].mean(),
            "std": feature_panel[col].std(),
            "min": feature_panel[col].min(),
            "max": feature_panel[col].max(),
            "null_count": feature_panel[col].isnull().sum(),
        })
    
    return pd.DataFrame(stats)