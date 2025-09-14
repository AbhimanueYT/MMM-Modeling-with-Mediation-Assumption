"""
Utility functions for MMM modeling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility"""
    np.random.seed(seed)


def adstock_transform(x: np.ndarray, decay: float = 0.5, max_lag: int = 4) -> np.ndarray:
    """
    Apply adstock transformation to media spend data
    
    Args:
        x: Input time series
        decay: Decay rate (0-1)
        max_lag: Maximum lag to consider
        
    Returns:
        Adstock transformed series
    """
    if decay <= 0 or decay >= 1:
        raise ValueError("Decay must be between 0 and 1")
    
    result = np.zeros_like(x)
    for i in range(len(x)):
        for lag in range(min(i + 1, max_lag + 1)):
            result[i] += x[i - lag] * (decay ** lag)
    
    return result


def saturation_transform(x: np.ndarray, saturation_point: float = 0.5, 
                        shape: float = 1.0) -> np.ndarray:
    """
    Apply saturation transformation using Hill function
    
    Args:
        x: Input values
        saturation_point: Point where saturation begins
        shape: Shape parameter (higher = more linear)
        
    Returns:
        Saturated values
    """
    # Handle edge case where all values are zero
    if np.all(x == 0):
        return np.zeros_like(x)
    
    # Normalize x to [0, 1] range
    x_max = x.max()
    if x_max == 0:
        return np.zeros_like(x)
    
    x_norm = x / x_max
    
    # Hill transformation
    saturated = (x_norm ** shape) / (x_norm ** shape + saturation_point ** shape)
    
    # Ensure no NaN values
    saturated = np.nan_to_num(saturated, nan=0.0, posinf=1.0, neginf=0.0)
    
    return saturated


def create_weekly_seasonality(n_weeks: int, amplitude: float = 0.3) -> np.ndarray:
    """
    Create weekly seasonality pattern
    
    Args:
        n_weeks: Number of weeks
        amplitude: Amplitude of seasonality
        
    Returns:
        Seasonal component
    """
    weeks = np.arange(n_weeks)
    # Weekly pattern: higher on weekends, lower mid-week
    seasonal = amplitude * np.sin(2 * np.pi * weeks / 7) + \
               0.1 * amplitude * np.sin(4 * np.pi * weeks / 7)
    return seasonal


def create_trend(n_weeks: int, trend_strength: float = 0.02) -> np.ndarray:
    """
    Create linear trend component
    
    Args:
        n_weeks: Number of weeks
        trend_strength: Strength of trend
        
    Returns:
        Trend component
    """
    weeks = np.arange(n_weeks)
    return trend_strength * weeks


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Square Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def time_series_split(data: pd.DataFrame, n_splits: int = 5, 
                     test_size: float = 0.2) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create time series cross-validation splits
    
    Args:
        data: Time series data
        n_splits: Number of splits
        test_size: Proportion of data for testing
        
    Returns:
        List of (train_indices, test_indices) tuples
    """
    n_samples = len(data)
    test_samples = int(n_samples * test_size)
    
    splits = []
    for i in range(n_splits):
        # Calculate split point
        split_point = n_samples - test_samples - (n_splits - 1 - i) * (test_samples // n_splits)
        
        train_indices = np.arange(split_point)
        test_indices = np.arange(split_point, min(split_point + test_samples, n_samples))
        
        splits.append((train_indices, test_indices))
    
    return splits


def check_stationarity(series: pd.Series, significance_level: float = 0.05) -> Dict:
    """
    Check stationarity using Augmented Dickey-Fuller test
    
    Args:
        series: Time series to test
        significance_level: Significance level for test
        
    Returns:
        Dictionary with test results
    """
    from statsmodels.tsa.stattools import adfuller
    
    result = adfuller(series.dropna())
    
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] < significance_level
    }


def detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """
    Detect outliers using IQR method
    
    Args:
        series: Input series
        factor: IQR factor for outlier detection
        
    Returns:
        Boolean series indicating outliers
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    return (series < lower_bound) | (series > upper_bound)


def create_interaction_features(df: pd.DataFrame, 
                              feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Create interaction features between specified pairs
    
    Args:
        df: Input dataframe
        feature_pairs: List of (feature1, feature2) tuples
        
    Returns:
        Dataframe with interaction features added
    """
    df_interact = df.copy()
    
    for feat1, feat2 in feature_pairs:
        if feat1 in df.columns and feat2 in df.columns:
            interaction_name = f"{feat1}_x_{feat2}"
            df_interact[interaction_name] = df[feat1] * df[feat2]
    
    return df_interact
