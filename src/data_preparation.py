"""
Data preparation pipeline for MMM modeling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer

try:
    from .utils import (
        set_random_seed, adstock_transform, saturation_transform,
        create_weekly_seasonality, create_trend, check_stationarity,
        detect_outliers_iqr, create_interaction_features
    )
except ImportError:
    from utils import (
        set_random_seed, adstock_transform, saturation_transform,
        create_weekly_seasonality, create_trend, check_stationarity,
        detect_outliers_iqr, create_interaction_features
    )

warnings.filterwarnings('ignore')


class DataPreparator:
    """
    Comprehensive data preparation pipeline for MMM modeling
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        set_random_seed(random_seed)
        
        # Initialize transformers
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.imputer = SimpleImputer(strategy='median')
        
        # Store transformation parameters
        self.adstock_params = {}
        self.saturation_params = {}
        self.feature_names = []
        
    def generate_synthetic_data(self, n_weeks: int = 104) -> pd.DataFrame:
        """
        Generate realistic synthetic MMM dataset
        
        Args:
            n_weeks: Number of weeks of data (default 104 = 2 years)
            
        Returns:
            Synthetic MMM dataset
        """
        print(f"Generating {n_weeks} weeks of synthetic MMM data...")
        
        # Create date range
        start_date = datetime(2022, 1, 1)
        dates = [start_date + timedelta(weeks=i) for i in range(n_weeks)]
        
        # Base components
        trend = create_trend(n_weeks, trend_strength=0.015)
        seasonal = create_weekly_seasonality(n_weeks, amplitude=0.2)
        
        # Media spend variables (with realistic patterns)
        media_vars = {}
        
        # Google (mediator) - influenced by social channels
        google_base = np.random.exponential(5000, n_weeks)
        google_trend = trend * 1000
        google_seasonal = seasonal * 2000
        media_vars['google_spend'] = np.maximum(0, google_base + google_trend + google_seasonal)
        
        # Social channels (influencing Google)
        facebook_base = np.random.exponential(3000, n_weeks)
        facebook_trend = trend * 800
        facebook_seasonal = seasonal * 1500
        media_vars['facebook_spend'] = np.maximum(0, facebook_base + facebook_trend + facebook_seasonal)
        
        tiktok_base = np.random.exponential(2000, n_weeks)
        tiktok_trend = trend * 1200  # Growing faster
        tiktok_seasonal = seasonal * 1000
        media_vars['tiktok_spend'] = np.maximum(0, tiktok_base + tiktok_trend + tiktok_seasonal)
        
        snapchat_base = np.random.exponential(1500, n_weeks)
        snapchat_trend = trend * 600
        snapchat_seasonal = seasonal * 800
        media_vars['snapchat_spend'] = np.maximum(0, snapchat_base + snapchat_trend + snapchat_seasonal)
        
        # Direct response channels
        email_base = np.random.poisson(50, n_weeks)
        email_trend = trend * 10
        media_vars['email_volume'] = np.maximum(0, email_base + email_trend)
        
        sms_base = np.random.poisson(30, n_weeks)
        sms_trend = trend * 5
        media_vars['sms_volume'] = np.maximum(0, sms_base + sms_trend)
        
        # Business variables
        # Price with some seasonality and trend
        base_price = 100
        price_trend = trend * 2
        price_seasonal = seasonal * 5
        price_noise = np.random.normal(0, 3, n_weeks)
        media_vars['average_price'] = np.maximum(50, base_price + price_trend + price_seasonal + price_noise)
        
        # Promotions (binary with some seasonality)
        promo_prob = 0.3 + 0.2 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)  # Annual seasonality
        media_vars['promotions'] = np.random.binomial(1, promo_prob, n_weeks)
        
        # Followers (growing over time)
        followers_base = 100000
        followers_trend = trend * 5000
        followers_noise = np.random.normal(0, 2000, n_weeks)
        media_vars['followers'] = np.maximum(50000, followers_base + followers_trend + followers_noise)
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'week': np.arange(1, n_weeks + 1),
            **media_vars
        })
        
        # Add mediation effect: social channels influence Google spend
        mediation_effect = (
            0.3 * df['facebook_spend'] + 
            0.4 * df['tiktok_spend'] + 
            0.2 * df['snapchat_spend']
        ) * 0.1  # 10% of social spend influences Google
        
        df['google_spend'] += mediation_effect
        
        # Generate revenue with realistic relationships
        revenue = self._generate_revenue(df)
        df['revenue'] = revenue
        
        # Add some zero-spend periods (realistic for media data)
        zero_periods = np.random.choice(n_weeks, size=int(0.05 * n_weeks), replace=False)
        for period in zero_periods:
            channel = np.random.choice(['facebook_spend', 'tiktok_spend', 'snapchat_spend'])
            df.loc[period, channel] = 0
        
        print(f"Generated dataset with {len(df)} weeks of data")
        print(f"Columns: {list(df.columns)}")
        
        return df
    
    def _generate_revenue(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate revenue based on realistic MMM relationships
        """
        n_weeks = len(df)
        
        # Base revenue components
        base_revenue = 50000
        
        # Media effects (with adstock and saturation)
        google_effect = self._calculate_media_effect(
            df['google_spend'], decay=0.6, saturation_point=0.7, shape=1.2
        ) * 8  # High ROI for Google
        
        facebook_effect = self._calculate_media_effect(
            df['facebook_spend'], decay=0.4, saturation_point=0.5, shape=0.8
        ) * 5
        
        tiktok_effect = self._calculate_media_effect(
            df['tiktok_spend'], decay=0.3, saturation_point=0.4, shape=0.6
        ) * 6
        
        snapchat_effect = self._calculate_media_effect(
            df['snapchat_spend'], decay=0.5, saturation_point=0.6, shape=0.9
        ) * 4
        
        # Direct response effects
        email_effect = df['email_volume'] * 50  # $50 per email
        sms_effect = df['sms_volume'] * 80  # $80 per SMS
        
        # Price elasticity (negative relationship)
        price_elasticity = -0.3
        price_effect = price_elasticity * (df['average_price'] - df['average_price'].mean()) / df['average_price'].mean()
        
        # Promotion lift
        promo_lift = df['promotions'] * 0.15  # 15% lift during promotions
        
        # Follower effect (brand awareness)
        follower_effect = (df['followers'] - df['followers'].mean()) / df['followers'].mean() * 0.1
        
        # Trend and seasonality
        trend_component = create_trend(n_weeks, trend_strength=0.01) * 10000
        seasonal_component = create_weekly_seasonality(n_weeks, amplitude=0.1) * 5000
        
        # Combine all effects
        revenue = (
            base_revenue +
            google_effect +
            facebook_effect +
            tiktok_effect +
            snapchat_effect +
            email_effect +
            sms_effect +
            price_effect * 10000 +
            promo_lift * 10000 +
            follower_effect * 10000 +
            trend_component +
            seasonal_component
        )
        
        # Add noise
        noise = np.random.normal(0, 5000, n_weeks)
        revenue += noise
        
        # Ensure positive revenue
        revenue = np.maximum(10000, revenue)
        
        return revenue
    
    def _calculate_media_effect(self, spend: pd.Series, decay: float, 
                              saturation_point: float, shape: float) -> np.ndarray:
        """Calculate media effect with adstock and saturation"""
        # Apply adstock transformation
        adstocked = adstock_transform(spend.values, decay=decay)
        
        # Apply saturation transformation
        saturated = saturation_transform(adstocked, saturation_point, shape)
        
        return saturated
    
    def prepare_data(self, data: pd.DataFrame, 
                    apply_transformations: bool = True) -> pd.DataFrame:
        """
        Prepare data for modeling
        
        Args:
            data: Raw MMM data
            apply_transformations: Whether to apply adstock/saturation
            
        Returns:
            Prepared data ready for modeling
        """
        print("Preparing data for modeling...")
        
        df = data.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Final check for any remaining NaN values
        if df.isnull().any().any():
            print("Warning: NaN values detected after preprocessing. Filling with 0.")
            df = df.fillna(0)
        
        # Detect and handle outliers
        df = self._handle_outliers(df)
        
        # Create additional features
        df = self._create_features(df)
        
        if apply_transformations:
            # Apply media transformations
            df = self._apply_media_transformations(df)
            
            # Create interaction features
            interaction_pairs = [
                ('google_spend', 'average_price'),
                ('facebook_spend', 'promotions'),
                ('tiktok_spend', 'promotions'),
                ('email_volume', 'sms_volume')
            ]
            df = create_interaction_features(df, interaction_pairs)
        
        # Store feature names
        self.feature_names = [col for col in df.columns 
                            if col not in ['date', 'week', 'revenue']]
        
        print(f"Data preparation complete. Features: {len(self.feature_names)}")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        print("Handling missing values...")
        
        # For media spend, fill with 0 (no spend)
        media_cols = [col for col in df.columns if 'spend' in col.lower()]
        for col in media_cols:
            df[col] = df[col].fillna(0)
        
        # For other numeric columns, use median imputation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in media_cols and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in the dataset"""
        print("Handling outliers...")
        
        # Don't treat media spend outliers as outliers (they might be real campaigns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_media_cols = [col for col in numeric_cols if 'spend' not in col.lower()]
        
        for col in non_media_cols:
            outliers = detect_outliers_iqr(df[col])
            if outliers.sum() > 0:
                # Cap outliers instead of removing them
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
        
        return df
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features"""
        print("Creating additional features...")
        
        # Time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Lag features for key variables
        for lag in [1, 2, 4]:
            df[f'revenue_lag_{lag}'] = df['revenue'].shift(lag)
            df[f'google_spend_lag_{lag}'] = df['google_spend'].shift(lag)
        
        # Rolling averages
        for window in [4, 8, 12]:
            df[f'revenue_ma_{window}'] = df['revenue'].rolling(window=window).mean()
            df[f'google_spend_ma_{window}'] = df['google_spend'].rolling(window=window).mean()
        
        # Total media spend
        media_cols = [col for col in df.columns if 'spend' in col.lower()]
        df['total_media_spend'] = df[media_cols].sum(axis=1)
        
        # Media mix ratios
        for col in media_cols:
            df[f'{col}_ratio'] = df[col] / (df['total_media_spend'] + 1e-8)
        
        return df
    
    def _apply_media_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply adstock and saturation transformations to media variables"""
        print("Applying media transformations...")
        
        media_cols = [col for col in df.columns if 'spend' in col.lower()]
        
        # Define transformation parameters for each channel
        transformation_params = {
            'google_spend': {'decay': 0.6, 'saturation_point': 0.7, 'shape': 1.2},
            'facebook_spend': {'decay': 0.4, 'saturation_point': 0.5, 'shape': 0.8},
            'tiktok_spend': {'decay': 0.3, 'saturation_point': 0.4, 'shape': 0.6},
            'snapchat_spend': {'decay': 0.5, 'saturation_point': 0.6, 'shape': 0.9}
        }
        
        for col in media_cols:
            if col in transformation_params:
                params = transformation_params[col]
                
                # Apply adstock transformation
                adstocked = adstock_transform(df[col].values, 
                                            decay=params['decay'])
                df[f'{col}_adstock'] = adstocked
                
                # Apply saturation transformation
                saturated = saturation_transform(adstocked,
                                               saturation_point=params['saturation_point'],
                                               shape=params['shape'])
                df[f'{col}_saturated'] = saturated
                
                # Store parameters
                self.adstock_params[col] = params['decay']
                self.saturation_params[col] = {
                    'saturation_point': params['saturation_point'],
                    'shape': params['shape']
                }
        
        return df
    
    def scale_features(self, df: pd.DataFrame, 
                      target_col: str = 'revenue') -> Tuple[pd.DataFrame, Dict]:
        """
        Scale features for modeling
        
        Args:
            df: Input dataframe
            target_col: Target variable column name
            
        Returns:
            Scaled dataframe and scaling parameters
        """
        print("Scaling features...")
        
        # Separate features and target
        feature_cols = [col for col in df.columns 
                       if col not in ['date', 'week', target_col]]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Fit scaler on training data
        X_scaled = self.scaler.fit_transform(X)
        
        # Create scaled dataframe
        df_scaled = df.copy()
        df_scaled[feature_cols] = X_scaled
        
        # Store scaling parameters
        scaling_params = {
            'feature_means': self.scaler.center_,
            'feature_scales': self.scaler.scale_,
            'feature_names': feature_cols
        }
        
        return df_scaled, scaling_params
    
    def get_feature_importance_data(self) -> Dict:
        """Get feature importance and transformation data"""
        return {
            'adstock_params': self.adstock_params,
            'saturation_params': self.saturation_params,
            'feature_names': self.feature_names
        }
