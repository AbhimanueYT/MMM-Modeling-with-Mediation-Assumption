"""
Mediation-aware MMM model implementation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import joblib

try:
    from .utils import (
        set_random_seed, time_series_split, calculate_mape, 
        calculate_rmse, calculate_r2
    )
except ImportError:
    from utils import (
        set_random_seed, time_series_split, calculate_mape, 
        calculate_rmse, calculate_r2
    )

warnings.filterwarnings('ignore')


class MediationMMM:
    """
    Mediation-aware Media Mix Model
    
    Implements a two-stage approach where Google spend mediates
    the relationship between social channels and revenue.
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        set_random_seed(random_seed)
        
        # Model components
        self.stage1_model = None  # Google spend = f(social channels)
        self.stage2_model = None  # Revenue = f(Google spend, other variables)
        
        # Model parameters
        self.stage1_params = {}
        self.stage2_params = {}
        
        # Results storage
        self.results = {}
        self.feature_importance = {}
        
    def fit(self, data: pd.DataFrame, 
            target_col: str = 'revenue',
            google_col: str = 'google_spend',
            social_cols: List[str] = None,
            direct_cols: List[str] = None) -> Dict[str, Any]:
        """
        Fit the two-stage mediation model
        
        Args:
            data: Prepared MMM data
            target_col: Target variable column
            google_col: Google spend column (mediator)
            social_cols: Social media columns (predictors for stage 1)
            direct_cols: Direct response columns (predictors for stage 2)
            
        Returns:
            Model results and diagnostics
        """
        print("Fitting mediation-aware MMM model...")
        
        # Set default columns if not provided
        if social_cols is None:
            social_cols = [col for col in data.columns 
                          if any(social in col.lower() for social in ['facebook', 'tiktok', 'snapchat'])]
        
        if direct_cols is None:
            direct_cols = [col for col in data.columns 
                          if any(direct in col.lower() for direct in ['email', 'sms', 'price', 'promo', 'follower'])]
        
        # Prepare data for modeling
        X_social = data[social_cols].values
        y_google = data[google_col].values
        
        # Stage 2: Include Google spend and other direct variables
        stage2_features = [google_col] + direct_cols
        X_stage2 = data[stage2_features].values
        y_revenue = data[target_col].values
        
        # Fit Stage 1: Google spend as function of social channels
        print("Fitting Stage 1: Social channels -> Google spend")
        self.stage1_model = self._fit_stage1_model(X_social, y_google, social_cols)
        
        # Fit Stage 2: Revenue as function of Google spend and direct variables
        print("Fitting Stage 2: Google spend + direct variables -> Revenue")
        self.stage2_model = self._fit_stage2_model(X_stage2, y_revenue, stage2_features)
        
        # Calculate predictions and diagnostics
        results = self._calculate_results(data, social_cols, stage2_features, target_col)
        
        # Store results
        self.results = results
        
        print("Model fitting complete!")
        return results
    
    def _fit_stage1_model(self, X: np.ndarray, y: np.ndarray, 
                         feature_names: List[str]) -> Any:
        """Fit Stage 1 model: Social channels -> Google spend"""
        
        # Use ElasticNet with cross-validation for regularization
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],  # Mix of L1 and L2 regularization
            alphas=np.logspace(-4, 1, 20),  # Range of regularization strengths
            cv=TimeSeriesSplit(n_splits=5),  # Time series cross-validation
            random_state=self.random_seed,
            max_iter=2000
        )
        
        model.fit(X, y)
        
        # Store model parameters
        self.stage1_params = {
            'alpha': model.alpha_,
            'l1_ratio': model.l1_ratio_,
            'feature_names': feature_names,
            'coefficients': model.coef_,
            'intercept': model.intercept_
        }
        
        return model
    
    def _fit_stage2_model(self, X: np.ndarray, y: np.ndarray, 
                         feature_names: List[str]) -> Any:
        """Fit Stage 2 model: Google spend + direct variables -> Revenue"""
        
        # Use ElasticNet with cross-validation
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            alphas=np.logspace(-4, 1, 20),
            cv=TimeSeriesSplit(n_splits=5),
            random_state=self.random_seed,
            max_iter=2000
        )
        
        model.fit(X, y)
        
        # Store model parameters
        self.stage2_params = {
            'alpha': model.alpha_,
            'l1_ratio': model.l1_ratio_,
            'feature_names': feature_names,
            'coefficients': model.coef_,
            'intercept': model.intercept_
        }
        
        return model
    
    def _calculate_results(self, data: pd.DataFrame, 
                          social_cols: List[str], 
                          stage2_features: List[str],
                          target_col: str) -> Dict[str, Any]:
        """Calculate model results and diagnostics"""
        
        # Get predictions
        X_social = data[social_cols].values
        X_stage2 = data[stage2_features].values
        y_google = data['google_spend'].values
        y_revenue = data[target_col].values
        
        # Stage 1 predictions
        google_pred = self.stage1_model.predict(X_social)
        
        # Stage 2 predictions
        revenue_pred = self.stage2_model.predict(X_stage2)
        
        # Calculate metrics
        stage1_metrics = {
            'r2': calculate_r2(y_google, google_pred),
            'rmse': calculate_rmse(y_google, google_pred),
            'mape': calculate_mape(y_google, google_pred)
        }
        
        stage2_metrics = {
            'r2': calculate_r2(y_revenue, revenue_pred),
            'rmse': calculate_rmse(y_revenue, revenue_pred),
            'mape': calculate_mape(y_revenue, revenue_pred)
        }
        
        # Calculate mediation effects
        mediation_effects = self._calculate_mediation_effects(
            data, social_cols, stage2_features
        )
        
        # Feature importance
        feature_importance = self._calculate_feature_importance()
        
        return {
            'stage1_metrics': stage1_metrics,
            'stage2_metrics': stage2_metrics,
            'google_predictions': google_pred,
            'revenue_predictions': revenue_pred,
            'mediation_effects': mediation_effects,
            'feature_importance': feature_importance,
            'model_params': {
                'stage1': self.stage1_params,
                'stage2': self.stage2_params
            }
        }
    
    def _calculate_mediation_effects(self, data: pd.DataFrame,
                                   social_cols: List[str],
                                   stage2_features: List[str]) -> Dict[str, float]:
        """Calculate mediation effects for each social channel"""
        
        mediation_effects = {}
        
        # Get Google spend coefficient from Stage 2
        google_idx = stage2_features.index('google_spend')
        google_coef = self.stage2_params['coefficients'][google_idx]
        
        # Calculate total effect for each social channel
        for i, social_col in enumerate(social_cols):
            # Direct effect (if social channel is in stage 2)
            direct_effect = 0
            if social_col in stage2_features:
                social_idx = stage2_features.index(social_col)
                direct_effect = self.stage2_params['coefficients'][social_idx]
            
            # Indirect effect through Google (mediation)
            social_coef_stage1 = self.stage1_params['coefficients'][i]
            indirect_effect = social_coef_stage1 * google_coef
            
            # Total effect
            total_effect = direct_effect + indirect_effect
            
            mediation_effects[social_col] = {
                'direct_effect': direct_effect,
                'indirect_effect': indirect_effect,
                'total_effect': total_effect,
                'mediation_ratio': indirect_effect / (total_effect + 1e-8)
            }
        
        return mediation_effects
    
    def _calculate_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Calculate feature importance for both stages"""
        
        importance = {
            'stage1': {},
            'stage2': {}
        }
        
        # Stage 1 importance (absolute coefficients)
        for i, feature in enumerate(self.stage1_params['feature_names']):
            importance['stage1'][feature] = abs(self.stage1_params['coefficients'][i])
        
        # Stage 2 importance (absolute coefficients)
        for i, feature in enumerate(self.stage2_params['feature_names']):
            importance['stage2'][feature] = abs(self.stage2_params['coefficients'][i])
        
        return importance
    
    def predict(self, data: pd.DataFrame, 
                social_cols: List[str] = None,
                stage2_features: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Make predictions using the fitted model
        
        Args:
            data: Input data
            social_cols: Social media columns
            stage2_features: Stage 2 feature columns
            
        Returns:
            Dictionary with predictions
        """
        if self.stage1_model is None or self.stage2_model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Use stored feature names if not provided
        if social_cols is None:
            social_cols = self.stage1_params['feature_names']
        if stage2_features is None:
            stage2_features = self.stage2_params['feature_names']
        
        # Check for NaN values in input data and handle them
        data_copy = data.copy()
        
        # Check which social columns actually exist in the data
        existing_social_cols = [col for col in social_cols if col in data_copy.columns]
        missing_social_cols = [col for col in social_cols if col not in data_copy.columns]
        
        if missing_social_cols:
            print(f"Warning: Missing social columns: {missing_social_cols}")
        
        if existing_social_cols and data_copy[existing_social_cols].isnull().any().any():
            print("Warning: NaN values detected in social features. Filling with 0.")
            for col in existing_social_cols:
                data_copy[col] = data_copy[col].fillna(0)
        
        # Check which stage2 columns actually exist in the data
        existing_stage2_cols = [col for col in stage2_features if col in data_copy.columns]
        missing_stage2_cols = [col for col in stage2_features if col not in data_copy.columns]
        
        if missing_stage2_cols:
            print(f"Warning: Missing stage2 columns: {missing_stage2_cols}")
        
        if existing_stage2_cols and data_copy[existing_stage2_cols].isnull().any().any():
            print("Warning: NaN values detected in stage2 features. Filling with 0.")
            for col in existing_stage2_cols:
                data_copy[col] = data_copy[col].fillna(0)
        
        # Stage 1 predictions - use the same features that were used during training
        if hasattr(self, 'stage1_params') and 'feature_names' in self.stage1_params:
            stage1_training_features = self.stage1_params['feature_names']
            available_stage1_features = [col for col in stage1_training_features if col in data_copy.columns]
            if len(available_stage1_features) != len(stage1_training_features):
                print(f"Warning: Only {len(available_stage1_features)}/{len(stage1_training_features)} stage1 training features available")
            X_social = data_copy[available_stage1_features].values
        else:
            X_social = data_copy[existing_social_cols].values
        google_pred = self.stage1_model.predict(X_social)
        
        # Stage 2 predictions - use the same features that were used during training
        if hasattr(self, 'stage2_params') and 'feature_names' in self.stage2_params:
            stage2_training_features = self.stage2_params['feature_names']
            available_stage2_features = [col for col in stage2_training_features if col in data_copy.columns]
            if len(available_stage2_features) != len(stage2_training_features):
                print(f"Warning: Only {len(available_stage2_features)}/{len(stage2_training_features)} stage2 training features available")
            X_stage2 = data_copy[available_stage2_features].values
        else:
            X_stage2 = data_copy[existing_stage2_cols].values
        revenue_pred = self.stage2_model.predict(X_stage2)
        
        return {
            'google_predictions': google_pred,
            'revenue_predictions': revenue_pred
        }
    
    def cross_validate(self, data: pd.DataFrame, 
                      n_splits: int = 5,
                      target_col: str = 'revenue') -> Dict[str, List[float]]:
        """
        Perform time series cross-validation
        
        Args:
            data: Input data
            n_splits: Number of CV splits
            target_col: Target variable column
            
        Returns:
            CV results
        """
        print(f"Performing {n_splits}-fold time series cross-validation...")
        
        # Create time series splits
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_results = {
            'stage1_r2': [],
            'stage1_rmse': [],
            'stage2_r2': [],
            'stage2_rmse': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(data)):
            print(f"  Fold {fold + 1}/{n_splits}")
            
            # Split data
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # Fit model on training data
            temp_model = MediationMMM(random_seed=self.random_seed)
            temp_results = temp_model.fit(train_data, target_col=target_col)
            
            # Make predictions on test data
            predictions = temp_model.predict(test_data)
            
            # Calculate metrics
            y_google_test = test_data['google_spend'].values
            y_revenue_test = test_data[target_col].values
            
            stage1_r2 = calculate_r2(y_google_test, predictions['google_predictions'])
            stage1_rmse = calculate_rmse(y_google_test, predictions['google_predictions'])
            
            stage2_r2 = calculate_r2(y_revenue_test, predictions['revenue_predictions'])
            stage2_rmse = calculate_rmse(y_revenue_test, predictions['revenue_predictions'])
            
            cv_results['stage1_r2'].append(stage1_r2)
            cv_results['stage1_rmse'].append(stage1_rmse)
            cv_results['stage2_r2'].append(stage2_r2)
            cv_results['stage2_rmse'].append(stage2_rmse)
        
        # Calculate summary statistics
        cv_summary = {}
        for metric, values in cv_results.items():
            cv_summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
        
        print("Cross-validation complete!")
        return cv_summary
    
    def save_model(self, filepath: str) -> None:
        """Save the fitted model"""
        model_data = {
            'stage1_model': self.stage1_model,
            'stage2_model': self.stage2_model,
            'stage1_params': self.stage1_params,
            'stage2_params': self.stage2_params,
            'results': self.results,
            'random_seed': self.random_seed
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a fitted model"""
        model_data = joblib.load(filepath)
        self.stage1_model = model_data['stage1_model']
        self.stage2_model = model_data['stage2_model']
        self.stage1_params = model_data['stage1_params']
        self.stage2_params = model_data['stage2_params']
        self.results = model_data['results']
        self.random_seed = model_data['random_seed']
        print(f"Model loaded from {filepath}")
    
    def get_attribution(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate attribution for each channel
        
        Args:
            data: Input data
            
        Returns:
            Attribution dataframe
        """
        if self.stage1_model is None or self.stage2_model is None:
            raise ValueError("Model must be fitted before calculating attribution")
        
        # Get predictions
        predictions = self.predict(data)
        
        # Calculate attribution
        attribution_data = []
        
        for i, row in data.iterrows():
            # Base revenue (intercept)
            base_revenue = self.stage2_params['intercept']
            
            # Google attribution (through mediation)
            google_idx = self.stage2_params['feature_names'].index('google_spend')
            google_coef = self.stage2_params['coefficients'][google_idx]
            google_attribution = row['google_spend'] * google_coef
            
            # Social channel attributions (through Google)
            social_attributions = {}
            for j, social_col in enumerate(self.stage1_params['feature_names']):
                social_coef = self.stage1_params['coefficients'][j]
                social_contribution = row[social_col] * social_coef
                social_attribution = social_contribution * google_coef
                social_attributions[social_col] = social_attribution
            
            # Direct channel attributions
            direct_attributions = {}
            for j, feature in enumerate(self.stage2_params['feature_names']):
                if feature != 'google_spend':
                    coef = self.stage2_params['coefficients'][j]
                    direct_attributions[feature] = row[feature] * coef
            
            # Combine all attributions
            total_attribution = (
                base_revenue + 
                google_attribution + 
                sum(social_attributions.values()) + 
                sum(direct_attributions.values())
            )
            
            attribution_data.append({
                'date': row['date'],
                'base_revenue': base_revenue,
                'google_attribution': google_attribution,
                **social_attributions,
                **direct_attributions,
                'total_attribution': total_attribution,
                'actual_revenue': row['revenue'],
                'predicted_revenue': predictions['revenue_predictions'][i]
            })
        
        return pd.DataFrame(attribution_data)
