"""
Diagnostic and validation framework for MMM modeling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_breuschpagan
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    from .utils import (
        calculate_mape, calculate_rmse, calculate_r2, 
        check_stationarity, detect_outliers_iqr
    )
except ImportError:
    from utils import (
        calculate_mape, calculate_rmse, calculate_r2, 
        check_stationarity, detect_outliers_iqr
    )

warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class MMMDiagnostics:
    """
    Comprehensive diagnostic framework for MMM models
    """
    
    def __init__(self, model_results: Dict[str, Any], data: pd.DataFrame):
        self.results = model_results
        self.data = data
        self.diagnostic_results = {}
        
    def run_all_diagnostics(self) -> Dict[str, Any]:
        """Run all diagnostic tests"""
        print("Running comprehensive MMM diagnostics...")
        
        diagnostics = {}
        
        # 1. Model Performance Diagnostics
        diagnostics['performance'] = self._performance_diagnostics()
        
        # 2. Residual Analysis
        diagnostics['residuals'] = self._residual_analysis()
        
        # 3. Time Series Diagnostics
        diagnostics['time_series'] = self._time_series_diagnostics()
        
        # 4. Stability Checks
        diagnostics['stability'] = self._stability_checks()
        
        # 5. Sensitivity Analysis
        diagnostics['sensitivity'] = self._sensitivity_analysis()
        
        # 6. Business Logic Validation
        diagnostics['business_logic'] = self._business_logic_validation()
        
        self.diagnostic_results = diagnostics
        return diagnostics
    
    def _performance_diagnostics(self) -> Dict[str, Any]:
        """Comprehensive model performance diagnostics"""
        print("  Running performance diagnostics...")
        
        y_true = self.data['revenue'].values
        y_pred = self.results['revenue_predictions']
        
        # Basic metrics
        metrics = {
            'r2': calculate_r2(y_true, y_pred),
            'rmse': calculate_rmse(y_true, y_pred),
            'mape': calculate_mape(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred)
        }
        
        # Directional accuracy
        directional_accuracy = np.mean(
            np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_pred[:-1])
        )
        metrics['directional_accuracy'] = directional_accuracy
        
        # Performance by time period
        n_periods = len(y_true)
        q1_end = n_periods // 4
        q2_end = n_periods // 2
        q3_end = 3 * n_periods // 4
        
        period_metrics = {}
        periods = [
            ('Q1', slice(0, q1_end)),
            ('Q2', slice(q1_end, q2_end)),
            ('Q3', slice(q2_end, q3_end)),
            ('Q4', slice(q3_end, None))
        ]
        
        for period_name, period_slice in periods:
            y_true_period = y_true[period_slice]
            y_pred_period = y_pred[period_slice]
            
            period_metrics[period_name] = {
                'r2': calculate_r2(y_true_period, y_pred_period),
                'mape': calculate_mape(y_true_period, y_pred_period),
                'rmse': calculate_rmse(y_true_period, y_pred_period)
            }
        
        return {
            'overall_metrics': metrics,
            'period_metrics': period_metrics
        }
    
    def _residual_analysis(self) -> Dict[str, Any]:
        """Comprehensive residual analysis"""
        print("  Running residual analysis...")
        
        y_true = self.data['revenue'].values
        y_pred = self.results['revenue_predictions']
        residuals = y_true - y_pred
        
        # Basic residual statistics
        residual_stats = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals),
            'jarque_bera_stat': stats.jarque_bera(residuals)[0],
            'jarque_bera_pvalue': stats.jarque_bera(residuals)[1]
        }
        
        # Normality tests
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        residual_stats['shapiro_stat'] = shapiro_stat
        residual_stats['shapiro_pvalue'] = shapiro_p
        residual_stats['is_normal'] = shapiro_p > 0.05
        
        # Heteroscedasticity test
        try:
            # Create a simple regression for heteroscedasticity test
            X_test = np.column_stack([y_pred, y_pred**2])
            bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X_test)
            residual_stats['breusch_pagan_stat'] = bp_stat
            residual_stats['breusch_pagan_pvalue'] = bp_p
            residual_stats['is_homoscedastic'] = bp_p > 0.05
        except:
            residual_stats['breusch_pagan_stat'] = None
            residual_stats['breusch_pagan_pvalue'] = None
            residual_stats['is_homoscedastic'] = None
        
        # Autocorrelation in residuals
        autocorr_lag1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        residual_stats['autocorr_lag1'] = autocorr_lag1
        
        # Outlier detection in residuals
        residual_outliers = detect_outliers_iqr(pd.Series(residuals))
        residual_stats['outlier_count'] = residual_outliers.sum()
        residual_stats['outlier_percentage'] = (residual_outliers.sum() / len(residuals)) * 100
        
        return {
            'residual_stats': residual_stats,
            'residuals': residuals,
            'outliers': residual_outliers
        }
    
    def _time_series_diagnostics(self) -> Dict[str, Any]:
        """Time series specific diagnostics"""
        print("  Running time series diagnostics...")
        
        # Stationarity tests
        revenue_stationarity = check_stationarity(self.data['revenue'])
        residuals = self.data['revenue'].values - self.results['revenue_predictions']
        residual_stationarity = check_stationarity(pd.Series(residuals))
        
        # Trend analysis
        revenue_trend = np.polyfit(range(len(self.data)), self.data['revenue'], 1)[0]
        residual_trend = np.polyfit(range(len(residuals)), residuals, 1)[0]
        
        # Seasonality analysis
        revenue_seasonal = self._detect_seasonality(self.data['revenue'])
        residual_seasonal = self._detect_seasonality(pd.Series(residuals))
        
        return {
            'revenue_stationarity': revenue_stationarity,
            'residual_stationarity': residual_stationarity,
            'revenue_trend': revenue_trend,
            'residual_trend': residual_trend,
            'revenue_seasonality': revenue_seasonal,
            'residual_seasonality': residual_seasonal
        }
    
    def _detect_seasonality(self, series: pd.Series, period: int = 52) -> Dict[str, Any]:
        """Detect seasonality in time series"""
        if len(series) < 2 * period:
            return {'detected': False, 'strength': 0}
        
        # Calculate autocorrelation at seasonal lag
        autocorr = series.autocorr(lag=period)
        
        # Simple seasonality strength measure
        seasonal_strength = abs(autocorr) if not np.isnan(autocorr) else 0
        
        return {
            'detected': seasonal_strength > 0.3,
            'strength': seasonal_strength,
            'autocorr': autocorr
        }
    
    def _stability_checks(self) -> Dict[str, Any]:
        """Model stability checks"""
        print("  Running stability checks...")
        
        # Rolling window performance
        window_size = 26  # 6 months
        rolling_metrics = self._rolling_window_analysis(window_size)
        
        # Coefficient stability (if available)
        coefficient_stability = self._coefficient_stability_analysis()
        
        # Prediction stability
        prediction_stability = self._prediction_stability_analysis()
        
        return {
            'rolling_metrics': rolling_metrics,
            'coefficient_stability': coefficient_stability,
            'prediction_stability': prediction_stability
        }
    
    def _rolling_window_analysis(self, window_size: int) -> Dict[str, Any]:
        """Analyze model performance over rolling windows"""
        y_true = self.data['revenue'].values
        y_pred = self.results['revenue_predictions']
        
        rolling_r2 = []
        rolling_mape = []
        
        for i in range(window_size, len(y_true)):
            window_true = y_true[i-window_size:i]
            window_pred = y_pred[i-window_size:i]
            
            rolling_r2.append(calculate_r2(window_true, window_pred))
            rolling_mape.append(calculate_mape(window_true, window_pred))
        
        return {
            'r2_mean': np.mean(rolling_r2),
            'r2_std': np.std(rolling_r2),
            'r2_min': np.min(rolling_r2),
            'r2_max': np.max(rolling_r2),
            'mape_mean': np.mean(rolling_mape),
            'mape_std': np.std(rolling_mape),
            'mape_min': np.min(rolling_mape),
            'mape_max': np.max(rolling_mape),
            'r2_values': rolling_r2,
            'mape_values': rolling_mape
        }
    
    def _coefficient_stability_analysis(self) -> Dict[str, Any]:
        """Analyze coefficient stability across time"""
        # This would require refitting the model on different time windows
        # For now, return placeholder
        return {
            'analysis_available': False,
            'note': 'Coefficient stability analysis requires refitting model on rolling windows'
        }
    
    def _prediction_stability_analysis(self) -> Dict[str, Any]:
        """Analyze prediction stability"""
        y_pred = self.results['revenue_predictions']
        
        # Calculate prediction volatility
        prediction_changes = np.diff(y_pred)
        volatility = np.std(prediction_changes)
        
        # Calculate prediction range
        prediction_range = np.max(y_pred) - np.min(y_pred)
        
        return {
            'volatility': volatility,
            'range': prediction_range,
            'coefficient_of_variation': volatility / np.mean(y_pred)
        }
    
    def _sensitivity_analysis(self) -> Dict[str, Any]:
        """Sensitivity analysis for key business variables"""
        print("  Running sensitivity analysis...")
        
        sensitivity_results = {}
        
        # Price sensitivity
        price_sensitivity = self._analyze_price_sensitivity()
        sensitivity_results['price'] = price_sensitivity
        
        # Promotion sensitivity
        promo_sensitivity = self._analyze_promotion_sensitivity()
        sensitivity_results['promotions'] = promo_sensitivity
        
        # Media spend sensitivity
        media_sensitivity = self._analyze_media_sensitivity()
        sensitivity_results['media'] = media_sensitivity
        
        return sensitivity_results
    
    def _analyze_price_sensitivity(self) -> Dict[str, Any]:
        """Analyze sensitivity to price changes"""
        if 'average_price' not in self.data.columns:
            return {'analysis_available': False}
        
        # Calculate price elasticity
        price_changes = np.diff(self.data['average_price'])
        revenue_changes = np.diff(self.data['revenue'])
        
        # Remove zero price changes to avoid division by zero
        non_zero_mask = price_changes != 0
        if non_zero_mask.sum() == 0:
            return {'elasticity': 0, 'analysis_available': False}
        
        price_elasticity = np.mean(
            (revenue_changes[non_zero_mask] / self.data['revenue'].iloc[1:][non_zero_mask]) /
            (price_changes[non_zero_mask] / self.data['average_price'].iloc[1:][non_zero_mask])
        )
        
        return {
            'elasticity': price_elasticity,
            'analysis_available': True,
            'interpretation': 'Negative elasticity indicates price sensitivity'
        }
    
    def _analyze_promotion_sensitivity(self) -> Dict[str, Any]:
        """Analyze sensitivity to promotions"""
        if 'promotions' not in self.data.columns:
            return {'analysis_available': False}
        
        # Compare revenue during promotions vs non-promotions
        promo_revenue = self.data[self.data['promotions'] == 1]['revenue']
        non_promo_revenue = self.data[self.data['promotions'] == 0]['revenue']
        
        if len(promo_revenue) == 0 or len(non_promo_revenue) == 0:
            return {'analysis_available': False}
        
        promo_lift = (promo_revenue.mean() - non_promo_revenue.mean()) / non_promo_revenue.mean()
        
        # Statistical significance test
        t_stat, p_value = stats.ttest_ind(promo_revenue, non_promo_revenue)
        
        return {
            'promo_lift': promo_lift,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'analysis_available': True
        }
    
    def _analyze_media_sensitivity(self) -> Dict[str, Any]:
        """Analyze sensitivity to media spend changes"""
        media_cols = [col for col in self.data.columns if 'spend' in col.lower()]
        
        media_sensitivity = {}
        for col in media_cols:
            spend_changes = np.diff(self.data[col])
            revenue_changes = np.diff(self.data['revenue'])
            
            # Calculate correlation between spend changes and revenue changes
            correlation = np.corrcoef(spend_changes, revenue_changes)[0, 1]
            
            # Calculate ROI proxy (revenue per dollar spent)
            total_spend = self.data[col].sum()
            total_revenue = self.data['revenue'].sum()
            roi_proxy = total_revenue / (total_spend + 1e-8)
            
            media_sensitivity[col] = {
                'correlation': correlation,
                'roi_proxy': roi_proxy,
                'total_spend': total_spend,
                'revenue_contribution': correlation * total_spend
            }
        
        return media_sensitivity
    
    def _business_logic_validation(self) -> Dict[str, Any]:
        """Validate business logic and model assumptions"""
        print("  Running business logic validation...")
        
        validation_results = {}
        
        # Check for negative coefficients where they shouldn't be
        negative_coef_issues = self._check_negative_coefficients()
        validation_results['negative_coefficients'] = negative_coef_issues
        
        # Check for unrealistic effect sizes
        unrealistic_effects = self._check_unrealistic_effects()
        validation_results['unrealistic_effects'] = unrealistic_effects
        
        # Check mediation assumption
        mediation_validation = self._validate_mediation_assumption()
        validation_results['mediation'] = mediation_validation
        
        return validation_results
    
    def _check_negative_coefficients(self) -> Dict[str, Any]:
        """Check for problematic negative coefficients"""
        issues = []
        
        # Check stage 2 coefficients (revenue model)
        stage2_coefs = self.results['model_params']['stage2']['coefficients']
        stage2_features = self.results['model_params']['stage2']['feature_names']
        
        for i, (coef, feature) in enumerate(zip(stage2_coefs, stage2_features)):
            if coef < 0 and 'spend' in feature.lower():
                issues.append({
                    'feature': feature,
                    'coefficient': coef,
                    'issue': 'Negative media spend coefficient'
                })
        
        return {
            'issues_found': len(issues),
            'issues': issues
        }
    
    def _check_unrealistic_effects(self) -> Dict[str, Any]:
        """Check for unrealistic effect sizes"""
        issues = []
        
        # Check for extremely large coefficients
        stage2_coefs = self.results['model_params']['stage2']['coefficients']
        stage2_features = self.results['model_params']['stage2']['feature_names']
        
        for coef, feature in zip(stage2_coefs, stage2_features):
            if abs(coef) > 1000:  # Arbitrary threshold
                issues.append({
                    'feature': feature,
                    'coefficient': coef,
                    'issue': 'Extremely large coefficient'
                })
        
        return {
            'issues_found': len(issues),
            'issues': issues
        }
    
    def _validate_mediation_assumption(self) -> Dict[str, Any]:
        """Validate the mediation assumption"""
        mediation_effects = self.results['mediation_effects']
        
        validation = {
            'social_channels_mediate': {},
            'overall_mediation_strength': 0
        }
        
        total_mediation = 0
        for channel, effects in mediation_effects.items():
            mediation_ratio = effects['mediation_ratio']
            validation['social_channels_mediate'][channel] = {
                'mediation_ratio': mediation_ratio,
                'strongly_mediated': mediation_ratio > 0.5,
                'indirect_effect': effects['indirect_effect'],
                'total_effect': effects['total_effect']
            }
            total_mediation += abs(effects['indirect_effect'])
        
        validation['overall_mediation_strength'] = total_mediation
        
        return validation
    
    def create_diagnostic_plots(self, save_path: str = None) -> None:
        """Create comprehensive diagnostic plots"""
        print("Creating diagnostic plots...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Actual vs Predicted Revenue',
                'Residuals Over Time',
                'Residual Distribution',
                'Q-Q Plot',
                'Feature Importance (Stage 1)',
                'Feature Importance (Stage 2)'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Actual vs Predicted
        fig.add_trace(
            go.Scatter(
                x=self.data['revenue'],
                y=self.results['revenue_predictions'],
                mode='markers',
                name='Predictions',
                marker=dict(color='blue', opacity=0.6)
            ),
            row=1, col=1
        )
        
        # Add perfect prediction line
        min_val = min(self.data['revenue'].min(), self.results['revenue_predictions'].min())
        max_val = max(self.data['revenue'].max(), self.results['revenue_predictions'].max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        # 2. Residuals over time
        residuals = self.data['revenue'].values - self.results['revenue_predictions']
        fig.add_trace(
            go.Scatter(
                x=list(range(len(residuals))),
                y=residuals,
                mode='lines',
                name='Residuals',
                line=dict(color='green')
            ),
            row=1, col=2
        )
        
        # 3. Residual distribution
        fig.add_trace(
            go.Histogram(
                x=residuals,
                name='Residual Distribution',
                nbinsx=30,
                marker_color='orange'
            ),
            row=2, col=1
        )
        
        # 4. Q-Q plot
        from scipy import stats
        qq_data = stats.probplot(residuals, dist="norm")
        fig.add_trace(
            go.Scatter(
                x=qq_data[0][0],
                y=qq_data[0][1],
                mode='markers',
                name='Q-Q Plot',
                marker=dict(color='purple')
            ),
            row=2, col=2
        )
        
        # Add Q-Q line
        fig.add_trace(
            go.Scatter(
                x=qq_data[0][0],
                y=qq_data[1][1] + qq_data[1][0] * qq_data[0][0],
                mode='lines',
                name='Q-Q Line',
                line=dict(color='red')
            ),
            row=2, col=2
        )
        
        # 5. Feature importance Stage 1
        stage1_importance = self.results['feature_importance']['stage1']
        fig.add_trace(
            go.Bar(
                x=list(stage1_importance.keys()),
                y=list(stage1_importance.values()),
                name='Stage 1 Importance',
                marker_color='lightblue'
            ),
            row=3, col=1
        )
        
        # 6. Feature importance Stage 2
        stage2_importance = self.results['feature_importance']['stage2']
        fig.add_trace(
            go.Bar(
                x=list(stage2_importance.keys()),
                y=list(stage2_importance.values()),
                name='Stage 2 Importance',
                marker_color='lightcoral'
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="MMM Model Diagnostics",
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Actual Revenue", row=1, col=1)
        fig.update_yaxes(title_text="Predicted Revenue", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_yaxes(title_text="Residuals", row=1, col=2)
        fig.update_xaxes(title_text="Residual Value", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)
        fig.update_xaxes(title_text="Features", row=3, col=1)
        fig.update_yaxes(title_text="Importance", row=3, col=1)
        fig.update_xaxes(title_text="Features", row=3, col=2)
        fig.update_yaxes(title_text="Importance", row=3, col=2)
        
        if save_path:
            fig.write_html(save_path)
            print(f"Diagnostic plots saved to {save_path}")
        else:
            fig.show()
    
    def generate_diagnostic_report(self) -> str:
        """Generate a comprehensive diagnostic report"""
        if not self.diagnostic_results:
            self.run_all_diagnostics()
        
        report = []
        report.append("# MMM Model Diagnostic Report")
        report.append("=" * 50)
        report.append("")
        
        # Performance Summary
        report.append("## Model Performance Summary")
        perf = self.diagnostic_results['performance']['overall_metrics']
        report.append(f"- R²: {perf['r2']:.4f}")
        report.append(f"- RMSE: {perf['rmse']:.2f}")
        report.append(f"- MAPE: {perf['mape']:.2f}%")
        report.append(f"- Directional Accuracy: {perf['directional_accuracy']:.2f}")
        report.append("")
        
        # Residual Analysis
        report.append("## Residual Analysis")
        residual_stats = self.diagnostic_results['residuals']['residual_stats']
        report.append(f"- Mean Residual: {residual_stats['mean']:.2f}")
        report.append(f"- Residual Std: {residual_stats['std']:.2f}")
        report.append(f"- Skewness: {residual_stats['skewness']:.4f}")
        report.append(f"- Kurtosis: {residual_stats['kurtosis']:.4f}")
        report.append(f"- Normality (Shapiro): p = {residual_stats['shapiro_pvalue']:.4f}")
        report.append(f"- Autocorrelation (lag 1): {residual_stats['autocorr_lag1']:.4f}")
        report.append("")
        
        # Business Logic Validation
        report.append("## Business Logic Validation")
        business_logic = self.diagnostic_results['business_logic']
        
        neg_coef_issues = business_logic['negative_coefficients']['issues_found']
        report.append(f"- Negative Coefficient Issues: {neg_coef_issues}")
        
        unrealistic_effects = business_logic['unrealistic_effects']['issues_found']
        report.append(f"- Unrealistic Effect Issues: {unrealistic_effects}")
        
        mediation_validation = business_logic['mediation']
        report.append(f"- Overall Mediation Strength: {mediation_validation['overall_mediation_strength']:.2f}")
        report.append("")
        
        # Sensitivity Analysis
        report.append("## Sensitivity Analysis")
        sensitivity = self.diagnostic_results['sensitivity']
        
        if sensitivity['price']['analysis_available']:
            price_elasticity = sensitivity['price']['elasticity']
            report.append(f"- Price Elasticity: {price_elasticity:.4f}")
        
        if sensitivity['promotions']['analysis_available']:
            promo_lift = sensitivity['promotions']['promo_lift']
            report.append(f"- Promotion Lift: {promo_lift:.2%}")
        
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("Based on the diagnostic analysis:")
        
        if perf['r2'] < 0.7:
            report.append("- Consider additional features or model complexity")
        
        if residual_stats['autocorr_lag1'] > 0.3:
            report.append("- Address autocorrelation in residuals")
        
        if not residual_stats['is_normal']:
            report.append("- Consider transforming the target variable")
        
        if neg_coef_issues > 0:
            report.append("- Review negative coefficients for business logic")
        
        return "\n".join(report)
