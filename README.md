# MMM Modeling with Mediation Assumption

## Overview
This project implements a Media Mix Model (MMM) that treats Google spend as a mediator between social/display channels (Facebook, TikTok, Snapchat) and Revenue. The model uses a causal perspective to understand how social media can stimulate search intent, which then influences Google spend and ultimately affects revenue.

## Project Structure
```
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── data/                     # Data directory
│   ├── raw/                 # Raw data files
│   └── processed/           # Processed data files
├── notebooks/               # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_mediation_modeling.ipynb
│   └── 04_diagnostics_insights.ipynb
├── src/                     # Source code
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── mediation_model.py
│   ├── diagnostics.py
│   └── utils.py
├── results/                 # Model outputs and visualizations
└── docs/                    # Additional documentation
```

## Key Features

### 1. Causal Framework
- **Mediation Assumption**: Google spend mediates the relationship between social channels and revenue
- **Two-stage modeling**: First stage models Google spend as function of social channels, second stage models revenue
- **DAG-consistent feature design**: Respects causal relationships in feature engineering

### 2. Time Series Considerations
- **Proper CV**: Time-aware cross-validation with no look-ahead bias
- **Seasonality**: Weekly seasonality and trend decomposition
- **Zero-spend handling**: Robust treatment of periods with no media spend

### 3. Model Architecture
- **Regularized regression**: Elastic net with cross-validation
- **Feature engineering**: Adstock transformations, saturation curves, interaction terms
- **Validation**: Rolling window and blocked time series CV

### 4. Diagnostics & Insights
- **Out-of-sample performance**: Comprehensive validation metrics
- **Residual analysis**: Time series and distribution diagnostics
- **Sensitivity analysis**: Price elasticity and promotion lift analysis
- **Business insights**: Actionable recommendations for marketing teams

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the notebooks in order:
   - `01_data_exploration.ipynb`
   - `02_data_preparation.ipynb`
   - `03_mediation_modeling.ipynb`
   - `04_diagnostics_insights.ipynb`

## Usage

The main modeling pipeline can be run through the notebooks or imported as modules:

```python
from src.data_preparation import DataPreparator
from src.mediation_model import MediationMMM

# Prepare data
prep = DataPreparator()
data = prep.prepare_data('data/raw/mmm_data.csv')

# Build mediation model
model = MediationMMM()
results = model.fit(data)
```

## Results

Key outputs include:
- Model performance metrics and diagnostics
- Attribution analysis with mediation effects
- Price elasticity and promotion lift estimates
- Budget allocation recommendations
- Sensitivity analysis for key business levers

## Methodology

### Mediation Framework
The model implements a two-stage approach:
1. **Stage 1**: Google spend = f(Facebook, TikTok, Snapchat, controls)
2. **Stage 2**: Revenue = f(Google_spend, direct_channels, price, promotions, controls)

This structure explicitly models the causal pathway where social channels influence search intent, which drives Google spend, which then affects revenue.

### Validation Strategy
- **Time Series CV**: Rolling window validation respecting temporal order
- **Blocked CV**: Ensures no data leakage across time periods
- **Out-of-sample testing**: Hold-out validation on most recent data

## Business Applications

The model provides actionable insights for:
- **Budget allocation**: Optimal spend across channels considering mediation effects
- **Price optimization**: Understanding price elasticity and demand response
- **Promotion planning**: Quantifying lift from promotional activities
- **Channel strategy**: Understanding how social channels drive search behavior

## Technical Notes

- All results are reproducible with fixed random seeds
- Model handles missing data and zero-spend periods robustly
- Feature engineering includes adstock transformations and saturation curves
- Regularization prevents overfitting in high-dimensional feature space
