# QUANT PROJECT 3: ML MODELS FOR BAC NEXT-DAY PRICE PREDICTION

This project builds and compares machine learning models to predict the next-day closing price of Bank of America (BAC) using peer bank stocks and macro variables. Models include Decision Tree, Random Forest, K-Nearest Neighbors (KNN), and Support Vector Regression (SVR).

The full pipeline covers data collection, feature engineering, ML modeling, model diagnostics, and forward testing on a held-out time slice.

---

## OBJECTIVE
Predict BAC’s next-day close using t-1 lagged equity features (BAC + peers + SPY), t-1 macro features (VIX, 10Y yield, USD index, crude oil, gold), and simple technical indicators (MA, rolling vol).

Note: Features are shifted by one trading day; the target is unshifted closing price → effectively a one-day-ahead forecast.

---

## SETUP INSTRUCTIONS
# Clone your repo (replace with your URL)
git clone https://github.com/rithik279/Project3_ML_BAC_PricePrediction
cd Project3_ML_BAC_PricePrediction

# Create env (optional) and install deps
pip install -r requirements.txt

---

## PROJECT PIPELINE

### 1) DATA COLLECTION
Source: Yahoo Finance via yfinance  
Tickers:  
- Equities: BAC, JPM, MS, C, WFC, SPY  
- Macros: ^VIX (VIX), ^TNX (10Y UST yield), DX-Y.NYB (DXY), CL=F (WTI crude), GC=F (Gold)  
Range: 2002-01-01 to 2025-01-01  
Missing data handling: forward fill (ffill) across all series  

### 2) FEATURE ENGINEERING
- Lagged (t-1) equity features: BAC, JPM, MS, C, WFC, SPY  
- Lagged (t-1) macro features: VIX, 10Y_Yield, USDollar_Futures (DXY), Crude_Oil_Futures, Gold_Futures  
- Technical indicators (on BAC):  
  - BAC_MA5, BAC_MA10 (both shifted by 1)  
  - BAC_Volatility5 = 5-day pct change (shifted by 1)  
- Target: BAC close (t)  

### 3) MODELING
All models trained on identical features with a time-series split (no shuffle).  
- Train/Test Split: 90/10 chronological (train_test_split(..., shuffle=False, test_size=0.10))  
- Models:  
  - DecisionTreeRegressor (max_depth=4)  
  - RandomForestRegressor (n_estimators=100)  
  - KNeighborsRegressor (n_neighbors=5)  
  - SVR (default RBF kernel)  
- Artifacts/Insights:  
  - Decision tree structure plot (plot_tree)  
  - Feature importances from the trained tree  
  - Consolidated comparison of predictions across models  

### 4) PERFORMANCE EVALUATION
- Metrics: R², MSE, RMSE (via helper evaluate_model)  
- Visuals & Diagnostics:  
  - Actual vs Predicted line plot (over test window) for all models  
  - Decision tree flowchart  
  - Top feature importances table (from the tree)  
  - (Optional): add residual analysis and error distribution plots  

### 5) TESTING ON FUTURE DATA
The final 10% of the time series is held out for forward testing with the same preprocessing:  
- Apply trained models to the test slice  
- Compare Actual vs Predicted visually and via metrics  

---

## VISUALIZATIONS
- Price comparison: Major Banks vs SPY (full history)  
- Actual vs Predicted (DT, RF, KNN, SVR) on test set  
- Decision tree flowchart  
- Feature importance table (tree)  

---

## KEY FINDINGS (INITIAL)
- Tree-based models (RF, DT) capture nonlinearity and interactions better than linear baselines (not included in this project) but require tuning to generalize.  
- SVR can perform competitively on smooth series but is sensitive to scaling and hyperparameters.  
- KNN provides a simple nonlinear baseline; performance depends heavily on neighbor choice and feature scaling.  
- Even simple t-1 lags + macros provide meaningful signal; adding richer technical & market structure features should improve results.  

---

## REQUIREMENTS
Install via:  
pip install -r requirements.txt  

Major dependencies:  
- yfinance  
- pandas, numpy  
- matplotlib  
- scikit-learn  
- (Optional dev) jupyter  

---

## FILES INCLUDED
- notebooks/model_dev.ipynb – end-to-end exploration, training, and plots  
- logic/ – Python modules for data prep and model training (if modularized)  
- requirements.txt – package versions  
- README.md – project overview and documentation  

---

## AUTHOR NOTES
This project extends the Quant Series from linear models to nonlinear ML using macro + peer inputs. It’s designed to build intuition about:  
- Feature lagging for next-day prediction  
- Model class trade-offs (variance, bias, interpretability)  
- Chronological validation for financial time series  

---

## TO-DO / NEXT STEPS
### Data/Features
- Add returns, log returns, and z-scored features (helps SVR/KNN).  
- Expand technicals: RSI, ATR, rolling skew/kurtosis, EWMA vol.  
- Add calendar features (month, day-of-week) and earnings/holiday flags.  

### Modeling
- Pipeline with StandardScaler (especially for SVR/KNN).  
- Hyperparameter tuning (GridSearchCV/TimeSeriesSplit).  
- Add Gradient Boosting / XGBoost / LightGBM and CatBoost.  

### Evaluation
- Residual plots, QQ plots, ACF/PACF of residuals.  
- Diebold–Mariano test vs baseline (e.g., naïve random walk).  

### Robustness
- Rolling/expanding walk-forward backtests.  
- Out-of-sample stress windows (e.g., 2008, 2020).  

### Deployment
- Streamlit dashboard for on-demand forecasts.  
- Scheduled retraining with model/version registry.  

---

## LIMITATIONS & RISK
- Look-ahead bias avoided via t-1 features; maintain strict time ordering.  
- Non-stationarity: Market regimes shift; periodic retraining is required.  
- Overfitting: Tune depth/trees/neighbors/C/γ; prefer walk-forward validation.  
- Data quality: Ffill can imprint stale macro values over long closures.  

---

© 2025 Rithik Singh — Quant Finance Project Series
