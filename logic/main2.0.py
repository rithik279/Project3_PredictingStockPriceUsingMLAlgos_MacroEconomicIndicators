# Project 3 (Patched): Predict BAC Next-Day Close with ML + Macro Factors

# =========================
# 1) Imports
# =========================
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# =========================
# 2) Universe & Download
# =========================
tickers = ['BAC', 'JPM', "MS", "C", "WFC", "SPY", "^VIX", "^TNX", "DX-Y.NYB","CL=F","GC=F"]
data = yf.download(tickers, start="2002-01-01", end="2025-01-01")["Close"]

# Basic sanity prints (optional)
# print(data.info())
# print(data.describe())
# print(data.isnull().sum())

# Forward-fill gaps (holidays / non-trading days in some series)
data = data.ffill()

# =========================
# 3) Feature Engineering
# =========================
df = pd.DataFrame(index=data.index)

# --- Lagged equity features (t-1)
df['BAC(t-1)'] = data["BAC"].shift(1)
df['JPM(t-1)'] = data["JPM"].shift(1)
df['MS(t-1)']  = data["MS"].shift(1)
df['C(t-1)']   = data["C"].shift(1)
df['WFC(t-1)'] = data["WFC"].shift(1)
df['SPY(t-1)'] = data["SPY"].shift(1)

# --- Lagged macro features (t-1)
df['VIX(t-1)']           = data["^VIX"].shift(1)
df['10Y_Yield(t-1)']     = data["^TNX"].shift(1)         # Note: TNX is yield*10
df['Gold_Futures(t-1)']  = data["GC=F"].shift(1)
df['USDollar_Fut(t-1)']  = data["DX-Y.NYB"].shift(1)
df['Crude_Oil_Fut(t-1)'] = data["CL=F"].shift(1)

# --- Technical indicators (lagged)
df["BAC_MA5"]        = data["BAC"].rolling(window=5).mean().shift(1)
df["BAC_MA10"]       = data["BAC"].rolling(window=10).mean().shift(1)  # FIXED
df["BAC_Volatility5"] = data["BAC"].pct_change(5).shift(1)

# --- Target: Next-day price (shift -1)
df["Target"] = data['BAC'].shift(-1)

# Drop rows with NA created by rolling/shift
df = df.dropna()

# =========================
# 4) Train/Test Split (Chronological)
# =========================
X = df.drop(columns=['Target'])
y = df['Target']

# Last 10% for test, no shuffle
X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=False, test_size=0.10
)

# =========================
# 5) Define Models
# =========================
# Tree models (no scaling needed)
dt_model = DecisionTreeRegressor(max_depth=4, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Distance/kernel models — use Pipeline with StandardScaler
knn_model = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor(n_neighbors=5))
])

svr_model = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('svr', SVR())  # defaults: kernel='rbf', C=1.0, epsilon=0.1
])

models = {
    "Decision Tree": dt_model,
    "Random Forest": rf_model,
    "KNN": knn_model,
    "SVR": svr_model
}

# =========================
# 6) Time-Series Cross-Validation on TRAIN only
# =========================
tscv = TimeSeriesSplit(n_splits=5)

def cv_report(model, X_tr, y_tr, name):
    # R2 via cross_val_score across time splits
    r2_scores = cross_val_score(model, X_tr, y_tr, cv=tscv, scoring='r2')
    print(f"[CV] {name} | R2 mean={r2_scores.mean():.4f}, std={r2_scores.std():.4f}")

for name, mdl in models.items():
    cv_report(mdl, X_train, y_train, name)

# =========================
# 7) Fit on TRAIN and Predict on TEST
# =========================
for mdl in models.values():
    mdl.fit(X_train, y_train)

dt_pred  = models["Decision Tree"].predict(X_test)
rf_pred  = models["Random Forest"].predict(X_test)
knn_pred = models["KNN"].predict(X_test)
svr_pred = models["SVR"].predict(X_test)

# =========================
# 8) Evaluation Helpers
# =========================
def evaluate(y_true, y_pred, model_name):
    r2   = r2_score(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    # Directional accuracy (up/down)
    # Compare actual t vs t-1 sign and predicted vs t-1 sign
    actual_ret = np.sign(np.array(y_true) - np.array(X_test['BAC(t-1)']))
    pred_ret   = np.sign(np.array(y_pred) - np.array(X_test['BAC(t-1)']))
    dir_acc    = (actual_ret == pred_ret).mean()
    print(f"Model: {model_name}")
    print(f" R2   : {r2:.6f}")
    print(f" MSE  : {mse:.6f}")
    print(f" RMSE : {rmse:.6f}")
    print(f" MAE  : {mae:.6f}")
    print(f" Dir% : {dir_acc*100:.2f}%")
    print("")

print("\n=== Test Performance (OOS) ===")
evaluate(y_test, dt_pred,  "Decision Tree")
evaluate(y_test, rf_pred,  "Random Forest")
evaluate(y_test, knn_pred, "KNN (Scaled)")
evaluate(y_test, svr_pred, "SVR (Scaled)")

# =========================
# 9) Feature Importances (Tree-based)
# =========================
# Decision Tree importances
dt_importance = models["Decision Tree"].feature_importances_
dt_feat_imp = pd.DataFrame({"Feature": X_train.columns, "Importance": dt_importance}) \
                .sort_values(by="Importance", ascending=False)
print("\nDecision Tree Feature Importances:\n", dt_feat_imp.head(15))

# Random Forest importances
rf_importance = models["Random Forest"].feature_importances_
rf_feat_imp = pd.DataFrame({"Feature": X_train.columns, "Importance": rf_importance}) \
                .sort_values(by="Importance", ascending=False)
print("\nRandom Forest Feature Importances:\n", rf_feat_imp.head(15))

# Optional: visualize the DT structure
plt.figure(figsize=(20, 10))
plot_tree(models["Decision Tree"], feature_names=X.columns, filled=True, rounded=True, precision=2)
plt.title("Decision Tree Flowchart")
plt.show()

# =========================
# 10) Result Table & Plots
# =========================
result = pd.DataFrame(index=y_test.index)
result['Actual']          = y_test.values
result['DT Prediction']   = dt_pred
result['RF Prediction']   = rf_pred
result['KNN Prediction']  = knn_pred
result['SVR Prediction']  = svr_pred
print("\nHead of Result Frame:\n", result.head())

# Plot: Actual vs each model
plt.figure(figsize=(14, 8))
plt.plot(result.index, result['Actual'], label="Actual BAC (t+1)")
plt.plot(result.index, result['DT Prediction'], label="Decision Tree")
plt.plot(result.index, result['RF Prediction'], label="Random Forest")
plt.plot(result.index, result['KNN Prediction'], label="KNN (Scaled)")
plt.plot(result.index, result['SVR Prediction'], label="SVR (Scaled)")
plt.title("Actual vs Predicted Next-Day BAC Price (Test Set)")
plt.xlabel("Date"); plt.ylabel("Price ($)")
plt.grid(); plt.legend(); plt.show()
