"""
model.py
Trains ML models to forecast monthly sales revenue.
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
Also forecasts next 6 months and saves all outputs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/sales.csv")
df["date"] = pd.to_datetime(df["date"])

CAT_COLORS = {
    "Electronics": "#3498db", "Clothing": "#e74c3c",
    "Groceries": "#2ecc71",  "Furniture": "#f39c12", "Sports": "#9b59b6",
}

# ─── Feature Engineering ─────────────────────────────────────────────────────
le_cat = LabelEncoder()
le_reg = LabelEncoder()
df["category_enc"] = le_cat.fit_transform(df["category"])
df["region_enc"]   = le_reg.fit_transform(df["region"])

# Time index (months since start)
df = df.sort_values("date").reset_index(drop=True)
start_date = df["date"].min()
df["time_index"] = ((df["date"].dt.year - start_date.year) * 12 +
                    (df["date"].dt.month - start_date.month))

# Month sin/cos encoding for cyclical seasonality
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

FEATURES = [
    "time_index", "month_sin", "month_cos", "quarter",
    "category_enc", "region_enc",
    "ad_spend", "discount_pct", "units_sold", "price_per_unit"
]

X = df[FEATURES]
y = df["revenue"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_scaled, y, df.index, test_size=0.2, random_state=42
)

# ─── Train Models ────────────────────────────────────────────────────────────
models = {
    "Linear Regression":       LinearRegression(),
    "Random Forest":           RandomForestRegressor(n_estimators=150, random_state=42),
    "Gradient Boosting":       GradientBoostingRegressor(n_estimators=150, random_state=42),
}

results = {}
print("=" * 58)
print("  MODEL TRAINING & EVALUATION")
print("=" * 58)

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2   = r2_score(y_test, pred)
    results[name] = {"model": model, "pred": pred, "mae": mae, "rmse": rmse, "r2": r2}
    print(f"\n  {name}")
    print(f"    MAE  : ₹{mae:,.0f}")
    print(f"    RMSE : ₹{rmse:,.0f}")
    print(f"    R²   : {r2:.4f}")

best_name  = max(results, key=lambda n: results[n]["r2"])
best_model = results[best_name]["model"]
best_pred  = results[best_name]["pred"]
print(f"\n  ⭐ Best model: {best_name}  (R² = {results[best_name]['r2']:.4f})")

# ── Chart 09: Actual vs Predicted (best model) ───────────────────────────────
y_test_arr  = np.array(y_test)
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(y_test_arr / 1e3, best_pred / 1e3, alpha=0.55, color="#3498db",
           edgecolors="white", linewidths=0.4, s=50, label="Predictions")
lim = [min(y_test_arr.min(), best_pred.min()) / 1e3 - 10,
       max(y_test_arr.max(), best_pred.max()) / 1e3 + 10]
ax.plot(lim, lim, "r--", linewidth=1.8, label="Perfect Prediction")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_title(f"{best_name}: Actual vs Predicted Revenue", fontsize=15, fontweight="bold", pad=12)
ax.set_xlabel("Actual Revenue (₹K)", fontsize=12)
ax.set_ylabel("Predicted Revenue (₹K)", fontsize=12)
ax.legend(fontsize=11)
ax.set_facecolor("#f8f9fa")
ax.grid(alpha=0.3, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/09_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  ✅ Saved: 09_actual_vs_predicted.png")

# ── Chart 10: Model Comparison Bar ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metric_keys  = ["r2",  "mae",   "rmse"]
metric_labels = ["R² Score", "MAE (₹)", "RMSE (₹)"]
bar_colors   = ["#3498db", "#e74c3c", "#9b59b6"]
model_names  = list(results.keys())
short_names  = ["LR", "RF", "GB"]

for i, (mk, ml) in enumerate(zip(metric_keys, metric_labels)):
    ax = axes[i]
    vals = [results[n][mk] for n in model_names]
    bars = ax.bar(short_names, vals, color=bar_colors, edgecolor="white", linewidth=1, width=0.5)
    for bar, v in zip(bars, vals):
        label = f"{v:.4f}" if mk == "r2" else f"₹{v:,.0f}"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                label, ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title(ml, fontsize=13, fontweight="bold")
    ax.set_facecolor("#f8f9fa")
    ax.grid(axis="y", alpha=0.35, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("Model Performance Comparison  (LR = Linear, RF = Random Forest, GB = Gradient Boosting)",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("outputs/10_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: 10_model_comparison.png")

# ── Chart 11: Feature Importance (RF) ────────────────────────────────────────
rf_model = results["Random Forest"]["model"]
feat_labels = ["Time Index", "Month (sin)", "Month (cos)", "Quarter",
               "Category", "Region", "Ad Spend", "Discount %", "Units Sold", "Price/Unit"]
importances = rf_model.feature_importances_
sorted_idx  = np.argsort(importances)
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(sorted_idx)))

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh([feat_labels[i] for i in sorted_idx], importances[sorted_idx],
        color=colors, edgecolor="white", linewidth=0.8)
ax.set_title("Feature Importance for Revenue Forecasting", fontsize=15, fontweight="bold", pad=12)
ax.set_xlabel("Importance Score", fontsize=12)
ax.set_facecolor("#f8f9fa")
ax.grid(axis="x", alpha=0.35, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/11_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: 11_feature_importance.png")

# ── Chart 12: 6-Month Forecast ───────────────────────────────────────────────
# Build future rows for each category × region for next 6 months
future_months = pd.date_range("2025-01-01", periods=6, freq="MS")
future_rows = []
categories = df["category"].unique()
regions    = df["region"].unique()

for fdate in future_months:
    for cat in categories:
        for reg in regions:
            ti = ((fdate.year - start_date.year) * 12 + (fdate.month - start_date.month))
            future_rows.append({
                "time_index":   ti,
                "month_sin":    np.sin(2 * np.pi * fdate.month / 12),
                "month_cos":    np.cos(2 * np.pi * fdate.month / 12),
                "quarter":      (fdate.month - 1) // 3 + 1,
                "category_enc": le_cat.transform([cat])[0],
                "region_enc":   le_reg.transform([reg])[0],
                "ad_spend":     25000,      # average assumed
                "discount_pct": 15.0,
                "units_sold":   1000,
                "price_per_unit": 300.0,
                "date":         fdate,
                "category":     cat,
            })

future_df = pd.DataFrame(future_rows)
X_future  = scaler.transform(future_df[FEATURES])
future_df["predicted_revenue"] = best_model.predict(X_future)

# Aggregate by date + category
forecast_cat = future_df.groupby(["date", "category"])["predicted_revenue"].sum().reset_index()
forecast_total = future_df.groupby("date")["predicted_revenue"].sum().reset_index()

# Historical total for context (last 12 months)
hist_monthly = df.groupby("date")["revenue"].sum().reset_index()
hist_last12  = hist_monthly[hist_monthly["date"] >= "2024-01-01"]

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(hist_last12["date"], hist_last12["revenue"] / 1e6, color="#3498db",
        linewidth=2.2, marker="o", markersize=5, label="Historical (2024)")
ax.plot(forecast_total["date"], forecast_total["predicted_revenue"] / 1e6,
        color="#e74c3c", linewidth=2.2, marker="s", markersize=5,
        linestyle="--", label="Forecast (Jan–Jun 2025)")
ax.axvline(pd.Timestamp("2025-01-01"), color="gray", linewidth=1.2, linestyle=":", alpha=0.8)
ax.text(pd.Timestamp("2025-01-01"), ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0,
        "  Forecast →", fontsize=10, color="gray")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:.0f}M"))
ax.set_title("Sales Forecast: Next 6 Months (Jan–Jun 2025)", fontsize=15, fontweight="bold", pad=14)
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Total Revenue", fontsize=12)
ax.legend(fontsize=11)
ax.set_facecolor("#f8f9fa")
ax.grid(alpha=0.35, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/12_sales_forecast.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: 12_sales_forecast.png")

# Print forecast summary
print("\n" + "=" * 58)
print("  6-MONTH FORECAST SUMMARY (All Categories + Regions)")
print("=" * 58)
for _, row in forecast_total.iterrows():
    print(f"  {row['date'].strftime('%b %Y')} : ₹{row['predicted_revenue']/1e6:.2f}M")

print(f"\n{'─'*58}")
print(f"  Best Model → {best_name}")
print(f"  R²  : {results[best_name]['r2']:.4f}")
print(f"  MAE : ₹{results[best_name]['mae']:,.0f}")
print(f"{'─'*58}")
print("✅ All model outputs saved to outputs/")
