# 📈 Sales Forecasting

**CodTech IT Solutions — ML Internship**

| Field | Details |
|---|---|
| **Intern ID** | CITS5203 |
| **Full Name** | Anagh Pandey |
| **No. of Weeks** | 4 |
| **Project Name** | Sales Forecasting |
| **Project Scope** | Machine Learning |

---

## 📌 Project Overview

A machine learning project that analyzes 3 years of multi-category retail sales data (2022–2024) and forecasts the next 6 months of revenue. Includes full exploratory data analysis, seasonality detection, and model comparison.

---

## 🗂️ Project Structure

```
sales-forecasting/
├── data/
│   └── sales.csv                     # Generated dataset (720 records)
├── outputs/                          # All output charts
│   ├── 01_monthly_revenue_trend.png
│   ├── 02_revenue_by_category.png
│   ├── 03_revenue_by_region.png
│   ├── 04_seasonality_heatmap.png
│   ├── 05_category_share_pie.png
│   ├── 06_adspend_vs_revenue.png
│   ├── 07_quarterly_revenue.png
│   ├── 08_discount_vs_revenue.png
│   ├── 09_actual_vs_predicted.png
│   ├── 10_model_comparison.png
│   ├── 11_feature_importance.png
│   └── 12_sales_forecast.png
├── generate_data.py                  # Generates 3-year sales dataset
├── visualize.py                      # EDA charts and trend analysis
├── model.py                          # ML training + 6-month forecast
├── main.py                           # Run everything with one command
└── requirements.txt
```

---

## ⚙️ Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python main.py
```

Or step by step:
```bash
python generate_data.py   # Generate dataset
python visualize.py       # EDA charts
python model.py           # Train & forecast
```

---

## 📊 Dataset Features

| Feature | Description |
|---|---|
| `date` | Monthly sales date (2022–2024) |
| `category` | Product category (Electronics, Clothing, etc.) |
| `region` | Sales region (North, South, East, West) |
| `ad_spend` | Monthly advertising spend (₹) |
| `discount_pct` | Discount percentage applied |
| `units_sold` | Units sold that month |
| `price_per_unit` | Average price per unit (₹) |
| `revenue` | Total monthly revenue (₹) |

---

## 🤖 Models Used

| Model | MAE | R² Score |
|---|---|---|
| Linear Regression | ~₹56,000 | 0.875 |
| Random Forest Regressor | ~₹18,000 | 0.976 |
| **Gradient Boosting Regressor** | **~₹13,700** | **0.992** |

> **Best model: Gradient Boosting Regressor**

---

## 📸 Output Images

All 12 charts are saved to `outputs/` automatically when you run the project.

---

## 🛠️ Tech Stack

- **Python 3**
- **pandas** — data handling
- **numpy** — numerical operations
- **matplotlib / seaborn** — visualizations
- **scikit-learn** — ML models (LinearRegression, RandomForest, GradientBoosting)
# sales-forceasting
# sales-forceasting
