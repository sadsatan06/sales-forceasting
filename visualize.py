"""
visualize.py
Generates all exploratory analysis charts for the sales dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/sales.csv")
df["date"] = pd.to_datetime(df["date"])

CAT_COLORS = {
    "Electronics": "#3498db",
    "Clothing":    "#e74c3c",
    "Groceries":   "#2ecc71",
    "Furniture":   "#f39c12",
    "Sports":      "#9b59b6",
}
REG_COLORS = {"North": "#1abc9c", "South": "#e67e22", "East": "#8e44ad", "West": "#2980b9"}

def fmt_cr(x, _=None):
    return f"₹{x/1e6:.1f}M"

print("📊 Generating visualizations...")

# ── 1. Total Monthly Revenue Trend ───────────────────────────────────────────
monthly = df.groupby("date")["revenue"].sum().reset_index()
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(monthly["date"], monthly["revenue"], color="#3498db", linewidth=2.2, marker="o", markersize=4)
ax.fill_between(monthly["date"], monthly["revenue"], alpha=0.12, color="#3498db")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_cr))
ax.set_title("Total Monthly Revenue Trend (2022–2024)", fontsize=16, fontweight="bold", pad=14)
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Revenue", fontsize=12)
ax.set_facecolor("#f8f9fa")
ax.grid(alpha=0.35, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/01_monthly_revenue_trend.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 01_monthly_revenue_trend.png")

# ── 2. Revenue by Category (Line) ────────────────────────────────────────────
cat_monthly = df.groupby(["date", "category"])["revenue"].sum().reset_index()
fig, ax = plt.subplots(figsize=(12, 6))
for cat in df["category"].unique():
    sub = cat_monthly[cat_monthly["category"] == cat]
    ax.plot(sub["date"], sub["revenue"], label=cat, color=CAT_COLORS[cat],
            linewidth=2, marker="o", markersize=3.5)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_cr))
ax.set_title("Monthly Revenue by Category (2022–2024)", fontsize=16, fontweight="bold", pad=14)
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Revenue", fontsize=12)
ax.legend(title="Category", fontsize=10, title_fontsize=11)
ax.set_facecolor("#f8f9fa")
ax.grid(alpha=0.35, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/02_revenue_by_category.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 02_revenue_by_category.png")

# ── 3. Revenue by Region (Stacked Bar per Year) ───────────────────────────────
yearly_reg = df.groupby(["year", "region"])["revenue"].sum().unstack()
fig, ax = plt.subplots(figsize=(9, 5))
yearly_reg.plot(kind="bar", ax=ax, color=[REG_COLORS[r] for r in yearly_reg.columns],
                edgecolor="white", linewidth=0.8, width=0.6)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_cr))
ax.set_title("Annual Revenue by Region", fontsize=16, fontweight="bold", pad=14)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Revenue", fontsize=12)
ax.legend(title="Region", fontsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_facecolor("#f8f9fa")
ax.grid(axis="y", alpha=0.35, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/03_revenue_by_region.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 03_revenue_by_region.png")

# ── 4. Monthly Seasonality Heatmap (category × month avg) ────────────────────
season_df = df.groupby(["category", "month"])["revenue"].mean().unstack()
month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
season_df.columns = month_labels

fig, ax = plt.subplots(figsize=(13, 5))
sns.heatmap(season_df / 1e6, annot=True, fmt=".1f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, annot_kws={"size": 9},
            cbar_kws={"label": "Revenue (M)"})
ax.set_title("Average Monthly Revenue Heatmap by Category (₹M)", fontsize=14, fontweight="bold", pad=14)
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Category", fontsize=12)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/04_seasonality_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 04_seasonality_heatmap.png")

# ── 5. Category Revenue Share (Pie) ──────────────────────────────────────────
cat_total = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 7))
wedges, texts, autotexts = ax.pie(
    cat_total.values,
    labels=cat_total.index,
    autopct="%1.1f%%",
    colors=[CAT_COLORS[c] for c in cat_total.index],
    startangle=140,
    pctdistance=0.82,
    wedgeprops=dict(edgecolor="white", linewidth=2),
)
for t in autotexts:
    t.set_fontsize(11)
    t.set_fontweight("bold")
ax.set_title("Total Revenue Share by Category (2022–2024)", fontsize=15, fontweight="bold", pad=16)
plt.tight_layout()
plt.savefig("outputs/05_category_share_pie.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 05_category_share_pie.png")

# ── 6. Ad Spend vs Revenue Scatter ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
for cat in df["category"].unique():
    sub = df[df["category"] == cat]
    ax.scatter(sub["ad_spend"], sub["revenue"] / 1e3,
               color=CAT_COLORS[cat], alpha=0.5, s=30, label=cat, edgecolors="none")
z = np.polyfit(df["ad_spend"], df["revenue"] / 1e3, 1)
p = np.poly1d(z)
x_l = np.linspace(df["ad_spend"].min(), df["ad_spend"].max(), 200)
ax.plot(x_l, p(x_l), "k--", linewidth=1.8, alpha=0.5, label="Trend")
ax.set_title("Ad Spend vs Revenue", fontsize=16, fontweight="bold", pad=14)
ax.set_xlabel("Ad Spend (₹)", fontsize=12)
ax.set_ylabel("Revenue (₹K)", fontsize=12)
ax.legend(title="Category", fontsize=9)
ax.set_facecolor("#f8f9fa")
ax.grid(alpha=0.3, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/06_adspend_vs_revenue.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 06_adspend_vs_revenue.png")

# ── 7. Quarterly Revenue Growth ───────────────────────────────────────────────
df["year_q"] = df["year"].astype(str) + "-Q" + df["quarter"].astype(str)
qtr = df.groupby("year_q")["revenue"].sum().reset_index()
# Sort chronologically
qtr = qtr.sort_values("year_q")
fig, ax = plt.subplots(figsize=(12, 5))
colors_q = ["#3498db" if "2022" in r else "#2ecc71" if "2023" in r else "#e74c3c" for r in qtr["year_q"]]
bars = ax.bar(qtr["year_q"], qtr["revenue"] / 1e6, color=colors_q, edgecolor="white", linewidth=0.8, width=0.6)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:.0f}M"))
ax.set_title("Quarterly Revenue (2022–2024)", fontsize=16, fontweight="bold", pad=14)
ax.set_xlabel("Quarter", fontsize=12)
ax.set_ylabel("Revenue", fontsize=12)
ax.set_xticklabels(qtr["year_q"], rotation=45, ha="right")
ax.set_facecolor("#f8f9fa")
ax.grid(axis="y", alpha=0.35, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
# Legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#3498db", label="2022"),
                   Patch(color="#2ecc71", label="2023"),
                   Patch(color="#e74c3c", label="2024")], fontsize=10)
plt.tight_layout()
plt.savefig("outputs/07_quarterly_revenue.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 07_quarterly_revenue.png")

# ── 8. Discount % vs Revenue ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
for cat in df["category"].unique():
    sub = df[df["category"] == cat]
    ax.scatter(sub["discount_pct"], sub["revenue"] / 1e3,
               color=CAT_COLORS[cat], alpha=0.45, s=28, label=cat, edgecolors="none")
ax.set_title("Discount % vs Revenue", fontsize=16, fontweight="bold", pad=14)
ax.set_xlabel("Discount (%)", fontsize=12)
ax.set_ylabel("Revenue (₹K)", fontsize=12)
ax.legend(title="Category", fontsize=9)
ax.set_facecolor("#f8f9fa")
ax.grid(alpha=0.3, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/08_discount_vs_revenue.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 08_discount_vs_revenue.png")

print("\n✅ All 8 visualizations saved to outputs/")
