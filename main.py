"""
main.py
Run this to execute the full Sales Forecasting pipeline.
  1. Generates synthetic sales dataset (2022–2024)
  2. Creates all EDA visualizations
  3. Trains ML models and forecasts next 6 months
"""

print("=" * 62)
print("  SALES FORECASTING")
print("  CodTech IT Solutions — ML Internship Task 2")
print("=" * 62)

import subprocess, sys

steps = [
    ("Step 1: Generating sales dataset...",       "generate_data.py"),
    ("Step 2: Creating visualizations...",         "visualize.py"),
    ("Step 3: Training models & forecasting...",   "model.py"),
]

for msg, script in steps:
    print(f"\n{'─'*62}")
    print(f"  {msg}")
    print(f"{'─'*62}")
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"\n❌ Error in {script}. Stopping.")
        sys.exit(1)

print("\n" + "=" * 62)
print("  ✅ ALL DONE! Check the outputs/ folder for images.")
print("=" * 62)
