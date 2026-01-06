 Project Name: Visual Analytics for Financial Data
 Version: 1.0
 Author: Hariom Sawant

Description:

This application is a Flask-based web tool for stock-market prediction that combines
machine-learning accuracy with end-user interpretability.

• A Random Forest classifier is trained on key stock indicators
  (price_change, volume, momentum, trend) to classify “Buy” or “Sell.”
• SHAP (SHapley Additive exPlanations) visualizations show feature
  contributions (global bar plots and local beeswarm plots) for transparency.
• A Blockly-style interface lets non-programmers create their own decision
  rules visually, generate Python code, and benchmark their strategies
  against the machine-learning model.

Contents:

- app.py               : Main Flask application.
- stock_data.csv       : Example stock dataset (~10,000 records).
- Literature_Survey.xlsx / User_Survey.xlsx : Reference and survey data.
- static/              : SHAP plots and other static images.
- templates/           : HTML templates (summary.html, custom.html).

Installation:

1. Ensure Python 3.9+ is installed.
2. Unzip `2428568.zip` and open a terminal in the `2428568` folder.
3. Install required packages (Flask, scikit-learn, pandas, numpy, matplotlib, shap):
       pip install flask scikit-learn pandas numpy matplotlib shap

Usage:

1. Start the Flask app:
       python app.py
2. Open a browser at:
       http://127.0.0.1:5000/

Features:

- Summary Page (`/`): Shows Random Forest accuracy, extracted rules,
  most common decision paths, and SHAP plots for Buy/Sell.
- Custom Rules Page (`/custom`): Paste Blockly-generated Python logic
  and view accuracy compared to the Random Forest model.
- Blockly Demo (`/build`): Opens Blockly environment to design rules visually.

Dataset:

The included `stock_data.csv` contains features:
- price_change, volume, momentum, trend (categorical)
with target label `expected` (“Buy”/“Sell”).
Trend is encoded numerically (up=1, down=-1, neutral=0).

Contact:

For questions or support, contact: 2428568@swansea.ac.uk
