Credit Card Default Prediction: Cost-Sensitive Optimization & SHAP

A business-driven machine learning project that predicts credit card default risk while optimizing lending decisions for net cash flow rather than classification accuracy alone.

## Project Overview

Financial institutions must balance expected lending returns against default risk. The costs are asymmetric:

- **False Negative:** A defaulter is approved, resulting in a potential credit loss.
- **True Negative:** A non-defaulter is approved, generating lending profit.
- **False Positive:** A good customer is rejected, creating an opportunity cost. This opportunity cost is excluded from the project's cash-flow objective to avoid encouraging overly aggressive lending.

The project compares five classification models, selects the best model using validation AUC, optimizes the decision threshold using a custom business cost function, and applies SHAP to explain both global and individual predictions.

## Key Results

| Metric | Result |
| :--- | :--- |
| Best model | LightGBM Classifier (Cross-Entropy) |
| Validation AUC | 0.7642 |
| Test AUC | 0.7659 |
| Cost-optimal threshold | 0.13 |
| Net cash flow | $5.75 million on 4,076 test customers |
| Approval rate | 40.7% |
| Default catch rate | 83.3% |
| Profit per approved customer | $3,466 |

## Methodology

### Model Comparison

Five models were benchmarked across interpretable and tree-based approaches:

- **Glassbox models:** Ridge Regression, Logistic Regression, and Linear SVM.
- **Blackbox models:** LightGBM Regressor and LightGBM Classifier.

LightGBM with cross-entropy loss achieved the highest validation AUC and generalized well to the test set, with an AUC gap of only 0.0017.

### Cost-Sensitive Threshold Optimization

Instead of applying the default 0.50 cutoff, the project evaluates lending outcomes using a custom net cash-flow objective based on a 50% loss-given-default assumption and a 10% profit margin. The selected threshold of 0.13 minimizes net business cost under this framework.

### Feature Engineering

Eight behavioral features were engineered, including:

- `MAX_DELINQUENCY`: Worst payment delay observed over six months.
- `DELINQUENCY_PERSISTENCE`: Consecutive months with late payments.
- `REPAYMENT_RATIO`: Total payments divided by total billed amounts.
- `PAY_SLOPE`: Trend in payment amounts over time.
- `UTILIZATION_TREND`: Change in credit utilization over time.

### SHAP Explainability

SHAP analysis provides:

- **Global explanations:** Payment history and delinquency behavior dominate model importance, while demographic variables rank substantially lower.
- **Local explanations:** Individual waterfall plots show which factors increase or reduce each customer's predicted risk.

![Global SHAP Feature Importance](static/images/global_feature_shap_absolute.png)

## Repository Structure

```text
credit_default.ipynb     End-to-end data preparation, modelling and evaluation
app.py                   Flask dashboard application
data/                    Source and dashboard-ready data
report/                  LaTeX source and final project report
static/                  Dashboard styles and visual assets
templates/               Flask HTML templates
requirements.txt         Python dependencies
```

## Run the Analysis

Python 3.8 or later is recommended.

```bash
pip install -r requirements.txt
pip install notebook
jupyter notebook credit_default.ipynb
```

Run the notebook from top to bottom to reproduce data cleaning, feature engineering, model comparison, threshold optimization, and SHAP analysis.

## Run the Dashboard Locally

The previous hosted demo is no longer active. The complete dashboard source remains available in this repository and can be run locally:

```bash
pip install -r requirements.txt
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in a browser. The dashboard uses `data/webapp_data.csv` to display customer-level predictions and interactive SHAP explanations.

## Tech Stack

Python, pandas, NumPy, scikit-learn, LightGBM, SHAP, Flask, Matplotlib and Seaborn.
