# Real Estate Investment Advisor
## Predicting Property Profitability & Future Value

A Python/Jupyter project based on the two supplied project briefs. It analyzes Indian housing data, creates an explainable investment classification target, estimates a five-year property value scenario, evaluates machine-learning models, and exports trained pipelines.

## Project files

- `Real_Estate_Investment_Advisor.ipynb` — complete end-to-end notebook.
- `india_housing_prices.csv` — expected input dataset (not included in the uploaded project briefs).
- `artifacts/` — generated model files, metadata and evaluation metrics after running the notebook.

## Important: dataset and target limitations

The project documents refer to `india_housing_prices.csv`, but the CSV itself was not supplied with the documents. The notebook searches for this file in the same folder. If it is missing, the notebook generates a **synthetic demonstration dataset** so the workflow can run from beginning to end. Synthetic data is for code demonstration only and must not be used to claim real market insights.

The brief does not supply observed five-year resale prices or a verified `Good Investment` label. To make the workflow executable, the notebook uses two clearly stated proxy targets:

1. **Good Investment (classification):** 1 when the property's price per square foot is at or below the median for its city and property type. If the group has fewer than 10 records, it falls back to the city median, then the overall median.
2. **Future Price after 5 Years (regression):** current price multiplied by `(1 + annual_appreciation_rate) ** 5`. The default annual appreciation assumption is 8%, matching the example in the project brief.

These models learn the proxy definitions. Their scores are **not evidence of actual investment returns or future market accuracy**. For a defensible forecasting project, obtain historical transactions with sale dates and observed resale values after five years. For investment classification, define a return measure that accounts for costs, rental income, taxes and risk.

## Dataset schema

The notebook expects the fields described in the project brief, including:

- `ID`, `State`, `City`, `Locality`, `Property_Type`
- `BHK`, `Size_in_SqFt`, `Price_in_Lakhs`, `Price_per_SqFt`
- `Year_Built`, `Furnished_Status`, `Floor_No`, `Total_Floors`
- `Nearby_Schools`, `Nearby_Hospitals`, `Public_Transport_Accessibility`
- `Parking_Space`, `Security`, `Amenities`, `Facing`, `Owner_Type`, `Availability_Status`

At minimum, `Price_in_Lakhs` and `Size_in_SqFt` are required. Missing optional columns are handled by the notebook with placeholder values, but real analysis is stronger when the complete dataset is supplied.

## Requirements

Recommended: Python 3.10 or later.

Install packages:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

Optional MLflow support:

```bash
python -m pip install mlflow
```

## Run the notebook

1. Download or obtain the actual `india_housing_prices.csv` dataset.
2. Place the CSV in the same directory as the notebook.
3. Open a terminal in that directory.
4. Start Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open `Real_Estate_Investment_Advisor.ipynb`.
6. Run all cells from top to bottom.

If the CSV is not present, the notebook will use synthetic demonstration data and print the data source clearly.

## Workflow

1. Load and inspect data.
2. Normalize column names, remove duplicate rows, handle invalid numeric values and derive price per square foot.
3. Engineer property age, floor ratio and school-density features.
4. Explore price, size, location, amenities, transport, furnishing, ownership and numeric correlations.
5. Create the documented proxy target variables.
6. Build preprocessing pipelines with median/mode imputation, scaling and one-hot encoding.
7. Train and compare classification and regression models.
8. Evaluate classification with accuracy, precision, recall, F1 and confusion matrix; evaluate regression with MAE, RMSE and R².
9. Calculate permutation feature importance.
10. Save model pipelines, metadata and metric CSV files to `artifacts/`.
11. Optionally log parameters, metrics and models with MLflow.

## Generated artifacts

After a successful run:

- `artifacts/investment_classifier.joblib`
- `artifacts/future_price_regressor.joblib`
- `artifacts/model_metadata.json`
- `artifacts/classification_metrics.csv`
- `artifacts/regression_metrics.csv`

The `.joblib` files contain complete scikit-learn pipelines, including preprocessing. Use the same feature names and compatible package versions when loading them.

## Optional MLflow

In the notebook, set:

```python
ENABLE_MLFLOW = True
```

Then run the MLflow cell. To open the local tracking UI, run:

```bash
mlflow ui
```

Open the local address printed by MLflow (typically `http://127.0.0.1:5000`).

## Suggested future improvements

- Replace proxy future prices with actual time-based resale outcomes.
- Use time-aware train/test splitting when historical transaction dates are available.
- Compare Logistic Regression, Random Forest, XGBoost and other models with cross-validation.
- Add RERA verification, title/approval checks and transaction-cost estimates where reliable data exists.
- Build a Streamlit interface using the exported pipelines.
- Add drift monitoring and retraining as new transactions become available.

## Disclaimer

This is an educational machine-learning project. Predictions are estimates based on the available data and explicit assumptions. They are not financial, legal, valuation or investment advice. Independently verify property documents and consult qualified professionals before making a purchase.
