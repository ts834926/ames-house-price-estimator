# Ames House Price Estimator

A learning project that explores the Ames, Iowa house-sales data, compares five regression models, and provides a Streamlit form for estimating a home's sale price. The app uses a saved scikit-learn pipeline trained on the labeled sales data.

## Live app

Try the deployed Streamlit app: [Ames House Price Estimator](https://ames-house-price-estimator.streamlit.app/).

## Project structure

```text
Ames House Price Estimator/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   │   ├── data_description.txt
│   │   ├── test.csv
│   │   └── train.csv
│   └── processed/
│       ├── feature_importance.csv
│       ├── holdout_predictions.csv
│       └── model_comparison.csv
├── models/
│   ├── ames_price_pipeline.joblib
│   └── model_metrics.json
└── notebooks/
    └── ames_home_price_estimator.ipynb
```

## File guide

| File | Purpose |
| --- | --- |
| `README.md` | Project overview, file guide, setup, and instructions for running the notebook and app. |
| `app.py` | Streamlit interface. Collects 14 home details, loads the saved model pipeline, displays an estimate, and shows the saved evaluation metrics and charts when their data files are present. |
| `requirements.txt` | Python package dependencies and supported version ranges for the notebook and app. |
| `notebooks/ames_home_price_estimator.ipynb` | End-to-end project notebook: explores the data, prepares features, compares five regressors on a fixed holdout split, saves the selected pipeline and evaluation files, and examines feature importance. Run its cells from top to bottom. |
| `data/raw/train.csv` | Kaggle training data: 1,460 Ames home sales with the `SalePrice` target used to train and evaluate the models. |
| `data/raw/test.csv` | Kaggle competition examples without sale prices. The notebook loads this file for reference, but it is not used for the local holdout evaluation or to train the saved app model. |
| `data/raw/data_description.txt` | Kaggle data dictionary describing the housing fields and category codes. |
| `data/processed/model_comparison.csv` | Held-out MAE, MSE, RMSE, and R² results for Linear Regression, Ridge, Lasso, Decision Tree, and Random Forest. The app uses it for the model comparison table and RMSE chart. |
| `data/processed/holdout_predictions.csv` | Actual prices, predictions, and residuals for the selected model on the held-out sales. Used for analysis in the notebook. |
| `data/processed/feature_importance.csv` | Feature-importance scores from the selected model, grouped back to the original input fields. The app uses this file for its feature chart. |
| `models/ames_price_pipeline.joblib` | Saved scikit-learn preprocessing and estimator pipeline. The app loads this artifact to make predictions. |
| `models/model_metrics.json` | Selected model name, holdout metrics, row counts, and input feature names. The app uses it to display the selected model and its MAE and R². |

The `data/raw/` directory holds the source competition files. The notebook writes evaluation outputs to `data/processed/` and the trained pipeline and its metrics to `models/`.

## Modeling workflow

The notebook uses 14 inputs: nine numeric features (`OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`, `YearBuilt`, `FullBath`, `BedroomAbvGr`, `LotArea`, and `Fireplaces`) and five categorical features (`KitchenQual`, `Neighborhood`, `ExterQual`, `BsmtQual`, and `CentralAir`). It fills numeric gaps with training-set medians, fills categorical gaps with the most common values, scales numeric inputs, and one-hot encodes categories inside each model pipeline.

The five models are compared on the same 20% holdout from `train.csv` using `random_state=42`. The model with the lowest holdout RMSE is selected. The notebook then refits that complete pipeline on all 1,460 labeled sales for use by the app. The holdout metrics continue to describe predictions made on data excluded from model fitting.

## Data files

The raw CSVs and data dictionary come from Kaggle's [House Prices: Advanced Regression Techniques data page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data). Accept the competition rules and place the three files in `data/raw/` if they are missing.

## Set up and run on Windows

In PowerShell, from the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m jupyter lab
```

Open `notebooks/ames_home_price_estimator.ipynb` and run all cells from top to bottom. This regenerates the saved model and outputs. Then, in a PowerShell terminal with the environment activated, start the app:

```powershell
python -m streamlit run app.py
```

The app requires `models/ames_price_pipeline.joblib` and `models/model_metrics.json`. Its comparison and feature-importance sections appear when the corresponding CSV files exist in `data/processed/`.

## Current holdout result

The saved comparison selects **Random Forest** by RMSE. On the fixed 292-sale holdout, it has an MAE of about **$18,126**, RMSE of about **$29,117**, and R² of **0.889**. These results describe this split only; they are not a guarantee of performance on other homes or future sales.

MAE is the average absolute prediction error in dollars. RMSE is also in dollars and gives larger errors more weight. R² compares the predictions with a simple average-price baseline; it is not a percent accuracy score.

## Limitations

This dataset covers historical sales in one city, and the model uses only 14 of the available property details. Its estimate cannot account for unprovided features or future market changes. Use the app as a learning demonstration, not as an appraisal or a substitute for professional advice.
