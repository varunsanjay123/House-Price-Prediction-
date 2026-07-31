# End-to-End Housing Price Prediction

This repository contains an end-to-end machine learning pipeline to predict median house values for residential blocks. The project includes exploratory data analysis, preprocessing, model comparison with cross-validation, hyperparameter tuning, final evaluation, and an inference helper.

## Problem Statement

Predict the `median_house_value` using tabular housing features such as location (`longitude`, `latitude`), structural attributes (`total_rooms`, `total_bedrooms`, `households`), demographics (`population`, `median_income`), and `ocean_proximity`.

## Dataset

CSV file: `House Price Prediction/housing.csv` (Kaggle-style housing dataset).

## Pipeline Overview

- Data loading and basic EDA
- Missing-value handling (median for numerical, most frequent for categorical)
- One-hot encoding for categorical features
- Scaling of numerical features with `StandardScaler`
- Built using `ColumnTransformer` and `Pipeline`

## Models Evaluated

- Linear Regression (baseline)
- Ridge, Lasso
- Random Forest Regressor
- HistGradientBoostingRegressor (selected and tuned)

## Final Model & Output

The final tuned model is `HistGradientBoostingRegressor` wrapped in a scikit-learn `Pipeline` (named `hgb_best` in the notebook). The model predicts a single numeric value: the median house value (float) for an input row.

## Evaluation Metrics

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

## How to run

1. Open `House Price Prediction/house_prediction.ipynb` and run cells top-to-bottom.
2. Ensure `housing.csv` is in the `House Price Prediction/` folder or update `CSV_PATH` in the notebook.

## Notes

- Confirm your scikit-learn version supports `root_mean_squared_error`; otherwise compute RMSE with `np.sqrt(mean_squared_error(...))`.
- GridSearchCV with `n_jobs=-1` can be compute-intensive.

## Author

Varunsanjay
