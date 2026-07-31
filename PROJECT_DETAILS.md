# Project Details — End-to-End Housing Price Prediction

**Project Title:** End-to-End Housing Price Prediction (Machine Learning)

**Problem Statement:**
Predict the median house value (median_house_value) for residential blocks using tabular housing features (location, rooms, population, income, and proximity to ocean).

**Dataset:**
House Prices dataset (Kaggle-style). CSV located at `House Price Prediction/housing.csv`.

**What I used (Tech stack & key libraries):**
- Python
- pandas, numpy
- scikit-learn (Pipeline, ColumnTransformer, imputation, encoders, model selection)
- matplotlib, seaborn

**Preprocessing steps:**
- Identify numerical and categorical features
- Impute missing numerical values with median
- Impute missing categorical values with most frequent
- One-hot encode categorical features (`OneHotEncoder(handle_unknown='ignore')`)
- Scale numerical features with `StandardScaler`
- Built using `ColumnTransformer` and `Pipeline`

**Models tried:**
- Linear Regression (baseline)
- Ridge, Lasso (regularized linear models)
- Random Forest Regressor
- HistGradientBoostingRegressor (boosting)

**Selected / Final model:**
- `HistGradientBoostingRegressor` (tuned)
- Tuned hyperparameters used in the notebook (hard-coded):
  - `l2_regularization=0.1`
  - `learning_rate=0.1`
  - `max_depth=None`
  - `max_leaf_nodes=63`
  - `min_samples_leaf=20`

**Evaluation Metrics (computed in notebook):**
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

**Final Output:**
- A numeric prediction: median house value (float) for a single input row.
- Trained model artifacts exist in-memory in the notebook as `hgb_best` (Pipeline), and an inference helper `predict_house_price(model, ...)` to get predictions for one house.


