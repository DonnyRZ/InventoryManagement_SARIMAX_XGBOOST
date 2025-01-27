# Install required libraries
!pip install optuna
!pip install pmdarima

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb
import optuna
from pmdarima import auto_arima

# Step 1: Upload Dataset
uploaded = files.upload()
df = pd.read_excel(next(iter(uploaded.values())), sheet_name=0)

# Step 2: Enhanced Data Preprocessing
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df[~df.index.duplicated(keep='first')].sort_index()

# Convert Product ID to numeric
df['Product ID'] = pd.to_numeric(df['Product ID'], errors='coerce')

# Handle missing values
target_column = 'Quantity in Stock (liters/kg)'
df.dropna(subset=[target_column], inplace=True)

# Encode categorical variables
cat_cols = [
    'Product Name', 'Brand', 'Sales Channel', 'Storage Condition',
    'Location', 'Customer Location', 'Farm Size'
]
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_df = pd.DataFrame(
    encoder.fit_transform(df[cat_cols]).toarray(),
    columns=encoder.get_feature_names_out(cat_cols),
    index=df.index
)
df = pd.concat([df.drop(cat_cols, axis=1), encoded_df], axis=1)

# Feature engineering
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

sold_column = 'Quantity Sold (liters/kg)'
for lag in [7, 14, 21]:
    df[f'lag_{lag}'] = df[sold_column].shift(lag)

df['rolling_mean_7'] = df[sold_column].rolling(7).mean()
df['rolling_std_14'] = df[sold_column].rolling(14).std()

# Fourier transforms
for k in [1, 2]:
    df[f'sin_{k}'] = np.sin(2 * np.pi * k * df['day_of_week']/7)
    df[f'cos_{k}'] = np.cos(2 * np.pi * k * df['day_of_week']/7)
    df[f'month_sin_{k}'] = np.sin(2 * np.pi * k * df['month']/12)
    df[f'month_cos_{k}'] = np.cos(2 * np.pi * k * df['month']/12)

# Interaction terms
df['lag_7_x_day'] = df['lag_7'] * df['day_of_week']
df['month_x_product'] = df['month'] * df['Product ID']

# Final cleaning and scaling
numerical_cols = [
    'Total Land Area (acres)', 'Number of Cows', 'rolling_mean_7',
    'lag_7', 'lag_14', 'lag_21'  # Added lag features for scaling
]
df[numerical_cols] = RobustScaler().fit_transform(df[numerical_cols])
df.dropna(inplace=True)

# Verify data integrity
assert not df.isna().any().any(), "NaN values present in final dataset"
assert df.select_dtypes(include=['object']).empty, "Categorical columns remain"

# Temporal split
train_size = int(0.8 * len(df))
train_X, val_X = df.iloc[:train_size].drop(target_column, axis=1), df.iloc[train_size:].drop(target_column, axis=1)
train_y, val_y = df.iloc[:train_size][target_column], df.iloc[train_size:][target_column]

# Step 3: SARIMAX Model with Validation
sarimax_exog_cols = ['day_of_week', 'lag_7', 'lag_14', 'lag_21']

print("Training SARIMAX model...")
sarimax_model = auto_arima(
    train_y,
    exogenous=train_X[sarimax_exog_cols],
    seasonal=True,
    m=7,
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore'
)
print("SARIMAX model summary:")
print(sarimax_model.summary())

# Generate predictions with validation
train_predictions = pd.Series(
    sarimax_model.predict_in_sample(exogenous=train_X[sarimax_exog_cols]),
    index=train_X.index
)
train_residuals = (train_y - train_predictions).dropna()
train_X_clean = train_X.loc[train_residuals.index]

# Step 4: XGBoost Residual Modeling with Error Handling
def objective(trial):
    # SARIMAX forecast
    try:
        val_forecast = pd.Series(
            sarimax_model.predict(n_periods=len(val_X), exogenous=val_X[sarimax_exog_cols]),
            index=val_X.index
        )
    except ValueError as e:
        print(f"SARIMAX prediction error: {e}")
        return float('inf')

    # Handle NaNs in forecast
    val_forecast = val_forecast.dropna()
    if len(val_forecast) == 0:
        return float('inf')

    val_X_clean = val_X.loc[val_forecast.index]
    val_y_clean = val_y.loc[val_forecast.index]
    val_residuals = (val_y_clean - val_forecast).values

    if len(val_residuals) == 0:
        return float('inf')

    # XGBoost parameters
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.5)
    }

    # Train and validate
    try:
        model = xgb.XGBRegressor(**params, random_state=42)
        model.fit(train_X_clean, train_residuals)
        preds = model.predict(val_X_clean)
        return mean_absolute_error(val_residuals, preds)
    except Exception as e:
        print(f"XGBoost error: {e}")
        return float('inf')

# Optimize hyperparameters
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, timeout=3600)

# Final model
best_xgb = xgb.XGBRegressor(**study.best_params, random_state=42)
best_xgb.fit(train_X_clean, train_residuals)

# Generate final predictions
val_forecast = sarimax_model.predict(n_periods=len(val_X), exogenous=val_X[sarimax_exog_cols])
xgb_corrections = best_xgb.predict(val_X)
final_predictions = val_forecast + xgb_corrections

# Evaluation
mae = mean_absolute_error(val_y, final_predictions)
print(f"Optimized MAE: {mae:.2f} liters/kg")

# Visualization
plt.figure(figsize=(14, 6))
plt.plot(val_y.index, val_y, label='Actual Stock')
plt.plot(val_y.index, final_predictions, label='Hybrid Forecast', alpha=0.8)
plt.title('Inventory Prediction: SARIMAX + XGBoost Hybrid Model')
plt.legend()
plt.grid(True)
plt.show()
