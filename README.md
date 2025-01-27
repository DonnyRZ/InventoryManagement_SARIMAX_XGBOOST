# **Guide to Running the Inventory Prediction Code in Google Colab**

This guide explains step-by-step how to use the provided Python code in Google Colab to build and evaluate an inventory prediction model using SARIMAX and XGBoost.

---

### **1. Set Up the Environment**

1. **Install Required Libraries:**
   The code uses external libraries such as `optuna`, `pmdarima`, `xgboost`, and others. Install these libraries by running the following commands:
   ```python
   !pip install optuna
   !pip install pmdarima
   ```

2. **Import Libraries:**
   Ensure that all required libraries are imported for data preprocessing, visualization, and modeling:
   ```python
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
   ```

---

### **2. Upload Your Dataset**

1. **Upload an Excel File:**
   Use the file upload feature in Google Colab to upload your dataset. The following code snippet uploads the file and reads the first sheet into a Pandas DataFrame:
   ```python
   uploaded = files.upload()
   df = pd.read_excel(next(iter(uploaded.values())), sheet_name=0)
   ```

2. **Dataset Requirements:**
   - Ensure the dataset contains a `Date` column (in a recognizable datetime format).
   - The target variable for prediction is `Quantity in Stock (liters/kg)`.
   - Ensure other relevant columns, such as product details, sales data, and location information, are present.

---

### **3. Data Preprocessing**

The preprocessing steps include:

1. **Datetime Conversion:**
   Convert the `Date` column to a datetime format and set it as the DataFrame index.
   ```python
   df['Date'] = pd.to_datetime(df['Date'])
   df.set_index('Date', inplace=True)
   ```

2. **Handle Missing Values:**
   Drop rows where the target variable (`Quantity in Stock (liters/kg)`) has missing values.
   ```python
   df.dropna(subset=['Quantity in Stock (liters/kg)'], inplace=True)
   ```

3. **Encode Categorical Variables:**
   Use `OneHotEncoder` to convert categorical columns into numerical features:
   ```python
   encoder = OneHotEncoder(handle_unknown='ignore')
   encoded_df = pd.DataFrame(
       encoder.fit_transform(df[cat_cols]).toarray(),
       columns=encoder.get_feature_names_out(cat_cols),
       index=df.index
   )
   df = pd.concat([df.drop(cat_cols, axis=1), encoded_df], axis=1)
   ```

4. **Feature Engineering:**
   Add time-based features (e.g., day of the week, month, and weekend indicator) and lagged values of `Quantity Sold (liters/kg)` for temporal patterns:
   ```python
   df['day_of_week'] = df.index.dayofweek
   df['month'] = df.index.month
   df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

   for lag in [7, 14, 21]:
       df[f'lag_{lag}'] = df[sold_column].shift(lag)

   df['rolling_mean_7'] = df[sold_column].rolling(7).mean()
   df['rolling_std_14'] = df[sold_column].rolling(14).std()
   ```

5. **Scaling Numerical Features:**
   Scale numerical columns to normalize their values:
   ```python
   df[numerical_cols] = RobustScaler().fit_transform(df[numerical_cols])
   df.dropna(inplace=True)
   ```

---

### **4. Temporal Data Split**

Split the data into training and validation sets:
```python
train_size = int(0.8 * len(df))
train_X, val_X = df.iloc[:train_size].drop(target_column, axis=1), df.iloc[train_size:].drop(target_column, axis=1)
train_y, val_y = df.iloc[:train_size][target_column], df.iloc[train_size:][target_column]
```

---

### **5. Train SARIMAX Model**

1. **Configure SARIMAX:**
   Use `pmdarima.auto_arima` to find the optimal parameters for the SARIMAX model:
   ```python
   sarimax_model = auto_arima(
       train_y,
       exogenous=train_X[sarimax_exog_cols],
       seasonal=True,
       m=7,
       stepwise=True,
       suppress_warnings=True,
       error_action='ignore'
   )
   print(sarimax_model.summary())
   ```

2. **Validate SARIMAX Predictions:**
   Generate predictions on the validation set:
   ```python
   val_forecast = sarimax_model.predict(n_periods=len(val_X), exogenous=val_X[sarimax_exog_cols])
   ```

---

### **6. Optimize XGBoost Residual Model**

1. **Define Objective Function for Hyperparameter Tuning:**
   Optimize XGBoost using `optuna`:
   ```python
   def objective(trial):
       params = {
           'max_depth': trial.suggest_int('max_depth', 3, 9),
           'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
           'n_estimators': trial.suggest_int('n_estimators', 100, 300),
           'subsample': trial.suggest_float('subsample', 0.7, 1.0),
           'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
           'gamma': trial.suggest_float('gamma', 0, 0.5)
       }
       model = xgb.XGBRegressor(**params, random_state=42)
       model.fit(train_X_clean, train_residuals)
       preds = model.predict(val_X_clean)
       return mean_absolute_error(val_residuals, preds)
   ```

2. **Run the Optimization:**
   ```python
   study = optuna.create_study(direction='minimize')
   study.optimize(objective, n_trials=50, timeout=3600)
   ```

---

### **7. Evaluate the Hybrid Model**

1. **Train the Final XGBoost Model:**
   ```python
   best_xgb = xgb.XGBRegressor(**study.best_params, random_state=42)
   best_xgb.fit(train_X_clean, train_residuals)
   ```

2. **Generate Final Predictions:**
   Combine SARIMAX predictions with XGBoost corrections:
   ```python
   final_predictions = val_forecast + best_xgb.predict(val_X)
   ```

3. **Calculate and Display Results:**
   Evaluate the model’s performance using Mean Absolute Error (MAE):
   ```python
   mae = mean_absolute_error(val_y, final_predictions)
   print(f"Optimized MAE: {mae:.2f} liters/kg")
   ```

4. **Visualize the Results:**
   Plot actual vs. predicted values:
   ```python
   plt.figure(figsize=(14, 6))
   plt.plot(val_y.index, val_y, label='Actual Stock')
   plt.plot(val_y.index, final_predictions, label='Hybrid Forecast', alpha=0.8)
   plt.title('Inventory Prediction: SARIMAX + XGBoost Hybrid Model')
   plt.legend()
   plt.grid(True)
   plt.show()
   ```

---

For more specific queries, please email me on donny.landscape@gmail.com

