import pandas as pd

# Load the dataset
df_car = pd.read_csv('car data.csv')

# Initial inspection
print("First 5 rows:")
print(df_car.head())

print("\nData Info:")
print(df_car.info())

print("\nCategorical Column Unique Values:")
cat_cols = ['Fuel_Type', 'Selling_type', 'Transmission', 'Owner']
for col in cat_cols:
    print(f"{col}: {df_car[col].unique()}")

print("\nSummary Statistics:")
print(df_car.describe())

import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np

# 1. Feature Engineering
current_year = 2024
df_car['Age'] = current_year - df_car['Year']

# 2. Drop irrelevant columns
# Car_Name has many unique values (98), might lead to overfitting if not handled well. 
# For this task, we will drop it to focus on numeric/categorical features.
df_model = df_car.drop(['Car_Name', 'Year'], axis=1)

# 3. One-Hot Encoding for categorical variables
df_model = pd.get_dummies(df_model, columns=['Fuel_Type', 'Selling_type', 'Transmission'], drop_first=True)

# 4. Split Data
X = df_model.drop('Selling_Price', axis=1)
y = df_model['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 6. Evaluate
y_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared Score: {r2:.4f}")

# 7. Visualizations
# Predicted vs Actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Car Prices')
plt.savefig('car_price_prediction_scatter.png')

# Feature Importance
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
importances.plot(kind='bar')
plt.title('Key Factors Affecting Car Price (Feature Importance)')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('car_feature_importance.png')

# Distribution of Errors
plt.figure(figsize=(10, 6))
sns.histplot(y_test - y_pred, kde=True)
plt.title('Distribution of Prediction Errors (Residuals)')
plt.xlabel('Error (Actual - Predicted)')
plt.savefig('car_price_residuals.png')

print("\nFeature Importances:")
print(importances)