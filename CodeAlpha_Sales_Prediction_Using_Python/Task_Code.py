import pandas as pd

# Load the dataset
df_adv = pd.read_csv('Advertising.csv')

# Inspect the data
print("First 5 rows:")
print(df_adv.head())

print("\nData Info:")
print(df_adv.info())

print("\nSummary Statistics:")
print(df_adv.describe())

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. Cleaning
df_adv = df_adv.drop('Unnamed: 0', axis=1)

# 2. EDA - Correlations
plt.figure(figsize=(10, 8))
sns.heatmap(df_adv.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap: Advertising Channels vs Sales')
plt.savefig('adv_correlation_heatmap.png')

# Pairplot to see relationships
sns.pairplot(df_adv, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=5, aspect=0.8, kind='reg')
plt.savefig('adv_pairplot_regression.png')

# 3. Model Training
X = df_adv[['TV', 'Radio', 'Newspaper']]
y = df_adv['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"RMSE: {rmse:.4f}")
print(f"R-squared: {r2:.4f}")

# Coefficients (Impact Analysis)
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\nImpact Analysis (Coefficients):")
print(coefficients)

# 5. Visualizing Predictions vs Actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.savefig('adv_sales_prediction_scatter.png')