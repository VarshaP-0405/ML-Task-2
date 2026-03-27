#Import required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#1.	Load Data: Import CSV using pandas.
df = pd.read_csv("Boston.csv")  
print("Sample Data")
print(df.head())

#2.	Explore: Check missing values, stats (df.describe()), scatter plots or correlation heatmap.
print(df.info())
print(df.isnull().sum())
print(df.describe())
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

#3.	Prepare: Select features (RM, LSTAT, PTRATIO, INDUS, NOX, AGE) and target (MEDV), split train/test (80/20).
X = df[['RM', 'LSTAT', 'PTRATIO', 'INDUS', 'NOX', 'AGE']]
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#4.	Build Model: Use LinearRegression(), fit on training data.
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#5.	Evaluate: Predict, calculate MSE, and R².
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

#6.	Interpret: Check coefficients to see which features influence house prices most, plot predicted vs actual prices.
coeff_df = pd.DataFrame({'Feature': X.columns,'Coefficient': model.coef_})
print(coeff_df)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices (MEDV)")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
plt.show()

