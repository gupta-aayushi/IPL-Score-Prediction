import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os

# Loading the dataset
file_path = 'C:\\Users\\user\\Downloads\\New folder\\Projects\\IPL Score Prediction\\ipl.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file '{file_path}' does not exist. Please ensure it is in the correct directory.")

data = pd.read_csv(file_path)

# Displaying basic information about the dataset
print("Dataset Shape:", data.shape)
data.head()

### Step 1: Data Exploration

# Checking for null values
print("Missing Values:")
print(data.isnull().sum())

# Summary statistics
print("\nSummary Statistics:")
print(data.describe())

### Step 2: Data Preprocessing

# Filter teams to ensure consistency
data = data[(data['bat_team'].isin([
    'Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
    'Delhi Daredevils', 'Sunrisers Hyderabad'
])) &
(data['bowl_team'].isin([
    'Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
    'Delhi Daredevils', 'Sunrisers Hyderabad'
]))]

# Removing unwanted columns
data = data[data['overs'] >= 5]

# Dropping irrelevant columns
columns_to_drop = ['batsman', 'bowler', 'striker', 'non-striker']  # Drop player-specific columns
data.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Ensure 'date' column is processed correctly
if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year  # Extract year
    data.drop('date', axis=1, inplace=True)  # Drop the original date column

# Encoding categorical variables
categorical_columns = ['bat_team', 'bowl_team', 'venue']  # Include all potential categorical columns
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Ensure all columns are numeric
assert data.select_dtypes(include=['object']).empty, "Non-numeric columns still exist in the dataset."

### Step 3: Splitting Data

# Features and target
target = 'total'
if target not in data.columns:
    raise KeyError(f"Target column '{target}' is missing from the dataset.")

X = data.drop(columns=[target])
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Step 4: Model Training

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

### Step 5: Evaluation

# Predictions
lr_preds = lr_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

# Metrics
print("\nLinear Regression:")
print("MAE:", mean_absolute_error(y_test, lr_preds))
print("R2 Score:", r2_score(y_test, lr_preds))

print("\nRandom Forest:")
print("MAE:", mean_absolute_error(y_test, rf_preds))
print("R2 Score:", r2_score(y_test, rf_preds))

### Step 6: Visualization

# Comparison of actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, lr_preds, alpha=0.5, label='Linear Regression')
plt.scatter(y_test, rf_preds, alpha=0.5, label='Random Forest', color='red')
plt.plot([0, max(y_test)], [0, max(y_test)], '--', color='gray', label='Ideal Fit')
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.legend()
plt.title('Actual vs Predicted Scores')
plt.show()
