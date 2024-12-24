# IPL First Innings Score Prediction

This project predicts the total score of a team in the first innings of an IPL match using machine learning models. It leverages historical IPL data to train and evaluate prediction models.

## Tech Stack
- **Programming Language**: Python
- **Libraries**:
  - Data Processing: Pandas, NumPy
  - Visualization: Matplotlib, Seaborn
  - Machine Learning: Scikit-learn
  - Models: Linear Regression, Random Forest Regressor

## Features
- Data preprocessing:
  - Filtering consistent teams
  - Encoding categorical variables
  - Dropping irrelevant columns
- Model training and evaluation:
  - Linear Regression
  - Random Forest Regressor
- Visualization of predictions vs actual scores.

## Dataset
The dataset should be a CSV file named `ipl.csv`, containing historical IPL match data with the following key columns:
- `date`: Date of the match
- `bat_team`: Batting team name
- `bowl_team`: Bowling team name
- `overs`: Overs completed
- `runs`: Runs scored
- `wickets`: Wickets lost
- `runs_last_5`: Runs scored in the last 5 overs
- `wickets_last_5`: Wickets lost in the last 5 overs
- `total`: Total score in the innings

Place the dataset in the following path:
```
C:\Users\user\Downloads\New folder\Projects\IPL Score Prediction\ipl.csv
```

## Installation
1. Clone the repository.
2. Install the required Python libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Ensure the dataset is placed in the correct path.

## Usage
1. Run the script:
   ```bash
   python ipl_score_prediction.py
   ```
2. The script will:
   - Preprocess the dataset
   - Train Linear Regression and Random Forest models
   - Evaluate the models
   - Display predictions and visualization

## Example Prediction
To predict the score for a specific scenario, use the following sample code snippet:

```python
# Create a sample input for prediction
new_data = {
    'runs': [50],
    'wickets': [2],
    'overs': [7.3],
    'runs_last_5': [36],
    'wickets_last_5': [1],
    'year': [2023],
    'bat_team_Chennai Super Kings': [1],
    'bowl_team_Mumbai Indians': [1],
    # Include all other team/venue dummies as needed
}

new_data_df = pd.DataFrame(new_data)

# Predict scores
lr_prediction = lr_model.predict(new_data_df)[0]
rf_prediction = rf_model.predict(new_data_df)[0]

print(f"Linear Regression Prediction: {lr_prediction:.2f}")
print(f"Random Forest Prediction: {rf_prediction:.2f}")
```

## Output
- Model evaluation metrics:
  - Mean Absolute Error (MAE)
  - R-squared (R²)
- Visualization of actual vs predicted scores.

## Contributing
Feel free to fork this repository and create pull requests for enhancements or bug fixes.
