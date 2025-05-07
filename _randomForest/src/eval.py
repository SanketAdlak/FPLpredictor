import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

def load_test_data(test_season='2023-24'):
    """Load test data from a specific season."""
    # Try loading from season directory first
    season_file = os.path.join('data', test_season, 'merged_gw.csv')
    if os.path.exists(season_file):
        df = pd.read_csv(season_file)
        df['Season'] = test_season
    else:
        # If not found, try loading from main CSV
        df = pd.read_csv('data/fantasy_football_data.csv')
        df = df[df['Season'] == test_season]
    
    print(f"Loaded {len(df)} records from {test_season} season")
    
    # Load feature information
    feature_info = joblib.load('models/feature_names.joblib')
    numeric_features = feature_info['numeric_features']
    
    # Create position dummies
    position_dummies = pd.get_dummies(df['position'], prefix='pos')
    
    # Create team dummies
    team_dummies = pd.get_dummies(df['team'], prefix='team')
    
    # Combine features
    X = pd.concat([
        df[numeric_features],  # Numeric features
        position_dummies,      # Position encoding
        team_dummies          # Team encoding
    ], axis=1)
    
    # Ensure all columns from training exist
    for col in feature_info['all_features']:
        if col not in X.columns:
            X[col] = 0
    
    # Reorder columns to match training data
    X = X[feature_info['all_features']]
    
    y = df['total_points']
    
    return X, y, df

def evaluate_model(test_season='2023-24'):
    print(f"Evaluating model on {test_season} season...")
    X_test, y_test, test_data = load_test_data(test_season)
    
    print("\nLoading model and scaler...")
    model = joblib.load('models/fantasy_football_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    feature_info = joblib.load('models/feature_names.joblib')
    
    # Scale only numeric features
    X_test_scaled = X_test.copy()
    X_test_scaled[feature_info['numeric_features']] = scaler.transform(X_test[feature_info['numeric_features']])
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test_scaled)
    
    # Calculate overall metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print evaluation metrics
    print("\nOverall Model Evaluation Metrics:")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Calculate and display metrics by position
    print("\nPerformance by Player Position:")
    positions = test_data['position'].unique()
    position_metrics = []
    for pos in positions:
        pos_mask = test_data['position'] == pos
        pos_rmse = np.sqrt(mean_squared_error(y_test[pos_mask], y_pred[pos_mask]))
        pos_mae = mean_absolute_error(y_test[pos_mask], y_pred[pos_mask])
        pos_r2 = r2_score(y_test[pos_mask], y_pred[pos_mask])
        
        position_metrics.append({
            'position': pos,
            'count': sum(pos_mask),
            'rmse': pos_rmse,
            'mae': pos_mae,
            'r2': pos_r2
        })
        
        print(f"\n{pos} Players (Count: {sum(pos_mask)}):")
        print(f"  RMSE: {pos_rmse:.4f}")
        print(f"  MAE: {pos_mae:.4f}")
        print(f"  R²: {pos_r2:.4f}")
    
    # Calculate metrics by price range
    print("\nPerformance by Price Range:")
    price_ranges = [(0, 5.0), (5.0, 7.0), (7.0, 9.0), (9.0, float('inf'))]
    for min_price, max_price in price_ranges:
        price_mask = (test_data['value'] >= min_price*10) & (test_data['value'] < max_price*10)
        if sum(price_mask) == 0:
            continue
        
        price_rmse = np.sqrt(mean_squared_error(y_test[price_mask], y_pred[price_mask]))
        price_mae = mean_absolute_error(y_test[price_mask], y_pred[price_mask])
        price_r2 = r2_score(y_test[price_mask], y_pred[price_mask])
        
        print(f"\nPrice {min_price}-{max_price}m (Count: {sum(price_mask)}):")
        print(f"  RMSE: {price_rmse:.4f}")
        print(f"  MAE: {price_mae:.4f}")
        print(f"  R²: {price_r2:.4f}")
    
    # Calculate metrics by team
    print("\nPerformance by Team:")
    teams = test_data['team'].unique()
    team_metrics = []
    for team in sorted(teams):
        team_mask = test_data['team'] == team
        if sum(team_mask) < 10:  # Skip teams with too few samples
            continue
            
        team_rmse = np.sqrt(mean_squared_error(y_test[team_mask], y_pred[team_mask]))
        team_mae = mean_absolute_error(y_test[team_mask], y_pred[team_mask])
        team_r2 = r2_score(y_test[team_mask], y_pred[team_mask])
        
        team_metrics.append({
            'team': team,
            'count': sum(team_mask),
            'rmse': team_rmse,
            'mae': team_mae,
            'r2': team_r2
        })
        
        print(f"\n{team} (Count: {sum(team_mask)}):")
        print(f"  RMSE: {team_rmse:.4f}")
        print(f"  MAE: {team_mae:.4f}")
        print(f"  R²: {team_r2:.4f}")
    
    # Save evaluation results
    results = {
        'season': test_season,
        'overall_rmse': rmse,
        'overall_mae': mae,
        'overall_r2': r2,
        'num_samples': len(y_test)
    }
    
    # Add position-wise metrics
    for metrics in position_metrics:
        pos = metrics['position']
        results[f'count_{pos}'] = metrics['count']
        results[f'rmse_{pos}'] = metrics['rmse']
        results[f'mae_{pos}'] = metrics['mae']
        results[f'r2_{pos}'] = metrics['r2']
    
    # Add team-wise metrics
    for metrics in team_metrics:
        team = metrics['team'].replace(' ', '_')
        results[f'count_{team}'] = metrics['count']
        results[f'rmse_{team}'] = metrics['rmse']
        results[f'mae_{team}'] = metrics['mae']
        results[f'r2_{team}'] = metrics['r2']
    
    pd.DataFrame([results]).to_csv('models/evaluation_results.csv', index=False)
    print("\nEvaluation results saved to 'models/evaluation_results.csv'")

if __name__ == "__main__":
    evaluate_model('2023-24') 