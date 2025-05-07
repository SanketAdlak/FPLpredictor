# Import required libraries
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_model_artifacts():
    """Load the latest model artifacts."""
    print("Loading model artifacts...")
    model = joblib.load('models/model_latest')
    scaler = joblib.load('models/scaler_latest')
    with open('models/info_latest', 'r') as f:
        feature_info = json.load(f)
    return model, scaler, feature_info

def prepare_features(df, feature_info):
    """Prepare features for evaluation."""
    # Initialize feature matrix with zeros
    X = pd.DataFrame(0, index=df.index, columns=feature_info['all_features'])
    
    # Fill numeric features
    for col in feature_info['numeric_features']:
        if col in df.columns:
            X[col] = df[col]
    
    # Fill position dummies
    for pos_col in feature_info['positions']:
        pos = pos_col.replace('pos_', '')
        X[pos_col] = (df['position'] == pos).astype(int)
    
    # Fill team dummies
    for team_col in feature_info['teams']:
        team = team_col.replace('team_', '')
        X[team_col] = (df['team'] == team).astype(int)
    
    return X

def evaluate_model():
    """Evaluate model performance on test data."""
    # Load model artifacts
    position_models, scaler, feature_info = load_model_artifacts()
    
    # Load test data
    print("\nLoading test data...")
    df = pd.read_csv('data/fantasy_football_data.csv')
    
    # Prepare features
    X = prepare_features(df, feature_info)
    
    # Generate synthetic target variable for demonstration
    y_true = (df['goals_scored'] * 4 + df['assists'] * 3 + df['clean_sheets'] * 4 + 
              df['saves'] * 0.5 + df['bonus'] + df['bps'] * 0.1)
    
    # Scale features
    X_scaled = X.copy()
    X_scaled[feature_info['numeric_features']] = scaler.transform(X[feature_info['numeric_features']])
    
    # Make predictions for each position
    predictions = []
    for pos in ['GKP', 'DEF', 'MID', 'FWD']:
        pos_mask = X[f'pos_{pos}'] == 1
        if pos_mask.any():
            model = position_models[pos]
            pos_preds = model.predict(X_scaled[pos_mask])
            predictions.extend(pos_preds)
    
    y_pred = np.array(predictions)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print("\nOverall Performance Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Calculate position-wise metrics
    print("\nPosition-wise Performance:")
    for pos in ['GKP', 'DEF', 'MID', 'FWD']:
        pos_mask = df['position'] == pos
        if pos_mask.any():
            pos_rmse = np.sqrt(mean_squared_error(y_true[pos_mask], y_pred[pos_mask]))
            pos_mae = mean_absolute_error(y_true[pos_mask], y_pred[pos_mask])
            pos_r2 = r2_score(y_true[pos_mask], y_pred[pos_mask])
            print(f"\n{pos}:")
            print(f"RMSE: {pos_rmse:.4f}")
            print(f"MAE: {pos_mae:.4f}")
            print(f"R²: {pos_r2:.4f}")
    
    # Create evaluation directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = f'models/eval_{timestamp}'
    os.makedirs(eval_dir, exist_ok=True)
    
    # Save results
    results = {
        'timestamp': timestamp,
        'overall_metrics': {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        },
        'position_metrics': {}
    }
    
    for pos in ['GKP', 'DEF', 'MID', 'FWD']:
        pos_mask = df['position'] == pos
        if pos_mask.any():
            results['position_metrics'][pos] = {
                'rmse': float(np.sqrt(mean_squared_error(y_true[pos_mask], y_pred[pos_mask]))),
                'mae': float(mean_absolute_error(y_true[pos_mask], y_pred[pos_mask])),
                'r2': float(r2_score(y_true[pos_mask], y_pred[pos_mask]))
            }
    
    with open(f'{eval_dir}/metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create plots
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('True Points')
    plt.ylabel('Predicted Points')
    plt.title('Predicted vs Actual Points')
    plt.savefig(f'{eval_dir}/predictions_vs_actual.png')
    plt.close()
    
    # Create residual plot
    residuals = y_pred - y_true
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Points')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.savefig(f'{eval_dir}/residuals.png')
    plt.close()
    
    # Create symlink to latest evaluation
    latest_link = 'models/eval_latest'
    if os.path.exists(latest_link):
        os.remove(latest_link)
    os.symlink(f'eval_{timestamp}', latest_link)
    
    print(f"\nEvaluation results and plots saved to {eval_dir}")

if __name__ == "__main__":
    evaluate_model() 