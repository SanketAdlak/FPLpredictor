import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LinearRegression, RidgeCV, LassoCV, ElasticNetCV
)
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import json
from datetime import datetime

def prepare_features(df, numeric_features):
    """Prepare features in a consistent way."""
    # Create position and team dummies
    position_dummies = pd.get_dummies(df['position'], prefix='pos')
    team_dummies = pd.get_dummies(df['team_x'], prefix='team')
    
    # Get all possible feature names
    all_positions = ['DEF', 'FWD', 'GKP', 'MID']
    all_teams = sorted(df['team_x'].unique().tolist())
    
    # Create complete feature names
    position_features = [f'pos_{pos}' for pos in all_positions]
    team_features = [f'team_{team}' for team in all_teams]
    all_features = numeric_features + position_features + team_features
    
    # Initialize feature matrix with zeros
    X = pd.DataFrame(0, index=df.index, columns=all_features)
    
    # Fill numeric features
    for col in numeric_features:
        if col in df.columns:
            X[col] = df[col]
    
    # Fill position dummies
    for pos in all_positions:
        col = f'pos_{pos}'
        X[col] = (df['position'] == pos).astype(int)
    
    # Fill team dummies
    for team in all_teams:
        col = f'team_{team}'
        X[col] = (df['team_x'] == team).astype(int)
    
    feature_info = {
        'numeric_features': numeric_features,
        'positions': position_features,
        'teams': team_features,
        'all_features': all_features
    }
    
    return X, feature_info

def load_data():
    """Load and prepare training data."""
    print("Loading data...")
    
    df = pd.read_csv('data/fantasy_football_data.csv')
    print(f"\nTotal records in dataset: {len(df)}")
    
    # Map position codes
    position_mapping = {
        'GK': 'GKP',
        'DEF': 'DEF',
        'MID': 'MID',
        'FWD': 'FWD'
    }
    df['position'] = df['position'].map(position_mapping)
    
    # Feature selection
    numeric_features = [
        'minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
        'own_goals', 'penalties_missed', 'yellow_cards', 'red_cards', 'saves',
        'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index',
        'now_cost', 'xP', 'expected_assists', 'expected_goal_involvements',
        'expected_goals', 'expected_goals_conceded'
    ]
    
    # Prepare features
    X, feature_info = prepare_features(df, numeric_features)
    
    # Use xP (expected points) as target variable if available, otherwise generate synthetic target
    if 'xP' in df.columns:
        y = df['xP']
    else:
        y = (df['goals_scored'] * 4 + df['assists'] * 3 + df['clean_sheets'] * 4 + 
             df['saves'] * 0.5 + df['bonus'] + df['bps'] * 0.1)
    
    return X, y, feature_info

def train_models():
    """Train multiple models and select the best one."""
    # Load and prepare data
    X, y, feature_info = load_data()
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    
    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    
    X_train_scaled[feature_info['numeric_features']] = scaler.fit_transform(X_train[feature_info['numeric_features']])
    X_val_scaled[feature_info['numeric_features']] = scaler.transform(X_val[feature_info['numeric_features']])
    
    # Train and select model for each position
    position_models = {}
    positions = ['GKP', 'DEF', 'MID', 'FWD']
    
    for pos in positions:
        print(f"\nTraining models for {pos}...")
        # Filter data for this position
        pos_mask = X_train[f'pos_{pos}'] == 1
        X_train_pos = X_train_scaled[pos_mask]
        y_train_pos = y_train[pos_mask]
        
        if len(X_train_pos) < 2:
            print(f"Not enough samples for {pos}, using default Linear Regression")
            model = LinearRegression()
            model.fit(X_train_pos, y_train_pos)
            position_models[pos] = model
            continue
        
        # Define models to try
        models = {
            'Linear': LinearRegression(),
            'Ridge': RidgeCV(alphas=[0.1, 1.0, 10.0], cv=min(3, len(X_train_pos))),
            'Lasso': LassoCV(alphas=[0.1, 1.0, 10.0], cv=min(3, len(X_train_pos)), max_iter=10000),
            'ElasticNet': ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=[0.1, 1.0, 10.0], cv=min(3, len(X_train_pos)), max_iter=10000)
        }
        
        # Train and evaluate models
        best_score = -np.inf
        best_model = None
        best_model_name = None
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train_pos, y_train_pos)
            
            # Make predictions
            train_pred = model.predict(X_train_pos)
            val_mask = X_val[f'pos_{pos}'] == 1
            if val_mask.any():
                val_pred = model.predict(X_val_scaled[val_mask])
                val_r2 = r2_score(y_val[val_mask], val_pred)
            else:
                val_r2 = 0
            
            train_r2 = r2_score(y_train_pos, train_pred)
            
            print(f"Training R²: {train_r2:.4f}")
            print(f"Validation R²: {val_r2:.4f}")
            
            # Track best model (use training R² if no validation data)
            score = val_r2 if val_mask.any() else train_r2
            if score > best_score:
                best_score = score
                best_model = model
                best_model_name = name
        
        if best_model is None:
            print(f"No good model found for {pos}, using default Linear Regression")
            best_model = LinearRegression()
            best_model.fit(X_train_pos, y_train_pos)
            best_model_name = 'Linear'
        
        print(f"\nBest model for {pos}: {best_model_name}")
        position_models[pos] = best_model
    
    # Create timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model artifacts
    os.makedirs('models', exist_ok=True)
    
    model_path = f'models/model_{timestamp}.joblib'
    scaler_path = f'models/scaler_{timestamp}.joblib'
    info_path = f'models/info_{timestamp}.json'
    results_path = f'models/results_{timestamp}.json'
    
    # Save model and scaler
    joblib.dump(position_models, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Save feature information
    with open(info_path, 'w') as f:
        json.dump(feature_info, f, indent=4)
    
    # Save training results
    results = {
        'timestamp': timestamp,
        'feature_info': feature_info,
        'model_info': {pos: str(model) for pos, model in position_models.items()}
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create symlinks to latest versions
    for prefix in ['model', 'scaler', 'info', 'results']:
        latest_link = f'models/{prefix}_latest'
        if os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(f'{prefix}_{timestamp}.joblib' if prefix in ['model', 'scaler'] else f'{prefix}_{timestamp}.json',
                  latest_link)
    
    print("\nModel artifacts saved:")
    print(f"- Model: {model_path}")
    print(f"- Scaler: {scaler_path}")
    print(f"- Feature info: {info_path}")
    print(f"- Results: {results_path}")
    print("\nSymlinks to latest versions created in models/ directory")

if __name__ == "__main__":
    train_models() 