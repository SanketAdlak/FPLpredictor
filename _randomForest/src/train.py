import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def load_season_data(season_dir):
    """Load data for a specific season."""
    file_path = os.path.join('data', season_dir, 'merged_gw.csv')
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found")
        return None
    
    df = pd.read_csv(file_path)
    df['Season'] = season_dir
    return df

def load_data():
    print("Reading data from all seasons...")
    
    # List of seasons to use for training
    train_seasons = ['2021-22', '2022-23', '2023-24']
    
    # Load and combine data from all training seasons
    all_data = []
    for season in train_seasons:
        season_data = load_season_data(season)
        if season_data is not None:
            print(f"Loaded {len(season_data)} records from {season}")
            all_data.append(season_data)
    
    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal records in combined dataset: {len(df)}")
    print(f"Seasons in combined data: {df['Season'].unique()}")
    
    # Numeric features that don't need preprocessing
    numeric_features = [
        'minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
        'own_goals', 'penalties_missed', 'yellow_cards', 'red_cards', 'saves',
        'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index',
        'value'
    ]
    
    # Create position dummies
    position_dummies = pd.get_dummies(df['position'], prefix='pos')
    
    # Add team difficulty ratings
    team_dummies = pd.get_dummies(df['team'], prefix='team')
    
    # Combine all features
    X = pd.concat([
        df[numeric_features],  # Numeric features
        position_dummies,      # Position encoding
        team_dummies          # Team encoding
    ], axis=1)
    
    y = df['total_points']  # Target variable
    
    print("\nFeature groups:")
    print(f"Numeric features: {numeric_features}")
    print(f"Position features: {position_dummies.columns.tolist()}")
    print(f"Team features: {team_dummies.columns.tolist()}")
    print(f"\nFinal training data shape: X={X.shape}, y={y.shape}")
    
    return X, y

def train_model():
    print("Loading data...")
    X, y = load_data()
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Validation set shape: X_val={X_val.shape}, y_val={y_val.shape}")
    
    # Scale only the numeric features
    scaler = StandardScaler()
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    
    X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_val_scaled[numeric_features] = scaler.transform(X_val[numeric_features])
    
    # Create and train Random Forest model
    print("\nTraining Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on validation set
    val_score = model.score(X_val_scaled, y_val)
    print(f"Validation R² score: {val_score:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    # Create directory for model artifacts
    os.makedirs('models', exist_ok=True)
    
    # Save the model, scaler, and feature information
    print("\nSaving model artifacts...")
    joblib.dump(model, 'models/fantasy_football_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump({
        'all_features': list(X_train.columns),
        'numeric_features': list(numeric_features),
        'categorical_features': list(X_train.select_dtypes(include=['uint8']).columns)
    }, 'models/feature_names.joblib')
    
    print("Training completed! Model artifacts saved in 'models' directory.")

if __name__ == "__main__":
    train_model() 