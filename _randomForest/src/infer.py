import pandas as pd
import numpy as np
import joblib
from pulp import *
import argparse
import os

def load_model_and_data():
    """Load the trained model, scaler, and current season data."""
    print("Loading model artifacts...")
    model = joblib.load('models/fantasy_football_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    feature_info = joblib.load('models/feature_names.joblib')
    
    print("Loading player data...")
    # Try to load from merged_gw.csv first
    if os.path.exists('data/2024-25/merged_gw.csv'):
        df = pd.read_csv('data/2024-25/merged_gw.csv')
    else:
        df = pd.read_csv('data/fantasy_football_data.csv')
    
    print(f"Loaded {len(df)} players")
    return model, scaler, feature_info, df

def prepare_features(df, feature_info):
    """Prepare features for prediction."""
    # Create position dummies
    position_dummies = pd.get_dummies(df['position'], prefix='pos')
    
    # Create team dummies
    team_dummies = pd.get_dummies(df['team'], prefix='team')
    
    # Combine features
    X = pd.concat([
        df[feature_info['numeric_features']],  # Numeric features
        position_dummies,                      # Position encoding
        team_dummies                          # Team encoding
    ], axis=1)
    
    # Ensure all columns from training exist
    for col in feature_info['all_features']:
        if col not in X.columns:
            X[col] = 0
    
    # Reorder columns to match training data
    X = X[feature_info['all_features']]
    
    return X

def select_optimal_team(df, predicted_points, budget=100.0, num_players=15):
    """
    Select optimal team using linear programming.
    
    Args:
        df: DataFrame with player information
        predicted_points: Predicted points for each player
        budget: Total budget in millions
        num_players: Total number of players to select
    """
    # Create optimization problem
    prob = LpProblem("Fantasy_Team_Selection", LpMaximize)
    
    # Create binary variables for each player
    player_vars = LpVariable.dicts("player",
                                 ((i) for i in range(len(df))),
                                 0, 1, LpBinary)
    
    # Objective: Maximize total predicted points
    prob += lpSum([predicted_points[i] * player_vars[i] for i in range(len(df))])
    
    # Constraints
    # 1. Total cost constraint
    prob += lpSum([df['value'].iloc[i]/10 * player_vars[i] for i in range(len(df))]) <= budget
    
    # 2. Position constraints
    gk_indices = [i for i, pos in enumerate(df['position']) if pos == 'GK']
    def_indices = [i for i, pos in enumerate(df['position']) if pos == 'DEF']
    mid_indices = [i for i, pos in enumerate(df['position']) if pos == 'MID']
    fwd_indices = [i for i, pos in enumerate(df['position']) if pos == 'FWD']
    
    prob += lpSum([player_vars[i] for i in gk_indices]) == 2  # 2 goalkeepers
    prob += lpSum([player_vars[i] for i in def_indices]) == 5  # 5 defenders
    prob += lpSum([player_vars[i] for i in mid_indices]) == 5  # 5 midfielders
    prob += lpSum([player_vars[i] for i in fwd_indices]) == 3  # 3 forwards
    
    # 3. Team constraint (max 3 players from same team)
    for team in df['team'].unique():
        team_indices = [i for i, t in enumerate(df['team']) if t == team]
        prob += lpSum([player_vars[i] for i in team_indices]) <= 3
    
    # Solve the problem
    prob.solve()
    
    # Get selected players
    selected_indices = [i for i in range(len(df)) if player_vars[i].value() == 1]
    selected_players = df.iloc[selected_indices].copy()
    selected_players['predicted_points'] = predicted_points[selected_indices]
    
    return selected_players

def suggest_starting_eleven(selected_team):
    """Suggest best starting 11 players based on predicted points."""
    # Must include 1 goalkeeper
    gk = selected_team[selected_team['position'] == 'GK'].nlargest(1, 'predicted_points')
    
    # Get remaining players sorted by predicted points
    outfield = selected_team[selected_team['position'] != 'GK'].sort_values('predicted_points', ascending=False)
    
    # Try different formations (ensuring at least 3 defenders)
    formations = [(3,5,2), (3,4,3), (4,4,2), (4,3,3), (4,5,1), (5,4,1), (5,3,2)]
    best_score = 0
    best_eleven = None
    best_formation = None
    
    for def_count, mid_count, fwd_count in formations:
        # Select players for this formation
        defenders = outfield[outfield['position'] == 'DEF'].head(def_count)
        midfielders = outfield[outfield['position'] == 'MID'].head(mid_count)
        forwards = outfield[outfield['position'] == 'FWD'].head(fwd_count)
        
        # Calculate total predicted points
        total_points = (gk['predicted_points'].sum() +
                       defenders['predicted_points'].sum() +
                       midfielders['predicted_points'].sum() +
                       forwards['predicted_points'].sum())
        
        if total_points > best_score:
            best_score = total_points
            best_eleven = pd.concat([gk, defenders, midfielders, forwards])
            best_formation = (def_count, mid_count, fwd_count)
    
    return best_eleven, best_formation

def main():
    parser = argparse.ArgumentParser(description='Select optimal fantasy football team')
    parser.add_argument('--budget', type=float, default=100.0, help='Total budget in millions')
    parser.add_argument('--players', type=int, default=15, help='Number of players to select')
    args = parser.parse_args()
    
    # Load model and data
    model, scaler, feature_info, df = load_model_and_data()
    
    # Prepare features
    X = prepare_features(df, feature_info)
    
    # Scale features and make predictions
    X_scaled = X.copy()
    X_scaled[feature_info['numeric_features']] = scaler.transform(X[feature_info['numeric_features']])
    predicted_points = model.predict(X_scaled)
    
    print("\nSelecting optimal team...")
    selected_team = select_optimal_team(df, predicted_points, args.budget, args.players)
    
    # Get best starting eleven
    best_eleven, formation = suggest_starting_eleven(selected_team)
    
    # Display results
    total_cost = selected_team['value'].sum() / 10
    total_points = selected_team['predicted_points'].sum()
    
    print("\nOptimal 15-Player Squad:")
    for position in ['GK', 'DEF', 'MID', 'FWD']:
        pos_players = selected_team[selected_team['position'] == position].sort_values('predicted_points', ascending=False)
        print(f"\n{position}:")
        for _, player in pos_players.iterrows():
            price = player['value'] / 10
            print(f"  {player['name']} ({player['team']}) - £{price:.1f}m, Predicted Points: {player['predicted_points']:.1f}")
    
    print(f"\nTotal Squad Cost: £{total_cost:.1f}m")
    print(f"Predicted Total Points: {total_points:.1f}")
    
    print(f"\nRecommended Starting XI (Formation: {formation[0]}-{formation[1]}-{formation[2]}):")
    for position in ['GK', 'DEF', 'MID', 'FWD']:
        pos_players = best_eleven[best_eleven['position'] == position].sort_values('predicted_points', ascending=False)
        if len(pos_players) > 0:
            print(f"\n{position}:")
            for _, player in pos_players.iterrows():
                print(f"  {player['name']} ({player['team']}) - Predicted: {player['predicted_points']:.1f}")
    
    # Save team selections
    selected_team.to_csv('optimal_squad.csv', index=False)
    best_eleven.to_csv('starting_eleven.csv', index=False)
    print("\nTeam selections saved to 'optimal_squad.csv' and 'starting_eleven.csv'")

if __name__ == "__main__":
    main() 