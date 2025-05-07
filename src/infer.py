import pandas as pd
import numpy as np
import joblib
import json
import argparse
from pulp import *
import contextlib
import io
import sys
import os
from tabulate import tabulate

def load_model_artifacts():
    """Load the latest model artifacts."""
    print("Loading model artifacts...")
    model = joblib.load('models/model_latest')
    scaler = joblib.load('models/scaler_latest')
    with open('models/info_latest', 'r') as f:
        feature_info = json.load(f)
    return model, scaler, feature_info

def prepare_player_features(player_data, feature_info):
    """Prepare features for a single player or multiple players."""
    # If input is already a DataFrame, use it directly
    if isinstance(player_data, pd.DataFrame):
        df = player_data.copy()
    else:
        # Convert dict or list of dicts to DataFrame
        df = pd.DataFrame(player_data if isinstance(player_data, list) else [player_data])
    
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
        X[team_col] = (df['team_x'] == team).astype(int)
    
    # Ensure all features exist with correct ordering
    return X[feature_info['all_features']]

def select_optimal_team(players_df, predicted_points, budget=100.0, num_players=15):
    """Select optimal team within budget and position constraints."""
    # Create optimization problem
    prob = LpProblem("FantasyTeam", LpMaximize)
    
    # Create binary variables for each player
    player_vars = LpVariable.dicts("player",
                                 ((i) for i in range(len(players_df))),
                                 cat='Binary')
    
    # Objective: Maximize predicted points
    prob += lpSum(predicted_points[i] * player_vars[i]
                 for i in range(len(players_df)))
    
    # Constraint: Total cost within budget
    prob += lpSum(players_df.iloc[i]['now_cost'] * player_vars[i]
                 for i in range(len(players_df))) <= budget * 10  # Convert budget to same units as value
    
    # Constraint: Total number of players
    prob += lpSum(player_vars[i] for i in range(len(players_df))) == num_players
    
    # Position constraints with strict requirements
    position_requirements = {
        'GKP': 2,
        'DEF': 5,
        'MID': 5,
        'FWD': 3
    }
    
    for pos, required_count in position_requirements.items():
        pos_indices = [i for i, p in enumerate(players_df['position']) if p == pos]
        if pos_indices:  # Only add constraint if we have players in this position
            prob += lpSum(player_vars[i] for i in pos_indices) == required_count
    
    # Team constraint: Maximum 3 players from each team
    for team in players_df['team_x'].unique():
        team_indices = [i for i, t in enumerate(players_df['team_x']) if t == team]
        prob += lpSum(player_vars[i] for i in team_indices) <= 3
    
    # Add constraint to prevent duplicate players
    for name in players_df['name'].unique():
        name_indices = [i for i, n in enumerate(players_df['name']) if n == name]
        if len(name_indices) > 1:
            prob += lpSum(player_vars[i] for i in name_indices) <= 1
    
    # Solve the problem with suppressed output
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    status = prob.solve()
    sys.stdout = old_stdout
    
    if status != 1:  # Not optimal
        print("\nWarning: Could not find optimal solution with strict constraints. Trying with relaxed constraints...")
        # Try with relaxed budget constraint
        prob = LpProblem("FantasyTeam_Relaxed", LpMaximize)
        
        # Recreate variables
        player_vars = LpVariable.dicts("player",
                                     ((i) for i in range(len(players_df))),
                                     cat='Binary')
        
        # Objective
        prob += lpSum(predicted_points[i] * player_vars[i]
                     for i in range(len(players_df)))
        
        # Position constraints (must be met)
        for pos, required_count in position_requirements.items():
            pos_indices = [i for i, p in enumerate(players_df['position']) if p == pos]
            if pos_indices:  # Only add constraint if we have players in this position
                prob += lpSum(player_vars[i] for i in pos_indices) == required_count
        
        # Team constraint (relaxed to 4 players)
        for team in players_df['team_x'].unique():
            team_indices = [i for i, t in enumerate(players_df['team_x']) if t == team]
            prob += lpSum(player_vars[i] for i in team_indices) <= 4
        
        # Add constraint to prevent duplicate players
        for name in players_df['name'].unique():
            name_indices = [i for i, n in enumerate(players_df['name']) if n == name]
            if len(name_indices) > 1:
                prob += lpSum(player_vars[i] for i in name_indices) <= 1
        
        # Solve with relaxed constraints
        sys.stdout = open(os.devnull, 'w')
        prob.solve()
        sys.stdout = old_stdout
    
    # Get selected players
    selected_indices = [i for i in range(len(players_df)) if value(player_vars[i]) == 1]
    selected_players = players_df.iloc[selected_indices].copy()
    
    return selected_players

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make fantasy football predictions')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file with player data')
    parser.add_argument('--budget', type=float, default=100.0, help='Total budget in millions')
    parser.add_argument('--optimize', action='store_true', help='Select optimal team')
    parser.add_argument('--num_players', type=int, default=15, help='Number of players to select')
    return parser.parse_args()

def main():
    """Main function to run inference."""
    # Parse command line arguments
    args = parse_args()
    
    # Load model artifacts
    position_models, scaler, feature_info = load_model_artifacts()
    
    # Load player data
    print(f"\nLoading player data from {args.input}...")
    players_df = pd.read_csv(args.input)
    
    # Clean and preprocess the data
    players_df = players_df.drop_duplicates(subset=['name', 'team_x'], keep='first')
    
    # Map position codes
    position_mapping = {
        'GK': 'GKP',
        'DEF': 'DEF',
        'MID': 'MID',
        'FWD': 'FWD'
    }
    players_df['position'] = players_df['position'].map(position_mapping)
    
    # Filter out invalid positions and ensure we have enough players in each position
    valid_positions = ['GKP', 'DEF', 'MID', 'FWD']
    players_df = players_df[players_df['position'].isin(valid_positions)]
    
    # Ensure we have enough players in each position
    min_players = {
        'GKP': 2,
        'DEF': 5,
        'MID': 5,
        'FWD': 3
    }
    
    for pos, count in min_players.items():
        pos_count = len(players_df[players_df['position'] == pos])
        if pos_count < count:
            print(f"Warning: Not enough {pos} players ({pos_count} available, {count} required)")
            return
    
    # Prepare features
    X = prepare_player_features(players_df, feature_info)
    
    # Scale numeric features
    X_scaled = X.copy()
    X_scaled[feature_info['numeric_features']] = scaler.transform(X[feature_info['numeric_features']])
    
    # Make predictions for each position
    predicted_points = np.zeros(len(players_df))  # Initialize with zeros
    for pos in valid_positions:
        pos_mask = X[f'pos_{pos}'] == 1
        if pos_mask.any():
            model = position_models[pos]
            pos_preds = model.predict(X_scaled[pos_mask])
            predicted_points[pos_mask] = pos_preds
    
    if args.optimize:
        # Select optimal team
        selected_players = select_optimal_team(
            players_df,
            predicted_points,
            budget=args.budget,
            num_players=args.num_players
        )
        
        # Prepare data for table display
        table_data = []
        total_cost = 0
        total_points = 0
        
        # Sort players by position for display
        position_order = {'GKP': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
        selected_players['pos_order'] = selected_players['position'].map(position_order)
        selected_players = selected_players.sort_values('pos_order')
        
        for _, player in selected_players.iterrows():
            cost = player['now_cost'] / 10  # Convert to millions
            points = predicted_points[players_df.index.get_loc(player.name)]
            table_data.append([
                player['position'],
                player['name'],
                player['team_x'],
                f"£{cost:.1f}m",
                f"{points:.1f}"
            ])
            total_cost += cost
            total_points += points
        
        # Print the table
        print("\nOptimal FPL Team Selection")
        print("=" * 75)
        print(tabulate(table_data, 
                      headers=['Position', 'Name', 'Team', 'Cost', 'Predicted Points'],
                      tablefmt='grid'))
        
        print("\n" + "=" * 75)
        print(f"Total Cost: £{total_cost:.1f}m")
        print(f"Total Predicted Points: {total_points:.1f}")
        
        # Print team distribution
        print("\nTeam Distribution:")
        print("-" * 20)
        team_dist = selected_players['team_x'].value_counts()
        for team, count in team_dist.items():
            print(f"{team}: {count} players")
    else:
        # Print predictions for all players
        results = pd.DataFrame({
            'name': players_df['name'],
            'position': players_df['position'],
            'team': players_df['team_x'],
            'cost': players_df['now_cost'] / 10,  # Convert to millions
            'predicted_points': predicted_points
        })
        print("\nPredicted points for all players:")
        print(tabulate(results.sort_values('predicted_points', ascending=False),
                      headers='keys',
                      tablefmt='grid',
                      floatfmt='.1f'))

if __name__ == "__main__":
    main() 