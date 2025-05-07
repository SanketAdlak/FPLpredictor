# Fantasy Football Prediction System

This system predicts fantasy football player performance using machine learning models trained on historical Premier League data.


## Overview

This system uses historical FPL data to:
1. Train position-specific machine learning models for point predictions
2. Evaluate model performance with detailed metrics
3. Select optimal teams within FPL constraints
4. Provide detailed analysis and visualizations

## Requirements

- Python 3.8+
- Required packages listed in `requirements.txt`
- Jupyter Notebook (for data preparation)

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

1. Run the Jupyter notebook to prepare the training data:
```bash
jupyter notebook Data_Preparation_updated.ipynb
```

The notebook will:
- Load historical player data from multiple seasons (2021-22, 2022-23, 2023-24)
- Process team difficulty ratings based on final league positions
- Generate features including:
  - Performance metrics (goals, assists, clean sheets, etc.)
  - Expected points (xP)
  - Team and position encodings
  - Player statistics (bonus points, influence, creativity, threat)
- Create the final dataset `fantasy_football_data.csv`

## Training Models

Run the training script to build position-specific prediction models:

```bash
python src/train.py
```

This will:
1. Load the prepared dataset
2. Split data into training and validation sets
3. Train separate models for each position (GKP, DEF, MID, FWD)
4. Select the best performing model for each position
5. Save model artifacts in the `models/` directory:
   - Trained models (`.joblib`)
   - Feature scalers
   - Feature information
   - Training results

The script automatically creates timestamped versions and symlinks to the latest models.

## Model Features

The system uses the following features for prediction:
- Match statistics (minutes, goals, assists, clean sheets)
- Performance metrics (bonus points, BPS)
- Advanced metrics (influence, creativity, threat, ICT index)
- Expected stats (xG, xA, xGI)
- Team and position encodings
- Player cost

## Features

### 1. Position-Specific Models
- Separate models for each position (GK, DEF, MID, FWD)
- Model selection per position based on validation performance
- Supported models:
  - Linear Regression
  - Ridge Regression (with cross-validation)
  - Lasso Regression (with cross-validation)
  - ElasticNet Regression (with cross-validation)

### 2. Feature Engineering
- Player performance metrics:
  - Match statistics (minutes, goals, assists, etc.)
  - Bonus point indicators (BPS, influence, creativity, threat)
  - Form indicators (rolling averages)
- Team and opposition factors:
  - Team strength indicators
  - Home/away performance
  - Opposition difficulty ratings

### 3. Team Optimization
- Maximizes predicted total points
- Enforces FPL constraints:
  - Budget limit (default: £100m)
  - Squad composition:
    - 2 Goalkeepers
    - 5 Defenders
    - 5 Midfielders
    - 3 Forwards
  - Maximum 3 players per team
- Provides detailed team analysis:
  - Position-wise breakdown
  - Team distribution
  - Value distribution

## Installation

1. Clone the repository:
```bash
git clone <repository_url>
cd fantasy-football-predictor
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install pandas numpy scikit-learn joblib pulp matplotlib seaborn
```

## Data Requirements

### Training Data Format
Each season directory (`data/YYYY-YY/`) should contain:

1. `players_raw.csv`:
```
name,position,team,value,now_cost,element_type,...
```

2. `merged_gw.csv`:
```
name,position,team,minutes,goals_scored,assists,...
```

### Inference Data Format
Input CSV must contain:
```
name,position,team,value,minutes,goals_scored,assists,clean_sheets,
goals_conceded,own_goals,penalties_missed,yellow_cards,red_cards,
saves,bonus,bps,influence,creativity,threat,ict_index
```

## Usage Guide

### 1. Training Models

```bash
python src/train.py
```

Process:
1. Loads historical data from season directories
2. Preprocesses and engineers features
3. Trains position-specific models
4. Performs model selection using validation data
5. Saves model artifacts:
   - `model_latest.joblib`: Position-specific models
   - `scaler_latest.joblib`: Feature scaler
   - `info_latest.json`: Feature information
   - `results_latest.json`: Training results

### 2. Model Evaluation

```bash
python src/eval.py
```

Generates:
1. Overall metrics:
   - Root Mean Square Error (RMSE)
   - Mean Absolute Error (MAE)
   - R² Score

2. Position-wise metrics:
   - Performance breakdown by position
   - Position-specific error analysis

3. Visualizations:
   - Predicted vs Actual points scatter plot
   - Residual analysis plot
   - Feature importance plots

### 3. Making Predictions

```bash
# Get predictions for all players
python src/infer.py --input data/fantasy_football_data.csv

# Select optimal team
python src/infer.py --input data/fantasy_football_data.csv --optimize --budget 100.0
```

Options:
- `--input`: Path to player data CSV
- `--optimize`: Enable team optimization
- `--budget`: Team budget in millions (default: 100.0)
- `--num_players`: Squad size (default: 15)

## Model Performance

Current model performance on example data:

1. Overall Metrics:
   - RMSE: 5.14
   - MAE: 4.29
   - R²: 0.93

2. Position-wise Performance:
   ```
   GKP:
   - RMSE: 3.97
   - MAE: 3.69
   - R²: 0.89

   DEF:
   - RMSE: 4.93
   - MAE: 3.58
   - R²: 0.91

   MID:
   - RMSE: 5.85
   - MAE: 5.10
   - R²: 0.90

   FWD:
   - RMSE: 5.59
   - MAE: 4.80
   - R²: 0.88
   ```

## Best Practices

1. Data Preparation:
   - Use recent seasons for training (e.g., last 3 seasons)
   - Ensure consistent team and player names
   - Handle missing values appropriately

2. Model Training:
   - Regular retraining with new data
   - Monitor position-wise performance
   - Validate feature importance

3. Team Selection:
   - Consider upcoming fixtures
   - Balance team value distribution
   - Account for team rotation

## Limitations

1. Model Assumptions:
   - Past performance indicates future results
   - Linear relationships between features
   - Independent feature contributions

2. Data Limitations:
   - Limited historical data
   - Player transfers between seasons
   - Team strategy changes

3. External Factors:
   - Injuries and suspensions
   - Team formation changes
   - Weather conditions

## Future Improvements

1. Model Enhancements:
   - Deep learning models
   - Ensemble methods
   - Time series features

2. Additional Features:
   - Fixture difficulty prediction
   - Player price change prediction
   - Transfer suggestion system

3. UI/UX:
   - Web interface
   - Real-time updates
   - Team visualization
