
# Fantasy Football Prediction System

This system predicts Fantasy Premier League (FPL) player performance using machine learning models trained on historical data. It also selects an optimal 15-player squad using point predictions, following official FPL rules.


##  Overview

The pipeline performs:

1. **Training** position-specific regression models using historical season data.
2. **Evaluation** using standard metrics (MAE, RMSE, R²).
3. **Inference** to predict points for a given season and generate an optimal squad.

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

##  Requirements

- Python 3.8+
- All dependencies are listed in `requirements.txt`


##  requirements.txt

```txt
pandas
numpy
scikit-learn
pyarrow
pulp
```

Install with:

```bash
pip install -r requirements.txt
```

---
### Installation

```bash
git clone https://github.com/SanketAdlak/FPLpredictor.git
cd smai_a2

python -m venv venv
venv\Scripts\activate             # On Windows
# OR
source venv/bin/activate          # On macOS/Linux

pip install -r requirements.txt
````


##  Project Structure

```
src/
├── data/
│   └── 2021-22/
│   └── 2022-23/
│   └── 2023-24/
│   └── 2024-25/
│       ├── players_raw.csv
│       └── merged_gw.csv
├── models/              # Auto-created during training
├── processed/           # Auto-created feature storage
├── train.py             # Train or fine-tune model
├── eval.py              # Evaluate model on test season
├── infer.py             # Predict points and select squad
├── requirements.txt
└── README.md
```


## How to Use

### 1. Train the Model

```bash
python train.py
```

Saves:

* `models/*.joblib` (1 per position)
* `models/config.json`
* `processed/features.parquet`

#### Quick Start (Skip Training)
- Pretrained models are included in the repository.  
- You can skip train.py and directly run eval.py and infer.py

---

### 2. Evaluate Model Performance

```bash
python eval.py
```

#### **Evaluation Results**

```
MAE by position:
  GK: 1.894
 DEF: 2.182
 MID: 1.858
 FWD: 2.587
 ALL: 2.046

MSE by position:
  GK: 6.536
 DEF: 7.267
 MID: 8.156
 FWD: 11.766
 ALL: 8.053

R² by position:
  GK: 0.090
 DEF: 0.001
 MID: 0.095
 FWD: 0.082
 ALL: 0.077
```

---

### 3. Predict and Optimise Squad

```bash
python infer.py data/2024-25/merged_gw.csv \
                --players_raw data/2024-25/players_raw.csv \
                --season 2024-25 \
                --out predictions_2024-25.csv
```

####  **Optimal Squad Output**


|name      |     team     |    position | cost_m | predicted_points | Starting|
| --- | --- | --- | --- | --- | --- |
|Daniel Muñoz  |   Crystal Palace   |      DEF   |   5.2   |      5.61    |     True|
| Diogo Dalot Teixeira    |     Man Utd    |     DEF   |   5.0    |     4.35    |     True|
| Marc Cucurella     |    Chelsea     |    DEF  |    5.4     |    4.68      |   True|
| Nikola Milenković  |  Nott'm Forest    |     DEF   |   5.1    |     4.78     |    True|
|Jørgen Strand Larsen  |        Wolves    |     FWD   |   5.4    |     4.15     |    True|
|     Dean Henderson   |  Crystal Palace  |       GK   |    4.6  |       4.16    |     True|
|Bruno Borges Fernandes  |      Man Utd   |      MID   |   8.6    |     4.77     |    True|
|         Ismaïla Sarr   |  Crystal Palace|        MID  |    5.7  |       4.49   |      True|
|        Jarrod Bowen   |     West Ham   |      MID    |  7.6     |    4.40     |    True|
|        Mohamed Salah   |    Liverpool   |      MID    | 13.8      |   4.90     |    True|
|       Morgan Rogers   |   Aston Villa  |       MID  |    5.6    |     4.57    |     True|
|      Jurriën Timber   |      Arsenal   |      DEF   |   5.6     |    4.29     |    False|
|           Liam Delap  |       Ipswich  |       FWD  |    5.6     |    3.68    |     False|
|          Yoane Wissa  |     Brentford  |       FWD   |   6.6     |    3.82     |    False|
|             Matz Sels |   Nott'm Forest |        GK   |    5.1   |      3.97  |       False|


---
This above output is the prediction for Gameweek 32.  

  


> **Note:** For visualizations and plots (e.g., feature importance, prediction vs. actual), you can use the provided data in a **Jupyter Notebook**.  
> The notebook environment is ideal for exploring results interactively.  
> Before running notebook, uncommnet the packages from requirements.txt and install all of them.

---

## Input Data Format

Each season folder (e.g., `data/2024-25/`) contains:

### 1. `players_raw.csv`

| name       | position | team | now\_cost | element\_type | ... |
| ---------- | -------- | ---- | --------- | ------------- | --- |
| Haaland E. | FWD      | MCI  | 125       | 4             | ... |

### 2. `merged_gw.csv`

| name       | round | minutes | goals\_scored | assists | ... |
| ---------- | ----- | ------- | ------------- | ------- | --- |
| Haaland E. | 1     | 90      | 1             | 0       | ... |



## Model Features

* Rolling stats from previous games (e.g., 5-game window)
* Match stats: goals, assists, minutes, clean sheets
* Advanced stats: BPS, influence, creativity, threat
* Context features: home/away, opponent strength

---

## Team Optimization Rules

* Max budget: £100.0m
* Squad size: 15 players
* Max 3 players per club
* Valid formation enforced

  * 2 GKs, 5 DEFs, 5 MIDs, 3 FWDs
  * At least 3 DEF, 2 MID, 1 FWD in starting XI

* Provides detailed team analysis:
  * Position-wise breakdown
  * Team distribution
  * Value distribution



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



##  Future Enhancements

* Fixture-aware prediction
* Transfer suggestion engine
* Player form tracking
* Web-based UI

