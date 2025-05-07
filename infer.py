#!/usr/bin/env python3
"""
Updated **infer.py** – now robust to different column names (`team_x`, `team_id`,
`value`, etc.) and prints a full optimal squad table.

Run example
-----------
    python infer.py data/2024-25/merged_gw.csv \
                 --players_raw data/2024-25/players_raw.csv \
                 --season 2024-25 \
                 --out predictions_2024-25.csv

Dependencies: `pip install pyarrow pulp`
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, PULP_CBC_CMD

##############################################################################
# --------------------- helper functions & constants -----------------------#
##############################################################################
TEAM_DIFFICULTY = {1: 4, 2: 3, 3: 2, 4: 3, 5: 2, 6: 4, 7: 3, 8: 2, 9: 2, 10: 2,
                   11: 2, 12: 5, 13: 5, 14: 4, 15: 4, 16: 3, 17: 2, 18: 2, 19: 4, 20: 3}
HOME_ADV, AWAY_ADV = 0.8, 1.2


def adjusted_difficulty(row):
    base = TEAM_DIFFICULTY.get(int(row["opponent_team"]), 3)
    return base * (HOME_ADV if row.get("was_home", False) else AWAY_ADV)


def merge_meta(gw: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Merge basic player meta into merged_gw rows."""
    if {"first_name", "second_name"}.issubset(players.columns):
        players["name"] = players["first_name"].str.strip() + " " + players["second_name"].str.strip()
    elif "web_name" in players.columns:
        players["name"] = players["web_name"].str.strip()

    meta_cols = [c for c in ["id", "name", "position", "now_cost", "team"] if c in players.columns]
    out = gw.merge(players[meta_cols], on="name", how="left", suffixes=("", "_ply"))
    return out


def build_recent_window(df: pd.DataFrame, n_prev: int) -> pd.DataFrame:
    stats = [
        "minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded",
        "own_goals", "penalties_saved", "penalties_missed", "yellow_cards",
        "red_cards", "saves", "bonus", "bps", "total_points",
    ]
    rows = []
    for _, grp in df.groupby("name", sort=False):
        grp = grp.sort_values("round").reset_index(drop=True)
        if len(grp) < n_prev:
            continue
        cur = grp.iloc[-1].copy()
        window = grp.iloc[-n_prev:]
        for s in stats:
            cur[f"prev{n_prev}_{s}"] = window[s].sum()
        cur["home_game"] = int(cur.get("was_home", False))
        cur["opp_difficulty"] = adjusted_difficulty(cur)
        rows.append(cur)
    return pd.DataFrame(rows)


##############################################################################
# --------------------------  squad optimiser  ---------------------------- #
##############################################################################

def ensure_team_col(df: pd.DataFrame) -> pd.DataFrame:
    if "team" not in df.columns:
        for alt in ["team_x", "team_id", "team_ply", "team_y"]:
            if alt in df.columns:
                df["team"] = df[alt]
                break
    if "team" not in df.columns:
        raise KeyError("Could not find a team column in the input DataFrame.")
    return df


def ensure_cost_col(df: pd.DataFrame) -> pd.DataFrame:
    if "now_cost" not in df.columns:
        if "value" in df.columns:
            df["now_cost"] = df["value"]            # Vaastav merged_gw uses 'value'
        elif "value_tenths" in df.columns:
            df["now_cost"] = df["value_tenths"]
    if "now_cost" not in df.columns:
        raise KeyError("Cannot find player cost column (now_cost/value) in data.")
    df["cost_m"] = df["now_cost"] / 10.0
    return df


def optimise_squad(players: pd.DataFrame, bench_weight: float = 0.1):
    players = players.reset_index(drop=True).copy()
    players = ensure_team_col(players)
    players = ensure_cost_col(players)

    N = len(players)
    x = [LpVariable(f"x_{i}", cat="Binary") for i in range(N)]
    y = [LpVariable(f"y_{i}", cat="Binary") for i in range(N)]

    prob = LpProblem("FPL_Team_Selection", LpMaximize)
    prob += lpSum(players.loc[i, "predicted_points"] * (y[i] + bench_weight * (x[i] - y[i])) for i in range(N))

    prob += lpSum(x) == 15
    prob += lpSum(y) == 11

    pos = players["position"]
    prob += lpSum(x[i] for i in range(N) if pos[i] == "GK") == 2
    prob += lpSum(x[i] for i in range(N) if pos[i] == "DEF") == 5
    prob += lpSum(x[i] for i in range(N) if pos[i] == "MID") == 5
    prob += lpSum(x[i] for i in range(N) if pos[i] == "FWD") == 3

    prob += lpSum(y[i] for i in range(N) if pos[i] == "GK") == 1
    prob += lpSum(y[i] for i in range(N) if pos[i] == "DEF") >= 3
    prob += lpSum(y[i] for i in range(N) if pos[i] == "MID") >= 2
    prob += lpSum(y[i] for i in range(N) if pos[i] == "FWD") >= 1

    teams = players["team"]
    for t in teams.unique():
        prob += lpSum(x[i] for i in range(N) if teams[i] == t) <= 3

    prob += lpSum(players.loc[i, "cost_m"] * x[i] for i in range(N)) <= 100

    for i in range(N):
        prob += y[i] <= x[i]

    prob.solve(PULP_CBC_CMD(msg=False))

    sel = [i for i in range(N) if x[i].value() == 1]
    starts = [i for i in sel if y[i].value() == 1]

    squad = players.loc[sel].copy()
    squad["Starting"] = squad.index.isin(starts)
    return squad.sort_values(["Starting", "position"], ascending=[False, True])


##############################################################################
# --------------------------------  main  ----------------------------------#
##############################################################################

def main():
    ap = argparse.ArgumentParser(description="Predict points & pick squad for a season")
    ap.add_argument("merged_gw", help="merged_gw.csv path")
    ap.add_argument("--players_raw", required=True, help="players_raw.csv path")
    ap.add_argument("--season", required=True, help="Season label, e.g. 2024-25")
    ap.add_argument("--out", default="predictions.csv", help="CSV to save predictions")
    ap.add_argument("--model_dir", default="models", help="Where trained models live")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    cfg = json.load(open(model_dir / "config.json"))
    n_prev = cfg["n_prev_games"]
    feature_cols = cfg["feature_cols"]

    # Load models
    models = {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        m_path, s_path = model_dir / f"{pos}_model.pkl", model_dir / f"{pos}_scaler.pkl"
        if m_path.exists():
            models[pos] = (joblib.load(m_path), joblib.load(s_path))

    # Data prep
    gw = pd.read_csv(args.merged_gw)
    players_raw = pd.read_csv(args.players_raw)
    df = merge_meta(gw, players_raw)
    df["Season"] = args.season

    feats = build_recent_window(df, n_prev)
    if feats.empty:
        raise SystemExit("No player has ≥ n_prev_games history – cannot predict.")

    # Predict
    feats["predicted_points"] = 0.0
    for pos, (mdl, sc) in models.items():
        msk = feats["position"] == pos
        if msk.any():
            X = sc.transform(feats.loc[msk, feature_cols].fillna(0.0))
            feats.loc[msk, "predicted_points"] = mdl.predict(X)

    feats.to_csv(args.out, index=False)
    print(f"✔ Predictions saved → {args.out}  (rows={len(feats)})")

    # Optimal squad (based on latest gw per player)
    latest = feats.sort_values("round").groupby("name", as_index=False).last()
    squad = optimise_squad(latest)

    print("\n================  Optimal Squad  ================")
    print(squad[["name", "team", "position", "cost_m", "predicted_points", "Starting"]]
              .to_string(index=False, formatters={"cost_m": "{:.1f}".format,
                                                 "predicted_points": "{:.2f}".format}))


if __name__ == "__main__":
    main()
