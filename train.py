#!/usr/bin/env python3
"""
train.py – prepare data, train position‑specific regression models, and
save all artefacts needed by test.py & infer.py.

Usage (from repo root):
    python train.py                # uses default config
    python train.py --config cfg.json

Outputs (relative to repo root):
    models/⟨POS⟩_model.pkl      – trained estimator for each position (GK/DEF/MID/FWD)
    models/⟨POS⟩_scaler.pkl     – fitted StandardScaler for the same features
    models/config.json           – config actually used (so test/infer stay in sync)
    processed/features.parquet   – engineered feature table (optional, speeds‑up test)
"""

import argparse
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

###########################################################################
# -----------------------------  CONFIG  ---------------------------------#
###########################################################################
DEFAULT_CONFIG = {
    "train_seasons": ["2021-22", "2022-23", "2023-24"],
    "target_season": "2024-25",
    "n_prev_games": 5,
    "data_root": "data",       # expected folder layout: data/<season>/<csvs>
    "model_dir": "models",
    "processed_dir": "processed"
}

# Premier‑league style difficulty mapping   (ID → difficulty 1‑5)
TEAM_DIFFICULTY = {
    1: 4,  2: 3,  3: 2,  4: 3,  5: 2,
    6: 4,  7: 3,  8: 2,  9: 2, 10: 2,
    11: 2, 12: 5, 13: 5, 14: 4, 15: 4,
    16: 3, 17: 2, 18: 2, 19: 4, 20: 3,
}
HOME_ADV = 0.8   # opponent easier when *they* are away
AWAY_ADV = 1.2   # opponent harder when *they* are at home

###########################################################################
# -----------------------  HELPER  FUNCTIONS  ----------------------------#
###########################################################################

def load_season_data(season: str, data_root: str) -> pd.DataFrame:
    """Return merged_gw with basic player meta for a season."""
    p_csv = Path(data_root) / season / "players_raw.csv"
    gw_csv = Path(data_root) / season / "merged_gw.csv"
    players = pd.read_csv(p_csv)
    gws = pd.read_csv(gw_csv)

    # build common name field in players
    if {"first_name", "second_name"}.issubset(players.columns):
        players["name"] = players["first_name"].str.strip() + " " + players["second_name"].str.strip()
    elif "web_name" in players.columns:
        players["name"] = players["web_name"].str.strip()
    else:
        players["name"] = players["name"].str.strip()

    # clean names in gw
    if "name" in gws.columns:
        gws["name"] = (gws["name"].astype(str)
                          .str.replace("_", " ")
                          .str.replace(r"\d+", "", regex=True)
                          .str.strip())

    # derive position if missing in gw
    if "position" not in gws.columns:
        if "element_type" in players.columns:
            players["position"] = players["element_type"].map({1: "GK", 2: "DEF", 3: "MID", 4: "FWD"})

    meta_cols = [c for c in ["id", "name", "position", "now_cost", "team"] if c in players.columns]
    merged = gws.merge(players[meta_cols], on="name", how="left")
    merged["Season"] = season
    return merged


def adjusted_difficulty(row) -> float:
    """Return opponent difficulty taking home/away into account."""
    base = TEAM_DIFFICULTY.get(int(row["opponent_team"]), 3)
    return base * (HOME_ADV if row.get("was_home", False) else AWAY_ADV)


def build_feature_frame(df_all: pd.DataFrame, n_prev: int) -> pd.DataFrame:
    """Rolling‑window feature engineering – returns new DF ready for modelling."""
    stats = [
        "minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded",
        "own_goals", "penalties_saved", "penalties_missed", "yellow_cards",
        "red_cards", "saves", "bonus", "bps", "total_points",
    ]

    out_records = []
    grouped = df_all.groupby(["Season", "name"], sort=False)
    for (_, _), grp in grouped:
        grp = grp.sort_values("round").reset_index(drop=True)
        for i in range(n_prev, len(grp)):
            cur = grp.iloc[i].copy()
            window = grp.iloc[i - n_prev:i]
            for s in stats:
                cur[f"prev{n_prev}_{s}"] = window[s].sum()
            cur["home_game"] = int(cur.get("was_home", False))
            cur["opp_difficulty"] = adjusted_difficulty(cur)
            out_records.append(cur)

    feats = pd.DataFrame(out_records)
    minute_col = f"prev{n_prev}_minutes"
    feats = feats[feats[minute_col] >= 90]  # ensure player played ≥ 90′ in window
    return feats.reset_index(drop=True)

###########################################################################
# ----------------------------  TRAIN  -----------------------------------#
###########################################################################

def train_models(train_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Train + return dict {position: (estimator, scaler)}"""
    positions = ["GK", "DEF", "MID", "FWD"]
    models = {}
    cand = {
        "Linear": LinearRegression(),
        "Ridge": RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5),
        "Lasso": LassoCV(alphas=[0.1, 1.0, 10.0], cv=5, max_iter=10000),
        "Elastic": ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], alphas=[0.1, 1.0, 10.0], cv=5,
                                 max_iter=10000),
    }
    for pos in positions:
        subset = train_df[train_df["position"] == pos]
        if subset.empty:
            continue
        X = subset[feature_cols].fillna(0.0).to_numpy()
        y = subset["total_points"].to_numpy()
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)

        best_mse, best_name, best_model = float("inf"), None, None
        for name, mdl in cand.items():
            mse = -cross_val_score(mdl, Xs, y, cv=5, scoring="neg_mean_squared_error").mean()
            if mse < best_mse:
                best_mse, best_name, best_model = mse, name, mdl
        best_model.fit(Xs, y)
        print(f"{pos:<3}  →  {best_name:8s}  CV‑MSE={best_mse:6.2f}")
        models[pos] = (best_model, scaler)
    return models

###########################################################################
# ---------------------------  MAIN CLI  ---------------------------------#
###########################################################################

def main():
    ap = argparse.ArgumentParser(description="Train FPL prediction models")
    ap.add_argument("--config", type=str, help="Path to JSON with overrides", default=None)
    args = ap.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    if args.config and Path(args.config).is_file():
        cfg.update(json.load(open(args.config)))

    Path(cfg["model_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["processed_dir"]).mkdir(parents=True, exist_ok=True)

    # ------------------ DATA PREP ------------------ #
    seasons = cfg["train_seasons"] + [cfg["target_season"]]
    frames = [load_season_data(s, cfg["data_root"]) for s in seasons]
    df_all = pd.concat(frames, ignore_index=True)

    features = build_feature_frame(df_all, cfg["n_prev_games"])
    features.to_parquet(Path(cfg["processed_dir"]) / "features.parquet", index=False)

    # ------------------ TRAIN ---------------------- #
    feature_cols = [c for c in features.columns if c.startswith(f"prev{cfg['n_prev_games']}_")]
    feature_cols += ["home_game", "opp_difficulty"]

    train_df = features[features["Season"].isin(cfg["train_seasons"])]
    models = train_models(train_df, feature_cols)

    # ------------------ SAVE ----------------------- #
    for pos, (mdl, sc) in models.items():
        joblib.dump(mdl, Path(cfg["model_dir"]) / f"{pos}_model.pkl")
        joblib.dump(sc,  Path(cfg["model_dir"]) / f"{pos}_scaler.pkl")
    cfg["feature_cols"] = feature_cols
    with open(Path(cfg["model_dir"]) / "config.json", "w") as fh:
        json.dump(cfg, fh, indent=2)
    print("\n✔ Training complete. Models stored in", cfg["model_dir"])


if __name__ == "__main__":
    main()
