#!/usr/bin/env python3
"""
test.py – evaluate the trained models on the *target* season & print metrics.

The script assumes that train.py has already produced:
    models/config.json             (hyper‑parameters, feature list)
    models/[POS]_model.pkl
    models/[POS]_scaler.pkl
    processed/features.parquet     (feature table for all seasons)

Usage:
    python test.py                 # evaluate using defaults from config.json
    python test.py --metric mae    # choose r2 / mse / mae (default mae)
"""
import argparse
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

METRICS = {
    "mae": mean_absolute_error,
    "mse": mean_squared_error,
    "r2" : r2_score,
}

def load_config(model_dir: Path):
    with open(model_dir / "config.json") as fh:
        return json.load(fh)

def main():
    ap = argparse.ArgumentParser(description="Evaluate FPL models on hold‑out season")
    ap.add_argument("--model_dir", default="models", help="Directory produced by train.py")
    ap.add_argument("--processed_dir", default="processed", help="Where features.parquet sits")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    cfg = load_config(model_dir)

    feats = pd.read_parquet(Path(args.processed_dir) / "features.parquet")
    test_df = feats[feats["Season"] == cfg["target_season"]].copy()

    results = {metric: {} for metric in METRICS}

    for pos in ["GK", "DEF", "MID", "FWD"]:
        mdl_path = model_dir / f"{pos}_model.pkl"
        sc_path = model_dir / f"{pos}_scaler.pkl"
        if not mdl_path.exists():
            continue
        model = joblib.load(mdl_path)
        scaler = joblib.load(sc_path)
        mask = test_df["position"] == pos
        if mask.sum() == 0:
            continue
        X = scaler.transform(test_df.loc[mask, cfg["feature_cols"]].fillna(0.0))
        y_true = test_df.loc[mask, "total_points"].to_numpy()
        y_pred = model.predict(X)
        for name, fn in METRICS.items():
            results[name][pos] = fn(y_true, y_pred)

    # Overall metrics
    all_true, all_pred = [], []
    for pos in ["GK", "DEF", "MID", "FWD"]:
        if pos not in results["mae"]:
            continue
        model = joblib.load(model_dir / f"{pos}_model.pkl")
        scaler = joblib.load(model_dir / f"{pos}_scaler.pkl")
        mask = test_df["position"] == pos
        X = scaler.transform(test_df.loc[mask, cfg["feature_cols"]].fillna(0.0))
        y_true = test_df.loc[mask, "total_points"].to_numpy()
        y_pred = model.predict(X)
        all_true.extend(y_true)
        all_pred.extend(y_pred)
    for name, fn in METRICS.items():
        results[name]["ALL"] = fn(np.array(all_true), np.array(all_pred))

    # Print results
    print("\nEvaluation Results\n" + "-" * 40)
    for name in ["mae", "mse", "r2"]:
        print(f"\n{name.upper():>4} by position:")
        for pos in ["GK", "DEF", "MID", "FWD", "ALL"]:
            if pos in results[name]:
                print(f"  {pos:>3}: {results[name][pos]:.3f}")

if __name__ == "__main__":
    main()
