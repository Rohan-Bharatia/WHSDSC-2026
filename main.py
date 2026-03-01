# MIT License
#
# Copyright (c) 2026 Rohan Bharatia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from src import *

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
WHL_FILE = RAW_DIR / "whl25.csv"
MATCHUPS_FILE = RAW_DIR / "matchups.csv"
DICTIONARY_FILE = RAW_DIR / "dictionary.csv"

def load_raw_data():
    info("Creating database tables...")
    create_tables()

    if not WHL_FILE.exists():
        error(f"Missing file: {WHL_FILE}")

    info(f"Loading shift data from {WHL_FILE}")
    load_csv_to_db(WHL_FILE)

    info("Raw data loaded successfully.")

def prepare_training_data():
    info("Building line matchup features...")
    df = build_line_matchup_features()

    if df.empty:
        error("No data returned from the feature builder")

    df = df[df["toi"] > 0]
    x = df[["home_xg", "away_xg", "toi", "xg_per_60"]].copy()
    y = df["xg_per_60"]

    info(f"Training rows: {len(x)}")
    return df, x, y

def train_goal_model(x, y):
    info("Training LightGBM goal model...")

    model = LightGBMGoalModel()
    model.fit(x, y)

    info("Model training complete.")
    return model

def build_elo_from_games(df):
    info("Building ELO ratings from previous games...")

    elo = EloSystem()
    games = (
        df.groupby(["home_team", "away_team"])
        .agg({
            "home_xg": "sum",
            "away_xg": "sum"
        })
        .reset_index()
    )

    for _, row in games.iterrows():
        elo.update(
            row["home_team"],
            row["away_team"],
            row["home_xg"],
            row["away_xg"]
        )

    info("ELO ratings initialized")
    return elo

def simulate_season(df, model, elo):
    info("Simulating season...")

    simulator = SeasonSimulator(model, elo)
    results = []
    games = (
        df.groupby(["home_team", "away_team"])
        .agg({
            "home_xg": "sum",
            "away_xg": "sum",
            "toi": "sum"
        })
        .reset_index()
    )

    for _, row in games.iterrows():
        features = pd.DataFrame([{
            "home_xg": row["home_xg"],
            "away_xg": row["away_xg"],
            "toi": row["toi"],
            "xg_per_60": row["home_xg"] / row["toi"] * 3600 if row["toi"] > 0 else 0
        }])

        goals_a, goals_b = simulator.simulate_game(
            row["home_team"],
            row["away_team"],
            features,
            features
        )

        results.append((
            row["home_team"],
            row["away_team"],
            goals_a,
            goals_b,
            False # TODO: Add overtime tracking
        ))

    info("Season simulation complete")
    return results

def main():
    try:
        load_raw_data()

        df, x, y = prepare_training_data()
        model = train_goal_model(x, y)
        elo = elo = build_elo_from_games(df)
        results = simulate_season(df, model, elo)
        standings = compute_standings(results)

        info("Final Standings")
        print("-" * 70)
        for rank, (team, stats) in enumerate(standings, start=1):
            print(
                f"{rank:2d}. {team:<22} "
                f"{stats['points']:3d} pts  "
                f"GF: {stats['gf']:3d}  "
                f"GA: {stats['ga']:3d}"
            )
        print("-" * 70)

    except Exception:
        raise

if __name__ == "__main__":
    main()
