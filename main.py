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

import sys
from pathlib import Path
import pandas as pd
from src import *

BASE_DIR = Path(__file__).resolve().parent
DB_FILE = BASE_DIR / "data" / "whl.db"
RAW_DIR = BASE_DIR / "data" / "raw"
WHL_FILE = RAW_DIR / "whl25.csv"
MATCHUPS_FILE = RAW_DIR / "matchups.csv"
DICTIONARY_FILE = RAW_DIR / "dictionary.csv"

def load_raw_data():
    print("\033[32;1m[INFO]\033[0m Creating database tables...")
    if DB_FILE.exists():
        print("\033[33;1m[WARNING]\033[0m Database already exists, deleting...")
        DB_FILE.unlink()
    create_tables()

    if not WHL_FILE.exists():
        print(f"\033[31;1m[ERROR]\033[0m Missing file: {WHL_FILE}")
        sys.exit(1)

    print(f"\033[32;1m[INFO]\033[0m Loading shift data from {WHL_FILE}")
    load_csv_to_db(WHL_FILE)

    print("\033[32;1m[INFO]\033[0m Raw data loaded successfully.")

def prepare_training_data():
    print("\033[32;1m[INFO]\033[0m Building line matchup features...")
    df = build_line_matchup_features()

    if df.empty:
        print("\033[31;1m[ERROR]\033[0m No data returned from the feature builder")
        sys.exit(1)

    df = df[df["toi"] > 0]

    elo = build_elo_from_games(df)
    df["elo_home"] = df["home_team"].map(elo.ratings)
    df["elo_away"] = df["away_team"].map(elo.ratings)
    df["elo_diff"] = df["elo_home"] - df["elo_away"]

    x = df[[
        "home_xg",
        "away_xg",
        "toi",
        "xg_per_60",
        "goals_per_60",
        "elo_home",
        "elo_away",
        "elo_diff"
    ]].copy()
    y_home = df["home_goals"]
    y_away = df["away_goals"]

    print(f"\033[32;1m[INFO]\033[0m Training rows: {len(x)}")
    return df, x, y_home, y_away, elo

def train_goal_model(x, y_home, y_away):
    print("\033[32;1m[INFO]\033[0m Training LightGBM goal model...")

    model_home = LightGBMGoalModel()
    model_home.fit(x, y_home)

    model_away = LightGBMGoalModel()
    model_away.fit(x, y_away)

    print("\033[32;1m[INFO]\033[0m Model training complete.")
    return model_home, model_away

def build_elo_from_games(df):
    print("\033[32;1m[INFO]\033[0m Building ELO ratings from previous games...")

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

    print("\033[32;1m[INFO]\033[0m ELO ratings initialized")
    return elo

def simulate_season(df, model_home, model_away, elo):
    print("\033[32;1m[INFO]\033[0m Simulating season...")

    simulator = SeasonSimulator(model_home, model_away, elo)
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
            "xg_per_60": row["home_xg"] / row["toi"] * 3600 if row["toi"] > 0 else 0,
            "goals_per_60": 0,
            "elo_home": elo.ratings[row["home_team"]],
            "elo_away": elo.ratings[row["away_team"]],
            "elo_diff": elo.ratings[row["home_team"]] - elo.ratings[row["away_team"]],
        }])

        goals_a, goals_b, went_ot = simulator.simulate_game(
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
            went_ot
        ))

    print("\033[32;1m[INFO]\033[0m Season simulation complete")
    return results

def main():
    try:
        load_raw_data()

        df, x, y_home, y_away, elo = prepare_training_data()
        model_home, model_away = train_goal_model(x, y_home, y_away)
        results = simulate_season(df, model_home, model_away, elo)
        standings = compute_standings(results)

        print("\033[32;1m[INFO]\033[0m Final Standings:")
        for rank, (team, stats) in enumerate(standings, start=1):
            print(
                f" {rank:2d}. {team:<18} "
                f" {stats['points']:3d} pts  "
                f" GF: {stats['gf']:3d}  "
                f" GA: {stats['ga']:3d}"
            )

    except Exception as e:
        print(f"\033[31;1m[ERROR]\033[0m A fatal error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
