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
import numpy as np
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

    print("\033[32;1m[INFO]\033[0m Raw data loaded successfully")

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

    print("\033[32;1m[INFO]\033[0m Model training complete")
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
        features_home = pd.DataFrame([{
            "home_xg": row["home_xg"],
            "away_xg": row["away_xg"],
            "toi": row["toi"],
            "xg_per_60": row["home_xg"] / row["toi"] * 3600 if row["toi"] > 0 else 0,
            "goals_per_60": 0,
            "elo_home": elo.ratings[row["home_team"]],
            "elo_away": elo.ratings[row["away_team"]],
            "elo_diff": elo.ratings[row["home_team"]] - elo.ratings[row["away_team"]],
        }])

        features_away = pd.DataFrame([{
            "home_xg": row["away_xg"],  # swapped
            "away_xg": row["home_xg"],  # swapped
            "toi": row["toi"],
            "xg_per_60": row["away_xg"] / row["toi"] * 3600 if row["toi"] > 0 else 0,
            "goals_per_60": 0,
            "elo_home": elo.ratings[row["away_team"]],
            "elo_away": elo.ratings[row["home_team"]],
            "elo_diff": elo.ratings[row["away_team"]] - elo.ratings[row["home_team"]],
        }])

        goals_a, goals_b, went_ot = simulator.simulate_game(
            row["home_team"],
            row["away_team"],
            features_home,
            features_away
        )

        results.append((
            row["home_team"],
            row["away_team"],
            goals_a,
            goals_b,
            went_ot
        ))

    return results

def simulate_seasons(df, model_home, model_away, elo, n=1000):
    print(f"\033[32;1m[INFO]\033[0m Simulating {n} season{"s" if n > 1 else ""} (this might take a while)...")

    all_standings = {}

    for i in range(n):
        sim_elo = EloSystem()
        sim_elo.ratings.update(elo.ratings.copy())

        results = simulate_season(df, model_home, model_away, sim_elo)
        standings = compute_standings(results)

        for rank, (team, stats) in enumerate(standings, start=1):
            if team not in all_standings:
                all_standings[team] = []

            all_standings[team].append((rank, stats))

        print(f"  \033[32;1m[INFO]\033[0m Season {i + 1} complete")

    print("\033[32;1m[INFO]\033[0m Season simulation complete")
    return all_standings

def sort_data(results):
    summary = []

    for team, results_list in results.items():
        ranks = []
        points = []
        gf = []
        ga = []
        championships = 0

        for rank, stats in results_list:
            ranks.append(rank)
            points.append(stats["points"])
            gf.append(stats["gf"])
            ga.append(stats["ga"])

            if rank == 1:
                championships += 1

        avg_rank = sum(ranks) / len(ranks)
        avg_points = sum(points) / len(points)
        avg_gf = sum(gf) / len(gf)
        avg_ga = sum(ga) / len(ga)
        win_prob = championships / len(ranks)

        summary.append({
            "team": team,
            "avg_rank": avg_rank,
            "avg_points": avg_points,
            "avg_gf": avg_gf,
            "avg_ga": avg_ga,
            "championship_prob": win_prob
        })

    summary = sorted(summary, key=lambda x: x["avg_rank"])
    return summary

def main():
    try:
        np.random.seed(42)

        load_raw_data()

        df, x, y_home, y_away, elo = prepare_training_data()
        model_home, model_away = train_goal_model(x, y_home, y_away)
        results = simulate_seasons(df, model_home, model_away, elo)

        print("\033[32;1m[INFO]\033[0m Final Standings:")
        summary = sort_data(results)
        for i, team_data in enumerate(summary, start=1):
            print(
                f"{i:2d}. {team_data['team']:<18} "
                f"Avg Rank: {team_data['avg_rank']:.2f}  "
                f"Avg Pts: {team_data['avg_points']:.1f}  "
                f"GF: {team_data['avg_gf']:.1f}  "
                f"GA: {team_data['avg_ga']:.1f}  "
                f"Title%: {team_data['championship_prob']*100:.1f}%"
            )

    except Exception as e:
        print(f"\033[31;1m[ERROR]\033[0m A fatal error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
