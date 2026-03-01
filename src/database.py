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

import sqlite3
import pandas as pd
from pathlib import Path

DB_FILE = Path("./data/whl.db")

def connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_FILE)

def create_tables():
    conn = connect()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS shifts (
        game_id TEXT,
        record_id TEXT,
        home_team TEXT,
        away_team TEXT,
        went_ot INTEGER,
        home_off_line TEXT,
        home_def_pairing TEXT,
        away_off_line TEXT,
        away_def_pairing TEXT,
        home_goalie TEXT,
        away_goalie TEXT,
        toi REAL,
        home_assists INTEGER,
        home_shots INTEGER,
        home_xg REAL,
        home_max_xg REAL,
        home_goals INTEGER,
        away_assists INTEGER,
        away_shots INTEGER,
        away_xg REAL,
        away_max_xg REAL,
        away_goals INTEGER,
        home_penalties_committed INTEGER,
        home_penalty_minutes INTEGER,
        away_penalties_committed INTEGER,
        away_penalty_minutes INTEGER
    )
    """)

    conn.commit()
    conn.close()

def load_csv_to_db(csv_path):
    df = pd.read_csv(csv_path)
    conn = connect()
    df.to_sql("shifts", conn, if_exists="append", index=False)
    conn.close()

