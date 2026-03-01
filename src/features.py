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

import pandas as pd
from .database import connect

def build_line_matchup_features():
    conn = connect()

    df = pd.read_sql("""
        SELECT *
        FROM shifts
        WHERE home_off_line != 'empty_net_line'
        AND away_off_line != 'empty_net_line'
    """, conn)

    grouped = df.groupby([
        "home_team",
        "away_team",
        "home_off_line",
        "away_def_pairing"
    ]).agg({
        "home_xg": "sum",
        "away_xg": "sum",
        "home_goals": "sum",
        "toi": "sum"
    }).reset_index()

    grouped["xg_per_60"] = grouped["home_xg"] / grouped["toi"] * 3600
    grouped["goals_per_60"] = grouped["home_goals"] / grouped["toi"] * 3600

    return grouped
