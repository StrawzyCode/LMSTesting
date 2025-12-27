from __future__ import annotations
import re
import numpy as np
import pandas as pd

TEAM_COLS = ["Home Team", "Away Team"]
ROUND_COL = "Round Number"

def load_fixture_results_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    # Standardise columns
    df = df.rename(columns={
        "Round Number": "Round",
        "Home Team": "HomeTeam",
        "Away Team": "AwayTeam",
        "Date": "DateTime",
        "Result": "Result",
    })
    # Round numeric
    df["Round"] = pd.to_numeric(df["Round"], errors="coerce").astype("Int64")

    # Parse datetime (your example: "15/08/2025 20:00")
    df["DateTime"] = pd.to_datetime(df["DateTime"], dayfirst=True, errors="coerce")

    # Parse result like "4 - 2"
    def parse_score(s):
        if not isinstance(s, str):
            return (np.nan, np.nan)
        m = re.search(r"(\d+)\s*-\s*(\d+)", s)
        if not m:
            return (np.nan, np.nan)
        return (int(m.group(1)), int(m.group(2)))

    scores = df["Result"].apply(parse_score)
    df["FTHG"] = scores.apply(lambda x: x[0])
    df["FTAG"] = scores.apply(lambda x: x[1])

    # FTR: H/A/D
    def ftr(row):
        if not np.isfinite(row["FTHG"]) or not np.isfinite(row["FTAG"]):
            return np.nan
        if row["FTHG"] > row["FTAG"]:
            return "H"
        if row["FTHG"] < row["FTAG"]:
            return "A"
        return "D"

    df["FTR"] = df.apply(ftr, axis=1)
    return df

def list_teams(df: pd.DataFrame) -> list[str]:
    return sorted(set(df["HomeTeam"].dropna().unique()).union(set(df["AwayTeam"].dropna().unique())))

def build_round_outcome_matrix(df: pd.DataFrame, teams: list[str]) -> np.ndarray:
    """
    R[round_idx, team_idx] = True if team won in that round, else False if played and didn't win, else nan if didn't play
    """
    max_round = int(df["Round"].max())
    R = np.full((max_round, len(teams)), np.nan, dtype=object)

    for _, row in df.iterrows():
        r = int(row["Round"]) - 1
        h = row["HomeTeam"]; a = row["AwayTeam"]
        if h not in teams or a not in teams:
            continue
        ih = teams.index(h); ia = teams.index(a)
        ftr = row["FTR"]
        if ftr == "H":
            R[r, ih] = True;  R[r, ia] = False
        elif ftr == "A":
            R[r, ih] = False; R[r, ia] = True
        elif ftr == "D":
            R[r, ih] = False; R[r, ia] = False

    return R
