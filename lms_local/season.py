from __future__ import annotations
import re
import numpy as np
import pandas as pd
from .odds import probs_3way

BOOKIE_3WAY_RE = re.compile(r"^([A-Za-z0-9]+)(H|D|A)$")

def detect_teams(df: pd.DataFrame) -> list[str]:
    teams = sorted(set(df["HomeTeam"].dropna().unique()).union(set(df["AwayTeam"].dropna().unique())))
    return teams

def detect_3way_bookies(df: pd.DataFrame) -> list[str]:
    """
    Detect all bookie prefixes that have H/D/A columns, e.g.
    B365H,B365D,B365A ; PSH,PSD,PSA ; BFDH,BFDD,BFDA ; etc.
    """
    cols = set(df.columns)
    prefixes = set()
    for c in cols:
        m = BOOKIE_3WAY_RE.match(c)
        if not m:
            continue
        prefix = m.group(1)
        if f"{prefix}H" in cols and f"{prefix}D" in cols and f"{prefix}A" in cols:
            prefixes.add(prefix)
    return sorted(prefixes)

def parse_date(df: pd.DataFrame) -> pd.Series:
    # football-data Date is typically dd/mm/yy or dd/mm/yyyy
    return pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

def assign_rounds_by_chunk(df: pd.DataFrame, matches_per_round: int = 10) -> pd.DataFrame:
    """
    EPL: 10 matches per round. We approximate 'Round' by sorting by (Date, Time)
    then chunking into blocks of 10 matches.
    """
    d = df.copy()
    d["ParsedDate"] = parse_date(d)
    sort_cols = ["ParsedDate"]
    if "Time" in d.columns:
        sort_cols.append("Time")
    d = d.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    d["Round"] = (d.index // matches_per_round) + 1
    return d

def build_round_team_matrices(
    df: pd.DataFrame,
    teams: list[str],
    bookie: str,
    prob_method: str,
    matches_per_round: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    X[r,t] = P(team t wins in round r)
    R[r,t] = did team t win in round r? (True/False/nan)
    """
    d = assign_rounds_by_chunk(df, matches_per_round=matches_per_round)

    n_rounds = int(d["Round"].max())
    n_teams = len(teams)

    X = np.full((n_rounds, n_teams), np.nan, dtype=float)
    R = np.full((n_rounds, n_teams), np.nan, dtype=object)

    for _, row in d.iterrows():
        h = row["HomeTeam"]
        a = row["AwayTeam"]
        if h not in teams or a not in teams:
            continue

        oh = row.get(f"{bookie}H", np.nan)
        od = row.get(f"{bookie}D", np.nan)
        oa = row.get(f"{bookie}A", np.nan)

        if not (np.isfinite(oh) and np.isfinite(od) and np.isfinite(oa)):
            continue

        ph, pdw, pa = probs_3way(float(oh), float(od), float(oa), prob_method)

        r = int(row["Round"]) - 1
        ih = teams.index(h)
        ia = teams.index(a)

        X[r, ih] = ph
        X[r, ia] = pa

        ftr = row.get("FTR", None)
        if ftr == "H":
            R[r, ih] = True
            R[r, ia] = False
        elif ftr == "A":
            R[r, ih] = False
            R[r, ia] = True
        elif ftr == "D":
            R[r, ih] = False
            R[r, ia] = False

    return X, R
