from __future__ import annotations
import numpy as np
import pandas as pd

def detect_3way_bookies(columns: list[str]) -> list[str]:
    cols = set(columns)
    prefixes = []
    for c in columns:
        if c.endswith("H"):
            p = c[:-1]
            if (p + "D") in cols and (p + "A") in cols:
                prefixes.append(p)
    preferred = ["B365", "PS", "WH", "BW", "VC", "LB", "Avg", "Max"]
    prefixes = sorted(set(prefixes), key=lambda x: (0 if x in preferred else 1, x))
    return prefixes

def choose_bookie(df: pd.DataFrame, prefer: list[str] | None = None) -> str:
    prefixes = detect_3way_bookies(list(df.columns))
    if not prefixes:
        raise ValueError("No 3-way odds columns found (need *H/*D/*A).")
    if prefer:
        for p in prefer:
            if p in prefixes:
                return p
    return prefixes[0]

def probs_from_decimal_odds(oh: float, od: float, oa: float) -> tuple[float,float,float]:
    oh, od, oa = float(oh), float(od), float(oa)
    inv = np.array([1.0/oh, 1.0/od, 1.0/oa], dtype=float)
    s = inv.sum()
    if not np.isfinite(s) or s <= 0:
        return (np.nan, np.nan, np.nan)
    p = inv / s
    return (float(p[0]), float(p[1]), float(p[2]))

def apply_odds_to_X(df_with_rounds: pd.DataFrame, X: np.ndarray, teams: list[str], bookie: str) -> np.ndarray:
    """
    Overwrite X using odds-implied probs where available.
    X is assumed shape (38, 20).
    """
    hcol, dcol, acol = f"{bookie}H", f"{bookie}D", f"{bookie}A"
    for _, row in df_with_rounds.iterrows():
        r = int(row["Round"])
        if r < 1 or r > X.shape[0]:
            continue

        home, away = row["HomeTeam"], row["AwayTeam"]
        if home not in teams or away not in teams:
            continue

        oh, od, oa = row.get(hcol, np.nan), row.get(dcol, np.nan), row.get(acol, np.nan)
        if not (np.isfinite(oh) and np.isfinite(od) and np.isfinite(oa)):
            continue

        ph, pd, pa = probs_from_decimal_odds(oh, od, oa)
        X[r-1, teams.index(home)] = ph
        X[r-1, teams.index(away)] = pa
    return X
