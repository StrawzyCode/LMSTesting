from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable

def survival_from_picks(R: np.ndarray, start_round: int, picks: list[int]) -> int:
    survived = 0
    for i, t in enumerate(picks):
        r = start_round + i
        if r >= R.shape[0]:
            break
        if R[r, t] is True:
            survived += 1
        else:
            break
    return survived

def backtest_over_season(X: np.ndarray, R: np.ndarray, strategy: Callable[[np.ndarray, int], list[int]]) -> pd.DataFrame:
    rows = []
    for start in range(X.shape[0]):
        picks = strategy(X, start)
        s = survival_from_picks(R, start, picks)
        rows.append({"start_round": start + 1, "survived": s})
    return pd.DataFrame(rows)

def summarize(df: pd.DataFrame) -> dict:
    return {
        "mean_survived": float(df["survived"].mean()),
        "median_survived": float(df["survived"].median()),
        "p_survive_5+": float((df["survived"] >= 5).mean()),
        "p_survive_10+": float((df["survived"] >= 10).mean()),
        "p_survive_15+": float((df["survived"] >= 15).mean()),
    }
