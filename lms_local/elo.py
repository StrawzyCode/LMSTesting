from __future__ import annotations
import math
import numpy as np
import pandas as pd

def elo_expected(ra: float, rb: float) -> float:
    # expected score for A vs B
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

def elo_train(df_matches: pd.DataFrame, k: float = 20.0, home_adv: float = 60.0) -> dict[str, float]:
    """
    Train Elo on historical matches.
    Expects columns: HomeTeam, AwayTeam, FTR (H/A/D)
    """
    teams = sorted(set(df_matches["HomeTeam"]).union(set(df_matches["AwayTeam"])))
    rating = {t: 1500.0 for t in teams}

    for _, row in df_matches.iterrows():
        h = row["HomeTeam"]; a = row["AwayTeam"]
        ftr = row["FTR"]
        if ftr not in ("H", "A", "D"):
            continue

        rh = rating[h] + home_adv
        ra = rating[a]
        eh = elo_expected(rh, ra)
        ea = 1.0 - eh

        # actual score: 1 win, 0 loss, 0.5 draw
        if ftr == "H":
            sh, sa = 1.0, 0.0
        elif ftr == "A":
            sh, sa = 0.0, 1.0
        else:
            sh, sa = 0.5, 0.5

        rating[h] = rating[h] + k * (sh - eh)
        rating[a] = rating[a] + k * (sa - ea)

    return rating

def prob_home_win_from_elo(r_home: float, r_away: float, home_adv: float = 60.0) -> float:
    return elo_expected(r_home + home_adv, r_away)
