from __future__ import annotations
import numpy as np
import pandas as pd
from .elo import prob_home_win_from_elo

def build_X_from_elo(fixtures: pd.DataFrame, teams: list[str], elo: dict[str,float], home_adv: float = 60.0) -> np.ndarray:
    """
    X[round_idx, team_idx] = P(team wins in that round)
    """
    max_round = int(fixtures["Round"].max())
    X = np.full((max_round, len(teams)), np.nan, dtype=float)

    for _, row in fixtures.iterrows():
        r = int(row["Round"]) - 1
        h = row["HomeTeam"]; a = row["AwayTeam"]
        if h not in teams or a not in teams:
            continue
        ih = teams.index(h); ia = teams.index(a)

        ph = prob_home_win_from_elo(elo.get(h,1500.0), elo.get(a,1500.0), home_adv=home_adv)
        pa = 1.0 - ph  # (draws ignored)

        X[r, ih] = ph
        X[r, ia] = pa

    return X
