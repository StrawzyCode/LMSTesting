from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .strategies import hungarian_assignment


def _opponent_for_round(fixtures: pd.DataFrame, rnd: int, team: str):
    f = fixtures[fixtures["Round"] == rnd]
    for _, row in f.iterrows():
        if row["HomeTeam"] == team:
            return row["AwayTeam"], "H"
        if row["AwayTeam"] == team:
            return row["HomeTeam"], "A"
    return None, None


def _make_table(fixtures: pd.DataFrame,
                X: np.ndarray,
                teams: list[str],
                perm: list[int],
                start_round: int = 1,
                forbidden_teams: set[str] | None = None) -> pd.DataFrame:
    if forbidden_teams is None:
        forbidden_teams = set()

    rows = []
    used = set(forbidden_teams)
    horizon = min(len(perm), X.shape[0] - (start_round - 1))

    for k in range(horizon):
        rnd = start_round + k
        team_idx = int(perm[k])
        team = teams[team_idx]

        # enforce no repeats
        if team in used:
            # try to pick best unused for this round
            best_t, best_p = None, -1.0
            for t in range(len(teams)):
                if teams[t] in used:
                    continue
                p = X[rnd - 1, t]
                if np.isfinite(p) and float(p) > best_p:
                    best_p = float(p)
                    best_t = t
            if best_t is None:
                # no valid pick
                break
            team_idx = best_t
            team = teams[team_idx]

        used.add(team)
        opp, venue = _opponent_for_round(fixtures, rnd, team)
        pwin = float(X[rnd - 1, team_idx]) if np.isfinite(X[rnd - 1, team_idx]) else np.nan

        rows.append({
            "Round": rnd,
            "Pick": team,
            "Opponent": opp,
            "Venue": venue,
            "P_win_model": pwin,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["logP"] = np.log(np.clip(df["P_win_model"].astype(float), 1e-12, 1.0))
        df["cum_logP"] = df["logP"].cumsum()
        df["cum_survival_proxy"] = np.exp(df["cum_logP"])
    else:
        df["logP"] = []
        df["cum_logP"] = []
        df["cum_survival_proxy"] = []
    return df


def plan_season_optimal(fixtures: pd.DataFrame,
                        X: np.ndarray,
                        teams: list[str],
                        start_round: int = 1,
                        forbidden_teams: set[str] | None = None) -> pd.DataFrame:
    """
    Whole-distance optimal (max product of win probs) starting at start_round.
    """
    n_rounds, _ = X.shape
    subX = X[start_round - 1:n_rounds, :]
    perm = hungarian_assignment(subX, start_round=0)
    return _make_table(fixtures, X, teams, perm,
                       start_round=start_round,
                       forbidden_teams=forbidden_teams)


def plan_season_frontload(fixtures: pd.DataFrame,
                          X: np.ndarray,
                          teams: list[str],
                          start_round: int = 1,
                          forbidden_teams: set[str] | None = None,
                          decay: float = 0.90) -> pd.DataFrame:
    """
    Front-loaded survival:
    maximise sum_r (decay^(r-1)) * log(p_r), starting at start_round.
    decay < 1 ⇒ earlier rounds matter more (safer early).
    """
    if forbidden_teams is None:
        forbidden_teams = set()

    n_rounds, n_teams = X.shape
    start_idx = start_round - 1
    remaining_rounds = n_rounds - start_idx
    available_team_indices = [i for i, t in enumerate(teams) if t not in forbidden_teams]

    horizon = min(remaining_rounds, len(available_team_indices))
    if horizon <= 0:
        return pd.DataFrame(columns=[
            "Round", "Pick", "Opponent", "Venue",
            "P_win_model", "logP", "cum_logP", "cum_survival_proxy"
        ])

    P = X[start_idx:start_idx + horizon, :][:, available_team_indices].copy()
    eps = 1e-12
    P = np.where(np.isfinite(P), np.clip(P, eps, 1.0), eps)

    weights = (decay ** np.arange(horizon)).reshape(-1, 1)
    cost = -(weights * np.log(P))  # maximise ⇒ minimise negative

    row_ind, col_ind = linear_sum_assignment(cost)
    order = np.argsort(row_ind)
    perm_mapped = [available_team_indices[int(col_ind[i])] for i in order]

    return _make_table(fixtures, X, teams, perm_mapped,
                       start_round=start_round,
                       forbidden_teams=forbidden_teams)
