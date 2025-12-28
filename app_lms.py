from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt

from lms_local.fixtures2526 import load_fixture_results_csv, list_teams
from lms_local.elo import elo_train
from lms_local.future_matrix import build_X_from_elo
from lms_local.merge_rounds import merge_rounds
from lms_local.odds import choose_bookie, apply_odds_to_X
from lms_local.planner import plan_season_frontload


RAW_DIR = "data/raw"

# Current-season URLs
URL_ODDS_2526 = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
URL_FIX_2526 = "https://fixturedownload.com/download/csv/epl-2025"

# Big 6 teams (used for the "save big teams" penalty)
BIG6_TEAMS = {
    "Arsenal",
    "Chelsea",
    "Liverpool",
    "Manchester City",
    "Man City",
    "Manchester United",
    "Man United",
    "Tottenham",
    "Tottenham Hotspur",
}


# -------------------------------------------------------------------
# Shared helpers
# -------------------------------------------------------------------

def download_if_new(url: str, save_to: str):
    """Download file from url if missing or older than 6 hours."""
    os.makedirs(os.path.dirname(save_to), exist_ok=True)

    need = True
    if os.path.exists(save_to):
        mtime = os.path.getmtime(save_to)
        age = pd.Timestamp.now() - pd.to_datetime(mtime, unit="s")
        if age < pd.Timedelta(hours=6):
            need = False

    if not need:
        return

    r = requests.get(url, timeout=15)
    r.raise_for_status()
    with open(save_to, "wb") as f:
        f.write(r.content)


def probs_from_odds(odds_home, odds_draw, odds_away):
    """
    Convert 1X2 decimal odds to implied probabilities with overround removed.
    """
    if np.any(pd.isna([odds_home, odds_draw, odds_away])):
        return None, None, None
    if odds_home <= 0 or odds_draw <= 0 or odds_away <= 0:
        return None, None, None

    sum_recip = 1.0 / odds_home + 1.0 / odds_draw + 1.0 / odds_away
    summand = -(1.0 / 3.0) * sum_recip + (1.0 / 3.0)
    ph = 1.0 / odds_home + summand
    pd_ = 1.0 / odds_draw + summand
    pa = 1.0 / odds_away + summand
    return ph, pd_, pa


def show_df_with_download(df: pd.DataFrame, title: str, filename: str, key: str):
    """Small helper: show dataframe + CSV download button."""
    st.subheader(title)
    st.dataframe(df)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"⬇️ Download {title}",
        data=csv,
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def range_controls(label: str,
                   default_min: float,
                   default_max: float,
                   default_step: float,
                   key_prefix: str):
    """Triplet of number_inputs for min/max/step."""
    c1, c2, c3 = st.columns(3)
    with c1:
        vmin = st.number_input(
            f"Min {label}", value=default_min, key=f"{key_prefix}_min"
        )
    with c2:
        vmax = st.number_input(
            f"Max {label}", value=default_max, key=f"{key_prefix}_max"
        )
    with c3:
        vstep = st.number_input(
            f"{label} step", value=default_step, key=f"{key_prefix}_step"
        )
    return float(vmin), float(vmax), float(vstep)


def scalar_control(label: str,
                   default: float,
                   min_value: float,
                   max_value: float,
                   step: float,
                   key: str):
    """
    Single number_input for a scalar hyper-parameter.
    Used for the current-season page or when a knob is fixed in the grid.
    """
    return float(
        st.number_input(
            label,
            min_value=min_value,
            max_value=max_value,
            value=default,
            step=step,
            key=key,
        )
    )


# -------------------------------------------------------------------
# Current-season model builder (Elo + Odds blend)
# -------------------------------------------------------------------

def build_X_elo_plus_odds(fixtures_path: str,
                          odds_path: str,
                          hist_files: list[str],
                          home_adv: float = 60.0,
                          w_odds: float = 0.75,
                          k_elo: float = 20.0):
    """
    Build X using Elo on past seasons + current season,
    then blend Elo and odds:

        X_final = w_odds * X_odds + (1 - w_odds) * X_elo

    where odds exist; otherwise we just use X_elo.
    """
    fixtures = load_fixture_results_csv(fixtures_path)
    teams = list_teams(fixtures)

    # --- Train Elo on multiple seasons + current season ---
    dfs = []
    for fname in hist_files:
        path = os.path.join(RAW_DIR, fname)
        df = pd.read_csv(path, encoding="latin-1")
        dfs.append(df[["HomeTeam", "AwayTeam", "FTR"]].dropna())

    cur = pd.read_csv(odds_path, encoding="latin-1")
    if {"HomeTeam", "AwayTeam", "FTR"}.issubset(cur.columns):
        dfs.append(cur[["HomeTeam", "AwayTeam", "FTR"]].dropna())

    hist = pd.concat(dfs, ignore_index=True)
    elo = elo_train(hist, k=k_elo, home_adv=home_adv)

    # Base Elo probabilities for ALL rounds
    X_elo = build_X_from_elo(fixtures, teams, elo, home_adv=home_adv)

    # Overwrite with odds where we actually have them
    merged = merge_rounds(fixtures_path, odds_path)
    if len(merged) > 0:
        bookie = choose_bookie(merged, prefer=["B365", "PS"])
        X_odds_applied = apply_odds_to_X(merged, X_elo.copy(), teams, bookie)

        diff = np.abs(X_odds_applied - X_elo)
        mask = diff > 1e-12

        X_final = X_elo.copy()
        X_final[mask] = w_odds * X_odds_applied[mask] + (1.0 - w_odds) * X_elo[mask]
    else:
        bookie = "ELO_ONLY"
        X_final = X_elo

    X_final = np.clip(X_final, 1e-6, 0.999999)
    return fixtures, teams, X_final, bookie, merged


def candidate_table_for_round(fixtures: pd.DataFrame,
                              X: np.ndarray,
                              teams: list[str],
                              round_num: int,
                              forbidden_teams: set[str]) -> pd.DataFrame:
    """
    For a given round, list *all* legal team options (not already used),
    with opponent, venue and P_win_model, sorted by P_win_model desc.
    """
    rows = []
    this_round = fixtures[fixtures["Round"] == round_num]

    for _, row in this_round.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        # Home option
        if home in teams and home not in forbidden_teams:
            hi = teams.index(home)
            p = float(X[round_num - 1, hi])
            rows.append(
                dict(Round=round_num, Pick=home, Opponent=away, Venue="H", P_win_model=p)
            )

        # Away option
        if away in teams and away not in forbidden_teams:
            ai = teams.index(away)
            p = float(X[round_num - 1, ai])
            rows.append(
                dict(Round=round_num, Pick=away, Opponent=home, Venue="A", P_win_model=p)
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("P_win_model", ascending=False)
    df = df.drop_duplicates(subset=["Pick"], keep="first")

    df["logP"] = np.log(np.clip(df["P_win_model"], 1e-12, 1.0))
    df["cum_logP"] = df["logP"]
    df["cum_survival_proxy"] = np.exp(df["cum_logP"])
    return df.sort_values("P_win_model", ascending=False)


# -------------------------------------------------------------------
# Backtest helpers and data structures
# -------------------------------------------------------------------

@dataclass
class SeasonComponents:
    name: str
    teams: list[str]
    X_odds: np.ndarray
    X_elo: np.ndarray
    R: np.ndarray
    X_index: np.ndarray
    matches: pd.DataFrame


def greedy_perm(X: np.ndarray):
    """Simple greedy LMS strategy: at round i pick highest P(win) remaining."""
    num_rounds, num_teams = X.shape
    visited = []
    for i in range(min(num_rounds, num_teams)):
        allowed = [j for j in range(num_teams) if j not in visited]
        if not allowed:
            break
        probs = X[i, allowed]
        if np.all(probs <= 0):
            break
        j_sel = allowed[int(np.argmax(probs))]
        visited.append(j_sel)
    return visited


def survival_from_perm(perm, R: np.ndarray):
    """
    Given a permutation (list of team indices) and results matrix R,
    return how many weeks you'd survive if you followed that order.
    """
    weeks = 0
    for i, j in enumerate(perm):
        if i >= R.shape[0]:
            break
        if R[i, j] == 1:
            weeks += 1
        else:
            break
    return weeks


def perm_report_df(perm, X, R, teams):
    """
    Build a per-week dataframe: round, team, p(win), actual result, alive?
    """
    rows = []
    alive = True
    for i, j in enumerate(perm):
        if i >= X.shape[0]:
            break
        p = float(X[i, j])
        actual_win = bool(R[i, j] == 1)
        alive = alive and actual_win
        rows.append(
            dict(
                Round=i + 1,
                Pick=teams[j],
                P_win_model=p,
                Actual="W" if actual_win else "L",
                Alive=alive,
            )
        )
        if not alive:
            break
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["logP"] = np.log(df["P_win_model"])
    df["cum_logP"] = df["logP"].cumsum()
    df["cum_survival_proxy"] = np.exp(df["cum_logP"])
    return df


def top3_per_round_df(X, X_index, df_matches, teams):
    """
    For each 'round' (i-th game per team), list the top 3 teams by P(win),
    with opponent, venue and actual result (W/D/L).
    """
    num_rounds, num_teams = X.shape
    rows = []

    for i in range(num_rounds):
        round_candidates = []

        for j in range(num_teams):
            p = float(X[i, j])
            idx = int(X_index[i, j])
            if idx < 0 or p <= 0.0:
                continue

            row = df_matches.iloc[idx]
            team = teams[j]

            home = row["HomeTeam"]
            away = row["AwayTeam"]
            ftr = row["FTR"]

            if team == home:
                opponent = away
                venue = "H"
                if ftr == "H":
                    actual = "W"
                elif ftr == "D":
                    actual = "D"
                else:
                    actual = "L"
            elif team == away:
                opponent = home
                venue = "A"
                if ftr == "A":
                    actual = "W"
                elif ftr == "D":
                    actual = "D"
                else:
                    actual = "L"
            else:
                continue

            round_candidates.append(
                dict(
                    Round=i + 1,
                    Team=team,
                    Opponent=opponent,
                    Venue=venue,
                    P_win_model=p,
                    Actual=actual,
                )
            )

        if not round_candidates:
            continue

        rc_df = pd.DataFrame(round_candidates)
        rc_df = rc_df.sort_values("P_win_model", ascending=False).head(3)
        rc_df = rc_df.reset_index(drop=True)
        rc_df["Rank"] = rc_df.index + 1

        rows.append(rc_df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out = out[["Round", "Rank", "Team", "Opponent", "Venue", "P_win_model", "Actual"]]
    return out


# -------------------------------------------------------------------
# Historical Elo + Odds components (season-only Elo)
# -------------------------------------------------------------------

def build_season_prob_components(
    season_path: str,
    label: str,
    bookie_prefix: str = "B365",
    k_elo: float = 20.0,
    home_adv: float = 60.0,
) -> SeasonComponents:
    """
    Build season-only Elo + odds components.

    Uses only matches from that season to train Elo, so there's no
    leakage from future seasons.
    """
    df = pd.read_csv(season_path, encoding="latin-1")
    df = df.copy().reset_index(drop=True)

    if not {"HomeTeam", "AwayTeam", "FTR"}.issubset(df.columns):
        raise ValueError(f"{season_path} missing HomeTeam/AwayTeam/FTR")

    if "Date" in df.columns:
        df["__Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.sort_values("__Date").reset_index(drop=True)
        df = df.drop(columns="__Date")

    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    num_teams = len(teams)
    num_rounds = 2 * (num_teams - 1)

    X_odds = np.zeros((num_rounds, num_teams), dtype=float)
    X_elo = np.zeros((num_rounds, num_teams), dtype=float)
    R = np.zeros((num_rounds, num_teams), dtype=int)
    X_index = np.full((num_rounds, num_teams), fill_value=-1, dtype=int)

    games_played_by_team = np.zeros(num_teams, dtype=int)

    ratings = {team: 1500.0 for team in teams}

    h_col = bookie_prefix + "H"
    d_col = bookie_prefix + "D"
    a_col = bookie_prefix + "A"

    if not {h_col, d_col, a_col}.issubset(df.columns):
        raise ValueError(f"{season_path} missing {bookie_prefix}H/D/A columns")

    for idx, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        ftr = row["FTR"]

        try:
            j_home = teams.index(home)
            j_away = teams.index(away)
        except ValueError:
            continue

        i_home = games_played_by_team[j_home]
        i_away = games_played_by_team[j_away]

        if i_home >= num_rounds or i_away >= num_rounds:
            continue

        # Odds-based
        oh, od, oa = row[h_col], row[d_col], row[a_col]
        ph_odds, _, pa_odds = probs_from_odds(oh, od, oa)

        # Elo expected (before update)
        Rh = ratings[home]
        Ra = ratings[away]
        exp_home = 1.0 / (1.0 + 10 ** (-(Rh + home_adv - Ra) / 400.0))
        exp_away = 1.0 - exp_home

        X_elo[i_home, j_home] = exp_home
        X_elo[i_away, j_away] = exp_away

        if ph_odds is not None and pa_odds is not None:
            X_odds[i_home, j_home] = ph_odds
            X_odds[i_away, j_away] = pa_odds

        X_index[i_home, j_home] = idx
        X_index[i_away, j_away] = idx

        # Actual result & Elo update
        if ftr == "H":
            s_home, s_away = 1.0, 0.0
            R[i_home, j_home] = 1
            R[i_away, j_away] = 0
        elif ftr == "A":
            s_home, s_away = 0.0, 1.0
            R[i_home, j_home] = 0
            R[i_away, j_away] = 1
        elif ftr == "D":
            s_home, s_away = 0.5, 0.5
        else:
            s_home = s_away = 0.5

        ratings[home] = Rh + k_elo * (s_home - exp_home)
        ratings[away] = Ra + k_elo * (s_away - exp_away)

        games_played_by_team[j_home] += 1
        games_played_by_team[j_away] += 1

    X_elo = np.clip(X_elo, 1e-6, 0.999999)
    return SeasonComponents(
        name=label,
        teams=teams,
        X_odds=X_odds,
        X_elo=X_elo,
        R=R,
        X_index=X_index,
        matches=df,
    )


@st.cache_data(show_spinner=False)
def build_season_prob_components_cached(
    season_path: str,
    label: str,
    bookie_prefix: str,
    k_elo: float,
    home_adv: float,
) -> SeasonComponents:
    """Cached wrapper for season components."""
    return build_season_prob_components(
        season_path,
        label=label,
        bookie_prefix=bookie_prefix,
        k_elo=k_elo,
        home_adv=home_adv,
    )


def blend_X_odds_elo(X_odds: np.ndarray, X_elo: np.ndarray, w_odds: float) -> np.ndarray:
    """
    Blend odds and Elo probabilities with weight w_odds.
    Where odds are missing (0), fall back to pure Elo.
    """
    X_elo_c = np.clip(X_elo, 1e-6, 0.999999)
    X_odds_c = np.clip(X_odds, 1e-6, 0.999999)

    mask_have_odds = X_odds > 0.0
    X = X_elo_c.copy()
    X[mask_have_odds] = (
        w_odds * X_odds_c[mask_have_odds] + (1.0 - w_odds) * X_elo_c[mask_have_odds]
    )
    X = np.clip(X, 1e-6, 0.999999)
    return X


def apply_away_penalty(X: np.ndarray,
                       X_index: np.ndarray,
                       df_matches: pd.DataFrame,
                       teams: list[str],
                       penalty: float) -> np.ndarray:
    """
    Knock an absolute 'penalty' off away probabilities.
    If penalty <= 0, return X unchanged.
    """
    if penalty <= 0.0:
        return X

    X_adj = X.copy()
    num_rounds, num_teams = X.shape

    for i in range(num_rounds):
        for j in range(num_teams):
            idx = int(X_index[i, j])
            if idx < 0:
                continue
            row = df_matches.iloc[idx]
            team = teams[j]
            if team == row["AwayTeam"]:
                X_adj[i, j] = max(1e-6, X_adj[i, j] - penalty)

    return X_adj


# ---------- Precomputed features for faster grid search ----------

def precompute_X_features(X: np.ndarray,
                          teams: list[str],
                          big_teams: set[str] | None = None):
    """
    Precompute log-probs, team max probs, and (optionally) a Big-6 mask
    for a given X matrix.
    """
    X_clipped = np.clip(X, 1e-12, 1.0)
    logX = np.log(X_clipped)                 # (rounds, teams)
    max_probs = X_clipped.max(axis=0)        # (teams,)

    if big_teams is not None:
        big_mask = np.array(
            [name in big_teams for name in teams], dtype=bool
        )
    else:
        big_mask = None

    return logX, max_probs, big_mask


def perm_from_decay_precomputed(logX: np.ndarray,
                                max_probs: np.ndarray,
                                big_mask: np.ndarray | None,
                                decay: float,
                                min_pick_prob: float = 0.0,
                                big_team_penalty: float = 0.0) -> list[int]:
    """
    Build a permutation using precomputed logX/max_probs/big_mask.
    """
    num_rounds, _ = logX.shape

    weights = decay ** np.arange(num_rounds)      # (rounds,)
    scores = (weights[:, None] * logX).sum(axis=0)

    if big_team_penalty > 0.0 and big_mask is not None:
        scores = scores.copy()
        scores[big_mask] -= big_team_penalty

    order = list(np.argsort(-scores))

    if min_pick_prob > 0.0:
        order = [j for j in order if max_probs[j] >= min_pick_prob]

    return order


def detailed_run_report(perm,
                        X: np.ndarray,
                        R: np.ndarray,
                        teams: list[str],
                        X_index: np.ndarray,
                        df_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Per-week report with opponents, venue, probs, actual result and survival flag.
    """
    rows = []
    alive = True
    num_rounds = X.shape[0]

    for i, j in enumerate(perm):
        if i >= num_rounds:
            break

        p = float(X[i, j])
        idx = int(X_index[i, j])
        team = teams[j]

        opponent = None
        venue = None
        actual_result = None

        if idx >= 0:
            row = df_matches.iloc[idx]
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            ftr = row["FTR"]

            if team == home:
                opponent = away
                venue = "H"
                if ftr == "H":
                    actual_result = "W"
                elif ftr == "D":
                    actual_result = "D"
                else:
                    actual_result = "L"
            elif team == away:
                opponent = home
                venue = "A"
                if ftr == "A":
                    actual_result = "W"
                elif ftr == "D":
                    actual_result = "D"
                else:
                    actual_result = "L"

        actual_win = bool(R[i, j] == 1)
        alive = alive and actual_win

        rows.append(
            dict(
                Round=i + 1,
                Pick=team,
                Opponent=opponent,
                Venue=venue,
                P_win_model=p,
                Actual=actual_result,
                Alive=alive,
            )
        )

        if not alive:
            break

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["logP"] = np.log(np.clip(df["P_win_model"], 1e-12, 1.0))
    df["cum_logP"] = df["logP"].cumsum()
    df["cum_survival_proxy"] = np.exp(df["cum_logP"])
    return df


# -------------------------------------------------------------------
# Main Streamlit app
# -------------------------------------------------------------------

def main():
    st.set_page_config(page_title="LMS Planner 25/26", layout="wide")

    tab_plan, tab_backtest = st.tabs(
        ["📅 Current season planner", "📈 Backtest previous seasons"]
    )

    # ================= TAB 1: CURRENT SEASON PLANNER =================
    with tab_plan:
        st.title("⚽ Last Man Standing Planner – Premier League 2025/26")

        st.markdown(
            """
            This tab uses **5 past seasons + current 2025/26 results** to build an Elo model,  
            then blends that with **bookmaker odds (25/26)** where available, and plans your LMS picks.
            """
        )

        fixtures_path = os.path.join(RAW_DIR, "E0_2025-26_fixtures.csv")
        odds_path = os.path.join(RAW_DIR, "E0_2025-26.csv")

        with st.spinner("Downloading latest fixtures + odds…"):
            download_if_new(URL_FIX_2526, fixtures_path)
            download_if_new(URL_ODDS_2526, odds_path)

        # Historical seasons for Elo training
        hist_files = [
            "E0_2020-21.csv",
            "E0_2021-22.csv",
            "E0_2022-23.csv",
            "E0_2023-24.csv",
            "E0_2024-25.csv",
        ]
        missing_hist = [hf for hf in hist_files if not os.path.exists(os.path.join(RAW_DIR, hf))]
        if missing_hist:
            st.error("Missing historical CSV files in data/raw:")
            for m in missing_hist:
                st.write("-", m)
            st.stop()

        st.sidebar.header("Current Season Settings")

        # Odds vs Elo blend
        w_odds = st.sidebar.slider(
            "Odds vs Elo blend (reliance on bookies)",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.05,
        )
        st.sidebar.caption(
            "**0.0** = pure Elo (multi-season strength only)\n"
            "**1.0** = pure odds (bookmakers only)\n"
            "Recommended: **0.6–0.85** (trust odds, but keep some Elo)."
        )

        # Advanced Elo knobs (current season)
        with st.sidebar.expander("Advanced Elo settings (optional)"):
            st.markdown(
                """
                *These change how the Elo part of the model behaves for 25/26.*  
                If you leave them off, defaults of **K = 20**, **home_adv = 60** are used.
                """
            )
            use_custom_k_cur = st.checkbox("Custom Elo K-factor", value=False, key="cur_k")
            if use_custom_k_cur:
                k_elo_cur = scalar_control(
                    "Elo K-factor (responsiveness)",
                    default=20.0,
                    min_value=5.0,
                    max_value=40.0,
                    step=0.5,
                    key="cur_k_input",
                )
            else:
                k_elo_cur = 20.0

            use_custom_home_cur = st.checkbox(
                "Custom home advantage (rating pts)",
                value=False,
                key="cur_home",
            )
            if use_custom_home_cur:
                home_adv_cur = scalar_control(
                    "Home advantage (rating points)",
                    default=60.0,
                    min_value=20.0,
                    max_value=100.0,
                    step=1.0,
                    key="cur_home_input",
                )
            else:
                home_adv_cur = 60.0

        # Educational mirror of sim knobs
        with st.sidebar.expander("Advanced LMS strategy knobs (used in simulations)"):
            st.markdown(
                """
                These are the extra knobs the **Backtest** tab can use when simulating seasons:

                - **Minimum P(win)** – avoid teams whose *best* game all season is below this.  
                - **Away penalty** – subtracts a small value from away win probabilities.  
                - **Big-team penalty** – nudges the planner to *save* Big 6 sides for later.  

                Here you can see the ideas; the real controls live in the **Backtest** tab.
                """
            )
            st.checkbox("Enforce minimum P(win) (sim only)", value=False, key="cur_minpick_on")
            st.number_input(
                "Min P(win) over season (sim only)",
                min_value=0.50,
                max_value=0.80,
                value=0.55,
                step=0.01,
                key="cur_minpick_input",
            )
            st.checkbox("Apply away-game penalty (sim only)", value=False, key="cur_away_on")
            st.number_input(
                "Away penalty (absolute prob, sim only)",
                min_value=0.00,
                max_value=0.10,
                value=0.00,
                step=0.01,
                key="cur_away_input",
            )
            st.checkbox("Discourage using Big 6 early (sim only)",
                        value=False, key="cur_big_on")
            st.number_input(
                "Big 6 penalty (sim only)",
                min_value=0.0,
                max_value=2.0,
                value=0.0,
                step=0.1,
                key="cur_big_input",
            )

        @st.cache_data(show_spinner=True)
        def _build_model_cached(w_odds_: float, k_elo_: float, home_adv_: float):
            return build_X_elo_plus_odds(
                fixtures_path,
                odds_path,
                hist_files,
                home_adv=home_adv_,
                w_odds=w_odds_,
                k_elo=k_elo_,
            )

        fixtures, teams, X, bookie, merged = _build_model_cached(
            float(w_odds), float(k_elo_cur), float(home_adv_cur)
        )

        min_round = int(fixtures["Round"].min())
        max_round = int(fixtures["Round"].max())

        current_round = st.sidebar.number_input(
            "Round you're about to pick for",
            min_value=min_round,
            max_value=max_round,
            value=min_round,
            step=1,
        )

        used_teams = st.sidebar.multiselect(
            "Teams you've already used in LMS",
            options=sorted(teams),
        )

        decay = st.sidebar.slider(
            "Front-load strength (risk-aversion)",
            min_value=0.80,
            max_value=0.99,
            value=0.90,
            step=0.01,
        )
        st.sidebar.caption(
            "**What this does:**\n"
            "- The planner maximises a weighted product of win probabilities.\n"
            "- `decay` controls how much **early rounds matter more**.\n"
        )

        st.sidebar.write(f"Using bookie: **{bookie}**")
        if len(merged) > 0:
            st.sidebar.write(
                f"Odds currently cover rounds **{int(merged['Round'].min())}"
                f" → {int(merged['Round'].max())}**"
            )

        if st.button("🔮 Compute next pick", type="primary", key="plan_button"):
            forbidden = set(used_teams)

            plan = plan_season_frontload(
                fixtures=fixtures,
                X=X,
                teams=teams,
                start_round=int(current_round),
                forbidden_teams=forbidden,
                decay=float(decay),
            )

            if plan.empty:
                st.error(
                    "No valid plan could be generated.\n"
                    "- Check the round number.\n"
                    "- Make sure you haven't used too many teams already."
                )
            else:
                next_pick = plan.iloc[0]

                st.subheader(f"🎯 Best LMS pick for Round {int(current_round)}")
                st.success(
                    f"**{next_pick['Pick']}** vs **{next_pick['Opponent']}** "
                    f"({'home' if next_pick['Venue']=='H' else 'away'}) — "
                    f"model P(win) = **{next_pick['P_win_model']:.3f}**"
                )

                candidates = candidate_table_for_round(
                    fixtures=fixtures,
                    X=X,
                    teams=teams,
                    round_num=int(current_round),
                    forbidden_teams=forbidden,
                )

                st.subheader("Top 5 choices from this week (this round only)")
                if candidates.empty:
                    st.info("No fixtures found for this round or all teams forbidden.")
                else:
                    st.dataframe(
                        candidates.head(5)[
                            ["Round", "Pick", "Opponent", "Venue",
                             "P_win_model", "cum_survival_proxy"]
                        ]
                    )

                st.subheader("Full plan from this round onward")
                st.dataframe(plan)

                csv = plan.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download full schedule as CSV",
                    data=csv,
                    file_name=f"lms_plan_from_gw{int(current_round)}_{bookie}.csv",
                    mime="text/csv",
                    key="plan_download",
                )
        else:
            st.info("Pick a round + used teams on the left, then click **Compute next pick**.")

    # ================= TAB 2: BACKTEST / SETTINGS SEARCH =================
    with tab_backtest:
        st.title("📈 Backtest LMS Strategies on Previous Seasons")

        st.markdown(
            """
            This tab uses **historical season CSVs** from `data/raw` and a
            **season-only Elo + odds model** to test:

            - Different `w_odds` (odds vs Elo blend)  
            - Different `decay` (front-load strength) values  
            - Optional ranges for K-factor, home advantage, min P(win),
              away penalty and Big-6 penalty.

            For each full combination we record **how many weeks you survive**
            in each season.
            """
        )

        # Available seasons
        season_files = {
            "2020-21": "E0_2020-21.csv",
            "2021-22": "E0_2021-22.csv",
            "2022-23": "E0_2022-23.csv",
            "2023-24": "E0_2023-24.csv",
            "2024-25": "E0_2024-25.csv",
        }

        missing = [lab for lab, fn in season_files.items()
                   if not os.path.exists(os.path.join(RAW_DIR, fn))]
        if missing:
            st.error("Missing some historical files in data/raw. Can't backtest:")
            for lab in missing:
                st.write("-", lab, "→", season_files[lab])
            st.stop()

        # Allow choosing which seasons to include (speeds things up)
        st.subheader("📅 Seasons to include in search")
        selected_seasons = st.multiselect(
            "Select seasons for the grid search",
            options=list(season_files.keys()),
            default=list(season_files.keys()),
        )
        if not selected_seasons:
            st.warning("Select at least one season.")
            return

        # ---------- Quick odds-only backtest ----------
        st.subheader("⚡ Quick odds-only backtest (single season)")
        season_label = st.selectbox(
            "Season for quick odds-only backtest",
            options=list(season_files.keys()),
            index=len(season_files) - 1,
        )
        bookie_prefix = st.selectbox(
            "Bookmaker (1X2 odds prefix)",
            options=["B365"],
            index=0,
        )

        if st.button("🚀 Run quick odds-only backtest", key="quick_bt_btn"):
            path = os.path.join(RAW_DIR, season_files[season_label])
            try:
                comps = build_season_prob_components(
                    path,
                    label=season_label,
                    bookie_prefix=bookie_prefix,
                    k_elo=20.0,
                    home_adv=60.0,
                )
            except Exception as e:
                st.error(f"Error building components: {e}")
            else:
                X = comps.X_odds
                R = comps.R
                teams = comps.teams

                perm = greedy_perm(X)
                weeks = survival_from_perm(perm, R)
                report = perm_report_df(perm, X, R, teams)

                st.subheader(f"🧮 Greedy odds-only backtest for {season_label}")
                st.write(f"Greedy LMS (odds-only) survived: **{weeks} weeks**")

                if not report.empty:
                    show_df_with_download(
                        report,
                        "Per-week greedy LMS detail",
                        f"lms_backtest_{season_label}_{bookie_prefix}.csv",
                        key=f"bt_csv_{season_label}",
                    )

                top3_df = top3_per_round_df(
                    X, comps.X_index, comps.matches, comps.teams
                )
                if top3_df.empty:
                    st.info("Could not compute top 3 per round (missing odds or fixtures).")
                else:
                    show_df_with_download(
                        top3_df,
                        "Top 3 odds-only picks for each 'round'",
                        f"lms_top3_{season_label}_{bookie_prefix}.csv",
                        key=f"top3_csv_{season_label}",
                    )

        st.markdown("---")
        st.subheader("🎛 Settings search: grid over strategy knobs")

        st.markdown(
            """
            For each parameter you can choose:

            - **Scan in grid** – give a min / max / step.  
            - **Fixed** – give a single value used for all runs.

            The total number of combinations grows as the product of all
            grid sizes, so if that number gets huge the search will be slow.
            """
        )

        # ---------- Advanced simulation knobs (grid setup) ----------
        with st.expander("Advanced simulation knobs (grid setup)"):
            st.markdown("**Elo model knobs**")

            # K-factor
            scan_k = st.checkbox(
                "Scan Elo K-factor in grid",
                value=False,
                key="scan_k",
            )
            if scan_k:
                k_min, k_max, k_step = range_controls(
                    "K_elo",
                    default_min=20.0,
                    default_max=20.0,
                    default_step=1.0,
                    key_prefix="kgrid",
                )
                k_values = np.arange(k_min, k_max + 1e-9, k_step)
            else:
                k_single = scalar_control(
                    "K_elo (fixed for this run)",
                    default=20.0,
                    min_value=5.0,
                    max_value=40.0,
                    step=0.5,
                    key="k_single",
                )
                k_values = np.array([k_single])

            # Home advantage
            scan_home = st.checkbox(
                "Scan home advantage (rating pts) in grid",
                value=False,
                key="scan_home",
            )
            if scan_home:
                home_min, home_max, home_step = range_controls(
                    "Home advantage",
                    default_min=60.0,
                    default_max=60.0,
                    default_step=5.0,
                    key_prefix="homegrid",
                )
                home_adv_values = np.arange(home_min, home_max + 1e-9, home_step)
            else:
                home_single = scalar_control(
                    "Home advantage (fixed)",
                    default=60.0,
                    min_value=20.0,
                    max_value=100.0,
                    step=1.0,
                    key="home_single",
                )
                home_adv_values = np.array([home_single])

            st.markdown("---")
            st.markdown("**LMS strategy knobs**")

            # Minimum pick probability
            scan_minpick = st.checkbox(
                "Scan minimum P(win) in grid",
                value=False,
                key="scan_minpick",
            )
            if scan_minpick:
                mp_min, mp_max, mp_step = range_controls(
                    "Minimum P(win)",
                    default_min=0.55,
                    default_max=0.55,
                    default_step=0.01,
                    key_prefix="minpickgrid",
                )
                min_pick_values = np.arange(mp_min, mp_max + 1e-9, mp_step)
            else:
                mp_single = scalar_control(
                    "Minimum P(win) (fixed)",
                    default=0.55,
                    min_value=0.50,
                    max_value=0.80,
                    step=0.01,
                    key="minpick_single",
                )
                # 0.0 means "no constraint", so if you want 0.55 as a real
                # constraint you HAVE to scan or keep this non-zero.
                min_pick_values = np.array([mp_single])

            # Away penalty
            scan_away = st.checkbox(
                "Scan away-game penalty in grid",
                value=False,
                key="scan_away",
            )
            if scan_away:
                ap_min, ap_max, ap_step = range_controls(
                    "Away penalty",
                    default_min=0.02,
                    default_max=0.02,
                    default_step=0.01,
                    key_prefix="awaygrid",
                )
                away_penalty_values = np.arange(ap_min, ap_max + 1e-9, ap_step)
            else:
                ap_single = scalar_control(
                    "Away penalty (fixed)",
                    default=0.00,
                    min_value=0.0,
                    max_value=0.10,
                    step=0.01,
                    key="away_single",
                )
                away_penalty_values = np.array([ap_single])

            # Big-6 penalty
            scan_big = st.checkbox(
                "Scan Big-6 penalty in grid",
                value=False,
                key="scan_big",
            )
            if scan_big:
                bp_min, bp_max, bp_step = range_controls(
                    "Big-6 penalty",
                    default_min=0.5,
                    default_max=0.5,
                    default_step=0.1,
                    key_prefix="biggrid",
                )
                big_team_penalty_values = np.arange(bp_min, bp_max + 1e-9, bp_step)
            else:
                bp_single = scalar_control(
                    "Big-6 penalty (fixed)",
                    default=0.0,
                    min_value=0.0,
                    max_value=2.0,
                    step=0.1,
                    key="big_single",
                )
                big_team_penalty_values = np.array([bp_single])

        # ---------- Grid over w_odds & decay ----------
        st.markdown("**Core strategy grid: w_odds vs decay**")
        w_min, w_max, w_step = range_controls(
            "w_odds", default_min=0.5, default_max=0.9,
            default_step=0.1, key_prefix="wgrid_main"
        )
        decay_min, decay_max, decay_step = range_controls(
            "decay", default_min=0.80, default_max=0.96,
            default_step=0.02, key_prefix="dgrid_main"
        )

        if st.button("🔍 Search best settings across selected seasons",
                     key="settings_search_btn"):
            w_values = np.arange(w_min, w_max + 1e-9, w_step)
            decay_values = np.arange(decay_min, decay_max + 1e-9, decay_step)

            if len(w_values) == 0 or len(decay_values) == 0:
                st.error("Grids are empty – check min/max/step values.")
                return

            total_iters = (
                len(selected_seasons)
                * len(k_values)
                * len(home_adv_values)
                * len(w_values)
                * len(away_penalty_values)
                * len(min_pick_values)
                * len(big_team_penalty_values)
                * len(decay_values)
            )

            st.info(f"Total parameter combinations in this run: **{total_iters:,}**")

            rows = []
            best_run_weeks = -1
            best_run_info = None

            progress = st.progress(0.0, text="Running settings grid…")
            iter_count = 0
            # To reduce overhead, only update progress ~200 times max
            update_every = max(1, total_iters // 200)

            for season_lab in selected_seasons:
                path = os.path.join(RAW_DIR, season_files[season_lab])

                for k_elo_bt in k_values:
                    for home_adv_bt in home_adv_values:
                        try:
                            comps = build_season_prob_components_cached(
                                path,
                                label=season_lab,
                                bookie_prefix=bookie_prefix,
                                k_elo=float(k_elo_bt),
                                home_adv=float(home_adv_bt),
                            )
                        except Exception as e:
                            st.warning(
                                f"Skipping {season_lab} (K={k_elo_bt}, home={home_adv_bt}): {e}"
                            )
                            skip_inner = (
                                len(w_values)
                                * len(away_penalty_values)
                                * len(min_pick_values)
                                * len(big_team_penalty_values)
                                * len(decay_values)
                            )
                            iter_count += skip_inner
                            continue

                        for w in w_values:
                            X_blend = blend_X_odds_elo(
                                comps.X_odds, comps.X_elo, float(w)
                            )

                            for away_penalty in away_penalty_values:
                                X_use = apply_away_penalty(
                                    X_blend,
                                    comps.X_index,
                                    comps.matches,
                                    comps.teams,
                                    float(away_penalty),
                                )

                                # Precompute once per (season, K, home, w, away)
                                logX, max_probs, big_mask = precompute_X_features(
                                    X_use,
                                    comps.teams,
                                    BIG6_TEAMS if np.any(big_team_penalty_values > 0) else None,
                                )

                                for min_pick_prob in min_pick_values:
                                    for big_team_penalty in big_team_penalty_values:
                                        for d in decay_values:
                                            perm = perm_from_decay_precomputed(
                                                logX,
                                                max_probs,
                                                big_mask if big_team_penalty > 0 else None,
                                                float(d),
                                                min_pick_prob=float(min_pick_prob),
                                                big_team_penalty=float(big_team_penalty),
                                            )
                                            weeks = survival_from_perm(
                                                perm, comps.R
                                            )

                                            rows.append(
                                                dict(
                                                    Season=season_lab,
                                                    w_odds=float(w),
                                                    Decay=float(d),
                                                    K_elo=float(k_elo_bt),
                                                    HomeAdv=float(home_adv_bt),
                                                    MinPick=float(min_pick_prob),
                                                    AwayPenalty=float(away_penalty),
                                                    Big6Penalty=float(big_team_penalty),
                                                    WeeksSurvived=int(weeks),
                                                )
                                            )

                                            if weeks > best_run_weeks:
                                                best_run_weeks = weeks
                                                best_run_info = dict(
                                                    Season=season_lab,
                                                    w_odds=float(w),
                                                    Decay=float(d),
                                                    K_elo=float(k_elo_bt),
                                                    HomeAdv=float(home_adv_bt),
                                                    MinPick=float(min_pick_prob),
                                                    AwayPenalty=float(away_penalty),
                                                    Big6Penalty=float(big_team_penalty),
                                                    Weeks=int(weeks),
                                                    Perm=perm,
                                                    X_used=X_use,
                                                    R=comps.R,
                                                    Teams=comps.teams,
                                                    X_index=comps.X_index,
                                                    Matches=comps.matches,
                                                )

                                            iter_count += 1
                                            if (
                                                iter_count % update_every == 0
                                                or iter_count == total_iters
                                            ):
                                                progress.progress(
                                                    min(iter_count / total_iters, 1.0),
                                                    text=(
                                                        f"Running settings grid… "
                                                        f"{iter_count}/{total_iters}"
                                                    ),
                                                )

            progress.progress(1.0, text="Settings grid complete ✅")

            results_df = pd.DataFrame(rows)
            if results_df.empty:
                st.error("No results computed – something went wrong.")
                return

            colL, colR = st.columns([2, 1])

            with colL:
                show_df_with_download(
                    results_df,
                    "Settings grid (all seasons, all knobs)",
                    "lms_settings_grid_all_seasons_full.csv",
                    key="settings_grid_csv",
                )

                best_per_season = (
                    results_df.sort_values(
                        ["Season", "WeeksSurvived"],
                        ascending=[True, False],
                    )
                    .groupby("Season")
                    .head(1)
                    .reset_index(drop=True)
                )
                show_df_with_download(
                    best_per_season,
                    "🏅 Best settings per season",
                    "lms_best_settings_per_season.csv",
                    key="best_per_season_csv",
                )

                # Best overall by *full* parameter combo
                avg_over_seasons_full = (
                    results_df.groupby(
                        [
                            "w_odds",
                            "Decay",
                            "K_elo",
                            "HomeAdv",
                            "MinPick",
                            "AwayPenalty",
                            "Big6Penalty",
                        ]
                    )["WeeksSurvived"]
                    .mean()
                    .reset_index(name="MeanWeeksAcrossSeasons")
                )
                best_overall = avg_over_seasons_full.loc[
                    avg_over_seasons_full["MeanWeeksAcrossSeasons"].idxmax()
                ]

                st.subheader("🌍 Best settings by average survival (full combo)")
                st.success(
                    "Best overall:\n"
                    f"- **w_odds** = {best_overall['w_odds']:.2f}\n"
                    f"- **decay** = {best_overall['Decay']:.3f}\n"
                    f"- **K_elo** = {best_overall['K_elo']:.1f}\n"
                    f"- **HomeAdv** = {best_overall['HomeAdv']:.1f}\n"
                    f"- **MinPick** = {best_overall['MinPick']:.3f}\n"
                    f"- **AwayPenalty** = {best_overall['AwayPenalty']:.3f}\n"
                    f"- **Big6Penalty** = {best_overall['Big6Penalty']:.3f}\n"
                    f"- **Mean survival** = "
                    f"**{best_overall['MeanWeeksAcrossSeasons']:.2f} weeks**"
                )

                # Heatmap of mean survival vs w_odds & decay (averaged over other knobs)
                st.subheader("🔥 Mean survival heatmap (averaged over other knobs)")

                avg_over_seasons_wd = (
                    results_df.groupby(["w_odds", "Decay"])["WeeksSurvived"]
                    .mean()
                    .reset_index(name="MeanWeeksAcrossSeasons")
                )

                pivot = avg_over_seasons_wd.pivot(
                    index="Decay", columns="w_odds",
                    values="MeanWeeksAcrossSeasons"
                )

                fig, ax = plt.subplots(figsize=(7, 5))
                cax = ax.imshow(
                    pivot.values,
                    aspect="auto",
                    origin="lower",
                )

                num_x = len(pivot.columns)
                num_y = len(pivot.index)

                x_tick_step = max(1, num_x // 8)
                y_tick_step = max(1, num_y // 8)

                x_ticks = np.arange(0, num_x, x_tick_step)
                y_ticks = np.arange(0, num_y, y_tick_step)

                ax.set_xticks(x_ticks)
                ax.set_xticklabels(
                    [f"{pivot.columns[i]:.2f}" for i in x_ticks],
                    rotation=45,
                    ha="right",
                )

                ax.set_yticks(y_ticks)
                ax.set_yticklabels(
                    [f"{pivot.index[i]:.2f}" for i in y_ticks]
                )

                ax.set_xlabel("w_odds")
                ax.set_ylabel("decay")

                fig.colorbar(cax, ax=ax, label="Mean weeks survived")
                fig.tight_layout()
                st.pyplot(fig)

            with colR:
                st.subheader("🏆 Longest single run of all time")
                if best_run_info is None:
                    st.info("No valid runs found.")
                else:
                    st.markdown(
                        f"**Season:** {best_run_info['Season']}  \n"
                        f"**w_odds:** {best_run_info['w_odds']:.2f}  \n"
                        f"**decay:** {best_run_info['Decay']:.3f}  \n"
                        f"**K_elo:** {best_run_info['K_elo']:.1f}  \n"
                        f"**HomeAdv:** {best_run_info['HomeAdv']:.1f}  \n"
                        f"**MinPick:** {best_run_info['MinPick']:.3f}  \n"
                        f"**AwayPenalty:** {best_run_info['AwayPenalty']:.3f}  \n"
                        f"**Big6Penalty:** {best_run_info['Big6Penalty']:.3f}  \n"
                        f"**Weeks survived:** {best_run_info['Weeks']}"
                    )

                    run_df = detailed_run_report(
                        best_run_info["Perm"],
                        best_run_info["X_used"],
                        best_run_info["R"],
                        best_run_info["Teams"],
                        best_run_info["X_index"],
                        best_run_info["Matches"],
                    )

                    if run_df.empty:
                        st.info("Could not build detailed run report.")
                    else:
                        p_prod = float(run_df["P_win_model"].prod())
                        p_avg = float(run_df["P_win_model"].mean())
                        home_rate = float(
                            (run_df["Venue"] == "H").mean()
                        )
                        big6_picks = run_df["Pick"].isin(BIG6_TEAMS).sum()

                        st.metric(
                            "Overall survival proxy (product of probs)",
                            f"{p_prod:.4f}",
                        )
                        st.metric(
                            "Average P(win) of picks",
                            f"{p_avg:.3f}",
                        )
                        st.metric(
                            "% picks at home",
                            f"{home_rate*100:.1f}%",
                        )
                        st.metric(
                            "Big 6 picks",
                            str(big6_picks),
                        )

                        st.dataframe(run_df)


if __name__ == "__main__":
    main()
