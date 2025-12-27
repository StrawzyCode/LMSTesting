from __future__ import annotations

import os
import numpy as np
import pandas as pd
import streamlit as st
import requests

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


# ---------- Shared helpers ----------

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


# ---------- Current-season model builder (Elo + Odds blend) ----------

def build_X_elo_plus_odds(fixtures_path: str,
                          odds_path: str,
                          hist_files: list[str],
                          home_adv: float = 60.0,
                          w_odds: float = 0.75):
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
    elo = elo_train(hist, k=20.0, home_adv=home_adv)

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


# ---------- Backtest helpers (odds-only model for quick reports) ----------

def build_X_R_from_bookie(season_path: str,
                          bookie_prefix: str = "B365"):
    """
    Original odds-only model, kept for simple greedy + top3 reports.
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
    num_rounds = 2 * (num_teams - 1)  # EPL

    X = np.zeros((num_rounds, num_teams), dtype=float)
    R = np.zeros((num_rounds, num_teams), dtype=int)
    X_index = np.full((num_rounds, num_teams), fill_value=-1, dtype=int)

    games_played_by_team = np.zeros(num_teams, dtype=int)

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

        oh, od, oa = row[h_col], row[d_col], row[a_col]
        ph, _, pa = probs_from_odds(oh, od, oa)
        if ph is None or pa is None:
            continue

        X[i_home, j_home] = ph
        X[i_away, j_away] = pa

        X_index[i_home, j_home] = idx
        X_index[i_away, j_away] = idx

        R[i_home, j_home] = 1 if ftr == "H" else 0
        R[i_away, j_away] = 1 if ftr == "A" else 0

        games_played_by_team[j_home] += 1
        games_played_by_team[j_away] += 1

    X = np.clip(X, 1e-6, 0.999999)
    return X, R, teams, X_index, df


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


# ---------- New: components for Elo + Odds on historical seasons ----------

def build_season_prob_components(
    season_path: str,
    bookie_prefix: str = "B365",
    k_elo: float = 20.0,
    home_adv: float = 60.0,
):
    """
    Build:
      - X_odds[i,j]: odds-implied win prob for team j in its i-th game
      - X_elo[i,j]:  simple Elo win prob for team j in its i-th game
      - R[i,j]:      actual win (1) / not win (0) for team j in that game
      - X_index[i,j]: row index in the match dataframe for that game

    IMPORTANT: this uses **only matches from that season** to train Elo,
    so e.g. 2019-20 does NOT use any information from later seasons.
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

    # Season-only Elo ratings
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

        # ----- odds-based -----
        oh, od, oa = row[h_col], row[d_col], row[a_col]
        ph_odds, _, pa_odds = probs_from_odds(oh, od, oa)

        # ----- Elo expected (before update) -----
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
            s_home = s_away = 0.5  # weird code, treat as draw

        ratings[home] = Rh + k_elo * (s_home - exp_home)
        ratings[away] = Ra + k_elo * (s_away - exp_away)

        games_played_by_team[j_home] += 1
        games_played_by_team[j_away] += 1

    X_elo = np.clip(X_elo, 1e-6, 0.999999)
    return X_odds, X_elo, R, teams, X_index, df


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


# ---------- New: decay-based permutation + detailed run report ----------

def perm_from_decay(X: np.ndarray, decay: float) -> list[int]:
    """
    Build a season-long permutation of teams based on a 'front-load strength' decay.

    For each team j, define a score:

        score_j = sum_i [ (decay**i) * log P_ij ]

    Then pick teams in descending score order.
    """
    num_rounds, _ = X.shape
    logX = np.log(np.clip(X, 1e-12, 1.0))

    weights = decay ** np.arange(num_rounds)  # shape (num_rounds,)
    scores = (weights[:, None] * logX).sum(axis=0)

    perm = list(np.argsort(-scores))
    return perm


def detailed_run_report(
    perm,
    X: np.ndarray,
    R: np.ndarray,
    teams: list[str],
    X_index: np.ndarray,
    df_matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-week report with opponents, venue, probs, actual result and survival flag
    for a specific permutation (season+settings combo).
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


# ---------- Main Streamlit app ----------

def main():
    st.set_page_config(page_title="LMS Planner 25/26", layout="wide")

    tab_plan, tab_backtest = st.tabs(["📅 Current season planner", "📈 Backtest previous seasons"])

    # ===== TAB 1: CURRENT SEASON PLANNER =====
    with tab_plan:
        st.title("⚽ Last Man Standing Planner – Premier League 2025/26")

        st.markdown(
            """
            This tab uses **5 past seasons + current 2025/26 results** to build an Elo model,  
            then blends that with **bookmaker odds (25/26)** where available, and plans your LMS picks.

            It automatically downloads the latest:

            - `E0_2025-26.csv` from football-data (results + odds)  
            - `E0_2025-26_fixtures.csv` from fixturedownload (all fixtures / rounds)
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

        @st.cache_data(show_spinner=True)
        def _build_model_cached(w_odds_: float):
            return build_X_elo_plus_odds(
                fixtures_path,
                odds_path,
                hist_files,
                home_adv=60.0,
                w_odds=w_odds_,
            )

        fixtures, teams, X, bookie, merged = _build_model_cached(float(w_odds))

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
            "- `decay` controls how much **early rounds matter more**.\n\n"
            "**Guideline:**\n"
            "- 0.80–0.88 → very conservative early survival\n"
            "- 0.90–0.93 → balanced (recommended)\n"
            "- 0.95–0.99 → almost full-season optimal, less front-loading"
        )

        st.sidebar.write(f"Using bookie: **{bookie}**")
        if len(merged) > 0:
            st.sidebar.write(
                f"Odds currently cover rounds **{int(merged['Round'].min())} → "
                f"{int(merged['Round'].max())}**"
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

    # ===== TAB 2: BACKTEST PREVIOUS SEASONS =====
    with tab_backtest:
        st.title("📈 Backtest LMS Strategies on Previous Seasons")

        st.markdown(
            """
            This tab uses **historical season CSVs** from `data/raw` and:

            - An **odds-only model** for quick greedy backtests  
            - A season-only **Elo + odds** model so we can tune:

              • `w_odds` – odds vs Elo blend  
              • `decay`  – front-load strength  

            We then see, for each season, which combination gives the longest run,
            and also the **single longest run of all time** with full details.
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
        else:
            season_label = st.selectbox(
                "Select season to inspect (quick reports use odds-only model)",
                options=list(season_files.keys()),
                index=len(season_files) - 1,
            )
            bookie_prefix = st.selectbox(
                "Bookmaker (1X2 odds prefix)",
                options=["B365"],
                index=0,
            )

            if st.button("🚀 Run quick backtest for selected season",
                         type="primary", key="backtest_button"):
                season_path = os.path.join(RAW_DIR, season_files[season_label])

                try:
                    X_bt, R_bt, teams_bt, X_idx_bt, df_bt = build_X_R_from_bookie(
                        season_path,
                        bookie_prefix=bookie_prefix,
                    )
                except Exception as e:
                    st.error(f"Error building season matrices: {e}")
                else:
                    perm = greedy_perm(X_bt)
                    weeks_survived = survival_from_perm(perm, R_bt)
                    report = perm_report_df(perm, X_bt, R_bt, teams_bt)

                    st.subheader(f"🧮 Greedy odds-only backtest for {season_label}")
                    st.write(f"Teams in league: **{len(teams_bt)}**")
                    st.write(f"Greedy LMS (odds-only) survived: **{weeks_survived} weeks**")

                    if not report.empty:
                        st.subheader("Per-week greedy LMS detail")
                        st.dataframe(report)

                        csv_bt = report.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Download backtest report as CSV",
                            data=csv_bt,
                            file_name=f"lms_backtest_{season_label}_{bookie_prefix}.csv",
                            mime="text/csv",
                            key="backtest_download",
                        )

                    st.subheader("🏆 Top 3 odds-only picks for each 'round'")
                    top3_df = top3_per_round_df(X_bt, X_idx_bt, df_bt, teams_bt)
                    if top3_df.empty:
                        st.info("Could not compute top 3 per round (missing odds or fixtures).")
                    else:
                        st.dataframe(top3_df)
                        csv_top3 = top3_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Download Top 3 per round as CSV",
                            data=csv_top3,
                            file_name=f"lms_top3_per_round_{season_label}_{bookie_prefix}.csv",
                            mime="text/csv",
                            key="top3_download",
                        )

        st.markdown("---")
        st.subheader("🎛 Settings search: Elo/odds blend + front-load strength")

        st.markdown(
            """
            This section replays the last 5 seasons using a **season-only Elo+odds model**.

            For each season and each `(w_odds, decay)` combination we:
            - Build blended probabilities from Elo + odds using that `w_odds`  
            - Build a full-season LMS order using that `decay`  
            - See how many weeks you **actually** survive using real results  

            Then we:
            - Show the best settings per season  
            - Show which settings have the best **average** survival over all seasons  
            - Show the **single longest run of all time** with full per-week details.
            """
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            w_min = st.number_input("Min w_odds", 0.0, 1.0, 0.50, 0.05)
        with col2:
            w_max = st.number_input("Max w_odds", 0.0, 1.0, 0.95, 0.05)
        with col3:
            w_step = st.number_input("w_odds step", 0.01, 0.5, 0.10, 0.01)

        col4, col5, col6 = st.columns(3)
        with col4:
            decay_min = st.number_input("Min decay", 0.70, 0.999, 0.80, 0.01)
        with col5:
            decay_max = st.number_input("Max decay", 0.70, 0.999, 0.98, 0.01)
        with col6:
            decay_step = st.number_input("Decay step", 0.005, 0.05, 0.02, 0.005)

        if st.button("🔍 Search best (w_odds, decay) across all seasons",
                     key="search_settings_all"):
            w_values = np.arange(w_min, w_max + 1e-9, w_step)
            decay_values = np.arange(decay_min, decay_max + 1e-9, decay_step)

            if len(w_values) == 0 or len(decay_values) == 0:
                st.error("Grids are empty – check min/max/step values.")
            else:
                # Precompute components per season (season-only Elo + odds)
                season_components = {}
                for lab, fname in season_files.items():
                    season_path = os.path.join(RAW_DIR, fname)
                    try:
                        X_odds_s, X_elo_s, R_s, teams_s, X_idx_s, df_s = build_season_prob_components(
                            season_path,
                            bookie_prefix=bookie_prefix,
                        )
                    except Exception as e:
                        st.warning(f"Skipping {lab}: error building components → {e}")
                        continue
                    season_components[lab] = (X_odds_s, X_elo_s, R_s,
                                              teams_s, X_idx_s, df_s)

                if not season_components:
                    st.error("Could not build components for any season.")
                else:
                    rows = []

                    # Track single best run of all time
                    best_run_weeks = -1
                    best_run_info = None

                    # Loop over all combos
                    for w in w_values:
                        for d in decay_values:
                            for lab, (X_odds_s, X_elo_s, R_s,
                                      teams_s, X_idx_s, df_s) in season_components.items():
                                X_blend = blend_X_odds_elo(X_odds_s, X_elo_s, float(w))
                                perm = perm_from_decay(X_blend, float(d))
                                weeks = survival_from_perm(perm, R_s)

                                rows.append(
                                    dict(
                                        Season=lab,
                                        w_odds=float(w),
                                        Decay=float(d),
                                        WeeksSurvived=int(weeks),
                                    )
                                )

                                # Update global best run
                                if weeks > best_run_weeks:
                                    best_run_weeks = weeks
                                    best_run_info = dict(
                                        Season=lab,
                                        w_odds=float(w),
                                        Decay=float(d),
                                        Weeks=int(weeks),
                                        Perm=perm,
                                        X_blend=X_blend,
                                        R=R_s,
                                        Teams=teams_s,
                                        X_index=X_idx_s,
                                        Matches=df_s,
                                    )

                    results_df = pd.DataFrame(rows)
                    if results_df.empty:
                        st.error("No results computed – something went wrong.")
                    else:
                        # Layout: summary on left, best-run detail on right
                        colL, colR = st.columns([2, 1])

                        with colL:
                            st.subheader("Full settings grid (all seasons)")
                            st.dataframe(results_df)

                            # Best per season
                            best_per_season = (
                                results_df.sort_values(
                                    ["Season", "WeeksSurvived"],
                                    ascending=[True, False],
                                )
                                .groupby("Season")
                                .head(1)
                                .reset_index(drop=True)
                            )
                            st.subheader("🏅 Best settings per season")
                            st.dataframe(best_per_season)

                            # Best overall by mean survival
                            avg_over_seasons = (
                                results_df.groupby(["w_odds", "Decay"])["WeeksSurvived"]
                                .mean()
                                .reset_index(name="MeanWeeksAcrossSeasons")
                            )
                            best_overall = avg_over_seasons.loc[
                                avg_over_seasons["MeanWeeksAcrossSeasons"].idxmax()
                            ]
                            st.subheader("🌍 Best settings by average survival")
                            st.success(
                                f"Best overall: **w_odds = {best_overall['w_odds']:.2f}**, "
                                f"**decay = {best_overall['Decay']:.3f}**, "
                                f"mean survival = "
                                f"**{best_overall['MeanWeeksAcrossSeasons']:.2f} weeks**"
                            )
                            st.dataframe(
                                avg_over_seasons.sort_values(
                                    "MeanWeeksAcrossSeasons", ascending=False
                                )
                            )

                        with colR:
                            st.subheader("🏆 Longest run of all time")
                            if best_run_info is None:
                                st.info("No valid runs found.")
                            else:
                                st.markdown(
                                    f"**Season:** {best_run_info['Season']}  \n"
                                    f"**w_odds:** {best_run_info['w_odds']:.2f}  \n"
                                    f"**decay:** {best_run_info['Decay']:.3f}  \n"
                                    f"**Weeks survived:** {best_run_info['Weeks']}"
                                )

                                run_df = detailed_run_report(
                                    best_run_info["Perm"],
                                    best_run_info["X_blend"],
                                    best_run_info["R"],
                                    best_run_info["Teams"],
                                    best_run_info["X_index"],
                                    best_run_info["Matches"],
                                )
                                st.dataframe(run_df)

                        # Downloads
                        csv_all = results_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Download full settings grid (per season)",
                            data=csv_all,
                            file_name="lms_settings_search_all_seasons.csv",
                            mime="text/csv",
                            key="settings_all_download",
                        )

                        csv_best_season = best_per_season.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Download best-per-season settings",
                            data=csv_best_season,
                            file_name="lms_best_settings_per_season.csv",
                            mime="text/csv",
                            key="settings_best_season_download",
                        )

                        csv_mean = avg_over_seasons.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Download mean survival per (w_odds, decay)",
                            data=csv_mean,
                            file_name="lms_settings_mean_across_seasons.csv",
                            mime="text/csv",
                            key="settings_mean_download",
                        )


if __name__ == "__main__":
    main()
