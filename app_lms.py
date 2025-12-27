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
    Convert 1X2 decimal odds to implied probabilities with overround removed,
    as in the original blog.
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
        # Apply odds on top of Elo to get "odds-only" matrix
        X_odds_applied = apply_odds_to_X(merged, X_elo.copy(), teams, bookie)

        # Where X_odds_applied differs from X_elo, odds were used
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

    # If a team appears twice somehow, keep max prob
    df = df.sort_values("P_win_model", ascending=False)
    df = df.drop_duplicates(subset=["Pick"], keep="first")

    df["logP"] = np.log(np.clip(df["P_win_model"], 1e-12, 1.0))
    df["cum_logP"] = df["logP"]          # single-round
    df["cum_survival_proxy"] = np.exp(df["cum_logP"])
    return df.sort_values("P_win_model", ascending=False)


# ---------- Backtest helpers (past seasons, odds-only model) ----------

def build_X_R_from_bookie(season_path: str,
                          bookie_prefix: str = "B365"):
    """
    Build (X, R, teams) from a historical season CSV using a given 1X2 bookie prefix.
    X[i, j] : model P(team j wins in its i-th league game)
    R[i, j] : 1 if team j actually won that game, else 0.

    This follows the spirit of the original blog's build_X, but also builds results.
    """
    df = pd.read_csv(season_path, encoding="latin-1")

    # Basic sanity
    if not {"HomeTeam", "AwayTeam", "FTR"}.issubset(df.columns):
        raise ValueError(f"{season_path} missing HomeTeam/AwayTeam/FTR")

    # Sort by date so "game index" is chronological
    if "Date" in df.columns:
        df = df.copy()
        df["__Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.sort_values("__Date").drop(columns="__Date")

    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    num_teams = len(teams)
    num_rounds = 2 * (num_teams - 1)  # EPL: 38

    X = np.zeros((num_rounds, num_teams), dtype=float)
    R = np.zeros((num_rounds, num_teams), dtype=int)

    games_played_by_team = np.zeros(num_teams, dtype=int)

    h_col = bookie_prefix + "H"
    d_col = bookie_prefix + "D"
    a_col = bookie_prefix + "A"

    if not {h_col, d_col, a_col}.issubset(df.columns):
        raise ValueError(f"{season_path} missing {bookie_prefix}H/D/A columns")

    for _, row in df.iterrows():
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

        # Actual results: 1 if win, else 0
        R[i_home, j_home] = 1 if ftr == "H" else 0
        R[i_away, j_away] = 1 if ftr == "A" else 0

        games_played_by_team[j_home] += 1
        games_played_by_team[j_away] += 1

    # Clip probabilities to sane range
    X = np.clip(X, 1e-6, 0.999999)
    return X, R, teams


def greedy_perm(X: np.ndarray):
    """
    Simple greedy LMS strategy over a whole season:
    at each 'round' i, pick the remaining team with highest model P(win).
    """
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

        # Historical seasons
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
                f"Odds currently cover rounds **{int(merged['Round'].min())} → {int(merged['Round'].max())}**"
            )

        if st.button("🔮 Compute next pick", type="primary", key="plan_button"):
            forbidden = set(used_teams)

            # Global front-loaded plan (for best overall choice)
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

                # --- Top 5 options in THIS round (not just the chosen one) ---
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
                            ["Round", "Pick", "Opponent", "Venue", "P_win_model", "cum_survival_proxy"]
                        ]
                    )

                # --- Full future plan ---
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
            This tab uses **historical season CSVs** from `data/raw` and a simple  
            **B365 implied-probability model** (like in the original blog) to see:

            - What order of teams the greedy LMS strategy would choose
            - How many weeks you would actually survive in that season
            - Per-week comparison: model P(win) vs actual W/L
            """
        )

        # Available seasons for backtest
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
                "Select season to backtest",
                options=list(season_files.keys()),
                index=len(season_files) - 1,  # default latest
            )
            bookie_prefix = st.selectbox(
                "Bookmaker (1X2 odds prefix)",
                options=["B365"],
                index=0,
            )

            if st.button("🚀 Run backtest", type="primary", key="backtest_button"):
                season_path = os.path.join(RAW_DIR, season_files[season_label])

                try:
                    X_bt, R_bt, teams_bt = build_X_R_from_bookie(
                        season_path,
                        bookie_prefix=bookie_prefix,
                    )
                except Exception as e:
                    st.error(f"Error building season matrices: {e}")
                else:
                    perm = greedy_perm(X_bt)
                    weeks_survived = survival_from_perm(perm, R_bt)
                    report = perm_report_df(perm, X_bt, R_bt, teams_bt)

                    st.subheader(f"🧮 Backtest result for {season_label} ({bookie_prefix})")
                    st.write(f"Teams in league: **{len(teams_bt)}**")
                    st.write(f"Greedy LMS strategy survived: **{weeks_survived} weeks** (max {len(perm)})")

                    if not report.empty:
                        st.subheader("Per-week detail")
                        st.dataframe(report)

                        csv_bt = report.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Download backtest report as CSV",
                            data=csv_bt,
                            file_name=f"lms_backtest_{season_label}_{bookie_prefix}.csv",
                            mime="text/csv",
                            key="backtest_download",
                        )
                    else:
                        st.info("No usable data for this season with the chosen bookie/odds.")

            else:
                st.info("Pick a season and click **Run backtest** to compare model vs actual results.")


if __name__ == "__main__":
    main()
