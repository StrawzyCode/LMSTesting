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

# URLs you gave
URL_ODDS = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
URL_FIX = "https://fixturedownload.com/download/csv/epl-2025"


# ---------- Helpers ----------

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

        # Blend Elo and odds only where odds exist
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


# ---------- Streamlit App ----------

def main():
    st.set_page_config(page_title="LMS Planner 25/26", layout="wide")
    st.title("⚽ Last Man Standing Planner – Premier League 2025/26")

    st.markdown(
        """
        This app uses **5 past seasons + current 2025/26 results** to build an Elo model,  
        then blends that with **bookmaker odds (25/26)** where available, and plans your LMS picks.

        It automatically downloads the latest:

        - `E0_2025-26.csv` from football-data (results + odds)  
        - `E0_2025-26_fixtures.csv` from fixturedownload (all fixtures / rounds)
        """
    )

    fixtures_path = os.path.join(RAW_DIR, "E0_2025-26_fixtures.csv")
    odds_path = os.path.join(RAW_DIR, "E0_2025-26.csv")

    with st.spinner("Downloading latest fixtures + odds…"):
        download_if_new(URL_FIX, fixtures_path)
        download_if_new(URL_ODDS, odds_path)

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

    # -------- SIDEBAR CONTROLS ----------
    st.sidebar.header("Settings")

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

    # -------- MAIN ACTION ----------
    if st.button("🔮 Compute next pick", type="primary"):
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
            return

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
        )

    else:
        st.info("Pick a round + used teams on the left, then click **Compute next pick**.")


if __name__ == "__main__":
    main()
