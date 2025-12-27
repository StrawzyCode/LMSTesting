from __future__ import annotations

import os
import pandas as pd

from lms_local.fixtures2526 import load_fixture_results_csv, list_teams
from lms_local.elo import elo_train
from lms_local.future_matrix import build_X_from_elo
from lms_local.merge_rounds import merge_rounds
from lms_local.odds import choose_bookie, apply_odds_to_X
from lms_local.planner import plan_season_frontload

RAW_DIR = "data/raw"
OUT_DIR = "data/outputs"

def main():
    fixtures_path = os.path.join(RAW_DIR, "E0_2025-26_fixtures.csv")  # has Round Number
    odds_path     = os.path.join(RAW_DIR, "E0_2025-26.csv")          # football-data (odds + results so far)

    start_round = 18
    decay = 0.90

    print("Loading fixtures:", fixtures_path)
    fixtures = load_fixture_results_csv(fixtures_path)
    teams = list_teams(fixtures)
    print("Teams:", len(teams))

    # 5 seasons history
    hist_files = [
        "E0_2020-21.csv",
        "E0_2021-22.csv",
        "E0_2022-23.csv",
        "E0_2023-24.csv",
        "E0_2024-25.csv",
    ]
    hist_paths = [os.path.join(RAW_DIR, f) for f in hist_files]
    dfs = []
    for p in hist_paths:
        df = pd.read_csv(p, encoding="latin-1")
        dfs.append(df[["HomeTeam", "AwayTeam", "FTR"]].dropna())

    # add current season results-so-far to train Elo more accurately
    cur = pd.read_csv(odds_path, encoding="latin-1")
    if {"HomeTeam","AwayTeam","FTR"}.issubset(cur.columns):
        cur_res = cur[["HomeTeam","AwayTeam","FTR"]].dropna()
        dfs.append(cur_res)

    hist = pd.concat(dfs, ignore_index=True)
    print("Training matches for Elo:", len(hist))

    elo = elo_train(hist, k=20.0, home_adv=60.0)

    # Base predictions for all 38 rounds from Elo
    X = build_X_from_elo(fixtures, teams, elo, home_adv=60.0)
    print("Base X shape (Elo):", X.shape)

    # Merge rounds into odds rows (fixes your “round 16 only” issue if caused by name mismatch)
    merged = merge_rounds(fixtures_path, odds_path)
    print("Odds rows matched to rounds:", len(merged))
    if len(merged) > 0:
        print("Rounds covered by odds:", merged["Round"].min(), "->", merged["Round"].max())

    bookie = choose_bookie(merged, prefer=["B365", "PS"]) if len(merged) else "B365"
    print("Using bookie:", bookie)

    # Overwrite Elo probs with odds-implied probs where we actually have odds
    X = apply_odds_to_X(merged, X, teams, bookie)

    # Put your actual picks from rounds 1-17 here (VERY IMPORTANT if you are continuing a real LMS run)
    already_used = set([
        # "Arsenal", "Liverpool", ...
    ])

    plan = plan_season_frontload(
        fixtures=fixtures,
        X=X,
        teams=teams,
        start_round=start_round,
        forbidden_teams=already_used,
        decay=decay,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"picks_2526_gw{start_round}_frontload_oddspluselo_{bookie}.csv")
    plan.to_csv(out_path, index=False)

    print("\nFirst 10 picks from GW", start_round)
    print(plan.head(10).to_string(index=False))
    print("\nSaved:", out_path)

if __name__ == "__main__":
    main()
