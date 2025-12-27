from __future__ import annotations

import os
import pandas as pd

from lms_local.fixtures2526 import load_fixture_results_csv, list_teams
from lms_local.elo import elo_train
from lms_local.future_matrix import build_X_from_elo
from lms_local.planner import plan_season_optimal, plan_season_frontload

RAW_DIR = "data/raw"
OUT_DIR = "data/outputs"


def main():
    print("✅ run_plan_2526.py started")

    fixture_path = os.path.join(RAW_DIR, "E0_2025-26_fixtures.csv")
    print("Looking for fixtures at:", fixture_path)
    if not os.path.exists(fixture_path):
        raise FileNotFoundError(f"Fixtures file not found: {fixture_path}")

    fixtures = load_fixture_results_csv(fixture_path)
    print("Fixtures loaded:", len(fixtures), "rows")
    print("Rounds in file:", fixtures["Round"].min(), "->", fixtures["Round"].max())

    # Last 5 seasons (edit these filenames if yours differ)
    hist_files = [
        "E0_2020-21.csv",
        "E0_2021-22.csv",
        "E0_2022-23.csv",
        "E0_2023-24.csv",
        "E0_2024-25.csv",
    ]
    hist_paths = [os.path.join(RAW_DIR, f) for f in hist_files]
    for p in hist_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing historical file: {p}")

    dfs = []
    for p in hist_paths:
        df = pd.read_csv(p, encoding="latin-1")
        need = {"HomeTeam", "AwayTeam", "FTR"}
        if not need.issubset(df.columns):
            raise ValueError(f"{os.path.basename(p)} missing required columns {need - set(df.columns)}")
        dfs.append(df[["HomeTeam", "AwayTeam", "FTR"]].dropna())

    hist = pd.concat(dfs, ignore_index=True)
    print("Historical matches loaded:", len(hist))

    # Train Elo
    elo = elo_train(hist, k=20.0, home_adv=60.0)
    print("Elo trained for teams:", len(elo))

    # Build probabilities for 25/26 fixtures
    teams = list_teams(fixtures)
    print("Teams in 25/26 fixture file:", len(teams))

    X = build_X_from_elo(fixtures, teams, elo, home_adv=60.0)
    print("X matrix shape:", X.shape)

    # Plans
    opt = plan_season_optimal(fixtures, X, teams)
    front = plan_season_frontload(fixtures, X, teams, decay=0.90)  # try 0.92/0.90/0.88

    os.makedirs(OUT_DIR, exist_ok=True)
    opt_path = os.path.join(OUT_DIR, "picks_2526_optimal.csv")
    front_path = os.path.join(OUT_DIR, "picks_2526_frontload.csv")

    opt.to_csv(opt_path, index=False)
    front.to_csv(front_path, index=False)

    print("\n📌 OPTIMAL (max whole-season survival) – first 15")
    print(opt.head(15).to_string(index=False))

    print("\n📌 FRONTLOAD (best early survival) – first 15")
    print(front.head(15).to_string(index=False))

    print("\nWhole-season survival proxy (product of p):")
    print("OPTIMAL:", float(opt["cum_survival_proxy"].iloc[-1]))
    print("FRONTLOAD:", float(front["cum_survival_proxy"].iloc[-1]))

    print("\nEarly survival proxy (first 10 rounds):")
    i10_opt = min(9, len(opt) - 1)
    i10_front = min(9, len(front) - 1)
    print("OPTIMAL:", float(opt["cum_survival_proxy"].iloc[i10_opt]))
    print("FRONTLOAD:", float(front["cum_survival_proxy"].iloc[i10_front]))

    print("\n✅ Saved:", opt_path)
    print("✅ Saved:", front_path)


if __name__ == "__main__":
    main()
