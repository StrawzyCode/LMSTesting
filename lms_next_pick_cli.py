from __future__ import annotations

import os
import sys
import pandas as pd

from lms_local.fixtures2526 import load_fixture_results_csv, list_teams
from lms_local.elo import elo_train
from lms_local.future_matrix import build_X_from_elo
from lms_local.merge_rounds import merge_rounds
from lms_local.odds import choose_bookie, apply_odds_to_X
from lms_local.planner import plan_season_frontload

RAW_DIR = "data/raw"


def ask_int(prompt: str, default: int | None = None) -> int:
    while True:
        s = input(f"{prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
        if not s and default is not None:
            return default
        try:
            return int(s)
        except ValueError:
            print("Please enter an integer.")


def ask_used_teams(teams: list[str]) -> set[str]:
    print("\nAvailable teams:")
    print(", ".join(sorted(teams)))
    print(
        "\nEnter teams you've ALREADY used so far in this LMS run, "
        "comma-separated (e.g. 'Arsenal,Liverpool,Spurs'). "
        "Leave blank if none / testing."
    )
    s = input("Already-used teams: ").strip()
    if not s:
        return set()
    raw = [x.strip() for x in s.split(",") if x.strip()]
    # try to match case-insensitively
    norm_map = {t.lower(): t for t in teams}
    used: set[str] = set()
    for r in raw:
        key = r.lower()
        if key in norm_map:
            used.add(norm_map[key])
        else:
            print(f"  ⚠️  Warning: '{r}' not recognised, ignoring.")
    return used


def build_X_elo_plus_odds(
    fixtures_path: str,
    odds_path: str,
    hist_files: list[str],
    home_adv: float = 60.0,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame, str, pd.DataFrame, "np.ndarray"]:
    """
    1) Load fixtures (with rounds)
    2) Train Elo on 5 seasons + current season so far
    3) Build X from Elo for all rounds
    4) Merge odds to attach rounds to odds rows
    5) Overwrite X with odds-implied probs where odds exist
    """
    import numpy as np

    print(f"\n📂 Loading fixtures from: {fixtures_path}")
    fixtures = load_fixture_results_csv(fixtures_path)
    teams = list_teams(fixtures)
    print(f"Teams in fixtures file: {len(teams)}")
    print(", ".join(sorted(teams)))

    # Train Elo
    dfs = []
    for fname in hist_files:
        path = os.path.join(RAW_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Historical file missing: {path}")
        df = pd.read_csv(path, encoding="latin-1")
        dfs.append(df[["HomeTeam", "AwayTeam", "FTR"]].dropna())

    # add current season results from odds file
    cur = pd.read_csv(odds_path, encoding="latin-1")
    if {"HomeTeam", "AwayTeam", "FTR"}.issubset(cur.columns):
        cur_res = cur[["HomeTeam", "AwayTeam", "FTR"]].dropna()
        dfs.append(cur_res)

    hist = pd.concat(dfs, ignore_index=True)
    print(f"Training matches for Elo: {len(hist)}")
    elo = elo_train(hist, k=20.0, home_adv=home_adv)
    print(f"Elo trained teams: {len(elo)}")

    # Base matrix from Elo
    X = build_X_from_elo(fixtures, teams, elo, home_adv=home_adv)
    print(f"Elo base X shape: {X.shape}")

    # Merge odds with rounds
    print(f"\n📂 Merging odds from: {odds_path}")
    merged = merge_rounds(fixtures_path, odds_path)
    print(f"Odds rows matched to fixture rounds: {len(merged)}")
    if len(merged):
        print("Rounds covered by odds:", int(merged["Round"].min()), "->", int(merged["Round"].max()))

    # Choose bookie + overwrite X where odds exist
    bookie = choose_bookie(merged, prefer=["B365", "PS"]) if len(merged) else "B365"
    print("Using bookie:", bookie)

    X = apply_odds_to_X(merged, X, teams, bookie)
    # sanity: clamp probs
    X = np.clip(X, 1e-6, 0.999999)

    return fixtures, teams, merged, bookie, hist, X


def main():
    print("=== Last Man Standing Next-Pick Helper ===")

    # Hard-coded season files for now
    fixtures_path = os.path.join(RAW_DIR, "E0_2025-26_fixtures.csv")  # has Round Number
    odds_path = os.path.join(RAW_DIR, "E0_2025-26.csv")  # football-data season file

    for p in [fixtures_path, odds_path]:
        if not os.path.exists(p):
            print(f"❌ File not found: {p}")
            print("Make sure you put the fixtures and E0_2025-26.csv into data/raw")
            sys.exit(1)

    hist_files = [
        "E0_2020-21.csv",
        "E0_2021-22.csv",
        "E0_2022-23.csv",
        "E0_2023-24.csv",
        "E0_2024-25.csv",
    ]

    fixtures, teams, merged, bookie, hist, X = build_X_elo_plus_odds(
        fixtures_path=fixtures_path,
        odds_path=odds_path,
        hist_files=hist_files,
        home_adv=60.0,
    )

    # Ask current gameweek
    print("\nWhat gameweek (round) are you about to pick for?")
    print("For example, if you're about to make your GW18 selection, type 18.")
    start_round = ask_int("Round number", default=18)

    # Ask which teams you've already used in this LMS
    already_used = ask_used_teams(teams)
    if already_used:
        print("You marked these as already used:")
        print(", ".join(sorted(already_used)))
    else:
        print("You marked no already-used teams (test / first run).")

    # Run front-loaded planner from this round forward
    decay = 0.90
    print(f"\n🧮 Planning from round {start_round} onwards with frontload decay={decay} ...")
    plan = plan_season_frontload(
        fixtures=fixtures,
        X=X,
        teams=teams,
        start_round=start_round,
        forbidden_teams=already_used,
        decay=decay,
    )

    if plan.empty:
        print("\n❌ No plan produced. Possible reasons:")
        print("- start_round is beyond the number of rounds in fixtures, or")
        print("- you have already used too many teams to make a valid pick.")
        return

    # Show next pick and a few backups
    print(f"\n✅ Recommended pick for round {start_round}:")
    row0 = plan.iloc[0]
    print(
        f"  -> {row0['Pick']} vs {row0['Opponent']} "
        f"({'home' if row0['Venue']=='H' else 'away'}) "
        f"with model P(win) ≈ {row0['P_win_model']:.3f}"
    )

    print("\nTop 5 planned picks from this round onwards:")
    print(plan.head(5).to_string(index=False))

    # Save full plan
    os.makedirs("data/outputs", exist_ok=True)
    out_path = os.path.join(
        "data/outputs",
        f"picks_next_from_gw{start_round}_frontload_{bookie}.csv",
    )
    plan.to_csv(out_path, index=False)
    print("\n📁 Full plan saved to:", out_path)


if __name__ == "__main__":
    main()
