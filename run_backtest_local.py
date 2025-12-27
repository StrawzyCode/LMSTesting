from __future__ import annotations

import glob
import os
import pandas as pd

from lms_local.season import detect_teams, detect_3way_bookies, build_round_team_matrices
from lms_local.strategies import (
    greedy,
    greedy_sampling_best_of_N,
    simulated_annealing_best_of_N,
    lp_assignment_cvxpy,
    hungarian_assignment,
)
from lms_local.backtest import backtest_over_season, summarize
from lms_local.report import plot_survival_curves

RAW_DIR = "data/raw"
OUT_DIR = "data/outputs"

# comparison sets
PROB_METHODS = ["implied_normalized", "proportional_vig"]
PREFERRED_BOOKIE_ORDER = ["B365", "PS", "BW", "WH", "LB", "BFD", "BMGM", "BVD", "CL", "BFE", "Max", "Avg"]

# budgets (tune if too slow)
GS_ITERS = 20000
SA_RUNS = 200
SA_BURNIN = 250
SA_BETA_I = 0.2
SA_BETA_F = 50.0
SA_ALPHA = 1.15

MATCHES_PER_ROUND = 10  # EPL

def pick_bookie(bookies_in_file: list[str]) -> str:
    for b in PREFERRED_BOOKIE_ORDER:
        if b in bookies_in_file:
            return b
    # fallback: first
    return bookies_in_file[0]

def load_csv(path: str) -> pd.DataFrame:
    # latin-1 avoids weird characters in Referee etc.
    return pd.read_csv(path, encoding="latin-1")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not files:
        raise SystemExit(f"No CSVs found in {RAW_DIR}. Put your season files there.")

    summary_rows = []

    for path in files:
        tag = os.path.splitext(os.path.basename(path))[0]
        print(f"\n=== {tag} ===")
        df = load_csv(path)

        # quick sanity check for required headers you listed
        required = {"Date","HomeTeam","AwayTeam","FTR"}
        missing = required - set(df.columns)
        if missing:
            print(f"[skip] missing required columns: {missing}")
            continue

        teams = detect_teams(df)
        bookies = detect_3way_bookies(df)
        if not bookies:
            print("[skip] no 3-way bookie columns found (..H/..D/..A).")
            continue

        bookie = pick_bookie(bookies)
        print(f"teams={len(teams)} | bookies_found={len(bookies)} | using_bookie={bookie}")

        for prob_method in PROB_METHODS:
            X, R = build_round_team_matrices(df, teams, bookie, prob_method, matches_per_round=MATCHES_PER_ROUND)

            # define strategies
            def s_greedy(X_, start): return greedy(X_, start)
            def s_gs(X_, start): return greedy_sampling_best_of_N(X_, start, iters=GS_ITERS, seed=1234)
            def s_sa(X_, start): return simulated_annealing_best_of_N(
                X_, start, runs=SA_RUNS, burn_in=SA_BURNIN, beta_i=SA_BETA_I, beta_f=SA_BETA_F, alpha=SA_ALPHA, seed=2025
            )
            def s_lp(X_, start): return lp_assignment_cvxpy(X_, start)
            def s_hung(X_, start): return hungarian_assignment(X_, start)

            strategies = {
                "greedy": s_greedy,
                "greedy_sampling": s_gs,
                "sim_anneal": s_sa,
                "lp_cvxpy": s_lp,
                "hungarian": s_hung,
            }

            dfs = {}
            for name, fn in strategies.items():
                print(f"  backtesting {name} | {prob_method}")
                bt = backtest_over_season(X, R, fn)
                dfs[name] = bt
                row = {
                    "season_file": tag,
                    "bookie": bookie,
                    "prob_method": prob_method,
                    "strategy": name,
                    **summarize(bt),
                }
                summary_rows.append(row)

            plot_tag = f"{tag}_{bookie}_{prob_method}"
            plot_path = plot_survival_curves(OUT_DIR, plot_tag, dfs)
            print(f"  saved plot: {plot_path}")

    # save summary table
    if summary_rows:
        out_csv = os.path.join(OUT_DIR, "summary.csv")
        pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
        print(f"\nSaved: {out_csv}")

        df_sum = pd.read_csv(out_csv)
        leaderboard = (
            df_sum.groupby(["strategy"])
            .agg(mean_survived=("mean_survived","mean"),
                 p10=("p_survive_10+","mean"),
                 p15=("p_survive_15+","mean"))
            .sort_values(["mean_survived","p10","p15"], ascending=False)
        )
        print("\n=== Leaderboard (averaged across all files & prob methods) ===")
        print(leaderboard.round(4).to_string())

if __name__ == "__main__":
    main()
