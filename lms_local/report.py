from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_survival_curves(out_dir: str, tag: str, dfs: dict[str, pd.DataFrame]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    for name, df in dfs.items():
        max_k = int(df["survived"].max())
        ks = list(range(0, max_k + 1))
        probs = [(df["survived"] >= k).mean() for k in ks]
        plt.plot(ks, probs, label=name)
    plt.xlabel("Rounds survived (k)")
    plt.ylabel("P(survive ≥ k)")
    plt.title(f"Survival curves - {tag}")
    plt.legend()
    path = os.path.join(out_dir, f"survival_{tag}.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path
