from __future__ import annotations
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

def safe_log(p: float, eps: float = 1e-12) -> float:
    if p is None or not np.isfinite(p):
        return math.log(eps)
    return math.log(max(eps, min(1.0, float(p))))

def eval_perm_logsum(X: np.ndarray, start_round: int, perm: list[int]) -> float:
    s = 0.0
    for i, t in enumerate(perm):
        r = start_round + i
        if r >= X.shape[0]:
            break
        s += safe_log(X[r, t])
    return s

def greedy(X: np.ndarray, start_round: int) -> list[int]:
    n_rounds, n_teams = X.shape
    used = set()
    horizon = min(n_teams, n_rounds - start_round)
    picks = []
    for i in range(horizon):
        r = start_round + i
        best = None
        bestp = -1.0
        for t in range(n_teams):
            if t in used:
                continue
            p = X[r, t]
            if np.isfinite(p) and p > bestp:
                bestp = float(p)
                best = t
        if best is None:
            # fallback: any unused
            best = next(t for t in range(n_teams) if t not in used)
        used.add(best)
        picks.append(best)
    return picks

def greedy_sample_once(X: np.ndarray, start_round: int, rng: np.random.Generator) -> list[int]:
    n_rounds, n_teams = X.shape
    used = set()
    horizon = min(n_teams, n_rounds - start_round)
    picks = []
    for i in range(horizon):
        r = start_round + i
        candidates = [t for t in range(n_teams) if t not in used]
        probs = np.array([X[r, t] if np.isfinite(X[r, t]) else 0.0 for t in candidates], dtype=float)
        s = probs.sum()
        if s <= 0:
            pick = int(rng.choice(candidates))
        else:
            probs = probs / s
            pick = int(rng.choice(candidates, p=probs))
        used.add(pick)
        picks.append(pick)
    return picks

def greedy_sampling_best_of_N(X: np.ndarray, start_round: int, iters: int, seed: int = 1234) -> list[int]:
    rng = np.random.default_rng(seed)
    best = greedy(X, start_round)
    bestv = eval_perm_logsum(X, start_round, best)
    for _ in range(iters):
        perm = greedy_sample_once(X, start_round, rng)
        v = eval_perm_logsum(X, start_round, perm)
        if v > bestv:
            best, bestv = perm, v
    return best

def sa_once(X, start_round, start_perm, burn_in, beta_i, beta_f, alpha, seed):
    rng = np.random.default_rng(seed)
    perm = start_perm[:]

    n = len(perm)
    if n < 2:
        return perm

    beta = beta_i
    while beta < beta_f:
        for _ in range(burn_in):
            n = len(perm)
            if n < 2:
                return perm
            i, j = rng.choice(n, size=2, replace=False)
            cand = perm[:]
            cand[i], cand[j] = cand[j], cand[i]
            cur = eval_perm_logsum(X, start_round, perm)
            nv = eval_perm_logsum(X, start_round, cand)
            if nv >= cur:
                perm = cand
            else:
                ap = math.exp(beta * (nv - cur))
                if rng.random() < ap:
                    perm = cand
        beta *= alpha
    return perm


def simulated_annealing_best_of_N(X, start_round, runs, burn_in, beta_i, beta_f, alpha, seed=2025):
    base = greedy(X, start_round)
    if len(base) < 2:
        return base
    best = base[:]
    bestv = eval_perm_logsum(X, start_round, best)
    for k in range(runs):
        perm = sa_once(X, start_round, base, burn_in, beta_i, beta_f, alpha, seed + k)
        v = eval_perm_logsum(X, start_round, perm)
        if v > bestv:
            best, bestv = perm, v
    return best

def lp_assignment_cvxpy(X: np.ndarray, start_round: int) -> list[int]:
    import cvxpy as cvx
    n_rounds, n_teams = X.shape
    horizon = min(n_teams, n_rounds - start_round)
    P = X[start_round:start_round+horizon, :].copy()
    eps = 1e-12
    P = np.where(np.isfinite(P), np.clip(P, eps, 1.0), eps)
    W = np.log(P)

    A = cvx.Variable((horizon, n_teams))
    obj = cvx.Maximize(cvx.sum(cvx.multiply(W, A)))
    constraints = [
        A >= 0, A <= 1,
        cvx.sum(A, axis=1) == 1,
        cvx.sum(A, axis=0) <= 1,
    ]
    prob = cvx.Problem(obj, constraints)
    prob.solve(solver=cvx.ECOS, verbose=False)

    M = A.value
    if M is None:
        return greedy(X, start_round)

    picks = []
    used = set()
    for r in range(horizon):
        order = np.argsort(-M[r, :])
        pick = None
        for t in order:
            t = int(t)
            if t not in used:
                pick = t
                break
        if pick is None:
            pick = next(t for t in range(n_teams) if t not in used)
        used.add(pick)
        picks.append(pick)
    return picks

def hungarian_assignment(X: np.ndarray, start_round: int) -> list[int]:
    n_rounds, n_teams = X.shape
    horizon = min(n_teams, n_rounds - start_round)
    P = X[start_round:start_round+horizon, :].copy()
    eps = 1e-12
    P = np.where(np.isfinite(P), np.clip(P, eps, 1.0), eps)
    cost = -np.log(P)
    row_ind, col_ind = linear_sum_assignment(cost)
    order = np.argsort(row_ind)
    return [int(col_ind[i]) for i in order]
