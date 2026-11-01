"""Simulation of the solvable model (K approaches, m obstructed) and helpers."""
from itertools import combinations
import numpy as np

K, m = 8, 4
v = K - m
N_SETS = np.array([sum(1 << a for a in c) for c in combinations(range(K), m)])
VIABLE = np.array([[a for a in range(K) if not (Nm >> a) & 1] for Nm in N_SETS])
NOT_IN = (((N_SETS[:, None] >> np.arange(K)) & 1) == 0).astype(float)
BIT = 1 << np.arange(K)


def bits(mask):
    return ((mask[:, None] >> np.arange(K)[None, :]) & 1).astype(np.float32)


def simulate(size, n, rng):
    """Archives of n problems from random families, and the next solution."""
    idx = rng.integers(len(N_SETS), size=size)
    N_true = N_SETS[idx]
    sol = np.zeros(size, dtype=np.int64)
    fail = np.zeros(size, dtype=np.int64)
    fail_obs = np.zeros(size, dtype=np.int64)
    fail_via = np.zeros(size, dtype=np.int64)
    for _ in range(n):
        s = VIABLE[idx, rng.integers(v, size=size)]
        order = np.argsort(rng.random((size, K)), axis=1)
        pos = np.argmax(order == s[:, None], axis=1)
        tried = np.arange(K)[None, :] < pos[:, None]
        F = (BIT[order] * tried).sum(axis=1)
        sol |= 1 << s
        fail |= F
        fail_obs |= F & N_true
        fail_via |= F & ~N_true
    y = VIABLE[idx, rng.integers(v, size=size)]
    masks = dict(sol=sol, fail=fail, fail_obs=fail_obs, fail_via=fail_via)
    feats = {
        "outcomes": bits(sol),
        "bare": np.hstack([bits(sol), bits(fail)]),
        "diagnosed": np.hstack([bits(sol), bits(fail_obs), bits(fail_via)]),
    }
    return feats, y, masks


def bayes(masks, kind):
    """Exact Bayes predictive distribution over the next solution."""
    sol = masks["sol"]
    if kind in ("outcomes", "bare"):
        cons = (N_SETS[None, :] & sol[:, None]) == 0
    else:
        obs = masks["fail_obs"]
        via = masks["fail_via"] | sol
        cons = ((N_SETS[None, :] & via[:, None]) == 0) & \
               ((N_SETS[None, :] & obs[:, None]) == obs[:, None])
    c = cons.astype(float)
    return (c @ NOT_IN) / (c.sum(axis=1, keepdims=True) * v)


def eval_pred(p, y):
    """Mean log-loss (bits) and expected attempts of a searcher ranking by p."""
    pt = np.clip(p[np.arange(len(y)), y], 1e-12, 1)
    greater = (p > pt[:, None] + 1e-9).sum(axis=1)
    ties = (np.abs(p - pt[:, None]) <= 1e-9).sum(axis=1) - 1
    return float(-np.log2(pt).mean()), float((1 + greater + ties / 2).mean())
