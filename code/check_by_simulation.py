"""
Independent check of worked_example.py by two different methods.

1. Formulas.  The entropies of Table 2 are recomputed from the closed-form
   expressions of Appendix A (and a hypergeometric formula for the labels),
   without enumerating the joint distribution.

2. Simulation.  The research process itself is simulated: a hidden obstructed
   set, a memoryless researcher who tries approaches in random order, and a
   next problem.  The Bayes posterior over all C(8,4) = 70 possible obstructed
   sets is computed by brute force for every simulated archive, with no
   sufficient statistics, no dynamic programme and no closed forms.  The
   simulated search cost and log-loss are compared with the exact values in
   results/worked_example.json.

Usage:  python code/check_by_simulation.py results/worked_example.json
Requires numpy.  Prints "simulation agrees with exact results" on success.
"""
import json
import sys
from itertools import combinations
from math import comb, factorial, log2

import numpy as np

K, m = 8, 4
v = K - m
SAMPLES = 200_000
rng = np.random.default_rng(2026)

exact = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/worked_example.json"))


def close(label, estimate, target, se=None, tol=None):
    """Compare an estimate with the exact value (within 4 standard errors)."""
    limit = tol if tol is not None else 4 * se + 1e-9
    ok = abs(estimate - target) <= limit
    print(f"  {label:52s} exact {target:8.4f}   check {estimate:8.4f}   {'ok' if ok else 'MISMATCH'}")
    assert ok, label


# ---------------------------------------------------------------------------
# 1. Closed-form formulas for one archived problem (Appendix A)
# ---------------------------------------------------------------------------
print("1. Formulas for one archived problem")
single = exact["single_problem"]
H_bare = log2(K) + sum(log2(factorial(K - 1) / factorial(K - 1 - f)) for f in range(K)) / K
order = sum(log2(factorial(f)) for f in range(K)) / K
dead_end_bare = H_bare - order
# Labels of a diagnosed trajectory: given S1 and |F| = f, the obstructed set N
# is a uniform m-subset of the K-1 other approaches; a labelling with t
# obstructed failures has probability C(K-1-f, m-t) / C(K-1, m).
labels = 0.0
for f in range(K):
    H_T = 0.0
    for t in range(0, min(f, m) + 1):
        P = comb(K - 1 - f, m - t) / comb(K - 1, m)
        if P > 0:
            H_T -= comb(f, t) * P * log2(P)
    labels += H_T / K
H_Y_given_E = -(1 / v) * log2(1 / v) - (1 - 1 / v) * log2((1 - 1 / v) / (K - 1))

close("Delta I, bare trajectories", H_bare, single["DeltaI_bare = H(Gamma_bare|S1)"], tol=1e-9)
close("Delta I, diagnosed trajectories", H_bare + labels, single["DeltaI_diag = H(Gamma_diag|S1)"], tol=1e-9)
close("order-of-attempts term", order, single["E[log2|F|!] (order term)"], tol=1e-9)
close("dead-end term, bare", dead_end_bare, single["H(F|S1)"], tol=1e-9)
close("dead-end term, diagnosed", dead_end_bare + labels, single["H(F,T|S1)"], tol=1e-9)
close("H(Y | E)", H_Y_given_E, single["H(S2|S1)"], tol=1e-9)

# ---------------------------------------------------------------------------
# 2. Simulation of the research process with brute-force Bayes posteriors
# ---------------------------------------------------------------------------
N_SETS = np.array([sum(1 << a for a in c) for c in combinations(range(K), m)])     # 70 bitmasks
VIABLE = np.array([[a for a in range(K) if not (Nm >> a) & 1] for Nm in N_SETS])   # 70 x v
NOT_IN = (((N_SETS[:, None] >> np.arange(K)) & 1) == 0).astype(float)              # 70 x K
BIT = 1 << np.arange(K)


def predictive(consistent):
    """Bayes predictive P(approach a solves the next problem) from a boolean
    matrix (samples x 70) of obstructed sets consistent with each archive."""
    c = consistent.astype(float)
    return (c @ NOT_IN) / (c.sum(axis=1, keepdims=True) * v)


def search_cost_and_logloss(pred, y):
    """Expected attempts of a searcher who tries approaches in decreasing
    predictive probability (random tie-breaking), and log-loss in bits."""
    p_true = pred[np.arange(len(y)), y]
    greater = (pred > p_true[:, None] + 1e-12).sum(axis=1)
    ties = (np.abs(pred - p_true[:, None]) <= 1e-12).sum(axis=1) - 1
    return 1 + greater + ties / 2, -np.log2(p_true)


print(f"\n2. Simulation ({SAMPLES:,} simulated archives for each n)")
for n in range(0, 7):
    row = exact["archive"][n]
    idx = rng.integers(len(N_SETS), size=SAMPLES)
    N_true = N_SETS[idx]
    solutions = np.zeros(SAMPLES, dtype=np.int64)       # bitmask of past solutions
    seen_obstructed = np.zeros(SAMPLES, dtype=np.int64)  # failures labelled obstructed
    seen_viable = np.zeros(SAMPLES, dtype=np.int64)      # solutions + failures labelled viable
    for _ in range(n):
        s = VIABLE[idx, rng.integers(v, size=SAMPLES)]
        order_ = np.argsort(rng.random((SAMPLES, K)), axis=1)  # random order of attempts
        pos = np.argmax(order_ == s[:, None], axis=1)
        tried_before = (np.arange(K)[None, :] < pos[:, None])
        F = (BIT[order_] * tried_before).sum(axis=1)          # set of failed attempts
        solutions |= 1 << s
        seen_obstructed |= F & N_true
        seen_viable |= (F & ~N_true) | (1 << s)
    y = VIABLE[idx, rng.integers(v, size=SAMPLES)]          # solution of the next problem

    cons_out = (N_SETS[None, :] & solutions[:, None]) == 0
    cons_diag = ((N_SETS[None, :] & seen_viable[:, None]) == 0) & \
                ((N_SETS[None, :] & seen_obstructed[:, None]) == seen_obstructed[:, None])
    T_out, L_out = search_cost_and_logloss(predictive(cons_out), y)
    T_diag, L_diag = search_cost_and_logloss(predictive(cons_diag), y)
    gain = L_out - L_diag
    se = lambda x: x.std(ddof=1) / np.sqrt(SAMPLES)

    print(f" n = {n}")
    close("expected attempts, outcomes only (= bare)", T_out.mean(), row["E[T] outcome-only (= bare)"], se(T_out))
    close("expected attempts, diagnosed", T_diag.mean(), row["E[T] diagnosed"], se(T_diag))
    close("task-relevant information Delta I_Y (bits)", gain.mean(),
          row["I(S_next; diagnosed trajectories | outcomes)"], se(gain))

# ---------------------------------------------------------------------------
# 3. The variant of Section 5.4 (p = 1/2), by simulation
# ---------------------------------------------------------------------------
# Each viable approach solves a problem independently with probability 1/2,
# conditioned on at least one solution; the record is the first success.
# The target is the set of solutions of the next problem.  With p = 1/2 every
# non-empty subset of the viable set is equally likely (probability 1/15).
print(f"\n3. Variant with p = 1/2 ({SAMPLES:,} simulated problems)")


def solution_sets(idx):
    """Sample non-empty solution sets (bitmasks) by rejection."""
    out = np.zeros(len(idx), dtype=np.int64)
    todo = np.arange(len(idx))
    while len(todo):
        draw = rng.random((len(todo), v)) < 0.5
        mask = (draw * (1 << VIABLE[idx[todo]])).sum(axis=1)
        out[todo] = mask
        todo = todo[mask == 0]
    return out


idx = rng.integers(len(N_SETS), size=SAMPLES)
N_true = N_SETS[idx]
sol1 = solution_sets(idx)
order_ = np.argsort(rng.random((SAMPLES, K)), axis=1)
success = ((sol1[:, None] >> order_) & 1) == 1
pos = np.argmax(success, axis=1)
s1 = order_[np.arange(SAMPLES), pos]
F = (BIT[order_] * (np.arange(K)[None, :] < pos[:, None])).sum(axis=1)
sol2 = solution_sets(idx)

valid = (N_SETS[None, :] & (1 << s1)[:, None]) == 0
# Likelihood of the bare failure set: every viable approach in F failed (1/2 each).
POPCOUNT = np.array([bin(i).count("1") for i in range(1 << K)])
w_bare = valid * 0.5 ** POPCOUNT[F[:, None] & ~N_SETS[None, :]]
w_out = valid.astype(float)
fits = ((N_SETS[None, :] & sol2[:, None]) == 0).astype(float)   # solution set avoids N
p_bare = (w_bare * fits).sum(axis=1) / w_bare.sum(axis=1) / 15
p_out = (w_out * fits).sum(axis=1) / w_out.sum(axis=1) / 15
gain = np.log2(p_bare) - np.log2(p_out)
close("I(next solutions; bare failures | record), bits", gain.mean(),
      exact["variant_I(Sol2;bare|S1)"]["1/2"], gain.std(ddof=1) / np.sqrt(SAMPLES))

print("\nsimulation agrees with exact results")
