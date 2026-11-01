"""
Numerical experiments of Section 6: navigational information estimated from
data with trained predictors, compared with the exact values.

A. The solvable model of Section 5 (K = 8, m = 4).  Archives are simulated,
   predictors of the next solution are trained on outcomes only, on bare
   trajectories and on diagnosed trajectories, and the task-relevant
   information is estimated as the difference of held-out log-losses
   (Proposition 2(a)).  The predictor class is an exchangeable softmax: the
   logit of an approach depends on its status in the archive (solved, failed,
   failed and labelled viable or obstructed, not yet seen) and on the number of
   approaches in each status.  A generic multilayer perceptron is run for
   comparison.

B. Information versus computation (Section 4.3).  A procedure computes the
   parity of d = 10 bits step by step.  The record is the input, the
   trajectory is the sequence of intermediate results, and the target is the
   output.  In the computable variant the trajectory is a function of the
   record (Delta I_Y = 0).  In the hidden-fault variant, with probability 1/2
   the device flips the running value at a random intermediate step, which
   the record does not show (Delta I_Y = 1 bit).  Usable navigational
   information is estimated for multilayer perceptrons of increasing width.

Usage:  python code/learned_estimates.py results/worked_example.json results/learned_estimates.json
Requires numpy, scipy, scikit-learn, joblib.
"""
import json
import sys
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from ni_model import K, simulate, eval_pred
from symmetric_softmax import SymmetricSoftmax, statuses

warnings.filterwarnings("ignore", category=ConvergenceWarning)
exact = json.load(open(sys.argv[1]))
out_path = sys.argv[2]
KINDS = ("outcomes", "bare", "diagnosed")
TEST = 100_000


def run_symmetric(n, n_train, seed):
    rng = np.random.default_rng(seed)
    _, ytr, mtr = simulate(n_train, n, rng)
    _, yte, mte = simulate(TEST, n, rng)
    res = {}
    for kind in KINDS:
        S, cnt, C = statuses(mtr, kind)
        model = SymmetricSoftmax().fit(S, cnt, ytr, C)
        S2, cnt2, _ = statuses(mte, kind)
        res[kind] = eval_pred(model.predict_proba(S2, cnt2), yte)
    return dict(n=n, n_train=n_train, seed=seed,
                logloss={k: res[k][0] for k in KINDS},
                attempts={k: res[k][1] for k in KINDS},
                gap_diagnosed=res["outcomes"][0] - res["diagnosed"][0],
                gap_bare=res["outcomes"][0] - res["bare"][0])


def run_mlp_model(n, n_train, seed):
    rng = np.random.default_rng(10_000 + seed)
    tr, ytr, _ = simulate(n_train, n, rng)
    te, yte, _ = simulate(TEST, n, rng)
    res = {}
    for kind in ("outcomes", "diagnosed"):
        clf = MLPClassifier(hidden_layer_sizes=(128,), alpha=0.0, early_stopping=True,
                            n_iter_no_change=40, tol=1e-8, learning_rate_init=1e-3,
                            batch_size=256, max_iter=400, random_state=seed)
        clf.fit(tr[kind], ytr)
        p = np.zeros((len(yte), K))
        p[:, clf.classes_] = clf.predict_proba(te[kind])
        res[kind] = eval_pred(p, yte)
    return dict(seed=seed, gap_diagnosed=res["outcomes"][0] - res["diagnosed"][0],
                logloss={k: res[k][0] for k in res})


def parity_data(size, d, hidden_fault, rng):
    x = rng.integers(0, 2, size=(size, d))
    running = np.cumsum(x, axis=1) % 2
    if hidden_fault:
        fault = rng.integers(0, 2, size=size)
        step = rng.integers(0, d - 1, size=size)          # an intermediate step
        flip = (np.arange(d)[None, :] >= step[:, None]) & (fault[:, None] == 1)
        running = running ^ flip
    return x, running[:, :-1], running[:, -1]           # record, trajectory, target


def bits_logloss(clf, X, y):
    p = clf.predict_proba(X)
    pt = np.clip(p[np.arange(len(y)), np.searchsorted(clf.classes_, y)], 1e-12, 1)
    return float(-np.log2(pt).mean())


def make_learner(width, seed):
    if width == 0:
        return LogisticRegression(max_iter=5000)
    # No early stopping: scikit-learn stops early on validation accuracy, which
    # saturates long before the log-loss has converged.
    return MLPClassifier(hidden_layer_sizes=(width,), alpha=1e-5, learning_rate_init=3e-3,
                         n_iter_no_change=30, tol=1e-6, max_iter=400, random_state=seed)


def run_parity(width, hidden_fault, seed, d=10, n_train=20_000):
    rng = np.random.default_rng(100 * seed + 7 * width + int(hidden_fault))
    xtr, gtr, ytr = parity_data(n_train, d, hidden_fault, rng)
    xte, gte, yte = parity_data(20_000, d, hidden_fault, rng)
    out = make_learner(width, seed).fit(xtr, ytr)
    traj = make_learner(width, seed).fit(np.hstack([xtr, gtr]), ytr)
    lo = bits_logloss(out, xte, yte)
    lt = bits_logloss(traj, np.hstack([xte, gte]), yte)
    return dict(width=width, hidden_fault=hidden_fault, seed=seed,
                logloss_record=lo, logloss_trajectory=lt, gap=lo - lt)


t0 = time.time()
results = {"exact": {"n": [r["n"] for r in exact["archive"]],
                     "Delta_I_Y_diagnosed": [r["I(S_next; diagnosed trajectories | outcomes)"]
                                             for r in exact["archive"]],
                     "attempts_outcomes": [r["E[T] outcome-only (= bare)"] for r in exact["archive"]],
                     "attempts_diagnosed": [r["E[T] diagnosed"] for r in exact["archive"]]}}

only_b = len(sys.argv) > 3 and sys.argv[3] == "B"   # recompute part B only
if only_b:
    results = json.load(open(out_path))
sizes = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
if not only_b:
    jobs = [(1, N, s) for N in sizes for s in range(5)]
    results["A_learning_curve"] = Parallel(n_jobs=2)(delayed(run_symmetric)(*j) for j in jobs)
    print(f"A1 done ({time.time() - t0:.0f}s)", flush=True)

    jobs = [(n, 32000, 100 + s) for n in range(0, 11) for s in range(3)]
    results["A_archive_size"] = Parallel(n_jobs=2)(delayed(run_symmetric)(*j) for j in jobs)
    print(f"A2 done ({time.time() - t0:.0f}s)", flush=True)

    results["A_generic_mlp"] = Parallel(n_jobs=2)(delayed(run_mlp_model)(1, 64000, s) for s in range(3))
    print(f"A3 done ({time.time() - t0:.0f}s)", flush=True)

widths = [0, 2, 4, 8, 16, 32, 64, 128, 256]
jobs = [(w, f, s) for w in widths for f in (False, True) for s in range(3)]
results["B_capacity"] = Parallel(n_jobs=2)(delayed(run_parity)(*j) for j in jobs)
print(f"B done ({time.time() - t0:.0f}s)", flush=True)

json.dump(results, open(out_path, "w"), indent=1)
print("saved", out_path)
