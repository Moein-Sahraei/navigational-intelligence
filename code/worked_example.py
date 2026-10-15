"""
Worked example for the navigational-intelligence paper:
search among K candidate approaches, m of which are structurally obstructed.

Model
-----
* K approaches; a latent set N of m obstructed approaches (uniform m-subset).
  V = complement of N, |V| = v = K - m.
* Problem k has solution S_k ~ Uniform(V), independently across k given N.
* A memoryless researcher tries approaches in a uniformly random order until S_k.
* Bare trajectory:      the ordered list of attempted approaches (ending at S_k).
* Diagnosed trajectory: the same list, with each failure labelled
                        'obstructed' (a in N) or 'viable, not the solution'.
* Published outcome:    E_k = S_k.
* Downstream target:    Y = S_{n+1}, the solution of the next problem.

All quantities are computed exactly (enumeration for n = 1, exact dynamic
programming over sufficient statistics for general n). Entropies in bits.
"""
from itertools import combinations
from math import comb, factorial, log2
from fractions import Fraction
import json
import sys

K, m = 8, 4
v = K - m
A = range(K)


# ---------------------------------------------------------------------------
# Part 1: exact enumeration for a single archived problem (n = 1)
# ---------------------------------------------------------------------------
def p_prefix_set(f):
    """P(set of approaches tried before S equals a given set F, |F| = f | S)."""
    return Fraction(factorial(f) * factorial(K - 1 - f), factorial(K))


def H(dist):
    return -sum(float(p) * log2(float(p)) for p in dist.values() if p > 0)


def enumerate_joint():
    """Joint distribution over (N, S1, F) with exact Fractions."""
    Ns = list(combinations(A, m))
    pN = Fraction(1, len(Ns))
    joint = {}
    for N in Ns:
        Nset = frozenset(N)
        V = [a for a in A if a not in Nset]
        for s in V:
            others = [a for a in A if a != s]
            for f in range(K):
                for F in combinations(others, f):
                    joint[(Nset, s, frozenset(F))] = pN * Fraction(1, v) * p_prefix_set(f)
    return joint


def exact_single_problem():
    joint = enumerate_joint()
    total = sum(joint.values())
    assert total == 1, total

    # Marginals needed for entropies
    from collections import defaultdict
    pS = defaultdict(Fraction)
    pSF = defaultdict(Fraction)
    pSFT = defaultdict(Fraction)  # T = F ∩ N (diagnosis)
    for (N, s, F), p in joint.items():
        T = F & N
        pS[s] += p
        pSF[(s, F)] += p
        pSFT[(s, F, T)] += p

    H_S = H(pS)
    H_SF = H(pSF)
    H_SFT = H(pSFT)
    # Order of attempts given the set F is uniform over |F|! orderings,
    # independent of everything else: contributes E[log2 |F|!].
    E_log_order = sum(float(p) * log2(factorial(len(F))) for (s, F), p in pSF.items())

    res = {}
    res["H(F|S1)"] = H_SF - H_S
    res["H(F,T|S1)"] = H_SFT - H_S
    res["E[log2|F|!] (order term)"] = E_log_order
    res["DeltaI_bare = H(Gamma_bare|S1)"] = (H_SF - H_S) + E_log_order
    res["DeltaI_diag = H(Gamma_diag|S1)"] = (H_SFT - H_S) + E_log_order

    # Task-relevant information about S2 (independent draw from V given N).
    # P(S2 = a | N) = 1/v for a not in N.
    def cond_entropy_S2(keyfun):
        """H(S2 | key) where key is a function of (N, s, F)."""
        from collections import defaultdict
        pk = defaultdict(Fraction)
        pk_s2 = defaultdict(Fraction)
        for (N, s, F), p in joint.items():
            k = keyfun(N, s, F)
            pk[k] += p
            for a in A:
                if a not in N:
                    pk_s2[(k, a)] += p * Fraction(1, v)
        return H(pk_s2) - H(pk)

    H_S2 = cond_entropy_S2(lambda N, s, F: 0)
    H_S2_given_S1 = cond_entropy_S2(lambda N, s, F: s)
    H_S2_given_S1_F = cond_entropy_S2(lambda N, s, F: (s, F))
    H_S2_given_S1_F_T = cond_entropy_S2(lambda N, s, F: (s, F, F & N))
    res["H(S2)"] = H_S2
    res["H(S2|S1)"] = H_S2_given_S1
    res["I(S2;Gamma_bare|S1)"] = H_S2_given_S1 - H_S2_given_S1_F
    res["I(S2;Gamma_diag|S1)"] = H_S2_given_S1 - H_S2_given_S1_F_T
    res["I(S2;S1) (outcome-only information)"] = H_S2 - H_S2_given_S1

    # Expected attempts on problem 2 under the Bayes-optimal search order,
    # computed directly from the enumerated joint distribution.
    def expected_attempts(keyfun):
        from collections import defaultdict
        post = defaultdict(lambda: defaultdict(Fraction))   # key -> a -> mass
        for (N, s, F), p in joint.items():
            k = keyfun(N, s, F)
            for a in A:
                if a not in N:
                    post[k][a] += p * Fraction(1, v)
        total = Fraction(0)
        for k, d in post.items():
            probs = sorted([d.get(a, Fraction(0)) for a in A], reverse=True)
            total += sum((r + 1) * q for r, q in enumerate(probs))
        return total

    res["E[T2] no memory"] = expected_attempts(lambda N, s, F: 0)
    res["E[T2] outcome-only"] = expected_attempts(lambda N, s, F: s)
    res["E[T2] bare"] = expected_attempts(lambda N, s, F: (s, F))
    res["E[T2] diagnosed"] = expected_attempts(lambda N, s, F: (s, F, F & N))
    return res


# ---------------------------------------------------------------------------
# Part 2: exact dynamic programming over sufficient statistics, n = 0..nmax
# ---------------------------------------------------------------------------
# For diagnosed archives the posterior over N depends only on
#   o = number of approaches revealed obstructed,
#   u = number of approaches revealed viable (solutions + viable failures).
# For outcome-only (and bare) archives it depends only on
#   u = number of distinct previous solutions (o = 0).
def predictive_stats(o, u):
    """Return (expected attempts, expected log-loss in bits) for the next
    problem under the Bayes-optimal search order / predictor, given (o, u)."""
    w = K - o - u                       # unrevealed approaches
    q = Fraction(v - u, w * v) if w > 0 else Fraction(0)  # prob. for each unrevealed
    # optimal order: revealed-viable first (prob 1/v each), then unrevealed
    ET = Fraction(u * (u + 1), 2 * v) + q * Fraction((u + w) * (u + w + 1) - u * (u + 1), 2)
    LL = 0.0
    if u > 0:
        LL += (u / v) * log2(v)
    if v - u > 0:
        LL += ((v - u) / v) * (-log2(float(q)))
    return ET, LL


def hyper(counts, draws):
    """Multivariate hypergeometric pmf over (x0, x1, x2) with sum = draws."""
    n0, n1, n2 = counts
    tot = n0 + n1 + n2
    out = {}
    for x0 in range(0, min(n0, draws) + 1):
        for x1 in range(0, min(n1, draws - x0) + 1):
            x2 = draws - x0 - x1
            if x2 < 0 or x2 > n2:
                continue
            out[(x0, x1)] = Fraction(comb(n0, x0) * comb(n1, x1) * comb(n2, x2), comb(tot, draws))
    return out


def dp_diagnosed(nmax):
    dists = [{(0, 0): Fraction(1)}]
    for _ in range(nmax):
        new = {}
        for (o, u), p in dists[-1].items():
            for s_new, ps in ((True, Fraction(v - u, v)), (False, Fraction(u, v))):
                if ps == 0:
                    continue
                n_unrev_obs = m - o
                n_unrev_via = v - u - (1 if s_new else 0)
                n_rev = o + u - (0 if s_new else 1)
                for f in range(K):          # |F| uniform on 0..K-1
                    pf = Fraction(1, K)
                    for (x, y), ph in hyper((n_unrev_obs, n_unrev_via, n_rev), f).items():
                        key = (o + x, u + y + (1 if s_new else 0))
                        new[key] = new.get(key, Fraction(0)) + p * ps * pf * ph
        assert sum(new.values()) == 1
        dists.append(new)
    return dists


def dp_outcome(nmax):
    dists = [{0: Fraction(1)}]
    for _ in range(nmax):
        new = {}
        for u, p in dists[-1].items():
            if v - u > 0:
                new[u + 1] = new.get(u + 1, Fraction(0)) + p * Fraction(v - u, v)
            if u > 0:
                new[u] = new.get(u, Fraction(0)) + p * Fraction(u, v)
        dists.append(new)
    return dists


def archive_curves(nmax=10):
    dd = dp_diagnosed(nmax)
    do = dp_outcome(nmax)
    rows = []
    for n in range(nmax + 1):
        ET_d = sum(p * predictive_stats(o, u)[0] for (o, u), p in dd[n].items())
        LL_d = sum(float(p) * predictive_stats(o, u)[1] for (o, u), p in dd[n].items())
        ET_o = sum(p * predictive_stats(0, u)[0] for u, p in do[n].items())
        LL_o = sum(float(p) * predictive_stats(0, u)[1] for u, p in do[n].items())
        rows.append({
            "n": n,
            "E[T] outcome-only (= bare)": float(ET_o),
            "E[T] diagnosed": float(ET_d),
            "H(S_next | outcomes)": LL_o,
            "H(S_next | outcomes, diagnosed trajectories)": LL_d,
            "I(S_next; diagnosed trajectories | outcomes)": LL_o - LL_d,
        })
    return rows


def variant_bare_information(p):
    """Variant: each viable approach solves a problem independently with
    probability p, conditioned on at least one solution. The memoryless
    researcher tries a uniformly random order until the first success
    (= the published record). Returns I(Sol2; bare failure set | S1) in bits,
    where Sol2 is the set of solutions of the next problem."""
    from collections import defaultdict
    p = Fraction(p)
    norm = 1 - (1 - p) ** v
    s1 = 0                                   # by symmetry, condition on S1 = 0
    others = [a for a in A if a != s1]
    joint = defaultdict(Fraction)            # (N, F) -> P(N, F, S1 = 0) (unnormalised)
    for N in combinations(A, m):
        N = frozenset(N)
        if s1 in N:
            continue
        for f in range(K):
            for F in combinations(others, f):
                F = frozenset(F)
                n_viable_failed = len([a for a in F if a not in N])
                joint[(N, F)] += (p_prefix_set(f) * (1 - p) ** n_viable_failed * p / norm)
    Z = sum(joint.values())

    def sol_dist(N):
        V = [a for a in A if a not in N]
        out = {}
        for r in range(1, v + 1):
            for c in combinations(V, r):
                out[frozenset(c)] = p ** r * (1 - p) ** (v - r) / norm
        return out
    cache = {}

    def H_sol2_given(keyfun):
        groups = defaultdict(lambda: defaultdict(Fraction))
        for (N, F), w in joint.items():
            groups[keyfun(N, F)][N] += w / Z
        tot = 0.0
        for g, wN in groups.items():
            pg = sum(wN.values())
            pred = defaultdict(Fraction)
            for N, w in wN.items():
                if N not in cache:
                    cache[N] = sol_dist(N)
                for sol, ps in cache[N].items():
                    pred[sol] += w / pg * ps
            tot += float(pg) * H(pred)
        return tot

    return H_sol2_given(lambda N, F: 0) - H_sol2_given(lambda N, F: F)


if __name__ == "__main__":
    single = exact_single_problem()
    curves = archive_curves(10)

    # Cross-checks: enumeration (n = 1) against the dynamic programme,
    # and both against the closed forms of Appendix A.
    assert abs(single["I(S2;Gamma_diag|S1)"] - curves[1]["I(S_next; diagnosed trajectories | outcomes)"]) < 1e-12
    assert single["E[T2] outcome-only"] == single["E[T2] bare"] == Fraction(4)
    assert single["E[T2] diagnosed"] == Fraction(3) and single["E[T2] no memory"] == Fraction(9, 2)
    assert abs(curves[1]["E[T] diagnosed"] - 3.0) < 1e-12 and abs(curves[1]["E[T] outcome-only (= bare)"] - 4.0) < 1e-12
    for r in curves:
        n = r["n"]
        cf_out = (v + 1) / 2 + m / 2 * (1 - 1 / v) ** n
        cf_diag = (v + 1) / 2 + m / 2 * ((v - 1) / (3 * v)) ** n
        assert abs(r["E[T] outcome-only (= bare)"] - cf_out) < 1e-12, (n, r, cf_out)
        assert abs(r["E[T] diagnosed"] - cf_diag) < 1e-12, (n, r, cf_diag)
    variant = {str(pp): variant_bare_information(pp) for pp in ("1/4", "1/2", "3/4")}

    single = {k: (float(x) if isinstance(x, Fraction) else x) for k, x in single.items()}
    out = {"K": K, "m": m, "single_problem": single, "archive": curves,
           "variant_I(Sol2;bare|S1)": variant}
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "worked_example.json", "w"), indent=2)
    for k, val in single.items():
        print(f"{k:45s} {val:.4f}")
    print()
    for r in curves:
        print(r)
    print("variant I(Sol2; bare failures | S1):", variant)
    print("all checks passed")
