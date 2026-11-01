"""Exchangeable softmax predictor: each approach is described by its status in
the archive (e.g. solved / failed / not yet seen), and the logit of an approach
depends on its status and on the counts of approaches in each status."""
import numpy as np
from scipy.optimize import minimize
from ni_model import K


def statuses(masks, kind):
    """Status of every approach (N x K ints) and context (counts per status)."""
    sol = (masks["sol"][:, None] >> np.arange(K)) & 1
    if kind == "outcomes":
        S = np.where(sol == 1, 1, 0)
        C = 2
    elif kind == "bare":
        fail = (masks["fail"][:, None] >> np.arange(K)) & 1
        S = np.where(sol == 1, 1, np.where(fail == 1, 2, 0))
        C = 3
    elif kind == "diagnosed":
        fv = (masks["fail_via"][:, None] >> np.arange(K)) & 1
        fo = (masks["fail_obs"][:, None] >> np.arange(K)) & 1
        S = np.where(sol == 1, 1, np.where(fv == 1, 2, np.where(fo == 1, 3, 0)))
        C = 4
    counts = np.stack([(S == c).sum(axis=1) for c in range(1, C)], axis=1)
    return S.astype(np.int64), counts, C


class SymmetricSoftmax:
    def __init__(self, l2=1e-6):
        self.l2 = l2

    def _ctx(self, counts, grow):
        idx = np.empty(len(counts), dtype=np.int64)
        for i, row in enumerate(map(tuple, counts)):
            j = self.ctx.get(row)
            if j is None:
                if grow:
                    j = self.ctx[row] = len(self.ctx)
                else:
                    j = -1
            idx[i] = j
        return idx

    def _logits(self, theta, S, c):
        w0 = theta[: self.C]
        W = theta[self.C:].reshape(self.C, self.M)
        L = w0[S].astype(float)
        ok = c >= 0
        L[ok] += W[S[ok], c[ok][:, None]]
        return L

    def fit(self, S, counts, y, C):
        self.C, self.ctx = C, {}
        c = self._ctx(counts, grow=True)
        self.M = len(self.ctx)
        n = len(y)
        rows = np.arange(n)

        def f(theta):
            L = self._logits(theta, S, c)
            L -= L.max(axis=1, keepdims=True)
            P = np.exp(L); P /= P.sum(axis=1, keepdims=True)
            loss = -np.log(np.clip(P[rows, y], 1e-300, None)).mean() + self.l2 * (theta ** 2).sum()
            G = P; G[rows, y] -= 1; G /= n
            g0 = np.bincount(S.ravel(), weights=G.ravel(), minlength=C)
            flat = (S * self.M + c[:, None]).ravel()
            gW = np.bincount(flat, weights=G.ravel(), minlength=C * self.M)
            grad = np.concatenate([g0, gW]) + 2 * self.l2 * theta
            return loss, grad

        theta0 = np.zeros(C + C * self.M)
        res = minimize(f, theta0, jac=True, method="L-BFGS-B", options=dict(maxiter=5000, gtol=1e-10))
        self.theta = res.x
        return self

    def predict_proba(self, S, counts):
        c = self._ctx(counts, grow=False)
        L = self._logits(self.theta, S, c)
        L -= L.max(axis=1, keepdims=True)
        P = np.exp(L)
        return P / P.sum(axis=1, keepdims=True)
