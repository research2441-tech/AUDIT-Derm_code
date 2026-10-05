"""AUDIT-Derm: audit-guided conformal risk control for skin malignancy triage.

Components
  * Paired acquisition transport (PAT): ridge-to-identity affine map of logit scores from an
    imaging condition to the dermoscopic reference, fitted on images of the same lesion.
  * Probability-based audit (PBA): every case carries a recorded inclusion probability for
    outcome ascertainment; auto-benign cases are sampled in proportion to predicted outcome
    uncertainty under a budget.
  * Model-assisted (AIPW) increments for linearised decision risks.
  * Anytime-valid certification: mixture-of-bets test supermartingales per risk, combined by an
    intersection-union rule and scanned over a fixed sequence of conformal candidate policies.
  * Controller: deploy the least-referral certified candidate; otherwise suspend auto-benign
    assessment (refer everything), which also restores full outcome ascertainment.
"""
import numpy as np

EPS = 1e-9


def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


# ----------------------------------------------------------------------------- transport
def fit_transport(u_c, u_ref, kappa=5.0):
    """argmin_{a,b} sum (u_ref - a u_c - b)^2 + kappa[(a-1)^2 + b^2]  (closed form)."""
    X = np.c_[u_c, np.ones_like(u_c)]
    A = X.T @ X + kappa * np.eye(2)
    rhs = X.T @ u_ref + kappa * np.array([1.0, 0.0])
    a, b = np.linalg.solve(A, rhs)
    resid = u_ref - (a * u_c + b)
    return dict(a=float(a), b=float(b), sigma=float(resid.std(ddof=2)), n=int(len(u_c)))


def apply_transport(u, cond, maps):
    out = u.copy()
    for c, m in maps.items():
        k = cond == c
        out[k] = m['a'] * u[k] + m['b']
    return out


# ----------------------------------------------------------------------------- calibration
def fit_platt(u, y, w=None, iters=100, l2=1e-3):
    """Weighted 1-D logistic calibration p = sigmoid(c1*u + c0) by Newton steps."""
    w = np.ones_like(u) if w is None else w
    c = np.array([1.0, 0.0])
    X = np.c_[u, np.ones_like(u)]
    for _ in range(iters):
        p = sigmoid(X @ c)
        g = X.T @ (w * (p - y)) + l2 * (c - np.array([1.0, 0.0]))
        H = (X * (w * p * (1 - p))[:, None]).T @ X + l2 * np.eye(2)
        step = np.linalg.solve(H, g)
        c -= step
        if np.abs(step).max() < 1e-8:
            break
    return c


def platt(u, c):
    return sigmoid(c[0] * u + c[1])


def prior_shift(p, pi_new, pi_0):
    """Bayes re-weighting of posteriors from development prevalence pi_0 to pi_new."""
    a = p * pi_new / pi_0
    b = (1 - p) * (1 - pi_new) / (1 - pi_0)
    return a / (a + b)


def em_prevalence(p, pi_0, iters=200):
    """Label-free EM estimate of the target prevalence from calibrated posteriors."""
    pi = pi_0
    for _ in range(iters):
        new = prior_shift(p, pi, pi_0).mean()
        if abs(new - pi) < 1e-7:
            break
        pi = new
    return float(pi)


# ----------------------------------------------------------------------------- conformal grid
def mondrian_thresholds(u_cal_malignant, eps_grid):
    """tau(eps): malignant label excluded iff u < tau(eps). Class-conditional split conformal:
    tau is the floor(eps (n1+1))-th smallest malignant calibration score, giving
    P(u < tau | Y=1) <= eps under exchangeability with the calibration data."""
    s = np.sort(u_cal_malignant)
    n1 = len(s)
    taus = []
    for e in eps_grid:
        k = int(np.floor(e * (n1 + 1)))
        taus.append(-np.inf if k < 1 else s[k - 1])
    return np.array(taus)


# ----------------------------------------------------------------------------- audit design
def design_scale(m_auto, budget, pi_min=0.05):
    """Scale c of the audit design pi(x) = clip(c sqrt(m(1-m)), pi_min, 1), chosen *before* the
    period from unlabelled predictions so that the expected audited fraction of auto-benign cases
    equals the budget. Fixing c in advance keeps pi_i a deterministic function of x_i."""
    if len(m_auto) == 0:
        return 1.0
    r = np.sqrt(np.clip(m_auto * (1 - m_auto), 1e-6, None))
    lo, hi = 0.0, 1e4
    for _ in range(80):
        c = 0.5 * (lo + hi)
        if np.clip(c * r, pi_min, 1).mean() > budget:
            hi = c
        else:
            lo = c
    return lo


def inclusion_probs(m, auto, c, pi_ref=1.0, pi_min=0.05, adaptive=True, budget=0.1):
    """Referred cases: outcome ascertained with probability pi_ref. Auto-benign cases:
    uncertainty-proportional probability (adaptive) or the flat budget (uniform design)."""
    pi = np.full(len(m), pi_ref, float)
    k = auto.astype(bool)
    if adaptive:
        pi[k] = np.clip(c * np.sqrt(np.clip(m[k] * (1 - m[k]), 1e-6, None)), pi_min, 1)
    else:
        pi[k] = max(pi_min, budget)
    return pi


def psi_upper(c, pi_ref, pi_min, adaptive, budget, suspended):
    """Largest possible value of psi for a case of a given period: psi <= 1 / (smallest inclusion
    probability the period design can assign). Depends on the period design only."""
    if suspended:
        return 1.0 / pi_ref
    lo = max(pi_min, min(1.0, c * 1e-3)) if adaptive else max(pi_min, budget)
    return float(max(1.0 / lo, 1.0 / pi_ref))


# ----------------------------------------------------------------------------- increments
def aipw_terms(y, o, pi, m, use_aipw=True):
    """Model-assisted outcome term psi_i with E[psi | x] = P(Y=1 | x) under known pi."""
    yo = np.where(o, y, 0.0)
    if use_aipw:
        return m + o / pi * (yo - m)
    return o / pi * yo


def risk_increments(u_t, taus, psi, alpha_f, alpha_m, margin=0.0):
    """Linearised risks for every candidate (n x K):
       FOR <= alpha_f  <=>  E[A (Y - alpha_f)] <= 0
       MISS <= alpha_m <=>  E[Y (A - alpha_m)] <= 0
    Both summands A(Y-alpha) and Y(A-alpha) have range 1, so testing E[z] <= -rho protects
    against a total-variation drift of at most rho between window and deployment period."""
    A = (u_t[:, None] < taus[None, :]).astype(float)
    zf = A * (psi[:, None] - alpha_f) + margin
    zm = (A - alpha_m) * psi[:, None] + margin
    return zf, zm


# ----------------------------------------------------------------------------- tests
def betting_reject(z, upper, delta, n_lambda=10, c=0.5):
    """Mixture-of-bets test of H0: E[z] >= 0 against E[z] < 0.
    `upper` is a per-row bound fixed by the deployment design of the period the row comes from
    (known before any outcome of that period is observed), so the bet
    lambda_{ij} = c 2^-j / upper_i is predictable. Each wealth process prod_i (1 - lambda_ij z_i) is a
    nonnegative supermartingale under H0; their uniform mixture is one too, and Ville's inequality
    gives P(sup_n W_n >= 1/delta) <= delta. Returns (reject, final log-wealth) per column."""
    if z.shape[0] == 0:
        return np.zeros(z.shape[1], bool), np.zeros(z.shape[1])
    upper = np.broadcast_to(np.asarray(upper, float), (z.shape[0],))
    lam = c * 2.0 ** -np.arange(n_lambda)[None, :] / upper[:, None]          # n x L
    logw = np.cumsum(np.log1p(-lam[:, :, None] * z[:, None, :]), axis=0)     # n x L x K
    mx = logw.max(axis=1, keepdims=True)
    mix = np.log(np.mean(np.exp(logw - mx), axis=1)) + mx[:, 0, :]
    run = mix.max(axis=0)
    return run >= np.log(1 / delta), mix[-1]


def hoeffding_reject(z, lower, upper, delta):
    n = z.shape[0]
    if n == 0:
        return np.zeros(z.shape[1], bool)
    return z.mean(0) + (upper - lower) * np.sqrt(np.log(1 / delta) / (2 * n)) <= 0


def wald_reject(z, delta):
    from scipy.stats import norm
    n = z.shape[0]
    if n < 2:
        return np.zeros(z.shape[1], bool)
    return z.mean(0) + norm.ppf(1 - delta) * z.std(0, ddof=1) / np.sqrt(n) <= 0


def fixed_sequence(reject):
    """Scan candidates in the pre-specified order; certified set = prefix of rejections."""
    k = 0
    while k < len(reject) and reject[k]:
        k += 1
    return k  # number of certified candidates (indices 0..k-1)


def betting_ucb(z, upper, delta, active=None, n_lambda=6, c=0.95, iters=18):
    """Upper confidence bound for E[z] from a mixture of constant bets (fixed-n form).
    For a null value mu0 the wealth is W(mu0) = mean_j prod_i (1 - lam_ij (z_i - mu0)), with
    lam_ij = c 2^-j / (upper_i - mu0) and upper_i fixed by the design of row i's period. Under
    H0: E[z] >= mu0, W is a nonnegative supermartingale, so P(W >= 1/delta) <= delta (Ville/Markov).
    The UCB is the smallest rejected mu0 (bisection, per column of z). Rows outside `active`
    (a design-determined mask) are identically zero and contribute in closed form."""
    n, K = z.shape
    if n == 0:
        return np.full(K, np.inf)
    upper = np.broadcast_to(np.asarray(upper, float), (n,))
    active = np.ones(n, bool) if active is None else active
    zz, uu = z[active], upper[active]
    n0 = int((~active).sum())
    u0 = float(uu.max()) if len(uu) else 1.0
    lo = np.full(K, -1.0)
    hi = np.full(K, float(max(u0, 1.0)))
    jj = c * 2.0 ** -np.arange(n_lambda)
    thr = np.log(1 / delta)
    for _ in range(iters):
        mu = 0.5 * (lo + hi)
        lw = np.zeros((n_lambda, K))
        if len(zz):
            lam = jj[None, :, None] / (uu[:, None, None] - mu[None, None, :])        # m x L x K
            lw += np.log1p(-lam * (zz[:, None, :] - mu[None, None, :])).sum(axis=0)
        if n0:
            lw += n0 * np.log1p(jj[:, None] / (u0 - mu[None, :]) * mu[None, :])
        mx = lw.max(axis=0)
        mix = np.log(np.mean(np.exp(lw - mx), axis=0)) + mx
        rej = mix >= thr
        hi = np.where(rej, mu, hi)
        lo = np.where(rej, lo, mu)
    return hi


def betting_logwealth(z, upper, mu0, active=None, n_lambda=6, c=0.95):
    """Mixture log-wealth of the bet against H0: E[z] >= mu0 (mu0 may differ per column)."""
    n, K = z.shape
    upper = np.broadcast_to(np.asarray(upper, float), (n,))
    active = np.ones(n, bool) if active is None else active
    zz, uu = z[active], upper[active]
    n0 = int((~active).sum())
    u0 = float(uu.max()) if len(uu) else 1.0
    jj = c * 2.0 ** -np.arange(n_lambda)
    mu0 = np.broadcast_to(np.asarray(mu0, float), (K,))
    lw = np.zeros((n_lambda, K))
    if len(zz):
        lam = jj[None, :, None] / (uu[:, None, None] - mu0[None, None, :])
        lw += np.log1p(-lam * (zz[:, None, :] - mu0[None, None, :])).sum(axis=0)
    if n0:
        lw += n0 * np.log1p(jj[:, None] / (u0 - mu0[None, :]) * mu0[None, :])
    mx = lw.max(axis=0)
    return np.log(np.mean(np.exp(lw - mx), axis=0)) + mx


def split_certify(z, B, uR, uB, delta, margin=0.0):
    """Reject H0: E[z] >= -margin using two disjoint row streams (referred rows R and audited
    auto-benign rows B). UCB_R is computed at level delta/2; the B stream is then tested at the
    complementary null value -UCB_R - margin at level delta/2. A rejection implies
    UCB_R + UCB_B + margin < 0, so P(false rejection) <= delta by a union bound."""
    R = ~B
    ucb_r = betting_ucb(z * R[:, None], uR, delta / 2, R)
    if not B.any():
        return ucb_r + margin < 0
    mu0 = -ucb_r - margin
    ok = mu0 > -1.0
    lw = betting_logwealth(z * B[:, None], uB, np.where(ok, mu0, 0.0), B)
    return ok & (lw >= np.log(2 / delta))
