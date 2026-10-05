"""Deployment simulation over real image pools: periods, audits, delayed outcomes, policies."""
import numpy as np, pandas as pd, json
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
import audit_derm as ad

CFG = dict(T=12, n=2500, alpha_f=0.05, alpha_m=0.10, delta=0.10, budget=0.40, pi_ref=1.0,
           pi_min=0.20, delay=1, window=3, kappa=5.0, n_min=150, eta=0.5)
EPS_GRID = np.geomspace(0.005, 0.6, 40)


# ============================================================================ data
class Data:
    def __init__(self, path='cases.csv', emb='emb_fp16.npz'):
        df = pd.read_csv(path)
        self.df = df
        self.emb = (np.load(emb) if emb.endswith('.npy') else np.load(emb)['emb']).astype(np.float32)
        self.emb /= np.linalg.norm(self.emb, axis=1, keepdims=True)
        from sklearn.decomposition import PCA
        tr = (df.split == 'train').values
        self.pca = PCA(32, random_state=0).fit(self.emb[tr]).transform(self.emb)
        cal = df[df.split == 'cal']
        self.u_cal, self.y_cal = cal.u.values, cal.y.values
        self.idx_cal = cal.index.values
        self.pi0 = float(self.y_cal.mean())
        self.c_src = ad.fit_platt(self.u_cal, self.y_cal)
        self.u_cal1 = self.u_cal[self.y_cal == 1]
        self.u_cal0 = self.u_cal[self.y_cal == 0]
        self.maps = self._fit_maps()
        df['ut'] = ad.apply_transport(df.u.values, df.condition.values, self.maps)
        self.pools = {
            'MSKCC': df.index[(df.cohort == 'MSKCC') & (df.split == 'pool')].values,
            'HIBA': df.index[(df.cohort == 'HIBA') & (df.split == 'pool')].values,
            'PAD': df.index[(df.cohort == 'PAD') & (df.split == 'pool')].values,
            'MILK': df.index[(df.cohort == 'MILK') & (df.split == 'pool')].values,
            'SRC': df.index[df.split == 'test'].values,
        }

    def pairs(self, site, cond):
        d = self.df[(self.df.cohort == site) & (self.df.split == 'pairfit')]
        a = d[d.condition == cond].groupby('group').u.mean()
        b = d[d.condition == 'dermoscopic'].groupby('group').u.mean()
        k = a.index.intersection(b.index)
        return a[k].values, b[k].values

    def _fit_maps(self):
        maps = {}
        uc, ur = self.pairs('MILK', 'closeup')
        maps['closeup'] = ad.fit_transport(uc, ur, CFG['kappa'])
        uo, ur2 = self.pairs('HIBA', 'overview')
        maps['overview'] = ad.fit_transport(uo, ur2, CFG['kappa'])
        maps['smartphone'] = dict(maps['closeup'])  # borrowed: no paired smartphone images exist
        return maps


# ============================================================================ scenarios
def period_weights(D, site, scen, t, T):
    """Sampling weights over the site pool for period t (prevalence and condition mix)."""
    idx = D.pools['SRC' if scen.startswith('prev') else site]
    y = D.df.y.values[idx]
    cond = D.df.condition.values[idx]
    if scen == 'acq':
        prev, mix = D.pi0, None
    elif scen == 'prev_lo':
        prev, mix = 0.10, None
    elif scen == 'prev_hi':
        prev, mix = 0.40, None
    elif scen == 'comp':
        prev, mix = 0.10, None
    elif scen == 'drift':
        prev = 0.10 if t < T // 2 else 0.30
        mix = 0.3 if t < T // 2 else 0.7  # share of non-dermoscopic images where both exist
    else:
        raise ValueError(scen)
    w = np.zeros(len(idx))
    conds = np.unique(cond)
    q = {c: np.mean(cond == c) for c in conds}
    if mix is not None and 'dermoscopic' in conds and len(conds) > 1:
        nd = [c for c in conds if c != 'dermoscopic']
        q = {'dermoscopic': 1 - mix}
        for c in nd:
            q[c] = mix * np.mean(cond == c) / np.mean(cond != 'dermoscopic')
    for c in conds:
        for lab, pl in ((1, prev), (0, 1 - prev)):
            k = (cond == c) & (y == lab)
            if k.sum():
                w[k] = q[c] * pl / k.sum()
    return idx, w / w.sum()


def pop_risk(score, y, w, tau):
    A = score < tau
    pa = (w * A).sum()
    num = (w * A * y).sum()
    prev = (w * y).sum()
    FOR = num / pa if pa > 0 else 0.0
    MISS = num / prev
    return FOR, MISS, 1 - pa


# ============================================================================ methods
METHODS = ['static', 'static_pat', 'wcp', 'ls_plugin', 'monitor', 'online', 'naive_audit',
           'ltt_ipw', 'audit', 'oracle']
ABLATIONS = ['audit_noPAT', 'audit_noAIPW', 'audit_uniform', 'audit_hoeffding', 'audit_bonf',
             'audit_nosuspend', 'audit_noupdate', 'audit_wald']


def flags(name):
    f = dict(pat=True, aipw=True, adaptive=True, test='bet', iut=True, suspend=True, update=True,
             weighted=True)
    if name == 'audit_noPAT': f['pat'] = False
    if name == 'audit_noAIPW': f['aipw'] = False
    if name == 'audit_uniform': f['adaptive'] = False
    if name == 'audit_hoeffding': f['test'] = 'hoeff'
    if name == 'audit_wald': f['test'] = 'wald'
    if name == 'audit_bonf': f['iut'] = False
    if name == 'audit_nosuspend': f['suspend'] = False
    if name == 'audit_noupdate': f['update'] = False
    if name == 'naive_audit': f['weighted'] = False
    if name == 'ltt_ipw':
        f.update(pat=False, aipw=False, adaptive=False, test='hoeff', iut=False)
    return f


def working_model(D, score_hist, y_hist, o_hist, pi_hist, score_win, update, c_src):
    """m(x): calibrated malignancy probability fixed before window outcomes are seen.
    Uses outcomes from periods older than the certification window (IPW-weighted Platt);
    otherwise source calibration shifted to the label-free EM prevalence of the window."""
    if update and len(score_hist) and o_hist.sum() >= 50 and y_hist[o_hist].sum() >= 5:
        k = o_hist
        c = ad.fit_platt(score_hist[k], y_hist[k].astype(float), w=1 / pi_hist[k])
        return lambda s: ad.platt(s, c)
    p_win = ad.platt(score_win, c_src)
    pi_em = ad.em_prevalence(p_win, D.pi0) if len(score_win) else D.pi0
    return lambda s: ad.prior_shift(ad.platt(s, c_src), pi_em, D.pi0)


def run_stream(D, site, scen, seed, methods, cfg=CFG, extra=None):
    """Simulate one deployment stream; every method sees the same case sequence and the same
    audit uniforms (common random numbers)."""
    extra = extra or {}
    cfg = dict(cfg, **extra)
    rng = np.random.default_rng(seed)
    T, n = cfg['T'], cfg['n']
    df = D.df
    batches, weights = [], []
    for t in range(T):
        idx, w = period_weights(D, site, scen, t, T)
        batches.append(rng.choice(idx, size=n, replace=True, p=w))
        weights.append((idx, w))
    audit_u = [rng.random(n) for _ in range(T)]
    # delayed-outcome arrival (used only by the informative-delay sensitivity study)
    arrive_u = [rng.random(n) for _ in range(T)]
    taus_grid = ad.mondrian_thresholds(D.u_cal1, EPS_GRID)
    taus_all = taus_grid
    out = []
    for name in methods:
        f = flags(name)
        col = 'ut' if (f['pat'] and name not in ('static', 'wcp', 'ls_plugin', 'monitor', 'online')) else 'u'
        if name in ('static_pat',):
            col = 'ut'
        score = df[col].values
        hist = []  # per period: dict(idx, A, pi, O)
        tau_prev = ad.mondrian_thresholds(D.u_cal1, [cfg['alpha_m']])[0]
        for t in range(T):
            b = batches[t]
            s_b, y_b = score[b], df.y.values[b]
            suspended = False
            info = {}
            # ------------------------------------------------ decide tau_t
            if name in ('static', 'static_pat'):
                tau = ad.mondrian_thresholds(D.u_cal1, [cfg['alpha_m']])[0]
            elif name == 'oracle':
                idx, w = weights[t]
                ok = [k for k, tk in enumerate(taus_all)
                      if all(np.array(pop_risk(score[idx], df.y.values[idx], w, tk)[:2]) <=
                             [cfg['alpha_f'], cfg['alpha_m']])]
                tau = taus_all[max(ok)] if ok else -np.inf
                suspended = not ok
            elif name == 'monitor':
                p = ks_2samp(s_b, D.u_cal).pvalue
                suspended = p < 0.01
                tau = -np.inf if suspended else ad.mondrian_thresholds(D.u_cal1, [cfg['alpha_m']])[0]
            elif name == 'wcp':
                Xs = D.pca[D.idx_cal]
                Xt = D.pca[b]
                lr = LogisticRegression(C=1.0, max_iter=500).fit(
                    np.r_[Xs, Xt], np.r_[np.zeros(len(Xs)), np.ones(len(Xt))])
                pr = lr.predict_proba(Xs)[:, 1]
                wcal = np.clip(pr / (1 - pr), 1e-3, 1e3)[D.y_cal == 1]
                o = np.argsort(D.u_cal1)
                cw = np.cumsum(wcal[o]) / (wcal.sum() + np.median(wcal))
                k = np.searchsorted(cw, cfg['alpha_m'], side='right')
                tau = -np.inf if k == 0 else D.u_cal1[o][k - 1]
            elif name == 'ls_plugin':
                pi_hat = ad.em_prevalence(ad.platt(s_b, D.c_src), D.pi0)
                F1 = np.array([(D.u_cal1 < tk).mean() for tk in taus_all])
                F0 = np.array([(D.u_cal0 < tk).mean() for tk in taus_all])
                FOR = pi_hat * F1 / np.maximum(pi_hat * F1 + (1 - pi_hat) * F0, 1e-12)
                ok = np.where((FOR <= cfg['alpha_f']) & (F1 <= cfg['alpha_m']))[0]
                tau = taus_all[ok.max()] if len(ok) else -np.inf
                suspended = len(ok) == 0
                info['pi_hat'] = pi_hat
            elif name == 'online':
                tau = tau_prev
                td = t - cfg['delay']
                if td >= 0:
                    h = hist[td]
                    yo = np.where(h['O'], df.y.values[h['idx']], 0)
                    num = (h['A'] * h['O'] / h['pi'] * yo).mean()
                    pa = h['A'].mean()
                    prev = (h['O'] / h['pi'] * yo).mean()
                    g = max((num / pa if pa > 0 else 0) / cfg['alpha_f'],
                            (num / prev if prev > 0 else 0) / cfg['alpha_m']) - 1
                    tau = tau_prev - cfg['eta'] * g
                tau_prev = tau
            else:  # audit-based certification (AUDIT-Derm, ablations, naive, LTT)
                lo, hi = t - cfg['delay'] - cfg['window'] + 1, t - cfg['delay']
                win = [hist[j] for j in range(max(0, lo), hi + 1)] if hi >= 0 else []
                old = [hist[j] for j in range(0, max(0, lo))]
                if win:
                    wi = np.concatenate([h['idx'] for h in win])
                    wpi = np.concatenate([h['pi'] for h in win])
                    wU = np.concatenate([np.full(len(h['idx']), h['U']) for h in win])
                    wB = np.concatenate([h['A'] for h in win]).astype(bool)
                    obs = []
                    for j in range(max(0, lo), hi + 1):
                        o = hist[j]['O'].copy()
                        if extra.get('informative') and j == hi:
                            # informative delay: only a fraction q of malignant outcomes of the most
                            # recent period has resolved; the window is (wrongly) used as if complete
                            ymal = df.y.values[hist[j]['idx']] == 1
                            o &= (~ymal) | (arrive_u[j] < extra['informative'])
                        obs.append(o)
                    wO = np.concatenate(obs)
                else:
                    wi = np.array([], int); wpi = np.array([]); wO = np.array([], bool); wU = np.array([]); wB = np.array([], bool)
                if old:
                    oi = np.concatenate([h['idx'] for h in old])
                    opi = np.concatenate([h['pi'] for h in old])
                    oO = np.concatenate([h['O'] for h in old])
                else:
                    oi = np.array([], int); opi = np.array([]); oO = np.array([], bool)
                mfun = working_model(D, score[oi], df.y.values[oi], oO, opi, score[wi],
                                     f['update'], D.c_src)
                n_res = int(wO.sum())
                # candidate list fixed without the window: keep policies whose auto-benign share on
                # unlabelled cases *outside* the window (older periods, else development data) is
                # at least f_min, so the scan never starts at practically empty auto-benign sets
                ref_s = score[oi] if len(oi) else D.u_cal
                taus_grid = taus_all[np.array([(ref_s < tk).mean() >= cfg.get('f_min', 0.10) for tk in taus_all])]
                if len(taus_grid) == 0:
                    taus_grid = taus_all[-1:]
                if len(wi) == 0 or n_res < cfg['n_min']:
                    k_cert = 0
                else:
                    y_w = df.y.values[wi].astype(float)
                    if extra.get('label_noise'):
                        flip = (np.random.default_rng(seed + 7).random(len(y_w)) < extra['label_noise']) & (y_w == 1)
                        y_w = np.where(flip, 0.0, y_w)
                    mg = cfg.get('margin', 0.0)
                    d = cfg['delta'] if f['iut'] else cfg['delta'] / 2
                    if f['weighted']:
                        psi = ad.aipw_terms(y_w, wO.astype(float), wpi, mfun(score[wi]), f['aipw'])
                        zf, zm = ad.risk_increments(score[wi], taus_grid, psi, cfg['alpha_f'], cfg['alpha_m'])
                        if f['test'] == 'bet':
                            # region-split certification: rows referred at deployment (complete
                            # ascertainment, small range) and audited auto-benign rows (wide range)
                            Bm = wB
                            uR = np.full(len(wi), 1.0 / cfg['pi_ref'])
                            rr = []
                            for z, sc in ((zf, 1.0), (zm, 1 - cfg['alpha_m'])):
                                rr.append(ad.split_certify(z, Bm, sc * uR, sc * wU, d, mg))
                            rf, rm = rr
                        elif f['test'] == 'hoeff':
                            Ug = wU.max()
                            rf = ad.hoeffding_reject(zf + mg, -Ug, Ug + mg, d)
                            rm = ad.hoeffding_reject(zm + mg, -Ug, Ug + mg, d)
                        else:
                            rf = ad.wald_reject(zf + mg, d)
                            rm = ad.wald_reject(zm + mg, d)
                    else:  # complete-case analysis that ignores the inclusion probabilities
                        psi = y_w[wO]
                        zf, zm = ad.risk_increments(score[wi][wO], taus_grid, psi, cfg['alpha_f'], cfg['alpha_m'])
                        rf = ad.betting_ucb(zf, 1.0, d) + mg < 0
                        rm = ad.betting_ucb(zm, 1 - cfg['alpha_m'], d) + mg < 0
                    k_cert = ad.fixed_sequence(rf & rm)
                if k_cert == 0:
                    if f['suspend']:
                        tau, suspended = -np.inf, True
                    else:
                        tau = ad.mondrian_thresholds(D.u_cal1, [cfg['alpha_m']])[0]
                else:
                    tau = taus_grid[k_cert - 1]
                info['k_cert'] = k_cert
                info['n_res'] = n_res
                m_b = mfun(s_b)
            # ------------------------------------------------ deploy for period t
            A = (s_b < tau)
            audit_like = name in ('audit', 'naive_audit', 'ltt_ipw') or name.startswith('audit_')
            if audit_like and f['adaptive']:
                # design scale fixed before the period from unlabelled recent cases
                ref_s = score[wi] if len(wi) else s_b[:0]
                c_t = ad.design_scale(mfun(ref_s[ref_s < tau]), cfg['budget'], cfg['pi_min'])
                pi = ad.inclusion_probs(m_b, A, c_t, cfg['pi_ref'], cfg['pi_min'], True)
            else:
                c_t = 1.0
                pi = ad.inclusion_probs(np.full(n, .5), A, c_t, cfg['pi_ref'], cfg['pi_min'],
                                        adaptive=False, budget=cfg['budget'])
            U_t = ad.psi_upper(c_t, cfg['pi_ref'], cfg['pi_min'], f['adaptive'] if audit_like else False,
                               cfg['budget'], suspended or not np.isfinite(tau))
            O = audit_u[t] < pi
            hist.append(dict(idx=b, A=A, pi=pi, O=O, U=U_t))
            idx, w = weights[t]
            FOR, MISS, REF = pop_risk(score[idx], df.y.values[idx], w, tau)
            out.append(dict(method=name, site=site, scen=scen, seed=seed, t=t, tau=float(tau),
                            FOR=FOR, MISS=MISS, REF=REF, susp=bool(suspended),
                            audits=int((O & A).sum()), n_auto=int(A.sum()),
                            viol=bool(FOR > cfg['alpha_f'] or MISS > cfg['alpha_m']), **info))
    return out
