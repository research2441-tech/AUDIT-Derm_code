"""E8: paired acquisition transport validation, prevalence estimation and shift attribution."""
import json, numpy as np, pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score
import audit_derm as ad
import simulate as S


def main():
    D = S.Data()
    df = D.df
    res = {'maps': D.maps}
    # ---- held-out pairs (lesions in the evaluation pools that have both conditions)
    hp = {}
    for site, cond in (('MILK', 'closeup'), ('HIBA', 'overview')):
        d = df[(df.cohort == site) & (df.split == 'pool')]
        a = d[d.condition == cond].groupby('group')[['u', 'ut']].mean()
        b = d[d.condition == 'dermoscopic'].groupby('group').u.mean()
        k = a.index.intersection(b.index)
        hp[cond] = dict(n=int(len(k)),
                        gap_raw=float(np.mean(np.abs(b[k].values - a.loc[k, 'u'].values))),
                        gap_pat=float(np.mean(np.abs(b[k].values - a.loc[k, 'ut'].values))),
                        bias_raw=float(np.mean(a.loc[k, 'u'].values - b[k].values)),
                        bias_pat=float(np.mean(a.loc[k, 'ut'].values - b[k].values)),
                        r=float(np.corrcoef(a.loc[k, 'u'].values, b[k].values)[0, 1]))
    res['heldout_pairs'] = hp
    # ---- discrimination per cohort / condition, and mixed-condition AUROC
    auc = {}
    for site in ['MSKCC', 'HIBA', 'PAD', 'MILK']:
        d = df.loc[D.pools[site]]
        auc[site] = dict(raw=float(roc_auc_score(d.y, d.u)), pat=float(roc_auc_score(d.y, d.ut)),
                         n=int(len(d)), prev=float(d.y.mean()))
        for c, g in d.groupby('condition'):
            auc[site][c] = dict(auc=float(roc_auc_score(g.y, g.u)), n=int(len(g)), prev=float(g.y.mean()))
    d = df.loc[D.pools['SRC']]
    auc['SRC'] = dict(raw=float(roc_auc_score(d.y, d.u)), n=int(len(d)), prev=float(d.y.mean()))
    res['auc'] = auc
    # ---- prevalence estimation and attribution on single batches
    rows = []
    src_mix = lambda pi, k: np.r_[np.random.default_rng(k).choice(D.u_cal1, int(4000 * pi)),
                                  np.random.default_rng(k + 1).choice(D.u_cal0, int(4000 * (1 - pi)))]
    for site, scen in [('MSKCC', 'acq'), ('HIBA', 'acq'), ('PAD', 'acq'), ('MILK', 'acq'),
                       ('SRC', 'prev_lo'), ('SRC', 'prev_hi'),
                       ('MSKCC', 'comp'), ('HIBA', 'comp'), ('PAD', 'comp'), ('MILK', 'comp')]:
        for seed in range(20):
            rng = np.random.default_rng(1000 + seed)
            idx, w = S.period_weights(D, site, scen, 0, 12)
            b = rng.choice(idx, 400, p=w)
            y = df.y.values[b]; u = df.u.values[b]; ut = df.ut.values[b]
            truth = float((w * df.y.values[idx]).sum())
            em_raw = ad.em_prevalence(ad.platt(u, D.c_src), D.pi0)
            em_pat = ad.em_prevalence(ad.platt(ut, D.c_src), D.pi0)
            # BBSE with a single source threshold (binary confusion inversion)
            thr = np.median(D.u_cal)
            tpr, fpr = (D.u_cal1 >= thr).mean(), (D.u_cal0 >= thr).mean()
            bb = lambda s: float(np.clip(((s >= thr).mean() - fpr) / (tpr - fpr), 0, 1))
            # audit: 10% uniform outcome sample, Horvitz-Thompson and model-assisted estimates
            pi = np.full(400, 0.10); O = rng.random(400) < pi
            ht = float(np.mean(O / pi * y))
            m = ad.prior_shift(ad.platt(ut, D.c_src), em_pat, D.pi0)
            aipw = float(np.mean(ad.aipw_terms(y.astype(float), O.astype(float), pi, m)))
            # attribution of the score shift (Wasserstein-1 on logit scores)
            ref0 = src_mix(D.pi0, seed)
            d0 = wasserstein_distance(u, ref0)
            d1 = wasserstein_distance(ut, ref0)
            d2 = wasserstein_distance(ut, src_mix(em_pat, seed))
            rows.append(dict(site=site, scen=scen, seed=seed, truth=truth, em_raw=em_raw, em_pat=em_pat,
                             bbse_raw=bb(u), bbse_pat=bb(ut), ht=ht, aipw=aipw, d0=d0, d1=d1, d2=d2))
    pr = pd.DataFrame(rows)
    pr.to_csv('E8_prevalence.csv', index=False)
    summ = {}
    for (site, scen), g in pr.groupby(['site', 'scen']):
        summ['%s/%s' % (site, scen)] = {k: float(np.mean(np.abs(g[k] - g.truth))) for k in
                                        ['em_raw', 'em_pat', 'bbse_raw', 'bbse_pat', 'ht', 'aipw']}
        summ['%s/%s' % (site, scen)].update(truth=float(g.truth.mean()), d0=float(g.d0.mean()),
                                            d1=float(g.d1.mean()), d2=float(g.d2.mean()),
                                            sd_ht=float(g.ht.std()), sd_aipw=float(g.aipw.std()))
    res['prevalence'] = summ
    json.dump(res, open('E8_transport.json', 'w'), indent=1)
    print(json.dumps(res, indent=1)[:6000])


if __name__ == '__main__':
    main()
