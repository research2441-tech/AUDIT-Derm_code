"""Monte Carlo check of the region-split betting bound: P(UCB_R + UCB_B < true mean) <= delta."""
import json, numpy as np
import simulate as S, audit_derm as ad

D = S.Data(); df = D.df
rng = np.random.default_rng(7)
idx, w = S.period_weights(D, 'HIBA', 'comp', 0, 12)
sc, yy = df.ut.values[idx], df.y.values[idx].astype(float)
taus = ad.mondrian_thresholds(D.u_cal1, S.EPS_GRID)
tau_dep = taus[18]
out = {}
for delta in (0.05, 0.10, 0.20):
    miss = {'F': 0, 'M': 0}; R = 600
    for r in range(R):
        b = rng.choice(len(idx), 7500, p=w)
        s, y = sc[b], yy[b]
        B = s < tau_dep
        m = np.clip(0.02 + 0.0 * s, 0, 1)
        pi = np.where(B, 0.2 + 0.4 * rng.random(7500) * 0, 1.0)
        O = rng.random(7500) < pi
        psi = ad.aipw_terms(y, O.astype(float), pi, m)
        zf, zm = ad.risk_increments(s, taus[10:30:4], psi, 0.05, 0.10)
        truef = np.array([(w * (sc < t) * (yy - 0.05)).sum() for t in taus[10:30:4]])
        truem = np.array([(w * yy * ((sc < t) - 0.10)).sum() for t in taus[10:30:4]])
        for z, tr, k, sc_ in ((zf, truef, 'F', 1.0), (zm, truem, 'M', 0.9)):
            u = ad.betting_ucb(z * (~B)[:, None], sc_, delta / 2, ~B) + ad.betting_ucb(z * B[:, None], sc_ * 5.0, delta / 2, B)
            miss[k] += int(np.any(u < tr))
    out['delta=%g' % delta] = {k: v / R for k, v in miss.items()}
    print(delta, out['delta=%g' % delta], flush=True)
json.dump(out, open('coverage_check.json', 'w'), indent=1)
