"""E9b: which malignant diagnoses are dismissed by the deployed thresholds (compound shift)."""
import json, numpy as np, pandas as pd
import simulate as S
D = S.Data(); df = D.df
E1 = pd.read_csv('E1_main.csv')
# evaluate every policy on the same periods: those in which AUDIT-Derm automated (compound shift)
_a = E1[(E1.method == 'audit') & (E1.scen == 'comp') & np.isfinite(E1.tau)]
AUTO_KEYS = set(zip(_a.site, _a.seed, _a.t))
E1 = E1[[k in AUTO_KEYS for k in zip(E1.site, E1.seed, E1.t)]]
dx = df.dx3.fillna('').str.lower()
cat = np.where(dx.str.contains('basal'), 'BCC', np.where(dx.str.contains('melanoma'), 'Melanoma',
               np.where(dx.str.contains('squamous|keratoacanthoma'), 'SCC/KA', 'other')))
out = {}
for m in ['audit', 'static', 'oracle', 'online']:
    rows = E1[(E1.method == m) & (E1.scen == 'comp') & np.isfinite(E1.tau)]
    acc = {c: [] for c in ['BCC', 'Melanoma', 'SCC/KA']}
    share = {c: [] for c in acc}
    for _, r in rows.iterrows():
        idx, w = S.period_weights(D, r.site, 'comp', int(r.t), 12)
        sc = df['u' if m in ('static', 'online') else 'ut'].values[idx]
        y = df.y.values[idx]; cc = cat[idx]
        mal = y == 1
        for c in acc:
            k = mal & (cc == c)
            if w[k].sum() <= 0:
                continue
            acc[c].append(float((w[k] * (sc[k] < r.tau)).sum() / w[k].sum()))
            share[c].append(float(w[k].sum() / w[mal].sum()))
    out[m] = {c: dict(miss=float(np.mean(acc[c])) if acc[c] else None, share=float(np.mean(share[c])) if share[c] else None,
                      n=len(acc[c])) for c in acc}
json.dump(out, open('E9b_dx.json', 'w'), indent=1)
print(json.dumps(out, indent=0))
