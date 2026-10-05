"""E9: subgroup risks of the thresholds actually deployed by AUDIT-Derm (compound scenario)."""
import json, numpy as np, pandas as pd
import simulate as S

D = S.Data()
df = D.df
E1 = pd.read_csv('E1_main.csv')
# evaluate every policy on the same periods: those in which AUDIT-Derm automated (compound shift)
_a = E1[(E1.method == 'audit') & (E1.scen == 'comp') & np.isfinite(E1.tau)]
AUTO_KEYS = set(zip(_a.site, _a.seed, _a.t))
E1 = E1[[k in AUTO_KEYS for k in zip(E1.site, E1.seed, E1.t)]]
rows = E1[(E1.method.isin(['audit', 'static', 'oracle'])) & (E1.scen == 'comp') & np.isfinite(E1.tau)]
age = df.age.values
groups = {
    'Female': (df.sex == 'female').values, 'Male': (df.sex == 'male').values,
    'Age < 50': age < 50, 'Age 50-69': (age >= 50) & (age < 70), 'Age >= 70': age >= 70,
    'Head/neck': df.site.astype(str).str.lower().str.contains('head|face|neck|scalp|ear|nose|lip').values,
    'Trunk': df.site.astype(str).str.lower().str.contains('trunk|back|chest|abdomen').values,
    'Extremities': df.site.astype(str).str.lower().str.contains('extremit|arm|leg|hand|foot').values,
    'Fitzpatrick I-II': df.fitz.astype(str).str.contains(r'^(I|II|1|2)$|^I$|^II$').values,
    'Fitzpatrick III-VI': df.fitz.astype(str).str.contains(r'^(III|IV|V|VI|3|4|5|6)$').values,
}
out = {}
for m in ['audit', 'static', 'oracle']:
    res = {g: dict(FOR=[], MISS=[], share=[]) for g in groups}
    for _, r in rows[rows.method == m].iterrows():
        idx, w = S.period_weights(D, r.site, 'comp', int(r.t), 12)
        sc = df['ut' if m != 'static' else 'u'].values[idx]
        y = df.y.values[idx]
        for g, mask in groups.items():
            k = mask[idx]
            if k.sum() < 30 or y[k].sum() < 5:
                continue
            ww = w[k] / w[k].sum()
            FOR, MISS, REF = S.pop_risk(sc[k], y[k], ww, r.tau)
            res[g]['FOR'].append(FOR); res[g]['MISS'].append(MISS); res[g]['share'].append(1 - REF)
    out[m] = {g: {k: (float(np.mean(v)) if v else None) for k, v in d.items()} | {
        'MISS_p95': float(np.percentile(d['MISS'], 95)) if d['MISS'] else None,
        'FOR_p95': float(np.percentile(d['FOR'], 95)) if d['FOR'] else None,
        'n_periods': len(d['MISS'])} for g, d in res.items()}
json.dump(out, open('E9_subgroups.json', 'w'), indent=1)
print(json.dumps(out['audit'], indent=0))
