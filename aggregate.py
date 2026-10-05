"""Aggregate experiment logs into exp/summary.json (all numbers used in the manuscript)."""
import json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
import simulate as S

rng = np.random.default_rng(0)
NAMES = {'static': 'Static Mondrian CP', 'static_pat': 'Static + transport', 'wcp': 'Weighted CP',
         'ls_plugin': 'Label-shift plug-in', 'monitor': 'Monitor-and-abstain', 'online': 'Online risk control',
         'naive_audit': 'Naive audit', 'ltt_ipw': 'LTT-IPW', 'audit': 'AUDIT-Derm', 'oracle': 'Oracle',
         'audit_noPAT': 'w/o paired transport', 'audit_noAIPW': 'w/o augmentation (IPW)',
         'audit_uniform': 'uniform audit allocation', 'audit_hoeffding': 'Hoeffding instead of betting',
         'audit_bonf': 'Bonferroni instead of IUT', 'audit_nosuspend': 'fallback instead of suspension',
         'audit_noupdate': 'w/o outcome-based recalibration', 'audit_wald': 'Wald (asymptotic) test'}


def boot(x, B=2000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    m = rng.choice(x, (B, len(x))).mean(1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def add_workload(df):
    n = df['x_n'] if 'x_n' in df else pd.Series(S.CFG['n'], index=df.index)
    n = n.fillna(S.CFG['n']) if hasattr(n, 'fillna') else n
    df['work'] = df.REF + df.audits / n
    return df


def stream_means(df, cols=('viol', 'REF', 'work', 'FOR', 'MISS', 'susp', 'audits')):
    return df.groupby(['method', 'site', 'scen', 'seed'])[list(cols)].mean().reset_index()


def main():
    out = {}
    E1 = add_workload(pd.read_csv('E1_main.csv'))
    sm = stream_means(E1)
    # ---------------- main table: scenario groups
    groups = {'acq': ['acq'], 'prev': ['prev_lo', 'prev_hi'], 'comp': ['comp'], 'drift': ['drift']}
    main = {}
    for m in S.METHODS:
        row = {}
        for g, sc in groups.items():
            x = sm[(sm.method == m) & sm.scen.isin(sc)]
            row[g] = dict(viol=float(x.viol.mean()), viol_ci=boot(x.viol), REF=float(x.REF.mean()),
                          work=float(x.work.mean()), FOR=float(x.FOR.mean()), MISS=float(x.MISS.mean()),
                          susp=float(x.susp.mean()))
        main[m] = row
    out['main'] = main
    # worst-case period risks and per-site detail
    persite = {}
    for (m, site, sc), x in sm.groupby(['method', 'site', 'scen']):
        persite.setdefault(m, {}).setdefault(site, {})[sc] = dict(viol=float(x.viol.mean()), REF=float(x.REF.mean()),
                                                                  work=float(x.work.mean()), susp=float(x.susp.mean()),
                                                                  MISS=float(x.MISS.mean()), FOR=float(x.FOR.mean()))
    out['persite'] = persite
    mx = E1.groupby('method')[['FOR', 'MISS']].max()
    out['max_risk'] = {m: dict(FOR=float(mx.loc[m, 'FOR']), MISS=float(mx.loc[m, 'MISS'])) for m in mx.index}
    out['audit_viol_periods'] = int(E1[(E1.method == 'audit')].viol.sum())
    out['audit_periods'] = int((E1.method == 'audit').sum())
    # ---------------- statistics: compound + drift streams, audit vs each comparator
    cd = sm[sm.scen.isin(['comp', 'drift'])]
    a = cd[cd.method == 'audit'].set_index(['site', 'scen', 'seed'])
    stats = {}
    ps = []
    for m in S.METHODS:
        if m == 'audit':
            continue
        b = cd[cd.method == m].set_index(['site', 'scen', 'seed']).loc[a.index]
        res = {}
        for col in ('viol', 'REF'):
            dlt = (a[col] - b[col]).values
            nz = dlt[dlt != 0]
            if len(nz) < 1:
                p, rbc = 1.0, 0.0
            else:
                p = float(wilcoxon(dlt, zero_method='wilcox').pvalue) if len(nz) > 0 else 1.0
                rp = pd.Series(np.abs(nz)).rank().values
                rbc = float((rp[nz > 0].sum() - rp[nz < 0].sum()) / rp.sum())
            res[col] = dict(diff=float(dlt.mean()), ci=boot(dlt), p=p, rbc=rbc)
            ps.append((m, col, p))
        stats[m] = res
    # Holm correction within each metric
    for col in ('viol', 'REF'):
        lst = sorted([(p, m) for m, c, p in ps if c == col])
        k = len(lst); prev = 0
        for i, (p, m) in enumerate(lst):
            adj = min(1.0, max(prev, (k - i) * p)); prev = adj
            stats[m][col]['p_holm'] = adj
    out['stats'] = stats
    out['n_streams_cd'] = int(len(a))
    # ---------------- ablation
    try:
        E2 = add_workload(pd.read_csv('E2_ablation.csv'))
        s2 = stream_means(E2)
        full = s2[s2.method == 'audit'].set_index(['site', 'scen', 'seed'])
        abl = {}
        for m in ['audit'] + S.ABLATIONS + ['oracle']:
            x = s2[s2.method == m].set_index(['site', 'scen', 'seed']).loc[full.index]
            dd = (x.REF - full.REF).values
            abl[m] = dict(viol=float(x.viol.mean()), viol_ci=boot(x.viol), REF=float(x.REF.mean()),
                          work=float(x.work.mean()), susp=float(x.susp.mean()), audits=float(x.audits.mean()),
                          dREF=float(dd.mean()), dREF_ci=boot(dd), MISS=float(x.MISS.mean()), FOR=float(x.FOR.mean()),
                          maxMISS=float(E2[E2.method == m].MISS.max()), maxFOR=float(E2[E2.method == m].FOR.max()))
        out['ablation'] = abl
    except FileNotFoundError:
        pass
    json.dump(out, open('summary_raw.json', 'w'), indent=1)
    print(json.dumps({k: out[k] for k in ['main']}, indent=0)[:4000])


if __name__ == '__main__':
    main()
