"""Summaries of the sensitivity (E3) and robustness (E4) studies -> exp/summary_sens.json."""
import json, numpy as np, pandas as pd
import simulate as S

rng = np.random.default_rng(1)


def boot(x, B=2000):
    x = np.asarray(x, float)
    m = rng.choice(x, (B, len(x))).mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summ(x, n=S.CFG['n']):
    sm = x.groupby(['site', 'seed']).agg(viol=('viol', 'mean'), REF=('REF', 'mean'), susp=('susp', 'mean'),
                                         aud=('audits', 'mean'), MISS=('MISS', 'max'), FOR=('FOR', 'max')).reset_index()
    return dict(viol=float(sm.viol.mean()), viol_ci=boot(sm.viol), REF=float(sm.REF.mean()), susp=float(sm.susp.mean()),
                work=float(sm.REF.mean() + sm.aud.mean() / n), maxMISS=float(x.MISS.max()), maxFOR=float(x.FOR.max()),
                n_streams=int(len(sm)))


def main():
    out = {}
    E1 = pd.read_csv('E1_main.csv')
    ref = E1[(E1.method == 'audit') & (E1.scen == 'comp') & (E1.seed < 10)]
    out['reference_comp'] = summ(ref)
    refd = E1[(E1.method == 'audit') & (E1.scen == 'drift') & (E1.seed < 10)]
    out['reference_drift'] = summ(refd)
    E3 = pd.read_csv('E3_sensitivity.csv')
    E4 = pd.read_csv('E4_robustness.csv')
    s3 = {}
    for b in sorted(E3.x_budget.dropna().unique()):
        for m in ('audit', 'audit_uniform', 'audit_noAIPW'):
            s3['budget=%g/%s' % (b, m)] = summ(E3[(E3.x_budget == b) & (E3.method == m)])
    for col, key in (('x_delta', 'delta'), ('x_window', 'window'), ('x_n', 'n')):
        if col in E3:
            for v in sorted(E3[col].dropna().unique()):
                x = E3[(E3[col] == v) & (E3.method == 'audit')]
                s3['%s=%g' % (key, v)] = summ(x, n=v if key == 'n' else S.CFG['n'])
    if 'x_alpha_f' in E3:
        for (af, am), x in E3[E3.x_alpha_f.notna()].groupby(['x_alpha_f', 'x_alpha_m']):
            s3['alpha=%g,%g/audit' % (af, am)] = summ(x[x.method == 'audit'])
            s3['alpha=%g,%g/oracle' % (af, am)] = summ(x[x.method == 'oracle'])
    out['sens'] = s3
    s4 = {}
    for col, key in (('x_delay', 'delay'), ('x_informative', 'informative'), ('x_label_noise', 'label_noise'),
                     ('x_pi_ref', 'pi_ref')):
        if col in E4:
            for v in sorted(E4[col].dropna().unique()):
                s4['%s=%g' % (key, v)] = summ(E4[(E4[col] == v) & (E4.method == 'audit')])
    if 'x_margin' in E4:
        for v in sorted(E4.x_margin.dropna().unique()):
            x = E4[(E4.x_margin == v)]
            s4['margin=%g/audit' % v] = summ(x[x.method == 'audit'])
            # post-change periods only (t = 6, 7, 8)
            s4['margin=%g/audit_post' % v] = summ(x[(x.method == 'audit') & x.t.isin([6, 7, 8])])
        s4['margin=0/audit_post'] = summ(refd[refd.t.isin([6, 7, 8])])
    out['robust'] = s4
    json.dump(out, open('summary_sens.json', 'w'), indent=1)
    for k, v in {**s3, **s4}.items():
        print(k.ljust(32), 'V %.3f REF %.3f W %.3f susp %.2f maxM %.3f maxF %.3f' % (v['viol'], v['REF'], v['work'], v['susp'], v['maxMISS'], v['maxFOR']))


if __name__ == '__main__':
    main()
