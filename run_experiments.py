"""Run all deployment experiments (E1-E9) and write long-format CSV logs to exp/."""
import sys, time, json, numpy as np, pandas as pd
from multiprocessing import Pool
import simulate as S

SITES = ['MSKCC', 'HIBA', 'PAD', 'MILK']
D = None


def init():
    global D
    D = S.Data()


def job(args):
    site, scen, seed, methods, extra = args
    t0 = time.time()
    rows = S.run_stream(D, site, scen, seed, methods, extra=extra)
    for r in rows:
        r.update({('x_' + k): v for k, v in extra.items()})
    return rows, time.time() - t0


def run(name, jobs, procs=2):
    t0 = time.time()
    out = []
    with Pool(procs, initializer=init) as p:
        for rows, dt in p.imap_unordered(job, jobs, chunksize=1):
            out.extend(rows)
    df = pd.DataFrame(out)
    df.to_csv('%s.csv' % name, index=False)
    print(name, len(jobs), 'streams', round(time.time() - t0, 1), 's', flush=True)
    return df


if __name__ == '__main__':
    which = sys.argv[1:]
    seeds = range(20)
    if 'main' in which:
        jobs = [(s, sc, k, S.METHODS, {}) for s in SITES for sc in ('acq', 'comp', 'drift') for k in seeds]
        jobs += [('SRC', sc, k, S.METHODS, {}) for sc in ('prev_lo', 'prev_hi') for k in seeds]
        run('E1_main', jobs)
    if 'ablation' in which:
        jobs = [(s, sc, k, ['audit'] + S.ABLATIONS + ['oracle'], {}) for s in SITES for sc in ('comp', 'drift') for k in seeds]
        run('E2_ablation', jobs)
    if 'sens' in which:
        jobs = []
        for b in (0.10, 0.20, 0.30, 0.60):
            jobs += [(s, 'comp', k, ['audit', 'audit_uniform', 'audit_noAIPW'], {'budget': b, 'pi_min': b / 2}) for s in SITES for k in range(10)]
        for d in (0.05, 0.20):
            jobs += [(s, 'comp', k, ['audit'], {'delta': d}) for s in SITES for k in range(10)]
        for L in (2, 4):
            jobs += [(s, 'comp', k, ['audit'], {'window': L}) for s in SITES for k in range(10)]
        for n in (1000, 4000):
            jobs += [(s, 'comp', k, ['audit'], {'n': n}) for s in SITES for k in range(10)]
        for af, am in ((0.05, 0.05), (0.03, 0.10), (0.10, 0.10)):
            jobs += [(s, 'comp', k, ['audit', 'oracle'], {'alpha_f': af, 'alpha_m': am}) for s in SITES for k in range(10)]
        run('E3_sensitivity', jobs)
    if 'volume' in which:
        jobs = []
        for b in (0.04, 0.06):
            jobs += [(s, 'comp', k, ['audit'], {'budget': b, 'pi_min': b / 2}) for s in SITES for k in range(10)]
        run('E3b_volume', jobs)
    if 'robust' in which:
        jobs = []
        for d in (2, 3):
            jobs += [(s, 'comp', k, ['audit'], {'delay': d}) for s in SITES for k in range(10)]
        for q in (0.3, 0.6):
            jobs += [(s, 'comp', k, ['audit'], {'informative': q}) for s in SITES for k in range(10)]
        for ln in (0.05, 0.10):
            jobs += [(s, 'comp', k, ['audit'], {'label_noise': ln}) for s in SITES for k in range(10)]
        jobs += [(s, 'comp', k, ['audit'], {'pi_ref': 0.8}) for s in SITES for k in range(10)]
        for mg in (0.005, 0.01, 0.02):
            jobs += [(s, 'drift', k, ['audit', 'oracle'], {'margin': mg}) for s in SITES for k in range(10)]
        run('E4_robustness', jobs)
