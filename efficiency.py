"""E6: computational cost of each stage (run on an otherwise idle machine)."""
import time, json, numpy as np, pandas as pd
import simulate as S, audit_derm as ad

D = S.Data()
out = {}
# per-case deployment work (transport + decision + audit draw), vectorised over a period
u = D.df.u.values[:2500]; cond = D.df.condition.values[:2500]
t0 = time.perf_counter()
for _ in range(200):
    ut = ad.apply_transport(u, cond, D.maps)
    A = ut < 0.0
    pi = ad.inclusion_probs(np.full(2500, .1), A, 2.0, 1.0, 0.2, True)
    O = np.random.random(2500) < pi
out['deploy_us_per_case'] = (time.perf_counter() - t0) / 200 / 2500 * 1e6
# certification of one window (N = 7500, K = 40)
rng = np.random.default_rng(0)
N, K = 7500, 40
taus = ad.mondrian_thresholds(D.u_cal1, S.EPS_GRID)
idx = rng.choice(D.pools['HIBA'], N)
y = D.df.y.values[idx].astype(float); s = D.df.ut.values[idx]
B = s < taus[15]
pi = np.where(B, 0.4, 1.0); O = rng.random(N) < pi
psi = ad.aipw_terms(y, O.astype(float), pi, np.full(N, 0.1))
zf, zm = ad.risk_increments(s, taus, psi, 0.05, 0.10)
times = []
for _ in range(5):
    t0 = time.perf_counter()
    ad.split_certify(zf, B, np.ones(N), np.full(N, 5.0), 0.1)
    ad.split_certify(zm, B, 0.9 * np.ones(N), np.full(N, 4.5), 0.1)
    times.append(time.perf_counter() - t0)
out['certify_s'] = float(np.median(times))
out['increment_MB'] = (zf.nbytes + zm.nbytes) / 1e6
# whole controller step of the full method inside a stream, amortised
t0 = time.perf_counter()
S.run_stream(D, 'HIBA', 'comp', 0, ['audit'])
out['stream_audit_s'] = time.perf_counter() - t0
t0 = time.perf_counter()
S.run_stream(D, 'HIBA', 'comp', 0, ['static'])
out['stream_static_s'] = time.perf_counter() - t0
t0 = time.perf_counter()
S.run_stream(D, 'HIBA', 'comp', 0, ['wcp'])
out['stream_wcp_s'] = time.perf_counter() - t0
t0 = time.perf_counter()
S.run_stream(D, 'HIBA', 'comp', 0, ['audit_hoeffding'])
out['stream_hoeff_s'] = time.perf_counter() - t0
out['encoder_img_per_s'] = 23427 / 1481.4
out['head_params'] = 513
json.dump(out, open('E6_efficiency.json', 'w'), indent=1)
print(out)
