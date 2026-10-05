"""Result figures (grayscale, publication sizes). Run from this folder."""
import json, sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 8, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'legend.frameon': False,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 600})
COL1 = 3.45
MK = {'audit': ('o', '0.0', '-'), 'static': ('s', '0.55', '--'), 'static_pat': ('D', '0.55', ':'),
      'wcp': ('^', '0.35', '-.'), 'ls_plugin': ('v', '0.35', '--'), 'monitor': ('P', '0.6', ':'),
      'online': ('X', '0.25', '-.'), 'naive_audit': ('<', '0.45', '--'), 'ltt_ipw': ('>', '0.7', ':'),
      'oracle': ('*', '0.0', ':')}
NAMES = {'static': 'Static Mondrian CP', 'static_pat': 'Static + transport', 'wcp': 'Weighted CP',
         'ls_plugin': 'Label-shift plug-in', 'monitor': 'Monitor-and-abstain', 'online': 'Online risk control',
         'naive_audit': 'Naive audit', 'ltt_ipw': 'LTT-IPW', 'audit': 'AUDIT-Derm', 'oracle': 'Oracle'}


def fig_transport():
    import simulate as S
    D = S.Data(); df = D.df
    r = json.load(open('E8_transport.json'))
    fig, ax = plt.subplots(2, 1, figsize=(COL1, 5.6), gridspec_kw=dict(height_ratios=[1.05, 1], hspace=0.62))
    a = ax[0]
    for (site, cond, mk, lab) in (('MILK', 'closeup', 'o', 'close-up (MILK10k)'), ('HIBA', 'overview', '^', 'overview (HIBA)')):
        d = df[(df.cohort == site) & (df.split == 'pool')]
        x = d[d.condition == cond].groupby('group').u.mean()
        y = d[d.condition == 'dermoscopic'].groupby('group').u.mean()
        k = x.index.intersection(y.index)
        a.scatter(x[k], y[k], s=9, marker=mk, facecolor='white' if mk == 'o' else '0.55', edgecolor='0.1', lw=0.5,
                  label='held-out pairs, ' + lab, zorder=3)
        mp = r['maps'][cond]
        xx = np.linspace(-6, 4, 50)
        a.plot(xx, mp['a'] * xx + mp['b'], color='0.1', lw=1.0, ls='-' if mk == 'o' else '--',
               label='$g_c$ fitted on other pairs (%s)' % ('close-up' if mk == 'o' else 'overview'))
    a.plot([-6, 4], [-6, 4], color='0.6', lw=0.7, ls=':', label='identity')
    a.set_xlim(-6, 4); a.set_ylim(-6, 4)
    a.set_xlabel('clinical-image logit $u^{c}$'); a.set_ylabel('dermoscopic logit $u^{\\mathrm{der}}$ (same lesion)')
    a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=5.6, handlelength=1.8, columnspacing=0.6)
    a.set_title('(a) Paired acquisition transport', fontsize=8, loc='left')
    a.set_aspect('equal')
    b = ax[1]
    pv = r['prevalence']
    cats = [('MSKCC/acq', 'MSKCC\nacq.'), ('HIBA/acq', 'HIBA\nacq.'), ('PAD/acq', 'PAD\nacq.'), ('MILK/acq', 'MILK\nacq.'),
            ('SRC/prev_lo', 'Int.\nprev 0.1'), ('SRC/prev_hi', 'Int.\nprev 0.4'), ('HIBA/comp', 'HIBA\ncomp.'), ('PAD/comp', 'PAD\ncomp.')]
    meths = [('em_raw', 'EM, raw scores', 'white', ''), ('em_pat', 'EM, transported', '0.75', ''),
             ('bbse_pat', 'BBSE, transported', 'white', '////'), ('ht', 'Audit HT (40 labels)', '0.35', ''),
             ('aipw', 'Audit AIPW (40 labels)', '0.05', '')]
    w = 0.16
    for j, (k, lab, fc, h) in enumerate(meths):
        vals = [pv[c][k] for c, _ in cats]
        b.bar(np.arange(len(cats)) + (j - 2) * w, vals, w, color=fc, edgecolor='0.1', lw=0.5, hatch=h, label=lab)
    b.set_xticks(np.arange(len(cats))); b.set_xticklabels([c[1] for c in cats], fontsize=5.6)
    b.legend_ = None
    b.set_ylabel('mean absolute prevalence error'); b.set_ylim(0, 1.02)
    b.legend(ncol=2, fontsize=5.6, loc='upper center', bbox_to_anchor=(0.5, -0.2), columnspacing=0.8, handlelength=1.4)
    b.set_title('(b) Prevalence error, one batch of 400 cases', fontsize=8, loc='left')
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.13, top=0.96)
    fig.savefig('fig2_transport.png'); plt.close(fig)


def fig_tradeoff():
    E1 = pd.read_csv('E1_main.csv')
    sm = E1.groupby(['method', 'scen', 'site', 'seed'])[['viol', 'REF']].mean().reset_index()
    groups = [('acq', ['acq'], '(a) Acquisition shift'), ('prev', ['prev_lo', 'prev_hi'], '(b) Prevalence shift'),
              ('comp', ['comp'], '(c) Compound shift'), ('drift', ['drift'], '(d) Compound drift')]
    fig, axs = plt.subplots(2, 2, figsize=(COL1, 3.5), sharex=True, sharey=True)
    for ax, (g, sc, title) in zip(axs.ravel(), groups):
        for m, (mk, c, _) in MK.items():
            x = sm[(sm.method == m) & sm.scen.isin(sc)]
            ax.scatter(x.REF.mean(), 100 * x.viol.mean(), marker=mk, s=30 if m != 'oracle' else 55,
                       facecolor=c if m in ('audit', 'oracle') else 'white', edgecolor=c if m not in ('audit',) else '0',
                       lw=0.9, zorder=3, label=NAMES[m])
        ax.axhline(10, color='0.4', lw=0.7, ls='--', zorder=1)
        ax.set_title(title, fontsize=7.5, loc='left', pad=2)
        ax.set_xlim(0.25, 1.03); ax.set_ylim(-5, 105)
    for ax in axs[1]: ax.set_xlabel('mean referral rate')
    for ax in axs[:, 0]: ax.set_ylabel('violating periods (%)')
    axs[0, 0].text(0.27, 13, '$\\delta=10\\%$', fontsize=6.5, color='0.3')
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=6.0, columnspacing=0.8, handletextpad=0.3, borderaxespad=0.2)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.95, bottom=0.27, hspace=0.3, wspace=0.12)
    fig.savefig('fig3_tradeoff.png'); plt.close(fig)


def fig_trajectory():
    E1 = pd.read_csv('E1_main.csv')
    x = E1[(E1.scen == 'drift') & (E1.site == 'HIBA')]
    ms = ['audit', 'static', 'ls_plugin', 'online', 'oracle']
    fig, axs = plt.subplots(3, 1, figsize=(COL1, 4.1), sharex=True)
    for m in ms:
        mk, c, ls = MK[m]
        g = x[x.method == m].groupby('t')
        for ax, col in zip(axs, ['MISS', 'FOR', 'REF']):
            mu = g[col].mean()
            if m == 'audit':
                lo, hi = g[col].quantile(0.1), g[col].quantile(0.9)
                ax.fill_between(mu.index + 1, lo.values, hi.values, color='0.75', lw=0, zorder=1,
                                label='AUDIT-Derm 10–90% of streams')
            ax.plot(mu.index + 1, mu.values, marker=mk, ms=3.2, color=c, ls=ls, lw=0.9, label=NAMES[m],
                    mfc='white' if m not in ('audit', 'oracle') else c, zorder=3)
    axs[0].axhline(0.10, color='0.5', lw=0.6, ls='--'); axs[1].axhline(0.05, color='0.5', lw=0.6, ls='--')
    for ax in axs:
        ax.axvspan(6.5, 12.5, facecolor='0.93', lw=0, zorder=0)
    axs[0].set_ylabel('MISS'); axs[1].set_ylabel('FOR'); axs[2].set_ylabel('referral rate')
    axs[2].set_xlabel('deployment period $t$'); axs[2].set_xticks(range(1, 13))
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, fontsize=6.2, columnspacing=0.9, handletextpad=0.3)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.2, hspace=0.12)
    fig.savefig('fig4_trajectory.png'); plt.close(fig)




def fig_bounds():
    import simulate as S, audit_derm as ad
    D = S.Data(); df = D.df
    rec = {}
    orig = ad.split_certify

    def spy(z, B, uR, uB, delta, margin=0.0):
        R = ~B
        ur = ad.betting_ucb(z * R[:, None], uR, delta / 2, R)
        ub = ad.betting_ucb(z * B[:, None], uB, delta / 2, B) if B.any() else np.zeros(z.shape[1])
        rec.setdefault(spy.t, []).append(dict(ur=ur, ub=ub, est=z.mean(0), nB=int(B.sum())))
        return orig(z, B, uR, uB, delta, margin)
    spy.t = -1
    ad.split_certify = spy
    # capture taus used and period index by wrapping risk_increments
    orig_ri = ad.risk_increments
    taus_seen = {}

    def ri(u, taus, psi, af, am, margin=0.0):
        spy.t += 1
        taus_seen[spy.t] = taus.copy()
        return orig_ri(u, taus, psi, af, am, margin)
    ad.risk_increments = ri
    S.run_stream(D, 'HIBA', 'comp', 3, ['audit'])
    ad.split_certify = orig; ad.risk_increments = orig_ri
    # pick the latest certification whose window contained audited auto-benign rows
    key = max(k for k, v in rec.items() if v[0]['nB'] > 0)
    taus = taus_seen[key]
    idx, w = S.period_weights(D, 'HIBA', 'comp', 0, 12)
    sc, y = df.ut.values[idx], df.y.values[idx]
    share = np.array([(w * (sc < t)).sum() for t in taus])
    trueF = np.array([(w * (sc < t) * (y - 0.05)).sum() for t in taus])
    trueM = np.array([(w * y * ((sc < t) - 0.10)).sum() for t in taus])
    fig, axs = plt.subplots(2, 1, figsize=(COL1, 4.6), sharey=False)
    for ax, (lab, j, tru) in zip(axs, [('false-omission constraint', 0, trueF), ('missed-malignancy constraint', 1, trueM)]):
        r = rec[key][j]
        tot = r['ur'] + r['ub']
        ok = tot < 0
        kmax = 0
        while kmax < len(ok) and ok[kmax]:
            kmax += 1
        ax.axvspan(share[0] - 0.01, share[kmax - 1] if kmax else share[0] - 0.01, color='0.9', lw=0, zorder=0)
        ax.plot(share, tru, color='0.0', lw=1.1, ls='-', marker='o', ms=2.5, label='true linear risk (population)')
        ax.plot(share, r['est'], color='0.45', lw=0.9, ls=':', marker='x', ms=3, label='window estimate')
        ax.plot(share, r['ur'], color='0.55', lw=0.9, ls='--', label='$\\bar U_{\\mathcal{R}}$ (referred rows)')
        ax.plot(share, r['ub'], color='0.55', lw=0.9, ls='-.', label='$\\bar U_{\\mathcal{B}}$ (auto-benign rows)')
        ax.plot(share, tot, color='0.0', lw=1.3, ls='--', label='$\\bar U_{\\mathcal{R}}+\\bar U_{\\mathcal{B}}$ (decision)')
        ax.axhline(0, color='0.3', lw=0.6)
        ax.set_xlabel('auto-benign share of candidate threshold')
        ax.set_title('(%s) %s' % ('ab'[j], lab), fontsize=8, loc='left')
    for ax in axs: ax.set_ylabel('linearised risk')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, fontsize=6.0, columnspacing=0.8, handlelength=2.0)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.95, bottom=0.2, hspace=0.45)
    fig.savefig('fig5_bounds.png'); plt.close(fig)
    json.dump(dict(period=int(key), nB=rec[key][0]['nB'], certified=int(kmax), share_cert=float(share[kmax - 1]) if kmax else 0.0),
              open('fig5_info.json', 'w'))


def fig_subgroups():
    r = json.load(open('E9_subgroups.json'))
    gs = [g for g in r['audit'] if r['audit'][g]['MISS'] is not None]
    fig, axs = plt.subplots(1, 2, figsize=(COL1, 2.7), sharey=True)
    yy = np.arange(len(gs))[::-1]
    for ax, col, lim in zip(axs, ['FOR', 'MISS'], [0.05, 0.10]):
        for y, g in zip(yy, gs):
            a = r['audit'][g]; o = r['oracle'][g]
            ax.plot([a[col], a[col + '_p95']], [y, y], color='0.3', lw=1.0, zorder=2)
            ax.scatter(a[col], y, s=16, color='0.0', zorder=3, label='AUDIT-Derm mean' if y == yy[0] else None)
            ax.scatter(a[col + '_p95'], y, s=16, marker='|', color='0.0', zorder=3, label='AUDIT-Derm 95th pct.' if y == yy[0] else None)
            ax.scatter(o[col], y, s=16, facecolor='white', edgecolor='0.4', marker='D', zorder=3, label='Oracle mean' if y == yy[0] else None)
        ax.axvline(lim, color='0.5', ls='--', lw=0.7)
        ax.set_xlabel(col + ' within subgroup')
        ax.set_xlim(0, 0.13 if col == 'FOR' else 0.22)
        ax.set_xticks([0, 0.05, 0.10] if col == 'FOR' else [0, 0.05, 0.10, 0.15, 0.20])
    axs[0].set_yticks(yy); axs[0].set_yticklabels(gs, fontsize=6.6)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=6.0, columnspacing=0.8, handletextpad=0.3)
    fig.subplots_adjust(left=0.25, right=0.98, top=0.97, bottom=0.27, wspace=0.12)
    fig.savefig('fig7_subgroups.png'); plt.close(fig)


def fig_budget():
    import simulate as S
    E3 = pd.read_csv('E3_sensitivity.csv')
    E1 = pd.read_csv('E1_main.csv')
    base = E1[(E1.scen == 'comp') & (E1.method == 'audit') & (E1.seed < 10)].copy()
    base['x_budget'] = S.CFG['budget']
    fig, axs = plt.subplots(2, 1, figsize=(COL1, 5.4), gridspec_kw=dict(hspace=0.75))
    a = axs[0]
    b3 = E3[E3.x_budget.notna() & E3.x_alpha_f.isna()] if 'x_alpha_f' in E3 else E3[E3.x_budget.notna()]
    for m, lab, mk, ls in (('audit', 'AUDIT-Derm (adaptive, AIPW)', 'o', '-'), ('audit_uniform', 'uniform allocation', 's', '--'),
                           ('audit_noAIPW', 'adaptive, IPW only', '^', ':')):
        x = b3[b3.method == m]
        if m == 'audit':
            x = pd.concat([x, base])
        n = x['x_n'].fillna(S.CFG['n']) if 'x_n' in x else S.CFG['n']
        x = x.assign(work=x.REF + x.audits / n)
        g = x.groupby('x_budget')
        a.plot(g.REF.mean().index, g.REF.mean().values, marker=mk, ms=3.5, color='0.0', ls=ls, lw=0.9, mfc='white', label=lab + ', referral')
        a.plot(g.work.mean().index, g.work.mean().values, marker=mk, ms=3.5, color='0.55', ls=ls, lw=0.9, label=lab + ', workload $\\Omega$')
    a.set_xlabel('audit budget $\\beta$ (floor $\\pi_{\\min}=\\beta/2$)'); a.set_ylabel('rate per case')
    a.set_title('(a) Referral and workload (compound shift)', fontsize=8, loc='left')
    a.legend(fontsize=5.4, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, columnspacing=0.6, handlelength=2.2)
    b = axs[1]
    rows = []
    def add(df, n, L, beta, pmin, am):
        for site, g in df.groupby('site'):
            rows.append(dict(site=site, idx=n * L * 0.10 * am * pmin, auto=1 - g.susp.mean()))
    add(base, S.CFG['n'], S.CFG['window'], S.CFG['budget'], S.CFG['pi_min'], S.CFG['alpha_m'])
    import os
    if os.path.exists('E3b_volume.csv'):
        E3b = pd.read_csv('E3b_volume.csv')
        for bb in sorted(E3b.x_budget.unique()):
            add(E3b[E3b.x_budget == bb], S.CFG['n'], S.CFG['window'], bb, bb / 2, S.CFG['alpha_m'])
    for bb in sorted(b3.x_budget.dropna().unique()):
        add(b3[(b3.method == 'audit') & (b3.x_budget == bb)], S.CFG['n'], S.CFG['window'], bb, bb / 2, S.CFG['alpha_m'])
    if 'x_n' in E3:
        for nn in sorted(E3.x_n.dropna().unique()):
            add(E3[(E3.method == 'audit') & (E3.x_n == nn)], nn, S.CFG['window'], S.CFG['budget'], S.CFG['pi_min'], S.CFG['alpha_m'])
    if 'x_window' in E3:
        for L in sorted(E3.x_window.dropna().unique()):
            add(E3[(E3.method == 'audit') & (E3.x_window == L)], S.CFG['n'], L, S.CFG['budget'], S.CFG['pi_min'], S.CFG['alpha_m'])
    if 'x_alpha_m' in E3:
        for am in sorted(E3.x_alpha_m.dropna().unique()):
            add(E3[(E3.method == 'audit') & (E3.x_alpha_m == am)], S.CFG['n'], S.CFG['window'], S.CFG['budget'], S.CFG['pi_min'], am)
    r = pd.DataFrame(rows)
    mks = {'HIBA': 'o', 'PAD': 's', 'MILK': '^', 'MSKCC': 'D'}
    dodge = {'HIBA': 0.94, 'MILK': 0.98, 'PAD': 1.02, 'MSKCC': 1.06}
    for site, g in r.groupby('site'):
        b.scatter(g.idx * dodge[site], g.auto, marker=mks[site], s=16, facecolor='white' if site != 'HIBA' else '0.2', edgecolor='0.1', lw=0.6, label=site)
    b.axvline(np.log(2 / 0.1) / 0.95, color='0.4', ls='--', lw=0.8)
    b.text(np.log(2 / 0.1) / 0.95 * 1.06, 0.93, 'Prop. 2 bound', fontsize=6.2, color='0.3')
    b.set_xscale('log'); b.set_xticks([1, 2, 3, 5, 10, 20]); b.set_xticklabels(['1', '2', '3', '5', '10', '20']); b.minorticks_off()
    b.set_xlabel('volume index $N\\,\\theta\\,\\alpha_M\\,\\pi_{\\min}$'); b.set_ylabel('share of non-suspended periods')
    b.set_ylim(-0.03, 1.03)
    b.set_title('(b) Automation against the volume condition', fontsize=8, loc='left')
    b.legend(fontsize=6.0, loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=4)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.96, bottom=0.12)
    fig.savefig('fig6_budget.png'); plt.close(fig)
    r.to_csv('fig6_volume.csv', index=False)



def fig_dynamics():
    E1 = pd.read_csv('E1_main.csv')
    a = E1[E1.method == 'audit']
    rows = [('HIBA', 'acq'), ('HIBA', 'comp'), ('HIBA', 'drift'), ('PAD', 'acq'), ('PAD', 'comp'), ('PAD', 'drift'),
            ('MILK', 'acq'), ('MILK', 'comp'), ('MILK', 'drift'), ('MSKCC', 'acq'), ('MSKCC', 'comp'), ('MSKCC', 'drift'),
            ('SRC', 'prev_lo'), ('SRC', 'prev_hi')]
    lab = {'acq': 'acq.', 'comp': 'comp.', 'drift': 'drift', 'prev_lo': 'prev 0.10', 'prev_hi': 'prev 0.40'}
    site = {'SRC': 'Internal', 'MILK': 'MILK10k', 'PAD': 'PAD', 'HIBA': 'HIBA', 'MSKCC': 'MSKCC'}
    M = np.array([[1 - a[(a.site == s) & (a.scen == c) & (a.t == t)].susp.mean() for t in range(12)] for s, c in rows])
    V = np.array([[a[(a.site == s) & (a.scen == c) & (a.t == t)].viol.mean() for t in range(12)] for s, c in rows])
    fig, ax = plt.subplots(figsize=(COL1, 3.3))
    im = ax.imshow(M, cmap='Greys', vmin=0, vmax=1, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(12):
            if V[i, j] > 0:
                ax.text(j, i, '%d' % round(100 * V[i, j]), ha='center', va='center', fontsize=5.4,
                        color='white' if M[i, j] > 0.55 else 'black')
    ax.set_xticks(range(12)); ax.set_xticklabels(range(1, 13), fontsize=6.5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(['%s, %s' % (site[s], lab[c]) for s, c in rows], fontsize=6.3)
    ax.set_xlabel('deployment period $t$')
    for k in (2.5, 5.5, 8.5, 11.5):
        ax.axhline(k, color='0.5', lw=0.6)
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label('share of streams automating', fontsize=6.5); cb.ax.tick_params(labelsize=6)
    fig.subplots_adjust(left=0.25, right=0.88, top=0.97, bottom=0.13)
    fig.savefig('fig8_dynamics.png'); plt.close(fig)



if __name__ == '__main__':
    for f in sys.argv[1:]:
        globals()['fig_' + f]()
        print('done', f)
