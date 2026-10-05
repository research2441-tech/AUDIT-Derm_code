"""Fig. 1: AUDIT-Derm architecture (grayscale, overlap- and routing-checked)."""
from diagram_lib import Diagram
from matplotlib.lines import Line2D

D = Diagram(7.1, 4.45)
# lanes
D.lane(0.05, 2.85, 7.0, 1.55, 'A. Development (offline, before deployment)')
D.lane(0.05, 1.66, 7.0, 1.07, 'B. Deployment period $t$ (frozen)')
D.lane(0.05, 0.36, 7.0, 1.20, 'C. Between periods: audit-based certification and policy update', tx=0.85)

# lane A, row 1 (forward training path)
y1, h = 3.48, 0.44
D.box('src', 0.15, y1, 1.30, h, 'Source dermoscopic\nimages (HAM, BCN)', 'data')
D.box('enc', 1.75, y1, 1.20, h, 'Frozen CLIP\nViT-B/32 encoder')
D.box('head', 3.25, y1, 1.20, h, 'Logistic head\n$u=w^{\\top}z+b$')
D.box('loss', 4.75, y1, 1.10, h, 'Cross-entropy\nloss (train split)')
# lane A, row 2
y2 = 2.95
D.box('pairs', 0.15, y2, 1.30, h, 'Paired lesions\n(clinical + dermoscopic)', 'data')
D.box('gc', 1.75, y2, 1.60, h, 'Paired acquisition\ntransport $g_c$', 'novel')
D.box('cal', 3.65, y2, 1.15, h, 'Calibration split\n(malignant scores)', 'data')
D.box('cand', 5.10, y2, 1.85, h, 'Mondrian conformal candidates\n$\\tau_1<\\tau_2<\\cdots<\\tau_K$')

# lane B
yb = 1.98
D.box('case', 0.15, yb, 1.05, h, 'Incoming case\n$x_i$, condition $c_i$', 'data')
D.box('score', 1.45, yb, 1.05, h, 'Frozen encoder\n+ head: $u_i$')
D.box('trans', 2.75, yb, 1.20, h, 'Transport\n$\\tilde u_i=g_{c_i}(u_i)$')
D.box('dec', 4.20, yb, 1.30, h, 'Auto-benign if $\\tilde u_i<\\tau_t$,\notherwise refer')
D.box('samp', 5.80, yb, 1.15, h, 'Audit sampler\n$\\pi_i$ recorded', 'novel')

# lane C
yc = 0.88
D.box('out', 5.80, yc, 1.15, h, 'Delayed outcome\n$Y_i$ if $O_i=1$', 'data')
D.box('win', 4.55, yc, 1.00, h, 'Completed\nwindow $\\mathcal{W}_t$')
D.box('aipw', 3.00, yc, 1.30, h, 'Model-assisted\nincrements $\\psi_i$', 'novel')
D.box('cert', 1.30, yc, 1.45, h, 'Betting IUT test +\nfixed-sequence scan', 'novel')
D.box('ctrl', 0.15, yc, 0.92, h, 'Controller:\n$\\tau_{t+1}$ or suspend', 'novel')
D.box('m', 3.00, 0.42, 1.30, 0.34, 'Working model $m(x)$', fs=6.4)

# forward arrows (solid)
D.arrow('src', 'r', 'enc', 'l')
D.arrow('enc', 'r', 'head', 'l')
D.arrow('head', 'r', 'loss', 'l')
D.arrow('pairs', 'r', 'gc', 'l')
D.arrow('cal', 'r', 'cand', 'l')
D.arrow('case', 'r', 'score', 'l')
D.arrow('score', 'r', 'trans', 'l')
D.arrow('trans', 'r', 'dec', 'l')
D.arrow('dec', 'r', 'samp', 'l')
D.arrow('gc', 'b', 'trans', 't', fa=0.75, fb=(1.75 + 1.6 * 0.75 - 2.75) / 1.20)
D.arrow('cand', 'b', 'dec', 't', fa=(5.30 - 5.10) / 1.85, fb=(5.30 - 4.20) / 1.30)
D.arrow('samp', 'b', 'out', 't')
D.arrow('out', 'l', 'win', 'r')
D.arrow('win', 'l', 'aipw', 'r')
D.arrow('aipw', 'l', 'cert', 'r')
D.arrow('cert', 'l', 'ctrl', 'r')
D.arrow('m', 't', 'aipw', 'b')
# reverse-mode gradient (dashed): loss -> head, routed above row 1
xg0 = 4.75 + 1.10 * 0.5; xg1 = 3.25 + 1.20 * 0.5
D.arrow('loss', 't', 'head', 't', via=[(xg0, 4.12), (xg1, 4.12)], style='dashed',
        label='reverse-mode gradient $\\nabla_{w,b}\\mathcal{L}$ (development only)', lpos=(4.05, 4.24))
# delayed feedback (dotted): controller -> decision of the next period
D.arrow('ctrl', 't', 'dec', 'b', via=[(0.61, 1.80), (4.85, 1.80)], style='dotted', fb=0.5,
        label='policy for period $t+1$', lpos=(2.55, 1.80))
# outcomes refresh the working model (dotted)
D.arrow('win', 'b', 'm', 'r', via=[(5.05, 0.59)], style='dotted')

# legend (outside the lanes, bottom strip)
ax = D.ax
items = [('-', 'data / decision flow'), ((0, (4, 2.2)), 'reverse-mode gradient (offline)'),
         ((0, (1, 1.6)), 'delayed feedback between periods')]
xs = [0.10, 1.62, 3.62]
for (ls, txt), x in zip(items, xs):
    ax.add_line(Line2D([x, x + 0.38], [0.16, 0.16], color='black', lw=0.9, ls=ls))
    t = ax.text(x + 0.46, 0.16, txt, fontsize=6.4, va='center', ha='left')
    D.texts.append(t)
ax.add_patch(__import__('matplotlib.patches', fromlist=['Rectangle']).Rectangle((5.80, 0.10), 0.16, 0.12, fc='0.82', ec='black', lw=1.2))
t = ax.text(6.02, 0.16, 'proposed component', fontsize=6.4, va='center', ha='left'); D.texts.append(t)
D.save('fig1_arch.png')
print('ok')
