"""Tiny box-and-arrow layout helper with automatic overlap and routing checks (grayscale)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix'})


class Diagram:
    def __init__(self, w, h):
        self.W, self.H = w, h
        self.fig = plt.figure(figsize=(w, h))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w); self.ax.set_ylim(0, h); self.ax.axis('off')
        self.boxes = {}
        self.arrows = []
        self.lanes = []
        self.texts = []

    def lane(self, x, y, w, h, label, right=False, tx=None):
        self.ax.add_patch(Rectangle((x, y), w, h, fc='0.94', ec='0.55', lw=0.6, ls=(0, (4, 2)), zorder=0))
        t = self.ax.text(tx if tx is not None else (x + w - 0.06 if right else x + 0.06), y + h - 0.05, label, ha='right' if right else 'left', va='top', fontsize=7.2, style='italic', color='0.15', zorder=1)
        self.lanes.append((x, y, w, h)); self.texts.append(t)

    def box(self, name, x, y, w, h, text, kind='plain', fs=6.6):
        fc = {'plain': 'white', 'novel': '0.82', 'data': '0.97'}[kind]
        lw = 1.3 if kind == 'novel' else 0.8
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.04',
                                         fc=fc, ec='black', lw=lw, zorder=3))
        t = self.ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, zorder=4,
                         linespacing=1.15, fontweight='bold' if kind == 'novel' else 'normal')
        self.boxes[name] = (x, y, w, h, t)

    def anchor(self, name, side, frac=0.5):
        x, y, w, h, _ = self.boxes[name]
        return {'l': (x, y + h * frac), 'r': (x + w, y + h * frac),
                't': (x + w * frac, y + h), 'b': (x + w * frac, y)}[side]

    def arrow(self, a, sa, b, sb, via=(), style='solid', label=None, lpos=None, fa=0.5, fb=0.5, lfs=6.0):
        p0 = self.anchor(a, sa, fa); p1 = self.anchor(b, sb, fb)
        pts = [p0] + list(via) + [p1]
        ls = {'solid': '-', 'dashed': (0, (4, 2.2)), 'dotted': (0, (1, 1.6))}[style]
        for i in range(len(pts) - 1):
            self.ax.add_line(Line2D([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]], color='black', lw=0.9, ls=ls, zorder=2))
        (xa, ya), (xb, yb) = pts[-2], pts[-1]
        L = ((xb - xa) ** 2 + (yb - ya) ** 2) ** .5
        k = min(1.0, 0.09 / max(L, 1e-9))
        self.ax.annotate('', xy=(xb, yb), xytext=(xb - (xb - xa) * k, yb - (yb - ya) * k), zorder=2,
                         arrowprops=dict(arrowstyle='-|>,head_length=0.45,head_width=0.22', color='black', lw=0.9,
                                         shrinkA=0, shrinkB=0))
        if label:
            t = self.ax.text(lpos[0], lpos[1], label, fontsize=lfs, ha='center', va='center', zorder=5,
                             bbox=dict(fc='white', ec='none', pad=0.6))
            self.texts.append(t)
        self.arrows.append((a, b, pts, style, t if label else None))

    # ------------------------------------------------------------------ checks
    def check(self, margin=0.05):
        errs = []
        names = list(self.boxes)
        for i, a in enumerate(names):
            ax, ay, aw, ah, _ = self.boxes[a]
            for b in names[i + 1:]:
                bx, by, bw, bh, _ = self.boxes[b]
                if ax < bx + bw + margin and bx < ax + aw + margin and ay < by + bh + margin and by < ay + ah + margin:
                    errs.append('box overlap/too close: %s %s' % (a, b))
        self.fig.canvas.draw()
        r = self.fig.canvas.get_renderer()
        inv = self.ax.transData.inverted()
        for n, (x, y, w, h, t) in self.boxes.items():
            bb = t.get_window_extent(r).transformed(inv)
            if bb.x0 < x + 0.02 or bb.x1 > x + w - 0.02 or bb.y0 < y + 0.01 or bb.y1 > y + h - 0.01:
                errs.append('text overflows box: %s' % n)
        # arrow segments must not pass through any box other than their own endpoints
        for a, b, pts, _, _l in self.arrows:
            for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
                for n, (x, y, w, h, _) in self.boxes.items():
                    if n in (a, b):
                        continue
                    if seg_hits_rect(x0, y0, x1, y1, x - 0.02, y - 0.02, w + 0.04, h + 0.04):
                        errs.append('arrow %s->%s crosses box %s' % (a, b, n))
        # free labels must not overlap boxes or each other
        tb = [t.get_window_extent(r).transformed(inv) for t in self.texts]
        lane_titles = set(id(t) for t in self.texts[:len(self.lanes)])
        for i, bb in enumerate(tb):
            for n, (x, y, w, h, _) in self.boxes.items():
                if bb.x0 < x + w and x < bb.x1 and bb.y0 < y + h and y < bb.y1:
                    errs.append('label "%s" overlaps box %s' % (self.texts[i].get_text()[:20], n))
            for j in range(i + 1, len(tb)):
                cc = tb[j]
                if bb.x0 < cc.x1 and cc.x0 < bb.x1 and bb.y0 < cc.y1 and cc.y0 < bb.y1:
                    errs.append('labels overlap: "%s" / "%s"' % (self.texts[i].get_text()[:15], self.texts[j].get_text()[:15]))
        # arrows must not run through any free text (lane titles, legends, other arrows' labels)
        for a, b, pts, _, own in self.arrows:
            for t in self.texts:
                if t is own:
                    continue
                bb = t.get_window_extent(r).transformed(inv)
                for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
                    if seg_hits_rect(x0, y0, x1, y1, bb.x0, bb.y0, bb.width, bb.height):
                        errs.append('arrow %s->%s crosses text "%s"' % (a, b, t.get_text()[:25]))
        # everything inside the canvas
        for t in self.texts + [v[4] for v in self.boxes.values()]:
            bb = t.get_window_extent(r).transformed(inv)
            if bb.x0 < 0.01 or bb.x1 > self.W - 0.01 or bb.y0 < 0.01 or bb.y1 > self.H - 0.01:
                errs.append('text outside canvas: "%s"' % t.get_text()[:25])
        return errs

    def save(self, path, dpi=600):
        errs = self.check()
        if errs:
            raise RuntimeError('\n'.join(errs))
        self.fig.savefig(path, dpi=dpi)
        plt.close(self.fig)


def seg_hits_rect(x0, y0, x1, y1, rx, ry, rw, rh, steps=200):
    for k in range(steps + 1):
        t = k / steps
        x = x0 + (x1 - x0) * t; y = y0 + (y1 - y0) * t
        if rx < x < rx + rw and ry < y < ry + rh:
            return True
    return False
