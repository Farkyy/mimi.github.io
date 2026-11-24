#!/usr/bin/env python3
"""
Version « gravure / tatouage fineline » : variante B (contours mesurés sur la photo)
enrichie d'ornements dessinés d'après les photos de référence (dossier references/) :
  - 4 chevaux de manège (crinière ondulée, bride à rosettes, plastron, tapis de selle
    à pierreries et languettes, jambes avant levées) sur des barres torsadées,
    placés sur de vraies barres détectées dans la photo
  - médaillons : double cadre, petit paysage peint au trait, volutes en C, coquille
  - crêtes du bandeau en feuilles d'acanthe
  - petites lampes sous chaque perle du feston
  - motif losange + rond dans chaque travée de la barrière
Traits noirs, épaisseur uniforme ; les formes blanches servent uniquement à masquer
les traits situés derrière (aucun remplissage visible).

Sorties : modeles/manege_grave.svg / .png  et  modeles/manege_grave_vs_B.png
"""
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

import contours_intermediaires as C
import dessin_cheval as D
import epurer_contours as E

S, W, H = E.S, E.W, E.H
OUT = E.OUT
HERE = Path(__file__).resolve().parent
STROKE = 3.0
# épaisseur uniforme pour tout le dessin des chevaux
D.W_MAIN = D.W_MID = D.W_FINE = D.W_HAIR = STROKE
AX = E.APEX[0]


def cp(x, y):
    """Coordonnées photo recadrée (816 x 925) -> coordonnées de travail (x3)."""
    return (x * S, y * S)


# ---------------------------------------------------------------------------
# 1. Variante B (mêmes réglages que contours_intermediaires.py)
# ---------------------------------------------------------------------------
gray = cv2.cvtColor(cv2.bilateralFilter(cv2.resize(E.photo, (W, H), interpolation=cv2.INTER_LANCZOS4), 9, 40, 9),
                    cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
g_canny = cv2.GaussianBlur(cv2.bilateralFilter(clahe.apply(gray), 11, 50, 11), (0, 0), 1.6)
cache = {}
b_lines, y_bottom = C.build(C.VARIANTS["B"], g_canny, E.smooth_gray(1.5, sp=4, sr=18), E.smooth_gray(2.5),
                            E.smooth_gray(2.0, sp=5, sr=26), cache)
geo = cache["geo"]

# chevaux : barres choisies (positions réelles), orientation, échelle, décalage vertical (crop px)
HORSE_BARS = [  # (x approx. de la barre, sens, échelle, décalage)
    (220, -1, 1.75, -72),
    (305, -1, 1.95, 36),
    (476, 1, 1.95, -10),
    (625, 1, 1.75, 30),
]
bars = geo["bars"]
chosen = []
for x0, facing, sc, dy in HORSE_BARS:
    bx = min(bars, key=lambda b: abs(b - x0 * S))
    chosen.append((bx, facing, sc, dy))
horse_xs = [c[0] for c in chosen]


def keep_b_line(l):
    m = l.mean(0)
    # fragments de B dans les zones des chevaux : remplacés par les chevaux dessinés
    if len(l) > 2 and any(C.in_box(m, b) for b in C.HORSES):
        return False
    # sous le toit : on ne garde que les barres et l'escalier (traits droits à 2 points)
    if 522 * S < m[1] < 765 * S and len(l) > 2 and C.length(l) < 0.15 * H:
        return False
    # bandeau : les petits fragments laissent la place aux ornements dessinés
    if 290 * S < m[1] < 400 * S and C.length(l) < 0.05 * H and len(l) > 2:
        return False
    # barres qui portent un cheval : remplacées par des barres torsadées
    if len(l) == 2 and abs(l[0, 0] - l[1, 0]) < 1 and any(abs(l[0, 0] - hx) < 3 for hx in horse_xs):
        return False
    return True


b_lines = [l for l in b_lines if keep_b_line(l)]

# ---------------------------------------------------------------------------
# 2. Ornements
# ---------------------------------------------------------------------------
svg = D.Svg(W, H)
P = D.pt
cat = D.catmull


def path(points, closed=False, mask=False):
    d = cat(points, closed)
    if mask:
        svg.path(d + ("" if closed else " Z"), STROKE, fill="#fff")
    else:
        svg.path(d, STROKE)


def poly(points):
    svg.path("M" + " L".join(P(p) for p in points), STROKE)


def ell_pts(x, y, rx, ry, a0=0, a1=2 * math.pi, n=60):
    return [(x + rx * math.cos(t), y + ry * math.sin(t)) for t in np.linspace(a0, a1, n)]


# --- 2a. Médaillons : double cadre + motif peint au trait + volutes + coquille
meds = sorted(geo["meds"], key=lambda m: m[0][0])
big = [m for m in meds if m[0][2] > 25 * S]
small = [m for m in meds if m[0][2] <= 25 * S]


def scroll(cx, cy, r, side, up=True):
    """Volute en C (rocaille) : courbe qui part du cadre et s'enroule en spirale."""
    pts = []
    turns = 1.35
    for t in np.linspace(0, turns * 2 * math.pi, 40):
        rr = r * (1 - t / (turns * 2 * math.pi) * 0.82)
        a = (math.pi if side < 0 else 0) + (-1 if up else 1) * side * t
        pts.append((cx + rr * math.cos(a) * side * -1 + side * r, cy + rr * math.sin(a)))
    return pts


def shell(x, y_base, w, h):
    """Coquille : éventail à côtes et bord festonné."""
    n = 7
    top = []
    for i in range(n + 1):
        a = math.pi * (1 - i / n)
        top.append((x + w / 2 * math.cos(a), y_base - h * math.sin(a)))
    outline = [(x - w * 0.18, y_base + h * 0.12)] + top + [(x + w * 0.18, y_base + h * 0.12)]
    path(outline + [(x, y_base + h * 0.3)], closed=True, mask=True)
    for i in range(1, n):
        p = top[i]
        poly([(x, y_base + h * 0.15), (p[0] * 0.85 + x * 0.15, p[1] * 0.85 + y_base * 0.15)])


def landscape(kind, x, y, rx, ry):
    """Petit paysage au trait dans l'ovale intérieur (d'après les médaillons peints, ref5)."""
    u = lambda px, py: (x + px * rx, y + py * ry)
    hz = 0.32
    poly([u(-0.82, hz), u(0.82, hz)])                                     # horizon / sol
    if kind == 0:     # chalet + sapin
        poly([u(-0.45, hz), u(-0.45, -0.05), u(-0.12, -0.38), u(0.22, -0.05), u(0.22, hz)])
        poly([u(-0.45, -0.05), u(0.22, -0.05)])
        poly([u(-0.22, hz), u(-0.22, 0.08), u(-0.04, 0.08), u(-0.04, hz)])           # porte
        poly([u(0.5, hz), u(0.5, 0.18)])
        poly([u(0.32, 0.18), u(0.5, -0.5), u(0.68, 0.18), u(0.32, 0.18)])            # sapin
        poly([u(0.38, -0.08), u(0.5, -0.5), u(0.62, -0.08)])
    elif kind == 1:   # église et clocher
        poly([u(-0.35, hz), u(-0.35, -0.02), u(0.25, -0.02), u(0.25, hz)])
        poly([u(-0.35, -0.02), u(-0.05, -0.22), u(0.25, -0.02)])
        poly([u(0.25, hz), u(0.25, -0.3), u(0.45, -0.3), u(0.45, hz)])
        poly([u(0.25, -0.3), u(0.35, -0.72), u(0.45, -0.3)])                          # flèche
        poly([u(0.35, -0.72), u(0.35, -0.82)])
        poly([u(-0.66, hz), u(-0.66, 0.0)])
        path([u(-0.8, 0.0), u(-0.66, -0.35), u(-0.52, 0.0)])                           # arbre
    else:             # paysage : collines + arbres
        path([u(-0.82, 0.12), u(-0.4, -0.12), u(0.0, 0.05), u(0.42, -0.2), u(0.82, 0.05)])
        for px in (-0.45, 0.35):
            poly([u(px, hz), u(px, 0.05)])
            path([u(px - 0.14, 0.05), u(px, -0.32), u(px + 0.14, 0.05)])
        path([u(-0.7, -0.5), u(-0.6, -0.56), u(-0.48, -0.5)])                          # oiseau


def rosette(x, y, r):
    for k in range(6):
        a = k * math.pi / 3
        px, py = x + r * 0.55 * math.cos(a), y + r * 0.55 * math.sin(a)
        svg.path(cat(ell_pts(px, py, r * 0.32, r * 0.32, n=16), True), STROKE)
    svg.circle((x, y), r * 0.18, STROKE)


for i, ((x, y, rx, ry), s) in enumerate(big):
    s = float(np.clip(s, 0.74, 0.84))
    # volutes de part et d'autre (si la place le permet)
    for side in (-1, 1):
        sx = x + side * rx
        if all(abs(sx + side * 0.3 * rx - m[0][0]) > m[0][2] + 0.3 * rx for m in small):
            path(scroll(sx, y - ry * 0.15, ry * 0.42, side, up=True))
            path(scroll(sx, y + ry * 0.45, ry * 0.3, side, up=False))
    # coquille au sommet
    shell(x, y - ry * 0.98, rx * 0.5, ry * 0.5)
    # cadre intérieur supplémentaire (double contour) + motif
    svg.path(cat(ell_pts(x, y, rx * (s + 0.07), ry * (s + 0.07)), True), STROKE)
    landscape(i % 3, x, y, rx * s * 0.92, ry * s * 0.92)
for (x, y, rx, ry), s in small:
    rosette(x, y, min(rx, ry) * 0.62)

# --- 2b. Crêtes du bandeau : feuilles d'acanthe (positions des crêtes sur la photo)
CRESTS = [(140, 338, 1.0), (290, 334, 1.0), (485, 334, 1.0), (645, 336, 1.0), (70, 350, 0.7), (730, 350, 0.7)]


def acanthus(x, yb, k):
    h, w = 40 * S * k, 9 * S * k
    L = [(x - w, yb), (x - w * 1.1, yb - h * 0.25), (x - w * 0.55, yb - h * 0.35), (x - w * 0.95, yb - h * 0.55),
         (x - w * 0.35, yb - h * 0.68), (x - w * 0.5, yb - h * 0.85), (x, yb - h)]
    R = [(2 * x - px, py) for px, py in L[::-1]]
    path(L + R[1:], closed=True, mask=True)
    path([(x, yb - h * 0.05), (x + w * 0.1, yb - h * 0.45), (x, yb - h * 0.9)])        # nervure
    path([(x - w * 0.15, yb - h * 0.18), (x - w * 0.55, yb - h * 0.3)])
    path([(x + w * 0.15, yb - h * 0.18), (x + w * 0.55, yb - h * 0.3)])
    svg.circle((x, yb - h - 2.6 * S * k), 2.4 * S * k, STROKE, fill="#fff")             # perle


for x, yb, k in CRESTS:
    acanthus(x * S, yb * S, k)

# --- 2c. Lampes sous chaque perle du feston
for x, y in geo["nodes"]:
    svg.path(f"M{P((x, y))} L{P((x, y + 2.2 * S))}", STROKE)
    svg.circle((x, y + 4.6 * S), 2.4 * S, STROKE, fill="#fff")

# --- 2d. Barrière : motif losange + rond dans chaque travée
af, bf, fcy, fh = geo["af"], geo["bf"], geo["fence_cy"], geo["fence_h"]


def fence_y(x, k):  # k = 0 pied, 1 main courante
    return fcy + bf * math.sqrt(max(0.0, 1 - ((x - AX) / af) ** 2)) - k * fh


post_x = np.linspace(AX - af + 3 * S, AX + af - 3 * S, 15)
for x in post_x[1:-1]:      # poteaux réguliers (en plus de ceux de B) avec boule
    poly([(x, fence_y(x, 0)), (x, fence_y(x, 1))])
    svg.circle((x, fence_y(x, 1) - 2.6 * S), 2.2 * S, STROKE, fill="#fff")
for x0, x1 in zip(post_x[:-1], post_x[1:]):
    xm = (x0 + x1) / 2
    top, bot = fence_y(xm, 0.8), fence_y(xm, 0.4)
    mid = (top + bot) / 2
    dx = (x1 - x0) * 0.24
    poly([(xm, top), (xm + dx, mid), (xm, bot), (xm - dx, mid), (xm, top)])
    svg.circle((xm, mid), min(dx, (bot - top) / 2) * 0.38, STROKE)

# --- 2e. Chevaux sur barres torsadées (du fond vers l'avant : extérieurs d'abord)
deck_y, a_, b_ = geo["deck_y"], geo["a"], geo["b"]


def floor_y(x):
    return deck_y + b_ * 0.55 * math.sqrt(max(0.0, 1 - ((x - AX) / a_) ** 2))


for bx, facing, sc, dy in sorted(chosen, key=lambda c: -abs(c[0] - AX)):
    y_top, y_bot = 524 * S, floor_y(bx)
    oy = 702 * S + dy * S
    D.twisted_pole(svg, bx, y_top, y_bot, r=6.5, pitch=24)
    D.horse(svg, bx, oy, scale=sc, facing=facing, detail=1.0)
    D.twisted_pole(svg, bx, y_top, oy - 33 * sc, r=6.5, pitch=24)   # la barre ressort au-dessus de la selle

# ---------------------------------------------------------------------------
# 3. Assemblage : B (traits) + ornements, recadré comme B
# ---------------------------------------------------------------------------
_, fw, fh_ = C.frame(b_lines, y_bottom)
pts = np.vstack(b_lines)
half = max(AX - pts[:, 0].min(), pts[:, 0].max() - AX) + C.MARGIN
x0, y0 = AX - half, pts[:, 1].min() - C.MARGIN


def d_of(p):
    if len(p) < 3:
        return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in p)
    s = [f"M{p[0][0]:.1f},{p[0][1]:.1f}"]
    for i in range(len(p) - 1):
        p0, p1, p2, p3 = p[max(i - 1, 0)], p[i], p[i + 1], p[min(i + 2, len(p) - 1)]
        c1, c2 = p1 + (p2 - p0) / 6, p2 - (p3 - p1) / 6
        s.append(f"C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}")
    return " ".join(s)


orn = "\n    ".join(svg.items)
orn = orn.replace('fill="#000"', 'fill="none"')                 # aucun aplat noir
import re
orn = re.sub(r'stroke-width="[\d.]+"', f'stroke-width="{STROKE}"', orn)
b_body = "\n    ".join(f'<path d="{d_of(l)}"/>' for l in b_lines)
out_svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{fw}" height="{fh_}" viewBox="0 0 {fw} {fh_}" '
           f'stroke-linecap="round" stroke-linejoin="round">\n'
           f'  <rect width="100%" height="100%" fill="#fff"/>\n'
           f'  <g transform="translate({-x0:.1f},{-y0:.1f})">\n'
           f'  <g fill="none" stroke="#000" stroke-width="{STROKE}">\n    {b_body}\n  </g>\n'
           f'    {orn}\n  </g>\n</svg>\n')
(OUT / "manege_grave.svg").write_text(out_svg)
subprocess.run(["node", str(HERE / "render.js"), str(OUT / "manege_grave.svg"), str(OUT / "manege_grave.png"), "1"],
               check=True)

# comparaison côte à côte avec B
a = cv2.imread(str(OUT / "manege_inter_B.png"), 0)
b = cv2.imread(str(OUT / "manege_grave.png"), 0)
h = 1400
f = lambda im: cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h), interpolation=cv2.INTER_AREA)
sep = np.full((h, 40), 255, np.uint8)
board = np.hstack([f(a), sep, f(b)])
board = np.vstack([np.full((70, board.shape[1]), 255, np.uint8), board])
cv2.putText(board, "Variante B", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 2, cv2.LINE_AA)
cv2.putText(board, "Version gravure (enrichie)", (f(a).shape[1] + 80, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 2,
            cv2.LINE_AA)
cv2.imwrite(str(OUT / "manege_grave_vs_B.png"), board)
print("ok", fw, fh_)
