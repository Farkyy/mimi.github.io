#!/usr/bin/env python3
"""
Contours « intermédiaires » du manège : entre manege_contours.png (trop chargé)
et manege_epure_*.png (trop vide).

Base : les mêmes contours Canny que manege_contours.png, transformés en lignes
(squelette -> polylignes), filtrés par longueur (seuil bas), lissés, épaisseur uniforme.
Corrections par zone, toutes mesurées sur la photo :
  - cône / toit évasé : les rayures Canny (2 bords par rayure) sont remplacées par
    une rayure sur deux mesurée sur le profil d'intensité
  - médaillons : contour extérieur (HoughCircles) + contour intérieur (pic de gradient radial)
  - sous le toit : barres, escalier, chevaux (zones des chevaux), le reste est filtré fort
  - plateforme + barrière : ellipse vue de face, dimensionnée sur la photo
    (largeur des barres, hauteur de la barrière, courbure du toit), poteaux aux positions réelles
  - recadrage : un seul manège, centré sur son axe, fond blanc

Sorties : modeles/manege_inter_A.png/.svg (plus détaillé) ... manege_inter_C (plus épuré)
"""
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

import epurer_contours as E

S, H, W = E.S, E.H, E.W
OUT = E.OUT
photo, mask_b = E.photo, E.mask_b
AX = E.APEX[0]                                   # axe vertical du manège
X_MIN, X_MAX = 95 * S, 715 * S                   # sous le toit : on coupe la cabine / les autres manèges
Y_ROOF, Y_UNDER_END = 520 * S, 762 * S

# Zones des chevaux (coordonnées photo x0, y0, x1, y1) – repérées sur la photo
HORSES = [(100, 650, 345, 768), (555, 640, 710, 768)]
DROP = [(462, 680, 560, 775)]                    # petite voiture de manège (pas un cheval)

# ---------------------------------------------------------------------------
# Squelette -> polylignes (rapide : parcours sur tableau)
# ---------------------------------------------------------------------------
NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def trace(skel):
    sk = np.pad(skel.astype(np.uint8), 1)
    deg = np.zeros_like(sk, np.int32)
    for dy, dx in NB:
        deg += np.roll(np.roll(sk, dy, 0), dx, 1)
    deg *= sk
    visited = np.zeros_like(sk, bool)
    lines = []

    def nbrs(y, x):
        return [(y + dy, x + dx) for dy, dx in NB if sk[y + dy, x + dx]]

    def walk(y, x):
        path = [(y, x)]
        visited[y, x] = True
        while True:
            nxt = [p for p in nbrs(*path[-1]) if not visited[p]]
            if not nxt:
                break
            p = nxt[0]
            path.append(p)
            visited[p] = True
            if deg[p] != 2:
                break
        return path

    ys, xs = np.nonzero((deg != 2) & (deg > 0))         # départs : extrémités et nœuds
    for y, x in zip(ys, xs):
        visited[y, x] = True
        for q in nbrs(y, x):
            if not visited[q]:
                lines.append([(y, x)] + walk(*q))
    ys, xs = np.nonzero(sk & ~visited)                  # boucles restantes
    for y, x in zip(ys, xs):
        if not visited[y, x]:
            lines.append(walk(y, x))
    return [np.array([(x - 1, y - 1) for y, x in l], np.float32) for l in lines if len(l) > 2]


def length(l):
    return float(np.sum(np.linalg.norm(np.diff(l, axis=0), axis=1)))


def smooth(l, win=9, eps=1.4):
    return E.smooth_poly(l, win=win, eps=eps)


def in_box(p, box):
    x0, y0, x1, y1 = box
    return x0 * S <= p[0] <= x1 * S and y0 * S <= p[1] <= y1 * S


def chord_vert(l):
    d = l[-1] - l[0]
    return 90 - abs(90 - abs(np.degrees(np.arctan2(d[1], d[0]))) % 180)


# ---------------------------------------------------------------------------
# Médaillons : contour extérieur + intérieur
# ---------------------------------------------------------------------------
def medallion_params(g):
    gg = g.copy()
    gg[~E.zone(290, 400)] = 200

    def detect(aspect, rmin, rmax, p2):
        small = cv2.resize(gg, None, fx=1 / aspect / S, fy=1 / S, interpolation=cv2.INTER_AREA)
        c = cv2.HoughCircles(small, cv2.HOUGH_GRADIENT, dp=1, minDist=rmin * 1.6, param1=60, param2=p2,
                             minRadius=rmin, maxRadius=rmax)
        return [] if c is None else [(x * aspect * S, y * S, r * aspect * S, r * S) for x, y, r in c.reshape(-1, 3)]

    big = []
    for e in detect(1.35, 20, 38, 18):
        if all(np.hypot(e[0] - o[0], e[1] - o[1]) > 0.95 * (e[2] + o[2]) for o in big):
            big.append(e)
        if len(big) == 5:
            break
    big = [(x, y, rx * 0.86, ry * 0.86) for x, y, rx, ry in big]
    out = list(big)
    for x, y, rx, ry in detect(1.0, 9, 17, 16):
        if not (340 * S < y < 378 * S):
            continue
        if any(((x - bx) / (bx_r + rx + 3 * S)) ** 2 + ((y - by) / (by_r + ry + 3 * S)) ** 2 < 1.0
               for bx, by, bx_r, by_r in out):
            continue
        out.append((x, y, rx, ry))
        if len(out) >= len(big) + 5:
            break
    return out


def inner_scale(g, m):
    """Échelle du cadre intérieur : pic de gradient le long des rayons de l'ellipse."""
    x, y, rx, ry = m
    gf = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.5)
    gx = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
    t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    best, best_s = -1, 0.8
    for s in np.arange(0.55, 0.9, 0.01):
        px, py = x + s * rx * np.cos(t), y + s * ry * np.sin(t)
        nx, ny = np.cos(t) / rx, np.sin(t) / ry
        n = np.hypot(nx, ny)
        v = np.abs(E.bilinear(gx, px[None], py[None])[0] * nx / n + E.bilinear(gy, px[None], py[None])[0] * ny / n)
        sc = np.median(v)
        if sc > best:
            best, best_s = sc, s
    return best_s


def ellipse_pts(x, y, rx, ry, n=100):
    t = np.linspace(0, 2 * np.pi, n + 1)
    return np.stack([x + rx * np.cos(t), y + ry * np.sin(t)], 1).astype(np.float32)



def festoon(g_fine):
    """Guirlande du toit évasé : ligne d'attache (haut) + un arc entre chaque perle.
    Les perles sont les maxima de luminosité réguliers le long de la guirlande ;
    la profondeur des arcs est mesurée (médiane) sur la photo."""
    gray = cv2.cvtColor(cv2.resize(photo, (W, H), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gs = cv2.GaussianBlur(gray, (0, 0), 3)
    gy = cv2.Sobel(gs, cv2.CV_32F, 0, 1, ksize=5)
    base = E.robust_curve(g_fine, 425, 450, 1, 30 * S, W - 30 * S)
    if not base:
        return []
    base = base[0]
    xs = np.arange(int(base[0, 0]), int(base[-1, 0]))
    yb = np.interp(xs, base[:, 0], base[:, 1])
    best = None
    for off in np.arange(-4, 14, 1.0):                     # rangée où la périodicité des perles est la plus nette
        prof = np.array([gs[int(y + off * S), x] for x, y in zip(xs, yb)])
        p = prof - cv2.GaussianBlur(prof.reshape(1, -1), (0, 0), 25).ravel()
        sp = np.abs(np.fft.rfft(p - p.mean()))
        f = np.fft.rfftfreq(len(p))
        sel = (f > 1 / (40 * S)) & (f < 1 / (12 * S))
        i = int(np.argmax(sp * sel))
        if best is None or sp[i] > best[0]:
            best = (sp[i], off, 1 / f[i], p)
    _, off, per, p = best
    w = int(per * 0.4)
    pk = [i for i in range(1, len(p) - 1) if p[i] >= p[i - 1] and p[i] > p[i + 1] and p[i] == p[max(0, i - w):i + w].max()]
    nodes = [(xs[i], yb[i] + off * S) for i in pk]
    # profondeur : bord clair -> foncé le plus net sous le milieu de chaque arc
    depths = []
    for (x0, y0), (x1, y1) in zip(nodes, nodes[1:]):
        xm, ym = int((x0 + x1) / 2), (y0 + y1) / 2
        col = -gy[int(ym):int(ym + 14 * S), xm]
        if len(col):
            depths.append(np.argmax(col))
    depth = float(np.clip(np.median(depths) if depths else 6 * S, 3 * S, 10 * S))
    out = [np.stack([xs, yb], 1).astype(np.float32)]          # ligne d'attache
    wave = []
    for (x0, y0), (x1, y1) in zip(nodes, nodes[1:]):
        if x1 - x0 > 2.2 * per:                               # trou (médaillon rond) : on coupe
            if len(wave) > 1:
                out.append(np.array(wave, np.float32))
            wave = []
            continue
        t = np.linspace(0, 1, 12)
        for tt in t[(1 if wave else 0):]:
            wave.append((x0 + (x1 - x0) * tt, y0 + (y1 - y0) * tt + depth * 4 * tt * (1 - tt)))
    if len(wave) > 1:
        out.append(np.array(wave, np.float32))
    return out

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def build(min_frac, g_canny, g_fine, g_mid, g_horse, cache):
    L = min_frac * H
    lines = []

    # 1) contours Canny (comme manege_contours.png), squelettisés et tracés
    if "polys" not in cache:
        inner = cv2.erode(E.mask, np.ones((9, 9), np.uint8)) > 0
        e = cv2.Canny(g_canny, 40, 100, L2gradient=True)
        e[~inner] = 0
        # chevaux : contours d'une image plus lissée (silhouettes sans détails internes)
        eh = cv2.Canny(g_horse, 30, 80, L2gradient=True)
        hz = np.zeros((H, W), bool)
        for x0, y0, x1, y1 in HORSES:
            hz[y0 * S:y1 * S, x0 * S:x1 * S] = True
        e[hz] = 0
        e[hz & inner] = eh[hz & inner] | e[hz & inner]
        cache["polys"] = trace(skeletonize(cv2.dilate(e, np.ones((3, 3), np.uint8)) > 0))
    polys = cache["polys"]

    # médaillons (pour vider l'intérieur des tableaux peints)
    if "med" not in cache:
        meds = medallion_params(g_mid)
        cache["med"] = [(m, inner_scale(g_mid, m)) for m in meds]
    meds = cache["med"]

    def inside_medallion(p, k=0.95):
        return any(((p[0] - x) / (rx * k)) ** 2 + ((p[1] - y) / (ry * k)) ** 2 < 1 for (x, y, rx, ry), _ in meds)

    y_cone_bot = E.Y["cone"][1] * S
    for l in polys:
        n = length(l)
        m = l.mean(0)
        cy = m[1]
        if cy < 172 * S:                                         # flèche : gardée
            ok = n > L
        elif cy < y_cone_bot:                                    # cône : les rayures sont redessinées
            ok = False
        elif cy < 400 * S:                                       # bandeau : décors, pas l'intérieur des médaillons
            ok = n > 1.3 * L and not inside_medallion(m)
        elif cy < 455 * S:                                       # toit évasé : seulement les lignes ~horizontales
            ok = n > 1.5 * L and chord_vert(l) < 20 and np.linalg.norm(l[-1] - l[0]) > 0.7 * n
        elif cy < Y_ROOF:                                        # franges : redessinées par le feston mesuré
            ok = n > 4 * L and chord_vert(l) < 10
        elif cy < Y_UNDER_END:                                   # sous le toit
            if not (X_MIN < m[0] < X_MAX) or any(in_box(m, b) for b in DROP):
                ok = False
            elif any(in_box(m, b) for b in HORSES):
                ok = n > 2.5 * L                                 # chevaux : seulement les grands contours
            elif E.STAIRS_X[0] < m[0] < E.STAIRS_X[1]:
                ok = False                                       # escalier redessiné proprement plus bas
            else:
                ok = n > 4.0 * L and (chord_vert(l) > 78 or chord_vert(l) < 10) and \
                    np.linalg.norm(l[-1] - l[0]) > 0.9 * n               # grandes lignes droites
        else:
            ok = False                                           # plateforme redessinée (ellipse)
        if ok:
            lines.append(smooth(l))

    # 2) silhouette extérieure du toit (crêtes comprises)
    cnts, _ = cv2.findContours(E.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    cur = []
    for p in np.vstack([c, c[:1]]):
        if p[1] < 503 * S and 6 < p[0] < W - 6:
            cur.append(p)
        elif cur:
            if len(cur) > 40:
                lines.append(E.smooth_poly(cur, win=21, eps=1.8))
            cur = []
    if len(cur) > 40:
        lines.append(E.smooth_poly(cur, win=21, eps=1.8))

    # 3) rayures du cône : une sur deux
    angs = sorted(E.stripe_angles(g_fine, E.APEX, 50 * S, 95 * S, (-75, 75), y_cone_bot) +
                  E.stripe_angles(g_fine, E.APEX, 90 * S, 140 * S, (-75, 75), y_cone_bot))
    merged = []
    for a in angs:
        if merged and a - merged[-1][-1] < 1.6:
            merged[-1].append(a)
        else:
            merged.append([a])
    for a in [float(np.mean(m)) for m in merged][::2]:
        seg = E.ray_until(E.APEX, a, 30 * S, y_cone_bot)
        if seg is not None:
            lines.append(seg)

    # 4) rayures du toit évasé : une sur deux (même mesure que epurer_contours.py)
    ya, yb, ym = 402 * S, 450 * S, 426 * S
    gf = g_fine.astype(np.float32)
    row = cv2.GaussianBlur(gf[ym - 4 * S:ym + 4 * S].mean(0).reshape(1, -1), (0, 0), 2).ravel()
    d = np.gradient(row)
    xs = [x for x in range(3, W - 3) if mask_b[ym, x] and d[x] < d[x - 1] and d[x] <= d[x + 1] and d[x] < -2.5]
    gx = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=5)
    gyy = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=5)
    Jxx, Jyy, Jxy = (cv2.GaussianBlur(a, (0, 0), 12) for a in (gx * gx, gyy * gyy, gx * gyy))
    for x in xs[::2]:
        th = 0.5 * np.arctan2(2 * Jxy[ym, x], Jxx[ym, x] - Jyy[ym, x])
        k = -np.tan(th)
        if abs(k) > 2.5 or (k * (x - AX) < 0 and abs(k) > 0.15):
            continue
        p0, p1 = (x + k * (ya - ym), ya), (x + k * (yb - ym), yb)
        if all(mask_b[int(p[1]), int(np.clip(p[0], 0, W - 1))] for p in (p0, p1)):
            lines.append(np.array([p0, p1], np.float32))


    # 4b) bord festonné : position réelle des perles de la guirlande + arcs entre elles
    lines += festoon(g_fine)

    # 5) médaillons : double contour
    for (x, y, rx, ry), s in meds:
        lines.append(ellipse_pts(x, y, rx, ry))
        lines.append(ellipse_pts(x, y, rx * s, ry * s))

    # 6) barres verticales (du lambrequin jusqu'au plancher, coupées par l'ellipse plus bas)
    e_fine = cv2.Canny(g_fine, 20, 60, L2gradient=True)
    span = (E.Y["under"][1] - E.Y["under"][0]) * S
    segs = E.hough(e_fine, E.zone(*E.Y["under"]), min_len=0.03 * H, gap=30)
    bars = E.merge_axis(segs, vertical=True, tol=4 * S, min_cover=0.22, span=span)
    bars = [b for b in bars if b[2] - b[1] > 0.62 * span and X_MIN < b[0] < X_MAX
            and not (E.STAIRS_X[0] - 6 * S < b[0] < E.STAIRS_X[1] + 6 * S)]
    kept_bars = []
    for b in bars:
        if not kept_bars or b[0] - kept_bars[-1] > 9 * S:
            kept_bars.append(b[0])

    # 7) plateforme ronde : ellipse vue de face
    #   - largeur : englobe toutes les barres
    #   - avant du plancher : bas de la barrière mesuré sur la photo
    #   - aplatissement : courbure du bord du toit (même angle de vue), corrigée de la hauteur
    e_mid = cv2.Canny(g_mid, 20, 60, L2gradient=True)
    hs = E.merge_axis(E.hough(e_mid, E.zone(*E.Y["fence"]), min_len=0.08 * H, gap=30),
                      vertical=False, tol=4 * S, min_cover=0.15, span=W)
    hs.sort(key=lambda t: t[0])
    y_rail = hs[0][0] if hs else 765 * S                    # main courante de la barrière (photo)
    y_ground = max([h[0] for h in hs if h[0] > 820 * S], default=860 * S)   # pied de la barrière (photo)
    y_deck = 792 * S                                        # plancher : bas de l'escalier (photo)
    y_skirt = 846 * S                                       # bas des panneaux décorés (photo)
    a = max(abs(x - AX) for x in kept_bars) + 28 * S
    # aplatissement de l'ellipse : courbure du bord du toit (même vue), au minimum 0,12 pour rester lisible
    roof = E.robust_curve(g_mid, 498, 530, -1, 40 * S, W - 40 * S)
    ratio = 0.12
    if roof:
        r = roof[0]
        sag = r[:, 1].max() - 0.5 * (r[0, 1] + r[-1, 1])
        u = min(0.5 * (r[-1, 0] - r[0, 0]) / (400 * S), 0.98)
        ratio = sag / (1 - np.sqrt(1 - u * u)) / (400 * S)
    ratio = float(np.clip(ratio, 0.12, 0.2))
    b = a * ratio
    deck_y = y_deck - b                                     # centre de l'ellipse du plancher
    thick = y_skirt - y_deck
    af, bf = a + 22 * S, (a + 22 * S) * ratio               # barrière au sol, autour de la plateforme
    fence_cy = y_ground - bf
    fence_h = y_ground - y_rail

    def arc(cx_, cy, aa, bb, t0, t1, n=120):
        t = np.linspace(t0, t1, n)
        return np.stack([cx_ + aa * np.cos(t), cy + bb * np.sin(t)], 1).astype(np.float32)

    lines.append(arc(AX, deck_y, a, b, 0, 2 * np.pi, 200))         # plancher (ellipse complète)
    lines.append(arc(AX, deck_y + thick, a, b, 0, np.pi))          # bas des panneaux (avant)
    lines.append(np.array([[AX - a, deck_y], [AX - a, deck_y + thick]], np.float32))
    lines.append(np.array([[AX + a, deck_y], [AX + a, deck_y + thick]], np.float32))
    lines.append(arc(AX, fence_cy - fence_h, af, bf, 0, np.pi))     # main courante
    lines.append(arc(AX, fence_cy - 0.15 * fence_h, af, bf, 0, np.pi))   # lisse basse
    posts = E.merge_axis(E.hough(e_fine, E.zone(int(y_rail / S), int(y_ground / S), 40, 776),
                                 min_len=0.02 * H, gap=10), vertical=True, tol=6 * S, min_cover=0.35,
                         span=fence_h)
    post_x = sorted({round(p[0]) for p in posts if abs(p[0] - AX) < af - 10 * S})
    xs_posts = [AX - af + 3 * S, AX + af - 3 * S]
    for x in post_x:
        if all(abs(x - q) > 25 * S for q in xs_posts):
            xs_posts.append(x)
    for x in xs_posts:
        yb_ = fence_cy + bf * np.sqrt(max(0, 1 - ((x - AX) / af) ** 2))
        lines.append(np.array([[x, yb_ - fence_h], [x, yb_]], np.float32))

    def floor_y(x):                     # point du plancher sous une barre (cercle des barres, côté avant)
        return deck_y + b * 0.55 * np.sqrt(max(0, 1 - ((x - AX) / a) ** 2))

    for x in kept_bars:
        lines.append(np.array([[x, 524 * S], [x, floor_y(x)]], np.float32))

    # 8) escalier central : limons + marches mesurées
    segs_v = E.hough(e_mid, E.zone(600, 765, E.STAIRS_X[0], E.STAIRS_X[1]), min_len=0.03 * H, gap=20)
    sides = E.merge_axis(segs_v, vertical=True, tol=4 * S, min_cover=0.35, span=165 * S)
    xl, xr = (sides[0][0], sides[-1][0]) if len(sides) >= 2 else (352 * S, 445 * S)
    y_st = floor_y(AX)
    lines.append(np.array([[xl, 600 * S], [xl, y_st]], np.float32))
    lines.append(np.array([[xr, 600 * S], [xr, y_st]], np.float32))
    segs_h = E.hough(cv2.Canny(g_fine, 25, 70), E.zone(612, 760, int(xl) + 6, int(xr) - 6),
                     min_len=0.35 * (xr - xl), gap=10)
    last = -1e9
    for y, *_ in E.merge_axis(segs_h, vertical=False, tol=2.5 * S, min_cover=0.45, span=xr - xl):
        if y - last > 9 * S and y < y_st - 4 * S:
            lines.append(np.array([[xl, y], [xr, y]], np.float32))
            last = y

    # 9) bords du toit : bas du bandeau (courbe lisse)
    lines += E.robust_curve(g_mid, 380, 412, -1, 40 * S, W - 40 * S)
    # bas du lambrequin (bord inférieur du toit évasé)
    lines += E.robust_curve(g_mid, 498, 530, -1, 40 * S, W - 40 * S)

    # nettoyage final : on retire ce qui sort du manège (hors silhouette du toit et hors plateforme)
    def inside_scene(p):
        if p[1] < Y_ROOF:
            return True
        return abs(p[0] - AX) <= af + 8 * S
    lines = [l for l in lines if inside_scene(l.mean(0))]
    return lines, (fence_cy + bf)


# ---------------------------------------------------------------------------
# Rendu : recadré, centré sur l'axe du manège, fond blanc
# ---------------------------------------------------------------------------
STROKE = 3.0
MARGIN = 60 * S


def frame(lines, y_bottom):
    pts = np.vstack(lines)
    half = max(AX - pts[:, 0].min(), pts[:, 0].max() - AX) + MARGIN
    x0 = AX - half
    y0 = pts[:, 1].min() - MARGIN
    w, h = int(2 * half), int(max(y_bottom, pts[:, 1].max()) + MARGIN - y0)
    return [l - np.array([x0, y0], np.float32) for l in lines], w, h


def render_png(lines, w, h, path, ss=2):
    img = np.full((h * ss, w * ss), 255, np.uint8)
    for l in lines:
        p = np.round(l * ss * 16).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [p], False, 0, int(round(STROKE * ss)), cv2.LINE_AA, shift=4)
    cv2.imwrite(str(path), cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA))


def render_svg(lines, w, h, path):
    def d(p):
        if len(p) < 3:
            return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in p)
        s = [f"M{p[0][0]:.1f},{p[0][1]:.1f}"]
        for i in range(len(p) - 1):
            p0, p1, p2, p3 = p[max(i - 1, 0)], p[i], p[i + 1], p[min(i + 2, len(p) - 1)]
            c1, c2 = p1 + (p2 - p0) / 6, p2 - (p3 - p1) / 6
            s.append(f"C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}")
        return " ".join(s)
    body = "\n".join(f'    <path d="{d(l)}"/>' for l in lines)
    Path(path).write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
        f'  <rect width="100%" height="100%" fill="#fff"/>\n'
        f'  <g fill="none" stroke="#000" stroke-width="{STROKE}" stroke-linecap="round" stroke-linejoin="round">\n'
        f'{body}\n  </g>\n</svg>\n')


VARIANTS = {"A": 0.010, "B": 0.014, "C": 0.018}   # longueur minimale (fraction de la hauteur)

if __name__ == "__main__":
    gray = cv2.cvtColor(cv2.bilateralFilter(cv2.resize(photo, (W, H), interpolation=cv2.INTER_LANCZOS4), 9, 40, 9),
                        cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    g_canny = cv2.GaussianBlur(cv2.bilateralFilter(clahe.apply(gray), 11, 50, 11), (0, 0), 1.6)
    g_fine = E.smooth_gray(1.5, sp=4, sr=18)
    g_mid = E.smooth_gray(2.5)
    g_horse = E.smooth_gray(2.0, sp=5, sr=26)
    cache = {}
    for name, frac in VARIANTS.items():
        lines, yb = build(frac, g_canny, g_fine, g_mid, g_horse, cache)
        lines, w, h = frame(lines, yb)
        render_png(lines, w, h, OUT / f"manege_inter_{name}.png")
        render_svg(lines, w, h, OUT / f"manege_inter_{name}.svg")
        print(f"manege_inter_{name}: {len(lines)} traits, {w}x{h}px")
