#!/usr/bin/env python3
"""
Contours très épurés du manège (base de tatouage fineline), extraits de la photo.

Rien n'est dessiné « à la main » : chaque trait est mesuré sur la photo, puis
nettoyé pour être continu et lisse.

Pipeline :
  1. photo recadrée + masque du manège (produit par prepare_modele.py)
  2. lissage fort (réduction de résolution + mean-shift + bilatéral, puis réagrandissement)
  3. détection par zone :
       - silhouette du toit   : contour extérieur du masque, lissé
       - rayures du cône/toit : segments de Hough orientés vers le sommet, fusionnés,
                                triés par angle -> on n'en garde qu'une sur N
       - médaillons           : ellipses ajustées sur les contours (cv2.fitEllipse)
       - bords horizontaux    : suivi du bord le plus contrasté colonne par colonne
       - barres verticales    : segments de Hough verticaux fusionnés par position
       - escalier             : 2 limons + marches (une sur N)
       - plateforme           : grandes lignes horizontales
  4. filtrage par longueur (en % de la hauteur), lissage, simplification (approxPolyDP)
  5. rendu PNG anti-aliasé à épaisseur fixe + SVG vectoriel (traits seuls)
"""
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "modeles"
S = 3                                   # facteur d'agrandissement (comme prepare_modele.py)
photo = cv2.imread(str(HERE / "manege.png"))[200:1125, 0:816]
mask = cv2.imread(str(OUT / "manege_masque.png"), 0)
if mask is None:
    raise SystemExit("Lance d'abord prepare_modele.py (il produit modeles/manege_masque.png)")
H, W = mask.shape
mask_b = mask > 127

# Repères mesurés sur la photo recadrée (816 x 925), multipliés par S
APEX = np.array([404.5, 148.5]) * S   # estimé par moindres carrés sur les rayures détectées          # sommet du cône
Y = dict(cone=(178, 296), band=(296, 405), skirt=(395, 462), under=(522, 762), fence=(748, 872))
STAIRS_X = (340 * S, 458 * S)


# ---------------------------------------------------------------------------
# Lissage
# ---------------------------------------------------------------------------
def smooth_gray(down, sp=6, sr=22):
    small = cv2.resize(photo, None, fx=1 / down, fy=1 / down, interpolation=cv2.INTER_AREA)
    small = cv2.pyrMeanShiftFiltering(small, sp, sr)
    small = cv2.bilateralFilter(small, 7, 45, 7)
    big = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    return cv2.GaussianBlur(g, (0, 0), 2.0)


def zone(y0, y1, x0=0, x1=None):
    z = np.zeros((H, W), bool)
    z[y0 * S:y1 * S, x0:(x1 if x1 is not None else W)] = True
    return z & mask_b


# ---------------------------------------------------------------------------
# Outils lignes
# ---------------------------------------------------------------------------
def hough(edges, region, min_len, gap):
    e = edges.copy()
    e[~region] = 0
    segs = cv2.HoughLinesP(e, 1, np.pi / 360, threshold=int(min_len * 0.5),
                           minLineLength=int(min_len), maxLineGap=int(gap))
    return [] if segs is None else [s.astype(float) for s in np.asarray(segs).reshape(-1, 4)]


def smooth_poly(pts, win=15, eps=1.5):
    pts = np.asarray(pts, np.float32)
    if len(pts) >= win:
        k = np.ones(win) / win
        pad = np.vstack([np.repeat(pts[:1], win // 2, 0), pts, np.repeat(pts[-1:], win // 2, 0)])
        pts = np.stack([np.convolve(pad[:, 0], k, "valid"), np.convolve(pad[:, 1], k, "valid")], 1)
    pts = np.ascontiguousarray(pts, np.float32)
    return cv2.approxPolyDP(pts.reshape(-1, 1, 2), eps, False).reshape(-1, 2).astype(np.float32)


def merge_radial(segs, center, tol_deg, min_total):
    """Regroupe les segments alignés sur un même rayon issu de `center`.
    Retourne [(angle, r_min, r_max)] pour chaque rayure."""
    items = []
    for x1, y1, x2, y2 in segs:
        a = np.array([x1, y1]) - center
        b = np.array([x2, y2]) - center
        d = b - a
        ang_seg = np.degrees(np.arctan2(d[0], d[1]))
        mid = (a + b) / 2
        ang_mid = np.degrees(np.arctan2(mid[0], mid[1]))
        diff = (ang_seg - ang_mid + 90) % 180 - 90
        if abs(diff) > 5:                        # pas orienté vers le sommet
            continue
        ra, rb = np.linalg.norm(a), np.linalg.norm(b)
        items.append((ang_mid, min(ra, rb), max(ra, rb)))
    items.sort()
    groups = []
    for it in items:
        if groups and it[0] - groups[-1][-1][0] < tol_deg:
            groups[-1].append(it)
        else:
            groups.append([it])
    out = []
    for g in groups:
        total = sum(r1 - r0 for _, r0, r1 in g)
        if total >= min_total:
            out.append((np.mean([a for a, _, _ in g]), min(r0 for _, r0, _ in g), max(r1 for _, _, r1 in g)))
    return out


def merge_axis(segs, vertical, tol, min_cover, span):
    """Regroupe des segments verticaux (par x) ou horizontaux (par y)."""
    items = []
    for x1, y1, x2, y2 in segs:
        if vertical:
            if abs(x2 - x1) > 0.07 * abs(y2 - y1):
                continue
            items.append(((x1 + x2) / 2, min(y1, y2), max(y1, y2)))
        else:
            if abs(y2 - y1) > 0.07 * abs(x2 - x1):
                continue
            items.append(((y1 + y2) / 2, min(x1, x2), max(x1, x2)))
    items.sort()
    groups = []
    for it in items:
        if groups and it[0] - groups[-1][-1][0] < tol:
            groups[-1].append(it)
        else:
            groups.append([it])
    out = []
    for g in groups:
        lo, hi = min(i[1] for i in g), max(i[2] for i in g)
        # couverture réelle (union des intervalles)
        iv = sorted((i[1], i[2]) for i in g)
        cov, cur = 0.0, list(iv[0])
        for a, b in iv[1:]:
            if a <= cur[1]:
                cur[1] = max(cur[1], b)
            else:
                cov += cur[1] - cur[0]
                cur = [a, b]
        cov += cur[1] - cur[0]
        if cov >= min_cover * span:
            out.append((np.mean([i[0] for i in g]), lo, hi, cov))
    return out


def follow_edge(g, y0, y1, x0, x1, step=3, min_strength=10):
    """Bord horizontal le plus contrasté entre y0 et y1 (coordonnées photo), colonne par colonne."""
    gy = np.abs(cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 0, 1, ksize=5)) / 32
    pts = []
    for x in range(int(x0), int(x1), step):
        col = gy[y0 * S:y1 * S, x].copy()
        col[~mask_b[y0 * S:y1 * S, x]] = 0
        i = int(np.argmax(col))
        if col[i] > min_strength:
            pts.append((x, y0 * S + i))
    if len(pts) < 10:
        return []
    pts = np.array(pts, np.float32)
    # médiane glissante pour éliminer les sauts, puis on coupe là où ça saute encore
    ys = pts[:, 1].copy()
    k = 7
    med = np.array([np.median(ys[max(0, i - k):i + k + 1]) for i in range(len(ys))])
    good = np.abs(ys - med) < 4 * S
    pts = np.stack([pts[:, 0], med], 1)[good]
    pieces, cur = [], [pts[0]]
    for p in pts[1:]:
        if p[0] - cur[-1][0] > 6 * step or abs(p[1] - cur[-1][1]) > 3 * S:
            pieces.append(np.array(cur))
            cur = [p]
        else:
            cur.append(p)
    pieces.append(np.array(cur))
    return [smooth_poly(p, win=11, eps=1.2) for p in pieces if len(p) > 1 and p[-1][0] - p[0][0] > 0.06 * W]


def bilinear(img, x, y):
    return cv2.remap(img, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR)


def stripe_angles(g, center, r0, r1, ang_range, y_max):
    """Profil d'intensité en fonction de l'angle autour de `center` (moyenne sur plusieurs rayons).
    Retourne les angles des fronts « clair -> sombre » (bord gauche de chaque rayure rouge)."""
    angs = np.arange(ang_range[0], ang_range[1], 0.05)
    prof = np.zeros_like(angs)
    cnt = np.zeros_like(angs)
    gf = g.astype(np.float32)
    for r in np.linspace(r0, r1, 40):
        x = center[0] + r * np.sin(np.radians(angs))
        y = center[1] + r * np.cos(np.radians(angs))
        ok = (y < y_max) & (x > 0) & (x < W - 1)
        ok &= mask_b[np.clip(y, 0, H - 1).astype(int), np.clip(x, 0, W - 1).astype(int)]
        v = bilinear(gf, x[None, :], y[None, :])[0]
        if ok.sum() < 20:
            continue
        v = (v - v[ok].mean()) / (v[ok].std() + 1e-6)      # normalisation par rayon
        prof[ok] += v[ok]
        cnt[ok] += 1
    prof = np.where(cnt > 8, prof / np.maximum(cnt, 1), 0)
    prof = cv2.GaussianBlur(prof.reshape(1, -1).astype(np.float32), (0, 0), 3).ravel()
    d = np.gradient(prof)
    peaks = [i for i in range(2, len(d) - 2) if d[i] < d[i - 1] and d[i] <= d[i + 1] and d[i] < -0.007]
    return [angs[i] for i in peaks]


def ray_until(center, ang, r_start, y_max):
    u = np.array([np.sin(np.radians(ang)), np.cos(np.radians(ang))])
    pts = []
    for r in np.arange(r_start, 400 * S, 2):
        p = center + u * r
        if p[1] >= y_max or not (0 <= p[0] < W) or not mask_b[int(p[1]), int(p[0])]:
            break
        pts.append(p)
    return np.array([pts[0], pts[-1]], np.float32) if len(pts) > 10 else None


def robust_curve(g, y0, y1, sign, x0, x1, deg=4, step=4):
    """Bord horizontal : point de plus fort gradient (signé) par colonne, puis
    ajustement polynomial robuste -> UNE courbe lisse et continue."""
    gy = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 0, 1, ksize=5) * sign / 32
    xs, ys, ws = [], [], []
    for x in range(int(x0), int(x1), step):
        col = gy[y0 * S:y1 * S, x].copy()
        col[~mask_b[y0 * S:y1 * S, x]] = -1e9
        i = int(np.argmax(col))
        if col[i] > 6:
            xs.append(x); ys.append(y0 * S + i); ws.append(col[i])
    if len(xs) < 20:
        return []
    xs, ys, ws = map(np.array, (xs, ys, ws))
    keep = np.ones(len(xs), bool)
    for _ in range(6):
        c = np.polyfit(xs[keep] / W, ys[keep], deg, w=np.sqrt(ws[keep]))
        res = np.abs(np.polyval(c, xs / W) - ys)
        keep = res < max(3 * S, np.percentile(res[keep], 60))
    xa, xb = xs[keep].min(), xs[keep].max()
    xx = np.linspace(xa, xb, 120)
    return [np.stack([xx, np.polyval(c, xx / W)], 1).astype(np.float32)]


def ellipse_poly(e, n=90):
    (cx, cy), (MA, ma), ang = e
    t = np.linspace(0, 2 * np.pi, n + 1)
    a = np.radians(ang)
    x = cx + MA / 2 * np.cos(t) * np.cos(a) - ma / 2 * np.sin(t) * np.sin(a)
    y = cy + MA / 2 * np.cos(t) * np.sin(a) + ma / 2 * np.sin(t) * np.cos(a)
    return np.stack([x, y], 1).astype(np.float32)


def medallions(g, n_big, n_small):
    """Médaillons du bandeau par HoughCircles (vote sur des arcs partiels, robuste aux cadres ornés).
    Grands médaillons peints : passe « ovale » (image compressée horizontalement).
    Petits médaillons à visage : passe ronde. On garde les votes les plus forts."""
    gg = g.copy()
    gg[~zone(290, 400)] = 200

    def detect(aspect, rmin, rmax, p2):
        small = cv2.resize(gg, None, fx=1 / aspect / S, fy=1 / S, interpolation=cv2.INTER_AREA)
        c = cv2.HoughCircles(small, cv2.HOUGH_GRADIENT, dp=1, minDist=rmin * 1.6, param1=60, param2=p2,
                             minRadius=rmin, maxRadius=rmax)
        return [] if c is None else [(x * aspect * S, y * S, r * aspect * S, r * S) for x, y, r in c.reshape(-1, 3)]

    big = []
    for e in detect(1.35, 20, 38, 18):            # grands médaillons : pas de chevauchement
        if all(np.hypot(e[0] - o[0], e[1] - o[1]) > 0.95 * (e[2] + o[2]) for o in big):
            big.append(e)
        if len(big) == n_big:
            break
    out = [e for e in big]
    for x, y, rx, ry in detect(1.0, 9, 17, 16):
        if not (340 * S < y < 378 * S):              # médaillons à visage : milieu du bandeau
            continue
        if sum(1 for e in out if e[2] < 25 * S) >= n_small and len(out) >= len(big) + n_small:
            break
        inside = any(((x - bx) / (bx_r + rx + 3 * S)) ** 2 + ((y - by) / (by_r + ry + 3 * S)) ** 2 < 1.0
                     for bx, by, bx_r, by_r in out)          # aucun chevauchement
        if not inside and len(out) < len(big) + n_small:
            out.append((x, y, rx, ry))
    # Hough surestime un peu le rayon des grands cadres ornés : on resserre le tracé
    return [ellipse_poly(((x, y), (2 * rx * (0.86 if rx > 25 * S else 1), 2 * ry * (0.86 if rx > 25 * S else 1)), 0))
            for x, y, rx, ry in out]


# ---------------------------------------------------------------------------
# Construction d'un niveau
# ---------------------------------------------------------------------------
LEVELS = {
    1: dict(name="manege_epure_1_leger", cone_every=2, skirt_every=2, medallions=(5, 4),
            bar_cover=0.22, bar_every=1, stair_every=1, fence_lines=3, follow=("band", "trim", "fringe")),
    2: dict(name="manege_epure_2_moyen", cone_every=3, skirt_every=3, medallions=(3, 3),
            bar_cover=0.28, bar_every=1, stair_every=2, fence_lines=2, follow=("band", "trim")),
    3: dict(name="manege_epure_3_tres_epure", cone_every=5, skirt_every=5, medallions=(3, 0),
            bar_cover=0.34, bar_every=2, stair_every=3, fence_lines=2, follow=("band", "trim")),
}


def build(P, g_fine, g_mid, g_coarse):
    lines = []
    min_len = 0.02 * H                                             # filtre de longueur : 2 % de la hauteur

    # --- silhouette du toit (contour extérieur du masque), jusqu'au bas du lambrequin
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    cur = []
    for p in np.vstack([c, c[:1]]):
        if p[1] < 503 * S and 6 < p[0] < W - 6:
            cur.append(p)
        elif cur:
            if len(cur) > 40:
                lines.append(smooth_poly(cur, win=45, eps=2.5))
            cur = []
    if len(cur) > 40:
        lines.append(smooth_poly(cur, win=45, eps=2.5))

    # --- rayures du cône : fronts mesurés sur le profil angulaire autour du sommet, une sur N
    y_cone_bot = Y["cone"][1] * S
    angs = sorted(stripe_angles(g_fine, APEX, 50 * S, 95 * S, (-75, 75), y_cone_bot) +
                  stripe_angles(g_fine, APEX, 90 * S, 140 * S, (-75, 75), y_cone_bot))
    merged = []
    for a in angs:                                   # fusion des détections des 2 bandes (< 1,6°)
        if merged and a - merged[-1][-1] < 1.6:
            merged[-1].append(a)
        else:
            merged.append([a])
    angs = [float(np.mean(m)) for m in merged]
    for ang in angs[::P["cone_every"]]:
        seg = ray_until(APEX, ang, 30 * S, y_cone_bot)
        if seg is not None and np.linalg.norm(seg[1] - seg[0]) > min_len:
            lines.append(seg)

    # --- rayures du toit évasé : fronts le long d'une rangée, orientation locale mesurée
    ya, yb, ym = 402 * S, 450 * S, 426 * S
    gf = g_fine.astype(np.float32)
    row = gf[ym - 4 * S:ym + 4 * S].mean(0)
    row = cv2.GaussianBlur(row.reshape(1, -1), (0, 0), 2).ravel()
    d = np.gradient(row)
    on = mask_b[ym]
    xs = [x for x in range(3, W - 3) if on[x] and d[x] < d[x - 1] and d[x] <= d[x + 1] and d[x] < -2.5]
    gx = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=5)
    gyy = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=5)
    Jxx = cv2.GaussianBlur(gx * gx, (0, 0), 12)
    Jyy = cv2.GaussianBlur(gyy * gyy, (0, 0), 12)
    Jxy = cv2.GaussianBlur(gx * gyy, (0, 0), 12)
    for x in xs[::P["skirt_every"]]:
        # direction des rayures = perpendiculaire au gradient dominant
        th = 0.5 * np.arctan2(2 * Jxy[ym, x], Jxx[ym, x] - Jyy[ym, x])
        k = -np.tan(th)                                   # dx/dy le long de la rayure
        if abs(k) > 2.5 or (k * (x - APEX[0]) < 0 and abs(k) > 0.15):
            continue
        p0, p1 = (x + k * (ya - ym), ya), (x + k * (yb - ym), yb)
        if all(mask_b[int(p[1]), int(np.clip(p[0], 0, W - 1))] for p in (p0, p1)):
            lines.append(np.array([p0, p1], np.float32))

    # --- médaillons
    lines += medallions(g_mid, *P["medallions"])

    # --- bords horizontaux : bas du bandeau, haut du lambrequin, bas des franges (courbes lisses)
    windows = {"band": (380, 412, -1), "trim": (446, 472, 1), "fringe": (498, 530, -1)}
    xs_m = np.where(mask_b[int(450 * S)])[0]
    for key in P["follow"]:
        y0, y1, sg = windows[key]
        lines += robust_curve(g_mid, y0, y1, sg, xs_m.min() + 15, xs_m.max() - 15)

    # --- barres verticales sous le toit (tracées d'un trait, du lambrequin à la plateforme)
    e_mid = cv2.Canny(g_mid, 20, 60, L2gradient=True)
    under = zone(*Y["under"])
    span = (Y["under"][1] - Y["under"][0]) * S
    e_fine = cv2.Canny(g_fine, 20, 60, L2gradient=True)
    segs = hough(e_fine, under, min_len=0.03 * H, gap=30)
    bars = merge_axis(segs, vertical=True, tol=4 * S, min_cover=P["bar_cover"], span=span)
    bars = [b for b in bars if b[2] - b[1] > 0.62 * span]   # une vraie barre traverse toute la hauteur
    merged = []                                             # 2 bords d'une même barre -> 1 trait
    for b in bars:
        if merged and b[0] - merged[-1][0] < 9 * S:
            continue
        merged.append(b)
    bars = merged
    bars = [b for b in bars if not (STAIRS_X[0] - 6 * S < b[0] < STAIRS_X[1] + 6 * S)]
    all_bars = list(bars)
    bars = bars[::P["bar_every"]]
    for x, lo, hi, _ in bars:
        lines.append(np.array([[x, 524 * S], [x, 760 * S]], np.float32))

    # --- escalier : deux limons + marches
    st_reg = zone(600, 765, STAIRS_X[0], STAIRS_X[1])
    segs_v = hough(e_mid, st_reg, min_len=0.03 * H, gap=20)
    sides = merge_axis(segs_v, vertical=True, tol=4 * S, min_cover=0.35, span=165 * S)
    if len(sides) >= 2:
        xl, xr = sides[0][0], sides[-1][0]
    else:
        xl, xr = 352 * S, 445 * S
    lines.append(np.array([[xl, 600 * S], [xl, 762 * S]], np.float32))
    lines.append(np.array([[xr, 600 * S], [xr, 762 * S]], np.float32))
    segs_h = hough(cv2.Canny(g_fine, 25, 70), zone(612, 760, int(xl) + 6, int(xr) - 6),
                   min_len=0.35 * (xr - xl), gap=10)
    steps = merge_axis(segs_h, vertical=False, tol=2.5 * S, min_cover=0.45, span=xr - xl)
    picked, last = [], -1e9
    for y, *_ in steps:
        if y - last > 9 * S:
            picked.append(y)
            last = y
    for y in picked[::P["stair_every"]]:
        lines.append(np.array([[xl, y], [xr, y]], np.float32))

    # --- plateforme / barrière : grandes horizontales
    fence = zone(*Y["fence"])
    segs = hough(e_mid, fence, min_len=0.08 * H, gap=30)
    hs = merge_axis(segs, vertical=False, tol=4 * S, min_cover=0.15, span=W)
    hs.sort(key=lambda t: t[0])
    if hs:
        top, bot = hs[0], max(hs, key=lambda t: t[0] + (t[3] if t[0] > 820 * S else -1e9))
        mids = [h for h in hs if top[0] + 15 * S < h[0] < bot[0] - 15 * S]
        chosen = [top, bot] + ([max(mids, key=lambda t: t[3])] if P["fence_lines"] == 3 and mids else [])
        lo = min(c[1] for c in chosen)
        hi = max(c[2] for c in chosen)
        if all_bars:                                     # plateforme centrée sur l'axe du manège,
            half = max(abs(b[0] - APEX[0]) for b in all_bars) + 22 * S   # englobant toutes les barres
            half = max(half, APEX[0] - lo, hi - APEX[0])
            lo, hi = APEX[0] - half, APEX[0] + half
        seen = set()
        for y, *_ in chosen:
            if round(y) in seen:
                continue
            seen.add(round(y))
            lines.append(np.array([[lo, y], [hi, y]], np.float32))
        if len(seen) >= 2:                               # côtés de la plateforme
            ys = sorted(seen)
            for x in (lo, hi):
                lines.append(np.array([[x, ys[0]], [x, ys[-1]]], np.float32))
    return lines


# ---------------------------------------------------------------------------
# Rendu
# ---------------------------------------------------------------------------
STROKE = 3.5


def render_png(lines, path, ss=2):
    img = np.full((H * ss, W * ss), 255, np.uint8)
    for l in lines:
        pts = np.round(np.asarray(l) * ss * 16).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, 0, int(round(STROKE * ss)), cv2.LINE_AA, shift=4)
    cv2.imwrite(str(path), cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA))


def render_svg(lines, path):
    def d(p):
        p = np.asarray(p)
        if len(p) < 3:
            return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in p)
        s = [f"M{p[0][0]:.1f},{p[0][1]:.1f}"]
        for i in range(len(p) - 1):                     # Catmull-Rom -> Bézier (courbes lisses)
            p0, p1, p2, p3 = p[max(i - 1, 0)], p[i], p[i + 1], p[min(i + 2, len(p) - 1)]
            c1, c2 = p1 + (p2 - p0) / 6, p2 - (p3 - p1) / 6
            s.append(f"C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}")
        return " ".join(s)

    body = "\n".join(f'    <path d="{d(l)}"/>' for l in lines)
    Path(path).write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
        f'  <rect width="100%" height="100%" fill="#fff"/>\n'
        f'  <g fill="none" stroke="#000" stroke-width="{STROKE}" stroke-linecap="round" stroke-linejoin="round">\n'
        f'{body}\n  </g>\n</svg>\n')


if __name__ == "__main__":
    g_fine = smooth_gray(1.5, sp=4, sr=18)      # garde les rayures
    g_mid = smooth_gray(2.5)                    # structure seulement
    g_coarse = smooth_gray(3.5, sp=8, sr=30)    # grandes formes (médaillons)
    for lvl, P in LEVELS.items():
        lines = build(P, g_fine, g_mid, g_coarse)
        render_png(lines, OUT / f"{P['name']}.png")
        render_svg(lines, OUT / f"{P['name']}.svg")
        print(f"{P['name']}: {len(lines)} traits")
