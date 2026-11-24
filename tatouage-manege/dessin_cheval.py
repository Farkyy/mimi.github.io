#!/usr/bin/env python3
"""
Manège ancien – line art fineline pour tatouage.

Tout est dessiné « à la main » en code : chaque élément est construit à partir
de points choisis à la main, lissés en courbes de Bézier (Catmull-Rom).
Traits noirs uniquement. Les formes au premier plan ont un fond blanc
uniquement pour masquer les traits situés derrière (aucun remplissage visible).

Usage :
    python3 manege_lineart.py            -> manege_lineart.svg + manege_simplifie.svg
"""
import math
from pathlib import Path

CX = 500.0  # axe central du manège
OUT = Path(__file__).resolve().parent

# Épaisseurs (px) : 3 = contours principaux, 2 = secondaires, 1.2 = détails
W_MAIN, W_MID, W_FINE, W_HAIR = 2.8, 2.0, 1.3, 1.0


# --------------------------------------------------------------------------
# Outils géométriques
# --------------------------------------------------------------------------
def f(v):
    return f"{v:.2f}".rstrip("0").rstrip(".")


def pt(p):
    return f"{f(p[0])},{f(p[1])}"


def catmull(points, closed=False):
    """Courbe lisse passant par tous les points (Catmull-Rom -> Bézier)."""
    pts = list(points)
    n = len(pts)
    if n < 2:
        return ""
    d = [f"M{pt(pts[0])}"]
    rng = range(n) if closed else range(n - 1)
    for i in rng:
        p0 = pts[(i - 1) % n] if (closed or i > 0) else pts[i]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n] if (closed or i + 2 < n) else p2
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        d.append(f"C{pt(c1)} {pt(c2)} {pt(p2)}")
    if closed:
        d.append("Z")
    return " ".join(d)


def catmull_cont(points):
    """Comme catmull() mais sans le M initial (pour enchaîner des tronçons)."""
    s = catmull(points)
    return s[s.index("C"):] if "C" in s else ""


def ell(cy, rx, ry, th_deg):
    """Point sur une ellipse horizontale vue de face ; th=0 -> devant."""
    t = math.radians(th_deg)
    return (CX + rx * math.sin(t), cy + ry * math.cos(t))


def front_arc(cy, rx, ry, a0=-90, a1=90):
    return f"M{pt(ell(cy, rx, ry, a0))} A{f(rx)},{f(ry)} 0 0 0 {pt(ell(cy, rx, ry, a1))}"


class Svg:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.items = []

    def path(self, d, w=W_MID, fill="none", extra=""):
        self.items.append(f'<path d="{d}" fill="{fill}" stroke="#000" stroke-width="{f(w)}"{extra}/>')

    def mask(self, d):
        """Forme blanche sans trait : sert seulement à cacher ce qui est derrière."""
        self.items.append(f'<path d="{d}" fill="#fff" stroke="none"/>')

    def circle(self, c, r, w=W_FINE, fill="none"):
        self.items.append(f'<circle cx="{f(c[0])}" cy="{f(c[1])}" r="{f(r)}" fill="{fill}" stroke="#000" stroke-width="{f(w)}"/>')

    def ellipse(self, c, rx, ry, w=W_FINE, fill="none", rot=0):
        tr = f' transform="rotate({f(rot)} {pt(c)})"' if rot else ""
        self.items.append(f'<ellipse cx="{f(c[0])}" cy="{f(c[1])}" rx="{f(rx)}" ry="{f(ry)}" fill="{fill}" stroke="#000" stroke-width="{f(w)}"{tr}/>')

    def raw(self, s):
        self.items.append(s)

    def save(self, path):
        body = "\n  ".join(self.items)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
               f'viewBox="0 0 {self.w} {self.h}" stroke-linecap="round" stroke-linejoin="round">\n'
               f'  <rect width="100%" height="100%" fill="#fff"/>\n  {body}\n</svg>\n')
        Path(path).write_text(svg)


# --------------------------------------------------------------------------
# Cheval de manège (dessiné de profil, tourné vers la droite, origine = barre)
# --------------------------------------------------------------------------
def rot(p, deg):
    a = math.radians(deg)
    return (p[0] * math.cos(a) - p[1] * math.sin(a), p[0] * math.sin(a) + p[1] * math.cos(a))


def limb(spine, hoof_len=10):
    """Jambe à partir d'une ligne médiane [(x, y, demi-largeur)].
    Retourne (contour ouvert, ligne de couronne du sabot, forme de masque)."""
    L, R = [], []
    for i, (x, y, w) in enumerate(spine):
        a = spine[max(i - 1, 0)]
        b = spine[min(i + 1, len(spine) - 1)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = math.hypot(dx, dy) or 1
        nx, ny = -dy / n, dx / n
        L.append((x + nx * w, y + ny * w))
        R.append((x - nx * w, y - ny * w))
    # direction de fin -> sabot
    (x1, y1, w1), (x0, y0, _) = spine[-1], spine[-2]
    dx, dy = x1 - x0, y1 - y0
    n = math.hypot(dx, dy)
    ux, uy = dx / n, dy / n
    nx, ny = -uy, ux
    hl = (L[-1][0] + ux * hoof_len + nx * 1.8, L[-1][1] + uy * hoof_len + ny * 1.8)
    hr = (R[-1][0] + ux * hoof_len * 0.75 - nx * 1.2, R[-1][1] + uy * hoof_len * 0.75 - ny * 1.2)
    d = catmull(L) + f" L{pt(hl)} L{pt(hr)} L{pt(R[-1])} " + catmull_cont(list(reversed(R)))
    d = d.replace(f"L{pt(R[-1])} C", f"L{pt(R[-1])} C", 1)
    coronet = f"M{pt(L[-1])} L{pt(R[-1])}"
    return d, coronet, L, R


def horse(svg, ox, oy, scale=1.0, facing=1, detail=1.0):
    """Dessine un cheval cabré. (ox, oy) = position de la barre au niveau de la selle."""
    TILT = -13  # cabré : l'avant se lève

    def T(p):
        q = rot(p, TILT)
        return (ox + facing * q[0] * scale, oy + q[1] * scale)

    def P(points):
        return [T(p) for p in points]

    sw = lambda w: w  # traits gardés en px réels (non mis à l'échelle)

    def part(stroke_pts, inner=None, w=W_MAIN, closed=False, smooth=True):
        pts = P(stroke_pts)
        d = catmull(pts, closed) if smooth else "M" + " L".join(pt(p) for p in pts)
        if closed:
            svg.path(d, sw(w), fill="#fff")
        else:
            if inner:
                svg.mask(d + " " + " ".join("L" + pt(T(q)) for q in inner) + " Z")
            svg.path(d, sw(w))

    def line(points, w=W_FINE):
        svg.path(catmull(P(points)), sw(w))

    def leg(spine, inner, w=W_MAIN):
        sp = [(T((x, y))[0], T((x, y))[1], wd * scale) for x, y, wd in spine]
        d, coronet, L, R = limb(sp, hoof_len=9 * scale)
        svg.mask(d + " " + " ".join("L" + pt(T(q)) for q in inner) + " Z")
        svg.path(d, sw(w))
        svg.path(coronet, sw(W_FINE))
        return L, R

    # ---------- Jambes côté éloigné (derrière le corps) ----------
    leg([(50, 14, 11), (62, 28, 9.5), (74, 42, 8.5), (70, 58, 6.4), (63, 69, 7.2), (60, 75, 6)],
        [(36, 0), (26, 4)], W_MID)
    leg([(-36, 14, 15), (-28, 42, 10.5), (-38, 62, 8.5), (-33, 79, 6.2), (-29, 88, 7), (-27, 94, 5.8)],
        [(-26, 0), (-42, 0)], W_MID)

    # ---------- Queue (mèches qui ondulent vers l'arrière) ----------
    tail_out = [(-70, -22), (-90, -20), (-104, -4), (-110, 20), (-104, 42), (-108, 60), (-118, 72)]
    tail_in = [(-108, 64), (-98, 48), (-94, 26), (-90, 6), (-76, -6)]
    svg.path(catmull(P(tail_out + tail_in)), W_MID, fill="#fff")
    line([(-84, -14), (-98, 2), (-102, 24), (-100, 46), (-110, 62)], W_HAIR)
    if detail > 0.5:
        line([(-94, -12), (-104, 6), (-106, 26)], W_HAIR)
        line([(-104, 48), (-112, 60), (-120, 72)], W_HAIR)

    # ---------- Corps ----------
    body = [(36, -36), (14, -29), (-6, -27), (-28, -29), (-48, -33), (-64, -28), (-74, -14), (-76, 4),
            (-70, 20), (-56, 30), (-34, 34), (-8, 36), (18, 35), (42, 30), (58, 20), (68, 6), (70, -10)]
    part(body, w=W_MAIN, closed=True)

    # ---------- Jambes côté proche ----------
    leg([(-54, 8, 19), (-46, 38, 12), (-60, 60, 9), (-56, 78, 6.4), (-52, 88, 7.4), (-49, 94, 6)],
        [(-40, -4), (-64, -8)], W_MAIN)

    # muscles (traits fins)
    line([(-48, -18), (-60, -2), (-58, 16)], W_HAIR)          # cuisse
    line([(44, -20), (52, -4), (48, 12)], W_HAIR)             # épaule
    if detail > 0.5:
        line([(-22, 30), (8, 31), (32, 27)], W_HAIR)          # ventre

    # ---------- Encolure arquée + tête ----------
    neck = [(24, -34), (36, -54), (50, -72), (64, -84), (74, -89), (86, -90), (97, -80),
            (107, -65), (116, -51), (119, -42), (114, -36), (105, -37), (96, -41), (88, -47),
            (87, -54), (90, -40), (87, -20), (78, 0)]
    part(neck, inner=[(36, -4), (20, -20)], w=W_MAIN)
    # avant proche : levée haut, avant-bras à l'horizontale, genou plié (devant l'encolure)
    leg([(52, 12, 12.5), (72, 10, 10.5), (94, 6, 9), (97, 22, 6.4), (93, 35, 7.2), (91, 41, 6)],
        [(40, -2), (52, -6)], W_MAIN)

    # oreilles
    part([(76, -88), (75, -98), (74, -108), (81, -99), (84, -90)], w=W_MID)
    part([(70, -85), (66, -95), (63, -104), (70, -96), (75, -88)], w=W_MID)
    # œil (amande) + arcade
    e = T((92, -73))
    svg.ellipse(e, 4.4 * scale, 2.4 * scale, W_FINE, rot=facing * 40)
    svg.circle(e, 1.3 * scale, W_HAIR, fill="#000")
    line([(86, -78), (92, -80), (98, -76)], W_HAIR)
    # naseau, bouche, ganache
    line([(112, -50), (115, -46), (112, -43)], W_FINE)
    line([(118, -38), (110, -37), (105, -39)], W_FINE)
    line([(86, -54), (92, -48), (100, -46)], W_HAIR)

    # ---------- Bride ----------
    line([(77, -86), (86, -68), (96, -54), (103, -42)], W_FINE)      # montant
    line([(106, -66), (102, -54), (98, -42)], W_FINE)                # muserolle
    line([(80, -84), (87, -86)], W_FINE)                             # frontal
    svg.circle(T((90, -61)), 3.0 * scale, W_FINE, fill="#fff")       # rosette
    svg.circle(T((90, -61)), 1.0 * scale, W_HAIR, fill="#000")
    svg.circle(T((105, -38)), 2.6 * scale, W_FINE, fill="#fff")      # anneau du mors
    line([(103, -37), (84, -30), (52, -36), (24, -44)], W_FINE)      # rênes

    # ---------- Crinière : mèches au-dessus de la crête, soufflées vers l'arrière ----------
    crest = [(26, -36), (36, -54), (50, -72), (64, -84), (74, -89)]

    def on_crest(t):
        k = t * (len(crest) - 1)
        i = min(int(k), len(crest) - 2)
        u = k - i
        a, b = crest[i], crest[i + 1]
        return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)

    n_locks = 7 if detail > 0.5 else 5
    for i in range(n_locks):          # de l'arrière (garrot) vers l'avant (nuque)
        t_back = i / n_locks
        t_front = (i + 1) / n_locks + 0.06
        a, b = on_crest(min(t_front, 1.0)), on_crest(max(t_back - 0.04, 0.0))
        ln = 13 + 3 * ((i * 7) % 3)
        tip = (b[0] - ln * 0.95, b[1] - ln * 0.25 + 4)
        shape = [(a[0] + 2, a[1] + 4), (a[0] - 3, a[1] - 6),
                 ((a[0] + tip[0]) / 2 + 2, (a[1] + tip[1]) / 2 - 7), (tip[0] - 2, tip[1] - 4), tip,
                 (tip[0] + 6, tip[1] + 1), ((b[0] + tip[0]) / 2 + 3, (b[1] + tip[1]) / 2 + 2), (b[0] + 2, b[1] + 6)]
        svg.path(catmull(P(shape)), W_MID, fill="#fff")
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        line([(mid[0] + 1, mid[1] - 1), ((mid[0] + tip[0]) / 2 + 2, (mid[1] + tip[1]) / 2 - 5), (tip[0] + 4, tip[1] - 2)], W_HAIR)
    # quelques mèches qui retombent sur l'encolure (côté visible)
    for (x, y) in ([(40, -54), (56, -74)] if detail > 0.5 else [(48, -64)]):
        svg.path(catmull(P([(x, y), (x - 3, y + 10), (x - 12, y + 18), (x - 6, y + 10), (x - 2, y + 2)])), W_FINE, fill="#fff")
    # toupet sur le front
    svg.path(catmull(P([(76, -90), (84, -88), (90, -80), (84, -82), (80, -76), (77, -84)])), W_FINE, fill="#fff")

    # ---------- Plastron décoré ----------
    line([(32, -32), (46, -18), (58, -2), (64, 10)], W_FINE)
    line([(28, -27), (41, -13), (52, 2), (57, 13)], W_FINE)
    for t in ([(38, -22), (49, -10), (58, 3)] if detail > 0.5 else [(49, -10)]):
        svg.circle(T(t), 2.6 * scale, W_FINE, fill="#fff")
    svg.path(catmull(P([(48, -7), (52, 2), (49, 10), (45, 2), (48, -7)])), W_FINE, fill="#fff")

    # ---------- Croupière ----------
    line([(-32, -32), (-50, -30), (-64, -22), (-72, -14)], W_FINE)
    if detail > 0.5:
        for t in [(-46, -30), (-59, -24)]:
            svg.circle(T(t), 2.0 * scale, W_HAIR, fill="#fff")

    # ---------- Selle + tapis décoré (d'après ref1 : bande de pierreries + languettes pointues) ----------
    # tapis : bord supérieur sous la selle, bord inférieur droit, languettes pointues en dessous
    blanket = [(-36, -30), (-38, -12), (-37, 6), (33, 6), (34, -12), (31, -32)]
    svg.path(catmull(P(blanket)) + " Z", W_MID, fill="#fff")
    # bande décorée : deux filets parallèles au bas du tapis + rang de pierreries
    line([(-37, -6), (0, -6), (33, -6)], W_FINE)
    line([(-37, 1), (0, 1), (33, 1)], W_FINE)
    for x in range(-31, 31, 8):
        svg.circle(T((x + 2, -2.5)), 1.7 * scale, W_FINE, fill="#fff")
    # languettes pointues (bordure festonnée en dents)
    tabs = []
    xs = list(range(-37, 34, 7))
    for x0, x1 in zip(xs, xs[1:]):
        tabs += [(x0, 6), ((x0 + x1) / 2, 15)]
    tabs.append((xs[-1], 6))
    svg.path("M" + " L".join(pt(T(p)) for p in tabs), W_FINE, fill="#fff")
    # galon en volute dans le haut du tapis
    line([(-30, -26), (-20, -16), (-6, -20), (8, -14), (22, -20), (27, -26)], W_FINE)
    # selle : troussequin, assise, pommeau
    seat = [(-32, -30), (-34, -40), (-30, -47), (-22, -42), (-10, -36), (6, -35), (16, -39), (23, -48),
            (29, -46), (29, -32)]
    svg.path(catmull(P(seat)), W_MAIN, fill="#fff")
    line([(-28, -34), (-12, -31), (6, -31), (22, -35)], W_HAIR)
    # étrivière + étrier
    line([(8, -30), (9, -8), (10, 14)], W_FINE)
    svg.path(catmull(P([(4, 14), (3, 22), (10, 25), (17, 22), (16, 14), (10, 13), (4, 14)])), W_FINE, fill="#fff")


# --------------------------------------------------------------------------
# Barre torsadée
# --------------------------------------------------------------------------
def twisted_pole(svg, x, y0, y1, r=5.0, pitch=13.0):
    svg.mask(f"M{f(x - r)},{f(y0)} L{f(x + r)},{f(y0)} L{f(x + r)},{f(y1)} L{f(x - r)},{f(y1)} Z")
    svg.path(f"M{f(x - r)},{f(y0)} L{f(x - r)},{f(y1)} M{f(x + r)},{f(y0)} L{f(x + r)},{f(y1)}", W_MID)
    y = y0 + 4
    while y < y1 - pitch * 0.6:
        svg.path(f"M{f(x - r)},{f(y)} C{f(x - r * 0.2)},{f(y + pitch * 0.15)} {f(x + r * 0.2)},{f(y + pitch * 0.55)} "
                 f"{f(x + r)},{f(y + pitch * 0.62)}", W_HAIR)
        y += pitch
    # bagues + pied
    for yy in (y0 + 2, y1 - 2):
        svg.ellipse((x, yy), r + 2.2, 2.2, W_FINE, fill="#fff")
    svg.path(f"M{f(x - r - 2)},{f(y1)} Q{f(x)},{f(y1 + 6)} {f(x + r + 2)},{f(y1)}", W_FINE)


# --------------------------------------------------------------------------
# Escalier en colimaçon
# --------------------------------------------------------------------------
def spiral_stairs(svg, y_top, y_bot, R=60, ry=13, turns=1.75, steps_per_turn=14, detail=1.0):
    """Escalier en colimaçon : fût central, marches en éventail, limon, rampe et palier haut."""
    total = int(turns * steps_per_turn)
    rise = (y_bot - y_top) / total
    pole_r = 6
    rail_h = 30
    d_th = 360.0 / steps_per_turn

    def hp(k, rad, dy=0.0):
        th = k * d_th + 90          # la marche du bas part du côté droit
        x, y = ell(0, rad, ry * rad / R, th)
        return (x, y + y_bot - k * rise + dy), math.cos(math.radians(th))

    def helix(rad, dy, front_only, w_front, w_back):
        prev = None
        for i in range(total * 8 + 1):
            k = i / 8
            p, c = hp(k, rad, dy)
            if prev is not None:
                if c > -0.02:
                    svg.path(f"M{pt(prev)} L{pt(p)}", w_front)
                elif not front_only:
                    svg.path(f"M{pt(prev)} L{pt(p)}", w_back)
            prev = p

    # 1) parties arrière (fines), 2) fût, 3) parties avant
    prev = None
    for i in range(total * 8 + 1):          # rampe arrière
        k = i / 8
        p, c = hp(k, R, -rail_h)
        if prev is not None and c <= -0.02:
            svg.path(f"M{pt(prev)} L{pt(p)}", W_HAIR)
        prev = p
    for k in range(total):                  # marches arrière : simple trait
        po, c = hp(k + 0.5, R)
        if c <= -0.02:
            pi_, _ = hp(k + 0.5, pole_r)
            svg.path(f"M{pt(pi_)} L{pt(po)}", W_HAIR)
    svg.mask(f"M{f(CX - pole_r)},{f(y_top)} L{f(CX + pole_r)},{f(y_top)} L{f(CX + pole_r)},{f(y_bot)} L{f(CX - pole_r)},{f(y_bot)} Z")
    svg.path(f"M{f(CX - pole_r)},{f(y_top)} L{f(CX - pole_r)},{f(y_bot)} M{f(CX + pole_r)},{f(y_top)} L{f(CX + pole_r)},{f(y_bot)}", W_MID)
    # marches avant : plateau en éventail (bord avant + contremarche)
    for k in range(total - 1, -1, -1):
        a0, a1 = k, k + 1
        p_in0, _ = hp(a0, pole_r)
        p_out0, c0 = hp(a0, R)
        p_out1, c1 = hp(a1, R)
        if c0 <= 0.0 and c1 <= 0.0:
            continue
        p_in1, _ = hp(a1, pole_r)
        top0 = (p_out0[0], p_out0[1] - rise)
        # plateau de la marche (au niveau k+1)
        tread = (f"M{pt((p_in0[0], p_in0[1] - rise))} L{pt(top0)} L{pt(p_out1)} L{pt(p_in1)} Z")
        svg.path(tread, W_FINE, fill="#fff")
        # contremarche (face verticale extérieure)
        svg.path(f"M{pt(p_out0)} L{pt(top0)}", W_FINE)
    helix(R, 0, True, W_FINE, None)          # limon
    for k in range(total + 1):               # balustres
        po, c = hp(k, R, -rise)
        if c > 0.0:
            tp, _ = hp(k, R, -rail_h)
            svg.path(f"M{pt(po)} L{pt(tp)}", W_HAIR)
    helix(R, -rail_h, True, W_MID, None)     # rampe avant


# --------------------------------------------------------------------------
# Scène complète
# --------------------------------------------------------------------------
def build(simple=False):
    W, H = 1000, 1060
    s = Svg(W, H)
    det = 0.0 if simple else 1.0

    # ---------------- Plateforme ----------------
    PCY, PRX, PRY, PTH = 880, 420, 52, 18
    s.path(f"M{pt((CX - PRX, PCY))} A{PRX},{PRY} 0 1 1 {pt((CX + PRX, PCY))} A{PRX},{PRY} 0 1 1 {pt((CX - PRX, PCY))} Z",
           W_MAIN, fill="#fff")
    s.path(f"M{pt((CX - PRX, PCY))} L{pt((CX - PRX, PCY + PTH))} A{PRX},{PRY} 0 0 0 {pt((CX + PRX, PCY + PTH))} "
           f"L{pt((CX + PRX, PCY))}", W_MAIN)
    if not simple:  # planches de la plateforme (lignes rayonnantes fines, seulement à l'avant)
        s.path(front_arc(PCY, PRX - 14, PRY - 2), W_HAIR)
        for th in range(-80, 81, 20):
            a, b = ell(PCY + PTH, PRX, PRY, th), ell(PCY, PRX, PRY, th)
            s.path(f"M{pt(b)} L{pt(a)}", W_HAIR)

    # ---------------- Escalier central ----------------
    spiral_stairs(s, 630, PCY + PRY * 0.2, R=62 if not simple else 58,
                  turns=1.75 if not simple else 1.5, steps_per_turn=14 if not simple else 12, detail=det)

    # ---------------- Chevaux ----------------
    if simple:
        horses = [(-38, 1.12), (38, 1.12)]
    else:
        horses = [(-64, 0.80), (64, 0.80), (-29, 0.92), (29, 0.92)]  # du fond vers l'avant
    HR = PRX * 0.86
    for th, sc in horses:
        bx, by = ell(PCY, HR, PRY * 0.86, th)
        top = ell(612, HR * 0.97, 44, th)[1] - 12
        oy = by - 132 * sc - (8 if abs(th) > 45 else 0)
        twisted_pole(s, bx, top, by, r=4.6 if not simple else 5.2)
        horse(s, bx, oy, scale=sc, facing=1 if th > 0 else -1, detail=det)
        # la barre ressort au-dessus de la selle (elle traverse le cheval)
        twisted_pole(s, bx, top, oy - 33 * sc, r=4.6 if not simple else 5.2)

    # ---------------- Second toit évasé (jupe) ----------------
    T1CY, T1RX, T1RY = 492, 266, 30
    T2CY, T2RX, T2RY = 598, 412, 46

    def skirt_curve(th):
        a, b = ell(T1CY, T1RX, T1RY, th), ell(T2CY, T2RX, T2RY, th)
        c = (a[0] + (b[0] - a[0]) * 0.22, a[1] + (b[1] - a[1]) * 0.78)
        return a, c, b

    aL, cL, bL = skirt_curve(-90)
    aR, cR, bR = skirt_curve(90)
    skirt = (f"M{pt(aL)} Q{pt(cL)} {pt(bL)} A{T2RX},{T2RY} 0 0 0 {pt(bR)} Q{pt(cR)} {pt(aR)} "
             f"A{T1RX},{T1RY} 0 0 1 {pt(aL)} Z")
    s.path(skirt, W_MAIN, fill="#fff")
    step = 7.5 if not simple else 15
    th = -90 + step
    while th < 90 - 1e-6:
        a, c, b = skirt_curve(th)
        s.path(f"M{pt(a)} Q{pt(c)} {pt(b)}", W_FINE if not simple else W_MID)
        th += step

    # bordure + lambrequin festonné + perles
    RIM = 9
    s.path(f"M{pt(bL)} L{pt((bL[0], bL[1] + RIM))} A{T2RX},{T2RY} 0 0 0 {pt((bR[0], bR[1] + RIM))} L{pt(bR)}",
           W_MID, fill="#fff")
    sc_step = 6 if not simple else 10
    pts = [ell(T2CY + RIM, T2RX, T2RY, a) for a in range(-90, 91, sc_step)]
    d = f"M{pt(pts[0])}"
    for p, q in zip(pts, pts[1:]):
        depth = 8 + 12 * (1 - abs((p[0] + q[0]) / 2 - CX) / T2RX) ** 0.5
        mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + depth * 1.9)
        d += f" Q{pt(mid)} {pt(q)}"
    s.path(d + " Z", W_MID, fill="#fff")
    for p in pts[1:-1]:
        s.circle((p[0], p[1] + 4), 2.6 if not simple else 3.2, W_FINE, fill="#fff")
    if not simple:
        for p, q in zip(pts, pts[1:]):
            m = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 - 4.5)
            s.circle(m, 1.5, W_HAIR, fill="#000")

    # ---------------- Bandeau décoratif ----------------
    BTOP, BBOT, BRX, BRY = 420, 492, 266, 30
    band = (f"M{pt((CX - BRX, BTOP))} L{pt((CX - BRX, BBOT))} A{BRX},{BRY} 0 0 0 {pt((CX + BRX, BBOT))} "
            f"L{pt((CX + BRX, BTOP))} A{BRX},{BRY} 0 0 1 {pt((CX - BRX, BTOP))} Z")
    # Le cône est dessiné avant le bandeau pour que le bandeau le recouvre.
    # ---------------- Cône rayé ----------------
    APEX = (CX, 168)
    CBY, CRX, CRY = 422, 252, 28

    def cone_curve(th):
        b = ell(CBY, CRX, CRY, th)
        m = ((APEX[0] + b[0]) / 2, (APEX[1] + b[1]) / 2)
        c = (m[0] - 24 * math.sin(math.radians(th)), m[1] + 10)
        return b, c

    bl, cl = cone_curve(-90)
    br, cr = cone_curve(90)
    s.path(f"M{pt(APEX)} Q{pt(cl)} {pt(bl)} A{CRX},{CRY} 0 0 0 {pt(br)} Q{pt(cr)} {pt(APEX)} Z", W_MAIN, fill="#fff")
    step = 9 if not simple else 18
    th = -90 + step
    while th < 90 - 1e-6:
        b, c = cone_curve(th)
        # démarre un peu sous la pointe pour éviter l'empâtement
        s.path(f"M{pt(ell(198, 19, 4, th))} Q{pt(c)} {pt(b)}", W_FINE if not simple else W_MID)
        th += step

    # lanterneau + mât + drapeau
    s.path(catmull([(CX - 20, 196), (CX - 13, 180), (CX - 6, 160), (CX - 7, 146)]) +
           " L" + pt((CX + 7, 146)) + " " + catmull_cont([(CX + 7, 146), (CX + 6, 160), (CX + 13, 180), (CX + 20, 196)]) +
           " Q" + pt((CX, 202)) + " " + pt((CX - 20, 196)) + " Z", W_MID, fill="#fff")
    for yy, rx in ((180, 13), (160, 7.5)):
        s.path(f"M{f(CX - rx)},{yy} Q{f(CX)},{yy + 3} {f(CX + rx)},{yy}", W_HAIR)
    s.ellipse((CX, 146), 9, 3, W_FINE, fill="#fff")
    s.circle((CX, 136), 6.5, W_MID, fill="#fff")              # boule
    s.path(f"M{f(CX)},{129.5} L{f(CX)},{52}", W_MID)        # mât
    s.circle((CX, 49), 3, W_FINE, fill="#fff")
    # drapeau flottant (fanion à deux pointes)
    flag = f"M{f(CX)},{56} C{f(CX + 22)},{48} {f(CX + 36)},{64} {f(CX + 60)},{55} L{f(CX + 50)},{69} L{f(CX + 62)},{82} " \
           f"C{f(CX + 38)},{90} {f(CX + 22)},{76} {f(CX)},{84} Z"
    s.path(flag, W_MID, fill="#fff")
    if not simple:
        s.path(f"M{f(CX + 6)},{70} C{f(CX + 22)},{63} {f(CX + 34)},{76} {f(CX + 48)},{69}", W_HAIR)

    # ---------------- Bandeau (dessus du cône) ----------------
    s.path(band, W_MAIN, fill="#fff")
    s.path(front_arc(BTOP + 8, BRX, BRY), W_FINE)          # moulure haute
    s.path(front_arc(BBOT - 8, BRX, BRY), W_FINE)          # moulure basse
    med_th = [-64, -32, 0, 32, 64] if not simple else [-50, 0, 50]
    mid_y = (BTOP + BBOT) / 2
    for th in med_th:
        c = ell(mid_y, BRX, BRY, th)
        k = math.cos(math.radians(th))
        rx, ry = 30 * k ** 0.9, 21
        # petite couronne au-dessus du médaillon (dépasse sur le cône)
        crown = [(c[0] - rx * 0.55, c[1] - ry - 1), (c[0] - rx * 0.3, c[1] - ry - 12), (c[0], c[1] - ry - 24),
                 (c[0] + rx * 0.3, c[1] - ry - 12), (c[0] + rx * 0.55, c[1] - ry - 1)]
        s.path(catmull(crown) + " Z", W_FINE, fill="#fff")
        s.circle((c[0], c[1] - ry - 11), 2.2, W_HAIR, fill="#fff")
        s.ellipse(c, rx + 4, ry + 4, W_MID, fill="#fff")
        s.ellipse(c, rx - 1, ry - 4, W_FINE)
        if not simple:  # petite étoile au centre
            s.circle(c, 1.6, W_HAIR, fill="#000")
    # pointes entre les médaillons
    pointe_th = [-80, -48, -16, 16, 48, 80] if not simple else [-78, -25, 25, 78]
    for th in pointe_th:
        b = ell(BTOP, BRX, BRY, th)
        k = math.cos(math.radians(th)) ** 0.5
        hw, hh = 5.5 * k + 1.5, 22
        d = (f"M{pt((b[0] - hw, b[1] + 1))} Q{pt((b[0] - hw * 0.4, b[1] - hh * 0.5))} {pt((b[0], b[1] - hh))} "
             f"Q{pt((b[0] + hw * 0.4, b[1] - hh * 0.5))} {pt((b[0] + hw, b[1] + 1))} Z")
        s.path(d, W_FINE, fill="#fff")
        s.circle((b[0], b[1] - hh - 3), 2.2, W_FINE, fill="#fff")
        if not simple:  # rosace sur le bandeau sous la pointe
            rb = ell(mid_y, BRX, BRY, th)
            s.circle(rb, 5 * k + 1.5, W_FINE, fill="#fff")
            s.circle(rb, 1.4, W_HAIR, fill="#000")

    # ---------------- Barrière ----------------
    GCY, GRX, GRY, GH = 900, 452, 64, 34
    a0, a1 = -84, 84
    s.path(front_arc(GCY - GH, GRX, GRY, a0, a1), W_MID)          # main courante
    if not simple:
        s.path(front_arc(GCY - GH + 6, GRX, GRY, a0, a1), W_HAIR)
    s.path(front_arc(GCY - 6, GRX, GRY, a0, a1), W_FINE)           # lisse basse
    post_step = 12 if not simple else 21
    th = a0
    while th <= a1 + 1e-6:
        b = ell(GCY, GRX, GRY, th)
        s.path(f"M{pt(b)} L{pt((b[0], b[1] - GH - 4))}", W_MID)
        s.circle((b[0], b[1] - GH - 6.5), 2.6, W_FINE, fill="#fff")
        th += post_step
    if not simple:  # barreaux fins entre les poteaux
        th = a0
        while th < a1 - 1e-6:
            for sub in (1 / 3, 2 / 3):
                t = th + post_step * sub
                b = ell(GCY - 6, GRX, GRY, t)
                s.path(f"M{pt(b)} L{pt((b[0], b[1] - GH + 6))}", W_HAIR)
            th += post_step
    return s


if __name__ == "__main__":
    build(False).save(OUT / "manege_lineart.svg")
    build(True).save(OUT / "manege_simplifie.svg")
    print("SVG générés dans", OUT)
