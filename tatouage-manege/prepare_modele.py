#!/usr/bin/env python3
"""
Prépare des modèles noir & blanc à partir de la photo du manège (manege.png),
pour servir de base à un dessin de tatouage en line art.
Rien n'est dessiné : toutes les images sont des traitements de la photo.

Sorties (dans ./modeles/) :
  manege_contraste.png            niveaux de gris contrastés, fond blanchi
  manege_seuil_1/2/3.png          noir pur / blanc pur, 3 seuils (clair -> foncé)
  manege_contours.png             contours épurés (Canny), traits noirs sur blanc
  manege_contours_adaptatif.png   variante (seuil adaptatif), traits plus « dessinés »
  manege_decalque.png             contours gris clair, pour repasser au stylet
  manege_masque.png               (contrôle) zone conservée
"""
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "modeles"
OUT.mkdir(exist_ok=True)

SCALE = 3  # 816 x 925 px -> 2448 x 2775 px

# ---------------------------------------------------------------------------
# 1. Recadrage : la photo est une capture d'écran de téléphone
# ---------------------------------------------------------------------------
shot = cv2.imread(str(HERE / "manege.png"))
photo = shot[200:1125, 0:816].copy()          # zone de l'image seule
H, W = photo.shape[:2]

# ---------------------------------------------------------------------------
# 2. Masque du manège (coordonnées dans la photo recadrée 816 x 925)
# ---------------------------------------------------------------------------
carousel_poly = np.array([
    (400, 18), (418, 18), (414, 95), (447, 150), (448, 172),     # flèche + couronne
    (555, 290), (650, 293), (700, 316), (730, 330), (733, 400),  # cône droit, bandeau
    (765, 430), (799, 458), (799, 502), (780, 522),              # toit évasé droit
    (702, 522), (702, 748), (816, 744), (816, 872),              # sous le toit, barrière
    (0, 872), (0, 762), (98, 762), (98, 522),                    # barrière gauche
    (0, 508), (0, 460), (40, 440), (68, 410), (66, 330),         # toit évasé gauche
    (100, 312), (165, 293), (242, 290), (362, 172), (362, 152),  # bandeau, cône gauche
    (396, 95),
], np.int32)

# zones à blanchir : personnes, icône Google Lens
erase_boxes = [
    (0, 680, 50, 875),      # homme à gauche
    (146, 688, 206, 868),   # homme veste sombre
    (640, 700, 692, 868),   # femme sac à dos
    (684, 692, 738, 868),   # femme manteau rouge
    (728, 695, 790, 775),   # femme à droite
]
lens_icon = ((68, 858), 40)

mask = np.full((H, W), cv2.GC_BGD, np.uint8)
cv2.fillPoly(mask, [carousel_poly], cv2.GC_PR_FGD)
# le toit (au-dessus du lambrequin) est sûr à 100 %
roof = np.zeros((H, W), np.uint8)
cv2.fillPoly(roof, [carousel_poly], 1)
roof[505:] = 0
hsv = cv2.cvtColor(photo, cv2.COLOR_BGR2HSV)
h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
sky = (h >= 95) & (h <= 130) & (s > 45)
mask[(roof == 1) & ~sky] = cv2.GC_FGD
mask[(roof == 1) & sky] = cv2.GC_BGD           # ciel visible entre les ornements du bandeau
# barrière : fait partie du manège, on la garde entière
fence = np.zeros((H, W), np.uint8)
cv2.fillPoly(fence, [carousel_poly], 1)
fence[:765] = 0
mask[fence == 1] = cv2.GC_FGD
# sous le toit : la végétation et le ciel visibles entre les barres -> probablement fond
vegetation = ((h >= 28) & (h <= 95) & (s > 40)) | ((h >= 95) & (h <= 130) & (s > 50))  # vert / bleu ciel
under = np.zeros((H, W), bool)
under[505:760] = True
mask[(mask == cv2.GC_PR_FGD) & vegetation & under] = cv2.GC_PR_BGD
# zones où l'on voit les arbres entre les barres (gauche et droite) : végétation = fond certain
trees = np.zeros((H, W), bool)
trees[505:655, 98:238] = True
trees[505:640, 590:702] = True
veg_wide = (h >= 30) & (h <= 135) & (s > 35)
mask[trees & veg_wide & (mask != cv2.GC_BGD)] = cv2.GC_BGD

bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
cv2.grabCut(photo, mask, None, bgd, fgd, 6, cv2.GC_INIT_WITH_MASK)
keep = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

for x0, y0, x1, y1 in erase_boxes:
    keep[y0:y1, x0:x1] = 0
cv2.circle(keep, *lens_icon, 0, -1)
# nettoyage : on supprime les petits îlots, on bouche les petits trous
keep = cv2.morphologyEx(keep, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
n, lab, stats, _ = cv2.connectedComponentsWithStats(keep, 8)
clean = np.zeros_like(keep)
for i in range(1, n):
    if stats[i, cv2.CC_STAT_AREA] >= 150:
        clean[lab == i] = 255
keep = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

# ---------------------------------------------------------------------------
# 3. Agrandissement haute résolution + débruitage doux
# ---------------------------------------------------------------------------
big = cv2.resize(photo, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LANCZOS4)
big = cv2.bilateralFilter(big, 9, 40, 9)
keep_big = cv2.resize(keep, (big.shape[1], big.shape[0]), interpolation=cv2.INTER_CUBIC)
keep_big = np.where(cv2.GaussianBlur(keep_big, (0, 0), 5) > 127, 255, 0).astype(np.uint8)  # bord lissé
soft = cv2.GaussianBlur(keep_big, (0, 0), 2.5).astype(np.float32) / 255.0   # bord adouci
hard = keep_big > 127
cv2.imwrite(str(OUT / "manege_masque.png"), keep_big)

gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

# ---------------------------------------------------------------------------
# Version 1 : noir & blanc contrasté, fond blanc
# ---------------------------------------------------------------------------
clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
g = clahe.apply(gray).astype(np.float32) / 255.0
lo, hi = np.percentile(g[hard], [2, 98])
g = np.clip((g - lo) / (hi - lo), 0, 1)
g = g ** 1.15                                   # assombrit légèrement les tons moyens
contrast = (g * soft + 1.0 * (1 - soft)) * 255
contrast = contrast.clip(0, 255).astype(np.uint8)
cv2.imwrite(str(OUT / "manege_contraste.png"), contrast)

# ---------------------------------------------------------------------------
# Version 2 : seuillage noir pur / blanc pur, 3 niveaux
# ---------------------------------------------------------------------------
smooth = cv2.GaussianBlur(contrast, (0, 0), 1.6)


def despeckle(bw_black, min_area):
    """bw_black : 255 = noir. Supprime les taches noires et les trous blancs trop petits."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw_black, 8)
    out = np.zeros_like(bw_black)
    big_ids = np.where(stats[1:, cv2.CC_STAT_AREA] >= min_area)[0] + 1
    out[np.isin(lab, big_ids)] = 255
    inv = 255 - out
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
    small = np.where(stats[1:, cv2.CC_STAT_AREA] < min_area // 2)[0] + 1
    out[np.isin(lab, small)] = 255
    return out


otsu, _ = cv2.threshold(smooth[hard], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
SEUILS = [("1", otsu - 22), ("2", otsu), ("3", otsu + 22)]   # 1 = plus clair, 3 = plus foncé
for name, t in SEUILS:
    black = np.where((smooth < t) & hard, 255, 0).astype(np.uint8)
    black = despeckle(black, 60)
    cv2.imwrite(str(OUT / f"manege_seuil_{name}.png"), 255 - black)
print("seuils utilisés :", [int(t) for _, t in SEUILS])

# ---------------------------------------------------------------------------
# Version 3 : contours épurés
# ---------------------------------------------------------------------------
ROOF_Y = 520 * SCALE  # sous le toit, l'image est très chargée : on ne garde que les traits plus longs
inner = cv2.erode(keep_big, np.ones((9, 9), np.uint8)) > 0   # évite le liseré du masque


def keep_long_strokes(edges, min_len):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(edges, 8)
    out = np.zeros_like(edges)
    w, h = stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]
    ok = np.where((stats[1:, cv2.CC_STAT_AREA] >= min_len) | (np.maximum(w, h) >= min_len))[0] + 1
    out[np.isin(lab, ok)] = 255
    return out




def filter_by_zone(edges, min_top, min_bottom):
    top, bottom = edges.copy(), edges.copy()
    top[ROOF_Y:] = 0
    bottom[:ROOF_Y] = 0
    return cv2.bitwise_or(keep_long_strokes(top, min_top), keep_long_strokes(bottom, min_bottom))


# a) Canny sur image très lissée (garde les formes, oublie la texture)
base = cv2.bilateralFilter(clahe.apply(gray), 11, 50, 11)
base = cv2.GaussianBlur(base, (0, 0), 1.6)
edges = cv2.Canny(base, 40, 100, L2gradient=True)
edges[~inner] = 0
# silhouette du manège : on ajoute le contour extérieur du masque
# (seulement le contour extérieur, et seulement pour le toit : plus bas le bord du masque est artificiel)
outline = np.zeros_like(keep_big)
cnts, _ = cv2.findContours(keep_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(outline, cnts, -1, 255, 2)
outline[ROOF_Y - 30 * SCALE:] = 0
# pas de trait autour des zones blanchies (personnes, icône) ni sur le bord de l'image
for x0, y0, x1, y1 in erase_boxes:
    outline[max(y0 * SCALE - 12, 0):y1 * SCALE + 12, max(x0 * SCALE - 12, 0):x1 * SCALE + 12] = 0
(cx_, cy_), r_ = lens_icon
cv2.circle(outline, (cx_ * SCALE, cy_ * SCALE), (r_ + 5) * SCALE, 0, -1)
outline[:, :8] = 0
outline[:, -8:] = 0
outline[-8:, :] = 0
edges = cv2.bitwise_or(edges, outline)
edges = filter_by_zone(edges, 60, 110)
edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
cv2.imwrite(str(OUT / "manege_contours.png"), 255 - edges)

# b) Seuil adaptatif : traits plus pleins, effet « dessin à l'encre »
ad_src = cv2.GaussianBlur(cv2.bilateralFilter(gray, 11, 70, 11), (0, 0), 2.2)
ad = cv2.adaptiveThreshold(ad_src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 16)
ad[~inner] = 0
ad = cv2.morphologyEx(ad, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
ad = filter_by_zone(ad, 140, 230)
ad = cv2.bitwise_or(ad, cv2.dilate(outline, np.ones((2, 2), np.uint8)))
cv2.imwrite(str(OUT / "manege_contours_adaptatif.png"), 255 - ad)

# ---------------------------------------------------------------------------
# Version 4 : décalque gris clair
# ---------------------------------------------------------------------------
LIGHT = 170  # gris très clair (0 = noir, 255 = blanc)
trace = cv2.GaussianBlur(edges, (0, 0), 0.8).astype(np.float32) / 255.0
decal = 255 - trace * (255 - LIGHT)
cv2.imwrite(str(OUT / "manege_decalque.png"), decal.clip(0, 255).astype(np.uint8))

print("Images écrites dans", OUT, "-", big.shape[1], "x", big.shape[0], "px")
