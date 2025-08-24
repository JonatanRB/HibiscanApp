import argparse
import json
import os
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import cv2
import numpy as np
from skimage.segmentation import watershed
from scipy import ndimage as ndi  # viene con scipy; si no lo tienes: pip install scipy
from PIL import Image

# -----------------------------
# Utilidades
# -----------------------------

def read_image(path: str) -> np.ndarray:
    # Soporta rutas con caracteres extraños y grandes tamaños
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        # fallback a cv2.imread si falla
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    return img

def to_hsv(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

def ensure_uint8(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8) * 255

def auto_area_bounds(img_shape: Tuple[int, int], scale: float = 0.0002, max_scale: float = 0.05) -> Tuple[int, int]:
    """Heurística: define área mínima y máxima esperada con base al tamaño de la imagen."""
    h, w = img_shape[:2]
    total = h * w
    min_area = max(20, int(total * scale))
    max_area = max(min_area + 50, int(total * max_scale))
    return min_area, max_area

@dataclass
class HSVRanges:
    lower1: Tuple[int, int, int]
    upper1: Tuple[int, int, int]
    lower2: Tuple[int, int, int]
    upper2: Tuple[int, int, int]

def default_red_ranges() -> HSVRanges:
    # OpenCV: H:0-179, S:0-255, V:0-255
    return HSVRanges(
        lower1=(0, 90, 60),
        upper1=(10, 255, 255),
        lower2=(165, 90, 60),
        upper2=(179, 255, 255),
    )

def build_mask(hsv: np.ndarray, ranges: HSVRanges) -> np.ndarray:
    m1 = cv2.inRange(hsv, np.array(ranges.lower1), np.array(ranges.upper1))
    m2 = cv2.inRange(hsv, np.array(ranges.lower2), np.array(ranges.upper2))
    mask = cv2.bitwise_or(m1, m2)
    # Limpieza básica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask

def split_touching(mask: np.ndarray) -> np.ndarray:
    """Separa objetos pegados usando distancia + watershed."""
    # Compute sure background
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(mask, kernel, iterations=2)

    # Distancia
    dist = cv2.distanceTransform(ensure_uint8(mask), cv2.DIST_L2, 5)
    # Normaliza y umbral para "sure foreground"
    if dist.max() > 0:
        dist_norm = dist / dist.max()
    else:
        dist_norm = dist
    sure_fg = (dist_norm > 0.35).astype(np.uint8)

    # Marcadores
    unknown = cv2.subtract(ensure_uint8(sure_bg), ensure_uint8(sure_fg))
    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed requiere una imagen 3 canales
    rgb = cv2.cvtColor(ensure_uint8(mask), cv2.COLOR_GRAY2BGR)
    markers = watershed(-dist, markers, mask=(mask > 0))

    # Vuelve a máscara por objeto
    return markers.astype(np.int32)

def contour_features(cnt: np.ndarray) -> Dict[str, Any]:
    area = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, True)
    if perim == 0:
        circ = 0
    else:
        circ = 4 * np.pi * area / (perim * perim)
    x, y, w, h = cv2.boundingRect(cnt)
    extent = area / (w * h + 1e-6)
    aspect = w / (h + 1e-6) if h > 0 else 0
    return {"area": area, "perim": perim, "circularity": circ, "extent": extent, "aspect": aspect}

def count_objects(mask: np.ndarray, img: np.ndarray, min_area: int, max_area: int) -> Tuple[int, np.ndarray, list]:
    """Cuenta objetos válidos y retorna imagen anotada + features por objeto."""
    # Separa objetos pegados
    markers = split_touching(mask)

    # Construye contornos por label
    annotated = img.copy()
    valid = 0
    feats = []
    unique_labels = [l for l in np.unique(markers) if l > 1]  # 0 fondo, 1 ruido inicial
    for label in unique_labels:
        obj_mask = (markers == label).astype(np.uint8) * 255
        # Contorno
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        f = contour_features(cnt)
        area = f["area"]
        circ = f["circularity"]
        extent = f["extent"]

        # Reglas simples para filtrar frutos plausibles
        if area < min_area or area > max_area:
            color = (0, 0, 255)  # rojo para descartado
            cv2.drawContours(annotated, [cnt], -1, color, 2)
            continue
        if circ < 0.3:  # muy irregular
            color = (0, 0, 255)
            cv2.drawContours(annotated, [cnt], -1, color, 2)
            continue
        if extent < 0.25:  # muy hueco/filamentoso
            color = (0, 0, 255)
            cv2.drawContours(annotated, [cnt], -1, color, 2)
            continue

        valid += 1
        feats.append(f)
        (x, y, w, h) = cv2.boundingRect(cnt)
        color = (0, 255, 0)  # verde válido
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(annotated, f"#{valid}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    cv2.putText(annotated, f"Conteo: {valid}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return valid, annotated, feats

# -----------------------------
# Algoritmo Genético para ajustar HSV por imagen
# -----------------------------

class SimpleGA:
    def __init__(self, pop_size=20, generations=15, elite=2, mutation_prob=0.3):
        self.pop_size = pop_size
        self.generations = generations
        self.elite = elite
        self.mutation_prob = mutation_prob
        rng = np.random.default_rng()
        self.rng = rng

    def random_individual(self):
        # h1 [0,15], h2 [160,179]; s_low [60,140], v_low [40,140], s_up [200,255], v_up [200,255]
        h1_lo = int(self.rng.integers(0, 8))
        h1_up = int(self.rng.integers(h1_lo + 6, 18))
        h2_lo = int(self.rng.integers(160, 173))
        h2_up = int(self.rng.integers(h2_lo + 6, 179))

        s_lo = int(self.rng.integers(60, 140))
        v_lo = int(self.rng.integers(40, 140))
        s_up = int(self.rng.integers(200, 256))
        v_up = int(self.rng.integers(200, 256))
        return np.array([h1_lo, h1_up, h2_lo, h2_up, s_lo, v_lo, s_up, v_up], dtype=int)

    def mutate(self, ind):
        idx = self.rng.integers(0, len(ind))
        jitter = int(self.rng.integers(-6, 7))
        new = ind.copy()
        new[idx] = np.clip(new[idx] + jitter, 0, 255)
        # re-orden en rangos si rompe
        new[1] = max(new[1], new[0] + 3)
        new[3] = max(new[3], new[2] + 3)
        new[1] = min(new[1], 179)
        new[3] = min(new[3], 179)
        return new

    def crossover(self, a, b):
        point = self.rng.integers(1, len(a)-1)
        child1 = np.concatenate([a[:point], b[point:]])
        child2 = np.concatenate([b[:point], a[point:]])
        return child1, child2

    def fitness(self, ind, hsv, img_shape):
        ranges = HSVRanges(
            lower1=(int(ind[0]), int(ind[4]), int(ind[5])),
            upper1=(int(ind[1]), int(ind[6]), int(ind[7])),
            lower2=(int(ind[2]), int(ind[4]), int(ind[5])),
            upper2=(int(ind[3]), int(ind[6]), int(ind[7])),
        )
        mask = build_mask(hsv, ranges)
        frac = mask.mean() / 255.0  # fracción de pixeles rojos
        min_area, max_area = auto_area_bounds(img_shape)

        # Conteo rápido mediante CC y filtros aproximados (sin watershed para velocidad)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([])
        # candidatos
        candidates = ((areas >= min_area) & (areas <= max_area)).sum()

        # penaliza exceso/defecto de máscara y ruido
        noise = max(0, len(areas) - candidates)
        # target de fracción en [0.03, 0.45] (heursística)
        target_low, target_high = 0.03, 0.45
        penalty_frac = 0
        if frac < target_low:
            penalty_frac = (target_low - frac) * 20
        elif frac > target_high:
            penalty_frac = (frac - target_high) * 20

        fitness = candidates - 0.2 * noise - penalty_frac
        return float(fitness), mask

    def optimize(self, hsv, img_shape):
        pop = [self.random_individual() for _ in range(self.pop_size)]
        best = None
        best_fit = -1e9
        best_mask = None

        for _ in range(self.generations):
            scored = []
            for ind in pop:
                fit, _ = self.fitness(ind, hsv, img_shape)
                scored.append((fit, ind))
                if fit > best_fit:
                    best_fit = fit
                    best = ind.copy()
            # selección por torneo
            scored.sort(key=lambda x: x[0], reverse=True)
            next_pop = [s[1] for s in scored[:self.elite]]  # elitismo
            while len(next_pop) < self.pop_size:
                a = scored[self.rng.integers(0, len(scored)//2)][1]
                b = scored[self.rng.integers(0, len(scored)//2)][1]
                c1, c2 = self.crossover(a, b)
                if self.rng.random() < self.mutation_prob:
                    c1 = self.mutate(c1)
                if self.rng.random() < self.mutation_prob:
                    c2 = self.mutate(c2)
                next_pop.extend([c1, c2])
            pop = next_pop[:self.pop_size]

        # máscara final con el mejor umbral
        _, best_mask = self.fitness(best, hsv, img_shape)
        ranges = HSVRanges(
            lower1=(int(best[0]), int(best[4]), int(best[5])),
            upper1=(int(best[1]), int(best[6]), int(best[7])),
            lower2=(int(best[2]), int(best[4]), int(best[5])),
            upper2=(int(best[3]), int(best[6]), int(best[7])),
        )
        return ranges, best_mask

# -----------------------------
# Pipeline principal
# -----------------------------

def process_image(image_path: str, optimize: bool = False, save_json: bool = False) -> Dict[str, Any]:
    bgr = read_image(image_path)
    hsv = to_hsv(bgr)

    if optimize:
        ga = SimpleGA(pop_size=24, generations=18, elite=3, mutation_prob=0.35)
        ranges, mask = ga.optimize(hsv, bgr.shape[:2])
    else:
        ranges = default_red_ranges()
        mask = build_mask(hsv, ranges)

    min_area, max_area = auto_area_bounds(bgr.shape[:2])
    count, annotated, feats = count_objects(mask, bgr, min_area, max_area)

    out_path = os.path.splitext(image_path)[0] + "_annotated.png"
    # Guardar respetando rutas unicode
    cv2.imencode(".png", annotated)[1].tofile(out_path)

    result = {
        "image": image_path,
        "output_image": out_path,
        "count": int(count),
        "min_area": int(min_area),
        "max_area": int(max_area),
        "hsv_ranges": {
            "lower1": ranges.lower1, "upper1": ranges.upper1,
            "lower2": ranges.lower2, "upper2": ranges.upper2,
        },
        "features_sample": feats[:10],
    }

    if save_json:
        json_path = os.path.splitext(image_path)[0] + "_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        result["metrics_json"] = json_path

    return result

def main():
    ap = argparse.ArgumentParser(description="Contador de frutos de jamaica por color rojo con GA opcional.")
    ap.add_argument("--image", required=True, help="Ruta a la imagen de entrada (JPG/PNG).")
    ap.add_argument("--optimize", action="store_true", help="Ajustar HSV con Algoritmo Genético por imagen.")
    ap.add_argument("--save-json", action="store_true", help="Guardar métricas en JSON.")
    args = ap.parse_args()

    result = process_image(args.image, optimize=args.optimize, save_json=args.save_json)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
