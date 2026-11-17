from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

from .common import (DATA_DIR, GENE_PROGRAMS, GENES, GRID, K_TILES, TILE_PX, save_json)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def render_tile(rng: np.random.Generator, size: int, atypia: float, steat: float, fib: float) -> np.ndarray:
    """Synthetic H&E-like tile: eosin background, haematoxylin nuclei, lipid vacuoles, collagen bands."""
    sc = size / 64.0
    # eosin-pink background with smooth texture
    base = np.array([226, 168, 198], dtype=np.float32)
    low = rng.normal(0, 1, (8, 8, 3)).astype(np.float32)
    tex = np.asarray(Image.fromarray(((low - low.min()) / (np.ptp(low) + 1e-6) * 255).astype(np.uint8))
                     .resize((size, size), Image.BICUBIC), dtype=np.float32) / 255.0 - 0.5
    img = np.clip(base[None, None] + tex * 28.0, 0, 255)
    im = Image.fromarray(img.astype(np.uint8))
    d = ImageDraw.Draw(im)

    # fibrosis: pale, wavy collagen bands
    for _ in range(int(round(fib * 4))):
        y0 = rng.uniform(0, size)
        amp, fr, ph = rng.uniform(2, 6) * sc, rng.uniform(0.05, 0.12) / sc, rng.uniform(0, 6.28)
        pts = [(x, y0 + amp * np.sin(fr * x + ph)) for x in range(0, size + 4, 4)]
        d.line(pts, fill=(246, 222, 232), width=max(2, int(rng.integers(3, 6) * sc)))

    # steatosis: white lipid vacuoles with thin membrane
    for _ in range(int(round(steat * 14))):
        r = rng.uniform(3, 8) * sc
        cx, cy = rng.uniform(0, size, 2)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(250, 246, 248), outline=(214, 150, 184))

    # nuclei: normal = few, small, pale; atypical = many, large, pleomorphic, hyperchromatic
    n_nuc = int(round(10 + 22 * atypia + rng.normal(0, 1.5)))
    for _ in range(max(n_nuc, 4)):
        rx = (2.4 + 3.6 * atypia * rng.uniform(0.5, 1.6)) * sc
        ry = rx * rng.uniform(0.7, 1.3)
        cx, cy = rng.uniform(0, size, 2)
        dark = 1.0 - 0.55 * atypia
        col = (int(95 * dark + 20), int(60 * dark + 10), int(150 * dark + 40))
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=col)

    im = im.filter(ImageFilter.GaussianBlur(0.6 * sc))
    arr = np.asarray(im, dtype=np.float32) + rng.normal(0, 4.0, (size, size, 3))
    return np.clip(arr, 0, 255).astype(np.uint8)


def make_cohort(n: int, size: int, seed: int):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[rng.permutation(n)[: int(round(0.45 * n))]] = 1       # ~45 % high grade

    sgn = 2 * y - 1
    m = 0.75 * sgn + rng.normal(0, 1, n)                      # molecular aggressiveness
    a = 0.7 * sgn + rng.normal(0, 1, n)                      # morphological atypia
    s = np.clip(rng.beta(2, 3, n) + 0.05 * sgn, 0, 1)        # steatosis
    f = np.clip(rng.beta(2, 3, n) + 0.10 * sgn, 0, 1)        # fibrosis
    inflam = rng.normal(0.15 * sgn, 1, n)                    # inflammation

    # ---- tiles --------------------------------------------------------------------------
    gy, gx = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    tiles = np.zeros((n, K_TILES, size, size, 3), dtype=np.uint8)
    atypia = np.zeros((n, K_TILES), dtype=np.float32)
    for p in range(n):
        cy, cx = rng.uniform(0.5, GRID - 1.5, 2)
        sigma = rng.uniform(0.8, 1.5)
        amp = _sigmoid(1.4 * a[p])      # focus intensity follows the latent atypia only (no direct label leak)
        blob = np.exp(-((gy - cy) ** 2 + (gx - cx) ** 2) / (2 * sigma ** 2)).ravel()
        at = np.clip(0.08 + amp * blob + rng.normal(0, 0.05, K_TILES), 0, 1)
        atypia[p] = at
        for k in range(K_TILES):
            st = np.clip(s[p] + rng.normal(0, 0.12), 0, 1)
            fb = np.clip(f[p] + rng.normal(0, 0.12), 0, 1)
            tiles[p, k] = render_tile(rng, size, at[k], st, fb)

    # ---- RNA-seq (log2 CPM latent -> Poisson counts) ----------------------------------
    G = len(GENES)
    base = rng.uniform(3.5, 9.5, G)
    load_m = np.zeros(G); load_f = np.zeros(G); load_s = np.zeros(G); load_i = np.zeros(G)
    for prog, spec in GENE_PROGRAMS.items():
        for g in spec["genes"]:
            j = GENES.index(g)
            if prog in ("Proliferation", "HCC / progenitor markers"):
                load_m[j] = spec["direction"] * rng.uniform(0.35, 0.65)
            elif prog in ("Wnt / beta-catenin", "Angiogenesis", "Tumour suppressors / drivers"):
                load_m[j] = spec["direction"] * rng.uniform(0.10, 0.30)
            elif prog == "Hepatocyte function":
                load_m[j] = spec["direction"] * rng.uniform(0.20, 0.45)
            if prog == "Fibrosis / stellate cell":
                load_f[j] = rng.uniform(0.5, 0.9); load_m[j] = rng.uniform(0.0, 0.12)
            if prog == "Inflammation":
                load_i[j] = rng.uniform(0.4, 0.8)
            if prog == "Lipid metabolism / steatosis":
                load_s[j] = rng.uniform(0.6, 1.2) * (-1 if g in ("PPARG",) else 1)
    # AFP / GPC3 are the strongest canonical HCC markers
    for g, w in (("AFP", 0.95), ("GPC3", 0.85), ("MKI67", 0.8)):
        load_m[GENES.index(g)] = w
    L = (base[None] + np.outer(m, load_m) + np.outer(f - f.mean(), load_f * 3)
         + np.outer(s - s.mean(), load_s * 3) + np.outer(inflam, load_i)
         + rng.normal(0, 0.8, (n, G)))
    lib = rng.lognormal(np.log(2.0e7), 0.25, n)                      # simulated library size
    counts = rng.poisson(2.0 ** L * lib[:, None] / 1e6).T               # genes x patients

    # ---- clinical metadata (NOT used as model input) -----------------------------------
    pid = [f"HF-{i + 1:03d}" for i in range(n)]
    etiology = np.where(s > 0.5, "MASH", rng.choice(["HBV", "HCV", "ALD", "MASH"], n, p=[0.3, 0.25, 0.25, 0.2]))
    clin = pd.DataFrame(dict(
        patient_id=pid, label=y,
        grade_label=np.where(y == 1, "High-Grade / Aggressive HCC", "Low-Grade / Early HCC"),
        age=np.clip(rng.normal(63 + 2 * y, 9, n), 35, 88).round().astype(int),
        sex=rng.choice(["M", "F"], n, p=[0.72, 0.28]),
        etiology=etiology,
        steatosis_pct=(s * 60).round(0).astype(int),
        fibrosis_ishak=np.clip((f * 7 + rng.normal(0, 0.6, n)).round(), 0, 6).astype(int),
        afp_ng_ml=np.exp(2.0 + 1.1 * m + rng.normal(0, 0.8, n)).round(1),
        library_size=lib.round(0).astype(np.int64),
    ))
    return tiles, atypia, counts, clin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--size", type=int, default=TILE_PX)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    tiles, atypia, counts, clin = make_cohort(args.n, args.size, args.seed)
    np.savez_compressed(DATA_DIR / "tiles.npz", tiles=tiles, atypia=atypia)
    clin.to_csv(DATA_DIR / "clinical.csv", index=False)
    pd.DataFrame(counts, index=GENES, columns=clin["patient_id"]).rename_axis("gene").to_csv(DATA_DIR / "expression_counts.csv")
    pd.DataFrame([(g, p, GENE_PROGRAMS[p]["direction"]) for p in GENE_PROGRAMS for g in GENE_PROGRAMS[p]["genes"]],
                 columns=["gene", "programme", "expected_direction_in_high_grade"]).to_csv(DATA_DIR / "gene_annotations.csv", index=False)
    save_json(dict(n=args.n, tiles_per_patient=K_TILES, grid=GRID, tile_px=args.size, genes=len(GENES), seed=args.seed,
                   synthetic=True), DATA_DIR / "cohort_meta.json")
    print(f"patients={args.n}  tiles={tiles.shape}  high-grade={int(clin.label.sum())}  genes={len(GENES)}")


if __name__ == "__main__":
    main()
