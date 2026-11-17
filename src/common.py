from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"

GRID = 5                      # 5 x 5 spatial tile grid per patient
K_TILES = GRID * GRID         # 25 tiles per bag
TILE_PX = 64                  # tile edge length in pixels (kept small for a light repo)
CLASS_NAMES = ["Low-Grade / Early HCC", "High-Grade / Aggressive HCC"]

# 60 liver-relevant genes grouped into biological programmes.
# `direction` is the sign of the effect modelled for aggressive (high-grade) HCC.
GENE_PROGRAMS: dict[str, dict] = {
    "Proliferation": dict(direction=+1, genes=["MKI67", "TOP2A", "CCNB1", "CDK1", "PCNA", "AURKA", "BIRC5"]),
    "HCC / progenitor markers": dict(direction=+1, genes=["AFP", "GPC3", "EPCAM", "KRT19", "SALL4", "DKK1", "SPINK1", "CD24"]),
    "Wnt / beta-catenin": dict(direction=+1, genes=["CTNNB1", "GLUL", "AXIN1"]),
    "Fibrosis / stellate cell": dict(direction=+1, genes=["COL1A1", "COL1A2", "COL3A1", "ACTA2", "TIMP1", "LOX", "TGFB1", "SPARC"]),
    "Inflammation": dict(direction=+1, genes=["IL6", "TNF", "IL1B", "CXCL8", "CCL2", "STAT3", "CD68"]),
    "Angiogenesis": dict(direction=+1, genes=["VEGFA", "KDR", "ANGPT2"]),
    "Hepatocyte function": dict(direction=-1, genes=["ALB", "TTR", "APOA1", "APOB", "CYP3A4", "CYP2E1", "HNF4A", "SERPINA1", "TF", "CPS1"]),
    "Lipid metabolism / steatosis": dict(direction=0, genes=["PPARG", "SREBF1", "FASN", "SCD", "CD36", "PNPLA3", "ACACA"]),
    "Tumour suppressors / drivers": dict(direction=+1, genes=["TP53", "MYC", "PTEN", "CDKN2A", "TERT", "ARID1A", "NFE2L2"]),
}
GENES: list[str] = [g for p in GENE_PROGRAMS.values() for g in p["genes"]]
GENE_TO_PROGRAM: dict[str, str] = {g: n for n, p in GENE_PROGRAMS.items() for g in p["genes"]}
assert len(GENES) == 60 and len(set(GENES)) == 60


def load_data(data_dir: Path = DATA_DIR) -> dict:
    """Load tiles, tile-level atypia (ground truth of the synthetic cohort), clinical + RNA-seq."""
    z = np.load(data_dir / "tiles.npz")
    clin = pd.read_csv(data_dir / "clinical.csv")
    counts = pd.read_csv(data_dir / "expression_counts.csv", index_col=0)  # genes x patients
    counts = counts.loc[GENES, clin["patient_id"]]
    lib = clin["library_size"].to_numpy(dtype=np.float64)
    log2cpm = np.log2(counts.to_numpy(dtype=np.float64).T / lib[:, None] * 1e6 + 1.0)  # patients x genes
    return dict(
        tiles=z["tiles"],                       # (N, 25, S, S, 3) uint8
        atypia=z["atypia"].astype(np.float32),  # (N, 25) in [0, 1]
        rna=log2cpm.astype(np.float32),         # (N, 60) log2(CPM+1)
        y=clin["label"].to_numpy(dtype=np.int64),
        clinical=clin,
        genes=list(GENES),
    )


def stratified_split(y: np.ndarray, seed: int, val_frac: float = 0.15, test_frac: float = 0.25):
    """Stratified train / val / test index split."""
    idx = np.arange(len(y))
    trainval, test = train_test_split(idx, test_size=test_frac, stratify=y, random_state=seed)
    train, val = train_test_split(trainval, test_size=val_frac / (1 - test_frac),
                                  stratify=y[trainval], random_state=seed)
    return np.sort(train), np.sort(val), np.sort(test)


def fit_scaler(rna_train: np.ndarray) -> dict:
    return dict(mean=rna_train.mean(0).astype(np.float32), std=(rna_train.std(0) + 1e-6).astype(np.float32))


def apply_scaler(rna: np.ndarray, sc: dict) -> np.ndarray:
    return ((rna - sc["mean"]) / sc["std"]).astype(np.float32)


def tiles_to_tensor(tiles_u8: np.ndarray):
    """(B, K, S, S, 3) uint8 -> (B, K, 3, S, S) float, ImageNet-normalised."""
    import torch
    x = torch.from_numpy(np.ascontiguousarray(tiles_u8)).float().div_(255.0).permute(0, 1, 4, 2, 3)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    return (x - mean) / std


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
