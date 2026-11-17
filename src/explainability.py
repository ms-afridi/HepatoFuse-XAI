from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from .common import (ART_DIR, GENE_TO_PROGRAM, GENES, GRID, apply_scaler, tiles_to_tensor)
from .models import build_model


# ----------------------------------------------------------------------------- loading
def load_checkpoint(name: str, art_dir: Path = ART_DIR, device: str = "cpu"):
    ck = torch.load(Path(art_dir) / f"{name}.pt", map_location=device, weights_only=False)
    model = build_model(name, n_genes=len(ck["genes"])).to(device)
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck


# ----------------------------------------------------------------------------- spatial attention
@torch.no_grad()
def tile_attention(model, tiles_u8: np.ndarray) -> np.ndarray:
    """tiles_u8: (K, S, S, 3) of one patient -> attention (K,) summing to 1 (image branch only)."""
    x = tiles_to_tensor(tiles_u8[None])
    _, a = model.encode_image(x)
    return a[0].cpu().numpy()


def attention_grid(attn: np.ndarray, normalise: bool = True) -> np.ndarray:
    g = np.asarray(attn, dtype=np.float32).reshape(GRID, GRID)    # row-major tile order = 5x5 grid
    if normalise:
        g = (g - g.min()) / (np.ptp(g) + 1e-9)
    return g


def make_mosaic(tiles_u8: np.ndarray) -> np.ndarray:
    """(K, S, S, 3) -> (GRID*S, GRID*S, 3) tissue mosaic in grid order."""
    k, s = tiles_u8.shape[:2]
    return tiles_u8.reshape(GRID, GRID, s, s, 3).transpose(0, 2, 1, 3, 4).reshape(GRID * s, GRID * s, 3)


def _colormap(v: np.ndarray) -> np.ndarray:
    """Dependency-free 'inferno-like' colormap, v in [0,1] -> RGB uint8."""
    stops = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    cols = np.array([[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]], dtype=np.float32)
    return np.stack([np.interp(v, stops, cols[:, c]) for c in range(3)], -1).astype(np.uint8)


def overlay_heatmap(mosaic: np.ndarray, grid_attn: np.ndarray, alpha: float = 0.55, upscale: int = 3) -> Image.Image:
    """Bilinear-upsampled attention heatmap blended over the H&E mosaic."""
    h, w = mosaic.shape[:2]
    base = Image.fromarray(mosaic).resize((w * upscale, h * upscale), Image.BICUBIC)
    heat = Image.fromarray((grid_attn * 255).astype(np.uint8)).resize(base.size, Image.BILINEAR)
    rgb = Image.fromarray(_colormap(np.asarray(heat, dtype=np.float32) / 255.0))
    return Image.blend(base.convert("RGB"), rgb, alpha)


def attention_mask_auroc(attn_all: np.ndarray, atypia_all: np.ndarray, thr: float = 0.4) -> float:
    """Mean per-patient tile-level AUROC of attention vs. ground-truth atypia mask (atypia > thr)."""
    vals = []
    for a, m in zip(attn_all, atypia_all):
        lab = (m > thr).astype(int)
        if 0 < lab.sum() < len(lab):
            vals.append(roc_auc_score(lab, a))
    return float(np.mean(vals)) if vals else float("nan")


# ----------------------------------------------------------------------------- gene attribution
class _RnaWrapper(torch.nn.Module):
    """f(rna) -> logit, holding the patient's WSI embedding fixed (fusion) or ignoring it (omics-only)."""

    def __init__(self, model, z_wsi=None):
        super().__init__()
        self.model, self.z = model, z_wsi

    def forward(self, rna):
        if self.z is None:                                     # omics-only
            return self.model.head(self.model.encoder(rna))
        z = self.z.expand(rna.shape[0], -1)
        return self.model.fuse(z, rna)[0].unsqueeze(-1)


def _prepare(model, tiles_u8):
    if hasattr(model, "fuse"):
        with torch.no_grad():
            z, _ = model.encode_image(tiles_to_tensor(tiles_u8[None]))
        return _RnaWrapper(model, z).eval()
    return _RnaWrapper(model, None).eval()


def integrated_gradients(wrapper, x: torch.Tensor, baseline: torch.Tensor | None = None, steps: int = 64):
    """Integrated Gradients (Sundararajan et al. 2017) for one (1, G) z-scored RNA vector.

    Returns (attr (G,), delta) where delta = sum(attr) - (f(x) - f(baseline)) (completeness error).
    Baseline = 0 = the training-cohort mean expression (inputs are z-scored)."""
    baseline = torch.zeros_like(x) if baseline is None else baseline
    alphas = torch.linspace(0, 1, steps + 1)[1:].view(-1, 1)
    path = (baseline + alphas * (x - baseline)).requires_grad_(True)
    out = wrapper(path).sum()
    grads, = torch.autograd.grad(out, path)
    attr = ((x - baseline) * grads.mean(0, keepdim=True)).squeeze(0).detach()
    with torch.no_grad():
        delta = float(attr.sum() - (wrapper(x) - wrapper(baseline)).sum())
    return attr.numpy(), delta


def shap_attributions(wrapper, x: torch.Tensor, background: torch.Tensor) -> np.ndarray:
    import shap
    expl = shap.GradientExplainer(wrapper, background)
    sv = expl.shap_values(x)
    sv = np.asarray(sv[0] if isinstance(sv, list) else sv)
    return sv.reshape(-1)[: x.shape[1]] if sv.size == x.shape[1] else sv.reshape(x.shape[1], -1)[:, 0]


def gene_attributions(model, tiles_u8: np.ndarray, rna_raw: np.ndarray, scaler: dict, method: str = "ig",
                      background_raw: np.ndarray | None = None):
    """Per-patient gene attributions for the high-grade logit.

    Returns a DataFrame ranked by |attribution| with gene, programme, z-expression and attribution.
    Positive attribution = pushes the prediction towards High-Grade / Aggressive HCC."""
    wrapper = _prepare(model, tiles_u8)
    x = torch.from_numpy(apply_scaler(rna_raw[None], scaler))
    delta = float("nan")
    if method == "shap" and background_raw is not None:
        try:
            bg = torch.from_numpy(apply_scaler(background_raw, scaler))
            attr = shap_attributions(wrapper, x, bg)
        except Exception as e:                                # pragma: no cover - version differences
            print(f"[explainability] SHAP failed ({e}); falling back to Integrated Gradients")
            attr, delta = integrated_gradients(wrapper, x)
    else:
        attr, delta = integrated_gradients(wrapper, x)
    df = pd.DataFrame(dict(gene=GENES, programme=[GENE_TO_PROGRAM[g] for g in GENES],
                           z_expression=x[0].numpy(), attribution=attr))
    df["abs_attr"] = df["attribution"].abs()
    df = df.sort_values("abs_attr", ascending=False).reset_index(drop=True)
    df.attrs["completeness_delta"] = delta
    return df


def programme_attribution(df: pd.DataFrame) -> pd.DataFrame:
    """Sum signed attributions by biological programme."""
    return (df.groupby("programme")["attribution"].sum().rename("attribution").reset_index()
              .assign(abs=lambda d: d.attribution.abs()).sort_values("abs", ascending=False)
              .drop(columns="abs").reset_index(drop=True))


def global_gene_importance(model, tiles_all, rna_all, scaler, idx, max_n: int = 30) -> pd.DataFrame:
    """Mean |IG attribution| per gene over a set of patients."""
    imp = np.zeros(len(GENES))
    idx = list(idx)[:max_n]
    for i in idx:
        d = gene_attributions(model, tiles_all[i], rna_all[i], scaler, "ig")
        imp += d.set_index("gene").loc[GENES, "abs_attr"].to_numpy()
    out = pd.DataFrame(dict(gene=GENES, programme=[GENE_TO_PROGRAM[g] for g in GENES], mean_abs_attr=imp / len(idx)))
    return out.sort_values("mean_abs_attr", ascending=False).reset_index(drop=True)
