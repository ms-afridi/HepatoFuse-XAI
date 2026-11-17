import numpy as np
import torch

from src.common import GENES, K_TILES, GRID, stratified_split
from src.explainability import attention_grid, integrated_gradients, make_mosaic, overlay_heatmap
from src.models import MODEL_REGISTRY, build_model


def test_gene_panel():
    assert len(GENES) == 60 and "AFP" in GENES and "GPC3" in GENES


def test_models_forward_shapes_and_attention_simplex():
    t, r = torch.randn(3, K_TILES, 3, 64, 64), torch.randn(3, 60)
    for n in MODEL_REGISTRY:
        o = build_model(n).eval()(t, r)
        assert o["logit"].shape == (3,)
        if o["attn"] is not None:
            assert o["attn"].shape == (3, K_TILES)
            assert torch.allclose(o["attn"].sum(1), torch.ones(3), atol=1e-5)


def test_stratified_split_is_disjoint_and_balanced():
    y = np.array([0] * 66 + [1] * 54)
    tr, va, te = stratified_split(y, 0)
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    assert abs(y[te].mean() - y.mean()) < 0.05


def test_mosaic_and_overlay():
    tiles = np.random.randint(0, 255, (K_TILES, 64, 64, 3), dtype=np.uint8)
    m = make_mosaic(tiles)
    assert m.shape == (GRID * 64, GRID * 64, 3)
    # tile (row 1, col 2) must sit at the right place in the mosaic
    assert np.array_equal(m[64:128, 128:192], tiles[1 * GRID + 2])
    g = attention_grid(np.random.rand(K_TILES))
    assert g.shape == (GRID, GRID) and overlay_heatmap(m, g).size == (GRID * 64 * 3, GRID * 64 * 3)


def test_integrated_gradients_completeness():
    from src.explainability import _RnaWrapper
    model = build_model("omics_only_mlp").eval()
    x = torch.randn(1, 60)
    attr, delta = integrated_gradients(_RnaWrapper(model), x, steps=256)
    assert attr.shape == (60,) and abs(delta) < 0.05
