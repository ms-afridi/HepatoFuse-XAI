from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from .common import K_TILES


class TileEncoder(nn.Module):
    """ResNet18 (stem + layers 1-3, 256-d) patch encoder with a small projection head.

    Truncating at layer3 keeps the model light (~2.8M params) for small tiles / CPU training.
    Pass `pretrained=True` to initialise from ImageNet (requires a download).
    """

    def __init__(self, out_dim: int = 128, pretrained: bool = False):
        super().__init__()
        r = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3,
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.proj = nn.Sequential(nn.Linear(256, out_dim), nn.ReLU(), nn.Dropout(0.2))
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:       # x: (B, K, 3, S, S)
        b, k = x.shape[:2]
        h = self.backbone(x.flatten(0, 1))
        return self.proj(h).view(b, k, -1)                     # (B, K, D)


class GatedAttention(nn.Module):
    """Gated attention MIL pooling (Ilse, Tomczak & Welling, ICML 2018).

    a_k = softmax_k( w^T [ tanh(V h_k) * sigmoid(U h_k) ] ),   z = sum_k a_k h_k
    """

    def __init__(self, in_dim: int, hid: int = 64, dropout: float = 0.25):
        super().__init__()
        self.V, self.U, self.w = nn.Linear(in_dim, hid), nn.Linear(in_dim, hid), nn.Linear(hid, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor):                        # h: (B, K, D)
        s = self.w(self.drop(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))).squeeze(-1)
        a = torch.softmax(s, dim=1)                            # (B, K), sums to 1 over tiles
        return (a.unsqueeze(-1) * h).sum(1), a


class OmicsEncoder(nn.Module):
    """LayerNorm -> Linear -> SiLU -> Dropout -> Linear -> SiLU gene-expression encoder."""

    def __init__(self, n_genes: int = 60, hid: int = 128, out_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(n_genes), nn.Linear(n_genes, hid), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(hid, out_dim), nn.SiLU())
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x)


class ImageOnlyABMIL(nn.Module):
    def __init__(self, embed_dim: int = 128, pretrained: bool = False):
        super().__init__()
        self.encoder = TileEncoder(embed_dim, pretrained)
        self.pool = GatedAttention(embed_dim)
        self.head = nn.Sequential(nn.Dropout(0.25), nn.Linear(embed_dim, 1))

    def encode_image(self, tiles):
        h = self.encoder(tiles)
        z, a = self.pool(h)
        return z, a

    def forward(self, tiles=None, rna=None):
        z, a = self.encode_image(tiles)
        return dict(logit=self.head(z).squeeze(-1), attn=a, gates=None)


class OmicsOnlyMLP(nn.Module):
    def __init__(self, n_genes: int = 60):
        super().__init__()
        self.encoder = OmicsEncoder(n_genes)
        self.head = nn.Linear(self.encoder.out_dim, 1)

    def forward(self, tiles=None, rna=None):
        return dict(logit=self.head(self.encoder(rna)).squeeze(-1), attn=None, gates=None)


class HepatoFuseMultimodal(nn.Module):
    """Cross-modal gated fusion of the WSI attention embedding and the RNA-seq embedding.

        h_i = proj_i(z_wsi),  h_r = proj_r(e_rna)
        [g_i, g_r] = sigmoid( W [h_i ; h_r] )        (each gate sees *both* modalities)
        fused = LayerNorm( [ g_i * h_i ; g_r * h_r ] )  ->  MLP  ->  logit

    Modality dropout (train only) randomly blanks one embedding so neither branch dominates and
    the model degrades gracefully when a modality is missing.
    """

    def __init__(self, n_genes: int = 60, embed_dim: int = 128, fusion_dim: int = 64,
                 pretrained: bool = False, modality_dropout: float = 0.15):
        super().__init__()
        self.encoder = TileEncoder(embed_dim, pretrained)
        self.pool = GatedAttention(embed_dim)
        self.omics = OmicsEncoder(n_genes)
        self.proj_i = nn.Linear(embed_dim, fusion_dim)
        self.proj_r = nn.Linear(self.omics.out_dim, fusion_dim)
        self.gate = nn.Linear(2 * fusion_dim, 2 * fusion_dim)
        self.norm = nn.LayerNorm(2 * fusion_dim)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * fusion_dim, 64), nn.SiLU(), nn.Dropout(0.2),
                                  nn.Linear(64, 1))
        self.fusion_dim, self.modality_dropout = fusion_dim, modality_dropout

    def encode_image(self, tiles):
        h = self.encoder(tiles)
        z, a = self.pool(h)
        return z, a

    def fuse(self, z_wsi: torch.Tensor, rna: torch.Tensor):
        """Fuse a (B, D) WSI embedding with (B, G) RNA values; differentiable w.r.t. `rna`."""
        h_i, h_r = self.proj_i(z_wsi), self.proj_r(self.omics(rna))
        if self.training and self.modality_dropout > 0:
            drop = torch.rand(h_i.shape[0], 1, device=h_i.device) < self.modality_dropout
            pick_i = torch.rand(h_i.shape[0], 1, device=h_i.device) < 0.5
            h_i = torch.where(drop & pick_i, torch.zeros_like(h_i), h_i)
            h_r = torch.where(drop & ~pick_i, torch.zeros_like(h_r), h_r)
        g = torch.sigmoid(self.gate(torch.cat([h_i, h_r], -1)))
        g_i, g_r = g[:, : self.fusion_dim], g[:, self.fusion_dim:]
        fused = self.norm(torch.cat([g_i * h_i, g_r * h_r], -1))
        gates = torch.stack([g_i.mean(-1), g_r.mean(-1)], -1)     # (B, 2): [image, RNA]
        return self.head(fused).squeeze(-1), gates

    def forward(self, tiles=None, rna=None):
        z, a = self.encode_image(tiles)
        logit, gates = self.fuse(z, rna)
        return dict(logit=logit, attn=a, gates=gates)


MODEL_REGISTRY = {"image_only_abmil": ImageOnlyABMIL, "omics_only_mlp": OmicsOnlyMLP,
                  "hepatofuse_multimodal": HepatoFuseMultimodal}
USES_IMAGE = {"image_only_abmil": True, "omics_only_mlp": False, "hepatofuse_multimodal": True}
USES_RNA = {"image_only_abmil": False, "omics_only_mlp": True, "hepatofuse_multimodal": True}


def build_model(name: str, n_genes: int = 60, pretrained: bool = False) -> nn.Module:
    if name == "image_only_abmil":
        return ImageOnlyABMIL(pretrained=pretrained)
    if name == "omics_only_mlp":
        return OmicsOnlyMLP(n_genes)
    if name == "hepatofuse_multimodal":
        return HepatoFuseMultimodal(n_genes, pretrained=pretrained)
    raise KeyError(name)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    t, r = torch.randn(2, K_TILES, 3, 64, 64), torch.randn(2, 60)
    for n in MODEL_REGISTRY:
        m = build_model(n).eval()
        o = m(t, r)
        print(n, count_params(m), o["logit"].shape, None if o["attn"] is None else o["attn"].shape)
