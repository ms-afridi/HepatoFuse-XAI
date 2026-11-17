from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from src.common import ART_DIR, CLASS_NAMES, GRID, apply_scaler, load_data, tiles_to_tensor
from src.explainability import (attention_grid, gene_attributions, load_checkpoint, make_mosaic,
                                overlay_heatmap, programme_attribution, tile_attention)
from src.models import MODEL_REGISTRY

st.set_page_config(page_title="HepatoFuse-XAI", page_icon="🔬", layout="wide")
LABELS = {"image_only_abmil": "Image-only ABMIL", "omics_only_mlp": "RNA-only MLP",
          "hepatofuse_multimodal": "HepatoFuse (WSI + RNA)"}
RED, BLUE = "#d7301f", "#2b8cbe"


@st.cache_data(show_spinner="Loading cohort...")
def get_data():
    return load_data()


@st.cache_resource(show_spinner="Loading models...")
def get_models():
    return {n: load_checkpoint(n) for n in MODEL_REGISTRY}


@st.cache_data
def get_artifacts():
    f = lambda n: pd.read_csv(ART_DIR / n) if (ART_DIR / n).exists() else None
    return dict(ablation=f("ablation_results.csv"), per_seed=f("per_seed_results.csv"), gene_global=f("global_gene_importance.csv"))


def predict_patient(models, d, i):
    tiles = tiles_to_tensor(d["tiles"][i:i + 1])
    out = {}
    for n, (m, ck) in models.items():
        rna = torch.from_numpy(apply_scaler(d["rna"][i:i + 1], ck["scaler"]))
        with torch.no_grad():
            o = m(tiles, rna)
        out[n] = dict(p=float(torch.sigmoid(o["logit"])[0]),
                      gates=None if o["gates"] is None else o["gates"][0].tolist())
    return out


# ------------------------------------------------------------------------------------- guard
if not (ART_DIR / "hepatofuse_multimodal.pt").exists():
    st.error("Model checkpoints not found. Run:  `python -m src.prepare_data && python -m src.train_eval`")
    st.stop()

d, models, arts = get_data(), get_models(), get_artifacts()
ck_f = models["hepatofuse_multimodal"][1]
split = ck_f["split"]
role = {i: "train" for i in split["train"]} | {i: "val" for i in split["val"]} | {i: "TEST" for i in split["test"]}
clin = d["clinical"]

# ------------------------------------------------------------------------------------- sidebar
st.sidebar.title("🔬 HepatoFuse-XAI")
st.sidebar.caption("ABMIL + RNA-seq multimodal fusion for HCC grade — spatial and molecular explanations.")
only_test = st.sidebar.checkbox("Show held-out TEST patients only", value=True)
cand = [i for i in range(len(clin)) if (not only_test or role[i] == "TEST")]
i = st.sidebar.selectbox("Patient", cand, format_func=lambda k: f"{clin.patient_id[k]}  [{role[k]}]")
att_model = st.sidebar.radio("Attention from", ["hepatofuse_multimodal", "image_only_abmil"], format_func=LABELS.get)
alpha = st.sidebar.slider("Heatmap opacity", 0.1, 0.9, 0.55, 0.05)
method = st.sidebar.radio("Gene attribution", ["Integrated Gradients", "SHAP (GradientExplainer)"])
attr_model = st.sidebar.radio("Attribution model", ["hepatofuse_multimodal", "omics_only_mlp"], format_func=LABELS.get)
st.sidebar.warning("Synthetic cohort · research prototype · **not** for clinical use.")

# ------------------------------------------------------------------------------------- header
row, preds = clin.iloc[i], predict_patient(models, d, i)
st.title(f"Patient {row.patient_id}")
st.caption(f"Split: **{role[i]}** · Ground-truth label (synthetic): **{row.grade_label}**")
c = st.columns(6)
for col, (k, v) in zip(c, [("Age", row.age), ("Sex", row.sex), ("Etiology", row.etiology),
                           ("Steatosis %", row.steatosis_pct), ("Fibrosis (Ishak)", row.fibrosis_ishak),
                           ("AFP ng/mL", row.afp_ng_ml)]):
    col.metric(k, v)

st.subheader("Model predictions · P(High-Grade / Aggressive HCC)")
pc = st.columns(3)
for col, n in zip(pc, MODEL_REGISTRY):
    p = preds[n]["p"]
    col.metric(LABELS[n], f"{p:.1%}", CLASS_NAMES[int(p >= .5)], delta_color="inverse" if p >= .5 else "normal")
    col.progress(min(max(p, 0.0), 1.0))
g = preds["hepatofuse_multimodal"]["gates"]
st.caption(f"Fusion gate activation — WSI: **{g[0]:.2f}**, RNA-seq: **{g[1]:.2f}** (mean sigmoid gate; indicates how much each modality is let through).")

tab1, tab2, tab3, tab4 = st.tabs(["🧫 Histology & attention", "🧬 Molecular attribution", "📊 Ablation", "ℹ️ About"])

# ------------------------------------------------------------------------------------- tab 1
with tab1:
    model_a = models[att_model][0]
    tiles = d["tiles"][i]
    attn = tile_attention(model_a, tiles)
    grid = attention_grid(attn)
    mosaic = make_mosaic(tiles)
    a, b, c3 = st.columns(3)
    a.markdown("**H&E tile mosaic (5×5)**"); a.image(mosaic, width="stretch")
    b.markdown(f"**Attention overlay — {LABELS[att_model]}**"); b.image(overlay_heatmap(mosaic, grid, alpha), width="stretch")
    fig = go.Figure(go.Heatmap(z=attn.reshape(GRID, GRID) * 100, colorscale="Inferno", colorbar=dict(title="% attn"),
                               text=np.round(attn.reshape(GRID, GRID) * 100, 1), texttemplate="%{text}"))
    fig.update_yaxes(autorange="reversed", title="row"); fig.update_xaxes(title="column")
    fig.update_layout(title="Tile attention weights (sum = 100 %)", height=360, margin=dict(l=10, r=10, t=40, b=10))
    c3.plotly_chart(fig, width="stretch")
    top = np.argsort(-attn)[:5]
    st.markdown("**Top-5 attended tiles**")
    cols = st.columns(5)
    for col, k in zip(cols, top):
        col.image(tiles[k], caption=f"tile r{k // GRID + 1}c{k % GRID + 1} · {attn[k]:.1%}", width="stretch")
    with st.expander("Validation against the synthetic ground truth (atypia map)"):
        at = d["atypia"][i].reshape(GRID, GRID)
        f2 = go.Figure(go.Heatmap(z=at, colorscale="Purples", zmin=0, zmax=1, colorbar=dict(title="atypia")))
        f2.update_yaxes(autorange="reversed"); f2.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                                                                 title="True (simulated) nuclear-atypia intensity per tile")
        st.plotly_chart(f2, width="stretch")
        st.caption("Only possible because the cohort is synthetic: the real-data analogue is a pathologist's tumour/grade annotation.")

# ------------------------------------------------------------------------------------- tab 2
with tab2:
    m_attr, ck_a = models[attr_model]
    meth = "shap" if method.startswith("SHAP") else "ig"
    bg = d["rna"][split["train"]][:40]
    df = gene_attributions(m_attr, d["tiles"][i], d["rna"][i], ck_a["scaler"], meth, bg)
    top = df.head(15).iloc[::-1]
    l, r = st.columns([3, 2])
    fig = go.Figure(go.Bar(x=top.attribution, y=top.gene, orientation="h",
                           marker_color=[RED if v > 0 else BLUE for v in top.attribution],
                           customdata=np.c_[top.programme, top.z_expression.round(2)],
                           hovertemplate="%{y}<br>%{customdata[0]}<br>z-expr %{customdata[1]}<br>attr %{x:.3f}<extra></extra>"))
    fig.update_layout(title=f"Top-15 genes · {method} · {LABELS[attr_model]}", height=520,
                      xaxis_title="← towards Low-grade     attribution to High-grade logit     towards High-grade →",
                      margin=dict(l=10, r=10, t=40, b=10))
    l.plotly_chart(fig, width="stretch")
    pa = programme_attribution(df)
    fig2 = go.Figure(go.Bar(x=pa.attribution, y=pa.programme, orientation="h",
                            marker_color=[RED if v > 0 else BLUE for v in pa.attribution]))
    fig2.update_yaxes(autorange="reversed"); fig2.update_layout(title="Biological programme level", height=520,
                                                                 margin=dict(l=10, r=10, t=40, b=10))
    r.plotly_chart(fig2, width="stretch")
    dl = df.attrs.get("completeness_delta", float("nan"))
    if not np.isnan(dl):
        st.caption(f"Integrated-Gradients completeness error |Σattr − (f(x) − f(baseline))| = {abs(dl):.4f} "
                   "(baseline = cohort-mean expression). For the fusion model the WSI embedding is held fixed.")
    tbl = df.drop(columns="abs_attr").head(20).round(3); tbl.attrs = {} 
    st.dataframe(tbl, width="stretch")
    if arts["gene_global"] is not None:
        with st.expander("Global importance (mean |IG| over held-out test patients, fusion model)"):
            gg = arts["gene_global"].head(15).iloc[::-1]
            st.plotly_chart(go.Figure(go.Bar(x=gg.mean_abs_attr, y=gg.gene, orientation="h", marker_color="#636363"))
                            .update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10)), width="stretch")

# ------------------------------------------------------------------------------------- tab 3
with tab3:
    ab = arts["ablation"]
    if ab is None:
        st.info("Run `python -m src.train_eval` to produce the ablation table.")
    else:
        ab = ab.copy(); ab["model"] = ab["model"].map(LABELS)
        show = pd.DataFrame({"Model": ab["model"]})
        for k, nm in [("auroc", "AUROC"), ("pr_auc", "PR-AUC"), ("f1", "F1"), ("accuracy", "Accuracy"),
                      ("attn_vs_mask_auroc", "Attention↔mask AUROC")]:
            show[nm] = [("–" if pd.isna(m) else f"{m:.3f} ± {0 if pd.isna(s) else s:.3f}") for m, s in zip(ab[f"{k}_mean"], ab[f"{k}_sd"])]
        show["Params"] = ab["params"].map("{:,}".format)
        st.dataframe(show, hide_index=True, width="stretch")
        fig = go.Figure(go.Bar(x=ab["model"], y=ab["auroc_mean"], error_y=dict(type="data", array=ab["auroc_sd"].fillna(0)),
                               marker_color=["#756bb1", "#3182bd", RED]))
        fig.update_layout(title="Held-out test AUROC (mean ± SD over seeds)", yaxis=dict(range=[0.5, 1.0]), height=380)
        st.plotly_chart(fig, width="stretch")
        st.caption("Small synthetic test sets (n=30 per split): differences between models can be within noise — see SD.")

# ------------------------------------------------------------------------------------- tab 4
with tab4:
    st.markdown("""
**What this is.** A prototype of an image–omics pipeline for chronic liver disease / HCC: attention-based multiple-instance
learning (Ilse et al., 2018) over a 5×5 grid of H&E tiles, an RNA-seq encoder over 60 liver-relevant genes, and a cross-modal
gated fusion head. Explanations are spatial (tile attention) and molecular (Integrated Gradients / SHAP aggregated to biological programmes).

**What this is not.** The cohort is **synthetic** (structured by known biology, with a known ground truth) — it demonstrates the
*method and engineering*, not clinical performance. Attention is a plausible-region indicator, not proof of causal relevance.
Swapping in real data (TCGA-LIHC / in-house WSI features + RNA-seq) only requires replacing the three files in `data/`.
""")
