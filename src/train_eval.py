from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
                             roc_auc_score)

from .common import (ART_DIR, apply_scaler, fit_scaler, load_data, save_json, stratified_split, tiles_to_tensor)
from .explainability import attention_mask_auroc, global_gene_importance
from .models import USES_IMAGE, USES_RNA, MODEL_REGISTRY, build_model, count_params

MODELS = list(MODEL_REGISTRY)


def set_seed(s: int):
    np.random.seed(s); torch.manual_seed(s)


def batches(idx, bs, shuffle, rng):
    idx = np.array(idx)
    if shuffle:
        idx = rng.permutation(idx)
    for i in range(0, len(idx), bs):
        yield idx[i:i + bs]


def make_inputs(name, d, rna_z, ids, augment=False, rng=None):
    tiles = rna = None
    if USES_IMAGE[name]:
        t = d["tiles"][ids]
        if augment:                                             # per-tile flips (grid order preserved)
            t = t.copy()
            for b in range(t.shape[0]):
                for k in range(t.shape[1]):
                    if rng.random() < 0.5: t[b, k] = t[b, k][:, ::-1]
                    if rng.random() < 0.5: t[b, k] = t[b, k][::-1]
        tiles = tiles_to_tensor(t)
    if USES_RNA[name]:
        rna = torch.from_numpy(rna_z[ids])
        if augment:
            rna = rna + 0.05 * torch.randn_like(rna)           # light expression noise
    return tiles, rna


@torch.no_grad()
def predict(model, name, d, rna_z, ids, bs=16):
    model.eval()
    probs, attns = [], []
    for b in batches(ids, bs, False, None):
        t, r = make_inputs(name, d, rna_z, b)
        o = model(t, r)
        probs.append(torch.sigmoid(o["logit"]).numpy())
        if o["attn"] is not None:
            attns.append(o["attn"].numpy())
    return np.concatenate(probs), (np.concatenate(attns) if attns else None)


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return dict(auroc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p), f1=f1_score(y, pred),
                accuracy=accuracy_score(y, pred), balanced_acc=balanced_accuracy_score(y, pred))


def train_one(name, d, rna_z, tr, va, args, seed):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(name, n_genes=rna_z.shape[1])
    lr = args.lr if USES_IMAGE[name] else 1e-3
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best, best_state, bad = (1e9, None, 0)
    y = d["y"]
    for ep in range(args.epochs):
        model.train()
        for b in batches(tr, args.bs, True, rng):
            t, r = make_inputs(name, d, rna_z, b, augment=True, rng=rng)
            loss = F.binary_cross_entropy_with_logits(model(t, r)["logit"], torch.from_numpy(y[b]).float())
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        pv, _ = predict(model, name, d, rna_z, va)
        vloss = float(F.binary_cross_entropy(torch.from_numpy(pv).clamp(1e-6, 1 - 1e-6), torch.from_numpy(y[va]).float()))
        if vloss < best - 1e-4:
            best, best_state, bad = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), best, ep + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--models", nargs="+", default=MODELS)
    args = ap.parse_args()

    ART_DIR.mkdir(exist_ok=True)
    d = load_data()
    y = d["y"]
    rows, preds = [], []
    for seed in range(args.seeds):
        tr, va, te = stratified_split(y, seed)
        sc = fit_scaler(d["rna"][tr])                          # scaler fitted on TRAIN only (no leakage)
        rna_z = apply_scaler(d["rna"], sc)
        for name in args.models:
            t0 = time.time()
            model, vloss, ep = train_one(name, d, rna_z, tr, va, args, seed)
            p_te, a_te = predict(model, name, d, rna_z, te)
            m = metrics(y[te], p_te)
            m.update(model=name, seed=seed, epochs_run=ep, val_loss=vloss, params=count_params(model),
                     attn_vs_mask_auroc=(attention_mask_auroc(a_te, d["atypia"][te]) if a_te is not None else np.nan),
                     n_train=len(tr), n_val=len(va), n_test=len(te), sec=round(time.time() - t0, 1))
            rows.append(m)
            print(f"seed {seed} {name:22s} AUROC={m['auroc']:.3f} PR-AUC={m['pr_auc']:.3f} F1={m['f1']:.3f} "
                  f"Acc={m['accuracy']:.3f} attnAUC={m['attn_vs_mask_auroc']:.3f} ({m['sec']}s, {ep} ep)", flush=True)
            if seed == 0:                                       # checkpoint + predictions used by the app
                torch.save(dict(state_dict=model.state_dict(), genes=d["genes"], scaler=sc, seed=0,
                                split=dict(train=tr.tolist(), val=va.tolist(), test=te.tolist())),
                           ART_DIR / f"{name}.pt")
                all_p, _ = predict(model, name, d, rna_z, np.arange(len(y)))
                preds.append(pd.DataFrame(dict(patient_id=d["clinical"].patient_id, model=name, prob_high_grade=all_p)))
                if name == "hepatofuse_multimodal":
                    split = dict(train=tr.tolist(), val=va.tolist(), test=te.tolist())
                    save_json(split, ART_DIR / "split.json")
                    gi = global_gene_importance(model, d["tiles"], d["rna"], sc, te)
                    gi.to_csv(ART_DIR / "global_gene_importance.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(ART_DIR / "per_seed_results.csv", index=False)
    pd.concat(preds).to_csv(ART_DIR / "predictions.csv", index=False)
    cols = ["auroc", "pr_auc", "f1", "accuracy", "balanced_acc", "attn_vs_mask_auroc"]
    g = df.groupby("model")[cols]
    summ = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).reindex(args.models)
    summ["params"] = df.groupby("model")["params"].first()
    summ.to_csv(ART_DIR / "ablation_results.csv")
    save_json(dict(n_seeds=args.seeds, args=vars(args), results=summ.reset_index().to_dict("records"),
                   per_seed=rows), ART_DIR / "ablation_results.json")
    print("\n", summ.round(3).to_string())


if __name__ == "__main__":
    main()
