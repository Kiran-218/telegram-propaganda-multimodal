#!/usr/bin/env python3
r"""
make_probe_figures.py -- Build dissertation figures from the embedding-probe
output CSVs.

Reads (whichever exist) in --outdir:
  probe_permutation_both.csv / _text.csv / _image.csv   (perm_auc column)
  probe_outer_folds_both.csv / _text.csv / _image.csv   (auc, f1_prop, chosen_C)

Produces in --figdir:
  fig_permutation_both.pdf/.png   histogram of the 1000 shuffled-label AUCs with
                                  the real AUC marked -- the "not noise" figure.
  fig_modality_auc.pdf/.png       per-modality AUC bar chart with chance line
                                  and permutation-95th-pct markers.
  fig_cosine_vs_probe.pdf/.png    the cosine-mismatch null vs the probe AUC.

Figures are vector PDF (for LaTeX \includegraphics) plus PNG (for quick viewing).
Colours are colour-blind-safe; no chartjunk; sized for a single column.

The real AUCs are read from the outer-fold CSVs (mean of the auc column), so the
figures always match your actual run -- nothing is hard-coded except the cosine
mismatch baseline (0.542) which comes from your evaluate_mismatch.py result.
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COSINE_MISMATCH_AUC = 0.542   # from evaluate_mismatch.py (the post-level null)

# colour-blind-safe palette (Wong)
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
GREY = "#999999"
RED = "#D55E00"

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def load_perm(outdir, mod):
    p = os.path.join(outdir, f"probe_permutation_{mod}.csv")
    if os.path.exists(p):
        return pd.read_csv(p)["perm_auc"].to_numpy()
    return None


def load_folds(outdir, mod):
    p = os.path.join(outdir, f"probe_outer_folds_{mod}.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return None


def real_auc(outdir, mod):
    f = load_folds(outdir, mod)
    return None if f is None else float(f["auc"].mean())


def perm_p(perm, real):
    return (np.sum(perm >= real) + 1) / (len(perm) + 1)


def fig_permutation(outdir, figdir, mod="both"):
    perm = load_perm(outdir, mod)
    r = real_auc(outdir, mod)
    if perm is None or r is None:
        print(f"[skip] permutation figure: missing files for '{mod}'")
        return
    p = perm_p(perm, r)
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.hist(perm, bins=40, color=GREY, alpha=0.8,
            label="shuffled-label AUCs (null)")
    ax.axvline(r, color=RED, lw=2,
               label=f"observed AUC = {r:.3f}")
    ax.axvline(float(np.percentile(perm, 95)), color=BLUE, lw=1.2, ls="--",
               label="null 95th percentile")
    ax.set_xlabel("AUC")
    ax.set_ylabel("permutations")
    ax.set_title(f"Label-permutation null vs observed ({mod}), "
                 f"p = {p:.3f}", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"fig_permutation_{mod}.{ext}"))
    plt.close(fig)
    print(f"[ok] fig_permutation_{mod}")


def fig_modality(outdir, figdir):
    mods = ["text", "image", "both", "latefusion"]
    labels = {"text": "Text only\n(512)", "image": "Image only\n(512)",
              "both": "Concat\n(1024)", "latefusion": "Late fusion\n(2x512)"}
    aucs, nulls95, present = [], [], []
    for m in mods:
        r = real_auc(outdir, m)
        perm = load_perm(outdir, m)
        if r is None:
            continue
        present.append(m)
        aucs.append(r)
        nulls95.append(float(np.percentile(perm, 95)) if perm is not None else None)
    if not present:
        print("[skip] modality figure: no fold files found")
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    x = np.arange(len(present))
    colors = {"text": ORANGE, "image": GREEN, "both": BLUE, "latefusion": "#56B4E9"}
    bars = ax.bar(x, aucs, color=[colors[m] for m in present], width=0.62)
    # chance line
    ax.axhline(0.5, color=GREY, ls=":", lw=1, label="chance (0.50)")
    # null 95th pct markers
    for xi, n95 in zip(x, nulls95):
        if n95 is not None:
            ax.plot([xi - 0.3, xi + 0.3], [n95, n95], color=RED, lw=1.2)
    ax.plot([], [], color=RED, lw=1.2, label="null 95th percentile")
    for xi, a in zip(x, aucs):
        ax.text(xi, a + 0.012, f"{a:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in present])
    ax.set_ylabel("cross-validated AUC")
    ax.set_ylim(0.4, max(aucs) + 0.08)
    ax.set_title("Discriminative signal by modality", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower center", ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"fig_modality_auc.{ext}"))
    plt.close(fig)
    print("[ok] fig_modality_auc")


def fig_cosine_vs_probe(outdir, figdir):
    r_both = real_auc(outdir, "both")
    if r_both is None:
        print("[skip] cosine-vs-probe figure: missing 'both' folds")
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.3))
    names = ["Cosine mismatch\n(1-dim)", "Embedding probe\n(1024-dim)"]
    vals = [COSINE_MISMATCH_AUC, r_both]
    bars = ax.bar([0, 1], vals, color=[GREY, BLUE], width=0.55)
    ax.axhline(0.5, color=GREY, ls=":", lw=1)
    for xi, v in zip([0, 1], vals):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(names)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.4, max(vals) + 0.08)
    ax.set_title("Same embeddings, two readouts", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"fig_cosine_vs_probe.{ext}"))
    plt.close(fig)
    print("[ok] fig_cosine_vs_probe")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--figdir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)
    fig_permutation(args.outdir, args.figdir, "both")
    # also make permutation figures for text/image if present
    for m in ("text", "image"):
        if load_perm(args.outdir, m) is not None:
            fig_permutation(args.outdir, args.figdir, m)
    fig_modality(args.outdir, args.figdir)
    fig_cosine_vs_probe(args.outdir, args.figdir)
    print(f"\nFigures written to {args.figdir}/ (PDF for LaTeX, PNG to preview)")


if __name__ == "__main__":
    main()
