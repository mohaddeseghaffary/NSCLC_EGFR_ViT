"""Rebuild all result figures from the logged CV metrics and the saved ViT-3D predictions.
Run from the repository root:  python results/make_figures.py
cv_metrics.csv holds the per-fold numbers printed at the end of notebooks 03 and 04.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score

OUT = "results/"
METRICS = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced Acc", "auc": "AUC", "f1": "F1"}
df = pd.read_csv(OUT + "cv_metrics.csv")

# summary table (mean ± std over folds, sample std)
g = df.groupby("model")[list(METRICS)].agg(["mean", "std"])
rows = []
for m in g.index:
    rows.append({"model": m, **{METRICS[k]: f"{g.loc[m,(k,'mean')]:.3f} ± {g.loc[m,(k,'std')]:.3f}" for k in METRICS}})
pd.DataFrame(rows).to_csv(OUT + "cv_summary.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))

# per-fold figures
for model, tag in [("2D Swin-T (slices)", "2d"), ("3D SwinUNETR (volume)", "3d")]:
    d = df[df.model == model]
    x = np.arange(len(d)); w = 0.2
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, (k, lab) in enumerate(METRICS.items()):
        ax.bar(x + (j - 1.5) * w, d[k], w, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([f"Fold {f}" for f in d.fold])
    ax.set_ylim(0, 1); ax.set_ylabel("Score"); ax.set_title(f"{model}: 5-fold CV metrics")
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.tight_layout(); fig.savefig(OUT + f"fig_cv_per_fold_{tag}.png", dpi=200); plt.close(fig)

# model comparison (mean ± std)
fig, ax = plt.subplots(figsize=(7, 4))
names = ["Balanced Acc", "AUC", "F1"]; keys = ["balanced_accuracy", "auc", "f1"]
x = np.arange(len(keys)); w = 0.35
for j, m in enumerate(g.index):
    ax.bar(x + (j - 0.5) * w, [g.loc[m, (k, "mean")] for k in keys], w,
           yerr=[g.loc[m, (k, "std")] for k in keys], capsize=3, label=m)
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1)
ax.axhline(0.5, color="grey", lw=0.8, ls="--")
ax.set_title("2D vs 3D Swin: mean ± std over 5 folds"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "fig_model_comparison.png", dpi=200); plt.close(fig)

# confusion matrix of the early ViT-3D baseline, from the real saved predictions
p = pd.read_csv(OUT + "vit3d_patient_predictions.csv")
pred = (p.prob >= 0.5).astype(int)
cm = confusion_matrix(p.label, pred)
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred Wildtype", "Pred Mutant"]); ax.set_yticklabels(["True Wildtype", "True Mutant"])
ax.set_title(f"ViT-3D baseline (n={len(p)}, thr=0.5)\nAUC = {roc_auc_score(p.label, p.prob):.3f}")
fig.tight_layout(); fig.savefig(OUT + "fig_confusion_matrix_vit3d.png", dpi=200); plt.close(fig)
print("confusion matrix:", cm.tolist())
