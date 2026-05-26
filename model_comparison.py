"""
Ablation study v1->v7 + calibration reliability diagrams.

Output:
  output/model_comparison.png - bar chart (Accuracy / Log Loss / Brier ± std)
  output/calibration_lr.png   - reliability diagrams for v1-v3 (1×3 grid)
  output/calibration_lgb.png  - reliability diagrams for v4-v7 (1×4 grid)
"""

import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

from features import (
    V1_NUMERIC,
    V3_NUMERIC,
    V5_NUMERIC,
    V6_PER_SNAP,
    V6_ROUND_CTX,
    V6_ALL_NUMERIC,
    V7_ALL_NUMERIC,
    MAP_FEATURE,
    EXCLUDE_MAPS,
    derive_runtime_features,
    derive_runtime_features_v6,
    derive_runtime_features_v7,
    melt_v7,
    add_delta_features,
    filter_present,
)

DATA_FILE = r"C:\Users\hasan\Downloads\CS2Predictor\training_data.parquet"
OUTPUT_DIR = r"C:\Users\hasan\Downloads\CS2Predictor\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


#Single-pass CV helper
def cv_eval(estimator, X, y, groups, cv=None):
    if cv is None:
        cv = GroupKFold(n_splits=5)

    is_df = isinstance(X, pd.DataFrame)
    n = len(y)
    proba_oof = np.zeros(n)
    fold_acc, fold_ll, fold_bs = [], [], []

    idx_arr = np.arange(n)
    for train_idx, test_idx in cv.split(idx_arr, y, groups):
        if is_df:
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        else:
            X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        est = clone(estimator)
        est.fit(X_tr, y_tr)
        p = est.predict_proba(X_te)[:, 1]
        proba_oof[test_idx] = p

        pred = (p >= 0.5).astype(int)
        fold_acc.append(accuracy_score(y_te, pred))
        fold_ll.append(log_loss(y_te, p))
        fold_bs.append(brier_score_loss(y_te, p))

    return proba_oof, {
        "accuracy": np.array(fold_acc),
        "log_loss": np.array(fold_ll),
        "brier": np.array(fold_bs),
    }


#Load base data
df = pl.read_parquet(DATA_FILE).filter(~pl.col(MAP_FEATURE).is_in(list(EXCLUDE_MAPS)))
df_pd = derive_runtime_features(df.to_pandas())

NUMERIC_V1_ALL = filter_present(V1_NUMERIC, df_pd)
NUMERIC_V3_ALL = filter_present(V3_NUMERIC, df_pd)
NUMERIC_V5_ALL = filter_present(V5_NUMERIC, df_pd)

le = LabelEncoder()
df_pd["map_encoded"] = le.fit_transform(df_pd[MAP_FEATURE])

y = df_pd["target"].values
groups = df_pd["match_id"].values
cv5 = GroupKFold(n_splits=5)

from lgb_config import LGB_V4_PARAMS, LGB_V5_PARAMS, LGB_V6_PARAMS, LGB_V7_PARAMS


def lgb_clf(params):
    return lgb.LGBMClassifier(**params)


#CV runs v1-v5
print("v1 (LR, 24 feat)...")
proba_v1, m_v1 = cv_eval(
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ]
    ),
    df_pd[NUMERIC_V1_ALL].values,
    y,
    groups,
    cv5,
)

print("v2 (LR, 24 feat + map)...")
proba_v2, m_v2 = cv_eval(
    Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), NUMERIC_V1_ALL),
                        (
                            "map",
                            OneHotEncoder(
                                drop="first",
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                            [MAP_FEATURE],
                        ),
                    ]
                ),
            ),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ]
    ),
    df_pd[NUMERIC_V1_ALL + [MAP_FEATURE]],
    y,
    groups,
    cv5,
)

print("v3 (LR, 32 feat + map)...")
proba_v3, m_v3 = cv_eval(
    Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), NUMERIC_V3_ALL),
                        (
                            "map",
                            OneHotEncoder(
                                drop="first",
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                            [MAP_FEATURE],
                        ),
                    ]
                ),
            ),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ]
    ),
    df_pd[NUMERIC_V3_ALL + [MAP_FEATURE]],
    y,
    groups,
    cv5,
)

print("v4 (LGB, 32 feat + map architecture isolation)...")
proba_v4, m_v4 = cv_eval(
    lgb_clf(LGB_V4_PARAMS),
    df_pd[NUMERIC_V3_ALL + ["map_encoded"]].values,
    y,
    groups,
    cv5,
)

print("v5 (LGB, 36 feat + map tactical features)...")
proba_v5, m_v5 = cv_eval(
    lgb_clf(LGB_V5_PARAMS),
    df_pd[NUMERIC_V5_ALL + ["map_encoded"]].values,
    y,
    groups,
    cv5,
)


#CV runs v6-v7 (t50+t75 melt)
def _melt_v6_local(df_wide):
    meta_cols = ["match_id", "round", MAP_FEATURE, "target"] + V6_ROUND_CTX

    def _snap(df, suffix, label):
        meta = (
            df[[c for c in meta_cols if c in df.columns]].copy().reset_index(drop=True)
        )
        if suffix == "":
            snap = (
                df[[c for c in V6_PER_SNAP if c in df.columns]]
                .copy()
                .reset_index(drop=True)
            )
        else:
            rename = {f"{c}_{suffix}": c for c in V6_PER_SNAP[:-1]}
            rename[f"bomb_planted_now_{suffix}"] = "bomb_planted_now"
            cols = [f"{c}_{suffix}" for c in V6_PER_SNAP[:-1]] + [
                f"bomb_planted_now_{suffix}"
            ]
            snap = (
                df[[c for c in cols if c in df.columns]]
                .rename(columns=rename)
                .reset_index(drop=True)
            )
        out = pd.concat([meta, snap], axis=1)
        out["_snap"] = label
        return out

    return add_delta_features(
        pd.concat(
            [_snap(df_wide, "", "t50"), _snap(df_wide, "t75", "t75")], ignore_index=True
        )
    )


_raw = (
    pl.read_parquet(DATA_FILE)
    .filter(~pl.col(MAP_FEATURE).is_in(list(EXCLUDE_MAPS)))
    .to_pandas()
)
_raw = derive_runtime_features_v6(_raw)
melted_v6_data = _melt_v6_local(_raw)
NUMERIC_V6_ALL = filter_present(V6_ALL_NUMERIC, melted_v6_data)
le_v6 = LabelEncoder()
melted_v6_data["map_encoded"] = le_v6.fit_transform(melted_v6_data[MAP_FEATURE])
y_v6 = melted_v6_data["target"].values
groups_v6 = melted_v6_data["match_id"].values
print(f"v6 (LGB t50+t75+Δ, {len(NUMERIC_V6_ALL)}+map feat)...")
proba_v6, m_v6 = cv_eval(
    lgb_clf(LGB_V6_PARAMS),
    melted_v6_data[NUMERIC_V6_ALL + ["map_encoded"]].values,
    y_v6,
    groups_v6,
    GroupKFold(n_splits=5),
)

_raw7 = (
    pl.read_parquet(DATA_FILE)
    .filter(~pl.col(MAP_FEATURE).is_in(list(EXCLUDE_MAPS)))
    .to_pandas()
)
_raw7 = derive_runtime_features_v7(_raw7)
melted_v7_data = melt_v7(_raw7)
NUMERIC_V7_ALL = filter_present(V7_ALL_NUMERIC, melted_v7_data)
le_v7 = LabelEncoder()
melted_v7_data["map_encoded"] = le_v7.fit_transform(melted_v7_data[MAP_FEATURE])
y_v7 = melted_v7_data["target"].values
groups_v7 = melted_v7_data["match_id"].values
print(f"v7 (LGB +Dyn/Wpn/Econ/Tac, {len(NUMERIC_V7_ALL)}+map feat)...")
proba_v7, m_v7 = cv_eval(
    lgb_clf(LGB_V7_PARAMS),
    melted_v7_data[NUMERIC_V7_ALL + ["map_encoded"]].values,
    y_v7,
    groups_v7,
    GroupKFold(n_splits=5),
)


#Console table
versions = ["v1", "v2", "v3", "v4(LGB)", "v5(LGB+tac)", "v6(LGB+\u0394)", "v7(LGB+Dyn)"]
all_m = [m_v1, m_v2, m_v3, m_v4, m_v5, m_v6, m_v7]
y_map = [y, y, y, y, y, y_v6, y_v7]
proba_map = [proba_v1, proba_v2, proba_v3, proba_v4, proba_v5, proba_v6, proba_v7]

print()
header = f"{'Metric':<14}" + "".join(f" {v:>22}" for v in versions)
print(header)
print("-" * len(header))
for metric_name, key in [
    ("Accuracy", "accuracy"),
    ("Log Loss", "log_loss"),
    ("Brier Score", "brier"),
]:
    row = f"{metric_name:<14}"
    for m in all_m:
        row += f" {m[key].mean():>10.4f}+-{m[key].std():<10.4f}"
    print(row)


#Bar chart
colors = {
    "v1": "#4C72B0",
    "v2": "#2ca02c",
    "v3": "#d62728",
    "v4": "#9467bd",
    "v5": "#8c564b",
    "v6": "#e377c2",
    "v7": "#ff7f0e",
}
keys = ["v1", "v2", "v3", "v4", "v5", "v6", "v7"]
labels = [
    "v1 (LR)",
    "v2 (LR+map)",
    "v3 (LR+ctx)",
    "v4 (LGB@v3)",
    "v5 (LGB+tac)",
    "v6 (LGB+\u0394)",
    "v7 (LGB+Dyn)",
]

metric_names = ["Accuracy", "Log Loss", "Brier Score"]
metric_keys = ["accuracy", "log_loss", "brier"]
x = np.arange(3)
n = 7
w = 0.11

fig, ax = plt.subplots(figsize=(16, 5))
for i, (k, lbl, m) in enumerate(zip(keys, labels, all_m)):
    offset = (i - (n - 1) / 2) * w
    vals = [m[mk].mean() for mk in metric_keys]
    errs = [m[mk].std() for mk in metric_keys]
    bars = ax.bar(
        x + offset,
        vals,
        w,
        yerr=errs,
        capsize=3,
        label=lbl,
        color=colors[k],
        edgecolor="white",
        error_kw={"linewidth": 1},
    )
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=5.5,
        )

ax.set_xticks(x)
ax.set_xticklabels(metric_names, fontsize=11)
ax.set_ylabel("Score")
ax.set_title(
    "Ablation study: v1 \u2192 v7\n(5-fold GroupKFold CV, grouped by match_id)",
    fontsize=11,
    fontweight="bold",
)
ax.legend(fontsize=8, ncol=4)
ax.spines[["top", "right"]].set_visible(False)
fig.text(
    0.5,
    -0.02,
    "Accuracy: higher is better   |   Log Loss & Brier Score: lower is better",
    ha="center",
    fontsize=8,
    color="grey",
)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150, bbox_inches="tight"
)
plt.close()
print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'model_comparison.png')}")


#Calibration curves
#Logistic Regression models
panels_lr = [
    (y, proba_v1, "v1 (LR, 24 feat)", "#4C72B0"),
    (y, proba_v2, "v2 (LR, 24+map)", "#2ca02c"),
    (y, proba_v3, "v3 (LR, 32+map)", "#d62728"),
]
fig_lr, axes_lr = plt.subplots(1, 3, figsize=(12, 4))
fig_lr.suptitle(
    "Calibration curves Logistic Regression (v1-v3)", fontsize=12, fontweight="bold"
)
for ax2, (y_true, proba, label, color) in zip(axes_lr, panels_lr):
    frac, mean = calibration_curve(y_true, proba, n_bins=10, strategy="uniform")
    ax2.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
    ax2.plot(mean, frac, "o-", color=color, linewidth=2, markersize=6, label=label)
    ax2.fill_between(mean, frac, mean, alpha=0.15, color=color)
    ll = log_loss(y_true, proba)
    bs = brier_score_loss(y_true, proba)
    ax2.set_title(f"{label}\nLog Loss={ll:.4f}  Brier={bs:.4f}", fontsize=9)
    ax2.set_xlabel("Mean predicted P(T wins)")
    ax2.set_ylabel("Actual fraction")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "calibration_lr.png"), dpi=150, bbox_inches="tight"
)
plt.close()
print(f"Saved: {os.path.join(OUTPUT_DIR, 'calibration_lr.png')}")

#LGB models
panels_lgb = [
    (y, proba_v4, f"v4 (LGB, {len(NUMERIC_V3_ALL)+1} feat)", "#9467bd"),
    (y, proba_v5, f"v5 (LGB, {len(NUMERIC_V5_ALL)+1} feat)", "#8c564b"),
    (y_v6, proba_v6, f"v6 (LGB t50+t75, {len(NUMERIC_V6_ALL)+1} feat)", "#e377c2"),
    (y_v7, proba_v7, f"v7 (LGB +Dyn, {len(NUMERIC_V7_ALL)+1} feat)", "#ff7f0e"),
]
fig_lgb, axes_lgb = plt.subplots(1, 4, figsize=(16, 4))
fig_lgb.suptitle("Calibration curves LightGBM (v4-v7)", fontsize=12, fontweight="bold")
for ax2, (y_true, proba, label, color) in zip(axes_lgb, panels_lgb):
    frac, mean = calibration_curve(y_true, proba, n_bins=10, strategy="uniform")
    ax2.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
    ax2.plot(mean, frac, "o-", color=color, linewidth=2, markersize=6, label=label)
    ax2.fill_between(mean, frac, mean, alpha=0.15, color=color)
    ll = log_loss(y_true, proba)
    bs = brier_score_loss(y_true, proba)
    ax2.set_title(f"{label}\nLog Loss={ll:.4f}  Brier={bs:.4f}", fontsize=9)
    ax2.set_xlabel("Mean predicted P(T wins)")
    ax2.set_ylabel("Actual fraction")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "calibration_lgb.png"), dpi=150, bbox_inches="tight"
)
plt.close()
print(f"Saved: {os.path.join(OUTPUT_DIR, 'calibration_lgb.png')}")
