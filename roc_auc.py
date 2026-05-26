#ROC curves and AUC scores for v1-v7.

import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import polars as pl
import pandas as pd
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_curve, auc

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
cv = GroupKFold(n_splits=5)

from lgb_config import LGB_V4_PARAMS, LGB_V5_PARAMS, LGB_V6_PARAMS, LGB_V7_PARAMS


def lgb_clf(params):
    return lgb.LGBMClassifier(**params)


#Out-of-fold predictions v1-v5
print("v1 (LR, 24 feat)...")
proba_v1 = cross_val_predict(
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ]
    ),
    df_pd[NUMERIC_V1_ALL].values,
    y,
    cv=cv,
    groups=groups,
    method="predict_proba",
)[:, 1]

print("v2 (LR, 24+map)...")
proba_v2 = cross_val_predict(
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
    cv=cv,
    groups=groups,
    method="predict_proba",
)[:, 1]

print("v3 (LR, 32+map)...")
proba_v3 = cross_val_predict(
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
    cv=cv,
    groups=groups,
    method="predict_proba",
)[:, 1]

print("v4 (LGB, 32+map architecture isolation)...")
proba_v4 = cross_val_predict(
    lgb_clf(LGB_V4_PARAMS),
    df_pd[NUMERIC_V3_ALL + ["map_encoded"]].values,
    y,
    cv=cv,
    groups=groups,
    method="predict_proba",
)[:, 1]

print("v5 (LGB, 36+map tactical features)...")
proba_v5 = cross_val_predict(
    lgb_clf(LGB_V5_PARAMS),
    df_pd[NUMERIC_V5_ALL + ["map_encoded"]].values,
    y,
    cv=cv,
    groups=groups,
    method="predict_proba",
)[:, 1]


#v6
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


_raw6 = (
    pl.read_parquet(DATA_FILE)
    .filter(~pl.col(MAP_FEATURE).is_in(list(EXCLUDE_MAPS)))
    .to_pandas()
)
_raw6 = derive_runtime_features_v6(_raw6)
melted_v6_data = _melt_v6_local(_raw6)
NUMERIC_V6_MELT = filter_present(V6_ALL_NUMERIC, melted_v6_data)
le_v6 = LabelEncoder()
melted_v6_data["map_encoded"] = le_v6.fit_transform(melted_v6_data[MAP_FEATURE])
y_v6 = melted_v6_data["target"].values
groups_v6 = melted_v6_data["match_id"].values
print(f"v6 (LGB t50+t75+Δ, {len(NUMERIC_V6_MELT)+1} feat)...")
proba_v6 = cross_val_predict(
    lgb_clf(LGB_V6_PARAMS),
    melted_v6_data[NUMERIC_V6_MELT + ["map_encoded"]].values,
    y_v6,
    cv=GroupKFold(n_splits=5),
    groups=groups_v6,
    method="predict_proba",
)[:, 1]


#v7
_raw7 = (
    pl.read_parquet(DATA_FILE)
    .filter(~pl.col(MAP_FEATURE).is_in(list(EXCLUDE_MAPS)))
    .to_pandas()
)
_raw7 = derive_runtime_features_v7(_raw7)
melted_v7_data = melt_v7(_raw7)
NUMERIC_V7_MELT = filter_present(V7_ALL_NUMERIC, melted_v7_data)
le_v7 = LabelEncoder()
melted_v7_data["map_encoded"] = le_v7.fit_transform(melted_v7_data[MAP_FEATURE])
y_v7 = melted_v7_data["target"].values
groups_v7 = melted_v7_data["match_id"].values
print(f"v7 (LGB +Dyn/Wpn/Econ/Tac, {len(NUMERIC_V7_MELT)+1} feat)...")
proba_v7 = cross_val_predict(
    lgb_clf(LGB_V7_PARAMS),
    melted_v7_data[NUMERIC_V7_MELT + ["map_encoded"]].values,
    y_v7,
    cv=GroupKFold(n_splits=5),
    groups=groups_v7,
    method="predict_proba",
)[:, 1]


#ROC and AUC
fpr_v1, tpr_v1, _ = roc_curve(y, proba_v1)
fpr_v2, tpr_v2, _ = roc_curve(y, proba_v2)
fpr_v3, tpr_v3, _ = roc_curve(y, proba_v3)
fpr_v4, tpr_v4, _ = roc_curve(y, proba_v4)
fpr_v5, tpr_v5, _ = roc_curve(y, proba_v5)
fpr_v6, tpr_v6, _ = roc_curve(y_v6, proba_v6)
fpr_v7, tpr_v7, _ = roc_curve(y_v7, proba_v7)

aucs = {
    "v1": auc(fpr_v1, tpr_v1),
    "v2": auc(fpr_v2, tpr_v2),
    "v3": auc(fpr_v3, tpr_v3),
    "v4": auc(fpr_v4, tpr_v4),
    "v5": auc(fpr_v5, tpr_v5),
    "v6": auc(fpr_v6, tpr_v6),
    "v7": auc(fpr_v7, tpr_v7),
}

print("\nAUC (out-of-fold):")
for k, v in aucs.items():
    print(f"  {k}: {v:.4f}")

#Plot
fig, ax = plt.subplots(figsize=(8, 6))

styles = [
    (fpr_v1, tpr_v1, "#4C72B0", "-", f"v1 LR, 24 feat         (AUC={aucs['v1']:.3f})"),
    (
        fpr_v2,
        tpr_v2,
        "#2ca02c",
        "--",
        f"v2 LR, 25 feat          (AUC={aucs['v2']:.3f})",
    ),
    (fpr_v3, tpr_v3, "#d62728", "-", f"v3 LR, 33 feat          (AUC={aucs['v3']:.3f})"),
    (
        fpr_v4,
        tpr_v4,
        "#9467bd",
        "--",
        f"v4 LGB, 33 feat          (AUC={aucs['v4']:.3f})",
    ),
    (
        fpr_v5,
        tpr_v5,
        "#8c564b",
        "-",
        f"v5 LGB, 37 feat          (AUC={aucs['v5']:.3f})",
    ),
    (
        fpr_v6,
        tpr_v6,
        "#e377c2",
        "--",
        f"v6 LGB t50+t75, {len(NUMERIC_V6_MELT)+1} feat  (AUC={aucs['v6']:.3f})",
    ),
    (
        fpr_v7,
        tpr_v7,
        "#ff7f0e",
        "-",
        f"v7 LGB +Dyn/Wpn, {len(NUMERIC_V7_MELT)+1} feat  (AUC={aucs['v7']:.3f})",
    ),
]
for fpr, tpr, color, ls, label in styles:
    ax.plot(fpr, tpr, ls, color=color, linewidth=2, label=label)
ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC=0.500)")

ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title(
    "ROC curves v1 \u2192 v7\n(out-of-fold, 5-fold GroupKFold CV)",
    fontsize=12,
    fontweight="bold",
)
ax.legend(loc="lower right", fontsize=8)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "roc_auc.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
