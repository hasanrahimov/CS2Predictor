#Sensitivity analysis for sample weights (t50 / t75).


import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.preprocessing import LabelEncoder

from features import (
    V6_PER_SNAP as PER_SNAP,
    V6_ROUND_CTX as ROUND_CTX,
    V6_ALL_NUMERIC,
    MAP_FEATURE,
    EXCLUDE_MAPS,
    derive_runtime_features_v6,
    add_delta_features,
    filter_present,
)
from lgb_config import LGB_V6_PARAMS

DATA_FILE = r"C:\Users\hasan\Downloads\CS2Predictor\training_data.parquet"

#Weight grid
W50_VALS = [1.0, 1.5, 2.0, 3.0, 4.0]
W75_VALS = [0.5, 1.0, 1.5, 2.0]
ROUND_META = ["match_id", "round", MAP_FEATURE, "target"]

#Load and melt
print("Loading data...")
df = pl.read_parquet(DATA_FILE)
df = df.filter(~pl.col(MAP_FEATURE).is_in(list(EXCLUDE_MAPS)))
df_pd = derive_runtime_features_v6(df.to_pandas())
print(f"  {len(df_pd)} rounds, {df_pd['match_id'].nunique()} matches")


def extract_snap(df: pd.DataFrame, suffix: str, label: str) -> pd.DataFrame:
    meta = df[ROUND_META + ROUND_CTX].copy().reset_index(drop=True)
    if suffix == "":
        snap = (
            df[[c for c in PER_SNAP if c in df.columns]].copy().reset_index(drop=True)
        )
    else:
        rename = {f"{c}_{suffix}": c for c in PER_SNAP[:-1]}
        rename[f"bomb_planted_now_{suffix}"] = "bomb_planted_now"
        cols = [f"{c}_{suffix}" for c in PER_SNAP[:-1]] + [f"bomb_planted_now_{suffix}"]
        snap = (
            df[[c for c in cols if c in df.columns]]
            .rename(columns=rename)
            .reset_index(drop=True)
        )
    out = pd.concat([meta, snap], axis=1)
    out["_snap"] = label
    return out


snap_t50 = extract_snap(df_pd, "", "t50")
snap_t75 = extract_snap(df_pd, "t75", "t75")
melted = pd.concat([snap_t50, snap_t75], ignore_index=True)
melted = add_delta_features(melted)
print(
    f"  Melted: {len(melted)} rows  "
    f"(t50={( melted['_snap']=='t50').sum()}, t75={(melted['_snap']=='t75').sum()})"
)

le = LabelEncoder()
melted["map_encoded"] = le.fit_transform(melted[MAP_FEATURE])
ALL_NUMERIC = filter_present(V6_ALL_NUMERIC, melted)
ALL_FEATURES = ALL_NUMERIC + ["map_encoded"]

X = melted[ALL_FEATURES].values
y = melted["target"].values
groups = melted["match_id"].values
is_t50 = (melted["_snap"] == "t50").values

#CV loop
cv = GroupKFold(n_splits=5)

#Run 5-fold CV with given weights; evaluate on t50 val rows only
def run_cv(w50: float, w75: float) -> dict:
    sw_map = np.where(is_t50, w50, w75)
    accs, lls, bss = [], [], []
    for train_idx, val_idx in cv.split(X, y, groups):
        clf = lgb.LGBMClassifier(**LGB_V6_PARAMS)
        clf.fit(X[train_idx], y[train_idx], sample_weight=sw_map[train_idx])
        # Evaluate only on t50 subset of validation fold
        val_t50 = val_idx[is_t50[val_idx]]
        if len(val_t50) == 0:
            continue
        proba = clf.predict_proba(X[val_t50])[:, 1]
        yv = y[val_t50]
        accs.append(accuracy_score(yv, (proba >= 0.5).astype(int)))
        lls.append(log_loss(yv, proba))
        bss.append(brier_score_loss(yv, proba))
    return {
        "acc": np.mean(accs),
        "ll": np.mean(lls),
        "brier": np.mean(bss),
    }


#Run grid
print(
    f"\nRunning {len(W50_VALS) * len(W75_VALS)} weight combinations "
    "(5-fold CV each, t50-only eval)..."
)
print("-" * 70)

results = []
for w50 in W50_VALS:
    for w75 in W75_VALS:
        r = run_cv(w50, w75)
        results.append({"w50": w50, "w75": w75, **r})
        marker = " <-- current" if w50 == 2.0 and w75 == 1.5 else ""
        print(
            f"  w50={w50:.1f}  w75={w75:.1f}  | "
            f"Acc={r['acc']:.4f}  LL={r['ll']:.4f}  Brier={r['brier']:.4f}{marker}"
        )

#Summary
res_df = pd.DataFrame(results)
best_acc = res_df.loc[res_df["acc"].idxmax()]
best_ll = res_df.loc[res_df["ll"].idxmin()]
best_brier = res_df.loc[res_df["brier"].idxmin()]

print("\n" + "=" * 70)
print(
    "BEST by Accuracy  : "
    f"w50={best_acc['w50']:.1f}  w75={best_acc['w75']:.1f}  "
    f"Acc={best_acc['acc']:.4f}"
)
print(
    "BEST by Log Loss  : "
    f"w50={best_ll['w50']:.1f}  w75={best_ll['w75']:.1f}  "
    f"LL={best_ll['ll']:.4f}"
)
print(
    "BEST by Brier     : "
    f"w50={best_brier['w50']:.1f}  w75={best_brier['w75']:.1f}  "
    f"Brier={best_brier['brier']:.4f}"
)

#Current combination
cur = res_df[(res_df["w50"] == 2.0) & (res_df["w75"] == 1.5)].iloc[0]
print(
    "\nCurrent (2.0/1.5) : "
    f"Acc={cur['acc']:.4f}  LL={cur['ll']:.4f}  Brier={cur['brier']:.4f}"
)

#Rank of current combination
rank_acc = (res_df["acc"] > cur["acc"]).sum() + 1
rank_ll = (res_df["ll"] < cur["ll"]).sum() + 1
rank_brier = (res_df["brier"] < cur["brier"]).sum() + 1
n = len(res_df)
print(
    f"Rank among {n}     : "
    f"Acc rank={rank_acc}/{n}  LL rank={rank_ll}/{n}  Brier rank={rank_brier}/{n}"
)

#Accuracy pivot table
print("\nAccuracy grid (rows=w50, cols=w75):")
pivot = res_df.pivot(index="w50", columns="w75", values="acc")
print(pivot.to_string(float_format="{:.4f}".format))

print("\nLog Loss grid (rows=w50, cols=w75):")
pivot_ll = res_df.pivot(index="w50", columns="w75", values="ll")
print(pivot_ll.to_string(float_format="{:.4f}".format))

print("\nDone.")
