#Empirical test of t25 exclusion claim.


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

#Load
print("Loading data...")
df = pl.read_parquet(DATA_FILE)
df = df.filter(~pl.col(MAP_FEATURE).is_in(list(EXCLUDE_MAPS)))
df_pd = df.to_pandas()

#Derive bomb_planted_now_t25 from bomb_countdown_t25
if "bomb_countdown_t25" in df_pd.columns:
    df_pd["bomb_planted_now_t25"] = (df_pd["bomb_countdown_t25"] < 40.0).astype(int)

derive_runtime_features_v6(df_pd)  # derives t50 + t75 bomb_planted_now
print(f"  {len(df_pd)} rounds, {df_pd['match_id'].nunique()} matches")

ROUND_META = ["match_id", "round", MAP_FEATURE, "target"]

#Pull one snapshot (suffix '' = t50, 't75', 't25') into a long row
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


snap_t25 = extract_snap(df_pd, "t25", "t25")
snap_t50 = extract_snap(df_pd, "", "t50")
snap_t75 = extract_snap(df_pd, "t75", "t75")

#Combine snapshot frames, add delta features, encode map
def make_dataset(snaps_with_weights: list):
    frames = [s for s, _ in snaps_with_weights]
    weights = {
        label: w for s, w in snaps_with_weights for label in [s["_snap"].iloc[0]]
    }
    melted = pd.concat(frames, ignore_index=True)
    melted = add_delta_features(melted)
    le = LabelEncoder()
    melted["map_encoded"] = le.fit_transform(melted[MAP_FEATURE])
    ALL_NUM = filter_present(V6_ALL_NUMERIC, melted)
    ALL_FEAT = ALL_NUM + ["map_encoded"]
    X = melted[ALL_FEAT].values
    y = melted["target"].values
    groups = melted["match_id"].values
    sw = melted["_snap"].map(weights).fillna(1.0).values
    is_t50 = (melted["_snap"] == "t50").values
    return X, y, groups, sw, is_t50, ALL_FEAT

# Define CV function that evaluates only on t50 rows
def run_cv(X, y, groups, sw, is_t50, label):
    cv = GroupKFold(n_splits=5)
    accs, lls, bss = [], [], []
    for train_idx, val_idx in cv.split(X, y, groups):
        clf = lgb.LGBMClassifier(**LGB_V6_PARAMS)
        clf.fit(X[train_idx], y[train_idx], sample_weight=sw[train_idx])
        val_t50 = val_idx[is_t50[val_idx]]
        proba = clf.predict_proba(X[val_t50])[:, 1]
        yv = y[val_t50]
        accs.append(accuracy_score(yv, (proba >= 0.5).astype(int)))
        lls.append(log_loss(yv, proba))
        bss.append(brier_score_loss(yv, proba))
    print(
        f"  {label:<35} Acc={np.mean(accs):.4f}  LL={np.mean(lls):.4f}"
        f"  Brier={np.mean(bss):.4f}"
    )
    return np.mean(accs), np.mean(lls), np.mean(bss)


#Experiment A: t50 + t75
print("\nRunning 5-fold CV (t50-only eval) for each configuration...")
print("-" * 70)
Xa, ya, ga, swa, is_t50a, _ = make_dataset(
    [
        (snap_t50, 2.0),
        (snap_t75, 1.5),
    ]
)
acc_a, ll_a, bs_a = run_cv(Xa, ya, ga, swa, is_t50a, "A: t50+t75 (w=2.0/1.5) [current]")

#Experiment B: t50 + t75 + t25 (w25=1.0)
Xb, yb, gb, swb, is_t50b, _ = make_dataset(
    [
        (snap_t25, 1.0),
        (snap_t50, 2.0),
        (snap_t75, 1.5),
    ]
)
acc_b, ll_b, bs_b = run_cv(Xb, yb, gb, swb, is_t50b, "B: t25+t50+t75 (w=1.0/2.0/1.5)")

#Experiment C: t50 + t75 + t25 (w25=0.5, soft)
Xc, yc, gc, swc, is_t50c, _ = make_dataset(
    [
        (snap_t25, 0.5),
        (snap_t50, 2.0),
        (snap_t75, 1.5),
    ]
)
acc_c, ll_c, bs_c = run_cv(Xc, yc, gc, swc, is_t50c, "C: t25+t50+t75 (w=0.5/2.0/1.5)")


print("\n" + "=" * 70)
print("SUMMARY  (eval on t50 validation rows only)")
print(f"  A [no t25]          Acc={acc_a:.4f}  LL={ll_a:.4f}  Brier={bs_a:.4f}")
print(f"  B [t25 w=1.0]       Acc={acc_b:.4f}  LL={ll_b:.4f}  Brier={bs_b:.4f}")
print(f"  C [t25 w=0.5]       Acc={acc_c:.4f}  LL={ll_c:.4f}  Brier={bs_c:.4f}")
print()
print(f"  B vs A  delta Acc = {acc_b - acc_a:+.4f}  ({(acc_b-acc_a)*100:+.2f} pp)")
print(f"  C vs A  delta Acc = {acc_c - acc_a:+.4f}  ({(acc_c-acc_a)*100:+.2f} pp)")
print()
if acc_b < acc_a and acc_c < acc_a:
    print("  => t25 iekļaušana pasliktina t50 precizitati (apstiprina izslēgšanu)")
elif acc_b > acc_a or acc_c > acc_a:
    print("  => t25 iekļaušana uzlabo t50 precizitati (apstridama izslegšana)")
else:
    print("  => t25 iekļaušana neietekmē t50 precizitati")
print("Done.")
