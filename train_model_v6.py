#LightGBM trained on t50+t75 snapshots with sample weights.

import polars as pl
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
import joblib

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

DATA_FILE = r"C:\Users\hasan\Downloads\CS2Predictor\training_data.parquet"
MODEL_FILE = r"C:\Users\hasan\Downloads\CS2Predictor\cs2_model_v6.joblib"

SNAP_WEIGHTS = {"t50": 2.0, "t75": 1.5}  # t25 not used
ROUND_META = ["match_id", "round", MAP_FEATURE, "target"]


#Load and filter
print("Loading training_data_v5.parquet...")
df = pl.read_parquet(DATA_FILE)
df = df.filter(~pl.col(MAP_FEATURE).is_in(list(EXCLUDE_MAPS)))
df_pd = derive_runtime_features_v6(df.to_pandas())

print(
    f"Loaded: {df_pd.shape}  ({df_pd['match_id'].nunique()} matches, "
    f"{len(df_pd)} rounds)"
)


#Melt: t50 + t75
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
    result = pd.concat([meta, snap], axis=1)
    result["_snap"] = label
    return result


snap_t50 = extract_snap(df_pd, "", "t50")
snap_t75 = extract_snap(df_pd, "t75", "t75")

melted = pd.concat([snap_t50, snap_t75], ignore_index=True)
melted = add_delta_features(melted)

print(f"\nMelted (t50+t75): {melted.shape}")
print(
    f"  t50: {(melted['_snap']=='t50').sum()}  |  t75: {(melted['_snap']=='t75').sum()}"
)

#Feature list
ALL_NUMERIC = filter_present(V6_ALL_NUMERIC, melted)
missing = set(V6_ALL_NUMERIC) - set(ALL_NUMERIC)
if missing:
    print(f"WARNING -- skipped: {missing}")

le = LabelEncoder()
melted["map_encoded"] = le.fit_transform(melted[MAP_FEATURE])
ALL_FEATURES = ALL_NUMERIC + ["map_encoded"]

X = melted[ALL_FEATURES].values
y = melted["target"].values
groups = melted["match_id"].values
sample_weight = melted["_snap"].map(SNAP_WEIGHTS).values

print(f"\nFeatures: {len(ALL_FEATURES)}  " f"({len(ALL_NUMERIC)} numeric + map)")
print(f"Sample weights:  t50={SNAP_WEIGHTS['t50']}  t75={SNAP_WEIGHTS['t75']}")

#LightGBM
from lgb_config import LGB_V6_PARAMS

lgb_clf = lgb.LGBMClassifier(**LGB_V6_PARAMS)

#CV
print("\nGroupKFold CV (5 folds, t50+t75 weighted)...")
cv = GroupKFold(n_splits=5)
scores = cross_validate(
    lgb_clf,
    X,
    y,
    groups=groups,
    cv=cv,
    scoring=["accuracy", "neg_log_loss", "neg_brier_score", "roc_auc"],
)

print(f"\n{'Metric':<20} {'Mean':>8} {'Std':>8}   (t50+t75 weighted)")
print("-" * 54)
for name, key, sign in [
    ("Accuracy", "test_accuracy", 1),
    ("Log Loss", "test_neg_log_loss", -1),
    ("Brier Score", "test_neg_brier_score", -1),
    ("ROC-AUC", "test_roc_auc", 1),
]:
    v = sign * scores[key]
    print(f"  {name:<18} {v.mean():>8.4f} {v.std():>8.4f}")

#CV t50 only
print("\nGroupKFold CV (t50 only -- v4 comparison)...")
m50 = melted["_snap"] == "t50"
scores50 = cross_validate(
    lgb_clf,
    melted.loc[m50, ALL_FEATURES].values,
    melted.loc[m50, "target"].values,
    groups=melted.loc[m50, "match_id"].values,
    cv=GroupKFold(n_splits=5),
    scoring=["accuracy", "neg_log_loss", "neg_brier_score", "roc_auc"],
)

print(f"\n{'Metric':<20} {'Mean':>8} {'Std':>8}   (t50 only)")
print("-" * 54)
for name, key, sign in [
    ("Accuracy", "test_accuracy", 1),
    ("Log Loss", "test_neg_log_loss", -1),
    ("Brier Score", "test_neg_brier_score", -1),
    ("ROC-AUC", "test_roc_auc", 1),
]:
    v = sign * scores50[key]
    print(f"  {name:<18} {v.mean():>8.4f} {v.std():>8.4f}")

#Train on full dataset
print("\nTraining final model on full dataset (t50+t75, weighted)...")
lgb_clf.fit(X, y, sample_weight=sample_weight)

#Feature importance
ranked = sorted(
    zip(ALL_FEATURES, lgb_clf.feature_importances_), key=lambda x: x[1], reverse=True
)
print(f"\n{'Feature':<35} {'Importance':>10}")
print("-" * 47)
for feat, imp in ranked[:30]:
    print(f"  {feat if feat != 'map_encoded' else 'map_name':<33} {imp:>10.0f}")
if len(ranked) > 30:
    print(f"  ... ({len(ranked) - 30} more)")

#Save
joblib.dump(
    {
        "model": lgb_clf,
        "numeric_features": ALL_NUMERIC,
        "map_feature": MAP_FEATURE,
        "map_encoder": le,
        "all_features": ALL_FEATURES,
    },
    MODEL_FILE,
)
print(f"\nSaved: {MODEL_FILE}")
