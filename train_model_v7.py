#LightGBM v7: v6 melt + dynamics, weapons, econ, tactical.


import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
import joblib

from features import (
    V7_ALL_NUMERIC,
    V7_DYN_BASE,
    MAP_FEATURE,
    EXCLUDE_MAPS,
    derive_runtime_features_v7,
    melt_v7,
    filter_present,
)

DATA_FILE = r"C:\Users\hasan\Downloads\CS2Predictor\training_data.parquet"
MODEL_FILE = r"C:\Users\hasan\Downloads\CS2Predictor\cs2_model_v7.joblib"

SNAP_WEIGHTS = {"t50": 2.0, "t75": 1.5}


#Load and filter
print("Loading training_data.parquet...")
df = pl.read_parquet(DATA_FILE)
df = df.filter(~pl.col(MAP_FEATURE).is_in(list(EXCLUDE_MAPS)))
df_pd = df.to_pandas()
df_pd = derive_runtime_features_v7(
    df_pd
)

print(
    f"Loaded: {df_pd.shape}  ({df_pd['match_id'].nunique()} matches, "
    f"{len(df_pd)} rounds)"
)


#Melt: t50 + t75 with dynamics deltas 
melted = melt_v7(df_pd)

print(f"\nMelted (t50+t75): {melted.shape}")
print(
    f"  t50: {(melted['_snap']=='t50').sum()}  |  t75: {(melted['_snap']=='t75').sum()}"
)


#Feature list
ALL_NUMERIC = filter_present(V7_ALL_NUMERIC, melted)
missing = set(V7_ALL_NUMERIC) - set(ALL_NUMERIC)
if missing:
    print(
        f"WARNING -- {len(missing)} v7 features missing (will skip): {sorted(missing)}"
    )

le = LabelEncoder()
melted["map_encoded"] = le.fit_transform(melted[MAP_FEATURE])
ALL_FEATURES = ALL_NUMERIC + ["map_encoded"]

X = melted[ALL_FEATURES].values
y = melted["target"].values
groups = melted["match_id"].values
sample_weight = melted["_snap"].map(SNAP_WEIGHTS).values

print(f"\nFeatures: {len(ALL_FEATURES)}  ({len(ALL_NUMERIC)} numeric + map)")
print(f"Sample weights:  t50={SNAP_WEIGHTS['t50']}  t75={SNAP_WEIGHTS['t75']}")


#LightGBM
from lgb_config import LGB_V7_PARAMS

lgb_clf = lgb.LGBMClassifier(**LGB_V7_PARAMS)


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
print("\nGroupKFold CV (t50 only -- v4/v5 comparison)...")
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
for feat, imp in ranked[:35]:
    print(f"  {feat if feat != 'map_encoded' else 'map_name':<33} {imp:>10.0f}")
if len(ranked) > 35:
    print(f"  ... ({len(ranked) - 35} more)")

#Save
joblib.dump(
    {
        "model": lgb_clf,
        "numeric_features": ALL_NUMERIC,
        "map_feature": MAP_FEATURE,
        "map_encoder": le,
        "all_features": ALL_FEATURES,
        "dyn_base": V7_DYN_BASE,
        "snap_weights": SNAP_WEIGHTS,
    },
    MODEL_FILE,
)
print(f"\nSaved: {MODEL_FILE}")
