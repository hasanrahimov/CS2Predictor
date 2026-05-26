#LightGBM with v3 features + tactical snapshot features.

import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
import joblib

from features import (
    V5_NUMERIC,
    MAP_FEATURE,
    EXCLUDE_MAPS,
    derive_runtime_features,
    filter_present,
)

DATA_FILE = r"C:\Users\hasan\Downloads\CS2Predictor\training_data.parquet"
MODEL_FILE = r"C:\Users\hasan\Downloads\CS2Predictor\cs2_model_v5.joblib"

#Load and filter
df = pl.read_parquet(DATA_FILE)
df = df.filter(~pl.col(MAP_FEATURE).is_in(list(EXCLUDE_MAPS)))
print(f"Dataset shape: {df.shape}")

df_pd = derive_runtime_features(df.to_pandas())
ALL_NUMERIC = filter_present(V5_NUMERIC, df_pd)
print(f"Using {len(ALL_NUMERIC)} numeric features + map_name")

#Label-encode map
le = LabelEncoder()
df_pd["map_encoded"] = le.fit_transform(df_pd[MAP_FEATURE])

ALL_FEATURES = ALL_NUMERIC + ["map_encoded"]
X = df_pd[ALL_FEATURES].values
y = df_pd["target"].values
groups = df_pd["match_id"].values

#LightGBM
from lgb_config import LGB_V4_PARAMS

lgb_clf = lgb.LGBMClassifier(**LGB_V4_PARAMS)

#Grouped CV
print("\nRunning GroupKFold cross-validation (5 folds)...")
cv = GroupKFold(n_splits=5)
scores = cross_validate(
    lgb_clf,
    X,
    y,
    groups=groups,
    cv=cv,
    scoring=["accuracy", "neg_log_loss", "neg_brier_score"],
)

acc = scores["test_accuracy"]
ll = -scores["test_neg_log_loss"]
brier = -scores["test_neg_brier_score"]

print(f"\n{'Metric':<20} {'Mean':>8} {'Std':>8}")
print("-" * 38)
print(f"{'Accuracy':<20} {acc.mean():>8.3f} {acc.std():>8.3f}")
print(f"{'Log Loss':<20} {ll.mean():>8.3f} {ll.std():>8.3f}")
print(f"{'Brier Score':<20} {brier.mean():>8.3f} {brier.std():>8.3f}")
print("\n(Brier: lower=better, perfect=0.0, random=0.25)")

#Train on full data
print("\nTraining final model on full dataset...")
lgb_clf.fit(X, y)

#Feature importance
importances = lgb_clf.feature_importances_
ranked = sorted(zip(ALL_FEATURES, importances), key=lambda x: x[1], reverse=True)
print(f"\n{'Feature':<35} {'Importance':>10}")
print("-" * 47)
for feat, imp in ranked:
    name = feat if feat != "map_encoded" else "map_name"
    print(f"  {name:<33} {imp:>10.0f}")

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
print(f"\nSaved to {MODEL_FILE}")
