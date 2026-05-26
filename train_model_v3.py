#Extended model with contextual features

import polars as pl
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_validate
import joblib

from features import (
    V2_NUMERIC,
    V3_NUMERIC,
    MAP_FEATURE,
    EXCLUDE_MAPS,
    derive_runtime_features,
    filter_present,
)

DATA_FILE = r"C:\Users\hasan\Downloads\CS2Predictor\training_data.parquet"
MODEL_FILE = r"C:\Users\hasan\Downloads\CS2Predictor\cs2_model_v3.joblib"

#Load and filter
df = pl.read_parquet(DATA_FILE)
df = df.filter(~pl.col(MAP_FEATURE).is_in(list(EXCLUDE_MAPS)))
print(f"Dataset shape: {df.shape}")
print(f"Maps: {sorted(df[MAP_FEATURE].unique().to_list())}")

df_pd = derive_runtime_features(df.to_pandas())

ALL_NUMERIC = filter_present(V3_NUMERIC, df_pd)
NEW_PRESENT = [f for f in ALL_NUMERIC if f not in V2_NUMERIC]
print(f"\nNew features included ({len(NEW_PRESENT)}): {NEW_PRESENT}")
print(f"Total numeric features: {len(ALL_NUMERIC)}")

#Feature coverage
print("\nNew feature coverage (non-zero rounds):")
for col in NEW_PRESENT:
    non_zero = int((df_pd[col] != 0).sum())
    print(f"  {col:<25} {non_zero:>6}/{len(df_pd)} ({non_zero/len(df_pd)*100:.1f}%)")

X = df_pd[ALL_NUMERIC + [MAP_FEATURE]]
y = df_pd["target"].values
groups = df_pd["match_id"].values

#Pipeline
preprocessor = ColumnTransformer(
    [
        ("num", StandardScaler(), ALL_NUMERIC),
        (
            "map",
            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            [MAP_FEATURE],
        ),
    ]
)
pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ]
)

#Grouped CV
print("\nRunning GroupKFold CV (5 folds)...")
cv = GroupKFold(n_splits=5)
scores = cross_validate(
    pipeline,
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
print(f"{'Accuracy':<20} {acc.mean():>8.4f} {acc.std():>8.4f}")
print(f"{'Log Loss':<20} {ll.mean():>8.4f} {ll.std():>8.4f}")
print(f"{'Brier Score':<20} {brier.mean():>8.4f} {brier.std():>8.4f}")
print("\n(v1 baseline: Accuracy=0.6636, Brier=0.2104)")
print("(v2 with map: Accuracy=0.6645, Brier=0.2104)")
print("(v2 with map: Accuracy=0.6642, Brier=0.2105)")

#Train model
print("\nTraining final model on full dataset...")
pipeline.fit(X, y)

#Feature importance
ohe_cats = (
    pipeline.named_steps["preprocessor"].named_transformers_["map"].categories_[0][1:]
)
all_names = ALL_NUMERIC + [f"map_{c}" for c in ohe_cats]
coefs = pipeline.named_steps["clf"].coef_[0]
ranked = sorted(zip(all_names, coefs), key=lambda x: abs(x[1]), reverse=True)

print(f"\n{'Feature':<30} {'Coef':>8}  Direction")
print("-" * 52)
for feat, coef in ranked:
    direction = "T wins" if coef > 0 else "CT wins"
    marker = " *" if feat in NEW_PRESENT else ""
    print(f"  {feat:<28} {coef:>+.4f}  {direction}{marker}")
print("\n* = new feature in v3")

#Save
joblib.dump(
    {
        "model": pipeline,
        "numeric_features": ALL_NUMERIC,
        "map_feature": MAP_FEATURE,
        "new_features": NEW_PRESENT,
    },
    MODEL_FILE,
)
print(f"\nSaved to {MODEL_FILE}")
