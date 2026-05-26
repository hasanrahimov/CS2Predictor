import polars as pl
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_validate
import joblib

from features import (
    V1_NUMERIC,
    MAP_FEATURE,
    EXCLUDE_MAPS,
    derive_runtime_features,
    filter_present,
)

DATA_FILE = r"C:\Users\hasan\Downloads\CS2Predictor\training_data.parquet"
MODEL_FILE = r"C:\Users\hasan\Downloads\CS2Predictor\cs2_model.joblib"

#Load
df = pl.read_parquet(DATA_FILE)
df = df.filter(~pl.col(MAP_FEATURE).is_in(list(EXCLUDE_MAPS)))
df_pd = derive_runtime_features(df.to_pandas())

FEATURES = filter_present(V1_NUMERIC, df_pd)
print(f"Using {len(FEATURES)} features")

X = df_pd[FEATURES].to_numpy().astype(float)
y = df_pd["target"].values
groups = df_pd["match_id"].values

#Pipeline
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ]
)

#Grouped CV
print("Running GroupKFold cross-validation (5 folds)...")
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
print(f"{'Accuracy':<20} {acc.mean():>8.3f} {acc.std():>8.3f}")
print(f"{'Log Loss':<20} {ll.mean():>8.3f} {ll.std():>8.3f}")
print(f"{'Brier Score':<20} {brier.mean():>8.3f} {brier.std():>8.3f}")
print("\n(Brier: lower=better, perfect=0.0, random=0.25)")

#Train on full data
print("\nTraining final model on full dataset...")
pipeline.fit(X, y)

#Feature importance
coefs = pipeline.named_steps["clf"].coef_[0]
importance = sorted(zip(FEATURES, coefs), key=lambda x: abs(x[1]), reverse=True)
print(f"\n{'Feature':<35} {'Coef':>8}  Direction")
print("-" * 60)
for feat, coef in importance:
    direction = "-> T wins" if coef > 0 else "-> CT wins"
    print(f"  {feat:<33} {coef:>+.3f}  {direction}")

#Save
joblib.dump({"model": pipeline, "features": FEATURES}, MODEL_FILE)
print(f"\nSaved to {MODEL_FILE}")
