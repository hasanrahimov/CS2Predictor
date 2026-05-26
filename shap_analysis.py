# Global SHAP summary plots for v4-v7.

import warnings

warnings.filterwarnings("ignore")
import os
import joblib
import shap
import pandas as pd
import polars as pl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features import (
    V3_NUMERIC,
    V5_NUMERIC,
    V6_PER_SNAP,
    V6_ROUND_CTX,
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


def shap_plot(model_path, X, feature_names, title, out_path, max_display=20):
    bundle = joblib.load(model_path)
    model = bundle["model"]
    print(f"Computing SHAP for {title}...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        sv = shap_vals[1]
    else:
        sv = shap_vals
    feat_display = [f if f != "map_encoded" else "map_name" for f in feature_names]
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        sv,
        X,
        feature_names=feat_display,
        max_display=max_display,
        show=False,
        plot_size=None,
        color_bar_label="Pazīmes vērtība",
    )
    plt.title(f"SHAP {title}", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("SHAP vērtība  (pozitīva -> T uzvar, negatīva -> CT uzvar)", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


#Load base data
df = (
    pl.read_parquet(DATA_FILE)
    .filter(~pl.col(MAP_FEATURE).is_in(list(EXCLUDE_MAPS)))
    .to_pandas()
)
df = derive_runtime_features(df)

NUMERIC_V3 = filter_present(V3_NUMERIC, df)
NUMERIC_V5 = filter_present(V5_NUMERIC, df)

#v4:
bundle_v4 = joblib.load(r"C:\Users\hasan\Downloads\CS2Predictor\cs2_model_v4.joblib")
feats_v4 = bundle_v4["all_features"]  # NUMERIC_V3 + ["map_encoded"]
le_v4 = bundle_v4["map_encoder"]
df["map_encoded"] = le_v4.transform(df[MAP_FEATURE])
X_v4 = df[feats_v4].values

shap_plot(
    r"C:\Users\hasan\Downloads\CS2Predictor\cs2_model_v4.joblib",
    X_v4,
    feats_v4,
    "v4 (LightGBM, 33 pazīmes arhitektūras izolācija)",
    os.path.join(OUTPUT_DIR, "shap_v4.png"),
)

#v5
bundle_v5 = joblib.load(r"C:\Users\hasan\Downloads\CS2Predictor\cs2_model_v5.joblib")
feats_v5 = bundle_v5["all_features"]  # NUMERIC_V5 + ["map_encoded"]
le_v5 = bundle_v5["map_encoder"]
df["map_encoded"] = le_v5.transform(df[MAP_FEATURE])
X_v5 = df[feats_v5].values

shap_plot(
    r"C:\Users\hasan\Downloads\CS2Predictor\cs2_model_v5.joblib",
    X_v5,
    feats_v5,
    "v5 (LightGBM, 37 pazīmes taktiskās pazīmes)",
    os.path.join(OUTPUT_DIR, "shap_v5.png"),
)


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
melted_v6 = _melt_v6_local(_raw6)

bundle_v6 = joblib.load(r"C:\Users\hasan\Downloads\CS2Predictor\cs2_model_v6.joblib")
feats_v6 = bundle_v6["all_features"]
le_v6 = bundle_v6["map_encoder"]
melted_v6["map_encoded"] = le_v6.transform(melted_v6[MAP_FEATURE])
sample_v6 = melted_v6[feats_v6].sample(n=min(10_000, len(melted_v6)), random_state=42)

shap_plot(
    r"C:\Users\hasan\Downloads\CS2Predictor\cs2_model_v6.joblib",
    sample_v6.values,
    feats_v6,
    "v6 (LightGBM, t50+t75, delta pazīmes, 41 pazīme)",
    os.path.join(OUTPUT_DIR, "shap_v6.png"),
)


#v7
_raw7 = (
    pl.read_parquet(DATA_FILE)
    .filter(~pl.col(MAP_FEATURE).is_in(list(EXCLUDE_MAPS)))
    .to_pandas()
)
_raw7 = derive_runtime_features_v7(_raw7)
melted_v7 = melt_v7(_raw7)

bundle_v7 = joblib.load(r"C:\Users\hasan\Downloads\CS2Predictor\cs2_model_v7.joblib")
feats_v7 = bundle_v7["all_features"]
le_v7 = bundle_v7["map_encoder"]
melted_v7["map_encoded"] = le_v7.transform(melted_v7[MAP_FEATURE])
sample_v7 = melted_v7[feats_v7].sample(n=min(10_000, len(melted_v7)), random_state=42)

shap_plot(
    r"C:\Users\hasan\Downloads\CS2Predictor\cs2_model_v7.joblib",
    sample_v7.values,
    feats_v7,
    "v7 (LightGBM, t50+t75, dinamika + ieroči + taktika, 71 pazīme)",
    os.path.join(OUTPUT_DIR, "shap_v7.png"),
)

print("\nDone. Saved: shap_v4.png, shap_v5.png, shap_v6.png, shap_v7.png")
