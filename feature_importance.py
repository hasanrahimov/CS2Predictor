
#Feature importance for all models.

import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#v1-v3: Logistic Regression coefficients
LR_MODELS = [
    (
        r"C:\Users\hasan\Downloads\CS2Predictor\cs2_model.joblib",
        "v1 (25 features, no map)",
    ),
    (
        r"C:\Users\hasan\Downloads\CS2Predictor\cs2_model_v2.joblib",
        "v2 (25 features + map)",
    ),
    (
        r"C:\Users\hasan\Downloads\CS2Predictor\cs2_model_v3.joblib",
        "v3 (33 features + map)",
    ),
]


def get_lr_feature_names(bundle):
    if "features" in bundle:
        return bundle["features"]
    ohe_cats = (
        bundle["model"]
        .named_steps["preprocessor"]
        .named_transformers_["map"]
        .categories_[0][1:]
    )
    return bundle["numeric_features"] + [f"map_{c}" for c in ohe_cats]


lr_data = []

for path, label in LR_MODELS:
    bundle = joblib.load(path)
    feature_names = get_lr_feature_names(bundle)
    new_feats = set(bundle.get("new_features", []))
    coefs = bundle["model"].named_steps["clf"].coef_[0]
    ranked = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
    lr_data.append((label, ranked, new_feats))

    print(f"\n{'=' * 62}")
    print(f"  {label}")
    print(f"{'=' * 62}")
    print(f"  {'Feature':<30} {'Coef':>8}  Direction")
    print(f"  {'-' * 54}")
    for feat, coef in ranked:
        direction = "T wins" if coef > 0 else "CT wins"
        marker = " *" if feat in new_feats else ""
        print(f"  {feat:<30} {coef:>+.4f}  {direction}{marker}")
    if new_feats:
        print("  (* = new in this version)")

#LR diverging bar charts — one file per version
COLOR_T = "#e05c5c"
COLOR_CT = "#4C72B0"
TOP_N = 20

patch_t = mpatches.Patch(color=COLOR_T, label="T wins (coef > 0)")
patch_ct = mpatches.Patch(color=COLOR_CT, label="CT wins (coef < 0)")

LR_OUT_NAMES = [
    r"C:\Users\hasan\Downloads\CS2Predictor\output\feature_importance_lr_v1.png",
    r"C:\Users\hasan\Downloads\CS2Predictor\output\feature_importance_lr_v2.png",
    r"C:\Users\hasan\Downloads\CS2Predictor\output\feature_importance_lr_v3.png",
]

for out_path, (label, ranked, new_feats) in zip(LR_OUT_NAMES, lr_data):
    top = ranked[:TOP_N]
    feats = [f for f, _ in top]
    coefs = [c for _, c in top]
    colors = [COLOR_T if c > 0 else COLOR_CT for c in coefs]

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.suptitle(
        f"Logistic Regression Coefficients {label}\n"
        "Red = T wins direction  |  Blue = CT wins direction",
        fontsize=12,
        fontweight="bold",
    )

    y = np.arange(len(feats))
    bars = ax.barh(y, coefs[::-1], color=colors[::-1], edgecolor="white", height=0.7)

    #Mark new features
    tick_labels = [f + " *" if f in new_feats else f for f in feats]
    ax.set_yticks(y)
    ax.set_yticklabels(tick_labels[::-1], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient value")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    #Value labels
    for bar, val in zip(bars, coefs[::-1]):
        x_pos = bar.get_width() + (0.01 if val >= 0 else -0.01)
        ha = "left" if val >= 0 else "right"
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center",
            ha=ha,
            fontsize=7.5,
        )

    fig.legend(
        handles=[patch_t, patch_ct],
        loc="lower center",
        ncol=2,
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved LR chart -> {out_path}")

#v4-v7: LightGBM feature importances (gain)
LGB_MODELS = [
    (
        r"C:\Users\hasan\Downloads\CS2Predictor\cs2_model_v4.joblib",
        "v4 (LightGBM, 33 features architecture isolation)",
        "#9467bd",
    ),
    (
        r"C:\Users\hasan\Downloads\CS2Predictor\cs2_model_v5.joblib",
        "v5 (LightGBM, 37 features tactical)",
        "#8c564b",
    ),
    (
        r"C:\Users\hasan\Downloads\CS2Predictor\cs2_model_v6.joblib",
        "v6 (LightGBM, 41 features, t50+t75)",
        "#e377c2",
    ),
    (
        r"C:\Users\hasan\Downloads\CS2Predictor\cs2_model_v7.joblib",
        "v7 (LightGBM, 71 features, t50+t75+dynamics)",
        "#ff7f0e",
    ),
]

lgb_data = []

for path, label, color in LGB_MODELS:
    if not os.path.exists(path):
        print(f"\nSkipping {label}: {path} not found.")
        lgb_data.append(None)
        continue

    bundle = joblib.load(path)
    model = bundle["model"]
    all_features = bundle["all_features"]
    display_names = [f if f != "map_encoded" else "map_name" for f in all_features]

    importances = model.feature_importances_
    ranked = sorted(zip(display_names, importances), key=lambda x: x[1], reverse=True)
    lgb_data.append((label, ranked, color))

    print(f"\n{'=' * 62}")
    print(f"  {label}")
    print(f"{'=' * 62}")
    print(f"  {'#':<4} {'Feature':<35} {'Gain':>10}")
    print(f"  {'-' * 52}")
    for rank, (feat, imp) in enumerate(ranked, 1):
        print(f"  {rank:<4} {feat:<35} {imp:>10.0f}")


TOP_N_LGB = 20


def _draw_lgb_panel(ax, label, ranked, color, top_n=TOP_N_LGB):
    """Horizontal bar chart of top-N features by gain (largest at the top)."""
    top = ranked[:top_n]
    feats = [f for f, _ in top]
    gains = [g for _, g in top]
    y = np.arange(len(feats))
    bars = ax.barh(y, gains[::-1], color=color, edgecolor="white", height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(feats[::-1], fontsize=8.5)
    ax.set_xlabel("Gain")
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    if gains:
        max_g = max(gains)
        for bar, val in zip(bars, gains[::-1]):
            ax.text(
                bar.get_width() + max_g * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{int(val)}",
                va="center",
                ha="left",
                fontsize=7,
            )


def _save_pair(pair_data, title, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 9))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    for ax, data in zip(axes, pair_data):
        if data is None:
            ax.set_visible(False)
            continue
        label, ranked, color = data
        _draw_lgb_panel(ax, label, ranked, color)
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved LGB chart -> {out_path}")


_OUT = r"C:\Users\hasan\Downloads\CS2Predictor\output"
_save_pair(
    lgb_data[:2],
    "LightGBM Feature Importance (gain) v4 / v5",
    os.path.join(_OUT, "feature_importance_lgb_v4_v5.png"),
)
_save_pair(
    lgb_data[2:],
    "LightGBM Feature Importance (gain) v6 / v7",
    os.path.join(_OUT, "feature_importance_lgb_v6_v7.png"),
)
