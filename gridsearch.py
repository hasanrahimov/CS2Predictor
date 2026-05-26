#Hyperparameter search + calibration diagnostic for all model versions
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_validate, cross_val_predict
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, brier_score_loss

#Calculate maximum calibration deviation from the perfect calibration line.
def max_calib_dev(y_true, proba, n_bins=10):
    frac, mean = calibration_curve(y_true, proba, n_bins=n_bins, strategy="uniform")
    return float(np.max(np.abs(frac - mean)))

#Grid search for num_leaves in LGBMClassifier with GroupKFold CV
def sweep_lgb_leaves(X, y, groups, label, leaves, subsample=0.8, colsample=0.8):
    print()
    print("=" * 80)
    print(f"{label} num_leaves sweep")
    print("=" * 80)
    print(
        f"{'num_leaves':>12}  {'Accuracy':>10}  {'LogLoss':>10}  {'Brier':>10}  {'MaxCalibDev':>12}"
    )
    print("-" * 60)
    out = []
    for nl in leaves:
        clf = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=nl,
            min_child_samples=20,
            subsample=subsample,
            colsample_bytree=colsample,
            random_state=42,
            verbose=-1,
        )
        proba = cross_val_predict(
            clf, X, y, cv=GroupKFold(n_splits=5), groups=groups, method="predict_proba"
        )[:, 1]
        acc = float(((proba >= 0.5).astype(int) == y).mean())
        ll = log_loss(y, proba)
        bs = brier_score_loss(y, proba)
        mcd = max_calib_dev(y, proba)
        print(f"{nl:>12}  {acc:>10.4f}  {ll:>10.4f}  {bs:>10.4f}  {mcd:>12.4f}")
        out.append(
            {"nl": nl, "acc": acc, "log_loss": ll, "brier": bs, "max_calib_dev": mcd}
        )
    return out

#Select the best model
def best_with_calib(results, calib_tolerance=1.3):
    min_mcd = min(r["max_calib_dev"] for r in results)
    threshold = min_mcd * calib_tolerance
    candidates = [r for r in results if r["max_calib_dev"] <= threshold]
    if not candidates:  # safety: fall back to full set
        candidates = results
    best = max(candidates, key=lambda x: x["acc"])
    return best, min_mcd, threshold


DATA_FILE = r"C:\Users\hasan\Downloads\CS2Predictor\training_data.parquet"
OUTPUT_DIR = r"C:\Users\hasan\Downloads\CS2Predictor\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUMERIC_BASE = [
    "ct_total_health",
    "ct_total_armor",
    "ct_players_alive",
    "ct_helmets",
    "ct_defusers",
    "ct_total_equip_value",
    "ct_primaries",
    "ct_flashbangs",
    "ct_hegrenades",
    "ct_smokes",
    "ct_molotovs",
    "ct_avg_money",
    "t_total_health",
    "t_total_armor",
    "t_players_alive",
    "t_helmets",
    "t_total_equip_value",
    "t_primaries",
    "t_flashbangs",
    "t_hegrenades",
    "t_smokes",
    "t_molotovs",
    "t_avg_money",
    "bomb_planted_now",
]
NUMERIC_CTX = [
    "round_num",
    "t_score",
    "ct_score",
    "t_loss_streak",
    "ct_loss_streak",
    "rank_diff",
    "t_dead_rank_avg",
    "ct_dead_rank_avg",
]
NUMERIC_V5_TACTICAL = [
    "ct_has_kit",
    "bomb_countdown",
    "round_time_remaining",
    "is_pistol_round",
]
MAP_FEATURE = "map_name"
EXCLUDE_MAPS = ["cs_office"]
C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
NUM_LEAVES = [15, 31, 63, 127, 255, 511]

df = pl.read_parquet(DATA_FILE).filter(~pl.col(MAP_FEATURE).is_in(EXCLUDE_MAPS))
df_pd = df.to_pandas()

df_pd["bomb_planted_now"] = (df_pd["bomb_countdown"] < 40.0).astype(int)
df_pd["is_pistol_round"] = (
    ((df_pd["t_score"] + df_pd["ct_score"]) % 12 == 0)
    & ((df_pd["t_score"] + df_pd["ct_score"]) < 24)
).astype(int)

NUMERIC_V1_ALL = [f for f in NUMERIC_BASE if f in df_pd.columns]
NUMERIC_V3_ALL = [f for f in NUMERIC_BASE + NUMERIC_CTX if f in df_pd.columns]
NUMERIC_V5_ALL = [
    f for f in NUMERIC_BASE + NUMERIC_CTX + NUMERIC_V5_TACTICAL if f in df_pd.columns
]

le = LabelEncoder()
df_pd["map_encoded"] = le.fit_transform(df_pd[MAP_FEATURE])

y = df_pd["target"].values
groups = df_pd["match_id"].values
cv = GroupKFold(n_splits=5)

X_v1 = df_pd[NUMERIC_V1_ALL].values
X_v2 = df_pd[NUMERIC_V1_ALL + [MAP_FEATURE]]
X_v3 = df_pd[NUMERIC_V3_ALL + [MAP_FEATURE]]
X_v3_enc = df_pd[NUMERIC_V3_ALL + ["map_encoded"]].values  # v4: LGB on v3 features
X_v5 = df_pd[NUMERIC_V5_ALL + ["map_encoded"]].values  # v5: LGB + tactical features

#GridSearch C for v1/v2/v3
results = {"v1": [], "v2": [], "v3": []}
SCORING = ["accuracy", "neg_brier_score"]

print("GridSearch C (v1/v2/v3):")
print(f"{'C':<10} {'v1 Acc':>8} {'v2 Acc':>8} {'v3 Acc':>8}")
print("-" * 40)

for C in C_VALUES:
    pipe_v1 = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=1000, random_state=42)),
        ]
    )
    s1 = cross_validate(pipe_v1, X_v1, y, cv=cv, groups=groups, scoring=SCORING)
    results["v1"].append(
        {
            "C": C,
            "acc": s1["test_accuracy"].mean(),
            "brier": -s1["test_neg_brier_score"].mean(),
        }
    )

    pipe_v2 = Pipeline(
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
            ("clf", LogisticRegression(C=C, max_iter=1000, random_state=42)),
        ]
    )
    s2 = cross_validate(pipe_v2, X_v2, y, cv=cv, groups=groups, scoring=SCORING)
    results["v2"].append(
        {
            "C": C,
            "acc": s2["test_accuracy"].mean(),
            "brier": -s2["test_neg_brier_score"].mean(),
        }
    )

    pipe_v3 = Pipeline(
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
            ("clf", LogisticRegression(C=C, max_iter=1000, random_state=42)),
        ]
    )
    s3 = cross_validate(pipe_v3, X_v3, y, cv=cv, groups=groups, scoring=SCORING)
    results["v3"].append(
        {
            "C": C,
            "acc": s3["test_accuracy"].mean(),
            "brier": -s3["test_neg_brier_score"].mean(),
        }
    )

    print(
        f"{C:<10g} {results['v1'][-1]['acc']:>8.4f} {results['v2'][-1]['acc']:>8.4f} {results['v3'][-1]['acc']:>8.4f}"
    )

for k in ["v1", "v2", "v3"]:
    best = max(results[k], key=lambda x: x["acc"])
    print(f"Best C for {k}: {best['C']}  (acc={best['acc']:.4f})")
best_v3_C = max(results["v3"], key=lambda x: x["acc"])["C"]

#GridSearch num_leaves for v4
results_lgb = {
    "v4": sweep_lgb_leaves(
        X_v3_enc,
        y,
        groups,
        "V4 (LGBM, v3 features, 33 feat architecture isolation)",
        NUM_LEAVES,
        subsample=0.8,
        colsample=0.8,
    )
}
best_v4, min_mcd_v4, thr_v4 = best_with_calib(results_lgb["v4"])
print(
    f"Best num_leaves for v4: {best_v4['nl']}  "
    f"(acc={best_v4['acc']:.4f}, MaxCalibDev={best_v4['max_calib_dev']:.4f})  "
    f"[calib threshold ≤{thr_v4:.4f} = 1.3 × min {min_mcd_v4:.4f}]"
)

#GridSearch num_leaves for v5
results_lgb["v5"] = sweep_lgb_leaves(
    X_v5,
    y,
    groups,
    "V5 (LGBM, single snapshot, 37 feat)",
    NUM_LEAVES,
    subsample=0.8,
    colsample=0.8,
)
best_v5, min_mcd_v5, thr_v5 = best_with_calib(results_lgb["v5"])
print(
    f"Best num_leaves for v5: {best_v5['nl']}  "
    f"(acc={best_v5['acc']:.4f}, MaxCalibDev={best_v5['max_calib_dev']:.4f})  "
    f"[calib threshold ≤{thr_v5:.4f} = 1.3 × min {min_mcd_v5:.4f}]"
)

#GridSearch num_leaves for v6
results_lgb["v6"] = []

PER_SNAP_V6 = [
    "ct_total_health",
    "ct_total_armor",
    "ct_players_alive",
    "ct_helmets",
    "ct_defusers",
    "ct_total_equip_value",
    "ct_primaries",
    "ct_flashbangs",
    "ct_hegrenades",
    "ct_smokes",
    "ct_molotovs",
    "ct_avg_money",
    "t_total_health",
    "t_total_armor",
    "t_players_alive",
    "t_helmets",
    "t_total_equip_value",
    "t_primaries",
    "t_flashbangs",
    "t_hegrenades",
    "t_smokes",
    "t_molotovs",
    "t_avg_money",
    "ct_has_kit",
    "bomb_countdown",
    "round_time_remaining",
    "bomb_planted_now",
]
ROUND_CTX_V6 = [
    "round_num",
    "t_score",
    "ct_score",
    "t_loss_streak",
    "ct_loss_streak",
    "rank_diff",
    "t_dead_rank_avg",
    "ct_dead_rank_avg",
    "is_pistol_round",
]
DELTA_V6 = ["health_share", "alive_diff", "equip_share", "money_diff"]


def _add_delta_v6(df):
    eps = 1e-6
    df = df.copy()
    df["health_share"] = df["ct_total_health"] / (
        df["ct_total_health"] + df["t_total_health"] + eps
    )
    df["alive_diff"] = df["ct_players_alive"] - df["t_players_alive"]
    df["equip_share"] = df["ct_total_equip_value"] / (
        df["ct_total_equip_value"] + df["t_total_equip_value"] + eps
    )
    df["money_diff"] = df["ct_avg_money"] - df["t_avg_money"]
    return df


def _melt_v6(df_wide):
    meta_cols = ["match_id", "round", MAP_FEATURE, "target"] + ROUND_CTX_V6

    def _snap(df, suffix, label):
        meta = (
            df[[c for c in meta_cols if c in df.columns]].copy().reset_index(drop=True)
        )
        if suffix == "":
            snap = (
                df[[c for c in PER_SNAP_V6 if c in df.columns]]
                .copy()
                .reset_index(drop=True)
            )
        else:
            rename = {f"{c}_{suffix}": c for c in PER_SNAP_V6[:-1]}
            rename[f"bomb_planted_now_{suffix}"] = "bomb_planted_now"
            cols = [f"{c}_{suffix}" for c in PER_SNAP_V6[:-1]] + [
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

    return _add_delta_v6(
        pd.concat(
            [_snap(df_wide, "", "t50"), _snap(df_wide, "t75", "t75")], ignore_index=True
        )
    )


if os.path.exists(DATA_FILE):
    _raw = (
        pl.read_parquet(DATA_FILE)
        .filter(~pl.col(MAP_FEATURE).is_in(EXCLUDE_MAPS))
        .to_pandas()
    )
    _raw["bomb_planted_now"] = (_raw["bomb_countdown"] < 40.0).astype(int)
    _raw["bomb_planted_now_t75"] = (_raw["bomb_countdown_t75"] < 40.0).astype(int)
    _raw["is_pistol_round"] = (
        ((_raw["t_score"] + _raw["ct_score"]) % 12 == 0)
        & ((_raw["t_score"] + _raw["ct_score"]) < 24)
    ).astype(int)
    melted_v6 = _melt_v6(_raw)
    NUMERIC_V6_ALL = [
        f for f in PER_SNAP_V6 + ROUND_CTX_V6 + DELTA_V6 if f in melted_v6.columns
    ]
    le_v6 = LabelEncoder()
    melted_v6["map_encoded"] = le_v6.fit_transform(melted_v6[MAP_FEATURE])
    X_v6 = melted_v6[NUMERIC_V6_ALL + ["map_encoded"]].values
    y_v6 = melted_v6["target"].values
    groups_v6 = melted_v6["match_id"].values
    results_lgb["v6"] = sweep_lgb_leaves(
        X_v6,
        y_v6,
        groups_v6,
        "V6 (LGBM, t50+t75 melted, delta features, 41 feat)",
        NUM_LEAVES,
        subsample=1.0,
        colsample=1.0,
    )
    best_v6, min_mcd_v6, thr_v6 = best_with_calib(results_lgb["v6"])
    print(
        f"Best num_leaves for v6: {best_v6['nl']}  "
        f"(acc={best_v6['acc']:.4f}, MaxCalibDev={best_v6['max_calib_dev']:.4f})  "
        f"[calib threshold ≤{thr_v6:.4f} = 1.3 × min {min_mcd_v6:.4f}]"
    )
else:
    print(f"\nSkipping v6 grid search: {DATA_FILE} not found.")

#GridSearch num_leaves for v7
results_lgb["v7"] = []
if os.path.exists(DATA_FILE):
    from features import (
        V7_ALL_NUMERIC,
        melt_v7 as _melt_v7_shared,
        derive_runtime_features_v7,
        filter_present,
    )

    _raw7 = (
        pl.read_parquet(DATA_FILE)
        .filter(~pl.col(MAP_FEATURE).is_in(EXCLUDE_MAPS))
        .to_pandas()
    )
    _raw7 = derive_runtime_features_v7(_raw7)
    melted_v7 = _melt_v7_shared(_raw7)
    NUMERIC_V7_ALL = filter_present(V7_ALL_NUMERIC, melted_v7)
    le_v7 = LabelEncoder()
    melted_v7["map_encoded"] = le_v7.fit_transform(melted_v7[MAP_FEATURE])
    X_v7 = melted_v7[NUMERIC_V7_ALL + ["map_encoded"]].values
    y_v7 = melted_v7["target"].values
    groups_v7 = melted_v7["match_id"].values
    results_lgb["v7"] = sweep_lgb_leaves(
        X_v7,
        y_v7,
        groups_v7,
        "V7 (LGBM, t50+t75 + dynamics/weapons/econ/tactical, 71 feat)",
        NUM_LEAVES,
        subsample=1.0,
        colsample=0.8,
    )
    best_v7, min_mcd_v7, thr_v7 = best_with_calib(results_lgb["v7"])
    print(
        f"Best num_leaves for v7: {best_v7['nl']}  "
        f"(acc={best_v7['acc']:.4f}, MaxCalibDev={best_v7['max_calib_dev']:.4f})  "
        f"[calib threshold ≤{thr_v7:.4f} = 1.3 × min {min_mcd_v7:.4f}]"
    )
else:
    print(f"\nSkipping v7 grid search: {DATA_FILE} not found.")

#Plot
cs = [r["C"] for r in results["v1"]]
acc_v1 = [r["acc"] for r in results["v1"]]
acc_v2 = [r["acc"] for r in results["v2"]]
acc_v3 = [r["acc"] for r in results["v3"]]
brier_v1 = [r["brier"] for r in results["v1"]]
brier_v2 = [r["brier"] for r in results["v2"]]
brier_v3 = [r["brier"] for r in results["v3"]]

nls = [r["nl"] for r in results_lgb["v5"]]
acc_v5 = [r["acc"] for r in results_lgb["v5"]]
mcd_v5 = [r["max_calib_dev"] for r in results_lgb["v5"]]

lr_styles = [
    ("v1 (bez kartes)", acc_v1, brier_v1, "#4C72B0", "o", "-"),
    ("v2 (ar karti)", acc_v2, brier_v2, "#2ca02c", "s", "--"),
    ("v3 (33 pazimes)", acc_v3, brier_v3, "#d62728", "^", "-"),
]

#LR GridSearch (v1/v2/v3)
fig_lr, axes_lr = plt.subplots(1, 2, figsize=(11, 4.5))
fig_lr.suptitle(
    "GridSearch: regulārizācijas parametrs C (v1–v3)", fontsize=13, fontweight="bold"
)

for ax, metric_idx, ylabel, title in [
    (axes_lr[0], 1, "Precizitate (Accuracy)", "Precizitate vs C"),
    (axes_lr[1], 2, "Brier Score (mazaks=labaks)", "Brier Score vs C"),
]:
    for label, acc_data, brier_data, color, marker, ls in lr_styles:
        data = acc_data if metric_idx == 1 else brier_data
        ax.semilogx(
            cs, data, marker + ls, color=color, linewidth=2, markersize=6, label=label
        )
    ax.set_xlabel("C (logaritmiska skala)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(best_v3_C, color="#d62728", linestyle=":", alpha=0.6, linewidth=1.2)

fig_lr.tight_layout()
out_lr = os.path.join(OUTPUT_DIR, "gridsearch_lr.png")
fig_lr.savefig(out_lr, dpi=150, bbox_inches="tight")
plt.close(fig_lr)
print(f"\nSaved: {out_lr}")


#LGB GridSearch plots v4-v7)
def plot_lgb_panel(ax, nls, acc, mcd, best_nl, color, title):
    """Accuracy (left y) + MaxCalibDev (right y) vs num_leaves."""
    axb = ax.twinx()
    ax.plot(nls, acc, "D-", color=color, linewidth=2, markersize=7, label="Precizitate")
    axb.plot(
        nls,
        mcd,
        "s--",
        color=color,
        linewidth=1.5,
        markersize=5,
        alpha=0.7,
        label="MaxCalibDev",
    )
    ax.axvline(best_nl, color=color, linestyle=":", alpha=0.6, linewidth=1.2)
    ax.set_xlabel("num_leaves")
    ax.set_ylabel("Precizitate")
    axb.set_ylabel("MaxCalibDev (mazāks = labāks)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", colors=color)
    axb.tick_params(axis="y", colors=color)
    lines = ax.get_legend_handles_labels()[0] + axb.get_legend_handles_labels()[0]
    lbls = ax.get_legend_handles_labels()[1] + axb.get_legend_handles_labels()[1]
    ax.legend(lines, lbls, fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines[["top"]].set_visible(False)


def save_lgb_fig(key, nls, acc, mcd, best, color, title, filename):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    plot_lgb_panel(ax, nls, acc, mcd, best["nl"], color, title)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


nls_v4 = [r["nl"] for r in results_lgb["v4"]]
acc_v4 = [r["acc"] for r in results_lgb["v4"]]
mcd_v4 = [r["max_calib_dev"] for r in results_lgb["v4"]]
save_lgb_fig(
    "v4",
    nls_v4,
    acc_v4,
    mcd_v4,
    best_v4,
    "#9467bd",
    "v4 LightGBM (v3 pazīmes, 33 pazīmes arhitektūras izolācija)\n"
    f"optimāls={best_v4['nl']}, acc={best_v4['acc']:.4f}, MCD={best_v4['max_calib_dev']:.4f}",
    "gridsearch_lgb_v4.png",
)

save_lgb_fig(
    "v5",
    nls,
    acc_v5,
    mcd_v5,
    best_v5,
    "#8c564b",
    "v5 LightGBM (viens momentuzņēmums, 37 pazīmes)\n"
    f"optimāls={best_v5['nl']}, acc={best_v5['acc']:.4f}, MCD={best_v5['max_calib_dev']:.4f}",
    "gridsearch_lgb_v5.png",
)

if results_lgb.get("v6"):
    nls_v6 = [r["nl"] for r in results_lgb["v6"]]
    acc_v6 = [r["acc"] for r in results_lgb["v6"]]
    mcd_v6 = [r["max_calib_dev"] for r in results_lgb["v6"]]
    save_lgb_fig(
        "v6",
        nls_v6,
        acc_v6,
        mcd_v6,
        best_v6,
        "#e377c2",
        "v6 LightGBM (t50+t75+\u0394, 41 pazīme)\n"
        f"optimāls={best_v6['nl']}, acc={best_v6['acc']:.4f}, MCD={best_v6['max_calib_dev']:.4f}",
        "gridsearch_lgb_v6.png",
    )

if results_lgb.get("v7"):
    nls_v7 = [r["nl"] for r in results_lgb["v7"]]
    acc_v7 = [r["acc"] for r in results_lgb["v7"]]
    mcd_v7 = [r["max_calib_dev"] for r in results_lgb["v7"]]
    save_lgb_fig(
        "v7",
        nls_v7,
        acc_v7,
        mcd_v7,
        best_v7,
        "#ff7f0e",
        "v7 LightGBM (+dinamika/ieroči/ekon./taktika, 71 pazīme)\n"
        f"optimāls={best_v7['nl']}, acc={best_v7['acc']:.4f}, MCD={best_v7['max_calib_dev']:.4f}",
        "gridsearch_lgb_v7.png",
    )
