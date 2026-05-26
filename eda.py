#Exploratory Data Analysis for CS2 round predictor dataset.


import os
import polars as pl
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_FILE = r"C:\Users\hasan\Downloads\CS2Predictor\training_data.parquet"
OUTPUT_DIR = r"C:\Users\hasan\Downloads\CS2Predictor\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUMERIC_FEATURES = [
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
]

df = pl.read_parquet(DATA_FILE)
df_pd = df.to_pandas()

#Console summary
print("=" * 55)
print("DATASET OVERVIEW")
print("=" * 55)
print(f"Shape:           {df.shape[0]} rounds × {df.shape[1]} columns")
print(f"Matches:         {df['match_id'].n_unique()}")
print(f"Missing values:  {df_pd[NUMERIC_FEATURES].isnull().sum().sum()}")

ct_wins = int((df["target"] == 0).sum())
t_wins = int((df["target"] == 1).sum())
total = len(df)
print("\nTarget distribution:")
print(f"  CT wins: {ct_wins:>6}  ({ct_wins/total*100:.1f}%)")
print(f"  T  wins: {t_wins:>6}  ({t_wins/total*100:.1f}%)")

if "map_name" in df.columns:
    print("\nMap distribution:")
    for row in (
        df.group_by("map_name")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .iter_rows()
    ):
        print(f"  {row[0]:<25} {row[1]:>5} rounds")
print("=" * 55)

#Plot 1: Target distribution
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
fig.suptitle("Round outcome distribution", fontsize=13, fontweight="bold")

labels = ["CT wins", "T wins"]
counts = [ct_wins, t_wins]
colors = ["#4C72B0", "#DD8452"]

axes[0].bar(labels, counts, color=colors, edgecolor="white", linewidth=0.8)
for i, v in enumerate(counts):
    axes[0].text(i, v + 60, f"{v}", ha="center", fontsize=10)
axes[0].set_ylabel("Number of rounds")
axes[0].set_ylim(0, max(counts) * 1.12)
axes[0].spines[["top", "right"]].set_visible(False)

axes[1].pie(
    counts,
    labels=labels,
    colors=colors,
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 1.2},
)

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "eda_01_target_dist.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

#Plot 2: Win rate per map
if "map_name" in df.columns:
    map_stats = (
        df.group_by("map_name")
        .agg(
            [
                (pl.col("target") == 0).sum().alias("ct_wins"),
                (pl.col("target") == 1).sum().alias("t_wins"),
                pl.len().alias("total"),
            ]
        )
        .with_columns(
            [
                (pl.col("ct_wins") / pl.col("total") * 100).alias("ct_pct"),
                (pl.col("t_wins") / pl.col("total") * 100).alias("t_pct"),
            ]
        )
        .sort("map_name")
    )
    maps = map_stats["map_name"].to_list()
    ct_pcts = map_stats["ct_pct"].to_list()
    t_pcts = map_stats["t_pct"].to_list()
    totals = map_stats["total"].to_list()

    x = np.arange(len(maps))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(
        x - w / 2, ct_pcts, w, label="CT win %", color="#4C72B0", edgecolor="white"
    )
    b2 = ax.bar(
        x + w / 2, t_pcts, w, label="T win %", color="#DD8452", edgecolor="white"
    )
    ax.axhline(50, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    for bar in list(b1) + list(b2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{bar.get_height():.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("de_", "") for m in maps], fontsize=10)
    ax.set_ylabel("Win rate (%)")
    ax.set_ylim(40, 62)
    ax.set_title("CT vs T win rate by map", fontsize=13, fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    ax2 = ax.twinx()
    ax2.set_ylabel("Rounds in dataset", fontsize=9, color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")
    ax2.set_ylim(0, max(totals) * 4)
    for i, n in enumerate(totals):
        ax2.text(i, 0, f"n={n}", ha="center", va="bottom", fontsize=7, color="grey")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "eda_02_map_winrates.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

#Plot 3: Correlation heatmap
corr = df_pd[NUMERIC_FEATURES].corr()
fig, ax = plt.subplots(figsize=(13, 11))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
ax.set_xticks(range(len(NUMERIC_FEATURES)))
ax.set_yticks(range(len(NUMERIC_FEATURES)))
short = [
    f.replace("ct_total_", "ct_")
    .replace("t_total_", "t_")
    .replace("_equip_value", "_equip")
    .replace("_players_alive", "_alive")
    for f in NUMERIC_FEATURES
]
ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7.5)
ax.set_yticklabels(short, fontsize=7.5)
ax.set_title("Feature correlation matrix (Pearson)", fontsize=13, fontweight="bold")
for i in range(len(NUMERIC_FEATURES)):
    for j in range(len(NUMERIC_FEATURES)):
        val = corr.values[i, j]
        if abs(val) > 0.4:
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color="white" if abs(val) > 0.7 else "black",
            )
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "eda_03_correlations.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

#Plot 4: Top-5 feature distributions by outcome
TOP5 = [
    "t_total_health",
    "ct_total_health",
    "t_total_equip_value",
    "ct_total_equip_value",
    "t_players_alive",
]

fig, axes = plt.subplots(1, 5, figsize=(16, 4))
fig.suptitle(
    "Key feature distributions: CT wins vs T wins", fontsize=12, fontweight="bold"
)

ct_df = df_pd[df_pd["target"] == 0]
t_df = df_pd[df_pd["target"] == 1]

for ax, feat in zip(axes, TOP5):
    bins = 20
    ax.hist(
        ct_df[feat],
        bins=bins,
        alpha=0.6,
        color="#4C72B0",
        label="CT wins",
        density=True,
    )
    ax.hist(
        t_df[feat], bins=bins, alpha=0.6, color="#DD8452", label="T wins", density=True
    )
    ax.set_title(
        feat.replace("_total_", "\n").replace("_equip_value", "\nequip_val"), fontsize=8
    )
    ax.set_xlabel("Value", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("Density")
axes[0].legend(fontsize=7)
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "eda_04_feature_by_outcome.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

print("\nEDA complete. All plots saved to output/")
