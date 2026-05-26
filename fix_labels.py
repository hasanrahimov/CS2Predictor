#fix for inverted team-code bug in training_data.parquet.
import polars as pl

DATA_FILE = r"C:\Users\hasan\Downloads\CS2Predictor\training_data.parquet"

SWAP_PAIRS = [
    ("ct_total_health", "t_total_health"),
    ("ct_total_armor", "t_total_armor"),
    ("ct_players_alive", "t_players_alive"),
    ("ct_helmets", "t_helmets"),
    ("ct_defusers", "t_defusers"),
    ("ct_total_equip_value", "t_total_equip_value"),
    ("ct_primaries", "t_primaries"),
    ("ct_flashbangs", "t_flashbangs"),
    ("ct_hegrenades", "t_hegrenades"),
    ("ct_smokes", "t_smokes"),
    ("ct_molotovs", "t_molotovs"),
    ("ct_avg_money", "t_avg_money"),
    ("ct_loss_streak", "t_loss_streak"),
    ("ct_dead_rank_avg", "t_dead_rank_avg"),
]


def apply_fix(path: str):
    df = pl.read_parquet(path)
    print(f"\n=== {path} ===")
    print(f"Shape before: {df.shape}")
    for r in df["target"].value_counts().sort("target").iter_rows():
        print(f"  target={r[0]}: {r[1]}")

    df = df.with_columns((1 - pl.col("target")).cast(pl.Int8).alias("target"))

    phase_a = {}
    for ct_col, t_col in SWAP_PAIRS:
        if ct_col in df.columns and t_col in df.columns:
            phase_a[ct_col] = "__tmp__" + t_col
            phase_a[t_col] = "__tmp__" + ct_col
        else:
            missing = [c for c in [ct_col, t_col] if c not in df.columns]
            if missing:
                print(f"  Warning: columns not found, skipping swap: {missing}")

    df = df.rename(phase_a)
    df = df.rename({c: c[7:] for c in df.columns if c.startswith("__tmp__")})

    for r in df["target"].value_counts().sort("target").iter_rows():
        print(f"  target={r[0]}: {r[1]}")

    df.write_parquet(path)
    print(f"Saved: {path}")


apply_fix(DATA_FILE)

df = pl.read_parquet(DATA_FILE)
for row in df["target"].value_counts().sort("target").iter_rows():
    print(f"  target={row[0]} -> {row[1]} ({row[1]/len(df)*100:.1f}%)")
