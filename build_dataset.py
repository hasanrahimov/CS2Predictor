#build_dataset.py -- Unified dataset builder for CS2 round predictor.

import polars as pl
from pathlib import Path

ARCHIVE_ROOT = r"C:\Users\hasan\Downloads\archive\csds"
OUTPUT_FILE = r"C:\Users\hasan\Downloads\CS2Predictor\training_data.parquet"
CT_CODE = 3
T_CODE = 2
BOMB_TIMER = 40.0
ROUND_TIME = 115.0
SNAPS = [("t25", 0.25), ("t75", 0.75)]

#v6 feature constants
WEAPON_SNIPERS = {9, 11, 38, 40}
WEAPON_RIFLES = {7, 8, 10, 13, 16, 39, 60}
WEAPON_SMGS = {17, 19, 23, 24, 26, 33, 34}


def compute_streaks(re_df: pl.DataFrame) -> pl.DataFrame:
    rows = re_df.select(["round", "winner_team_code"]).sort("round").to_pandas()
    t_s = ct_s = 0
    t_streaks, ct_streaks = [], []
    for _, row in rows.iterrows():
        t_streaks.append(t_s)
        ct_streaks.append(ct_s)
        if row["winner_team_code"] == CT_CODE:
            t_s += 1
            ct_s = 0
        else:
            ct_s += 1
            t_s = 0
    return pl.DataFrame(
        {
            "round": rows["round"].tolist(),
            "t_loss_streak": [int(x) for x in t_streaks],
            "ct_loss_streak": [int(x) for x in ct_streaks],
        }
    )


def aggregate_snapshot(ps_snap: pl.DataFrame, suffix: str) -> pl.DataFrame:
    cols = set(ps_snap.columns)

    agg_exprs = [
        pl.col("health").sum().alias("total_health"),
        pl.col("armor").sum().alias("total_armor"),
        (pl.col("health") > 0).sum().alias("players_alive"),
        pl.col("has_helmet").cast(pl.Int8).sum().alias("helmets"),
        pl.col("has_defuser").cast(pl.Int8).sum().alias("defusers"),
        pl.col("equipment_value_calc").sum().alias("total_equip_value"),
        pl.col("inv_primary").is_not_null().cast(pl.Int8).sum().alias("primaries"),
        pl.col("inv_flashbang").fill_null(0).sum().alias("flashbangs"),
        pl.col("inv_hegrenade").fill_null(0).sum().alias("hegrenades"),
        pl.col("inv_smokegrenade").fill_null(0).sum().alias("smokes"),
        pl.col("inv_molotov").fill_null(0).sum().alias("molotovs"),
        pl.col("money").mean().alias("avg_money"),
    ]

# v6 weapon class composition
    if "inv_primary" in cols:
        alive = pl.col("health") > 0
        agg_exprs.extend(
            [
                (pl.col("inv_primary").is_in(list(WEAPON_SNIPERS)) & alive)
                .cast(pl.Int8)
                .sum()
                .alias("snipers"),
                (pl.col("inv_primary").is_in(list(WEAPON_RIFLES)) & alive)
                .cast(pl.Int8)
                .sum()
                .alias("rifles"),
                (pl.col("inv_primary").is_in(list(WEAPON_SMGS)) & alive)
                .cast(pl.Int8)
                .sum()
                .alias("smgs"),
            ]
        )

# v6 tactical aggregates
    alive = pl.col("health") > 0
    if "is_in_bomb_zone" in cols:
        agg_exprs.append(
            (pl.col("is_in_bomb_zone").fill_null(False) & alive)
            .cast(pl.Int8)
            .sum()
            .alias("in_bomb_zone")
        )
    if "is_scoped" in cols:
        agg_exprs.append(
            (pl.col("is_scoped").fill_null(False) & alive)
            .cast(pl.Int8)
            .sum()
            .alias("scoped")
        )
    if "place_name" in cols:
        agg_exprs.extend(
            [
                (pl.col("place_name").fill_null("").str.contains("BombsiteA") & alive)
                .cast(pl.Int8)
                .sum()
                .alias("at_a_site"),
                (pl.col("place_name").fill_null("").str.contains("BombsiteB") & alive)
                .cast(pl.Int8)
                .sum()
                .alias("at_b_site"),
            ]
        )
    if "flash_duration" in cols:
        agg_exprs.append(
            ((pl.col("flash_duration").fill_null(0.0) > 0.0) & alive)
            .cast(pl.Int8)
            .sum()
            .alias("flashed")
        )

    agg = ps_snap.group_by(["round", "team_code"]).agg(agg_exprs)
    suf = f"_{suffix}" if suffix else ""
    ct = (
        agg.filter(pl.col("team_code") == CT_CODE)
        .drop("team_code")
        .rename(
            {c: f"ct_{c}{suf}" for c in agg.columns if c not in ["round", "team_code"]}
        )
    )
    t = (
        agg.filter(pl.col("team_code") == T_CODE)
        .drop("team_code")
        .rename(
            {c: f"t_{c}{suf}" for c in agg.columns if c not in ["round", "team_code"]}
        )
    )
    return ct.join(t, on="round", how="inner")


def process_match(folder: Path):
    #Required files
    try:
        ps = pl.read_parquet(folder / "player_status")
        re = pl.read_parquet(folder / "round_end")
        hdr = pl.read_parquet(folder / "header")
        pi = pl.read_parquet(folder / "player_info")
        map_name = hdr["map_name"][0]
        tick_rate = float(hdr["tick_rate"][0] or 64)
    except Exception:
        return None

    #Optional files
    try:
        rs = pl.read_parquet(folder / "round_state")
    except Exception:
        rs = None
    try:
        bs = pl.read_parquet(folder / "bomb_state")
    except Exception:
        bs = None
    try:
        pd_file = pl.read_parquet(folder / "player_death")
    except Exception:
        pd_file = None

    match_id = str(folder)
    rounds = re.select("round").unique().sort("round")["round"].to_list()

    #Labels
    labels = re.select(
        [
            pl.col("round"),
            (pl.col("winner_team_code") == T_CODE).cast(pl.Int8).alias("target"),
        ]
    )

    #team_code per player
    team_map = pi.select(["round", "player_id", "team_code"])
    ps = ps.join(team_map, on=["round", "player_id"], how="left")
    ps = ps.filter(
        pl.col("team_code").is_not_null() & pl.col("team_code").is_in([CT_CODE, T_CODE])
    )

    #round_end_tick and freeze_end_tick
    round_end_ticks = re.select(["round", "tick"]).rename({"tick": "round_end_tick"})
    if rs is not None:
        try:
            freeze_end_ticks = (
                rs.filter(pl.col("event_type") == "round_freeze_end")
                .select(["round", "tick"])
                .rename({"tick": "freeze_end_tick"})
            )
        except Exception:
            freeze_end_ticks = pl.DataFrame(
                {"round": rounds, "freeze_end_tick": [0] * len(rounds)}
            )
    else:
        freeze_end_ticks = pl.DataFrame(
            {"round": rounds, "freeze_end_tick": [0] * len(rounds)}
        )

    timing = round_end_ticks.join(
        freeze_end_ticks, on="round", how="left"
    ).with_columns(pl.col("freeze_end_tick").fill_null(0))

    #Snapshot ticks: t50 (mid), t25, t75
    timing = timing.with_columns(
        ((pl.col("freeze_end_tick") + pl.col("round_end_tick")) / 2.0).alias("snap_t50")
    )
    for suffix, frac in SNAPS:
        timing = timing.with_columns(
            (
                pl.col("freeze_end_tick")
                + frac * (pl.col("round_end_tick") - pl.col("freeze_end_tick"))
            ).alias(f"snap_{suffix}")
        )

    #bomb plant ticks
    if bs is not None:
        try:
            plant_ticks = (
                bs.filter(pl.col("event_type") == "bomb_planted")
                .select(["round", "tick"])
                .rename({"tick": "plant_tick"})
                .unique(subset=["round"])
            )
        except Exception:
            plant_ticks = pl.DataFrame({"round": [], "plant_tick": []})
    else:
        plant_ticks = pl.DataFrame({"round": [], "plant_tick": []})
    timing = timing.join(plant_ticks, on="round", how="left")

    #Filter freezetime ticks
    ps = ps.join(freeze_end_ticks, on="round", how="left")
    ps = ps.filter(pl.col("tick") > pl.col("freeze_end_tick").fill_null(0))

    #t50 snapshot
    ps_t50 = ps.join(timing.select(["round", "snap_t50"]), on="round", how="left")
    ps_t50 = ps_t50.with_columns(
        (pl.col("tick") - pl.col("snap_t50")).abs().alias("_dist")
    )
    ps_t50 = (
        ps_t50.sort(["round", "player_id", "_dist"])
        .group_by(["round", "player_id"])
        .first()
    )

    snapshot = aggregate_snapshot(ps_t50, "")

    kit_t50 = (
        ps_t50.filter(pl.col("team_code") == CT_CODE)
        .group_by("round")
        .agg(pl.col("has_defuser").cast(pl.Int8).max().alias("ct_has_kit"))
    )
    snapshot = snapshot.join(kit_t50, on="round", how="left")

    timing_t50 = timing.with_columns(
        [
            pl.when(
                pl.col("plant_tick").is_not_null()
                & (pl.col("plant_tick") <= pl.col("snap_t50"))
            )
            .then(
                (
                    pl.lit(BOMB_TIMER)
                    - (pl.col("snap_t50") - pl.col("plant_tick")) / tick_rate
                ).clip(lower_bound=0.0, upper_bound=BOMB_TIMER)
            )
            .otherwise(pl.lit(BOMB_TIMER))
            .alias("bomb_countdown"),
            # round_time_remaining
            (
                pl.lit(ROUND_TIME)
                - (pl.col("snap_t50") - pl.col("freeze_end_tick")) / tick_rate
            )
            .clip(lower_bound=0.0, upper_bound=ROUND_TIME)
            .alias("round_time_remaining"),
        ]
    )
    snapshot = snapshot.join(
        timing_t50.select(["round", "bomb_countdown", "round_time_remaining"]),
        on="round",
        how="left",
    )

    #t25 and t75 snapshots
    for suffix, _ in SNAPS:
        snap_col = f"snap_{suffix}"
        ps_s = ps.join(timing.select(["round", snap_col]), on="round", how="left")
        ps_s = ps_s.with_columns(
            (pl.col("tick") - pl.col(snap_col)).abs().alias("_dist")
        )
        ps_s = (
            ps_s.sort(["round", "player_id", "_dist"])
            .group_by(["round", "player_id"])
            .first()
        )

        snap_agg = aggregate_snapshot(ps_s, suffix)

        kit_s = (
            ps_s.filter(pl.col("team_code") == CT_CODE)
            .group_by("round")
            .agg(
                pl.col("has_defuser").cast(pl.Int8).max().alias(f"ct_has_kit_{suffix}")
            )
        )
        snap_agg = snap_agg.join(kit_s, on="round", how="left")

        timing_s = timing.with_columns(
            [
                pl.when(
                    pl.col("plant_tick").is_not_null()
                    & (pl.col("plant_tick") <= pl.col(snap_col))
                )
                .then(
                    (
                        pl.lit(BOMB_TIMER)
                        - (pl.col(snap_col) - pl.col("plant_tick")) / tick_rate
                    ).clip(lower_bound=0.0, upper_bound=BOMB_TIMER)
                )
                .otherwise(pl.lit(BOMB_TIMER))
                .alias(f"bomb_countdown_{suffix}"),
                # clock-based round_time_remaining
                (
                    pl.lit(ROUND_TIME)
                    - (pl.col(snap_col) - pl.col("freeze_end_tick")) / tick_rate
                )
                .clip(lower_bound=0.0, upper_bound=ROUND_TIME)
                .alias(f"round_time_remaining_{suffix}"),
            ]
        )
        snap_agg = snap_agg.join(
            timing_s.select(
                ["round", f"bomb_countdown_{suffix}", f"round_time_remaining_{suffix}"]
            ),
            on="round",
            how="left",
        )
        snap_agg = snap_agg.with_columns(
            [
                pl.col(f"ct_has_kit_{suffix}").fill_null(0).cast(pl.Int8),
                pl.col(f"bomb_countdown_{suffix}").fill_null(BOMB_TIMER),
                pl.col(f"round_time_remaining_{suffix}").fill_null(0.0),
            ]
        )
        snapshot = snapshot.join(snap_agg, on="round", how="left")

    #Context features (v3)
    try:
        streaks = compute_streaks(re)
    except Exception:
        streaks = pl.DataFrame(
            {
                "round": rounds,
                "t_loss_streak": [0] * len(rounds),
                "ct_loss_streak": [0] * len(rounds),
            }
        )

    if rs is not None:
        try:
            scores = (
                rs.filter(pl.col("event_type") == "round_prestart")
                .select(["round", "t_score", "ct_score"])
                .unique(subset=["round"])
            )
        except Exception:
            scores = pl.DataFrame(
                {
                    "round": rounds,
                    "t_score": [0] * len(rounds),
                    "ct_score": [0] * len(rounds),
                }
            )
    else:
        scores = pl.DataFrame(
            {
                "round": rounds,
                "t_score": [0] * len(rounds),
                "ct_score": [0] * len(rounds),
            }
        )

    rank_diff = 0.0
    try:
        t_r = float(hdr["t_starters_avg_rank"][0] or 0)
        ct_r = float(hdr["ct_starters_avg_rank"][0] or 0)
        if t_r > 0 and ct_r > 0:
            rank_diff = t_r - ct_r
    except Exception:
        pass

    t_dead_ranks, ct_dead_ranks = {}, {}
    if pd_file is not None:
        try:
            if "rank" in pi.columns and pi.filter(pl.col("rank") > 0).height > 0:
                rank_map = (
                    pi.filter(pl.col("rank") > 0)
                    .select(["round", "player_id", "rank", "team_code"])
                    .rename({"rank": "player_rank"})
                )
                deaths = (
                    pd_file.select(["round", "player_id"])
                    .join(rank_map, on=["round", "player_id"], how="left")
                    .filter(pl.col("player_rank").is_not_null())
                )
                for team_code, col_name in [
                    (T_CODE, "t_dead_rank_avg"),
                    (CT_CODE, "ct_dead_rank_avg"),
                ]:
                    agg_r = (
                        deaths.filter(pl.col("team_code") == team_code)
                        .group_by("round")
                        .agg(pl.col("player_rank").mean().alias(col_name))
                    )
                    for r, v in zip(
                        agg_r["round"].to_list(), agg_r[col_name].to_list()
                    ):
                        if team_code == T_CODE:
                            t_dead_ranks[r] = v
                        else:
                            ct_dead_ranks[r] = v
        except Exception:
            pass

    context = pl.DataFrame({"round": rounds})
    context = context.join(streaks, on="round", how="left")
    context = context.join(scores, on="round", how="left")
    context = context.with_columns(
        [
            pl.col("t_loss_streak").fill_null(0),
            pl.col("ct_loss_streak").fill_null(0),
            pl.col("t_score").fill_null(0),
            pl.col("ct_score").fill_null(0),
            pl.lit(rank_diff).alias("rank_diff"),
            pl.col("round")
            .map_elements(lambda r: t_dead_ranks.get(r, 0.0), return_dtype=pl.Float64)
            .alias("t_dead_rank_avg"),
            pl.col("round")
            .map_elements(lambda r: ct_dead_ranks.get(r, 0.0), return_dtype=pl.Float64)
            .alias("ct_dead_rank_avg"),
        ]
    )

    #Assembly
    snapshot = snapshot.join(labels, on="round", how="inner")
    snapshot = snapshot.join(context, on="round", how="left")
    snapshot = snapshot.with_columns(
        [
            pl.lit(match_id).alias("match_id"),
            pl.lit(map_name).alias("map_name"),
            pl.col("round").alias("round_num"),
            pl.col("ct_has_kit").fill_null(0).cast(pl.Int8),
            pl.col("bomb_countdown").fill_null(BOMB_TIMER),
            pl.col("round_time_remaining").fill_null(0.0),
        ]
    )
    return snapshot


#Main
folders = [p.parent for p in Path(ARCHIVE_ROOT).rglob("player_status") if p.is_file()]
print(f"Found {len(folders)} match folders")

all_snapshots = []
failed = 0
for i, folder in enumerate(folders):
    result = process_match(folder)
    if result is not None:
        all_snapshots.append(result)
    else:
        failed += 1
    if i % 50 == 0:
        print(
            f"  {i}/{len(folders)} processed -- {len(all_snapshots)} ok, {failed} failed"
        )

print(f"\nCombining {len(all_snapshots)} matches...")
final = pl.concat(all_snapshots, how="diagonal")
final.write_parquet(OUTPUT_FILE)
print(f"Saved: {OUTPUT_FILE}")
print(f"Shape: {final.shape}")
print(f"Target distribution:\n{final['target'].value_counts()}")
