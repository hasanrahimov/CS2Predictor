"""
Definitions for feature lists across all model versions.

All train_model_v*.py, model_comparison.py, roc_auc.py, calibration.py
import their feature definitions from here.
"""

#Per-team snapshot state without bomb (23 features)
NUMERIC_BASE_NOBOMB = [
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

#v1 base: per-team state + bomb_planted_now (24 features)
#bomb_planted_now is derived at runtime from bomb_countdown
NUMERIC_BASE = NUMERIC_BASE_NOBOMB + ["bomb_planted_now"]

#v3 additions: round context (8 features)
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

#v5 additions: tactical snapshot state (4 features)
#bomb_planted_now already in NUMERIC_BASE, so not repeated here.
NUMERIC_V5_TACTICAL = [
    "ct_has_kit",
    "bomb_countdown",
    "round_time_remaining",
    "is_pistol_round",
]

#v6 additions: relative team advantage delta features (4 features)
DELTA_FEATURES = [
    "health_share",
    "alive_diff",
    "equip_share",
    "money_diff",
]

#v6-specific snapshot structure (melt-aware)
#v6 trains on t50 + t75 snapshots and have suffix variants
#bomb_planted_now must stay last so the melt helper can rename the suffixed variant independently.
V6_PER_SNAP_TACTICAL = [
    "ct_has_kit",
    "bomb_countdown",
    "round_time_remaining",
]
V6_PER_SNAP = NUMERIC_BASE_NOBOMB + V6_PER_SNAP_TACTICAL + ["bomb_planted_now"]

#Round level context isn't snapshot-dependent
V6_ROUND_CTX = NUMERIC_CTX + ["is_pistol_round"]

V6_ALL_NUMERIC = V6_PER_SNAP + V6_ROUND_CTX + DELTA_FEATURES

#v7 + 4 feature groups(dynamics, weapons, tactical, econ) + melt-aware structure
# CS2 weapon-id
WEAPON_SNIPERS = {9, 11, 38, 40}  # AWP, G3SG1, SCAR-20, SSG08
WEAPON_RIFLES = {7, 8, 10, 13, 16, 39, 60}  # AK, AUG, FAMAS, Galil, M4A4, SG553, M4A1-S
WEAPON_SMGS = {17, 19, 23, 24, 26, 33, 34}  # MAC10, P90, MP5-SD, UMP45, Bizon, MP7, MP9

#Dynamics. Computed at snapshot-melt time as snap_value - prev_snap_value
V7_DYNAMICS = [
    "ct_health_change",
    "t_health_change",
    "ct_alive_change",
    "t_alive_change",
    "ct_equip_change",
    "t_equip_change",
    "ct_armor_change",
    "t_armor_change",
]

#Weapon class per-snapshot, melted
V7_WEAPONS = [
    "ct_snipers",
    "t_snipers",
    "ct_rifles",
    "t_rifles",
    "ct_smgs",
    "t_smgs",
]

#Tactical aggregates from player_status per-snapshot and melted.
V7_TACTICAL = [
    "ct_in_bomb_zone",
    "t_in_bomb_zone",
    "ct_scoped",
    "t_scoped",
    "ct_at_a_site",
    "ct_at_b_site",
    "t_at_a_site",
    "t_at_b_site",
    "ct_flashed",
    "t_flashed",
]

#Economy class, derived per-snapshot from equipment value.
V7_ECON = [
    "ct_eco",
    "ct_force",
    "ct_full_buy",
    "t_eco",
    "t_force",
    "t_full_buy",
]
# Per-snapshot v7 features.  bomb_planted_now remains last so the melt helper can rename the suffixed variant independently.
V7_PER_SNAP = (
    NUMERIC_BASE_NOBOMB
    + V6_PER_SNAP_TACTICAL
    + V7_WEAPONS
    + V7_TACTICAL
    + V7_ECON
    + V7_DYNAMICS
    + ["bomb_planted_now"]
)

#Round-level context is shared with v6.
V7_ROUND_CTX = V6_ROUND_CTX

V7_ALL_NUMERIC = V7_PER_SNAP + V7_ROUND_CTX + DELTA_FEATURES

#Economy thresholds (USD of equipment value, summed across the 5 players).
ECON_FORCE_MIN = 6000.0  # <6000: eco
ECON_FULL_MIN = 18000.0  # >18000: full buy

#Assembled version feature lists
V1_NUMERIC = list(NUMERIC_BASE)
V2_NUMERIC = list(NUMERIC_BASE)
V3_NUMERIC = list(NUMERIC_BASE + NUMERIC_CTX)
V5_NUMERIC = list(NUMERIC_BASE + NUMERIC_CTX + NUMERIC_V5_TACTICAL)

#Map/dataset config
MAP_FEATURE = "map_name"
EXCLUDE_MAPS = {"cs_office"}  # only 3 matches, not competitively representative
BOMB_TIMER = 40.0


#Runtime derivation helpers
def derive_bomb_planted_now(
    df, source_col="bomb_countdown", target_col="bomb_planted_now"
):
    #Set target_col = 1 if source_col < BOMB_TIMER else 0
    if source_col in df.columns:
        df[target_col] = (df[source_col] < BOMB_TIMER).astype(int)
    return df

#is_pistol_round = 1 for rounds 1 and 13 (first round of each half)
def derive_is_pistol_round(df):
    if "t_score" in df.columns and "ct_score" in df.columns:
        s = df["t_score"] + df["ct_score"]
        df["is_pistol_round"] = ((s % 12 == 0) & (s < 24)).astype(int)
    return df

#derive bomb_planted_now and is_pistol_round from raw columns at runtime
def derive_runtime_features(df):
    derive_bomb_planted_now(df)
    derive_is_pistol_round(df)
    return df

#v6 runtime features. bomb_planted_now_t75 + the v6 delta features
def derive_runtime_features_v6(df):
    derive_runtime_features(df)
    derive_bomb_planted_now(df, "bomb_countdown_t75", "bomb_planted_now_t75")
    return df

#v7 runtime features. v6 + the v7 econ classification
def add_delta_features(df):
    eps = 1e-6
    df["health_share"] = df["ct_total_health"] / (
        df["ct_total_health"] + df["t_total_health"] + eps
    )
    df["alive_diff"] = df["ct_players_alive"] - df["t_players_alive"]
    df["equip_share"] = df["ct_total_equip_value"] / (
        df["ct_total_equip_value"] + df["t_total_equip_value"] + eps
    )
    df["money_diff"] = df["ct_avg_money"] - df["t_avg_money"]
    return df

#Filter a list of features to present in the dataframe
def filter_present(feature_list, df):
    return [f for f in feature_list if f in df.columns]

#Derive econ classification (eco/force/full_buy) for the given side based on total equip value.
def derive_econ_classification(df, side):
    col = f"{side}_total_equip_value"
    if col not in df.columns:
        return df
    v = df[col].fillna(0)
    df[f"{side}_eco"] = (v < ECON_FORCE_MIN).astype(int)
    df[f"{side}_force"] = ((v >= ECON_FORCE_MIN) & (v < ECON_FULL_MIN)).astype(int)
    df[f"{side}_full_buy"] = (v >= ECON_FULL_MIN).astype(int)
    return df

#Derive v7 econ classification for both sides.
def derive_v7_econ(df):
    derive_econ_classification(df, "ct")
    derive_econ_classification(df, "t")
    return df

#Derive all v7 runtime features
def derive_runtime_features_v7(df):
    derive_runtime_features_v6(df)
    derive_v7_econ(df)
    return df

#v7 melt helper
V7_DYN_BASE = [
    ("ct_total_health", "ct_health_change"),
    ("t_total_health", "t_health_change"),
    ("ct_players_alive", "ct_alive_change"),
    ("t_players_alive", "t_alive_change"),
    ("ct_total_equip_value", "ct_equip_change"),
    ("t_total_equip_value", "t_equip_change"),
    ("ct_total_armor", "ct_armor_change"),
    ("t_total_armor", "t_armor_change"),
]

#v7 melt helper function
def melt_v7(df_wide):
    import pandas as pd
    meta_cols = ["match_id", "round", MAP_FEATURE, "target"] + V7_ROUND_CTX
    def _snap(df, suffix, label, prev_suffix):
        meta = (
            df[[c for c in meta_cols if c in df.columns]].copy().reset_index(drop=True)
        )
        if suffix == "":
            snap = (
                df[[c for c in V7_PER_SNAP if c in df.columns]]
                .copy()
                .reset_index(drop=True)
            )
        else:
            rename = {f"{c}_{suffix}": c for c in V7_PER_SNAP[:-1]}
            rename[f"bomb_planted_now_{suffix}"] = "bomb_planted_now"
            cols = [f"{c}_{suffix}" for c in V7_PER_SNAP[:-1]] + [
                f"bomb_planted_now_{suffix}"
            ]
            snap = (
                df[[c for c in cols if c in df.columns]]
                .rename(columns=rename)
                .reset_index(drop=True)
            )
        #Dynamics deltas
        cur_suf = f"_{suffix}" if suffix else ""
        prev_suf = f"_{prev_suffix}" if prev_suffix else ""
        for base, delta_name in V7_DYN_BASE:
            cur_col, prev_col = f"{base}{cur_suf}", f"{base}{prev_suf}"
            if cur_col in df.columns and prev_col in df.columns:
                snap[delta_name] = (
                    df[cur_col].fillna(0).reset_index(drop=True)
                    - df[prev_col].fillna(0).reset_index(drop=True)
                ).astype(float)
            else:
                snap[delta_name] = 0.0
        out = pd.concat([meta, snap], axis=1)
        out["_snap"] = label
        return out

    melted = pd.concat(
        [
            _snap(df_wide, "", "t50", prev_suffix="t25"),
            _snap(df_wide, "t75", "t75", prev_suffix=""),
        ],
        ignore_index=True,
    )
    add_delta_features(melted)
    derive_v7_econ(melted)
    return melted
