#parses CS2 GSI snapshots into model feature dicts.

from map_zones import classify_at_site

class MatchNotStarted(Exception):
    pass

class WarmUp(Exception):
    pass

class EmptyServer(Exception):
    pass

CT = "CT"
T = "T"

# Weapons that count as a primary
PRIMARIES = {
    "weapon_ak47",
    "weapon_m4a1",
    "weapon_m4a1_silencer",
    "weapon_m4a4",
    "weapon_aug",
    "weapon_sg556",
    "weapon_awp",
    "weapon_ssg08",
    "weapon_g3sg1",
    "weapon_scar20",
    "weapon_famas",
    "weapon_galilar",
    "weapon_mac10",
    "weapon_mp5sd",
    "weapon_mp7",
    "weapon_mp9",
    "weapon_bizon",
    "weapon_p90",
    "weapon_ump45",
    "weapon_negev",
    "weapon_m249",
    "weapon_nova",
    "weapon_xm1014",
    "weapon_mag7",
    "weapon_sawedoff",
}

# v6 weapon-class buckets
WEAPON_SNIPERS_NAMES = {
    "weapon_awp",
    "weapon_g3sg1",
    "weapon_scar20",
    "weapon_ssg08",
}
WEAPON_RIFLES_NAMES = {
    "weapon_ak47",
    "weapon_aug",
    "weapon_famas",
    "weapon_galilar",
    "weapon_m4a4",
    "weapon_m4a1",
    "weapon_m4a1_silencer",
    "weapon_sg556",
}
WEAPON_SMGS_NAMES = {
    "weapon_mac10",
    "weapon_p90",
    "weapon_mp5sd",
    "weapon_ump45",
    "weapon_bizon",
    "weapon_mp7",
    "weapon_mp9",
}

# v6 econ thresholds
ECON_FORCE_MIN = 6000.0
ECON_FULL_MIN = 18000.0

# Bomb-zone radius
BOMB_ZONE_RADIUS_SQ = 350.0**2

# state.flashed 0-255. >0 flashed
FLASH_THRESHOLD = 0

#Parse GSI 'x, y, z' position string -> (x, y, z) tuple of floats, or None.
def _parse_position(pos_str):

    if not pos_str:
        return None
    try:
        parts = [float(x) for x in pos_str.split(",")]
    except (ValueError, AttributeError):
        return None
    return tuple(parts) if len(parts) >= 2 else None

#Raise descriptive exceptions for unusable snapshots
def validate(snapshot):
    if not snapshot.get("allplayers"):
        raise MatchNotStarted("No allplayers -- not spectating a live match")
    if snapshot.get("map", {}).get("phase") == "warmup":
        raise WarmUp("Match is in warmup")
    if len(snapshot["allplayers"]) == 0:
        raise EmptyServer("Server is empty")
    return snapshot

#Convert a raw CS2 GSI snapshot into the feature dict used by v1-v6
def parse(snapshot) -> dict:
    players = snapshot["allplayers"]
    map_data = snapshot.get("map", {})
    map_name = map_data.get("name", "unknown")

    #per-team accumulators
    ct_health = ct_armor = ct_alive = ct_helmets = ct_defusers = 0
    ct_equip = ct_primaries = ct_flash = ct_he = ct_smoke = ct_molotov = 0
    ct_money = []
    ct_snipers = ct_rifles = ct_smgs = 0
    ct_scoped = ct_flashed = ct_in_bomb_zone = 0
    ct_at_a_site = ct_at_b_site = 0

    t_health = t_armor = t_alive = t_helmets = t_defusers = 0
    t_equip = t_primaries = t_flash = t_he = t_smoke = t_molotov = 0
    t_money = []
    t_snipers = t_rifles = t_smgs = 0
    t_scoped = t_flashed = t_in_bomb_zone = 0
    t_at_a_site = t_at_b_site = 0

    #Bomb info
    bomb_data = snapshot.get("bomb", {})
    bomb_state = bomb_data.get("state", "")
    bomb_planted = 1 if bomb_state == "planted" else 0
    bomb_pos = _parse_position(bomb_data.get("position", "")) if bomb_planted else None

    for pid, p in players.items():
        state = p.get("state", {})
        team = p.get("team", "")
        if team not in (CT, T):
            continue

        health = state.get("health", 0)
        alive = health > 0
        armor = state.get("armor", 0)
        helmet = state.get("helmet", False)
        defuser = state.get("defusekit", False)
        money = state.get("money", 0)
        equip = state.get("equip_value", 0)
        flashed = state.get("flashed", 0)

        #Weapon inventory
        weapons = p.get("weapons", {})
        weapon_names = [w.get("name", "") for w in weapons.values()]
        has_primary = any(n in PRIMARIES for n in weapon_names)
        flashes = sum(1 for n in weapon_names if n == "weapon_flashbang")
        hes = sum(1 for n in weapon_names if n == "weapon_hegrenade")
        smokes = sum(1 for n in weapon_names if n == "weapon_smokegrenade")
        molotovs = sum(
            1 for n in weapon_names if n in ("weapon_molotov", "weapon_incgrenade")
        )

        #Weapon classes
        has_sniper = (
            any(n in WEAPON_SNIPERS_NAMES for n in weapon_names) if alive else False
        )
        has_rifle = (
            any(n in WEAPON_RIFLES_NAMES for n in weapon_names) if alive else False
        )
        has_smg = any(n in WEAPON_SMGS_NAMES for n in weapon_names) if alive else False

        #Active weapon
        active_name = next(
            (w.get("name") for w in weapons.values() if w.get("state") == "active"),
            None,
        )
        is_scoped_proxy = alive and active_name in WEAPON_SNIPERS_NAMES

        #Parse position once
        ppos = _parse_position(p.get("position", "")) if alive else None

        #In-bomb-zone check
        in_zone = False
        if ppos is not None and bomb_pos is not None:
            dx = ppos[0] - bomb_pos[0]
            dy = ppos[1] - bomb_pos[1]
            in_zone = (dx * dx + dy * dy) < BOMB_ZONE_RADIUS_SQ

        #Bombsite zone classification
        site = classify_at_site(map_name, ppos) if ppos is not None else None
        at_a = site == "A"
        at_b = site == "B"

        is_flashed = alive and flashed > FLASH_THRESHOLD

        if team == CT:
            if alive:
                ct_alive += 1
            ct_health += health
            ct_armor += armor
            ct_helmets += int(helmet)
            ct_defusers += int(defuser)
            ct_equip += equip
            ct_primaries += int(has_primary)
            ct_flash += flashes
            ct_he += hes
            ct_smoke += smokes
            ct_molotov += molotovs
            ct_money.append(money)
            ct_snipers += int(has_sniper)
            ct_rifles += int(has_rifle)
            ct_smgs += int(has_smg)
            ct_scoped += int(is_scoped_proxy)
            ct_flashed += int(is_flashed)
            ct_in_bomb_zone += int(in_zone)
            ct_at_a_site += int(at_a)
            ct_at_b_site += int(at_b)
        else:
            if alive:
                t_alive += 1
            t_health += health
            t_armor += armor
            t_helmets += int(helmet)
            t_defusers += int(defuser)
            t_equip += equip
            t_primaries += int(has_primary)
            t_flash += flashes
            t_he += hes
            t_smoke += smokes
            t_molotov += molotovs
            t_money.append(money)
            t_snipers += int(has_sniper)
            t_rifles += int(has_rifle)
            t_smgs += int(has_smg)
            t_scoped += int(is_scoped_proxy)
            t_flashed += int(is_flashed)
            t_in_bomb_zone += int(in_zone)
            t_at_a_site += int(at_a)
            t_at_b_site += int(at_b)

    #v3 context
    round_num = map_data.get("round", 0)
    t_score = map_data.get("team_t", {}).get("score", 0)
    ct_score = map_data.get("team_ct", {}).get("score", 0)

    #v4 features
    ct_has_kit = 1 if ct_defusers > 0 else 0
    bomb_countdown = float(bomb_data.get("countdown", 40.0)) if bomb_planted else 40.0
    round_time_remaining = float(map_data.get("phase_ends_in", 0.0))

    #v5 features
    bomb_planted_now = bomb_planted
    total_score = t_score + ct_score
    is_pistol_round = 1 if (total_score % 12 == 0 and total_score < 24) else 0

    #v5 deltas
    eps = 1e-6
    ct_avg_money = sum(ct_money) / len(ct_money) if ct_money else 0
    t_avg_money = sum(t_money) / len(t_money) if t_money else 0
    health_share = ct_health / (ct_health + t_health + eps)
    alive_diff = ct_alive - t_alive
    equip_share = ct_equip / (ct_equip + t_equip + eps)
    money_diff = ct_avg_money - t_avg_money

    #v6 econ classification
    def _econ_flags(equip_total):
        return (
            int(equip_total < ECON_FORCE_MIN),  # eco
            int(ECON_FORCE_MIN <= equip_total < ECON_FULL_MIN),  # force
            int(equip_total >= ECON_FULL_MIN),
        )

    ct_eco, ct_force, ct_full_buy = _econ_flags(ct_equip)
    t_eco, t_force, t_full_buy = _econ_flags(t_equip)

    return {
        #v1 base
        "ct_total_health": ct_health,
        "ct_total_armor": ct_armor,
        "ct_players_alive": ct_alive,
        "ct_helmets": ct_helmets,
        "ct_defusers": ct_defusers,
        "ct_total_equip_value": ct_equip,
        "ct_primaries": ct_primaries,
        "ct_flashbangs": ct_flash,
        "ct_hegrenades": ct_he,
        "ct_smokes": ct_smoke,
        "ct_molotovs": ct_molotov,
        "ct_avg_money": ct_avg_money,
        "t_total_health": t_health,
        "t_total_armor": t_armor,
        "t_players_alive": t_alive,
        "t_helmets": t_helmets,
        "t_defusers": t_defusers,
        "t_total_equip_value": t_equip,
        "t_primaries": t_primaries,
        "t_flashbangs": t_flash,
        "t_hegrenades": t_he,
        "t_smokes": t_smoke,
        "t_molotovs": t_molotov,
        "t_avg_money": t_avg_money,
        "map_name": map_name,
        #v3 context
        "round_num": round_num,
        "t_score": t_score,
        "ct_score": ct_score,
        "bomb_planted": bomb_planted,
        #Not available from GSI
        "t_loss_streak": 0,
        "ct_loss_streak": 0,
        "rank_diff": 0,
        "t_dead_rank_avg": 0,
        "ct_dead_rank_avg": 0,
        #v4
        "ct_has_kit": ct_has_kit,
        "bomb_countdown": bomb_countdown,
        "round_time_remaining": round_time_remaining,
        #v5
        "bomb_planted_now": bomb_planted_now,
        "is_pistol_round": is_pistol_round,
        "health_share": health_share,
        "alive_diff": alive_diff,
        "equip_share": equip_share,
        "money_diff": money_diff,
        #v6 weapons
        "ct_snipers": ct_snipers,
        "t_snipers": t_snipers,
        "ct_rifles": ct_rifles,
        "t_rifles": t_rifles,
        "ct_smgs": ct_smgs,
        "t_smgs": t_smgs,
        #v6 tactical
        "ct_in_bomb_zone": ct_in_bomb_zone,
        "t_in_bomb_zone": t_in_bomb_zone,
        "ct_scoped": ct_scoped,
        "t_scoped": t_scoped,
        "ct_at_a_site": ct_at_a_site,
        "ct_at_b_site": ct_at_b_site,
        "t_at_a_site": t_at_a_site,
        "t_at_b_site": t_at_b_site,
        "ct_flashed": ct_flashed,
        "t_flashed": t_flashed,
        #v6 econ
        "ct_eco": ct_eco,
        "ct_force": ct_force,
        "ct_full_buy": ct_full_buy,
        "t_eco": t_eco,
        "t_force": t_force,
        "t_full_buy": t_full_buy,
    }
