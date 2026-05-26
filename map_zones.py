#approximate bombsite centers for the 8 training maps.


# (x, y, z, radius)
MAP_BOMBSITES = {
    "de_dust2": {"A": (1240, 2430, 30, 350), "B": (-1540, 2660, 10, 350)},
    "de_mirage": {"A": (1140, -200, -180, 350), "B": (-1950, -500, -180, 350)},
    "de_inferno": {"A": (1990, 320, 165, 350), "B": (370, 2870, 110, 350)},
    "de_nuke": {"A": (700, -830, -415, 300), "B": (600, -780, -770, 300)},
    "de_overpass": {"A": (-2050, 1020, 75, 400), "B": (-1770, 380, 480, 400)},
    "de_vertigo": {"A": (-700, 60, 11580, 350), "B": (-2040, 1270, 11420, 400)},
    "de_anubis": {"A": (930, -220, 0, 450), "B": (-900, -90, 0, 450)},
    "de_ancient": {"A": (-800, -490, -60, 400), "B": (720, 380, 50, 450)},
}

#Classify a player as being at A, B, or neither based on their position and map.
def classify_at_site(map_name: str, pos_xyz) -> str | None:
    sites = MAP_BOMBSITES.get(map_name)
    if not sites or pos_xyz is None:
        return None

    px, py = pos_xyz[0], pos_xyz[1]
    pz = pos_xyz[2] if len(pos_xyz) > 2 else 0.0

    best, best_dist = None, float("inf")
    for site, (x, y, z, r) in sites.items():
        dx, dy, dz = px - x, py - y, pz - z
        d = (dx * dx + dy * dy + dz * dz) ** 0.5
        if d < r and d < best_dist:
            best, best_dist = site, d
    return best
