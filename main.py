"""
CS2 real-time round predictor (v7 model, LightGBM + SHAP).
SHAP overlay shows *5-second window SHAP delta*. The features whose SHAP
contribution shifted the most over the last 5 seconds, so "LATEST FACTORS"
reflects what recently moved the prediction bar by the largest % swing.
"""

import os
import time
import warnings
from collections import deque
import joblib
import numpy as np
import shap
from gsi_pinger import get_snapshot
from snapshot_parser import validate, parse, MatchNotStarted, WarmUp, EmptyServer
from gui import PredictorGUI

#Seconds to look back when ranking features by SHAP change in the overlay
SHAP_WINDOW_SECS = 5.0

_V6 = r"C:\Users\hasan\Downloads\CS2Predictor\cs2_model_v6.joblib"
_V7 = r"C:\Users\hasan\Downloads\CS2Predictor\cs2_model_v7.joblib"
MODEL_FILE = _V7 if os.path.exists(_V7) else _V6


def load_model():
    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]
    numeric_feats = bundle["numeric_features"]
    map_encoder = bundle["map_encoder"]
    all_features = bundle["all_features"]
    explainer = shap.TreeExplainer(model)
    return model, numeric_feats, map_encoder, all_features, explainer


# Base columns used to derive the v6 dynamics deltas
DYNAMICS_BASE = [
    ("ct_total_health", "ct_health_change"),
    ("t_total_health", "t_health_change"),
    ("ct_players_alive", "ct_alive_change"),
    ("t_players_alive", "t_alive_change"),
    ("ct_total_equip_value", "ct_equip_change"),
    ("t_total_equip_value", "t_equip_change"),
    ("ct_total_armor", "ct_armor_change"),
    ("t_total_armor", "t_armor_change"),
]

#Populate v7 dynamics
def update_dynamics_deltas(features_dict: dict, baseline: dict | None) -> None:
    for cur, delta in DYNAMICS_BASE:
        if baseline is None:
            features_dict[delta] = 0.0
        else:
            features_dict[delta] = float(features_dict.get(cur, 0)) - float(
                baseline.get(cur, 0)
            )


LIVE_AVAILABLE = {
    # v1 base
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
    "bomb_planted",
    "bomb_planted_now",
    "map_name",
    "map_encoded",
    # v3/v4
    "round_num",
    "t_score",
    "ct_score",
    "t_loss_streak",
    "ct_loss_streak",
    "ct_has_kit",
    "bomb_countdown",
    "round_time_remaining",
    "is_pistol_round",
    # v5 deltas
    "health_share",
    "alive_diff",
    "equip_share",
    "money_diff",
    # v6 weapons
    "ct_snipers",
    "t_snipers",
    "ct_rifles",
    "t_rifles",
    "ct_smgs",
    "t_smgs",
    # v6 tactical
    "ct_in_bomb_zone",
    "t_in_bomb_zone",
    "ct_scoped",
    "t_scoped",
    "ct_flashed",
    "t_flashed",
    "ct_at_a_site",
    "ct_at_b_site",
    "t_at_a_site",
    "t_at_b_site",
    # v6 econ
    "ct_eco",
    "ct_force",
    "ct_full_buy",
    "t_eco",
    "t_force",
    "t_full_buy",
    # v6 dynamics
    "ct_health_change",
    "t_health_change",
    "ct_alive_change",
    "t_alive_change",
    "ct_equip_change",
    "t_equip_change",
    "ct_armor_change",
    "t_armor_change",
}

def _format_value(feat: str, val) -> str:
    # Binary / flag features
    BINARY = {
        "bomb_planted",
        "bomb_planted_now",
        "ct_has_kit",
        "is_pistol_round",
        "ct_eco",
        "ct_force",
        "ct_full_buy",
        "t_eco",
        "t_force",
        "t_full_buy",
    }
    if feat in BINARY:
        return "yes" if val else "no"
    if feat == "map_name":
        return str(val)
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return str(val)
    # Shares/ratios to %
    if feat in ("health_share", "equip_share"):
        return f"{fval * 100:.0f}%"
    # Money/equipment/dynamics equip deltas to int USD 
    if (
        feat.endswith("_money")
        or feat == "money_diff"
        or feat.endswith("_equip_value")
        or feat.endswith("_equip_change")
    ):
        return f"{'+' if fval > 0 else ''}${int(round(fval)):,}"
    #Dynamics deltas
    if feat.endswith("_change") or feat == "alive_diff":
        sign = "+" if fval > 0 else ""
        return f"{sign}{int(fval) if fval.is_integer() else f'{fval:.1f}'}"
    #Integer valued counters
    if fval.is_integer():
        return str(int(fval))
    return f"{fval:.2f}"


def predict(
    features_dict,
    model,
    numeric_feats,
    map_encoder,
    all_features,
    explainer,
    shap_history: deque | None = None,
):
    #Build feature vector
    row = {f: features_dict.get(f, 0) for f in numeric_feats}
    row["map_encoded"] = map_encoder.transform([features_dict["map_name"]])[0]
    X = np.array([[row[f] for f in all_features]])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        proba = model.predict_proba(X)[0]

    #SHAP values for this snapshot
    shap_vals_raw = explainer.shap_values(X)
    if isinstance(shap_vals_raw, list):
        sv = shap_vals_raw[1][0]  # shape (n_features,)
    else:
        sv = shap_vals_raw[0]

    #5-second window delta SHAP
    ref_sv = None
    if shap_history is not None:
        now = time.time()
        shap_history.append((now, sv.copy()))

        #Trim entries older than 2× window
        while len(shap_history) > 1 and now - shap_history[0][0] > SHAP_WINDOW_SECS * 2:
            shap_history.popleft()

        #Walk oldest->newest
        for ts, sv_old in shap_history:
            if now - ts >= SHAP_WINDOW_SECS:
                ref_sv = sv_old

    if ref_sv is not None:
        score = np.abs(sv - ref_sv)  # magnitude of shift over 5s window
        sign_src = sv - ref_sv  # positive -> moved prediction toward T
    else:
        score = np.abs(sv)
        sign_src = sv

    #Pick top-3 live-available features by 5s delta score
    ranked = sorted(
        zip(all_features, score, sign_src), key=lambda x: x[1], reverse=True
    )

    top3 = []
    for feat, delta_score, delta_sign in ranked:
        if len(top3) == 3:
            break
        display = feat if feat != "map_encoded" else "map_name"
        if display not in LIVE_AVAILABLE:
            continue
        raw_val = features_dict.get(display, features_dict.get(feat, 0))
        direction = "T" if delta_sign > 0 else "CT"
        top3.append(
            {
                "feature": display,
                "value": raw_val,
                "value_str": _format_value(display, raw_val),
                "shap": delta_sign,
                "direction": direction,
            }
        )

    result = {
        "ct_pct": round(proba[0] * 100, 1),
        "t_pct": round(proba[1] * 100, 1),
        "top3": top3,
    }
    return result


def main():
    print(f"Loading model from {os.path.basename(MODEL_FILE)}...")
    model, numeric_feats, map_encoder, all_features, explainer = load_model()
    print(f"Model loaded ({len(all_features)} features). Launching overlay...")

    version = "v7" if MODEL_FILE == _V7 else "v6"
    gui = PredictorGUI(model_version=version)
    gui.set_status("Waiting for CS2...")

    streaks = {"t": 0, "ct": 0}
    last_round = 0
    last_t_score = 0
    last_ct_score = 0
    round_baseline = None  
    shap_history = deque()

    while gui.alive:
        try:
            snapshot = get_snapshot()
            if snapshot is None:
                gui.pump()
                continue

            validate(snapshot)
            features_dict = parse(snapshot)

            #Track loss streaks + per-round baseline
            current_round = features_dict["round_num"]
            if current_round != last_round:
                if last_round > 0:
                    t_score_now = features_dict["t_score"]
                    ct_score_now = features_dict["ct_score"]
                    if t_score_now > last_t_score:
                        streaks["t"] = 0
                        streaks["ct"] += 1
                    elif ct_score_now > last_ct_score:
                        streaks["ct"] = 0
                        streaks["t"] += 1

                last_round = current_round
                last_t_score = features_dict["t_score"]
                last_ct_score = features_dict["ct_score"]
                round_baseline = dict(features_dict)  #freeze start of round
                shap_history.clear()  #reset 5s SHAP window on new round
            else:
                if last_round == 0:
                    last_t_score = features_dict["t_score"]
                    last_ct_score = features_dict["ct_score"]
                    round_baseline = dict(features_dict)

            features_dict["t_loss_streak"] = streaks["t"]
            features_dict["ct_loss_streak"] = streaks["ct"]

            #v7 dynamics deltas
            update_dynamics_deltas(features_dict, round_baseline)

            result = predict(
                features_dict,
                model,
                numeric_feats,
                map_encoder,
                all_features,
                explainer,
                shap_history=shap_history,
            )
            result["round_time_remaining"] = features_dict.get("round_time_remaining", 0)
            result["bomb_planted"] = bool(features_dict.get("bomb_planted", 0))
            result["bomb_countdown"] = features_dict.get("bomb_countdown", 40.0)
            gui.update(result)

        except MatchNotStarted:
            gui.set_status("Waiting for match to start...")
            time.sleep(1)
        except WarmUp:
            gui.set_status("Match in warmup, waiting...")
            time.sleep(2)
        except EmptyServer:
            gui.set_status("Server empty, waiting...")
            time.sleep(1)
        except KeyboardInterrupt:
            gui.close()
            break
        except Exception as e:
            gui.set_status(f"Error: {e}, retrying...", color="#ff6666")
            time.sleep(1)

    print("Exiting.")


if __name__ == "__main__":
    main()
