#Central definitions for LightGBM hyperparameters.
LGB_BASE = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

#v4
LGB_V4_PARAMS = {**LGB_BASE}

#v5
LGB_V5_PARAMS = {
    **LGB_BASE,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "num_leaves": 63,
}

# v6
LGB_V6_PARAMS = {
    **LGB_BASE,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "num_leaves": 127,
}

# v7
LGB_V7_PARAMS = {
    **LGB_BASE,
    "subsample": 1.0,
    "colsample_bytree": 0.8,
    "num_leaves": 127,
}
