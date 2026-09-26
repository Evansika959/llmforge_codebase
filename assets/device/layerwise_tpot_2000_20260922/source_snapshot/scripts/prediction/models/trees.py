"""Default XGBoost model specification, shared by independent and stacked fits."""
from xgboost import XGBRegressor


def make_xgboost(seed=42, depth=3):
    return XGBRegressor(n_estimators=1500, learning_rate=.03, max_depth=depth,
                        min_child_weight=3, reg_lambda=5, subsample=.85,
                        colsample_bytree=.9, objective='reg:squarederror',
                        tree_method='hist', n_jobs=4,
                        early_stopping_rounds=50, random_state=seed)
