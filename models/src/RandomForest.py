from __future__ import annotations

from typing import Any, Optional

import xgboost as xgb

from .model import Model


class RandomForest(Model):
    """Random-forest-like model built using XGBoost.

    Implements a classic RF-style ensemble by training *many trees in parallel* with:
    - num_boost_round = 1
    - num_parallel_tree = N
    - eta (learning_rate) = 1.0
    - subsample < 1 and colsample_bytree < 1 for randomness
    """

    def __init__(
        self,
        *,
        n_trees: int = 200,
        use_gpu: bool = False,
        enable_categorical: bool = True,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        max_depth: int = 8,
        seed: int = 42,
        **params: Any,
    ) -> None:
        default_params: dict[str, Any] = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "eta": 1.0,
            "num_parallel_tree": int(n_trees),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample_bytree),
            "max_depth": int(max_depth),
            "seed": int(seed),
            "nthread": -1,
        }

        if use_gpu:
            default_params.update(
                {
                    "tree_method": "gpu_hist",
                    "predictor": "gpu_predictor",
                }
            )
        else:
            default_params.update({"tree_method": "hist"})

        super().__init__(**{**default_params, **params})
        self.enable_categorical = enable_categorical
        self.model: Optional[xgb.Booster] = None

        self.logger.info("RandomForest (XGB-RF) initialized with parameters: %s", self.params)

    def _coerce_categoricals(self, X: Any) -> Any:
        if not self.enable_categorical:
            return X
        if not hasattr(X, "select_dtypes"):
            return X

        X_out = X.copy() if hasattr(X, "copy") else X
        cat_cols = X_out.select_dtypes(include=["object", "category"]).columns.to_list()
        for c in cat_cols:
            try:
                X_out[c] = X_out[c].astype("category")
            except Exception:
                pass
        return X_out

    def _make_dmatrix(self, X: Any, y: Any | None = None) -> xgb.DMatrix:
        X_prep = self._coerce_categoricals(X)
        if y is None:
            return xgb.DMatrix(X_prep, enable_categorical=self.enable_categorical)
        return xgb.DMatrix(X_prep, label=y, enable_categorical=self.enable_categorical)

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        eval_set: Optional[list[tuple[Any, Any]]] = None,
        verbose_eval: bool | int = False,
        **kwargs: Any,
    ) -> "RandomForest":
        _ = kwargs
        self.logger.info("Starting RandomForest (XGB-RF) training...")

        dtrain = self._make_dmatrix(X, y)

        evals: list[tuple[xgb.DMatrix, str]] = [(dtrain, "train")]
        if eval_set:
            for i, (Xe, ye) in enumerate(eval_set):
                evals.append((self._make_dmatrix(Xe, ye), f"valid_{i}"))

        # RF-like: a single boosting round with many parallel trees.
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=1,
            evals=evals,
            verbose_eval=verbose_eval,
        )

        self.is_fitted_ = True
        self.logger.info("Training completed.")
        return self

    def predict(self, X: Any, **kwargs: Any) -> Any:
        _ = kwargs
        if self.model is None:
            raise ValueError("Model is not fitted yet. Call fit(X, y) first.")
        dtest = self._make_dmatrix(X)
        return self.model.predict(dtest)
