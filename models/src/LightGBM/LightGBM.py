from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple, Union


ArrayLike = Any
EvalSet = Union[Tuple[ArrayLike, ArrayLike], Sequence[Tuple[ArrayLike, ArrayLike]]]


def _import_lightgbm():
    try:
        import lightgbm as lgb  # type: ignore

        return lgb
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "lightgbm is required to use LightGBM; install it (e.g. `pip install lightgbm`)."
        ) from exc


class LightGBM:
    def __init__(
        self,
        params: Optional[dict[str, Any]] = None,
        *,
        num_boost_round: int = 100,
        early_stopping_rounds: Optional[int] = None,
        feature_name: Union[str, Sequence[str]] = "auto",
        categorical_feature: Union[str, Sequence[str], Sequence[int]] = "auto",
    ) -> None:
        self.params: dict[str, Any] = dict(params or {})
        self.num_boost_round = int(num_boost_round)
        self.early_stopping_rounds = early_stopping_rounds
        self.feature_name = feature_name
        self.categorical_feature = categorical_feature

        self.booster_ = None
        self.best_iteration_: Optional[int] = None

    @property
    def is_fitted(self) -> bool:
        return self.booster_ is not None

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        eval_set: Optional[EvalSet] = None,
        eval_metric: Optional[Union[str, Sequence[str]]] = None,
        sample_weight: Optional[ArrayLike] = None,
        eval_sample_weight: Optional[Union[ArrayLike, Sequence[ArrayLike]]] = None,
        verbose_eval: Optional[Union[bool, int]] = None,
        callbacks: Optional[Iterable[Any]] = None,
    ) -> "LightGBM":
        lgb = _import_lightgbm()

        params = dict(self.params)
        if "objective" not in params:
            params["objective"] = "regression"
        if eval_metric is not None and "metric" not in params:
            params["metric"] = eval_metric

        train_data = lgb.Dataset(
            X,
            label=y,
            weight=sample_weight,
            feature_name=self.feature_name,
            categorical_feature=self.categorical_feature,
            free_raw_data=False,
        )

        valid_sets = None
        valid_names = None

        if eval_set is not None:
            eval_pairs: Sequence[Tuple[ArrayLike, ArrayLike]]
            if isinstance(eval_set, tuple) and len(eval_set) == 2:
                eval_pairs = [eval_set]
            else:
                eval_pairs = list(eval_set)  # type: ignore[arg-type]

            valid_sets = []
            valid_names = []
            for idx, (X_val, y_val) in enumerate(eval_pairs):
                weight = None
                if eval_sample_weight is not None:
                    if isinstance(eval_sample_weight, (list, tuple)):
                        weight = eval_sample_weight[idx]
                    else:
                        weight = eval_sample_weight
                valid_sets.append(
                    lgb.Dataset(
                        X_val,
                        label=y_val,
                        weight=weight,
                        feature_name=self.feature_name,
                        categorical_feature=self.categorical_feature,
                        reference=train_data,
                        free_raw_data=False,
                    )
                )
                valid_names.append(f"valid_{idx}")

        train_callbacks = list(callbacks or [])
        if self.early_stopping_rounds is not None and valid_sets is not None:
            train_callbacks.append(lgb.early_stopping(self.early_stopping_rounds))

        booster = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            verbose_eval=verbose_eval,
            callbacks=train_callbacks or None,
        )

        self.booster_ = booster
        self.best_iteration_ = getattr(booster, "best_iteration", None) or None
        return self

    def predict(self, X: ArrayLike, *, num_iteration: Optional[int] = None) -> Any:
        if self.booster_ is None:
            raise RuntimeError("Model is not fitted. Call `fit` or `load_model` first.")
        return self.booster_.predict(X, num_iteration=num_iteration)

    def feature_importance(self, *, importance_type: str = "split") -> Any:
        if self.booster_ is None:
            raise RuntimeError("Model is not fitted. Call `fit` or `load_model` first.")
        return self.booster_.feature_importance(importance_type=importance_type)

    def save_model(self, path: Union[str, Path], *, num_iteration: Optional[int] = None) -> None:
        if self.booster_ is None:
            raise RuntimeError("Model is not fitted. Call `fit` before saving.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster_.save_model(str(path), num_iteration=num_iteration)

    @classmethod
    def load_model(cls, path: Union[str, Path], *, params: Optional[dict[str, Any]] = None) -> "LightGBM":
        lgb = _import_lightgbm()
        instance = cls(params=params)
        instance.booster_ = lgb.Booster(model_file=str(path))
        instance.best_iteration_ = getattr(instance.booster_, "best_iteration", None) or None
        return instance
