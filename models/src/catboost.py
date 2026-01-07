from __future__ import annotations

from typing import Any, Optional

from catboost import CatBoostClassifier, CatBoostRegressor, Pool, cv

from sklearn.model_selection import train_test_split

from .model import Model


class CatBoost(Model):
    def __init__(
        self,
        *,
        use_gpu: bool = False,
        classification: bool = False,
        **params: Any,
    ) -> None:
        default_params: dict[str, Any] = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": 42,
            "verbose": False,
        }

        if use_gpu:
            default_params.update({"task_type": "GPU"})

        super().__init__(**{**default_params, **params})
        self.use_gpu = use_gpu
        self.classification = classification
        self.model: Optional[CatBoostRegressor | CatBoostClassifier] = None

        self.cv_results_: Any = None
        self.best_iteration_: Optional[int] = None

        self.logger.info("CatBoost initialized with parameters: %s", self.params)

    def _should_use_classifier(self) -> bool:
        if self.classification:
            return True

        # Heuristic: if classification-only params/metrics are present, pick classifier.
        loss_function = str(self.params.get("loss_function", "")).strip()
        eval_metric = str(self.params.get("eval_metric", "")).strip()
        if "auto_class_weights" in self.params:
            return True
        if loss_function in {"Logloss", "CrossEntropy", "MultiClass", "MultiClassOneVsAll"}:
            return True
        if eval_metric in {"AUC", "Accuracy", "F1", "Precision", "Recall", "TotalF1"}:
            return True
        return False

    def _metric_direction(self) -> str:
        """Return 'max' for metrics where higher is better, else 'min'."""
        metric = str(self.params.get("eval_metric", self.params.get("metric", ""))).strip().lower()
        if metric in {"auc", "accuracy", "f1", "precision", "recall", "average_precision"}:
            return "max"
        return "min"

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        eval_set: Optional[tuple[Any, Any]] = None,
        nfold: Optional[int] = None,
        cv_seed: Optional[int] = None,
        cv_shuffle: bool = True,
        early_stopping_rounds: int = 50,
        **kwargs: Any,
    ) -> "CatBoost":
        extra_fit_kwargs = dict(kwargs)
        # Prevent collisions with parameters managed by this wrapper.
        for k in {"cat_features", "eval_set", "early_stopping_rounds", "use_best_model"}:
            extra_fit_kwargs.pop(k, None)

        use_classifier = self._should_use_classifier()
        model_cls = CatBoostClassifier if use_classifier else CatBoostRegressor

        cat_features: list[int] = []
        if hasattr(X, "select_dtypes"):
            cat_cols = X.select_dtypes(include=["object", "category"]).columns.to_list()
            cat_features = [X.columns.get_loc(c) for c in cat_cols]

        # If the caller didn't provide an eval_set, create a small holdout split.
        # This also avoids the notebook error: "cannot unpack non-iterable NoneType object".
        if eval_set is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=0.2,
                stratify=y if use_classifier else None,
                random_state=42,
            )
            train_X, train_y = X_train, y_train
            eval_set = (X_val, y_val)
        else:
            train_X, train_y = X, y


        # Optional CV mode: use catboost.cv to pick a good iteration count,
        # then train a final model on the full training set.
        if nfold is not None and nfold > 1:
            seed = int(
                cv_seed
                if cv_seed is not None
                else self.params.get("random_seed", self.params.get("seed", 42))
            )

            train_pool = Pool(train_X, train_y, cat_features=cat_features)

            cv_params = dict(self.params)
            # Ensure iterations exists for CV; CatBoost expects it in params.
            cv_params.setdefault("iterations", 2000)

            self.logger.info("Running %d-fold CV to select iterations...", nfold)
            self.cv_results_ = cv(
                pool=train_pool,
                params=cv_params,
                fold_count=nfold,
                shuffle=cv_shuffle,
                stratified=True if use_classifier else False,
                partition_random_seed=seed,
                early_stopping_rounds=early_stopping_rounds,
                verbose=False,
            )

            # Pick best iteration from the CV table.
            best_iter: Optional[int] = None
            if hasattr(self.cv_results_, "columns"):
                test_mean_cols = [
                    c
                    for c in list(self.cv_results_.columns)
                    if isinstance(c, str) and c.startswith("test-") and c.endswith("-mean")
                ]
                if test_mean_cols:
                    metric_col = test_mean_cols[0]
                    try:
                        s = self.cv_results_[metric_col].astype(float)
                        best_iter = int(s.idxmax() if self._metric_direction() == "max" else s.idxmin())
                    except Exception:
                        best_iter = None

            if best_iter is None:
                # Fallback: last row.
                best_iter = int(len(self.cv_results_)) - 1 if self.cv_results_ is not None else None

            self.best_iteration_ = best_iter

            final_params = dict(self.params)
            if best_iter is not None:
                final_params["iterations"] = int(best_iter) + 1

            if use_classifier:
                self.model = model_cls(**final_params, use_best_model=True)
            else:
                self.model = model_cls(**final_params)

            fit_kwargs: dict[str, Any] = {
                "cat_features": cat_features,
            }
            fit_kwargs.update(extra_fit_kwargs)

            # If eval_set is provided, keep it for monitoring, but don't early-stop again;
            # CV already selected the iteration count.
            if eval_set is not None:
                X_val, y_val = eval_set
                fit_kwargs.update({"eval_set": (X_val, y_val)})

            self.logger.info("Starting CatBoost training (CV-selected iterations)...")
            self.model.fit(train_X, train_y, **fit_kwargs)
            self.is_fitted_ = True
            self.logger.info("Training completed.")
            return self

        self.model = model_cls(**self.params)

        fit_kwargs: dict[str, Any] = {
            "cat_features": cat_features,
            "use_best_model": True,
        }
        fit_kwargs.update(extra_fit_kwargs)

        if eval_set is not None:
            X_val, y_val = eval_set
            fit_kwargs.update(
                {
                    "eval_set": (X_val, y_val),
                    "early_stopping_rounds": early_stopping_rounds,
                }
            )

        self.logger.info("Starting CatBoost training...")
        self.model.fit(train_X, train_y, **fit_kwargs)
        self.is_fitted_ = True
        self.logger.info("Training completed.")
        return self

    def predict(self, X: Any, **kwargs: Any) -> Any:
        if self.model is None:
            raise ValueError("Model is not fitted yet. Call fit(X, y) first.")
        return self.model.predict(X, **kwargs)

    def predict_proba(self, X: Any, **kwargs: Any) -> Any:
        if self.model is None:
            raise ValueError("Model is not fitted yet. Call fit(X, y) first.")
        if not hasattr(self.model, "predict_proba"):
            raise TypeError("predict_proba is only available for classification models.")
        return self.model.predict_proba(X, **kwargs)
