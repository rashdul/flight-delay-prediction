from __future__ import annotations

import logging
from typing import Any, Optional

import lightgbm as lgb


class LightGBM:
	def __init__(self, classification: bool = False, **model_kwargs: Any) -> None:
		self.logger = logging.getLogger(self.__class__.__name__)
		self.logger.setLevel(logging.INFO)

		if not self.logger.handlers:
			handler = logging.StreamHandler()
			formatter = logging.Formatter("[%(levelname)s] %(name)s - %(message)s")
			handler.setFormatter(formatter)
			self.logger.addHandler(handler)

		# Avoid duplicate logs if the root logger is also configured elsewhere (common in notebooks).
		self.logger.propagate = False

		self.classification = classification
		self.is_fitted_ = False

		default_objective = "binary" if classification else "regression"
		default_metric = "binary_logloss" if classification else "rmse"

		default_params = {
			"objective": default_objective,
			"metric": default_metric,
			"learning_rate": 0.05,
			"num_leaves": 31,
			"feature_fraction": 0.9,
			"bagging_fraction": 0.9,
			"bagging_freq": 1,
			"seed": 42,
			"boosting_type": "gbdt",
			"verbose": -1,
			"max_depth": -1,
			"lambda_l1": 0.0,
			"lambda_l2": 0.0,
			"min_data_in_leaf": 20,
		}
		self.model_kwargs = {**default_params, **model_kwargs}
		self.model = None
		self.logger.info("LightGBM model initialized with parameters: %s", self.model_kwargs)

	def fit(
		self,
		X: Any,
		y: Any,
		cat_cols: Optional[list[str]] = None,
		sample_weight: Any = None,
		**fit_kwargs: Any,
	) -> "LightGBM":
		self.logger.info("Starting model training...")
		X = X.copy()

		# Build dataset (weights, if provided, apply to both CV and final training)
		dset = lgb.Dataset(
			X,
			label=y,
			weight=sample_weight,
			categorical_feature=cat_cols,
			free_raw_data=False,
		)

		# CV params
		cv_params = {
			"num_boost_round": 500,
			"nfold": 5,
			"stratified": bool(self.classification),
			"shuffle": True,
			"callbacks": [
				lgb.early_stopping(stopping_rounds=20),
				lgb.log_evaluation(period=50),
			],
		}
		updated_cv = {**cv_params, **fit_kwargs}

		# Run CV
		cv_results = lgb.cv(
			self.model_kwargs,
			dset,
			**updated_cv,
		)

		# Extract best iteration
		metric_name = self.model_kwargs["metric"]
		best_iter = len(cv_results[f"valid {metric_name}-mean"])
		self.logger.info(f"Best iteration from CV: {best_iter}")

		# Train final model
		self.model = lgb.train(
			self.model_kwargs,
			dset,
			num_boost_round=best_iter,
		)

		self.is_fitted_ = True
		self.logger.info("Model training completed.")
		return self


	def predict(self, X: Any, proba: bool = False, threshold: float = 0.5) -> Any:
		if self.model is None:
			raise ValueError("Model is not fitted yet. Call fit(X, y) first.")

		pred = self.model.predict(X)
		if not self.classification:
			return pred

		objective = str(self.model_kwargs.get("objective", "binary")).lower()
		if proba:
			return pred
		if objective.startswith("multiclass"):
			# LightGBM returns flattened probabilities for multiclass: (n_samples * n_classes,)
			num_class = int(self.model_kwargs.get("num_class", 0) or 0)
			if num_class <= 0:
				raise ValueError("For multiclass classification, set 'num_class' in model parameters or use proba=True.")
			return pred.reshape(-1, num_class).argmax(axis=1)

		# Binary classification
		return (pred >= float(threshold)).astype(int)

	def get_params(self, deep: bool = True) -> dict[str, Any]:
		_ = deep
		return dict(self.model_kwargs)

	def set_params(self, **params: Any) -> "LightGBM":
		self.model_kwargs.update(params)
		return self

	@property
	def feature_importances_(self) -> Any:
		if self.model is None:
			raise ValueError("Model is not fitted yet. Call fit(X, y) first.")
		return self.model.feature_importances_