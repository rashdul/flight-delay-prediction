from __future__ import annotations

from typing import Any, Optional

import xgboost as xgb

from .model import Model


class XGBoost(Model):
	def __init__(
		self,
		*,
		use_gpu: bool = False,
		enable_categorical: bool = True,
		**params: Any,
	) -> None:
		default_params: dict[str, Any] = {
			"objective": "reg:squarederror",
			"eval_metric": "rmse",
			"learning_rate": 0.01,  # alias: eta
			"max_depth": 6,
			"subsample": 0.8,
			"colsample_bytree": 0.8,
			"seed": 42,
			"nthread": -1,
		}

		# XGBoost GPU support depends on the installed build.
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
		self.use_gpu = use_gpu
		self.enable_categorical = enable_categorical
		self.model: Optional[xgb.Booster] = None
		self.cv_results_: Any = None
		self.best_iteration_: Optional[int] = None

		self.logger.info("XGBoost initialized with parameters: %s", self.params)

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
				# If conversion fails, keep original column.
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
		nfold: Optional[int] = None,
		cv_seed: Optional[int] = None,
		cv_shuffle: bool = True,
		num_boost_round: int = 2000,
		early_stopping_rounds: Optional[int] = 50,
		verbose_eval: bool | int = False,
		**kwargs: Any,
	) -> "XGBoost":
		_ = kwargs
		self.logger.info("Starting XGBoost training...")

		dtrain = self._make_dmatrix(X, y)
		print("We are here")

		# Optional CV mode: use xgboost.cv to select a good boosting round count,
		# then train a final model on the full training set.
		if nfold is not None and nfold > 1:
			seed = int(
				cv_seed
				if cv_seed is not None
				else self.params.get("seed", self.params.get("random_state", 42))
			)

			self.logger.info("Running %d-fold CV to select num_boost_round...", nfold)
			self.cv_results_ = xgb.cv(
				params=self.params,
				dtrain=dtrain,
				nfold=nfold,
				num_boost_round=num_boost_round,
				early_stopping_rounds=early_stopping_rounds,
				seed=seed,
				shuffle=cv_shuffle,
				verbose_eval=verbose_eval,
			)

			# cv_results_ is typically a DataFrame-like object with one row per round.
			best_round = int(len(self.cv_results_))
			self.best_iteration_ = best_round - 1 if best_round > 0 else None
			self.logger.info("CV selected best_round=%d", best_round)

			# Train final model on full data for the chosen number of rounds.
			evals: list[tuple[xgb.DMatrix, str]] = [(dtrain, "train")]
			if eval_set:
				for i, (Xe, ye) in enumerate(eval_set):
					evals.append((self._make_dmatrix(Xe, ye), f"valid_{i}"))

			self.model = xgb.train(
				params=self.params,
				dtrain=dtrain,
				num_boost_round=best_round,
				evals=evals,
				verbose_eval=verbose_eval,
			)

			self.is_fitted_ = True
			self.logger.info("Training completed.")
			return self

		evals: list[tuple[xgb.DMatrix, str]] = [(dtrain, "train")]
		if eval_set:
			for i, (Xe, ye) in enumerate(eval_set):
				evals.append((self._make_dmatrix(Xe, ye), f"valid_{i}"))

		self.model = xgb.train(
			params=self.params,
			dtrain=dtrain,
			num_boost_round=num_boost_round,
			evals=evals,
			early_stopping_rounds=early_stopping_rounds,
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

		# Version-compatible prediction at best iteration.
		best_ntree_limit = getattr(self.model, "best_ntree_limit", None)
		best_iteration = getattr(self.model, "best_iteration", None)

		if best_ntree_limit is not None:
			return self.model.predict(dtest, ntree_limit=best_ntree_limit)
		if best_iteration is not None:
			return self.model.predict(dtest, iteration_range=(0, best_iteration + 1))
		return self.model.predict(dtest)

	# get_params / set_params are inherited from Model

