from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any


class Model(ABC):
    """Small shared base for ML model wrappers in this repo."""

    def __init__(self, **params: Any) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(levelname)s] %(name)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Avoid duplicate logs if root logger is also configured elsewhere (common in notebooks).
        self.logger.propagate = False

        self.params: dict[str, Any] = dict(params)
        self.is_fitted_ = False

    @abstractmethod
    def fit(self, X: Any, y: Any, **kwargs: Any) -> "Model":
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        _ = deep
        return dict(self.params)

    def set_params(self, **params: Any) -> "Model":
        self.params.update(params)
        return self
