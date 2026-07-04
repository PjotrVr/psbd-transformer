from __future__ import annotations

from abc import ABC, abstractmethod


class DefenseBase(ABC):
    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError
