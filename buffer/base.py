import numpy as np
from abc import ABC, abstractmethod


class Buffer(ABC):
    def __init__(self, max_size: int, batch_size: int, shuffle: bool = True) -> None:
        super().__init__()
        self.shuffle = shuffle
        self.max_size = max_size
        self.batch_size = batch_size

    @abstractmethod
    def add(self, *args) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def get_len(self) -> int:
        pass

    @abstractmethod
    def sample(self):
        pass

    def __len__(self):
        return self.get_len()
