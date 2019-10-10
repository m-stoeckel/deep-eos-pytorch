import numpy as np


class AverageMeter:
    def __init__(self):
        self.vals = np.array([], dtype=np.float)
        self._avg = 0.0
        self._max = 0.0
        self._min = 0.0

    def update(self, val):
        self.vals = np.append(self.vals, val)
        self._avg = np.mean(self.vals)
        self._max = np.max(self.vals)
        self._min = np.min(self.vals)

    @property
    def avg(self):
        return self._avg

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max
