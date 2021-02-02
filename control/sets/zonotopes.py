import numpy as np

class Zonotope():
    def __init__(self, center, G):
        self._center = center
        self._G = G

    @property
    def center(self):
        return self._center

    @property
    def G(self):
        return self._G

    def __add__(self, S):
        if isinstance(S, Zonotope):
            return Zonotope(self.center + S.center, np.hstack(self.G, S.G))
        else:
            return NotImplemented
