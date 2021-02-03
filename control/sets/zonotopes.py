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

    @property
    def dim(self):
        return len(self._center)

    def __and__(self, Z):
        if isinstance(Z, Zonotope):
            return ZonotopeBundle(self, Z)
        else:
            return NotImplemented

    def __add__(self, Z):
        if isinstance(Z, Zonotope):
            return Zonotope(self.center + Z.center, np.hstack(self.G, Z.G))
        else:
            return NotImplemented

class ZonotopeBundle():
    def __init__(self, *args):
        for Z in args:
            if not isinstance(Z, Zonotope):
                raise TypeError('All elements of a Zonotope Bunlde must be '
                    'zonotopes.')

        self._zonotopes = args

    @property
    def zonotopes(self):
        return self._zonotopes

    @property
    def dim(self):
        return self.zonotopes[0].dim

    def __iter__(self):
        self._zidx = 0
        return self

    def __next__(self):
        if self._zidx < len(self):
            Z = self.zonotopes[self._zidx]
            self._zidx += 1
            return Z
        else:
            raise StopIteration

    def __len__(self):
        return len(self.zonotopes)

    def __and__(self, Z):
        if isinstance(Z, Zonotope):
            return ZonotopeBundle(*self.zonotopes, Z)
        elif isinstance(Z, ZonotopeBundle):
            return ZonotopeBundle(*self.zonotopes, *Z.zonotopes)
        else:
            return NotImplemented

def to_zonotope(P):
    raise NotImplementedError('In progress...')