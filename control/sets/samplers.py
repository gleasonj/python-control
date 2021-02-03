import numpy as np

from .supporsets import SupportSet, supvec

from .polytopes import VPolytope, HPolytope, Hyperrectangle

class UniformRejectionSampler():
    def __init__(self, S):
        try:
            self._bbox = S.bounding_box()
        except:
            raise ValueError('Could not compute bounding box for set provided.')

        try:
            S.contains(np.zeros(S.dim))
        except:
            raise ValueError('Could not check containment for set provided.')

        self._S = S

    @property
    def S(self):
        return self._S

    def sample(self, n=1):
        if n > 1:
            return np.array([self.sample() for _ in range(n)]).T
        else:
            x = self._bbox.sample()
            while not self.S.contains(x):
                x = self._bbox.sample()

            return x

class UniformPolytopeSampler():
    def __init__(self, P: (VPolytope, HPolytope)):
        if not isinstance(P, (VPolytope, HPolytope)):
            raise ValueError('Set to be sample must be a polytope')

        self._P = P
        self._bounding_box = P.bounding_box()

    @property
    def P(self):
        return self._P

    def sample(self, n=1):
        if n > 1:
            return np.array([self.sample() for _ in range(n)]).T
        else:
            x = self._bounding_box.sample()
            while not self.P.contains(x):
                x = self._bounding_box.sample()

            return x

class UniformDBallSampler():
    def __init__(self, d: int):
        if not isinstance(d, int) and d <= 0:
            raise ValueError('Dimension must be a positive integer.')

        self._d = d
        self._hr = Hyperrectangle(-np.ones(d), np.ones(d))

    @property
    def d(self):
        return self._d

    def sample(self, n=1):
        if n > 1:
            return np.array([self.sample() for _ in range(n)]).T
        else:
            x = self._hr.sample()
            while not np.linalg.norm(x) <= 1:
                x = self._hr.sample()

            return x

class UniformDSphereSampler():
    def __init__(self, d: int):
        if not isinstance(d, int) and d <= 0:
            raise ValueError('Dimension must be a positive integer.')

        self._d = d

    @property
    def d(self):
        return self._d

    def sample(self, n=1):
        if n > 1:
            return np.array([self.sample() for _ in range(n)]).T
        else:
            l = np.random.multivariate_normal(np.zeros(self.d), np.eye(self.d))

            return l / np.linalg.norm(l)

class ConvexWalkSampler():
    def __init__(self, S: (SupportSet, VPolytope)):
        if not isinstance(S, (SupportSet, VPolytope)):
            raise ValueError('Convex walk sampler only supports the following: '
                'SupportSet, VPolytope.')

        self._S = S
        self._x = self._get_vertex()

    @property
    def S(self):
        return self._S

    @property
    def x(self):
        return self._x

    def _get_vertex(self):
        if isinstance(self.S, VPolytope):
            return self.S.V[:, np.random.randint(self.S.nv)]
        elif isinstance(self.S, SupportSet):
            return supvec(self.S, UniformDSphereSampler(self.S.dim).sample())
        else:
            raise RuntimeError('Unhandled set type.')

    def sample(self, n=1):
        if n > 1:
            return np.array([self.sample() for _ in range(n)]).T
        else:
            lam = np.random.rand()

            self._x = lam * self._x + (1 - lam) * self._get_vertex()

            return self.x