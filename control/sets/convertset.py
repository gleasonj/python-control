import numpy as np

from .special import ZeroSet, Empty, Universe

from .polytopes import HPolytope, \
                       VPolytope, \
                       Hyperrectangle, \
                       _facet_enumeration, \
                       _vertex_enumeration

from .supporsets import SupportSet

from .zonotopes import Zonotope

from .hypperrectangle import Hyperrectangle

from scipy.optimize import linprog

def convertset(S, t):
    if isinstance(S, Hyperrectangle) and t == SupportSet:
        return HyperrectangleSupportSet(S)
    elif isinstance(S, HPolytope) and t == SupportSet:
        return HPolytopeSupportSet(S)
    elif isinstance(S, VPolytope) and t == SupportSet:
        return VPolytopeSupportSet(S)
    elif isinstance(S, Hyperrectangle) and t == HPolytope:
        return S
    elif isinstance(S, t):
        return S
    elif isinstance(S, HPolytope) and t == VPolytope:
        return _vertex_enumeration(S)
    elif isinstance(S, VPolytope) and t == HPolytope:
        return _facet_enumeration(S)
    else:
        raise ValueError('Cannot convert set from type {} to {}'.format(type(S), 
            t))

class HPolytopeSupportSet(SupportSet):
    ''' SupportSet wrapper for an HPolytope set 
    
    INPUTS:
        S   HPolytope set
    '''
    def __init__(self, P: HPolytope):
        if P == Empty():
            raise ValueError('Cannot create support set from empty HPolytope')

        self._P = P

    def __call__(self, l):
        res = linprog(l, self._P.A, self._P.b, bounds=(-np.inf, np.inf))

        return res.x

    def contains(self, x):
        return self._P.contains(x)

    @property
    def dim(self):
        return self._P.dim

class VPolytopeSupportSet(SupportSet):
    ''' SupportSet wrapper for an VPolytope set 
    
    INPUTS:
        S   VPolytope set
    '''
    def __init__(self, P: VPolytope):
        self._P = P

    def __call__(self, l):
        return self._P.V[:, np.argmax(l @ self._P.V)]

    def contains(self, x):
        return self._P.contains(x)

    @property
    def dim(self):
        return self._P.dim

class HyperrectangleSupportSet(HPolytopeSupportSet):
    ''' SupportSet wrapper for a Hyperrectangle set 
    
    INPUTS:
        P   Hyperrectangle set
    '''
    def __init__(self, P: Hyperrectangle):
        super().__init__(P)

    def __call__(self, l):
        v = np.zeros(self._P.dim)
        for i in range(self._P.dim):
            v[i] = self._P.lb[i] if l[i] < 0 else self._P.ub[i]

        return v

