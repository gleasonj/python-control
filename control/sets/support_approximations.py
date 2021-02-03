import numpy as np

from .supporsets import SupportSet, supvec, supfcn

from .samplers import UniformDSphereSampler

from .polytopes import HPolytope, VPolytope

def overapproximate(S, L):
    if not isinstance(S, SupportSet):
        raise TypeError('S needs to be a support set')

    if isinstance(L, int) and L > 0:
        L = UniformDSphereSampler(S.dim).sample(L)
    elif isinstance(L, int) and L <= 0:
        raise ValueError('L must be a positive integer or a set of direction '
            'vectors for sampling the support set.')
    
    if not isinstance(L, np.ndarray):
        raise TypeError('L must be a positive integer or a set of direction '
            'vectors for sampling the support set.')

    return HPolytope(L.T, supfcn(S, L))

def underapproximate(S, L):
    if not isinstance(S, SupportSet):
        raise TypeError('S needs to be a support set')

    if isinstance(L, int) and L > 0:
        L = UniformDSphereSampler(S.dim).sample(L)
    elif isinstance(L, int) and L <= 0:
        raise ValueError('L must be a positive integer or a set of direction '
            'vectors for sampling the support set.')
    
    if not isinstance(L, np.ndarray):
        raise TypeError('L must be a positive integer or a set of direction '
            'vectors for sampling the support set.')

    return VPolytope(supvec(S, L))