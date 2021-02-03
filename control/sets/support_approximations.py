import numpy as np

from .supporsets import SupportSet, supvec, supfcn

from .samplers import UniformDSphereSampler

from .polytopes import HPolytope, VPolytope

def overapproximate(S, L):
    ''' Overapproximate a given SupportSet using support function values

    We use support functions to determine an HPolytope overapproxatiom of a
    support set using a given set of directions, or through a given number
    of random directions on the unit d-sphere.

    INPUTS:
        S   SupportSet
        L   Either an numpy ndarray giving a set of directions to sample for
            the overapproximation, where each column represents a direction,
            or an integer specifying the number of directions to use from
            random generation on the unit d-sphere.

    OUTPUTS:
        P   HPolytope overapproximation of the set S
    '''
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
    ''' Under a given SupportSet using support vectors

    We use support vectors to determine a VPolytope underapproximation of a
    support set using a given set of directions, or through a given number
    of random directions on the unit d-sphere.

    INPUTS:
        S   SupportSet
        L   Either an numpy ndarray giving a set of directions to sample for
            the underapproximation, where each column represents a direction,
            or an integer specifying the number of directions to use from
            random generation on the unit d-sphere.

    OUTPUTS:
        P   VPolytope underapproximation of the set S
    '''
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