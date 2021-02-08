import numpy as np

from .polytopes import HPolytope, VPolytope, Hyperrectangle

from .supporsets import SupportSet

from .convertset import convertset

def supvec(S, l: np.ndarray):
    ''' Compute the support vector for given directions.

    The support vector of the set, S, in the direction, l, is defined as:
    
            argmax    l @ x      w.r.t x
        subject to    x in S

    INPUTS:
        l   Direction(s) in which to compute the support vectors. A single 
            direction is given by a 1d numpy array; multiple directions are 
            given by a 2d numpy array where each column is a direction.

    RETURNS:
        v   Support vector(s). A single direction is given by a 1d numpy array; 
            multiple vectors are given by a 2d numpy array where each column is 
            a direction.
    '''
    if not isinstance(l, np.ndarray):
        raise TypeError('Direction vector(s) must be a 1d numpy array or a '
            '2d numpy array where each column represents a direction.')
    
    if l.ndim > 2:
        raise ValueError('Direction vector(s) must be a 1d numpy array or a '
            '2d numpy array where each column represents a direction.')

    if l.ndim == 2 and not S.dim == l.shape[0]:
        raise ValueError('Dimension mismatch between direction(s) and set')

    if l.ndim == 2:
        return np.array([supvec(S, v) for v in l.T]).T
    
    if isinstance(S, SupportSet):
        return S(l / np.linalg.norm(l))
    elif isinstance(S, (Hyperrectangle, HPolytope, VPolytope)):
        return supvec(convertset(S, SupportSet), l)
    else:
        raise TypeError('Set must be one of the following types: \n\n'
            '    SupportSet, Hyperrectangle, HPolytope, VPolytope')

def supfcn(S: SupportSet, l: np.ndarray):
    ''' Compute the support function values for given directions.

    The support function value of the set, S, in the direction, l, is defined 
    as:
    
               max    l @ x      w.r.t x
        subject to    x in S

    INPUTS:
        l   Direction(s) in which to compute the support vectors. A single 
            direction is given by a 1d numpy array; multiple directions are 
            given by a 2d numpy array where each column is a direction.

    RETURNS:
        v   Support function value(s). A single value is given by a float; 
            multiple values are given by a 1d numpy array.
    '''
    if l.ndim == 2:
        return np.array([supfcn(S, x) for x in l.T])
    else:
        return l @ supvec(S, l)