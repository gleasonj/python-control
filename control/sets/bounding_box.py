import numpy as np

from .supvec_supfcn import supfcn

from .polytopes import HPolytope, \
    VPolytope, \
    Hyperrectangle, \
    PolytopeUnion, \
    _polytope_is_bounded

from .supporsets import SupportSet, \
                        SupportSetCartesianProduct

from .zonotopes import Zonotope

from .special import ZeroSet

from scipy.optimize import linprog

def bounding_box(S):
    ''' Return the bounding box of the set

    INPUTS:
        S   Set

    RETURNS:
        bbox    Hyperrectangle object that defines the bounding box for the 
                polytope
    '''
    if isinstance(S, HPolytope):
        A = np.vstack((-np.eye(S.dim), np.eye(S.dim)))
        b = np.zeros(A.shape[0])

        for i in range(len(b)):
            a = A[i]
            b[i] = -linprog(-a, S.A, S.b, bounds=(-np.inf, np.inf)).fun

        # Using a trick here: the floor divide // will automatically convert
        # the resulting division to an int. By definition, the len(b) is an
        # even number so this will work.
        return Hyperrectangle(-b[:len(b)//2], b[len(b)//2:])
    elif isinstance(S, VPolytope):
        return Hyperrectangle(np.min(S.V, axis=1), np.max(S.V, axis=1))
    elif isinstance(S, Hyperrectangle):
        return S
    elif isinstance(S, SupportSet):
        b = supfcn(S, np.vstack((-np.eye(S.dim), np.eye(S.dim))).T)
        return Hyperrectangle(-b[:len(b)//2], b[len(b)//2:])
    elif isinstance(S, PolytopeUnion):
        X = ZeroSet()

        for P in S:
            X += bounding_box(P)
        
        return X
    else:
        raise ValueError('Cannot compute bounding box for set of type {}'.format(type(S)))


def is_bounded(S):
    if isinstance(S, (HPolytope, VPolytope)):
        return _polytope_is_bounded(S)
    else:
        try:
            bbox = bounding_box(S)
        except:
            raise TypeError('Cannot comput bounds for set of type {}'.format(S))

        return not np.any(np.abs(np.concatenate((bbox.lb, bbox.ub))) == np.inf)