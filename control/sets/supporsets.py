import numpy as np

from .hypperrectangle import Hyperrectangle

class SupportSet():
    ''' Abstract Set discriptor

    An AbstratSupportSet should redefine the method supvec which for a given 
    direction l, which is a 1d numpy ndarray with unit 2-norm magnitude, returns
    the support vector.

    NOTE: Classes should only be defined as abstract support sets if they have
          the supvec method defined.
    '''
    def __call__(self, l: np.ndarray):
        raise NotImplementedError('Any subclass of SupportSet needs to define '
            'an appropriate __call__ function.')

    def __rmatmul__(self, M):
        return MatMulSupportSet(self, M)

    def __add__(self, B):
        if isinstance(B, SupportSet):
            return MinkowskiAdditionSupportSet(self, B)
        elif isinstance(B, np.ndarray):
            if len(B.shape) == 1:
                return MinkowskiAdditionSupportSet(self, 
                    SingletonSupportSet(B))
            else:
                return NotImplemented
        else:
            return NotImplemented

def supvec(S, l: np.ndarray):
    if not isinstance(S, (SupportSet, Hyperrectangle)):
        raise TypeError('Set must be a type of AbstractSupportSet')

    if not isinstance(l, np.ndarray):
        raise TypeError('Direction vector(s) must be a 1d numpy array or a '
            '2d numpy array where each column represents a direction.')
    
    if len(l.shape) > 2:
        raise ValueError('Direction vector(s) must be a 1d numpy array or a '
            '2d numpy array where each column represents a direction.')

    if len(l.shape) == 2 and not S.dim == l.shape[0]:
        raise ValueError('Dimension mismatch between direction(s) and set')

    if len(l.shape) == 2:
        return np.array([supvec(S, v) for v in l.T]).T
    
    if isinstance(S, SupportSet):
        return S(l / np.linalg.norm(l))
    elif isinstance(S, Hyperrectangle):
        return supvec(HyperrectangleSupportSet(S), l)
    else:
        raise TypeError('Cannot compute support function for set type {}. '
            'Available types include: SupportSet, Hyperrectangle')

def supfcn(S: SupportSet, l: np.ndarray):
    return l.T @ supvec(S, l)

class HyperrectangleSupportSet(SupportSet):
    def __init__(self, S: Hyperrectangle):
        self._S = S

    def __call__(self, l):
        v = np.zeros(self.dim)
        for i in range(self.dim):
            v[i] = self.lb[i] if l[i] < 0 else self.ub[i]

        return v

class SingletonSupportSet(SupportSet):
    def __init__(self, x: np.ndarray):
        self._x = x

    def __call__(self, l):
        return self._x

class MinkowskiAdditionSupportSet(SupportSet):
    def __init__(self, A: SupportSet, B: SupportSet):
        self._A = A
        self._B = B

    def __call__(self, l):
        return self._A(l) + self._B(l)

class MatMulSupportSet(SupportSet):
    def __init__(self, S: SupportSet, M: np.ndarray):
        self._S = S
        self._M = M

    def __call__(self, l):
        return self._M @ self._S(self._M.T @ l)

class SupportSetCartesianProduct(SupportSet):
    def __init__(self, A, B):
        self._A = A
        self._B = B

    def __call__(self, l):
        np.concatenate((self._A(l), self._B(l)))