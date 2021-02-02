import numpy as np

from .hypperrectangle import Hyperrectangle

from .polytopes import HPolytope, VPolytope

from scipy.optimize import linprog

class SupportSet():
    ''' Base Support Set Class

    A SupportSet sublcass should redefine the __call__ method where for a given 
    direction l, which is a 1d numpy ndarray with unit 2-norm magnitude, returns
    the support vector.
    '''
    __array_ufunc__ = None
    def __call__(self, l: np.ndarray):
        raise NotImplementedError('Any subclass of SupportSet needs to define '
            'an appropriate __call__ function.')

    def __mul__(self, S):
        if isinstance(S, SupportSet):
            return SupportSetCartesianProduct(self, S)
        elif isinstance(S, float):
            return ScalorMulSupportSet(self, S)
        elif isinstance(S, Hyperrectangle):
            return SupportSetCartesianProduct(self, HyperrectangleSupportSet(S))
        elif isinstance(S, HPolytope):
            return SupportSetCartesianProduct(self, HPolytopeSupportSet(S))
        elif isinstance(S, VPolytope):
            return SupportSetCartesianProduct(self, VPolytopeSupportSet(S))
        else:
            return NotImplemented

    def __rmul__(self, S):
        if isinstance(S, float):
            return ScalorMulSupportSet(self, S)
        elif isinstance(S, Hyperrectangle):
            return SupportSetCartesianProduct(HyperrectangleSupportSet(S), self)
        elif isinstance(S, HPolytope):
            return SupportSetCartesianProduct(HPolytopeSupportSet(S), self)
        elif isinstance(S, VPolytope):
            return SupportSetCartesianProduct(VPolytopeSupportSet(S), self)
        else:
            return NotImplemented

    def __rmatmul__(self, M):
        return MatMulSupportSet(self, M)

    def __add__(self, B):
        if isinstance(B, SupportSet):
            return MinkowskiAdditionSupportSet(self, B)
        elif isinstance(B, np.ndarray):
            if len(B.shape) == 1:
                return MinkowskiAdditionSupportSet(self, SingletonSupportSet(B))
            else:
                return NotImplemented
        elif isinstance(B, Hyperrectangle):
            return MinkowskiAdditionSupportSet(self, 
                HyperrectangleSupportSet(B))
        elif isinstance(B, HPolytope):
            return MinkowskiAdditionSupportSet(self, HPolytopeSupportSet(B))
        elif isinstance(B, VPolytope):
            return MinkowskiAdditionSupportSet(self, VPolytopeSupportSet(B))
        else:
            return NotImplemented

    @property
    def dim(self):
        return NotImplementedError('Any subclass of SupportSet needs to define '
            'the dimension of the set.')

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
    
    if l.ndiim > 2:
        raise ValueError('Direction vector(s) must be a 1d numpy array or a '
            '2d numpy array where each column represents a direction.')

    if l.ndiim == 2 and not S.dim == l.shape[0]:
        raise ValueError('Dimension mismatch between direction(s) and set')

    if l.ndiim == 2:
        return np.array([supvec(S, v) for v in l.T]).T
    
    if isinstance(S, SupportSet):
        return S(l / np.linalg.norm(l))
    elif isinstance(S, Hyperrectangle):
        return supvec(HyperrectangleSupportSet(S), l)
    elif isinstance(S, HPolytope):
        return supvec(HPolytopeSupportSet(S), l)
    elif isinstance(S, VPolytope):
        return supvec(VPolytopeSupportSet(S), l)
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
    return l.T @ supvec(S, l)

class HPolytopeSupportSet(SupportSet):
    ''' SupportSet wrapper for an HPolytope set 
    
    INPUTS:
        S   HPolytope set
    '''
    def __init__(self, P: HPolytope):
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

class HyperrectangleSupportSet(SupportSet):
    ''' SupportSet wrapper for a Hyperrectangle set 
    
    INPUTS:
        S   Hyperrectangle set
    '''
    def __init__(self, S: Hyperrectangle):
        self._S = S

    def __call__(self, l):
        v = np.zeros(self._S.dim)
        for i in range(self._S.dim):
            v[i] = self._S.lb[i] if l[i] < 0 else self._S.ub[i]

        return v

    def bounding_box(self):
        return self._S

    @property
    def dim(self):
        return self._S.dim

    def contains(self, x):
        return self._S.contains(x)

class SingletonSupportSet(SupportSet):
    ''' SupportSet wrapper for a numpy array 

    INPUTS:
        x   Numpy array
    '''
    def __init__(self, x: np.ndarray):
        self._x = x

    def __call__(self, l):
        return self._x

    @property
    def dim(self):
        return len(self._x)

class MinkowskiAdditionSupportSet(SupportSet):
    ''' SupportSet representing the Minkowski summation of two support sets.

    The support vector of a Minkowski addition of two support sets, A and B, 
    is given by:
    
        v[A+B](l) = v[A](l) + v[B](l)

    INPUTS:
        A   SupportSet
        B   SupportSet
    '''
    def __init__(self, A: SupportSet, B: SupportSet):
        if not A.dim == B.dim:
            raise ValueError('Dimension mistmatch between added support sets: '
                '{} != {}'.format(A.dim, B.dim))
        
        self._A = A
        self._B = B

    def __call__(self, l):
        return self._A(l) + self._B(l)

    @property
    def dim(self):
        return self._A.dim

class MatMulSupportSet(SupportSet):
    ''' SupportSet representing the matrix multiplicatoin of a support set.

    The support vector of M @ A, where M is a matrix and A is a support set is
    given by:
    
        v[M @ A](l) = M @ v[A](M.T @ l)

    INPUTS:
        S   SupportSet
        M   2d, square, numpy array
    '''
    def __init__(self, S: SupportSet, M: np.ndarray):
        if not S.dim == M.shape[1]:
            raise ValueError('Dimension mismatch in M @ S: {} != {}'.format(
                M.shape[1], S.dim))
            
        if not M.shape[0] == M.shape[1]:
            raise ValueError('Multiplying matrix must be square.')
        
        self._S = S
        self._M = M

    def __call__(self, l):
        return self._M @ self._S(self._M.T @ l)

    @property
    def dim(self):
        return self._S.dim

class ScalorMulSupportSet(SupportSet):
    def __init__(self, S: SupportSet, c: float):
        self._S = S
        self._c = c

    def __call__(self, l):
        return self._c * self._S(l)

    @property
    def dim(self):
        return self._S.dim

class SupportSetCartesianProduct(SupportSet):
    ''' SupportSet representing the cartesian product of support sets.

    INPUTS:
        A   SupportSet
        B   SupportSet
    '''
    def __init__(self, A, B):
        self._A = A
        self._B = B

    def __call__(self, l):
        np.concatenate((self._A(l), self._B(l)))

    @property
    def dim(self):
        return self._A.dim + self._B.dim

    def bounding_box(self):
        return self._A.bounding_box() * self._B.bounding_box()

    def contains(self, x):
        if x.ndim > 1:
            return np.arrray([self.contains(v) for v in x.T])
        else:
            return self._A.contains(x[:self._A.dim]) and \
                self._B.contains(x[:self._A.dim])

class BallpNormSupportSet(SupportSet):
    ''' SupportSet representation of a p-Norm Ball Set

    A p-Norm Ball set with a center point, c, and radius, r, is defined as:
    
        { x | ||x - c||_p <= r }

    INPUTS:
        p       Norm p-value
        center  Center point
        radius  Radius
    '''
    def __init__(self, p: int, center: np.ndarray, radius: float):
        if not isinstance(p, int):
            raise ValueError('P-value must be an integer')

        if not isinstance(center, np.ndarray) or len(center.shape) > 1:
            raise ValueError('Center must be a 1d numpy array')

        if not isinstance(radius, float) or radius <= 0:
            raise ValueError('Radius must be a positive floating point number')
        
        self._p = p
        self._center = center
        self._radius = radius

    @property
    def p(self):
        ''' Norm p-value '''
        return self._p

    @property
    def center(self):
        ''' Center point '''
        return self._center

    @property
    def radius(self):
        ''' Radius '''
        return self._radius

    @property
    def dim(self):
        ''' Dimension '''
        return len(self._center)

    def contains(self, x):
        if not isinstance(x, np.ndarray) or x.ndim > 2:
            raise TypeError('Point(s) must be a 1d numpy array or 2d numpy '
                'array where each column represents a point.')

        if x.ndim > 1:
            return np.array([self.contains(v) for v in x.T])
        else:
            return np.linalg.norm(self.center - x, self.p) <= self.radius

    def bounding_box(self):
        return Hyperrectangle(self.center-self.radius, self.center+self.radius)

class Ball1NormSupportSet(BallpNormSupportSet):
    def __init__(self, center: np.ndarray, radius: float):
        super().__init__(1, center, radius)

class Ball2NormSupportSet(BallpNormSupportSet):
    def __init__(self, center: np.ndarray, radius: float):
        super().__init__(2, center, radius)

class BallInfNormSupportSet(BallpNormSupportSet):
    def __init__(self, center: np.ndarray, radius: float):
        super().__init__(np.inf, center, radius)