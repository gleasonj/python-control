import numpy as np

from .special import Empty, Universe

from scipy.optimize import linprog
from scipy.linalg import block_diag

class HPolytope():
    __array_ufunc__ = None
    def __init__(self, A: np.ndarray, b: np.ndarray):
        if not isinstance(A, np.ndarray) or not isinstance(b, np.ndarray):
            raise TypeError('A and b must be numpy arrays.')

        if not len(A.shape) == 2:
            raise ValueError('A must be a matrix.')

        if not len(b.shape) == 1:
            raise ValueError('b must be an array.')

        self._A = A
        self._b = b

    @property
    def A(self):
        return self._A

    @property
    def b(self):
        return self._b

    @property
    def dim(self):
        return self.A.shape[1]

    @property
    def nf(self):
        ''' Number of facets '''
        return self.A.shape[0]

    def __iter__(self):
        self._iter_idx = 0
        return self
    
    def __next__(self):
        if self._iter_idx < self.nf:
            a = self.A[self._iter_idx]
            b = self.b[self._iter_idx]
            self._iter_idx += 1

            return (a, b)
        else:
            raise StopIteration
        
    def __mul__(self, P):
        if isinstance(P, VPolytope):
            return PolytopeCartesianProduct(self, P)
        elif isinstance(P, HPolytope):
            return HPolytope(block_diag(self.A, P.A), 
                np.concatenate((self.b, P.b)))
        else:
            return NotImplemented

    def __and__(self, P):
        if isinstance(P, HPolytope):
            return HPolytope(np.vstack((self.A, P.A)), 
                np.concatenate((self.b, P.b)))
        elif isinstance(P, PolytopeUnion):
            # Using distributive property of intersections and unions
            #   A & (B | C) = (A & B) | (A & C)
            return PolytopeUnion(*[poly and self for poly in P])
        else:
            return NotImplemented

    def __or__(self, P):
        if isinstance(P, (HPolytope, VPolytope)):
            return PolytopeUnion(self, P)
        elif isinstance(P, Empty):
            return self
        elif isinstance(P, Universe):
            return P
        else:
            return NotImplemented

    def __ror__(self, P):
        if isinstance(P, (HPolytope, VPolytope)):
            return PolytopeUnion(P, self)
        elif isinstance(P, (Empty, Universe)):
            return self.__or__(P)
        else:
            return NotImplemented

    def bounding_box(self):
        ''' Return the bounding box of the polytope 

        RETURNS:
            bbox    Hyperrectangle object that defines the bounding box for the 
                    polytope
        '''
        A = np.vstack((-np.eye(self.dim), np.eye(self.dim)))
        b = np.zeros(A.shape[0])

        for i in range(len(b)):
            a = A[i]
            if sum(a) == 1.0:
                b[i] = -linprog(-a, self.A, self.b, bounds=(-np.inf, np.inf)).fun
            else:
                b[i] = linprog(-a, self.A, self.b, bounds=(-np.inf, np.inf)).fun

        # Using a trick here: the floor divide // will automatically convert
        # the resulting division to an int. By definition, the len(b) is an
        # even number so this will work.
        return Hyperrectangle(b[:len(b)//2], b[len(b)//2:])

    def __rmatmul__(self, M):
        if isinstance(M, np.ndarray):
            if not len(M.shape) == 2 or not M.shape[0] == M.shape[1] or \
                np.linalg.det(M) == 0:
                raise ValueError('Can only matrix polytopes by invertible, '
                    'square 2d numpy arrays.')

            if not M.shape[0] == self.dim:
                raise ValueError('Dimension mistmatch between polytope and '
                    'multiplying matrix: {} != {}.'.format(self.dim, M.shape[0]))
            
            return HPolytope(self.A @ np.linalg.inv(M), self.b)
        else:
            return NotImplemented

    
    def contains(self, x: np.ndarray):
        if not isinstance(x, np.ndarray):
            raise TypeError('Point(s) must be a 1d numpy array or 2d numpy '
                'array where each column represents a point.')

        if x.ndim > 2:
            raise ValueError('Point(s) must be a 1d numpy array or 2d numpy '
                'array where each column represents a point.')

        return np.all(self.A @ x <= self.b, axis=0)

class VPolytope():
    __array_ufunc__ = None
    def __init__(self, V: np.ndarray):
        if not isinstance(V, np.ndarray):
            raise TypeError('V must be a 1d numpy array or 2d numpy '
                'array where each column represents a vertex.')

        if len(V.shape) > 2:
            raise ValueError('V must be a 1d numpy array or 2d numpy '
                'array where each column represents a vertex.')

        if len(V.shape) == 1:
            V = np.atleast_2d(V).T

        self._V = V

    @property
    def V(self):
        return self._V

    @property
    def dim(self):
        return self.V.shape[0]

    @property
    def nv(self):
        return self.V.shape[1]

    # The __iter__ and __next__ function allow for iteration through the
    # vertices of the HPolytope
    def __iter__(self):
        self._iter_idx = 0
        return self

    def __next__(self):
        if self._iter_idx < self.nv:
            v = self.V[:, self._iter_idx]
            self._iter_idx += 1
            return v
        else:
            raise StopIteration

    def __or__(self, P):
        if isinstance(P, (HPolytope, VPolytope)):
            return PolytopeUnion(self, P)
        elif isinstance(P, Empty):
            return self
        elif isinstance(P, Universe):
            return P
        else:
            return NotImplemented

    def __ror__(self, P):
        if isinstance(P, (HPolytope, VPolytope)):
            return PolytopeUnion(P, self)
        elif isinstance(P, (Empty, Universe)):
            return self.__or__(P)
        else:
            return NotImplemented

    def __mul__(self, P):
        if isinstance(P, HPolytope):
            return PolytopeCartesianProduct(self, P)
        elif isinstance(P, VPolytope):
            return PolytopeCartesianProduct(self, P)
        else:
            return NotImplemented

    def __add__(self, P):
        if isinstance(P, VPolytope):
            # NOTE  Minkowski summation of vertex polytopes can very quickly 
            #       become computationally intractible because of the 
            #       exponential growth in the number of vertices.
            return VPolytope(np.hstack([np.atleast_2d(v).T + self.V for v in P]))
        else: 
            return NotImplemented

    def bounding_box(self):
        return Hyperrectangle(np.min(self.V, axis=1), np.max(self.V, axis=1))

    def sample_vertex(self):
        return self.V[:, np.random.randint(self.nv)]

    def __rmatmul__(self, M):
        if isinstance(M, np.ndarray):
            if not len(M.shape) == 2:
                raise ValueError('Can only multiple VPolytopes with 2d '
                    'numpy arrays.')
            
            if not M.shape[1] == self.dim:
                raise ValueError('Dimension mismatch between multiplying '
                    'matrix and VPolytope: {} != {}'.format(M.shape[1], 
                    self.dim))

            return VPolytope(M @ self.V)
        else:
            return NotImplemented
    
    def contains(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError('Point(s) must be a 1d numpy array or 2d numpy '
                'array where each column represents a point.')

        if x.ndim > 2:
            raise ValueError('Point(s) must be a 1d numpy array or 2d numpy '
                'array where each column represents a point.')

        if x.ndim == 2:
            return np.array([self.contains(v) for v in x.T])

        res = linprog(np.zeros(self.nv), 
            A_eq=np.vstack((np.ones(self.nv), self.V)), 
            b_eq=np.concatenate((1, x)))

        return res.success

class PolytopeUnion():
    def __init__(self, *args):
        for P in args:
            if not isinstance(P, (VPolytope, HPolytope)):
                raise TypeError('All elements of a polytope union must be '
                    'polytopes.')

        self._polytopes = args

    @property
    def polytopes(self):
        return self._polytopes

    def __len__(self):
        return len(self._polytopes)

    def __iter__(self):
        self._pidx = 0
        return self

    def __next__(self):
        if self._pidx < len(self):
            P = self.polytopes[self._pidx]
            self._pidx += 1
            return P
        else:
            raise StopIteration

    def __or__(self, P):
        if isinstance(P, (HPolytope, VPolytope)):
            return PolytopeUnion(*self.polytopes, P)
        elif isinstance(P, PolytopeUnion):
            return PolytopeUnion(*self.polytopes, *P.polytopes)
        elif isinstance(P, Hyperrectangle):
            return PolytopeUnion(*self.polytopes, 
                P)
        elif isinstance(P, Empty):
            return self
        elif isinstance(P, Universe):
            return P
        else:
            return NotImplemented

    def __ror__(self, P):
        if isinstance(P, (HPolytope, VPolytope)):
            return PolytopeUnion(P, *self.polytopes)
        elif isinstance(P, PolytopeUnion):
            return PolytopeUnion(*P.polytopes, *self.polytopes)
        elif isinstance(P, Hyperrectangle):
            return PolytopeUnion(P, 
                *self.polytopes)
        elif isinstance(P, (Empty, Universe)):
            return self.__or__(P)
        else:
            return NotImplemented

    def contains(self, x):
        if x.ndim == 2:
            return np.array([self.contains(v) for v in x.T])
        else:
            return np.any([P.contains(x) for P in self.polytopes])

    def bounding_box(self):
        bbox = Empty()
        for poly in self:
            bbox += poly.bounding_box()

        return bbox

    @property
    def dim(self):
        return self.polytopes[0].dim

def _hypperrectangle_to_hpolytope(hr):
    return HPolytope(np.vstack((-np.eye(hr.dim), np.eye(hr.dim))),
        np.concatenate((-hr.lb, hr.ub)))

class PolytopeCartesianProduct():
    __array_ufunc__ = None
    def __init__(self, A, B):
        self._A = A
        self._B = B

    @property
    def dim(self):
        return self._A.dim + self._B.dim

    def bounding_box(self):
        return self._A.bounding_box() * self._B.bounding_box()

    def contains(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError('Point(s) must be a 1d numpy array or 2d numpy '
                'array where each column represents a point.')

        if x.ndim > 2:
            raise TypeError('Point(s) must be a 1d numpy array or 2d numpy '
                'array where each column represents a point.')

        if x.ndim == 2:
            return np.array([self.contains(v) for v in x.T])
        
        return self._A.contains(x[:self._A.dim]) and \
            self._B.contains(x[self._A.dim:])


class Hyperrectangle(HPolytope):
    ''' Hyperrectangle set

    A hyperrectangular set is given by lower and upper bounding vectors.

                    --------------------o upper bound
                    |                   |
                    |                   |
                    |                   |
        lower bound o--------------------

    INPUTS:
        lb  Lower bound vector
        ub  Upper bound vector

    RETURNS:
        Hyperrrectangle object
    '''
    __array_ufunc__ = None
    def __init__(self, lb: np.ndarray, ub: np.ndarray):
        assert len(lb.shape) == 1, 'Lower bound must be an array'
        assert len(ub.shape) == 1, 'Upper bound must be an array'
        assert len(lb) == len(ub), 'Lower and upper bounds must have same length'

        super().__init__(np.vstack((-np.eye(len(lb)), np.eye(len(lb)))),
            np.concatenate((-lb, ub)))

        self._lb = lb
        self._ub = ub

    def __rmul__(self, b):
        if isinstance(b, float):
            return self.__mul__(b)
        else:
            return NotImplemented

    def __mul__(self, b):
        if isinstance(b, Hyperrectangle):
            return Hyperrectangle(np.concatenate((self.lb, b.lb)),
                np.concatenate((self.ub, b.ub)))
        if isinstance(b, float):
            return self.__rmul__(b)
        else:
            return super().__mul__(b)

    def __and__(self, S):
        if isinstance(S, Hyperrectangle):
            return Hyperrectangle(np.minimum(self.lb, S.lb),
                np.maximum(self.ub, S.ub))
        else:
            return super().__and__(S)

    def __add__(self, S):
        if isinstance(S, Hyperrectangle):
            return Hyperrectangle(self.lb + S.lb, self.ub + S.ub)
        else:
            return NotImplemented

    def __radd__(self, S):
        return self.__add__(S)

    def __lt__(self, S):
        if isinstance(S, Universe):
            return True
        elif isinstance(S, Empty):
            return False
        elif isinstance(S, Hyperrectangle):
            return np.all(self.lb <= S.lb) and np.all(self.ub >= S.ub)
        else:
            return NotImplemented

    def __eq__(self, S):
        if isinstance(S, (Universe, Empty)):
            return False
        elif isinstance(S, Hyperrectangle):
            return np.all(self.lb == S.lb) and np.all(self.ub == S.ub)
        else:
            return NotImplemented

    def __gt__(self, S):
        if isinstance(S, Universe):
            return False
        elif isinstance(S, Empty):
            return True
        elif isinstance(S, Hyperrectangle):
            return S.__lt__(self)
        else:
            return NotImplemented

    @property
    def lb(self):
        ''' Lower bound vector '''
        return self._lb

    @property
    def ub(self):
        ''' Upper bound vector '''
        return self._ub

    @property
    def center(self):
        ''' Center point '''
        return (self.ub + self.lb) / 2

    @property
    def side_length(self):
        ''' Side length '''
        return (self.ub - self.lb)

    @property
    def half_length(self):
        ''' Half length (half of the side length) '''
        return self.side_length / 2

    @property
    def dim(self):
        ''' Set dimension '''
        return len(self.lb)

    @property
    def hpoly_A(self):
        return np.vstack((np.eye(self.dim), -np.eye(self.dim)))

    @property
    def hpoly_b(self):
        return np.vstack((np.ub, -np.lb))

    def bounding_box(self):
        return self

    def contains(self, x: np.ndarray):
        ''' Check is point(s) are contained in the set

        INPUTS:
            x   Point(s) to determine containment. This is either a 1d numpy 
                array, or a 2d numpy array where each column represents a single
                point.

        RETURNS:
            y   Boolean or boolean vector
        '''
        if not isinstance(x, np.ndarray) or x.ndim > 2:
            raise ValueError('Points must be provided as 1d numpy array or '
                '2d numpy array where each column represents a point.')

        if x.ndim == 1 and not len(x) == self.dim:
            raise ValueError('Dimension mismatch between given point and set.')
    
        if x.ndim == 2 and not x.shape[0] == self.dim:
            raise ValueError('Dimension mismatch between given points and set.')

        if x.ndim > 1:
            return np.array([self.contains(v) for v in x.T])

        return np.all(x <= self.ub) and np.all(x >= self.lb)

    def sample(self, n=1):
        ''' Sample points in set

        INPUTS:
            n   (Default 1) Number of sample points to generate

        RETURNS:
            p   Sample point(s). Either a 1d numpy ndarray for single point
                sampling or a 2d numpy array where each column represents a 
                point.
        '''
        if not isinstance(n, int) or n < 0:
            return ValueError('Number of samples must be a positive integer')

        if n > 1:
            return np.random.uniform(self.lb, self.ub, (n, self.dim)).T
        else:
            return np.random.uniform(self.lb, self.ub)

class UnitHyperrectangle(Hyperrectangle):
    def __init__(self, d: int):
        if not isinstance(d, int):
            raise TypeError('Dimension must be a positive integer.')
            
        if d < 1:
            raise ValueError('Dimension must be a positive integer.')

        super().__init__(-np.ones(d), np.ones(d))

class EuclideanRn(Hyperrectangle):
    ''' Euclidean Rn Space (Set)

    The Euclidean Rn set is a special type of hyperrectangle where its lower 
    and upper bounds are -inf and inf, respectively.

    INPUTS:
        n   The dimension of the Euclidean space
    '''
    def __init__(self, n: int):
        if not isinstance(n, int) or n < 1:
            raise ValueError('Dimension of the EuclideanRn set must be an '
                'integer greater than 0.')

        super().__init__(-np.inf * np.ones(n), np.inf * np.ones(n))