import numpy as np

from .hypperrectangle import Hyperrectangle

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
        elif isinstance(P, Hyperrectangle):
            return self * _hypperrectangle_to_hpolytope(P)
        else:
            return NotImplemented

    def __rmul__(self, P):
        if isinstance(P, Hyperrectangle):
            return _hypperrectangle_to_hpolytope(P) * self
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
        elif isinstance(P, Hyperrectangle):
            return PolytopeUnion(self, _hypperrectangle_to_hpolytope(P))
        else:
            return NotImplemented

    def __ror__(self, P):
        if isinstance(P, (HPolytope, VPolytope)):
            return PolytopeUnion(P, self)
        elif isinstance(P, Hyperrectangle):
            return PolytopeUnion(_hypperrectangle_to_hpolytope(P), self)
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

    def __mul__(self, P):
        if isinstance(P, HPolytope):
            return PolytopeCartesianProduct(self, P)
        elif isinstance(P, VPolytope):
            return PolytopeCartesianProduct(self, P)
        elif isinstance(P, Hyperrectangle):
            return PolytopeCartesianProduct(self, 
                _hypperrectangle_to_hpolytope(P))
        else:
            return NotImplemented

    def __rmul__(self, P):
        # Note that different from the __mull__ function we don't need to 
        # include what happens for H or VPolytopes. This is because for either
        # of those types, P.__mul__ should have been already been called.
        if isinstance(P, Hyperrectangle):
            return _hypperrectangle_to_hpolytope(P) * self
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
        else:
            return NotImplemented

    def __ror__(self, P):
        if isinstance(P, (HPolytope, VPolytope)):
            return PolytopeUnion(P, *self.polytopes)
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