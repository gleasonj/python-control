import numpy as np

from .special import Empty, Universe

from scipy.optimize import linprog

class Hyperrectangle():
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
        # Need to call the super to resolve the abstract type definitions
        super().__init__()

        assert len(lb.shape) == 1, 'Lower bound must be an array'
        assert len(ub.shape) == 1, 'Upper bound must be an array'
        assert len(lb) == len(ub), 'Lower and upper bounds must have same length'

        self._lb = lb
        self._ub = ub

    def __rmul__(self, b):
        if isinstance(b, Hyperrectangle):
            return b.__mul__(self)
        elif isinstance(b, float):
            return Hyperrectangle(self.lb * b, self.ub * b)
        else:
            return NotImplemented

    def __mul__(self, b):
        if isinstance(b, Hyperrectangle):
            return Hyperrectangle(np.concatenate((self.lb, b.lb)),
                np.concatenate((self.ub, b.ub)))
        if isinstance(b, float):
            return self.__rmul__(b)
        else:
            return NotImplemented

    def __and__(self, S):
        if isinstance(S, Hyperrectangle):
            return Hyperrectangle(np.minimum(self.lb, S.lb),
                np.maximum(self.ub, S.ub))
        else:
            return NotImplemented

    def __add__(self, S):
        if isinstance(S, Hyperrectangle):
            return Hyperrectangle(self.lb + S.lb, self.ub + S.ub)
        if isinstance(S, Empty):
            return self
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

class HyperrectangleUnion():
    __array_ufunc__ = None
    def __init__(self, *args):
        for r in args:
            if not isinstance(r, Hyperrectangle):
                raise TypeError('All elements of HyperrectangleUnion must be '
                    'Hyperrectangles')

        self._rectangles = args

    @property
    def rectangles(self):
        return self._rectangles

    @property
    def dim(self):
        return self.rectangles[0].dim

    def bounding_box(self):
        bbox = Empty()
        for r in self:
            bbox += r

        return bbox

    def __or__(self, S):
        if isinstance(S, Hyperrectangle):
            return HyperrectangleUnion(*self.rectangles, S)
        if isinstance(S, HyperrectangleUnion):
            return HyperrectangleUnion(*self.rectangles, *S.rectangles)
        else:
            return NotImplemented
    
    def __ror__(self, S):
        if isinstance(S, Hyperrectangle):
            return HyperrectangleUnion(S, *self.rectangles)
        elif isinstance(S, HyperrectangleUnion):
            return HyperrectangleUnion(*S.rectangles, *self.rectangles)
        else:
            return NotImplemented

    def __len__(self):
        return len(self.rectangles)

    def __iter__(self):
        self._ridx = 0
        return self

    def __next__(self):
        if self._ridx < len(self):
            R = self.rectangles[self._ridx]
            self._ridx += 1
            return R
        else:
            raise StopIteration