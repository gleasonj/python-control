import numpy as np

from .special import Empty, Universe

from scipy.optimize import linprog

class Hyperrectangle():
    __array_ufunc__ = None
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
        else:
            return NotImplemented

    def __mul__(self, b):
        if isinstance(b, Hyperrectangle):
            return Hyperrectangle(np.concatenate((self.lb, b.lb)),
                np.concatenate((self.ub, b.ub)))
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

    def contains(self, x: np.ndarray):
        ''' Check is point(s) are contained in the set

        INPUTS:
            x   Point(s) to determine containment. This is either a 1d numpy 
                array, or a 2d numpy array where each column represents a single
                point.

        RETURNS:
            y   Boolean or boolean vector
        '''
        if not isinstance(x, np.ndarray) or len(x.shape) > 2:
            raise ValueError('Points must be provided as 1d numpy array or '
                '2d numpy array where each column represents a point.')

        if len(x.shape) == 1 and not len(x) == self.dim:
            raise ValueError('Dimension mismatch between given point and set.')
    
        if len(x.shape) == 2 and not x.shape[0] == self.dim:
            raise ValueError('Dimension mismatch between given points and set.')

        if len(x.shape) > 1:
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
    