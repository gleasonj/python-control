import numpy as np

class ZeroSet():
    def contains(self, x: np.ndarray):
        ''' 
        '''
        if not isinstance(x, np.ndarray):
            raise TypeError('Point(s) must be a 1d numpy array or 2d numpy '
                'array where each column represents a point.')

        if x.ndim > 2:
            raise TypeError('Point(s) must be a 1d numpy array or 2d numpy '
                'array where each column represents a point.')

        if x.ndim == 2:
            return np.array([self.contains(v) for v in x.T])
        else:
            return np.all(x == 0)

class Empty():
    def contains(self, x: np.ndarray):
        ''' 
        '''
        if not isinstance(x, np.ndarray):
            raise TypeError('Point(s) must be a 1d numpy array or 2d numpy '
                'array where each column represents a point.')

        if x.ndim > 2:
            raise TypeError('Point(s) must be a 1d numpy array or 2d numpy '
                'array where each column represents a point.')

        if x.ndim == 2:
            return np.zeros(x.shape[1], dtype=bool)
        else:
            return False

class Universe():
    def contains(self, x: np.ndarray):
        if not isinstance(x, np.ndarray):
            raise TypeError('Point(s) must be a 1d numpy array or 2d numpy '
                'array where each column represents a point.')

        if x.ndim > 2:
            raise TypeError('Point(s) must be a 1d numpy array or 2d numpy '
                'array where each column represents a point.')
        
        if x.ndim == 2:
            return np.ones(x.shape[1], dtype=bool)
        else:
            return True