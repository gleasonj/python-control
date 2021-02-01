import numpy as np

class Empty():
    def contains(self, x: np.ndarray):
        ''' 
        '''
        check_valid_contains_points(x)

        if len(x.shape) == 2:
            return np.zeros(x.shape[1], dtype=bool)
        else:
            return False

class Universe():
    def contains(self, x: np.ndarray):
        check_valid_contains_points(x)
        
        if len(x.shape) == 2:
            return np.ones(x.shape[1], dtype=bool)
        else:
            return True