def check_valid_contains_points(x):
    if not isinstance(x, np.ndarray) or len(x.shape) > 2:
            raise ValueError('Points must be provided as 1d numpy array or '
                '2d numpy array where each column represents a point.')

