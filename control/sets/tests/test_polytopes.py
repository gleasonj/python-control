import unittest

import control.sets as cset

import numpy as np

class TestHPolytope(unittest.TestCase):
    def setUp(self):
        self.P = cset.HPolytope(np.vstack((np.eye(2), -np.eye(2))), np.ones(4))

    def test_iteration(self):
        for i, (a, b) in enumerate(self.P):
            self.assertTrue(np.all(a == self.P.A[i]) and b == self.P.b[i])

    def test_multipoint_contains(self):
        X = np.random.uniform(-1, 1, (2, 1000))
        self.assertTrue(np.all(self.P.contains(X)))

    def test_singlepoint_contains(self):
        self.assertTrue(self.P.contains(np.random.uniform(-1, 1, 2)))

    def test_empty_comparison(self):
        P = cset.HPolytope(self.P.A, -np.ones(4))
        self.assertTrue(P == cset.Empty())
        self.assertFalse(self.P == cset.Empty())

    def test_universe_comparison(self):
        self.assertTrue(self.P < cset.Universe())
        self.assertTrue(self.P <= cset.Universe())

if __name__ == '__main__':
    unittest.main()