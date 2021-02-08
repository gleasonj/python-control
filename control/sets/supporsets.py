# supportsets.py - Control Sets Support Sets
#
# Author: Joseph D. Gleason
# Date: 2021-02-04
#
# These are classes and function for creating support sets and functions to 
# compute the support function and support vector values.
#
# Copyright (c) 2021 by Joseph D. Gleason
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the California Institute of Technology nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
# OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#
# $Id$

import numpy as np

from scipy.optimize import linprog, minimize, NonlinearConstraint

from .hypperrectangle import Hyperrectangle

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
        else:
            return NotImplemented

    def __rmul__(self, S):
        if isinstance(S, float):
            return ScalorMulSupportSet(self, S)
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
        else:
            return NotImplemented

    @property
    def dim(self):
        return NotImplementedError('Any subclass of SupportSet needs to define '
            'the dimension of the set.')

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
            
        # if not M.shape[0] == M.shape[1]:
        #     raise ValueError('Multiplying matrix must be square.')
        
        self._S = S
        self._M = M

    def __call__(self, l):
        return self._M @ self._S(self._M.T @ l)

    @property
    def dim(self):
        return self._M.shape[0]

class ScalorMulSupportSet(SupportSet):
    def __init__(self, S: SupportSet, c: float):
        self._S = S
        self._c = c

    def __call__(self, l):
        return self._c * self._S(l)

    @property
    def dim(self):
        return self._S.dim

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
        return np.concatenate((self._A(l[:self._A.dim]), self._B(l[self._A.dim:])))

    @property
    def dim(self):
        return self._A.dim + self._B.dim

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
        if not (isinstance(p, int) or p == np.inf or p == -np.inf):
            raise ValueError('P-value must be an integer or numpy inf')

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

    def __call__(self, l):
        res = minimize(lambda x: -l.T @ x, np.zeros(self.dim), 
            constraints=NonlinearConstraint(
            lambda x: np.linalg.norm(x - self.center, ord=self.p),
            0, self.radius))

        return res.x

    def contains(self, x):
        if not isinstance(x, np.ndarray) or x.ndim > 2:
            raise TypeError('Point(s) must be a 1d numpy array or 2d numpy '
                'array where each column represents a point.')

        if x.ndim > 1:
            return np.array([self.contains(v) for v in x.T])
        else:
            return np.linalg.norm(self.center - x, self.p) <= self.radius

class Ball1NormSupportSet(BallpNormSupportSet):
    def __init__(self, center: np.ndarray, radius: float):
        super().__init__(1, center, radius)

class Ball2NormSupportSet(BallpNormSupportSet):
    def __init__(self, center: np.ndarray, radius: float):
        super().__init__(2, center, radius)

    def __call__(self, l):
        return self._radius * l

class BallInfNormSupportSet(BallpNormSupportSet):
    def __init__(self, center: np.ndarray, radius: float):
        super().__init__(np.inf, center, radius)
