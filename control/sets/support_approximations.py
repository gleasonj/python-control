# support_approximation.py - Control Sets Support Set approximations
#
# Author: Joseph D. Gleason
# Date: 2021-02-04
#
# These are basic functions for creating polytopic over and underapproximations
# of support sets
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

from .supporsets import SupportSet

from .supvec_supfcn import supvec, supfcn

from .samplers import UniformDSphereSampler

from .polytopes import HPolytope, VPolytope

def overapproximate(S, L):
    ''' Overapproximate a given SupportSet using support function values

    We use support functions to determine an HPolytope overapproxatiom of a
    support set using a given set of directions, or through a given number
    of random directions on the unit d-sphere.

    INPUTS:
        S   SupportSet
        L   Either an numpy ndarray giving a set of directions to sample for
            the overapproximation, where each column represents a direction,
            or an integer specifying the number of directions to use from
            random generation on the unit d-sphere.

    OUTPUTS:
        P   HPolytope overapproximation of the set S
    '''
    if not isinstance(S, SupportSet):
        raise TypeError('S needs to be a support set')

    if isinstance(L, int) and L > 0:
        L = UniformDSphereSampler(S.dim).sample(L)
    elif isinstance(L, int) and L <= 0:
        raise ValueError('L must be a positive integer or a set of direction '
            'vectors for sampling the support set.')
    
    if not isinstance(L, np.ndarray):
        raise TypeError('L must be a positive integer or a set of direction '
            'vectors for sampling the support set.')

    return HPolytope(L.T, supfcn(S, L))

def underapproximate(S, L):
    ''' Under a given SupportSet using support vectors

    We use support vectors to determine a VPolytope underapproximation of a
    support set using a given set of directions, or through a given number
    of random directions on the unit d-sphere.

    INPUTS:
        S   SupportSet
        L   Either an numpy ndarray giving a set of directions to sample for
            the underapproximation, where each column represents a direction,
            or an integer specifying the number of directions to use from
            random generation on the unit d-sphere.

    OUTPUTS:
        P   VPolytope underapproximation of the set S
    '''
    if not isinstance(S, SupportSet):
        raise TypeError('S needs to be a support set')

    if isinstance(L, int) and L > 0:
        L = UniformDSphereSampler(S.dim).sample(L)
    elif isinstance(L, int) and L <= 0:
        raise ValueError('L must be a positive integer or a set of direction '
            'vectors for sampling the support set.')
    
    if not isinstance(L, np.ndarray):
        raise TypeError('L must be a positive integer or a set of direction '
            'vectors for sampling the support set.')

    return VPolytope(supvec(S, L))