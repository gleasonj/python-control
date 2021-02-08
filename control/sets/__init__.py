# __init__.py - initialization for control systems sets toolbox
#
# Author: Joseph D. Gleason
# Date: 2021-02-04
#
# This file contains the initialization information from the control sets 
# package.
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

"""
The Python Control Sets provides common convex set descriptions and basic
tools for for performing set operations and their interaction with control
systems.
"""

from .special import Empty, Universe

from .supporsets import SupportSet, \
                        Ball2NormSupportSet, \
                        Ball1NormSupportSet, \
                        BallpNormSupportSet, \
                        BallInfNormSupportSet

from .supvec_supfcn import supvec, supfcn

from .convertset import convertset

from .support_approximations import underapproximate, overapproximate

from .polytopes import VPolytope, HPolytope, Hyperrectangle

from .samplers import UniformDBallSampler, \
                      UniformDSphereSampler, \
                      UniformRejectionSampler, \
                      ConvexWalkSampler
                      