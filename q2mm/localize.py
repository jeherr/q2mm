#!/usr/bin/env python
"""
Takes the hessian matrix and its eigenvals/eigenvectors
and finds a transformation to turn them into a set of
localized modes.

Useful when modes distant from the reaction center have
large errors from the ground-state FF but are mostly
irrelevant to the parameters being optimized by Q2MM.
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np

def localize_normal_modes(hess, evals, evecs):
    evals, evecs = np.linalg.eigh(hess)
    # print(np.dot(hess, evecs[:,0]) - evals[0] * evecs[:,0])
    evals = np.diag(np.dot(evecs.T, np.dot(hess, evecs)))
    return evals, evecs