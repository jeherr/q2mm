
from scipy.optimize import differential_evolution
import copy
import collections
import csv
import glob
import itertools
import logging
import logging.config
import numpy as np
import os
import re
import sys

import calculate
import compare
import constants as co
import datatypes
import opt as opt
import parameters

logger = logging.getLogger(__name__)

class DifferentialEvolution(opt.Optimizer):
    """
    Differential Evolution optimization (those dependent on derivatives of
    the penalty function). See `Optimizer` for repeated documentation.

    All cutoff attributes are a list of two positive floats. For new parameter
    sets, if the radius of unsigned parameter change doesn't lie between these
    floats, then that parameter set is ignored.

    All radii attributes are a list of many positive floats, as many as you'd
    like. This list is used to scale new parameter changes. For each new
    parameter set, it iterates through the list from lowest to highest values.
    If the radius of unsigned parameter change exceeds the given radius, then
    the parameter changes are scaled to match that radius. If the radius of
    unsigned parameter change is less than the given radius, the current
    parameter changes are applied without modification, and the remaining radii
    are not iterated through.

    Attributes
    ----------------
    do_basic : bool
    do_lagrange : bool
    do_levenberg : bool
    do_newton : bool
    do_svd : bool
    basic_cutoffs : list or None
    basic_radii : list or None
                  Default is [0.1, 1., 5., 10.].
    lagrange_factors : list
                       Default is [0.01, 0.1, 1., 10.].
    lagrange_cutoffs : list or None
    lagrange_radii : list or None
                     Default is [0.1, 1., 5., 10.].
    levenberg_factors : list
                        Default is [0.01, 0.1, 1., 10.].
    levenberg_cutoffs : list or None
    levenberg_radii : list or None
                      Default is [0.1, 1., 5., 10.].
    newton_cutoffs : list or None
    newton_radii : list or None
                   Default is [0.1, 1., 5., 10.].
    svd_factors : list or None
                  Default is [0.001, 0.01, 0.1, 1.].
    svd_cutoffs : list of None
                  Default is [0.1, 10.].
    svd_radii : list or None
    """
    def __init__(self,
                 direc=None,
                 ff=None,
                 ff_lines=None,
                 args_ff=None,
                 args_ref=None):
        super(DifferentialEvolution, self).__init__(
            direc, ff, ff_lines, args_ff, args_ref)

    def set_param_bounds(self):
        bounds = []
        for param in self.ff.params:
            if param.ptype == "be":
                bounds.append(0.7, 3.0)
            elif param.ptype == "bf":
                bounds.append(0.0, 20.0)
            elif param.ptype == "q":
                bounds.append(-10.0, 10.0)
            elif param.ptype == "ae":
                bounds.append(50.0, 180.0)
            elif param.ptype == "af":
                bounds.append(0.1, 10.0)
            elif param.ptype == "df":
                bounds.append(-20.0, 20.0)
        return bounds


    def objective_function(self, new_ff_params):
        new_ff = copy.deepcopy(self.ff)
        for idx, new_value in enumerate(new_ff_params):
            new_ff.params[idx].value = new_value
        new_ff.export_ff(lines=self.ff.lines)
        new_ff.data = calculate.main(self.args_ff)
        new_ff.score = compare.calculate_score(self.ref_data, new_ff.data)
        opt.pretty_ff_results(new_ff, level=20)
        return new_ff.score

    @opt.catch_run_errors
    def run(self, ref_data=None, restart=None):
        """
        Runs the gradient optimization.

        Ensure that the attributes in __init__ are set as you desire before
        using this function.

        Returns
        -------
        `datatypes.FF` (or subclass)
            Contains the best parameters.
        """
        # We need reference data if you didn't provide it.
        if ref_data is None:
            self.ref_data = opt.return_ref_data(self.args_ref)
        else:
            self.ref_data = ref_data

        # We need the initial FF data.
        if self.ff.data is None:
            logger.log(20, '~~ GATHERING INITIAL FF DATA ~~'.rjust(79, '~'))
            # Is opt.Optimizer.ff_lines used anymore?
            self.ff.export_ff()
            self.ff.data = calculate.main(self.args_ff)
            # Not 100% sure if this is necessary, but it certainly doesn't hurt.
            compare.correlate_energies(ref_data, self.ff.data)
        if self.ff.score is None:
            # Already zeroed reference and correlated the energies.
            self.ff.score = compare.calculate_score(ref_data, self.ff.data)

        logger.log(20, '~~ DIFFERENTIAL EVOLUTION OPTIMIZATION ~~'.rjust(79, '~'))
        logger.log(20, 'INIT FF SCORE: {}'.format(self.ff.score))
        opt.pretty_ff_results(self.ff, level=20)

        exit(0)
        param_bounds = self.set_param_bounds()
        objective = lambda x: self.objective_function(x)        
        result = differential_evolution(objective, param_bounds, maxiter=self.maxiter,
            popsize=10, disp=True, polish=False)
        print(result.x, result.fun)
        if result.fun < self.ff.score:
            ff = copy.deepcopy(self.ff)
            for idx, new_value in enumerate(result.x):
                new_ff.params[idx].value = new_value
            opt.pretty_ff_results(new_ff, level=20)
            return ff
        else:
            return self.ff
