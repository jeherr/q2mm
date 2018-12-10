
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
                bounds.append((0.9, 2.5))
            elif param.ptype == "bf":
                bounds.append((0.9, 13.0))
            elif param.ptype == "q":
                bounds.append((-4.5, 3.5))
            elif param.ptype == "ae":
                bounds.append((50.0, 180.0))
            elif param.ptype == "af":
                bounds.append((0.1, 9.0))
            elif param.ptype == "df":
                if param.mm3_col == 1:
                    bounds.append((-3.5, 5.0))
                elif param.mm3_col == 2:
                    bounds.append((-10.0, 16.0))
                elif param.mm3_col == 3:
                    bounds.append((-4.0, 4.0))
        return bounds


    def objective_function(self, new_ff_params):
        new_ff = copy.deepcopy(self.ff)
        for idx, new_value in enumerate(new_ff_params):
            new_ff.params[idx].value = new_value
        new_ff.export_ff(lines=self.ff.lines)
        try:
            new_ff.data = calculate.main(self.args_ff)
            new_ff.score = compare.calculate_score(self.ref_data, new_ff.data)
        except:
            new_ff.score = 9999999999.0
            pass
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

        param_bounds = self.set_param_bounds()
        objective = lambda x: self.objective_function(x)
        result = differential_evolution(objective, param_bounds, maxiter=self.maxiter,
            popsize=4, disp=True, polish=False)
        print(result.x, result.fun)
        if result.fun < self.ff.score:
            ff = copy.deepcopy(self.ff)
            for idx, new_value in enumerate(result.x):
                new_ff.params[idx].value = new_value
            opt.pretty_ff_results(new_ff, level=20)
            return ff
        else:
            return self.ff
