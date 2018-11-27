
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
        # The current defaults err on the side of simplicity, so these are
        # likely more ideal for smaller parameter sets. For larger parameter
        # sets, central differentiation will take longer, and so you you will
        # likely want to try more trial FFs per iteration. This would mean
        # adding more max radii (ex. lagrange_radii) or more factors (ex.
        # svd_factors).

        # Whether or not to generate parameters with these methods.
        self.do_lstsq = False
        self.do_lagrange = True
        self.do_levenberg = False
        self.do_newton = True
        self.do_svd = True

        # Particular settings for each method.
        # LEAST SQUARES
        self.lstsq_cutoffs = None
        self.lstsq_radii = [1., 10.]
        # LAGRANGE
        self.lagrange_factors = [0.01, 0.1, 1., 10.]
        self.lagrange_cutoffs = None
        self.lagrange_radii = [0.1, 10.]
        # LEVENBERG-MARQUARDT
        self.levenberg_factors = [0.01, 0.1, 1., 10.]
        self.levenberg_cutoffs = None
        self.levenberg_radii = [0.1, 10.]
        # NEWTON-RAPHSON
        self.newton_cutoffs = None
        self.newton_radii = [1., 10.]
        # SVD
        self.svd_factors = [0.001, 0.01, 0.1, 1., 10.]
        self.svd_cutoffs = [0.1, 10.]
        self.svd_radii = None

    # Don't worry that self.ff isn't included in self.new_ffs.
    # opt.catch_run_errors will know what to do if self.new_ffs
    # is None.
    @property
    def best_ff(self):
        return sorted(self.new_ffs, key=lambda x: x.score)[0]

	def objective_function(ff_params):
		compare.calculate_score(ref_data, self.ff.data)

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
            ref_data = opt.return_ref_data(self.args_ref)

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

        if restart:
            par_file = restart
            logger.log(20, '  -- Restarting gradient from central '
                       'differentiation file {}.'.format(par_file))
        else:
            # We need a file to hold the differentiated parameter data.
            par_files = glob.glob(os.path.join(self.direc, 'par_diff_???.txt'))
            if par_files:
                par_files.sort()
                most_recent_par_file = par_files[-1]
                most_recent_par_file = most_recent_par_file.split('/')[-1]
                most_recent_num = most_recent_par_file[9:12]
                num = int(most_recent_num) + 1
                par_file = 'par_diff_{:03d}.txt'.format(num)
            else:
                par_file = 'par_diff_001.txt'
            logger.log(20, '  -- Generating central differentiation '
                       'file {}.'.format(par_file))
            f = open(os.path.join(self.direc, par_file), 'w')
            csv_writer = csv.writer(f)
            # Row 1 - Labels
            # Row 2 - Weights
            # Row 3 - Reference data values
            # Row 4 - Initial FF data values
            csv_writer.writerow([x.lbl for x in ref_data])
            csv_writer.writerow([x.wht for x in ref_data])
            csv_writer.writerow([x.val for x in ref_data])
            csv_writer.writerow([x.val for x in self.ff.data])
            logger.log(20, '~~ DIFFERENTIATING PARAMETERS ~~'.rjust(79, '~'))
            # Save many FFs, each with their own parameter sets.
            ffs = opt.differentiate_ff(self.ff)
            logger.log(20, '~~ SCORING DIFFERENTIATED PARAMETERS ~~'.rjust(
                79, '~'))
            for ff in ffs:
                ff.export_ff(lines=self.ff.lines)
                logger.log(20, '  -- Calculating {}.'.format(ff))
                data = calculate.main(self.args_ff)
                ff.score = compare.compare_data(ref_data, data)
                opt.pretty_ff_results(ff)
                # Write the data rather than storing it in memory. For large
                # parameter sets, this could consume GBs of memory otherwise!
                csv_writer.writerow([x.val for x in data])
            f.close()

            # Make sure we have derivative information. Used for NR.
            #
            # The derivatives are useful for checking up on the progress of the
            # optimization and for deciding which parameters to use in a
            # subsequent simplex optimization.
            #
            # Still need a way to do this with the resatrt file.
            opt.param_derivs(self.ff, ffs)

        # Report how many trial FFs were generated.
        logger.log(20, '  -- Generated {} trial force field(s).'.format(
                len(self.new_ffs)))
        # If there are any trials, test them.
        if self.new_ffs:
            logger.log(20, '~~ EVALUATING TRIAL FF(S) ~~'.rjust(79, '~'))
            for ff in self.new_ffs:
                data = opt.cal_ff(ff, self.args_ff, parent_ff=self.ff)
                # Shouldn't need to zero anymore.
                ff.score = compare.compare_data(ref_data, data)
                opt.pretty_ff_results(ff)
            self.new_ffs = sorted(
                self.new_ffs, key=lambda x: x.score)
            # Check for improvement.
            if self.new_ffs[0].score < self.ff.score:
                ff = self.new_ffs[0]
                logger.log(
                    20, '~~ GRADIENT FINISHED WITH IMPROVEMENTS ~~'.rjust(
                        79, '~'))
                opt.pretty_ff_results(self.ff, level=20)
                opt.pretty_ff_results(ff, level=20)
                # Copy parameter derivatives from original FF to save time in
                # case we move onto simplex immediately after this.
                copy_derivs(self.ff, ff)
            else:
                ff = self.ff
        else:
            ff = self.ff
        return ff
