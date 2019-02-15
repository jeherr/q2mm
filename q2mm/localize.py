#!/usr/bin/env python
"""
Takes the hessian matrix and its eigenvals/eigenvectors and finds a
transformation to turn them into a set of localized modes.

Useful when modes distant from the reaction center have large errors from the
ground-state FF but are mostly irrelevant to the parameters being optimized
by Q2MM.

Please see J. Chem. Phys. 130, 084106 (2009) https://doi.org/10.1063/1.3077690
for theoretical background. This code was adapted from MoViPac v1.0.1 available
at http://www.reiher.ethz.ch/software/movipac.html (accessed 02/12/2019)
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np

def localize_normal_modes(hess, log, thresh=1e-6, thresh2=1e-4):
    # Get the eigenvalues and eigenvectors first
    structure = log.structures[0]
    evals, evecs = np.linalg.eigh(hess)
    # evals = np.dot(evecs.T, np.dot(hess, evecs))
    # print(evals)
    # exit(0)
    # Next remove the first 7 vectors and values which
    # are the rotations, translations, and the rxn coordinate
    evals, evecs = evals[7:], np.transpose(evecs)[7:]
    num_modes = len(evals)
    num_atoms = len(hess) // 3
    # Now get the starting transformation (identity) matrix to perform jacobi sweeps
    transform_mat = np.identity(num_modes)
    # evals = np.diag(np.dot(evecs.T, np.dot(hess, evecs)))
    err = 1e10
    err2 = 1e10
    isweep = 0
    # Perform the Jacobi sweeps to localize the modes
    while ((err > thresh) or (err2 > thresh2)) and (isweep < 50):
        isweep += 1

        err2 = 0.0
        old_p = calc_p(evecs, num_modes, num_atoms)
        for i in range(num_modes):
            for j in range(i + 1, num_modes):
                evecs, alpha = rotate(evecs, i, j)

                transform_mat = np.dot(rotation_matrix(i, j, alpha, num_modes), transform_mat)

                err2 += abs(alpha)

        p = calc_p(evecs, num_modes, num_atoms)
        err = p - old_p

        print(
            " Normal mode localization: Cycle %3i    p: %8.3f   change: %10.7f  %10.5f " % \
            (isweep, p, err, err2))
    evals = np.dot(evecs, np.dot(hess, evecs.T))
    # write_gausslog(evecs, evals, structure, num_modes, num_atoms, log, filename='g.log')
    log.evecs = evecs
    return evals

def calc_p(modes, num_modes, num_atoms):
    squared_modes = np.square(modes.reshape((num_modes, num_atoms, 3)))
    p = np.sum(np.square(np.sum(squared_modes, axis=-1)))
    return p

def rotate(modes, i, j):
    imode, jmode = modes[i].reshape((-1, 3)), modes[j].reshape((-1, 3))
    a, b = calc_ab(imode, jmode)

    alpha = np.arccos(-a / np.sqrt(a * a + b * b))

    if np.sin(alpha) - (b / np.sqrt(a * a + b * b)) > 1e-6:
        alpha = -alpha + 2.0 * np.pi
    if alpha < -np.pi:
        alpha = alpha + 2.0 * np.pi
    if alpha >= np.pi:
        alpha = alpha - 2.0 * np.pi
    alpha = alpha / 4.0

    rotated_modes = modes.copy()
    rotated_modes[i, :] = np.cos(alpha) * modes[i, :] + np.sin(alpha) * modes[j, :]
    rotated_modes[j, :] = - np.sin(alpha) * modes[i, :] + np.cos(alpha) * modes[j, :]

    return rotated_modes, alpha

def calc_ab(imode, jmode):
    cij = np.sum(imode * jmode, axis=-1)
    ci = np.sum(np.square(imode), axis=-1)
    cj = np.sum(np.square(jmode), axis=-1)
    a = np.sum(np.square(cij) - 0.25 * np.square(ci - cj))
    b = np.sum(cij * (ci - cj))
    return a, b

def rotation_matrix(i, j, alpha, num_modes):
    rmat = np.identity(num_modes)
    rmat[i, i] = np.cos(alpha)
    rmat[i, j] = np.sin(alpha)
    rmat[j, i] = -np.sin(alpha)
    rmat[j, j] = np.cos(alpha)
    return rmat

def get_coupling_matrix(hessian=False) :
    # coupling matrix is defined as in the paper:
    # eigenvectors are rows of U / columns of U^t = transmat
    # eigenvectors give normal mode in basis of localized modes
    if not hessian :
        diag = np.diag(startmodes.freqs)
    else :
        freqs = self.startmodes.freqs.copy() * 1e2 * Constants.cvel_ms
        # now freq is nu[1/s]
        omega = 2.0 * math.pi * freqs
        # now omega is omega[1/s]
        omega = omega**2
        omega = omega/(Constants.Hartree_in_Joule/(Constants.amu_in_kg*Constants.Bohr_in_Meter**2))
        diag =  numpy.diag(omega)

def write_g98out(modes, evals, structure, num_modes, num_atoms, filename='g98.out'):
    if num_modes % 3 == 0:
        temp_modes = modes
    else:
        temp_modes = np.zeros((3 * (num_modes // 3) + 3, 3 * num_atoms))
        temp_modes[:num_modes, :] = modes[:, :]

    temp_freqs = np.zeros((temp_modes.shape[0],))
    temp_freqs[:num_modes] = np.diag(evals)

    with open(filename, 'w') as f:
        f.write(' Entering Gaussian System \n')
        f.write(' *********************************************\n')
        f.write(' Gaussian 98:\n')
        f.write(' frequency fake output\n')
        f.write(' *********************************************\n')
        f.write('                         Standard orientation:\n')
        f.write(' --------------------------------------------------------------------\n')
        f.write('  Center     Atomic     Atomic              Coordinates (Angstroms)  \n')
        f.write('  Number     Number      Type              X           Y           Z \n')
        f.write(' --------------------------------------------------------------------\n')

        atnums = [structure.atoms[i].atomic_num for i in range(num_atoms)]
        coords = structure.coords

        for iatom in range(num_atoms):
            f.write(' %4i       %4i             0     %11.6f %11.6f %11.6f \n' %
                    (iatom + 1, atnums[iatom], coords[iatom][0], coords[iatom][1], coords[iatom][2]))

        f.write(' --------------------------------------------------------------------\n')
        f.write('     1 basis functions        1 primitive gaussians \n')
        f.write('     1 alpha electrons        1 beta electrons\n')
        f.write('\n')
        f.write(' Harmonic frequencies (cm**-1), IR intensities (KM/Mole), \n')
        f.write(' Raman scattering activities (A**4/amu), Raman depolarization ratios, \n')
        f.write(' reduced masses (AMU), force constants (mDyne/A) and normal coordinates: \n')

        for i in range(0, temp_modes.shape[0], 3):
            f.write('                   %4i                   %4i                  %4i \n' %
                    (i + 1, i + 2, i + 3))
            f.write('                     a                      a                      a  \n')
            f.write(' Frequencies -- %10.4f             %10.4f             %10.4f \n' %
                    (temp_freqs[i], temp_freqs[i + 1], temp_freqs[i + 2]))
            f.write(' Red. masses -- %10.4f             %10.4f             %10.4f \n' % (0.0, 0.0, 0.0))
            f.write(' Frc consts  -- %10.4f             %10.4f             %10.4f \n' % (0.0, 0.0, 0.0))
            f.write(' IR Inten    -- %10.4f             %10.4f             %10.4f \n' % (0.0, 0.0, 0.0))
            f.write(' Raman Activ --     0.0000                 0.0000                 0.0000 \n')
            f.write(' Depolar     --     0.0000                 0.0000                 0.0000 \n')
            f.write(' Atom AN      X      Y      Z        X      Y      Z        X      Y      Z \n')

            for iatom in range(num_atoms):
                atnum = atnums[iatom]
                f.write('%4i %3i   %6.2f %6.2f %6.2f   %6.2f %6.2f %6.2f   %6.2f %6.2f %6.2f \n' %
                        (iatom + 1, atnum,
                         temp_modes[i, 3 * iatom], temp_modes[i, 3 * iatom + 1], temp_modes[i, 3 * iatom + 2],
                         temp_modes[i + 1, 3 * iatom], temp_modes[i + 1, 3 * iatom + 1],
                         temp_modes[i + 1, 3 * iatom + 2],
                         temp_modes[i + 2, 3 * iatom], temp_modes[i + 2, 3 * iatom + 1],
                         temp_modes[i + 2, 3 * iatom + 2],))

def write_gausslog(modes, evals, structure, num_modes, num_atoms, log, filename='g.log'):
    if num_modes % 3 == 0:
        temp_modes = modes
    else:
        temp_modes = np.zeros((3 * (num_modes // 3) + 3, 3 * num_atoms))
        temp_modes[:num_modes, :] = modes[:, :]

    temp_freqs = np.zeros((temp_modes.shape[0],))
    temp_freqs[:num_modes] = np.diag(evals)

    atnums = [structure.atoms[i].atomic_num for i in range(num_atoms)]

    with open(log.path, 'r') as f:
        lines = f.readlines()

    with open(filename, 'w') as f:
        i = 0
        for j, line in enumerate(lines):
            if 'Frequencies' in line:
                f.write(' Frequencies -- %10.4f             %10.4f             %10.4f \n' %
                        (temp_freqs[i], temp_freqs[i + 1], temp_freqs[i + 2]))
            elif 'Red. masses' in line:
                f.write(' Red. masses -- %10.4f             %10.4f             %10.4f \n' % (0.0, 0.0, 0.0))
            elif 'Frc consts' in line:
                f.write(' Frc consts  -- %10.4f             %10.4f             %10.4f \n' % (0.0, 0.0, 0.0))
            elif 'IR Inten' in line:
                f.write(' IR Inten    -- %10.4f             %10.4f             %10.4f \n' % (0.0, 0.0, 0.0))
            elif 'Atom  AN      X      Y      Z' in line:
                f.write(line)
                for iatom in range(num_atoms):
                    atnum = atnums[iatom]
                    f.write('%4i %3i   %6.2f %6.2f %6.2f   %6.2f %6.2f %6.2f   %6.2f %6.2f %6.2f \n' %
                            (iatom + 1, atnum,
                             temp_modes[i, 3 * iatom], temp_modes[i, 3 * iatom + 1], temp_modes[i, 3 * iatom + 2],
                             temp_modes[i + 1, 3 * iatom], temp_modes[i + 1, 3 * iatom + 1],
                             temp_modes[i + 1, 3 * iatom + 2],
                             temp_modes[i + 2, 3 * iatom], temp_modes[i + 2, 3 * iatom + 1],
                             temp_modes[i + 2, 3 * iatom + 2],))
                i += 3
                next_line = j + num_atoms
            else:
                try:
                    if j < next_line:
                        continue
                    else:
                        f.write(line)
                except:
                    f.write(line)
                    pass