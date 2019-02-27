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

import constants as co

def localize_normal_modes(hess, log, thresh=1e-6, thresh2=1e-7):
    # Get the eigenvalues and eigenvectors first
    structure = log.structures[0]
    old_evals, old_evecs = np.linalg.eigh(hess)
    # Next remove the first 7 vectors and values which
    # are the rotations, translations, and the rxn coordinate
    evals, evecs = old_evals[7:], np.transpose(old_evecs)[7:]
    num_modes = len(evals)
    num_atoms = len(hess) // 3
    # Now get the starting transformation (identity) matrix to perform jacobi sweeps
    transform_mat = np.identity(num_modes)
    # evals = np.diag(np.dot(evecs.T, np.dot(hess, evecs)))
    err = 1e10
    err2 = 1e10
    combinations = np.math.factorial(num_modes) / (np.math.factorial(num_modes - 2) * np.math.factorial(2))
    isweep = 0
    # Perform the Jacobi sweeps to localize the modes
    while ((err > thresh) or (err2 > thresh2)) and (isweep < 200):
        isweep += 1

        err2 = 0.0
        old_p = calc_p(evecs, num_modes, num_atoms)
        for i in range(num_modes):
            for j in range(i + 1, num_modes):
                evecs, alpha = rotate(evecs, i, j)

                transform_mat = np.dot(rotation_matrix(i, j, alpha, num_modes), transform_mat)

                err2 += abs(alpha)
        err2 /= combinations
        p = calc_p(evecs, num_modes, num_atoms)
        err = p - old_p

        print(
            " Normal mode localization: Cycle %3i    p: %8.3f   change: %14.8f  %14.8f " % \
            (isweep, p, err, err2))
    similarity_matrix = get_cosine_similarity(evecs, np.transpose(old_evecs)[7:])
    check_prelocalization_criteria(similarity_matrix)
    evecs = np.append(np.transpose(old_evecs[:, 0:1]), evecs, axis=0)
    evals = np.dot(evecs, np.dot(hess, evecs.T))
    # write_gausslog(evecs, evals, structure, num_atoms, log)
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

def write_gausslog(modes, evals, structure, num_atoms, log):
    num_modes = modes.shape[0]
    if num_modes % 3 == 0:
        temp_modes = modes
    else:
        temp_modes = np.zeros((3 * (num_modes // 3) + 3, 3 * num_atoms))
        temp_modes[:num_modes, :] = modes[:, :]

    temp_freqs = np.zeros((temp_modes.shape[0],))
    temp_freqs[:num_modes] = np.diag(evals)

    atnums = [structure.atoms[i].atomic_num for i in range(num_atoms)]

    mass_sqrt = np.sqrt([list(co.MASSES.items())[atnums[i] - 1][1] for i in range(num_atoms)])
    temp_modes = temp_modes.reshape((num_modes, num_atoms, 3)) / np.expand_dims(mass_sqrt, axis=-1)
    temp_modes = temp_modes.reshape((num_modes, num_atoms * 3))

    red_masses = np.zeros((3 * (num_modes // 3) + 3),)
    red_masses[:num_modes] = np.sum(np.linalg.norm(modes.reshape((num_modes, num_atoms, 3)), axis=-1) / np.array(atnums), axis=-1)
    with open(log.path, 'r') as f:
        lines = f.readlines()

    filename = log.path[:-4]+'-local'+log.path[-4:]
    with open(filename, 'w') as f:
        i = 0
        for j, line in enumerate(lines):
            if 'Frequencies' in line:
                f.write(' Frequencies -- %10.4f             %10.4f             %10.4f \n' %
                        (temp_freqs[i], temp_freqs[i + 1], temp_freqs[i + 2]))
            elif 'Red. masses' in line:
                f.write(' Red. masses -- %10.4f             %10.4f             %10.4f \n' %
                        (red_masses[i], red_masses[i + 1], red_masses[i + 2]))
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
                next_line = j + num_atoms + 1
            else:
                try:
                    if j < next_line:
                        continue
                    else:
                        f.write(line)
                except:
                    f.write(line)
                    pass

def select_relevant_modes(modes):
    # Should select which modes are most relevant to the
    # parameters being optimized and discard the rest
    return modes

def get_cosine_similarity(modes, original_modes):
    # Returns the cosine similarity matrix between the
    # normal modes and the localized modes by taking all
    # possible dot products.
    similarity_matrix = np.zeros((len(modes), len(modes)))
    for i in range(len(modes)):
        ii_overlap = np.dot(modes[i], original_modes[i]) / (np.linalg.norm(modes[i]) * np.linalg.norm(original_modes[i]))
        similarity_matrix[i,i] = ii_overlap
        for j in range(i + 1, len(modes)):
            ij_overlap = np.dot(modes[i], original_modes[j]) / (np.linalg.norm(modes[i]) * np.linalg.norm(original_modes[j]))
            similarity_matrix[i,j] = ij_overlap
            similarity_matrix[j,i] = ij_overlap
    np.savetxt("cos_sim_modes.txt", similarity_matrix)
    return similarity_matrix

def check_prelocalization_criteria(similarity_matrix):
    max_args = np.argsort(similarity_matrix, axis=1)[::-1]
    matched_idx = []
    arg_order = []
    for i in range(len(similarity_matrix)):
        arg_trial_idx = 0
        while True:
            max_idx = max_args[i,arg_trial_idx]
            if max_idx in matched_idx:
                arg_trial_idx += 1
            else:
                matched_idx.append(max_idx)
                arg_order.append(arg_trial_idx)
                break
    switch_iter = 0
    while True:
        switch_iter += 1
        flag = True
        for i in range(len(similarity_matrix)):
            for j in range(i+1, len(similarity_matrix)):
                i_arg, j_arg = matched_idx[i], matched_idx[j]
                current_overlap = np.square(similarity_matrix[i,i_arg]) + np.square(similarity_matrix[j,j_arg])
                switch_overlap = np.square(similarity_matrix[i,j_arg]) + np.square(similarity_matrix[j,i_arg])
                if switch_overlap > current_overlap:
                    flag = False
                    matched_idx[j], matched_idx[i] = matched_idx[i], matched_idx[j]
        if flag:
            break
    overlaps = [similarity_matrix[i, matched_idx[i]] for i in range(len(similarity_matrix))]
    print(np.sqrt(np.sum(np.square(overlaps)) / len(similarity_matrix)))
    # print(np.sqrt(sum(overlaps) / len(overlap_matrix)))
    # print(min(overlaps))
    # print(max(overlaps))
    # for i in range(len(overlap_matrix)):
    #
    # Create a master set for the idx of all possible modes
    # modes_idx_list = set(range(len(overlap_matrix)))
    # mode_mode_dict = {}



