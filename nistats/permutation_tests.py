"""
Permutation tests for first level and second level analysis.
"""
# Author: Martin Perez-Guevara, <mperezguevara@gmail.com>, jan. 2016
import warnings
import numpy as np
from sklearn.utils import check_random_state
import sklearn.externals.joblib as joblib
from sklearn.base import clone
from nistats.design_matrix import make_design_matrix
import time
import sys


def _meet_contrast_conditions(con_val):
    """Check if permutation test is allowed for given contrast values."""
    # No F tests
    if type(con_val[0]) is list:
        return False
    # Not with less or more than 2 conditions
    if len(np.where(con_val != 0.0)[0]) != 2:
        return False
    return True


def _permutations_glm(original_stat, glm_ref, imgs, design_matrix_objs,
                      con_val, con_id, con_labels, n_perm_chunk=10000,
                      two_sided_test=True,
                      random_state=None,
                      n_perm=10000,
                      thread_id=1,
                      verbose=0):
    """Massively univariate group analysis with permuted OLS on a data chunk.
    To be used in a parallel computing context.
    Parameters
    ----------
    design_matrix_objs: list of [time_frames, paradigm, kwargs dict].
        frame_times is an array of shape (n_frames,) representing the timing
        of the scans in seconds. Paradigm is a DataFrame instance with the
        description of the experimental paradigm. kwargs are any other
        arguments that could be passed to the function make_design_matrix
        defined in nistats.design_matrix.

    original_stat : array-like, shape=(n_voxels)
        t-scores obtained for the original (non-permuted) data.

    n_perm_chunk : int,
        Number of permutations to be performed.

    two_sided_test : boolean,
        If True, performs an unsigned t-test. Both positive and negative
        effects are considered; the null hypothesis is that the effect is zero.
        If False, only positive effects are considered as relevant. The null
        hypothesis is that the effect is zero or negative.

    random_state : int or None,
        Seed for random number generator, to have the same permutations
        in each computing units.

    Returns
    -------
    unc_ranks_parts: array-like, shape=(n_voxels)
        Accumulated count of permuted stats over original stats,
        to compute uncorrected p-values. (limited to this permutation chunk).

    smax_parts : array-like, shape=(n_perm_chunk, )
        Distribution of the (max) t-statistic under the null hypothesis
        (limited to this permutation chunk).

    References
    ----------
    [1] Nichols, T. E., & Holmes, A. P. (2002). Nonparametric permutation
    tests for functional neuroimaging: a primer with examples. Human brain
    mapping, 15(1), 1-25..
    """
    # initialize the seed of the random generator
    rng = check_random_state(random_state)

    # run the permutations
    smax_parts = np.empty((n_perm_chunk))
    unc_rank_parts = np.zeros(len(original_stat))
    # Infer conditions to permute
    conditions = np.where(con_val != 0.0)
    t0 = time.time()
    for i in range(n_perm_chunk):
        # Shuffle labels of the conditions in the design matrix
        design_matrices = []

        for frame_times, paradigm, kwargs in design_matrix_objs:
            perm_paradigm = paradigm.copy()
            con_idx = np.where(con_val != 0.0)[0]
            conditions = [con_labels[x] for x in con_idx]
            selection = np.array([False for x in range(len(paradigm))])
            for condition in conditions:
                selection = selection | (perm_paradigm['name'] == condition)
            sel_labels = perm_paradigm[selection]
            labels_idx = sel_labels.index.tolist()
            perm_labels = rng.permutation(sel_labels['name'].tolist())
            for lidx, perm_label in zip(labels_idx, perm_labels):
                perm_paradigm.loc[lidx, 'name'] = perm_label

            design_matrices.append(make_design_matrix(frame_times,
                                                      perm_paradigm,
                                                      **kwargs))

        # Get permuted stat
        glm_obj = clone(glm_ref)
        permuted_stat, = glm_obj.fit(imgs, design_matrices) \
                                .transform(con_val, contrast_name=con_id,
                                           output_z=False, output_stat=True)
        permuted_stat = glm_obj.masker_.transform(permuted_stat)[0]

        # For uncorrected p-values
        if two_sided_test:
            unc_rank_parts += (np.fabs(permuted_stat) < original_stat)
        else:
            unc_rank_parts += (permuted_stat < original_stat)

        # For corrected p-values
        smax_parts[i] = np.max(permuted_stat)

        if verbose > 0:
            # We print every 10 permutations or more often
            step = 11 - min(verbose, 10)
            if (i % step == 0):
                # If there is only one job, progress information is fixed
                if n_perm == n_perm_chunk:
                    crlf = "\r"
                else:
                    crlf = "\n"
                percent = float(i) / n_perm_chunk
                percent = round(percent * 100, 2)
                dt = time.time() - t0
                # We use a max to avoid a division by zero
                remaining = (100. - percent) / max(0.01, percent) * dt
                sys.stderr.write(
                    "Job #%d, processed %d/%d permutations for contrast %s "
                    "(%0.2f%%, %i seconds remaining)%s"
                    % (thread_id, i, n_perm_chunk, con_id, percent, remaining,
                       crlf))

    return unc_rank_parts, smax_parts


def first_level_permutation_test(contrasts, glm_ref, imgs,
                                 design_matrix_objs, n_perm=50,
                                 two_sided_test=True, cluster_threshold=None,
                                 random_state=None, verbose=1, n_jobs=1):
    """Estimate uncorrected and FWE corrected p-values for voxel activation and
    cluster size.

    Caches voxel t distributions, whole brain max t distribution and whole
    brain max cluster size distribution if desired to quickly apply arbitrary
    thresholding to contrasts.

    Parameters
    ----------
    contrasts: dict with string as key and list of float as value,
        The key corresponds to the contrast name and the list size must
        corresponds to the number of columns in the design matrices.
        Currently only contrasts between two conditions can be computed.
        So every contrast value must be of the form [...,+-1.,...,-+1.,...].
        Contrasts considering more than two conditions will be ignored.

    glm_ref: FirstLevelGLM instance,
        To provide any GLM computation specification. This object will be
        taken as a reference, new copies will the same parameters will be
        created to be fitted on imgs and design_matrices.

    imgs: Niimg-like object or list of Niimg-like objects,
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        Data on which the GLM will be fitted. If this is a list,
        the affine is considered the same for all.

    design_matrix_objs: list of [time_frames, paradigm, kwargs dict].
        frame_times is an array of shape (n_frames,) representing the timing
        of the scans in seconds. Paradigm is a DataFrame instance with the
        description of the experimental paradigm. kwargs are any other
        arguments that could be passed to the function make_design_matrix
        defined in nistats.design_matrix.

    n_perm: int, optional
        Number of permutations. Greater than 0. Defaults to 10000.

    two_sided_test : boolean,
        If True, performs an unsigned t-test. Both positive and negative
        effects are considered; the null hypothesis is that the effect is zero.
        If False, only positive effects are considered as relevant. The null
        hypothesis is that the effect is zero or negative.

    cluster_threshold: None or float, optional
        Threshold for computation of cluster size statistics. If None will
        not compute cluster size statistics.

    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.

    verbose : integer, optional
        Indicate the level of verbosity. By default, nothing is printed.

    Returns
    -------
    contrast_results: dict with string as key and tuple of two Nifti1File
        objects as value. The key is the contrast name and the first
        nifti file contains the uncorrected p values derived from the
        permutations at the voxel level, while the second nifti file contains
        the corrected p values derived from the permutations whole brain max
        statistic.
    """
    # check n_jobs (number of CPUs)
    if n_jobs == 0:  # invalid according to joblib's conventions
        raise ValueError("'n_jobs == 0' is not a valid choice. "
                         "Please provide a positive number of CPUs, or -1 "
                         "for all CPUs, or a negative number (-i) for "
                         "'all but (i-1)' CPUs (joblib conventions).")
    elif n_jobs < 0:
        n_jobs = max(1, joblib.cpu_count() - int(n_jobs) + 1)
    else:
        n_jobs = min(n_jobs, joblib.cpu_count())

    # Check n_perm
    if n_perm <= 0:
        raise ValueError("'n_perm <= 0' is not a valid choice.")

    # Process one contrast at a time
    contrast_results = {}
    for (con_id, con_val) in contrasts.iteritems():
        # Check permutation can be performed on contrasts
        if not _meet_contrast_conditions(con_val):
            warnings.warn('Contrast %s do not meet conditions.'
                          'Computation ignored.' % (con_id,))
            continue

        # Get original stat
        design_matrices = []
        for time_frames, paradigm, kwargs in design_matrix_objs:
            design_matrices.append(make_design_matrix(time_frames, paradigm,
                                                      **kwargs))
        glm_obj = clone(glm_ref)
        original_stat, = glm_obj.fit(imgs, design_matrices) \
                                .transform(con_val, contrast_name=con_id,
                                           output_z=False, output_stat=True)
        original_stat = glm_obj.masker_.transform(original_stat)[0]
        if two_sided_test:
            original_stat = np.fabs(original_stat)
        # Check possible permutations

        # Distribute permutations across jobs
        if n_perm > n_jobs:
            n_perm_chunks = np.asarray([n_perm / n_jobs] * n_jobs,
                                       dtype=int)
            n_perm_chunks[-1] += n_perm % n_jobs
        elif n_perm > 0:
            warnings.warn('The specified number of permutations is %d and '
                          'the number of jobs to be performed in parallel was '
                          'set to %s. This is incompatible so only %d jobs will '
                          'be running. You may want to perform more permutations '
                          'in order to take the most of the available computing '
                          'resources.' % (n_perm, n_jobs, n_perm))
            n_perm_chunks = np.ones(n_perm, dtype=int)

        # initialize the seed of the random generator
        rng = check_random_state(random_state)

        per = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
              joblib.delayed(_permutations_glm)(
                  original_stat, glm_ref, imgs, design_matrix_objs,
                  con_val, con_id, design_matrices[0].columns.tolist(),
                  n_perm_chunk=n_perm_chunk,
                  two_sided_test=two_sided_test,
                  random_state=rng.random_integers(np.iinfo(np.int32).max - 1),
                  n_perm=n_perm, thread_id=thread_id, verbose=verbose)
              for thread_id, n_perm_chunk in enumerate(n_perm_chunks))

        unc_ranks_parts, smax_parts = zip(*per)
        # Get uncorrected p-values
        unc_ranks = np.zeros(len(original_stat))
        for unc_ranks_part in unc_ranks_parts:
            unc_ranks += unc_ranks_part
        unc_pvals = (n_perm + 1 - unc_ranks) / float(1 + n_perm)
        # Get corrected p-values
        smax = np.concatenate(smax_parts)
        cor_ranks = np.zeros(len(original_stat))
        if two_sided_test:
            for s in np.fabs(smax):
                cor_ranks += (s < original_stat)
        else:
            for s in smax:
                cor_ranks += (s < original_stat)

        cor_pvals = (n_perm + 1 - cor_ranks) / float(1 + n_perm)

        unc_pvals = glm_obj.masker_.inverse_transform(unc_pvals)
        cor_pvals = glm_obj.masker_.inverse_transform(cor_pvals)

        contrast_results[con_id] = (unc_pvals,
                                    cor_pvals)

    return contrast_results
