"""
Permutation tests for first level and second level analysis.
"""
# Author: Martin Perez-Guevara, <mperezguevara@gmail.com>, jan. 2016
import warnings
import numpy as np
from sklearn.utils import check_random_state
import sklearn.externals.joblib as joblib
import time
import sys
from scipy.ndimage import label

from .first_level_model import run_glm
from .contrasts import compute_contrast
from .thresholding import infer_threshold


def _maxstat_thresholding(stats, masker, threshold=0.001,
                          height_control='fpr'):
    """Compute max cluster size and max cluster mass."""
    z_th = infer_threshold(stats, threshold, height_control)

    # Embed stats back to 3D grid
    stat_map = masker.inverse_transform(stats).get_data()

    # Extract connected components above threshold
    label_map, n_labels = label(stat_map > z_th)
    labels = label_map[masker.mask_img_.get_data() > 0]

    # Get max size and max mass for extracted components
    clusters_size = np.empty(n_labels)
    clusters_mass = np.empty(n_labels)
    for label_ in range(1, n_labels + 1):
        clusters_size[label_ - 1] = np.sum(labels == label_)
        clusters_mass[label_ - 1] = np.sum(stats[labels == label_])
    csmax = np.max(clusters_size)
    cmmax = np.max(clusters_mass)

    return csmax, cmmax


def _original_stat(Y, design_matrix, con_val, stat_type, n_jobs=1):
    """Return original statistic."""
    labels, results = run_glm(Y, design_matrix.as_matrix(),
                              n_jobs=n_jobs, noise_model='ols')
    contrast = compute_contrast(labels, results, con_val, stat_type)
    original_stat = contrast.stat()
    del labels
    del results
    return original_stat


def _sign_flip_glm(Y, design_matrix, con_val, masker, original_stat,
                   stat_type=None, threshold=0.001, height_control='fpr',
                   two_sided_test=True, n_perm_chunk=10000, n_perm=10000,
                   random_state=None, thread_id=1, verbose=0, n_jobs=1):
    """sign flip permutations on data for OLS on a data chunk.
    To be used in a parallel computing context.

    Parameters
    ----------
    design_matrix_objs: list of [time_frames, paradigm, kwargs dict].
        frame_times is an array of shape (n_frames,) representing the timing
        of the scans in seconds. Paradigm is a DataFrame instance with the
        description of the experimental paradigm. kwargs are any other
        arguments that could be passed to the function make_design_matrix
        defined in nistats.design_matrix.

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

    cs_max_parts: array-like, shape=(n_perm_chunk, ):
        Distribution of the (max) cluster size. Limited to this permutation
        chunk.

    cm_max_parts: array-like, shape=(n_perm_chunk, ):
        Distribution of the (max) cluster mass. Limited to this permutation
        chunk.

    References
    ----------
    [1] Nichols, T. E., & Holmes, A. P. (2002). Nonparametric permutation
    tests for functional neuroimaging: a primer with examples. Human brain
    mapping, 15(1), 1-25..
    """
    # initialize the seed of the random generator
    rng = check_random_state(random_state)

    # Initialize result arrays for max stat, max cluster size and max
    # cluster mass
    unc_rank_parts = np.zeros(len(original_stat))
    smax_parts = np.empty((n_perm_chunk))
    cs_max_parts = np.empty((n_perm_chunk))
    cm_max_parts = np.empty((n_perm_chunk))

    # Avoid recomputing sign flips
    n_imgs = Y.shape[0]
    imgs = np.empty((n_imgs * 2, Y.shape[1]))
    for i in range(n_imgs):
        imgs[i * 2, :] = Y[i, :]
        imgs[i * 2 + 1, :] = -1 * Y[i, :]

    # Generate permutation set
    permutation_set = rng.bytes(n_imgs * n_perm_chunk)
    range_idx = np.array(range(n_imgs)) * 2

    # Compute permutations
    t0 = time.time()
    for perm in range(n_perm_chunk):
        perm_val = permutation_set[perm * n_imgs:(perm + 1) * n_imgs]
        perm_idx = np.multiply(perm_val, range_idx)
        labels, results = run_glm(imgs[perm_idx], design_matrix.as_matrix(),
                                  n_jobs=n_jobs, noise_model='ols')
        contrast = compute_contrast(labels, results, con_val, stat_type)
        permuted_stat = contrast.stat()
        del labels
        del results

        # For uncorrected p-values
        if two_sided_test:
            unc_rank_parts += (np.fabs(permuted_stat) < original_stat)
        else:
            unc_rank_parts += (permuted_stat < original_stat)

        # Get max statistic
        smax_parts[perm] = np.max(permuted_stat)
        # Get cluster stats
        csmax, cmmax = _maxstat_thresholding(permuted_stat, masker,
                                             threshold, height_control)
        cs_max_parts[perm] = csmax
        cm_max_parts[perm] = cmmax

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
                    % (thread_id, i, n_perm_chunk, con_val, percent, remaining,
                       crlf))

    return unc_rank_parts, smax_parts, cs_max_parts, cm_max_parts


def second_level_permutation(Y, design_matrix, con_val, masker, stat_type=None,
                             threshold=0.001, height_control='fpr',
                             two_sided_test=True, n_perm=10000,
                             random_state=None, verbose=1, n_jobs=1):
    """Estimate uncorrected and FWE corrected p-values for voxel activation and
    cluster size.

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
    unc_pvals: array-like, shape=(n_voxels)
        Uncorrected p values based on a voxel wise statistic of permutations.

    cor_pvals : array-like, shape=(n_voxels)
        FWE corrected p values based on max statistic of permutations.

    cs_max_parts: array-like, shape=(n_perm):
        Distribution of the (max) cluster size.

    cm_max_parts: array-like, shape=(n_perm):
        Distribution of the (max) cluster mass.
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

    # Compute original stat and clusters only once
    original_stat = _original_stat(Y, design_matrix, con_val, stat_type,
                                   n_jobs)

    # initialize the seed of the random generator
    rng = check_random_state(random_state)

    per = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
        joblib.delayed(_sign_flip_glm)(
            Y, design_matrix, con_val, masker, original_stat,
            stat_type=stat_type, threshold=threshold,
            height_control=height_control, two_sided_test=two_sided_test,
            n_perm_chunk=n_perm_chunk,
            random_state=rng.random_integers(np.iinfo(np.int32).max - 1),
            n_perm=n_perm, thread_id=thread_id, verbose=verbose)
        for thread_id, n_perm_chunk in enumerate(n_perm_chunks))

    unc_rank_parts, smax_parts, cs_max_parts, cm_max_parts = zip(*per)

    # Get uncorrected p-values
    unc_rank = np.zeros(len(original_stat))
    for unc_rank_part in unc_rank_parts:
        unc_rank += unc_rank_part
    unc_pvals = (n_perm + 1 - unc_rank) / float(1 + n_perm)

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

    # Get max cluster size distribution
    cs_max_dist = np.concatenate(cs_max_parts)
    # Get max cluster mass distribution
    cm_max_dist = np.concatenate(cm_max_parts)

    return unc_pvals, cor_pvals, cs_max_dist, cm_max_dist
