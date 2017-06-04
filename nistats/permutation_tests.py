"""
Permutation tests for first level and second level models.
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
    if n_labels > 0:
        clusters_size = np.empty(n_labels)
        clusters_mass = np.empty(n_labels)
        for label_ in range(1, n_labels + 1):
            clusters_size[label_ - 1] = np.sum(labels == label_)
            clusters_mass[label_ - 1] = np.sum(stats[labels == label_])
        csmax = int(np.max(clusters_size))
        cmmax = np.max(clusters_mass)
    else:
        csmax = 0
        cmmax = 0.0

    return csmax, cmmax


def _get_z_score(Y, design_matrix, con_val, stat_type, n_jobs=1):
    """Return original statistic."""
    labels, results = run_glm(Y, design_matrix.as_matrix(),
                              n_jobs=n_jobs, noise_model='ols')
    contrast = compute_contrast(labels, results, con_val, stat_type)
    original_stat = contrast.z_score()
    del labels
    del results
    return original_stat


def _sign_flip_glm(Y, design_matrix, con_val, masker,
                   stat_type=None, threshold=0.001, height_control='fpr',
                   two_sided_test=True, n_perm=10000, n_perm_chunk=10000,
                   random_state=None, thread_id=1, verbose=0, n_jobs=1):
    """sign flip permutations on data for OLS on a data chunk.
    To be used in a parallel computing context.

    Parameters
    ----------
    Y : array of shape (n_subjects, n_voxels)
        The fMRI data.

    design_matrix : pandas DataFrame (n_subjects, n_regressors)
        The design matrix.

    con_val : array of shape (n_regressors,)
        Contrast specification to compute statistic.

    masker : NiftiMasker object,
        It must be the same masker employed to extract Y from nifti images.
        It will be used to compute clusters in permutations.

    stat_type : {'t', 'F'}, optional
        Type of the contrast

    threshold: float, optional
        cluster forming threshold (either a p-value or z-scale value)

    height_control: string, optional
        false positive control meaning of cluster forming
        threshold: 'fpr'|'fdr'|'bonferroni'|'none'

    two_sided_test : boolean, optional
        If True, performs an unsigned t-test. Both positive and negative
        effects are considered; the null hypothesis is that the effect is zero.
        If False, only positive effects are considered as relevant. The null
        hypothesis is that the effect is zero or negative.

    n_perm : int, optional
        Total number of permutations.

    n_perm_chunk : int, optional
        Number of permutations to be performed in this chunk.

    two_sided_test : boolean, optional
        If True, performs an unsigned t-test. Both positive and negative
        effects are considered; the null hypothesis is that the effect is zero.
        If False, only positive effects are considered as relevant. The null
        hypothesis is that the effect is zero or negative.

    random_state : int or None, optional
        Seed for random number generator, to have the same permutations
        in each computing units.

    Returns
    -------
    smax_parts : array-like, shape=(n_perm_chunk, )
        Distribution of the (max) z-statistic under the null hypothesis
        (limited to this permutation chunk).

    cs_max_parts : array-like, shape=(n_perm_chunk, ):
        Distribution of the (max) cluster size. Limited to this permutation
        chunk.

    cm_max_parts : array-like, shape=(n_perm_chunk, ):
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
    permutation_set = rng.randint(0, 2, n_imgs * n_perm_chunk)

    # Compute permutations
    t0 = time.time()
    for perm in range(n_perm_chunk):
        perm_val = permutation_set[perm * n_imgs:(perm + 1) * n_imgs]
        perm_idx = perm_val + (np.array(range(n_imgs)) * 2)
        permuted_stat = _get_z_score(imgs[perm_idx], design_matrix, con_val,
                                     stat_type, n_jobs=n_jobs)
        if two_sided_test:
            permuted_stat = np.fabs(permuted_stat)

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

    return smax_parts, cs_max_parts, cm_max_parts


def cluster_p_value(stats, masker, stat_dist, cluster_stat,
                    threshold=0.001, height_control='fpr'):
    """Compute the p value of clusters for the chosen cluster statistic.

    Parameters
    ----------
    stats : array of shape (n_voxels)
        The image statistic to extract clusters from.

    masker : NiftiMasker object,
        It must be the same masker employed to extract stats from nifti images.
        It will be used to compute clusters in permutations.

    stat_dist : array of shape (n_perm)
        Statistical distribution corresponding to the selected cluster_stat.
        Obtained from n_perm permutations.

    cluster_stat : string,
        Type of cluster statistic to consider. Possible values are 'size' and
        'mass'.

    threshold: float, optional
        cluster forming threshold (either a p-value or z-scale value)

    height_control: string, optional
        false positive control meaning of cluster forming
        threshold: 'fpr'|'fdr'|'bonferroni'|'none'
    """
    if cluster_stat not in ['size', 'mass']:
        raise ValueError('cluster_stat must be "size" or "mass"')

    n_dist = len(stat_dist)
    z_th = infer_threshold(stats, threshold, height_control)

    # Embed stats back to 3D grid
    stat_map = masker.inverse_transform(stats).get_data()

    # Extract connected components above threshold
    label_map, n_labels = label(stat_map > z_th)
    labels = label_map[masker.mask_img_.get_data() > 0]

    # Fill all cluster voxels with the cluster p_value
    clusters_pval = np.zeros(len(stats))
    for label_ in range(1, n_labels + 1):
        if cluster_stat == 'size':
            cluster_stat = np.sum(labels == label_)
        elif cluster_stat == 'mass':
            cluster_stat = np.sum(stats[labels == label_])
        cluster_rank = np.sum(stat_dist < cluster_stat)
        cluster_pval = (n_dist + 1 - cluster_rank) / float(1 + n_dist)
        clusters_pval[labels == label_] = cluster_pval

    return clusters_pval


def second_level_permutation(Y, design_matrix, con_val, masker, stat_type=None,
                             threshold=0.001, height_control='fpr',
                             two_sided_test=True, n_perm=10000,
                             random_state=None, verbose=1, n_jobs=1):
    """Estimate uncorrected and FWE corrected p-values for voxel activation and
    cluster size.

    Parameters
    ----------
    Y : array of shape (n_subjects, n_voxels)
        The fMRI data.

    design_matrix : pandas DataFrame (n_subjects, n_regressors)
        The design matrix.

    con_val : array of shape (n_regressors,)
        Contrast specification to compute statistic.

    masker : NiftiMasker object,
        It must be the same masker employed to extract Y from nifti images.
        It will be used to compute clusters in permutations.

    stat_type : {'t', 'F'}, optional
        Type of the contrast

    threshold: float, optional
        cluster forming threshold (either a p-value or z-scale value)

    height_control: string, optional
        false positive control meaning of cluster forming
        threshold: 'fpr'|'fdr'|'bonferroni'|'none'

    two_sided_test : boolean,
        If True, performs an unsigned t-test. Both positive and negative
        effects are considered; the null hypothesis is that the effect is zero.
        If False, only positive effects are considered as relevant. The null
        hypothesis is that the effect is zero or negative.

    n_perm : int, optional
        Number of permutations. Greater than 0. Defaults to 10000.

    random_state : int or None,
        Seed for random number generator, to have the same permutations
        in each computing units.

    verbose : integer, optional
        Indicate the level of verbosity. By default, nothing is printed.

    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.

    Returns
    -------
    cor_pvals : array-like, shape=(n_voxels)
        FWE corrected p values based on max statistic of permutations.

    cs_max_parts : array-like, shape=(n_perm):
        Distribution of the (max) cluster size.

    cm_max_parts : array-like, shape=(n_perm):
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

    # initialize the seed of the random generator
    rng = check_random_state(random_state)

    per = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
        joblib.delayed(_sign_flip_glm)(
            Y, design_matrix, con_val, masker,
            stat_type=stat_type, threshold=threshold,
            height_control=height_control, two_sided_test=two_sided_test,
            n_perm_chunk=n_perm_chunk,
            random_state=rng.random_integers(np.iinfo(np.int32).max - 1),
            n_perm=n_perm, thread_id=thread_id, verbose=verbose)
        for thread_id, n_perm_chunk in enumerate(n_perm_chunks))

    smax_parts, cs_max_parts, cm_max_parts = zip(*per)

    # Compute original stat and clusters only once
    original_stat = _get_z_score(Y, design_matrix, con_val, stat_type, n_jobs)
    # Take absolute value of stat for two sided test
    if two_sided_test:
        original_stat = np.fabs(original_stat)

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
    cs_pvals = cluster_p_value(original_stat, masker, cs_max_dist,
                               'size', threshold=threshold,
                               height_control=height_control)
    # Get max cluster mass distribution
    cm_max_dist = np.concatenate(cm_max_parts)
    cm_pvals = cluster_p_value(original_stat, masker, cm_max_dist,
                               'mass', threshold=threshold,
                               height_control=height_control)

    return cor_pvals, cs_pvals, cm_pvals
