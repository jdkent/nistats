"""
Tests for second level sign flipping permutations
"""
import numpy as np
from scipy import stats
from sklearn.utils import check_random_state
import pandas as pd

from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
                           assert_array_less, assert_equal)
import nibabel as nib
from nibabel import load
from nilearn.input_data import NiftiMasker
from nilearn._utils.niimg_conversions import check_niimg
from nibabel.tmpdirs import InTemporaryDirectory

from nistats.second_level_model import SecondLevelModel
from nistats.tests.test_second_level_model import write_fake_fmri_data
from nistats.permutation_tests import (_get_z_score, _sign_flip_glm,
                                       _maxstat_thresholding, cluster_p_value,
                                       second_level_permutation)


def test_get_z_score():
    # Make sure the permutation stat computation is consistent
    with InTemporaryDirectory():
        shapes = ((7, 8, 9, 1),)
        mask, FUNCFILE, _ = write_fake_fmri_data(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)

        # estimate contrast from second level model
        model = SecondLevelModel(mask=mask)
        Y = [func_img] * 4
        X = pd.DataFrame([[1]] * 4, columns=['intercept'])

        model = model.fit(Y, design_matrix=X)
        stat1 = model.compute_contrast(output_type='z_score')
        stat1 = model.masker_.transform(stat1)[0, :]

        # estimate contrast from get_stat
        for niimg in Y:
            check_niimg(niimg, ensure_ndim=3)
        Y = model.masker_.transform(Y)
        stat2 = _get_z_score(Y, X, np.array([1.]), None)

        # stats should be the same
        assert_almost_equal(stat1, stat2, decimal=4)


def test_maxstat_thresholding():
    with InTemporaryDirectory():
        n_samples = 1
        shapes = [(100, 100, 100, 1)] * n_samples
        mask, Y, _ = write_fake_fmri_data(shapes, mask_th=-0.1, img_mean=0.)
        masker = NiftiMasker(mask).fit()

        effect = stats.norm.isf(0.0001)  # effect greater than p val 0.001
        # Create clusters, the biggest has to be captured
        Y = load(Y[0])
        clusters_matrix = Y.get_data()[:, :, :, 0]
        clusters_matrix[:, :, :] = 0.
        clusters_matrix[:5, :5, :5] = effect
        clusters_matrix[90:, 90:, 90:] = effect
        cluster_size = 1000
        cluster_mass = 1000 * effect
        clusters_img = nib.Nifti1Image(clusters_matrix, Y.get_affine())
        Y = masker.transform(clusters_img)[0, :]
        cluster_max = _maxstat_thresholding(Y, masker)
        assert_almost_equal(np.array([cluster_size, cluster_mass]),
                            np.array(cluster_max), decimal=4)


def test_sign_flip_glm_noeffect():
    with InTemporaryDirectory():
        # Create dummy gaussian dataset with no effects
        n_samples = 50
        res = 3
        shapes = [(res, res, res, 1)] * n_samples
        mask, Y, _ = write_fake_fmri_data(shapes, mask_th=-.1, img_mean=0)
        con_val = np.array([1.])
        X = pd.DataFrame([[1]] * n_samples, columns=['intercept'])
        masker = NiftiMasker(mask).fit()
        Y = masker.transform(Y)

        # Estimate Bonferroni corrected z threshold
        pval_th = 0.1  # Pick this value for perm_ranges multiples of 10
        bonferroni_th = stats.norm.isf(pval_th / (res ** 3))

        # As permutations increase we should approximate Bonferroni correction
        perm_ranges = [10, 100, 1000]
        perm_repetitions = 5
        perm_error = []
        for i, n_perm in enumerate(np.repeat(perm_ranges, perm_repetitions)):
            results = _sign_flip_glm(Y, X, con_val, masker,
                                     n_perm=n_perm, n_perm_chunk=n_perm,
                                     two_sided_test=False)
            smax, cs_max, cm_max = results
            assert_equal(len(smax), n_perm)

            # The permutation th is given by the [n_perm * (1. - pval_th) - 1]
            # index in the sorted array. Since we substract from n_perm + 1 all
            # the max stat lower than the original stat
            perm_th = sorted(smax)[int(n_perm * (1. - pval_th)) - 1]
            perm_error.append((perm_th - bonferroni_th) ** 2)

        # Consistency of the algorithm: the more permutations, the closer to
        # the bonferroni_z threshold
        perm_error = np.array(perm_error).reshape(len(perm_ranges),
                                                  perm_repetitions)
        assert_array_less(np.diff(np.mean(perm_error, axis=1)), 0)


def test_sign_flip_glm_sided_test(random_state=0):
    with InTemporaryDirectory():
        # Create dummy gaussian dataset with no effects
        n_samples = 50
        res = 4
        shapes = [(res, res, res, 1)] * n_samples
        mask, Y, _ = write_fake_fmri_data(shapes, mask_th=-.1, img_mean=0)
        con_val = np.array([1.])
        X = pd.DataFrame([[1]] * n_samples, columns=['intercept'])
        masker = NiftiMasker(mask).fit()
        Y = masker.transform(Y)

        original_stat = _get_z_score(Y, X, con_val, None)
        positive_effect_location = original_stat > 0.
        negative_effect_location = original_stat < 0.

        pvals_twosided, _, _ = second_level_permutation(
            Y, X, con_val, masker, n_perm=1000)
        pvals_onesided1, _, _ = second_level_permutation(
            Y, X, con_val, masker, n_perm=1000, two_sided_test=False)
        pvals_onesided2, _, _ = second_level_permutation(
            -Y, X, con_val, masker, n_perm=1000, two_sided_test=False)

        # Check that an effect is always better recovered with one-sided
        assert_equal(
            np.sum(pvals_onesided1[positive_effect_location] -
                   pvals_twosided[positive_effect_location] > 0), 0)
        assert_equal(
            np.sum(pvals_onesided2[negative_effect_location] -
                   pvals_twosided[negative_effect_location] > 0), 0)

        # check only negative or positive effects are captured with one-sided
        assert_equal(
            np.sum(pvals_onesided2[negative_effect_location] -
                   pvals_onesided1[negative_effect_location] > 0), 0)
        assert_equal(
            np.sum(pvals_onesided1[positive_effect_location] -
                   pvals_onesided2[positive_effect_location] > 0), 0)

        # Check positive and negative effects are captured with two-sided
        if np.sum(positive_effect_location) > 0:
            assert(np.sum(pvals_twosided[positive_effect_location] < 1.) > 0)
        if np.sum(negative_effect_location) > 0:
            assert(np.sum(pvals_twosided[negative_effect_location] < 1.) > 0)


def test_cluster_p_value():
    with InTemporaryDirectory():
        n_samples = 1
        shapes = [(100, 100, 100, 1)] * n_samples
        mask, Y, _ = write_fake_fmri_data(shapes, mask_th=-0.1, img_mean=0.)
        masker = NiftiMasker(mask).fit()

        effect = stats.norm.isf(0.0001)  # effect greater than p val 0.001
        # Create clusters, the biggest has to be captured
        Y = load(Y[0])
        clusters_matrix = Y.get_data()[:, :, :, 0]
        clusters_matrix[:, :, :] = 0.
        clusters_matrix[:10, :10, :10] = effect
        cluster_size = 1000
        cluster_mass = 1000 * effect
        clusters_img = nib.Nifti1Image(clusters_matrix, Y.get_affine())
        Y = masker.transform(clusters_img)[0, :]

        # Create fake distribution for mass and size cluster stats
        # The idea is to get back the expected p value from our fake dist
        size_dist = [0] * 90 + [cluster_size + 1] * 10  # 11/101. (.1089 pval)
        mass_dist = [0] * 90 + [cluster_mass + 1] * 10  # 11/101. (.1089 pval)

        size_pval = cluster_p_value(Y, masker, size_dist, 'size')
        mass_pval = cluster_p_value(Y, masker, mass_dist, 'mass')
        # The voxels in cluster are the only ones with p val > 0.1 (0.1089)
        assert(np.sum(size_pval > 0.1) == 1000)
        assert(np.sum(mass_pval > 0.1) == 1000)


def test_second_level_permutation():
    # Quick check between bonferroni correction and permuted corrected p values
    with InTemporaryDirectory():
        # Create dummy gaussian dataset with no effects
        n_samples = 50
        res = 3
        shapes = [(res, res, res, 1)] * n_samples
        mask, Y, _ = write_fake_fmri_data(shapes, mask_th=-.1, img_mean=0)
        con_val = np.array([1.])
        X = pd.DataFrame([[1]] * n_samples, columns=['intercept'])
        masker = NiftiMasker(mask).fit()
        Y = masker.transform(Y)
        original_stat = _get_z_score(Y, X, con_val, None)

        # Estimate Bonferroni corrected z threshold
        pval_th = 0.99  # Pick this value for perm_ranges multiples of 10
        bonferroni_th = stats.norm.isf(pval_th / (res ** 3))

        # Get the permutation corrected p values
        results = second_level_permutation(
            Y, X, con_val, masker, n_perm=10000, two_sided_test=False)
        cor_pvals, _, _ = results
        cor_pvals_passed_bonferroni = cor_pvals[original_stat > bonferroni_th]
        # Voxels that pass the bonferroni test should pass the permutation test
        assert(np.sum(cor_pvals_passed_bonferroni < pval_th) ==
               len(cor_pvals_passed_bonferroni))
