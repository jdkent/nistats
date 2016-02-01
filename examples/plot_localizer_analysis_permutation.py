"""
GLM fitting in fMRI
===================

Full step-by-step example of fitting a GLM to experimental data realizing
permutation tests and visualizing the results.

More specifically:

1. A sequence of fMRI volumes are loaded
2. A design matrix describing all the effects related to the data is computed
3. a mask of the useful brain volume is computed
4. A GLM is applied to the dataset (effect/covariance,
   then contrast estimation)
5. Permutation tests are performed to obtain uncorrected and corrected p values
at each voxel for a contrast comparing only two conditions.

"""

print(__doc__)

from os import mkdir, path

import numpy as np
import pandas as pd
from nilearn import plotting

from nistats.glm import FirstLevelGLM
from nistats.design_matrix import make_design_matrix
from nistats import datasets
from nistats.permutation_tests import first_level_permutation_test
from nistats.utils import z_score
import nibabel as nib


n_proc = 6

### Data and analysis parameters #######################################

# timing
n_scans = 128
tr = 2.4
frame_times = np.linspace(0.5 * tr, (n_scans - .5) * tr, n_scans)

# data
data = datasets.fetch_localizer_first_level()
paradigm_file = data.paradigm
fmri_img = data.epi_img

### Design matrix ########################################

paradigm = pd.read_csv(paradigm_file, sep=' ', header=None, index_col=None)
paradigm.columns = ['session', 'name', 'onset']
design_matrix = make_design_matrix(
    frame_times, paradigm, hrf_model='canonical with derivative',
    drift_model="cosine", period_cut=128)

### Perform a GLM analysis ########################################

fmri_glm = FirstLevelGLM().fit(fmri_img, design_matrix)

### Estimate contrasts #########################################

# Specify the contrasts
contrast_matrix = np.eye(design_matrix.shape[1])
contrasts = dict([(column, contrast_matrix[i])
                  for i, column in enumerate(design_matrix.columns)])

contrasts["audio"] = contrasts["clicDaudio"] + contrasts["clicGaudio"] +\
    contrasts["calculaudio"] + contrasts["phraseaudio"]
contrasts["video"] = contrasts["clicDvideo"] + contrasts["clicGvideo"] + \
    contrasts["calculvideo"] + contrasts["phrasevideo"]
contrasts["computation"] = contrasts["calculaudio"] + contrasts["calculvideo"]
contrasts["sentences"] = contrasts["phraseaudio"] + contrasts["phrasevideo"]

# Short list or more relevant contrasts
# contrasts = {
#     "left-right": (contrasts["clicGaudio"] + contrasts["clicGvideo"]
#                    - contrasts["clicDaudio"] - contrasts["clicDvideo"]),
#     "H-V": contrasts["damier_H"] - contrasts["damier_V"],
#     "audio-video": contrasts["audio"] - contrasts["video"],
#     "video-audio": -contrasts["audio"] + contrasts["video"],
#     "computation-sentences": (contrasts["computation"] -
#                               contrasts["sentences"]),
#     "reading-visual": contrasts["phrasevideo"] - contrasts["damier_H"]
#     }

contrasts = {
    "reading-visual": contrasts["phrasevideo"] - contrasts["damier_H"]
    }

### Permutation tests ###########################################

# Design matrix obj arrangement for paradigm permutations
# What about creating design_matrix class to wrap make design matrix and
# respective parameters. Then could just pass that object

design_matrix_obj = [frame_times, paradigm,
                     {'hrf_model': 'canonical with derivative',
                      'drift_model': 'cosine',
                      'period_cut': 128}]

# Run the permutations

res = first_level_permutation_test(contrasts, fmri_glm, fmri_img,
                                   [design_matrix_obj], n_perm=600,
                                   two_sided_test=True,
                                   cluster_threshold=None,
                                   random_state=None, verbose=10,
                                   n_jobs=n_proc)

### Plots #######################################################

# Create snapshots of the contrasts thresholded with parametric and non
# parametric methods.

# write directory
write_dir = 'results'
if not path.exists(write_dir):
    mkdir(write_dir)

for index, (contrast_id, (unc_pval, cor_pval)) in enumerate(res.iteritems()):
    z_map, = fmri_glm.transform(contrasts[contrast_id],
                                contrast_name=contrast_id,
                                output_z=True)
    nib.save(z_map, path.join(write_dir, '%s_z_map.nii' % contrast_id))
    z_map_data = fmri_glm.masker_.transform(z_map)[0]

    display = plotting.plot_stat_map(z_map, display_mode='z',
                                     threshold=z_score(0.1),
                                     title=contrast_id)
    display.savefig(path.join(write_dir, '%s_z_map.png' % contrast_id))

    nib.save(unc_pval, path.join(write_dir, '%s_unc_pval.nii' % contrast_id))
    nib.save(cor_pval, path.join(write_dir, '%s_cor_pval.nii' % contrast_id))

    threshold = (fmri_glm.masker_.transform(unc_pval)[0] < 0.1)
    thresholded_z_map = fmri_glm.masker_.inverse_transform(z_map_data * threshold)

    display = plotting.plot_stat_map(thresholded_z_map, display_mode='z',
                                     threshold=0.5, title=contrast_id)
    display.savefig(path.join(write_dir, '%s_unc_pval.png' % contrast_id))

    threshold = (fmri_glm.masker_.transform(cor_pval)[0] < 0.5)
    thresholded_z_map = fmri_glm.masker_.inverse_transform(z_map_data * threshold)
    display = plotting.plot_stat_map(thresholded_z_map, display_mode='z',
                                     threshold=0.5, title=contrast_id)
    display.savefig(path.join(write_dir, '%s_cor_pval.png' % contrast_id))
