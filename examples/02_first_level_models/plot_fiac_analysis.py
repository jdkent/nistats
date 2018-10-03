"""Simple example of two-session fMRI model fitting
================================================

Full step-by-step example of fitting a GLM to experimental data and visualizing
the results. This is done on two runs of one subject of the FIAC dataset.

For details on the data, please see:

Dehaene-Lambertz G, Dehaene S, Anton JL, Campagne A, Ciuciu P, Dehaene
G, Denghien I, Jobert A, LeBihan D, Sigman M, Pallier C, Poline
JB. Functional segregation of cortical language areas by sentence
repetition. Hum Brain Mapp. 2006: 27:360--371.
http://www.pubmedcentral.nih.gov/articlerender.fcgi?artid=2653076#R11

More specifically:

1. A sequence of fMRI volumes are loaded
2. A design matrix describing all the effects related to the data is computed
3. a mask of the useful brain volume is computed
4. A GLM is applied to the dataset (effect/covariance,
   then contrast estimation)

Technically, this example shows how to handle two sessions that
contain the same experimental conditions. The model directly returns a
fixed effect of the statistics across the two sessions.

"""


###############################################################################
# Create a write directory to work
# it will be a 'results' subdirectory of the current directory.
from os import mkdir, path, getcwd
write_dir = path.join(getcwd(), 'results')
if not path.exists(write_dir):
    mkdir(write_dir)

#########################################################################
# Prepare data and analysis parameters
# --------------------------------------
# 
# Note that there are two sessions

from nistats import datasets
data = datasets.fetch_fiac_first_level()
fmri_img = [data['func1'], data['func2']]

#########################################################################
# Create a mean image for plotting purpose
from nilearn.image import mean_img
mean_img_ = mean_img(fmri_img[0])

#########################################################################
# The design matrices were pre-computed, we simply put them in a list of DataFrames
design_files = [data['design_matrix1'], data['design_matrix2']]
import pandas as pd
import numpy as np
design_matrices = [pd.DataFrame(np.load(df)['X']) for df in design_files]

#########################################################################
# GLM estimation
# ----------------------------------
# GLM specification

from nistats.first_level_model import FirstLevelModel
fmri_glm = FirstLevelModel(mask=data['mask'], minimize_memory=True)

#########################################################################
# GLM fitting
fmri_glm = fmri_glm.fit(fmri_img, design_matrices=design_matrices)

#########################################################################
# Compute fixed effects of the two runs and compute related images
# For this, we first define the contrasts as we would do for a single session
n_columns = design_matrices[0].shape[1]

def pad_vector(contrast_, n_columns):
    """A small routine to append zeros in contrast vectors"""
    return np.hstack((contrast_, np.zeros(n_columns - len(contrast_))))

#########################################################################
# Contrast specification

contrasts = {'SStSSp_minus_DStDSp': pad_vector([1, 0, 0, -1], n_columns),
             'DStDSp_minus_SStSSp': pad_vector([-1, 0, 0, 1], n_columns),
             'DSt_minus_SSt': pad_vector([-1, -1, 1, 1], n_columns),
             'DSp_minus_SSp': pad_vector([-1, 1, -1, 1], n_columns),
             'DSt_minus_SSt_for_DSp': pad_vector([0, -1, 0, 1], n_columns),
             'DSp_minus_SSp_for_DSt': pad_vector([0, 0, -1, 1], n_columns),
             'Deactivation': pad_vector([-1, -1, -1, -1, 4], n_columns),
             'Effects_of_interest': np.eye(n_columns)[:5]}

#########################################################################
# Compute and plot statistics

from nilearn import plotting
print('Computing contrasts...')
for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
    print('  Contrast % 2i out of %i: %s' % (
        index + 1, len(contrasts), contrast_id))
    # estimate the contasts
    # note that the model implictly compute a fixed effects across the two sessions
    z_map = fmri_glm.compute_contrast(
        contrast_val, output_type='z_score')
    
    # Write the resulting stat images to file 
    z_image_path = path.join(write_dir, '%s_z_map.nii.gz' % contrast_id)
    z_map.to_filename(z_image_path)

#########################################################################
# make a snapshot of the 'Effects_of_interest' contrast map.
# We first compute a threshold corresponding to an FDR correction of .05
# We also discard isolated sets of less that 10 voxels
from nistats.thresholding import map_threshold
zmap = path.join(write_dir, 'Effects_of_interest_z_map.nii')
thresholded_map, threshold = map_threshold(
    zmap, height_control='fdr', threshold=.05, cluster_threshold=10)

#########################################################################
# Then display the map
display = plotting.plot_stat_map(
    thresholded_map, bg_img=mean_img_, threshold=threshold,
    title='%s, fdr=.05' % 'Effects of interest')

#########################################################################
# We can save the figure a posteriori
display.savefig(path.join(write_dir, '%s_z_map.png' % contrast_id))

plotting.show()

#########################################################################
# Generating a report
# -------------------
# Since we have already computed the FirstLevelModel and
# and have the contrast, we can quickly create a summary report.
from nistats.reporting import make_glm_report

report = make_glm_report(fmri_glm,
                         contrasts,
                         bg_img=mean_img_,
                         )

#########################################################################
# We have several ways to access the report:

report  # This report can be viewed in a notebook
# report.save_as_html('report.html')
# report.open_in_browser()
