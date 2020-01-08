"""
Predicted time series and residuals
===================================

Here we fit a First Level GLM with the `minimize_memory`-argument set to `False`.
By doing so, the `FirstLevelModel`-object stores the residuals, which we can then inspect.
Also, the predicted time series can be extracted, which is useful to assess the quality of the model fit.
"""


#########################################################################
# Import modules
# --------------
from nistats.datasets import fetch_spm_auditory
from nilearn import image
from nilearn import masking
import pandas as pd


# load fMRI data
subject_data = fetch_spm_auditory()
fmri_img = image.concat_imgs(subject_data.func)

# Make an average
mean_img = image.mean_img(fmri_img)
mask = masking.compute_epi_mask(mean_img)

# Clean and smooth data
fmri_img = image.clean_img(fmri_img, standardize=False)
fmri_img = image.smooth_img(fmri_img, 5.)

# load events
events = pd.read_table(subject_data['events'])


#########################################################################
# Fit model
# ---------
# Note that `minimize_memory` is set to `False` so that `FirstLevelModel`
# stores the residuals.
# `signal_scaling` is set to False, so we keep the same scaling as the
# original data in `fmri_img`.
from nistats.first_level_model import FirstLevelModel

fmri_glm = FirstLevelModel(t_r=7,
                           drift_model='cosine',
                           signal_scaling=False,
                           mask_img=mask,
                           minimize_memory=False)

fmri_glm = fmri_glm.fit(fmri_img, events)


#########################################################################
# Calculate and plot contrast
# ---------------------------
from nilearn import plotting

z_map = fmri_glm.compute_contrast('active - rest')

plotting.plot_stat_map(z_map, bg_img=mean_img, threshold=3.1)

#########################################################################
# Extract the largest clusters
# ----------------------------
from nistats.reporting import get_clusters_table
from nilearn import input_data

table = get_clusters_table(z_map, stat_threshold=3.1,
                           cluster_threshold=20).set_index('Cluster ID', drop=True)
table.head()

masker = input_data.NiftiSpheresMasker(table.loc[range(1, 7), ['X', 'Y', 'Z']].values)

real_timeseries = masker.fit_transform(fmri_img)
predicted_timeseries = masker.fit_transform(fmri_glm.predicted)


#########################################################################
# Plot predicted and actual time series for 6 most significant clusters
# ---------------------------------------------------------------------
import matplotlib.pyplot as plt

for i in range(1, 7):
    plt.subplot(2, 3, i)
    plt.title('Cluster peak {}\n'.format(table.loc[i, ['X', 'Y', 'Z']].tolist()))
    plt.plot(real_timeseries[:, i-1], c='k', lw=2)
    plt.plot(predicted_timeseries[:, i-1], c='r',  ls='--', lw=2)
    plt.xlabel('Time')
    plt.ylabel('Signal intensity')

plt.gcf().set_size_inches(12, 7)
plt.tight_layout()

#########################################################################
# Get residuals
# -------------
resid = masker.fit_transform(fmri_glm.residuals)


#########################################################################
# Plot distribution of residuals
# ------------------------------
# Note that residuals are not really distributed normally.


for i in range(1, 7):
    plt.subplot(2, 3, i)
    plt.title('Cluster peak {}\n'.format(table.loc[i, ['X', 'Y', 'Z']].tolist()))
    plt.hist(resid[:, i-1])
    print('Mean residuals: {}'.format(resid[:, i-1].mean()))

plt.gcf().set_size_inches(12, 7)
plt.tight_layout()


#########################################################################
# Plot R-squared
# --------------
# Because we stored the residuals, we can plot the R-squared: the proportion
# of explained variance of the GLM as a whole. Note that the R-squared is markedly
# lower deep down the brain, where there is more physiological noise and we 
# are further away from the receive coils. However, R-Squared should be interpreted
# with a grain of salt. The R-squared value will necessarily increase with
# the addition of more factors (such as rest, active, drift, motion) into the GLM.
# Additionally, we are looking at the overall fit of the model, so we are
# unable to say whether a voxel/region has a large R-squared value because
# the voxel/region is responsive to the experiment (such as active or rest)
# or because the voxel/region fits the noise factors (such as drift or motion)
# that could be present in the GLM. To isolate the influence of the experiment,
# we can use an F-test as shown in the next section.
plotting.plot_stat_map(fmri_glm.r_square, 
                       bg_img=mean_img, threshold=.1, display_mode='z', cut_coords=7)


#########################################################################
# Calculate and Plot F-test
# -------------------------
# The F-test tells you how well the GLM fits effects of interest such as 
# the active and rest conditions together. This is different from R-squared,
# which tells you how well the overall GLM fits the data, including active, rest
# and all the other columns in the design matrix such as drift and motion.
import numpy as np

design_matrix = fmri_glm.design_matrices_[0]
active = np.array([1 if c == 'active' else 0 for c in design_matrix.columns])
rest = np.array([1 if c == 'rest' else 0 for c in design_matrix.columns])
effects_of_interest = np.vstack((active, rest))

z_map = fmri_glm.compute_contrast(effects_of_interest,
                                  output_type='z_score')

plotting.plot_stat_map(z_map, 
                       bg_img=mean_img, threshold=2.33, display_mode='z', cut_coords=7)
