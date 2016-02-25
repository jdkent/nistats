""" Example of a simple GLM fit on an OpenfMRI dataset subject

The script defines a General Linear Model and fits it to OpenfMRI data.

Author: Martin Perez-Guevara, 2016
"""

import os
import glob
import itertools
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.plotting import plot_stat_map, show
from sklearn.datasets.base import Bunch

from nistats.design_matrix import make_design_matrix
from nistats.glm import FirstLevelGLM
from nistats.datasets import fetch_openfmri_dataset


# Utility to load data from openfmri dataset for a subject

# openfmri dataset extraction
OPENFMRI_SEP = '\t'
OTHER_FILE_SEP = ' '


def _glob_openfmri_data(dataset_dir, sub_id, model_id, task_id):
    '''Extracts model from openfmri dataset for given task and subject'''
    _subject_data = {}
    onsets_template = os.path.join(dataset_dir,
                                   'sub{0:03d}'.format(sub_id),
                                   'model',
                                   'model{0:03d}'.format(model_id),
                                   'onsets',
                                   'task{0:03d}'.format(task_id) + '_run{0}',
                                   '{1}.txt')
    condkey_file = os.path.join(dataset_dir, 'models',
                                'model{0:03d}'.format(model_id),
                                'condition_key.txt')
    cond_df = pd.read_csv(condkey_file,
                          sep=OTHER_FILE_SEP,
                          header=None)
    info_conditions = cond_df[cond_df[0] == 'task{0:03d}'.format(task_id)]
    conds = info_conditions[1].tolist()
    conds_name = info_conditions[2].tolist()
    run_path = os.path.join(dataset_dir, 'sub{0:03d}'.format(sub_id),
                            'BOLD', 'task{0:03d}_run*'.format(task_id),
                            'bold.nii.gz')
    runs = glob.glob(run_path)
    runs.sort()
    TR = float(open(os.path.join(dataset_dir, 'scan_key.txt'))
               .read().split(OTHER_FILE_SEP)[1])
    _run_events = {}
    all_event_files = []

    for run in runs:
        names = []
        allonsets = []
        alldurations = []
        event_files = []
        # This will be used to add dummy events.
        vol = nib.load(os.path.join(dataset_dir, 'sub{0:03d}', 'BOLD',
                                    'task{1:03d}_run001', 'bold.nii.gz')
                       .format(sub_id, task_id)).shape[3]
        for i in range(len(conds)):
            names.append(conds_name[i])
            run_id = run[run.index('run')+3:run.index('run')+6]
            cond_file = onsets_template.format(run_id, conds[i])
            event_files.append(cond_file)
            onsets = []
            durations = []
            # ASSUMING WE NEED AT LEAST TWO EVENTS FOR ANY CONDITION
            if os.stat(cond_file).st_size > 0:
                cond_info = pd.read_csv(cond_file,
                                        sep=OPENFMRI_SEP,
                                        header=None)
                onsets = cond_info[0].tolist()
                durations = cond_info[1].tolist()
            else:
                print 'empty file found: ' + cond_file
                onsets = [vol*TR - 0.1]
                durations = [0.0]
            allonsets.append(onsets)
            alldurations.append(durations)
            n_on = len(list(itertools.chain.from_iterable(allonsets)))
            n_dur = len(list(itertools.chain.from_iterable(alldurations)))
            assert(n_on == n_dur)

        _run_events[int(run_id)-1] = {}
        _run_events[int(run_id)-1]['conditions'] = names
        _run_events[int(run_id)-1]['onsets'] = allonsets
        _run_events[int(run_id)-1]['durations'] = alldurations
        all_event_files.append(event_files)

    _subject_data['TR'] = TR
    _subject_data['func'] = runs
    _subject_data['anat'] = os.path.join(dataset_dir,
                                         'sub{0:03d}'.format(sub_id),
                                         'anatomy', 'highres001.nii.gz')
    _subject_data['func_events'] = _run_events
    _subject_data['onset_files'] = all_event_files

    return Bunch(**_subject_data)


def _openfrmri_contrasts(ref_design_matrix, dataset_dir, model_id, task_id):
    # Build basic contrast_matrix.
    contrast_matrix = np.eye(ref_design_matrix.shape[1])
    contrasts = dict([(column, contrast_matrix[i])
                      for i, column in enumerate(ref_design_matrix.columns)])

    # Load information from openfmri files
    task = 'task{0:03d}'.format(task_id)
    condkey_file = os.path.join(dataset_dir, 'models',
                                'model{0:03d}'.format(model_id),
                                'condition_key.txt')
    cond_df = pd.read_csv(condkey_file,
                          sep=OTHER_FILE_SEP,
                          header=None)
    info_conditions = cond_df[cond_df[0] == task]
    conds_name = info_conditions[2].tolist()
    contrasts_file = os.path.join(dataset_dir, 'models',
                                  'model{0:03d}'.format(model_id),
                                  'task_contrasts.txt')
    contrasts_df = pd.read_csv(contrasts_file,
                               sep=OTHER_FILE_SEP,
                               header=None)
    task_contrasts = contrasts_df[contrasts_df[0] == task]

    # Complete contrast specification
    for idx, crow in task_contrasts.iterrows():
        contrasts[crow[1]] = np.zeros(ref_design_matrix.shape[1])
        for widx, weight in enumerate(crow[2:]):
            contrasts[crow[1]] += weight*contrasts[conds_name[widx]]

    return contrasts

# dataset parameters
sub_id = 1
model_id = 1
task_id = 1

# download openfmri dataset
dataset_dir = fetch_openfmri_dataset(dataset_id='ds001')

# fetch openfmri dataset
data = _glob_openfmri_data(dataset_dir, sub_id, model_id, task_id)

# construct design matrices
design_matrices = []
contrasts = []
for session in range(len(data.func_events)):
    onsets = []
    durations = []
    names = []
    n_cond = len(data.func_events[session]['conditions'])
    for c in range(n_cond):
        onsets += data.func_events[session]['onsets'][c]
        durations += data.func_events[session]['durations'][c]
        n_events = len(data.func_events[session]['durations'][c])
        names += [data.func_events[session]['conditions'][c]] * n_events
    paradigm = pd.DataFrame({'onset': onsets, 'duration': durations,
                             'name': names})
    n_scans = nib.load(data.func[session]).get_data().shape[3]
    frame_times = np.linspace(0, (n_scans - 1) * data.TR, n_scans)
    drift_model = 'Cosine'
    hrf_model = 'Canonical With Derivative'
    design_matrix = make_design_matrix(frame_times, paradigm,
                                       hrf_model=hrf_model,
                                       drift_model=drift_model)
    design_matrices.append(design_matrix)

    # specify contrasts
    contrasts.append(_openfrmri_contrasts(design_matrix, dataset_dir,
                                          model_id, task_id))

# fit GLM
print('\r\nFitting a GLM (this takes time) ..')
fmri_glm = FirstLevelGLM(noise_model='ar1', standardize=False).fit(
    data.func, design_matrices)

print("Computing contrasts ..")
output_dir = os.path.join('dataset_download', 'ds001_analysis',
                          'sub{0:03d}'.format(sub_id))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

anat_img = nib.load(data['anat'])
for contrast_id in contrasts[1].keys()[-6:]:
    print("\tcontrast id: %s" % contrast_id)
    z_map, t_map, eff_map, var_map = fmri_glm.transform(
        [scon[contrast_id] for scon in contrasts], contrast_name=contrast_id,
        output_z=True, output_stat=True, output_effects=True,
        output_variance=True)

    # store stat maps to disk
    for dtype, out_map in zip(['z', 't', 'effects', 'variance'],
                              [z_map, t_map, eff_map, var_map]):
        map_dir = os.path.join(output_dir, '%s_maps' % dtype)
        if not os.path.exists(map_dir):
            os.makedirs(map_dir)
        map_path = os.path.join(map_dir, '%s.nii.gz' % contrast_id)
        nib.save(out_map, map_path)
        print("\t\t%s map: %s" % (dtype, map_path))

    # plot one activation map

    display = plot_stat_map(z_map, bg_img=anat_img, threshold=3.0,
                            display_mode='z', cut_coords=3, black_bg=True,
                            title=contrast_id)

show()
