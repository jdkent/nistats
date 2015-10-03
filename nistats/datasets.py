"""
Utilities to download NeuroImaging datasets
"""
# Author: Gael Varoquaux
# License: simplified BSD

import os
import re
import glob
import nibabel
import urllib2
from sklearn.datasets.base import Bunch
import pandas as pd
import nibabel as nib
import itertools

from nilearn.datasets.utils import (
    _get_dataset_dir, _fetch_files, _fetch_file, _uncompress_file)

SPM_AUDITORY_DATA_FILES = ["fM00223/fM00223_%03i.img" % index
                           for index in range(4, 100)]
SPM_AUDITORY_DATA_FILES.append("sM00223/sM00223_002.img")


def fetch_localizer_first_level(data_dir=None, verbose=1):
    """ Download a first-level localizer fMRI dataset

    Parameters
    ----------
    data_dir: string
        directory where data should be downloaded and unpacked.

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        dictionary-like object, keys are:
        epi_img: the input 4D image
        paradigm: a csv file describing the paardigm
    """
    url = 'ftp://ftp.cea.fr/pub/dsv/madic/download/nipy'

    dataset_name = "localizer_first_level"
    files = dict(epi_img="s12069_swaloc1_corr.nii.gz",
                 paradigm="localizer_paradigm.csv")
    # The options needed for _fetch_files
    options = [(filename, os.path.join(url, filename), {})
               for _, filename in sorted(files.items())]

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    sub_files = _fetch_files(data_dir, options, resume=True,
                             verbose=verbose)

    params = dict(zip(sorted(files.keys()), sub_files))

    return Bunch(**params)


def fetch_spm_auditory(data_dir=None, data_name='spm_auditory',
                       subject_id="sub001", verbose=1):
    """Function to fetch SPM auditory single-subject data.

    Parameters
    ----------
    data_dir: string
        Path of the data directory. Used to force data storage in a specified
        location. If the data is already present there, then will simply
        glob it.

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are:
        - 'func': string list. Paths to functional images
        - 'anat': string list. Path to anat image

    References
    ----------
    :download:
        http://www.fil.ion.ucl.ac.uk/spm/data/auditory/

    """
    data_dir = _get_dataset_dir(data_name, data_dir=data_dir,
                                verbose=verbose)
    subject_dir = os.path.join(data_dir, subject_id)

    def _glob_spm_auditory_data():
        """glob data from subject_dir.

        """

        if not os.path.exists(subject_dir):
            return None

        subject_data = {}
        for file_name in SPM_AUDITORY_DATA_FILES:
            file_path = os.path.join(subject_dir, file_name)
            if os.path.exists(file_path):
                subject_data[file_name] = file_path
            else:
                print("%s missing from filelist!" % file_name)
                return None

        _subject_data = {}
        _subject_data["func"] = sorted(
            [subject_data[x] for x in subject_data.keys()
             if re.match("^fM00223_0\d\d\.img$", os.path.basename(x))])

        # volumes for this dataset of shape (64, 64, 64, 1); let's fix this
        for x in _subject_data["func"]:
            vol = nibabel.load(x)
            if len(vol.shape) == 4:
                vol = nibabel.Nifti1Image(vol.get_data()[:, :, :, 0],
                                          vol.get_affine())
                nibabel.save(vol, x)

        _subject_data["anat"] = [subject_data[x] for x in subject_data.keys()
                                 if re.match("^sM00223_002\.img$",
                                             os.path.basename(x))][0]

        # ... same thing for anat
        vol = nibabel.load(_subject_data["anat"])
        if len(vol.shape) == 4:
            vol = nibabel.Nifti1Image(vol.get_data()[:, :, :, 0],
                                      vol.get_affine())
            nibabel.save(vol, _subject_data["anat"])

        return Bunch(**_subject_data)

    # maybe data_dir already contains the data ?
    data = _glob_spm_auditory_data()
    if data is not None:
        return data

    # No. Download the data
    print("Data absent, downloading...")
    url = ("http://www.fil.ion.ucl.ac.uk/spm/download/data/MoAEpilot/"
           "MoAEpilot.zip")
    archive_path = os.path.join(subject_dir, os.path.basename(url))
    _fetch_file(url, subject_dir)
    try:
        _uncompress_file(archive_path)
    except:
        print("Archive corrupted, trying to download it again.")
        return fetch_spm_auditory(data_dir=data_dir, data_name="",
                                  subject_id=subject_id)

    return _glob_spm_auditory_data()


def fetch_spm_multimodal_fmri(data_dir=None, data_name="spm_multimodal_fmri",
                              subject_id="sub001", verbose=1):
    """Fetcher for Multi-modal Face Dataset.

    Parameters
    ----------
    data_dir: string
        path of the data directory. Used to force data storage in a specified
        location. If the data is already present there, then will simply
        glob it.

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are:
        - 'func1': string list. Paths to functional images for session 1
        - 'func2': string list. Paths to functional images for session 2
        - 'trials_ses1': string list. Path to onsets file for session 1
        - 'trials_ses2': string list. Path to onsets file for session 2
        - 'anat': string. Path to anat file

    References
    ----------
    :download:
        http://www.fil.ion.ucl.ac.uk/spm/data/mmfaces/

    """

    data_dir = _get_dataset_dir(data_name, data_dir=data_dir,
                                verbose=verbose)
    subject_dir = os.path.join(data_dir, subject_id)

    def _glob_spm_multimodal_fmri_data():
        """glob data from subject_dir."""
        _subject_data = {'slice_order': 'descending'}

        for session in range(2):
            # glob func data for session s + 1
            session_func = sorted(glob.glob(
                    os.path.join(
                        subject_dir,
                        ("fMRI/Session%i/fMETHODS-000%i-*-01.img" % (
                                session + 1, session + 5)))))
            if len(session_func) < 390:
                print("Missing %i functional scans for session %i." % (
                    390 - len(session_func), session))
                return None

            _subject_data['func%i' % (session + 1)] = session_func

            # glob trials .mat file
            sess_trials = os.path.join(
                subject_dir,
                "fMRI/trials_ses%i.mat" % (session + 1))
            if not os.path.isfile(sess_trials):
                print("Missing session file: %s" % sess_trials)
                return None

            _subject_data['trials_ses%i' % (session + 1)] = sess_trials

        # glob for anat data
        anat = os.path.join(subject_dir, "sMRI/smri.img")
        if not os.path.isfile(anat):
            print("Missing structural image.")
            return None

        _subject_data["anat"] = anat

        return Bunch(**_subject_data)

    # maybe data_dir already contains the data ?
    data = _glob_spm_multimodal_fmri_data()
    if data is not None:
        return data

    # No. Download the data
    print("Data absent, downloading...")
    urls = [
        # fmri
        ("http://www.fil.ion.ucl.ac.uk/spm/download/data/mmfaces/"
        "multimodal_fmri.zip"),

        # structural
        ("http://www.fil.ion.ucl.ac.uk/spm/download/data/mmfaces/"
         "multimodal_smri.zip")
        ]

    for url in urls:
        archive_path = os.path.join(subject_dir, os.path.basename(url))
        _fetch_file(url, subject_dir)
        try:
            _uncompress_file(archive_path)
        except:
            print("Archive corrupted, trying to download it again.")
            return fetch_spm_multimodal_fmri(data_dir=data_dir,
                                             data_name="",
                                             subject_id=subject_id)

    return _glob_spm_multimodal_fmri_data()


def fetch_fiac_first_level(data_dir=None, verbose=1):
    """ Download a first-level fiac fMRI dataset (2 sessions)

    Parameters
    ----------
    data_dir: string
        directory where data should be downloaded and unpacked.
    """
    data_dir = _get_dataset_dir('', data_dir=data_dir, verbose=verbose)

    def _glob_fiac_data():
        """glob data from subject_dir."""
        _subject_data = {}
        subject_dir = os.path.join(data_dir, 'nipy-data-0.2/data/fiac/fiac0')
        for session in [1, 2]:
            # glob func data for session session + 1
            session_func = os.path.join(subject_dir, 'run%i.nii.gz' % session)
            if not os.path.isfile(session_func):
                print('Missing functional scan for session %i.' % session)
                return None

            _subject_data['func%i' % session] = session_func

            # glob design matrix .npz file
            sess_dmtx = os.path.join(subject_dir, 'run%i_design.npz' % session)
            if not os.path.isfile(sess_dmtx):
                print('Missing session file: %s' % sess_dmtx)
                return None

            _subject_data['design_matrix%i' % session] = sess_dmtx

        # glob for mask data
        mask = os.path.join(subject_dir, 'mask.nii.gz')
        if not os.path.isfile(mask):
            print('Missing mask image.')
            return None

        _subject_data['mask'] = mask
        return Bunch(**_subject_data)

    # maybe data_dir already contains the data ?
    data = _glob_fiac_data()
    if data is not None:
        return data

    # No. Download the data
    print('Data absent, downloading...')
    url = 'http://nipy.sourceforge.net/data-packages/nipy-data-0.2.tar.gz'

    archive_path = os.path.join(data_dir, os.path.basename(url))
    _fetch_file(url, data_dir)
    try:
        _uncompress_file(archive_path)
    except:
        print('Archive corrupted, trying to download it again.')
        return fetch_fiac_first_level(data_dir=data_dir)

    return _glob_fiac_data()


def fetch_openfmri(data_dir=None, dataset_id=105, force_download=False,
                   sub_id=1, task_id=1, model_id=1, verbose=1):
    '''Download openfmri datasets.

    Currently the openfmri website employs 6 digits for the dataset id in
    the specification, but only 3 digits in the file name to download. So
    we assume datasets only until 999 with download link of the form
    http://openfmri.s3.amazonaws.com/tarballs/ds[000]{Letter}_raw{_part[0]}.

    Parameters
    ----------
    data_dir: string
        directory where data should be downloaded and unpacked
    dataset_id: int
        number of the dataset to download
    '''
    data_dir = _get_dataset_dir('', data_dir=data_dir, verbose=verbose)
    dataset_dir = os.path.join(data_dir, 'ds{0:03d}'.format(dataset_id))
    ONSETS_FILE_SEP = '\t'
    OTHER_FILE_SEP = ' '


    def _glob_openfmri_data():
        '''Extracts model from openfmri dataset for given task and subject'''
        _subject_data = {}
        onsets_template = os.path.join(dataset_dir,
                                       'sub{0:03d}'.format(sub_id),
                                       'model',
                                       'model{0:03d}'.format(model_id),
                                       'onsets',
                                       'task{0:03d}'.format(task_id)+'_run{0}',
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
        TR = float(open(os.path.join(dataset_dir,
                                     'scan_key.txt')).read().split(OTHER_FILE_SEP)[1])
        _subject_data['TR'] = TR
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
                                            sep=ONSETS_FILE_SEP,
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

            _run_events[run_id] = {}
            _run_events[run_id]['conditions'] = names
            _run_events[run_id]['onsets'] = allonsets
            _run_events[run_id]['durations'] = alldurations
            all_event_files.append(event_files)

        _subject_data['func'] = runs
        _subject_data['anat'] = os.path.join(dataset_dir,
                                             'sub{0:03d}'.format(sub_id),
                                             'anatomy', 'highres001.nii.gz')
        _subject_data['func_events'] = _run_events
        _subject_data['onset_files'] = all_event_files

        return Bunch(**_subject_data)

    # maybe data_dir already contains the data ?
    if os.path.exists(dataset_dir):
        print 'Dataset found'
        return _glob_openfmri_data()

    # No. Download the data
    def check_link_exist(link):
        try:
            urllib2.urlopen(link)
            return True
        except urllib2.HTTPError, e:
            return False
        except urllib2.URLError, e:
            return False

    def explore_posible_urls():
        files = []
        base_url = 'http://openfmri.s3.amazonaws.com/tarballs/ds{0:03d}{1}_raw{2}.tgz'
        dataset_links_found = False
        while(not dataset_links_found):
            checking_parts_and_groups = True
            checking_groups = True
            checking_parts = True
            group = 'A'
            part = 1
            while(checking_parts_and_groups):
                url = base_url.format(dataset_id, group, '_part%d' % part)
                if check_link_exist(url):
                    files.append(url)
                    part += 1
                elif part > 1:
                    part = 1
                    group += 1
                else:
                    checking_parts_and_groups = False
                    if files:
                        dataset_links_found = True
                        checking_parts = False
                        checking_groups = False
            while(checking_parts):
                url = base_url.format(dataset_id, '', '_part%d' % part)
                if check_link_exist(url):
                    files.append(url)
                    part += 1
                else:
                    checking_parts = False
                    if files:
                        dataset_links_found = True
                        checking_groups = False
            group = 'A'
            while(checking_groups):
                url = base_url.format(dataset_id, group, '')
                if check_link_exist(url):
                    files.append(url)
                    group += 1
                else:
                    checking_groups = False
                    if files:
                        dataset_links_found = True
            if not files:
                url = base_url.format(dataset_id, '', '')
                if check_link_exist(url):
                    files.append(url)
                else:
                    raise Exception('Can not find dataset %s' % dataset_id)
        return files

    urls = [('ds{0:03d}'.format(dataset_id), f,
             {'uncompress': True}) for f in explore_posible_urls()]
    output_dir = os.path.join(data_dir, 'ds{0:03d}'.format(dataset_id))
    if not os.path.exists(output_dir) and not force_download:
        _fetch_files(data_dir, urls, verbose=verbose)

    return _glob_openfmri_data()


def fetch_openfmri2(data_dir, dataset_id, force_download=False, verbose=1):
    files = {
        'ds001': ['ds001_raw'],
        'ds002': ['ds002_raw'],
        'ds003': ['ds003_raw'],
        'ds005': ['ds005_raw'],
        'ds006A': ['ds006A_raw'],
        'ds007': ['ds007_raw'],
        'ds008': ['ds008_raw'],
        'ds011': ['ds011_raw'],
        'ds017A': ['ds017A_raw'],
        'ds017B': ['ds017B_raw'],
        'ds051': ['ds051_raw'],
        'ds052': ['ds052_raw'],
        'ds101': ['ds101_raw'],
        'ds102': ['ds102_raw'],
        'ds105': ['ds105_raw'],
        'ds107': ['ds107_raw'],
        'ds108': ['ds108_raw_part1', 'ds108_raw_part2', 'ds108_raw_part3'],
        'ds109': ['ds109_raw'],
        'ds110': ['ds110_raw_part1', 'ds110_raw_part2', 'ds110_raw_part3',
                  'ds110_raw_part4', 'ds110_raw_part5', 'ds110_raw_part6']
        }

    if dataset_id not in files:
        raise Exception('Unknown dataset %s' % dataset_id)

    base_url = 'http://openfmri.s3.amazonaws.com/tarballs/%s.tgz'
    urls = [(dataset_id, base_url % f, {'uncompress':True}) for f in files[dataset_id]]
    temp_dir = os.path.join(data_dir, '_%s' % dataset_id, dataset_id)
    output_dir = os.path.join(data_dir, dataset_id)
    print data_dir, urls
    if not os.path.exists(output_dir) and not force_download:
        _fetch_files(data_dir, urls, verbose=verbose)
    return output_dir
