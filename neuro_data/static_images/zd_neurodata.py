from collections import OrderedDict
from functools import partial
from itertools import compress
from pprint import pformat

import datajoint as dj
import numpy as np
import pandas as pd

from neuro_data import logger as log
from neuro_data.utils.data import h5cached, SplineCurve, FilterMixin, fill_nans, NaNSpline
from neuro_data.static_images import datasets, configs as neuro_configs

dj.config['external-data'] = {'protocol': 'file', 'location': '/external/'}

experiment = dj.create_virtual_module('experiment', 'pipeline_experiment')
reso = dj.create_virtual_module('reso', 'pipeline_reso')
meso = dj.create_virtual_module('meso', 'pipeline_meso')
fuse = dj.create_virtual_module('fuse', 'pipeline_fuse')
pupil = dj.create_virtual_module('pupil', 'pipeline_eye')
stimulus = dj.create_virtual_module('stimulus', 'pipeline_stimulus')
shared = dj.create_virtual_module('shared', 'pipeline_shared')
anatomy = dj.create_virtual_module('anatomy', 'pipeline_anatomy')
treadmill = dj.create_virtual_module('treadmill', 'pipeline_treadmill')
base = dj.create_virtual_module('neurostatic_base', 'neurostatic_base')
imagenet = dj.create_virtual_module('pipeline_imagenet', 'pipeline_imagenet')
loop = dj.create_virtual_module('neurostatic_zhiwei_loop', 'neurostatic_zhiwei_loop')

schema = dj.schema('zhiwei_neuro_data')

# set of attributes that uniquely identifies the frame content
UNIQUE_FRAME = {
    'stimulus.Frame': ('image_id', 'image_class'),
    'stimulus.MonetFrame': ('rng_seed', 'orientation'),
    'stimulus.TrippyFrame': ('rng_seed',),
    'stimulus.ColorFrameProjector': ('image_id', 'image_class'),
}

IMAGE_CLASSES = 'image_class in ("imagenet", "masked_oracle", "masked_single", "diverse_mei", "searched_nat", "mei2", "imagenet_v2_gray", "imagenet_v2_rgb", "gaudy_imagenet2")' # all valid natural image classes
ORACLE_CLASSES = 'image_class in ("imagenet", "masked_oracle", "gaudy_imagenet2")'
TRAINING_CLASSES = 'image_class in ("imagenet", "masked_single", "searched_nat", "diverse_mei", "mei2", "gaudy_imagenet2")'
MASKED_CLASSES = ['diverse_mei', 'mei2', 'masked_single']
FF_CLASSES = ['imagenet', 'searched_nat', 'gaudy_imagenet2']
    
@schema
class StaticScanCandidate(dj.Manual):
    definition = """ # list of scans to process
    
    -> fuse.ScanDone
    ---
    candidate_notes='' : varchar(1024)
    """
    @staticmethod
    def fill(key, candidate_notes='', segmentation_method=6, spike_method=5,
             pipe_version=1):
        """ Fill an entry with key"""
        StaticScanCandidate.insert1({'segmentation_method': segmentation_method,
                                     'spike_method': spike_method,
                                     'pipe_version': pipe_version, **key,
                                     'candidate_notes': candidate_notes},
                                    skip_duplicates=True)

@schema
class StaticScan(dj.Computed):
    definition = """ # gatekeeper for scan and preprocessing settings
    
    -> fuse.ScanDone
    """

    class Unit(dj.Part):
        definition = """
        # smaller primary key table for data
        -> master        
        unit_id              : int                          # unique per scan & segmentation method
        ---
        -> fuse.ScanSet.Unit
        """

    key_source = fuse.ScanDone() & StaticScanCandidate & 'spike_method=5 and segmentation_method=6'

    @staticmethod
    def complete_key(key):
        return dict((dj.U('segmentation_method', 'pipe_version') &
                     (meso.ScanSet.Unit() & key)).fetch1(dj.key), **key)

    def make(self, key):
        self.insert(fuse.ScanDone() & key, ignore_extra_fields=True)
        pipe = (fuse.ScanDone() & key).fetch1('pipe')
        pipe = dj.create_virtual_module(pipe, 'pipeline_' + pipe)
        self.Unit().insert(fuse.ScanDone * pipe.ScanSet.Unit * pipe.MaskClassification.Type & key
                           & dict(pipe_version=1, segmentation_method=6, spike_method=5, type='soma'),
                           ignore_extra_fields=True)


@schema
class Tier(dj.Lookup):
    definition = """
    tier        : varchar(20)
    ---
    """

    @property
    def contents(self):
        yield from zip(["train", "test", "test_gaudy_imagenet2", "validation"])


@schema
class ExcludedTrial(dj.Manual):
    definition = """
    # trials to be excluded from analysis
    -> stimulus.Trial
    ---
    exclusion_comment='': varchar(64)   # reasons for exclusion
    """


@schema
class ImageNetSubset(dj.Lookup):
    definition = """
    subset_id: int 
    --- 
    description: varchar(255)
    subset_by  : varchar(16)   # the criteria by which the subsets are divided
    num_repeats: int          # number of repeats for each image
    """
    class TrainClass(dj.Part):
        definition = """
        -> master
        class_id: int
        ---
        image_class: varchar(45)  # a specific image class
        """
    class ValidClass(dj.Part):
        definition = """
        -> master
        class_id: int
        ---
        image_class: varchar(45)  # a specific image class
        """
    class TestClass(dj.Part):
        definition = """
        -> master
        class_id: int
        ---
        image_class: varchar(45)  # a specific image class
        """
# contents = [[1, 'train on "imagenet" and validate on "imagenet"', ['imagenet'], ['imagenet'], ['imagenet']], 
#             [2, 'train on "imagenet" and validate on "imagenet" and test on mixed', ['imagenet'], ['imagenet'], ['imagenet', 'gaudy_imagenet2']],
#             [3, 'train on "gaudy_imagenet2" and validate on "gaudy_imagenet2" and test on mixed', ['gaudy_imagenet2'], ['gaudy_imagenet2'], ['imagenet', 'gaudy_imagenet2']],
#             [4, 'train on "gaudy_imagenet2" and validate on "imagenet" and test on mixed', ['gaudy_imagenet2'], ['imagenet'], ['imagenet', 'gaudy_imagenet2']],
#             [5, 'train on mixed and validate on mixed and test on mixed', ['gaudy_imagenet2', 'imagenet'], ['gaudy_imagenet2', 'imagenet'], ['imagenet', 'gaudy_imagenet2']],
#             [6, 'train on mixed and validate on "imagenet" and test on mixed', ['gaudy_imagenet2', 'imagenet'], ['imagenet'], ['imagenet', 'gaudy_imagenet2']]
#             [7, 'train and validate on single trial responses', 'trial_number', 1]
#             [8, 'train and validate on trial-average responses', 'trial_number', 10]]
    def fill(contents):
        for entry in contents:
            subset_id, des, train, valid, test = entry
            ImageNetSubset.insert1([subset_id, des], skip_duplicates=True)
            for i, ic in enumerate(train):
                ImageNetSubset.TrainClass.insert1([subset_id, i+1, ic], skip_duplicates=True)
            for i, ic in enumerate(valid):
                ImageNetSubset.ValidClass.insert1([subset_id, i+1, ic], skip_duplicates=True)
            for i, ic in enumerate(test):
                ImageNetSubset.TestClass.insert1([subset_id, i+1, ic], skip_duplicates=True)

@schema
class SubsetImageNetSplit(dj.Lookup):
    definition = """ # split frames in each subset of image_class combinations for train, validation, and test sets
    -> StaticScan
    -> ImageNetSubset
    ---
    n_unique_images:    int  # total number of unique images in this subset
    """
    class Image(dj.Part):
        definition = """ 
        -> master
        -> stimulus.StaticImage.Image
        ---
        -> Tier
        """
    def fill(self, scan_key, subset_id):
        """ Assign each frame in the specific subset of the current scan to train/validation/test_{image_class} sets.
        Arguments:
            scan_key: An scan (animal_id, session, scan_idx) that has stimulus.Trials
                created. Usually one where the stimulus was presented.
            subset_id: subset_id in ImageNetSubset
        """
        subset_by, num_repeats = (ImageNetSubset & {'subset_id': subset_id}).fetch1('subset_by', 'num_repeats')
        key = (StaticScan * ImageNetSubset & scan_key & {'subset_id': subset_id}).fetch1('KEY')

        # Find out whether we are using the old pipeline (grayscale only) or the new version
        if stimulus.Frame & (stimulus.Trial & scan_key):
            frame_table = stimulus.Frame
        elif stimulus.ColorFrameProjector & (stimulus.Trial & scan_key):
            frame_table = stimulus.ColorFrameProjector
        else:
            print('Static images were not shown for this scan')

        # Make sure there are more than 0 frames in this scan
        all_frames = frame_table * stimulus.Trial & scan_key & IMAGE_CLASSES
        unique_frames = dj.U('image_id', 'image_class').aggr(all_frames, repeats='COUNT(*)')
        num_frames = len(unique_frames)
        assert num_frames != 0, 'unique_frames == 0'

        if subset_by == 'image_class':
            # Separate single and oracle images
            n = int(np.median(unique_frames.fetch('repeats')))  # HACK
            oracle_rel = unique_frames & 'repeats > {}'.format(n)
            single_rel = unique_frames.proj() - oracle_rel.proj()
            num_images_per_class = int(np.floor(len(single_rel) / len(dj.U('image_class') & single_rel)))
            # This implementation assumes there are equal numbers of images in each train_class 

            # Fetch key and all image classes in this subset of this scan
            train_classes = (ImageNetSubset.TrainClass & {'subset_id': subset_id}).fetch('image_class')
            valid_classes = (ImageNetSubset.ValidClass & {'subset_id': subset_id}).fetch('image_class')
            test_classes = (ImageNetSubset.TestClass & {'subset_id': subset_id}).fetch('image_class')
            self.insert1({**key, 'n_unique_images': len(oracle_rel)+num_images_per_class})

            # Insert test set images (one sub test set for each image_class in test_classes in this subset)
            for ic in test_classes:
                test_ids = (oracle_rel & {'image_class': ic}).fetch('image_id')
                if ic == 'imagenet':
                    self.Image.insert([{**key, 'image_id': iid, 'image_class': ic, 'tier': 'test'} for iid in test_ids])
                else: 
                    self.Image.insert([{**key, 'image_id': iid, 'image_class': ic, 'tier': 'test_{}'.format(ic)} for iid in test_ids])

            # Insert training and validation set images
            if len(train_classes) == 1:  # Assuming if there is only one train_class, then there is also only one valid_class
                train_ids = (single_rel & {'image_class': train_classes[0]}).fetch('image_id', order_by='repeats')
                # * NOTE: this fetches all images first and the rest in a "random" order;
                num_validation = int(np.ceil(len(train_ids) * 0.1))  # 10% validation examples
                self.Image.insert([{**key, 'image_id': iid, 'image_class': valid_classes[0], 'tier': 'validation'} for iid in train_ids[:num_validation]])
                self.Image.insert([{**key, 'image_id': iid, 'image_class': train_classes[0], 'tier': 'train'} for iid in train_ids[num_validation:]])

            elif len(train_classes) > 1: # When there are multiple train_class, there can be one or multiple valid_class
                all_selected = []
                for i, ic in enumerate(train_classes):
                    train_ids = (single_rel & {'image_class': ic}).fetch('image_id', order_by='repeats')
                    assert len(train_ids) == num_images_per_class, 'Unequal number of images in each image class!'

                    num_selected_per_class = int(np.floor(num_images_per_class / len(train_classes)))
                    # Select non-overlapping image_ids for different train_classes
                    np.random.seed(0)
                    if i == 0:
                        selected = np.random.choice(train_ids, num_selected_per_class, replace=False).tolist()
                    else:
                        rest = list(set(train_ids) - set(selected))
                        selected = np.random.choice(rest, num_selected_per_class, replace=False).tolist()
                        all_selected = all_selected + selected
                    
                    num_validation = int(np.ceil(len(selected) * 0.1))  # 10% validation examples

                    if len(valid_classes) == 1:
                        self.Image.insert([{**key, 'image_id': iid, 'image_class': valid_classes[0], 'tier': 'validation'} for iid in selected[:num_validation]])
                    else:  # Assumes if there are multiple valid_classes, then they are the same as train_classes (also in the same order)
                        self.Image.insert([{**key, 'image_id': iid, 'image_class': valid_classes[i], 'tier': 'validation'} for iid in selected[:num_validation]])
                    self.Image.insert([{**key, 'image_id': iid, 'image_class': ic, 'tier': 'train'} for iid in selected[num_validation:]])

        elif subset_by == 'trial_number':
            # get oracle images
            oracle_rel = stimulus.Trial * stimulus.Frame & scan_key & (imagenet.Album.Oracle() & 'image_class = "imagenet" and collection_id = 2')
            unique_oracle = dj.U('image_class', 'image_id') & oracle_rel
            test_image_classes, test_image_ids = (unique_oracle).fetch('image_class', 'image_id')
            num_oracles = len(unique_oracle)

            # split non-oracle images
            non_oracles = unique_frames - unique_oracle

            if num_repeats == 1:
                # Compute number of validation examples
                num_validation = int(np.ceil(len(non_oracles) * 0.1))  # 10% validation examples
                single_image_classes, single_image_ids = (non_oracles & 'repeats = 1').fetch('image_class', 'image_id', order_by='repeats DESC')
                repeat_image_classes, repeat_image_ids = (non_oracles & 'repeats > 1').fetch('image_class', 'image_id')
                self.insert1({**key, 'n_unique_images': len(unique_frames)})
                self.Image.insert([{**key, 'image_id': iid, 'image_class': ic, 'tier': 'test'} for iid, ic in
                                        zip(test_image_ids, test_image_classes)])
                self.Image.insert([{**key, 'image_id': iid, 'image_class': ic, 'tier': 'validation'} for iid, ic in 
                                zip(single_image_ids[:num_validation], single_image_classes[:num_validation])])
                self.Image.insert([{**key, 'image_id': iid, 'image_class': ic, 'tier': 'train'} for iid, ic in
                                zip(single_image_ids[num_validation:], single_image_classes[num_validation:])])
                self.Image.insert([{**key, 'image_id': iid, 'image_class': ic, 'tier': 'train'} for iid, ic in
                                        zip(repeat_image_ids, repeat_image_classes)])
            else:
                # Compute number of validation examples
                num_validation = int(np.ceil(len(non_oracles & 'repeats > 1') * 0.1))  # 10% validation examples
                repeat_image_classes, repeat_image_ids = (non_oracles & 'repeats > 1').fetch('image_class', 'image_id', order_by='repeats DESC')
                self.insert1({**key, 'n_unique_images': len(unique_frames & 'repeats > 1')})
                self.Image.insert([{**key, 'image_id': iid, 'image_class': ic, 'tier': 'test'} for iid, ic in
                                                    zip(test_image_ids, test_image_classes)])
                self.Image.insert([{**key, 'image_id': iid, 'image_class': ic, 'tier': 'validation'} for iid, ic in 
                                zip(repeat_image_ids[:num_validation], repeat_image_classes[:num_validation])])
                self.Image.insert([{**key, 'image_id': iid, 'image_class': ic, 'tier': 'train'} for iid, ic in
                                zip(repeat_image_ids[num_validation:], repeat_image_classes[num_validation:])])


@schema
class ConditionTier(dj.Computed):
    definition = """
    # split into train, test, validation, and test_masked (if masked oracles are used)

    -> stimulus.Condition
    -> StaticScan
    -> ImageNetSubset
    ---
    -> Tier
    """

    @property
    def dataset_compositions(self):
        return dj.U('animal_id', 'session', 'scan_idx', 'stimulus_type', 'tier').aggr(
            self * stimulus.Condition(), n='count(*)')

    @property
    def key_source(self):
        # all static scan with at least on recorded trial
        return StaticScan() * ImageNetSubset & stimulus.Trial()

    def check_train_test_split(self, frames, cond):
        stim = getattr(stimulus, cond['stimulus_type'].split('.')[-1])
        train_test = (dj.U(*UNIQUE_FRAME[cond['stimulus_type']]).aggr(frames * stim,
                                                                      train='sum(1-test)',
                                                                      test='sum(test)') &
                      'train>0 and test>0')
        assert len(train_test) == 0, 'Train and test clips do overlap'

    def fill_up(self, tier, frames, cond, key, m):
        existing = ConditionTier().proj() & (self & dict(tier=tier)) \
                   & (stimulus.Trial() * stimulus.Condition() & dict(key, **cond))
        n = len(existing)
        if n < m:
            # all hashes that are in clips but not registered for that animal and have the right tier
            candidates = dj.U('condition_hash') & \
                         (self & (dj.U('condition_hash') & (frames - self)) & dict(tier=tier))
            keys = candidates.fetch(dj.key)
            d = m - n
            update = min(len(keys), d)

            log.info('Inserting {} more existing {} trials'.format(update, tier))
            for k in keys[:update]:
                k = (frames & k).fetch1(dj.key)
                k['tier'] = tier
                self.insert1(k, ignore_extra_fields=True)

        existing = ConditionTier().proj() & (self & dict(tier=tier)) \
                   & (stimulus.Trial() * stimulus.Condition() & dict(key, **cond))
        n = len(existing)
        if n < m:
            keys = (frames - self).fetch(dj.key)
            update = m - n
            log.info('Inserting {} more new {} trials'.format(update, tier))

            for k in keys[:update]:
                k['tier'] = tier
                self.insert1(k, ignore_extra_fields=True)

    def make(self, key):
        log.info(80 * '-')
        log.info('Processing ' + pformat(key))
        # count the number of distinct conditions presented for each one of three stimulus types:
        # "stimulus.Frame","stimulus.MonetFrame", "stimulus.TrippyFrame"
        conditions = dj.U('stimulus_type').aggr(stimulus.Condition() & (stimulus.Trial() & key),
                                                count='count(*)') \
                     & 'stimulus_type in ("stimulus.Frame", "stimulus.MonetFrame", "stimulus.TrippyFrame", "stimulus.ColorFrameProjector")'
        for cond in conditions.fetch(as_dict=True):
            # hack for compatibility with previous datasets
            if cond['stimulus_type'] in ['stimulus.Frame', 'stimulus.ColorFrameProjector']:
                frame_table = (stimulus.Frame if cond['stimulus_type'] == 'stimulus.Frame' else stimulus.ColorFrameProjector)
                    
                # deal with ImageNet frames first
                log.info('Inserting assignment from ImageNetSplit')
                targets = StaticScan * frame_table * SubsetImageNetSplit.Image & (stimulus.Trial & key) & IMAGE_CLASSES
                print('Inserting {} imagenet conditions!'.format(len(targets)))
                self.insert(targets, ignore_extra_fields=True)
                
                # deal with MEI images, assigning tier test for all images
                assignment = (frame_table & 'image_class in ("cnn_mei", "lin_rf", "multi_cnn_mei", "multi_lin_rf")').proj(tier='"train"')
                self.insert(StaticScan * frame_table * assignment & (stimulus.Trial & key), ignore_extra_fields=True)

                # make sure that all frames were assigned
                remaining = (stimulus.Trial * frame_table & key) - self
                assert len(remaining) == 0, 'There are still unprocessed Frames'
                
                # make sure there is no overlap between train and test set
                log.info('Checking condition {stimulus_type} (n={count})'.format(**cond))
                frames = (stimulus.Condition() * StaticScan() & key & cond).aggr(stimulus.Trial(), repeats="count(*)",
                                                                             test='count(*) > 4')
                self.check_train_test_split(frames, cond)
                continue
                
            log.info('Checking condition {stimulus_type} (n={count})'.format(**cond))
            frames = (stimulus.Condition() * StaticScan() & key & cond).aggr(stimulus.Trial(), repeats="count(*)",
                                                                             test='count(*) > 4')
            self.check_train_test_split(frames, cond)

            m = len(frames)
            m_test = m_val = len(frames & 'test > 0') or max(m * 0.075, 1)
            log.info('Minimum test and validation set size will be {}'.format(m_test))
            log.info('Processing test conditions')

            # insert repeats as test trials
            self.insert((frames & dict(test=1)).proj(tier='"test"'), ignore_extra_fields=True)
            self.fill_up('test', frames, cond, key, m_test)

            log.info('Processing validation conditions')
            self.fill_up('validation', frames, cond, key, m_val)

            log.info('Processing training conditions')
            self.fill_up('train', frames, cond, key, m - m_test - m_val)
                
            
@schema
class Preprocessing(dj.Lookup):
    definition = """
    # settings for movie preprocessing

    preproc_id       : tinyint # preprocessing ID
    ---
    offset           : decimal(6,4) # offset to stimulus onset in s
    duration         : decimal(6,4) # window length in s
    row              : smallint     # row size of movies
    col              : smallint     # col size of movie
    filter           : varchar(24)  # filter type for window extraction
    gamma            : boolean      # whether to convert images to luminance values rather than pixel intensities
    """
    contents = [
        {'preproc_id': 0, 'offset': 0.05, 'duration': 0.5, 'row': 36, 'col': 64,
         'filter': 'hamming', 'gamma': False},  # this one was still processed with cropping
        {'preproc_id': 1, 'offset': 0.05, 'duration': 0.5, 'row': 36, 'col': 64,
         'filter': 'hamming', 'gamma': False},
        {'preproc_id': 2, 'offset': 0.05, 'duration': 0.5, 'row': 72, 'col': 128,
         'filter': 'hamming', 'gamma': False},
        {'preproc_id': 3, 'offset': 0.05, 'duration': 0.5, 'row': 36, 'col': 64,
         'filter': 'hamming', 'gamma': True},
    ]


def process_frame(preproc_key, frame):
    """
    Helper function that preprocesses a frame
    """
    import cv2
    imgsize = (Preprocessing() & preproc_key).fetch1('col', 'row')  # target size of movie frames
    log.info('Downsampling frame')
    if not frame.shape[0] / imgsize[1] == frame.shape[1] / imgsize[0]:
        log.warning('Image size would change aspect ratio.')

    return cv2.resize(frame, imgsize, interpolation=cv2.INTER_AREA).astype(np.float32)


@schema
class Frame(dj.Computed):
    definition = """ # frames downsampled

    -> stimulus.Condition
    -> Preprocessing
    ---
    frame                : external-data   # frame processed
    """

    @property
    def key_source(self):
        return stimulus.Condition() * Preprocessing() & ConditionTier()

    @staticmethod
    def load_frame(key):
        if stimulus.Frame & key:
            assert (stimulus.Frame & key).fetch1('pre_blank_period') > 0, 'we assume blank periods'
            return (stimulus.StaticImage.Image & (stimulus.Frame & key)).fetch1('image')
        elif stimulus.MonetFrame & key:
            assert (stimulus.MonetFrame & key).fetch1('pre_blank_period') > 0, 'we assume blank periods'
            return (stimulus.MonetFrame & key).fetch1('img')
        elif stimulus.TrippyFrame & key:
            assert (stimulus.TrippyFrame & key).fetch1('pre_blank_period') > 0, 'we assume blank periods'
            return (stimulus.TrippyFrame & key).fetch1('img')
        elif stimulus.ColorFrameProjector & key:
            # stimulus is type ColorFrameProjector which means we need to look up what channel was map to what and select base on
            assert (stimulus.ColorFrameProjector & key).fetch1('pre_blank_period') > 0, 'we assume blank periods'

            original_img = (stimulus.StaticImage.Image & (stimulus.ColorFrameProjector & key)).fetch1('image')
            if len(original_img.shape) == 2:
                # Only 1 channel
                return original_img
            else:
                # There is more then 1 channel, thus we need get the channel mappings for the project, where the number signifies which RGB channel maps to the project channels
                channel_mappings = (stimulus.ColorFrameProjector() & key).fetch1('channel_1', 'channel_2', 'channel_3')
                image_sub_channels_to_include = []
                for channel_mapping in channel_mappings:
                    if channel_mapping is not None:
                        image_sub_channels_to_include.append(original_img[:, :, channel_mapping - 1])
                return np.stack(image_sub_channels_to_include, axis=-1)
        else:
            raise KeyError('Cannot find matching stimulus relation')

    @staticmethod
    def get_stimulus_type(scan_key):
        """
        Function that returns a list of str indicating what stimulus_types are in the given condition_hash

        Args:
            scan_key (dict): A key that contains animial_id, session, scan_idx, pipe_version, segmentation_method, and spike_method. Most of the time the first 3 attributes are sufficient
        
        Returns:
            stimulus_types (list<str>): A list of string containing the stimulus_type name(s)
        """
        
        key = ConditionTier & scan_key
        stimulus_types = []

        if stimulus.Frame & key:
            stimulus_types.append('stimulus.Frame')
        if stimulus.MonetFrame & key:
            stimulus_types.append('stimulus.MonetFrame')
        if stimulus.TrippyFrame & key:
            stimulus_types.append('stimulus.TrippyFrame')
        if stimulus.ColorFrameProjector & key:
            stimulus_types.append('stimulus.ColorFrameProjector')

        return stimulus_types

    def make(self, key):
        log.info(80 * '-')
        log.info('Processing key ' + pformat(dict(key)))

        # get original frame
        frame = self.load_frame(key)

        # preprocess the frame
        frame = process_frame(key, frame)

        # --- generate response sampling points and sample movie frames relative to it
        self.insert1(dict(key, frame=frame))

@h5cached('/external/cache/', mode='array', transfer_to_tmp=False,
          file_format='static{animal_id}-{session}-{scan_idx}-preproc{preproc_id}-subset{subset_id}.h5')
@schema
class InputResponse(dj.Computed, FilterMixin):
    definition = """
    # responses of one neuron to the stimulus

    -> StaticScan
    -> Preprocessing
    ---
    """

    key_source = StaticScan * Preprocessing & Frame

    class Input(dj.Part):
        definition = """
            -> master
            -> stimulus.Trial
            -> Frame
            ---
            row_id           : int             # row id in the response block
            """

    class ResponseBlock(dj.Part):
        definition = """
            -> master
            ---
            responses           : external-data   # response of one neurons for all bins
            """

    class ResponseKeys(dj.Part):
        definition = """
            -> master.ResponseBlock
            -> fuse.Activity.Trace
            ---
            col_id           : int             # col id in the response block
            """

    def load_traces_and_frametimes(self, key):
        # -- find number of recording depths
        pipe = (fuse.Activity() & key).fetch('pipe')
        assert len(np.unique(pipe)) == 1, 'Selection is from different pipelines'
        pipe = dj.create_virtual_module(pipe[0], 'pipeline_' + pipe[0])
        k = dict(key)
        k.pop('field', None)
        ndepth = len(dj.U('z') & (pipe.ScanInfo.Field() & k))
        frame_times = (stimulus.Sync() & key).fetch1('frame_times').squeeze()[::ndepth]

        soma = pipe.MaskClassification.Type() & dict(type='soma')

        spikes = (dj.U('field', 'channel') * pipe.Activity.Trace() * StaticScan.Unit() \
                  * pipe.ScanSet.UnitInfo() & soma & key)
        traces, ms_delay, trace_keys = spikes.fetch('trace', 'ms_delay', dj.key,
                                                    order_by='animal_id, session, scan_idx, unit_id')
        delay = np.fromiter(ms_delay / 1000, dtype=np.float)
        frame_times = (delay[:, None] + frame_times[None, :])
        traces = np.vstack([fill_nans(tr.astype(np.float32)).squeeze() for tr in traces])
        traces, frame_times = self.adjust_trace_len(traces, frame_times)
        return traces, frame_times, trace_keys

    def adjust_trace_len(self, traces, frame_times):
        trace_len, nframes = traces.shape[1], frame_times.shape[1]
        if trace_len < nframes:
            frame_times = frame_times[:, :trace_len]
        elif trace_len > nframes:
            traces = traces[:, :nframes]
        return traces, frame_times

    def get_trace_spline(self, key, sampling_period):
        traces, frame_times, trace_keys = self.load_traces_and_frametimes(key)
        log.info('Loaded {} traces'.format(len(traces)))

        log.info('Generating lowpass filters to {}Hz'.format(1 / sampling_period))
        h_trace = self.get_filter(sampling_period, np.median(np.diff(frame_times)), 'hamming',
                                  warning=False)
        # low pass filter
        trace_spline = SplineCurve(frame_times,
                                   [np.convolve(trace, h_trace, mode='same') for trace in traces], k=1, ext=1)
        return trace_spline, trace_keys, frame_times.min(), frame_times.max()

    @staticmethod
    def stimulus_onset(flip_times, duration):
        n_ft = np.unique([ft.size for ft in flip_times])
        assert len(n_ft) == 1, 'Found inconsistent number of fliptimes'
        n_ft = int(n_ft)
        log.info('Found {} flip times'.format(n_ft))

        assert n_ft in (2, 3), 'Cannot deal with {} flip times'.format(n_ft)

        stimulus_onset = np.vstack(flip_times)  # columns correspond to  clear flip, onset flip
        ft = stimulus_onset[np.argsort(stimulus_onset[:, 0])]
        if n_ft == 2:
            assert np.median(ft[1:, 0] - ft[:-1, 1]) < duration + 0.05, 'stimulus duration off by more than 50ms'
        else:
            assert np.median(ft[:, 2] - ft[:, 1]) < duration + 0.05, 'stimulus duration off by more than 50ms'
        stimulus_onset = stimulus_onset[:, 1]

        return stimulus_onset

    def make(self, scan_key):
        self.insert1(scan_key)
        # integration window size for responses
        duration, offset = map(float, (Preprocessing() & scan_key).fetch1('duration', 'offset'))
        sample_point = offset + duration / 2

        log.info('Sampling neural responses at {}s intervals'.format(duration))

        trace_spline, trace_keys, ftmin, ftmax = self.get_trace_spline(scan_key, duration)
        # exclude trials marked in ExcludedTrial
        log.info('Excluding {} trials based on ExcludedTrial'.format(len(ExcludedTrial() & scan_key)))
        flip_times, trial_keys = (Frame * (stimulus.Trial - ExcludedTrial) & scan_key).fetch('flip_times', dj.key,
                                                                           order_by='condition_hash')
        flip_times = [ft.squeeze() for ft in flip_times]

        # If no Frames are present, skip this scan
        if len(flip_times) == 0:
            log.warning('No static frames were present to be processed for {}'.format(scan_key))
            return

        valid = np.array([ft.min() >= ftmin and ft.max() <= ftmax for ft in flip_times], dtype=bool)
        if not np.all(valid):
            log.warning('Dropping {} trials with dropped frames or flips outside the recording interval'.format(
                (~valid).sum()))

        stimulus_onset = self.stimulus_onset(flip_times, duration)
        log.info('Sampling {} responses {}s after stimulus onset'.format(valid.sum(), sample_point))
        R = trace_spline(stimulus_onset[valid] + sample_point, log=True).T

        self.ResponseBlock.insert1(dict(scan_key, responses=R))
        self.ResponseKeys.insert([dict(scan_key, **trace_key, col_id=i) for i, trace_key in enumerate(trace_keys)])
        self.Input.insert([dict(scan_key, **trial_key, row_id=i)
                           for i, trial_key in enumerate(compress(trial_keys, valid))])

    def compute_data(self, key):
        key = dict((self & key).fetch1(dj.key), **key)
        subset_by, num_repeats = (ImageNetSubset & key).fetch1('subset_by', 'num_repeats')

        log.info('Computing dataset for\n' + pformat(key, indent=20))

        # meso or reso?
        pipe = (fuse.ScanDone() * StaticScan() & key).fetch1('pipe')
        pipe = dj.create_virtual_module(pipe, 'pipeline_' + pipe)

        # get data relation
        include_behavior = bool(Eye.proj() * Treadmill.proj() & key)

        assert include_behavior, 'Behavior data is missing!'

        # make sure that including areas and layers does not decrease number of neurons
        assert len(pipe.ScanSet.UnitInfo() * experiment.Layer() * anatomy.AreaMembership() * anatomy.LayerMembership() & key) == \
               len(pipe.ScanSet.UnitInfo() & key), "AreaMembership decreases number of neurons"

        responses = (self.ResponseBlock & key).fetch1('responses')
        trials = Frame() * ConditionTier() * self.Input() * stimulus.Condition().proj('stimulus_type') & key
        if subset_by == 'trial_number':
            hashes, trial_idxs, tiers, types, images, row_ids = trials.fetch('condition_hash', 'trial_idx', 'tier',
                                                                'stimulus_type', 'frame', 'row_id', order_by='condition_hash, row_id')
        else:
            hashes, trial_idxs, tiers, types, images, row_ids = trials.fetch('condition_hash', 'trial_idx', 'tier',
                                                                'stimulus_type', 'frame', 'row_id', order_by='row_id')
        images = np.stack(images)

        # select a subset of trials
        responses = responses[row_ids]

        if len(images.shape) == 3:
            log.info('Adding channel dimension')
            images = images[:, None, ...]
        elif len(images.shape) == 4:
            images = images.transpose(0, 3, 1, 2)
        hashes = hashes.astype(str)
        types = types.astype(str)

        # gamma correction
        if (Preprocessing & key).fetch1('gamma'):
            log.info('Gamma correcting images.')
            from staticnet_analyses import multi_mei

            if len(multi_mei.ClosestCalibration & key) == 0:
                raise ValueError('No ClosestMonitorCalibration for this scan.')
            f, f_inv = (multi_mei.ClosestCalibration & key).get_fs()
            images = f(images)


        # --- extract infomation for each trial
        extra_info = pd.DataFrame({'condition_hash':hashes, 'trial_idx':trial_idxs})
        dfs = OrderedDict()

        # add information about each stimulus
        for t in map(lambda x: x.split('.')[1], np.unique(types)):
            stim = getattr(stimulus, t)
            rel = stim() * stimulus.Trial() & key
            df = pd.DataFrame(rel.proj(*rel.heading.non_blobs).fetch())
            dfs[t] = df

        on = ['animal_id', 'condition_hash', 'scan_idx', 'session', 'trial_idx']
        for t, df in dfs.items():
            mapping = {c:(t.lower() + '_' + c) for c in set(df.columns) - set(on)}
            dfs[t] = df.rename(columns=mapping)
        df = list(dfs.values())[0]
        for d in list(dfs.values())[1:]:
            df = df.merge(d, how='outer', on=on)
        extra_info = extra_info.merge(df, on=['condition_hash','trial_idx']) # align rows to existing data
        assert len(extra_info) == len(trial_idxs), 'Extra information changes in length'
        assert np.all(extra_info['condition_hash'] == hashes), 'Hash order changed'
        assert np.all(extra_info['trial_idx'] == trial_idxs), 'Trial idx order changed'
        row_info = {}

        for k in extra_info.columns:
            dt = extra_info[k].dtype
            if isinstance(extra_info[k][0], str):
                row_info[k] = np.array(extra_info[k], dtype='S')
            elif dt == np.dtype('O') or dt == np.dtype('<M8[ns]'):
                row_info[k] = np.array(list(map(repr, extra_info[k])), dtype='S')
            else:
                row_info[k] = np.array(extra_info[k])

        # extract behavior
        if include_behavior:
            pupil, dpupil, pupil_center, valid_eye = (Eye & key).fetch1('pupil', 'dpupil', 'center', 'valid')
            pupil = pupil[row_ids]
            dpupil = dpupil[row_ids]
            pupil_center = pupil_center.T[row_ids]
            valid_eye = valid_eye[row_ids]

            treadmill, valid_treadmill = (Treadmill & key).fetch1('treadmill', 'valid')
            treadmill = treadmill[row_ids]
            valid_treadmill = valid_treadmill[row_ids]
            valid = valid_eye & valid_treadmill
            if np.any(~valid):
                log.warning('Found {} invalid trials. Reducing data.'.format((~valid).sum()))
                hashes = hashes[valid]
                images = images[valid]
                responses = responses[valid]
                trial_idxs = trial_idxs[valid]
                tiers = tiers[valid]
                types = types[valid]
                pupil = pupil[valid]
                dpupil = dpupil[valid]
                pupil_center = pupil_center[valid]
                treadmill = treadmill[valid]
                for k in row_info:
                    row_info[k] = row_info[k][valid]
            behavior = np.c_[pupil, dpupil, treadmill]

        areas, layers, animal_ids, sessions, scan_idxs, unit_ids = (self.ResponseKeys
                                                                    * anatomy.AreaMembership
                                                                    * anatomy.LayerMembership & key).fetch('brain_area',
                                                                                                           'layer',
                                                                                                           'animal_id',
                                                                                                           'session',
                                                                                                           'scan_idx',
                                                                                                           'unit_id',
                                                                                                           order_by='col_id ASC')

        assert len(np.unique(unit_ids)) == len(unit_ids), \
            'unit ids are not unique, do you have more than one preprocessing method?'

        neurons = dict(
            unit_ids=unit_ids.astype(np.uint16),
            animal_ids=animal_ids.astype(np.uint16),
            sessions=sessions.astype(np.uint8),
            scan_idx=scan_idxs.astype(np.uint8),
            layer=layers.astype('S'),
            area=areas.astype('S')
        )
            
        def run_stats(selector, types, ix, per_image=False, axis=None):

            ret = {}
            for t in np.unique(types):
                if not np.any(ix & (types == t)):
                    continue
                data = selector(ix & (types == t))
                if per_image:
                    mean = np.mean(data.mean(axis=(2,3)))
                    std = np.mean(data.std(axis=(2, 3)))
                else:
                    mean = data.mean(axis=axis)
                    std = data.std(axis=axis)
                ret[t] = dict(
                    mean=mean.astype(np.float32),
                    std=std.astype(np.float32),
                    min=data.min(axis=axis).astype(np.float32),
                    max=data.max(axis=axis).astype(np.float32),
                    median=np.median(data, axis=axis).astype(np.float32)
                )
            data = selector(ix)
            ret['all'] = dict(
                mean=mean.astype(np.float32),
                std=std.astype(np.float32),
                min=data.min(axis=axis).astype(np.float32),
                max=data.max(axis=axis).astype(np.float32),
                median=np.median(data, axis=axis).astype(np.float32)
            )
            return ret

        # average across trial repeats if necessary 
        if subset_by == 'trial_number':
            if num_repeats == 1:
                test_idxs = np.where(tiers == 'test')[0]
                nontest_idxs = np.where(tiers != 'test')[0]
                # find unique image idxs of nontest images
                unique_nontest_idxs = nontest_idxs[np.unique(hashes[nontest_idxs], return_index=True)[1]]
                # find idxs of unique nontest images, and all test images, then sort them by idx
                unique_image_idxs = np.array(sorted(np.concatenate([test_idxs, unique_nontest_idxs])))

                responses = responses[unique_image_idxs]
                behavior = behavior[unique_image_idxs]
                pupil_center = pupil_center[unique_image_idxs]

            if num_repeats > 1:
                df = pd.DataFrame(row_info)
                resp_df = pd.concat([df[['condition_hash']], pd.DataFrame(responses)], axis=1)
                mean_resp_df = resp_df.groupby('condition_hash', sort=False).mean()
                responses = mean_resp_df.to_numpy()

                beh_df = pd.concat([df[['condition_hash']], pd.DataFrame(behavior)], axis=1)
                mean_beh_df = beh_df.groupby('condition_hash', sort=False).mean()
                behavior = mean_beh_df.to_numpy()

                pupil_df = pd.concat([df[['condition_hash']], pd.DataFrame(pupil_center)], axis=1)
                mean_pupil_df = pupil_df.groupby('condition_hash', sort=False).mean()
                pupil_center = mean_pupil_df.to_numpy()
            
                unique_image_idxs = np.array(sorted(np.unique(hashes, return_index=True)[1]))

            images = images[unique_image_idxs]
            types = types[unique_image_idxs]
            hashes = hashes[unique_image_idxs]
            tiers = tiers[unique_image_idxs]
            trial_idxs = trial_idxs[unique_image_idxs] # only save the trial_idx of first occurance of each unique image 
            for k in row_info:
                row_info[k] = row_info[k][unique_image_idxs]
                
        # --- compute statistics
        log.info('Computing statistics on training dataset')
        response_statistics = run_stats(lambda ix: responses[ix], types, tiers == 'train', axis=0)
        if (Preprocessing & key).fetch1('trainstats_per_image'):
            log.info('Computing training input statistics by averaging across the values computed for individual images')
            input_statistics = run_stats(lambda ix: images[ix], types, tiers == 'train', per_image=True)
        else:
            log.info('Computing training input statistics across all values of all images')
            input_statistics = run_stats(lambda ix: images[ix], types, tiers == 'train')

        statistics = dict(
            images=input_statistics,
            responses=response_statistics
        )

        if include_behavior:
            # ---- include statistics
            behavior_statistics = run_stats(lambda ix: behavior[ix], types, tiers == 'train', axis=0)
            eye_statistics = run_stats(lambda ix: pupil_center[ix], types, tiers == 'train', axis=0)

            statistics['behavior'] = behavior_statistics
            statistics['pupil_center'] = eye_statistics


        retval = dict(images=images,
                      responses=responses,
                      types=types.astype('S'),
                      subset_id=np.full(len(types), key['subset_id']).astype(np.uint32),
                      condition_hashes=hashes.astype('S'),
                      trial_idx=trial_idxs.astype(np.uint32),
                      neurons=neurons,
                      item_info=row_info,
                      tiers=tiers.astype('S'),
                      statistics=statistics
                      )
        if include_behavior:
            retval['behavior'] = behavior
            retval['pupil_center'] = pupil_center
            
        return retval


class BehaviorMixin:
    def load_frame_times(self, key):
        pipe = (fuse.Activity() & key).fetch('pipe')
        assert len(np.unique(pipe)) == 1, 'Selection is from different pipelines'
        pipe = dj.create_virtual_module(pipe[0], 'pipeline_' + pipe[0])
        k = dict(key)
        k.pop('field', None)
        ndepth = len(dj.U('z') & (pipe.ScanInfo.Field() & k))
        return (stimulus.Sync() & key).fetch1('frame_times').squeeze()[::ndepth]

    def load_eye_traces(self, key):
        #r, center = (pupil.FittedPupil.Ellipse() & key).fetch('major_r', 'center', order_by='frame_id ASC')
        r, center = (pupil.FittedPupil.Circle() & key).fetch('radius', 'center',
                                                             order_by='frame_id')
        detectedFrames = ~np.isnan(r)
        xy = np.full((len(r), 2), np.nan)
        xy[detectedFrames, :] = np.vstack(center[detectedFrames])
        xy = np.vstack(map(partial(fill_nans, preserve_gap=3), xy.T))
        if np.any(np.isnan(xy)):
            log.info('Keeping some nans in the pupil location trace')
        pupil_radius = fill_nans(r.squeeze(), preserve_gap=3)
        if np.any(np.isnan(pupil_radius)):
            log.info('Keeping some nans in the pupil radius trace')

        eye_time = (pupil.Eye() & key).fetch1('eye_time').squeeze()
        return pupil_radius, xy, eye_time

    def load_behavior_timing(self, key):
        log.info('Loading behavior frametimes')
        # -- find number of recording depths
        pipe = (fuse.Activity() & key).fetch('pipe')
        assert len(np.unique(pipe)) == 1, 'Selection is from different pipelines'
        pipe = dj.create_virtual_module(pipe[0], 'pipeline_' + pipe[0])
        k = dict(key)
        k.pop('field', None)
        ndepth = len(dj.U('z') & (pipe.ScanInfo.Field() & k))
        return (stimulus.BehaviorSync() & key).fetch1('frame_times').squeeze()[0::ndepth]

    def load_treadmill_velocity(self, key):
        t, v = (treadmill.Treadmill() & key).fetch1('treadmill_time', 'treadmill_vel')
        return v.squeeze(), t.squeeze()


@schema
class Eye(dj.Computed, FilterMixin, BehaviorMixin):
    definition = """
    # eye movement data

    -> InputResponse
    ---
    -> pupil.FittedPupil                 # tracking_method as a secondary attribute
    pupil              : external-data   # pupil dilation trace
    dpupil             : external-data   # derivative of pupil dilation trace
    center             : external-data   # center position of the eye
    valid              : external-data   # valid trials
    """

    @property
    def key_source(self):
        return InputResponse & pupil.FittedPupil & stimulus.BehaviorSync

    def make(self, scan_key):
        scan_key = {**scan_key, 'tracking_method': 2}
        log.info('Populating '+ pformat(scan_key))
        radius, xy, eye_time = self.load_eye_traces(scan_key)
        frame_times = self.load_frame_times(scan_key)
        behavior_clock = self.load_behavior_timing(scan_key)

        if len(frame_times) - len(behavior_clock) != 0:
            assert abs(len(frame_times) - len(behavior_clock)) < 2, 'Difference bigger than 2 time points'
            l = min(len(frame_times), len(behavior_clock))
            log.info('Frametimes and stimulus.BehaviorSync differ in length! Shortening it.')
            frame_times = frame_times[:l]
            behavior_clock = behavior_clock[:l]

        fr2beh = NaNSpline(frame_times, behavior_clock, k=1, ext=3)

        duration, offset = map(float, (Preprocessing() & scan_key).fetch1('duration', 'offset'))
        sample_point = offset + duration / 2

        log.info('Downsampling eye signal to {}Hz'.format(1 / duration))
        deye = np.nanmedian(np.diff(eye_time))
        h_eye = self.get_filter(duration, deye, 'hamming', warning=True)
        h_deye = self.get_filter(duration, deye, 'dhamming', warning=True)
        pupil_spline = NaNSpline(eye_time,
                                 np.convolve(radius, h_eye, mode='same'), k=1, ext=0)

        dpupil_spline = NaNSpline(eye_time,
                                  np.convolve(radius, h_deye, mode='same'), k=1, ext=0)
        center_spline = SplineCurve(eye_time,
                                    np.vstack([np.convolve(coord, h_eye, mode='same') for coord in xy]),
                                    k=1, ext=0)

        flip_times = (InputResponse.Input * Frame * stimulus.Trial & scan_key).fetch('flip_times',
                                                                                     order_by='row_id ASC')

        flip_times = [ft.squeeze() for ft in flip_times]

        # If no Frames are present, skip this scan
        if len(flip_times) == 0:
            log.warning('No static frames were present to be processed for {}'.format(scan_key))
            return

        stimulus_onset = InputResponse.stimulus_onset(flip_times, duration)
        t = fr2beh(stimulus_onset + sample_point)
        pupil = pupil_spline(t)
        dpupil = dpupil_spline(t)
        center = center_spline(t)
        valid = ~np.isnan(pupil + dpupil + center.sum(axis=0))
        if not np.all(valid):
            log.warning('Found {} NaN trials. Setting to -1'.format((~valid).sum()))
            pupil[~valid] = -1
            dpupil[~valid] = -1
            center[:, ~valid] = -1

        self.insert1(dict(scan_key, pupil=pupil, dpupil=dpupil, center=center, valid=valid))


@schema
class Treadmill(dj.Computed, FilterMixin, BehaviorMixin):
    definition = """
    # eye movement data

    -> InputResponse
    -> treadmill.Treadmill
    ---
    treadmill          : external-data   # treadmill speed (|velcolity|)
    valid              : external-data   # valid trials
    """

    @property
    def key_source(self):
        rel = InputResponse
        return rel & treadmill.Treadmill() & stimulus.BehaviorSync()

    def make(self, scan_key):
        log.info('Populating\n' + pformat(scan_key))
        v, treadmill_time = self.load_treadmill_velocity(scan_key)
        frame_times = self.load_frame_times(scan_key)
        behavior_clock = self.load_behavior_timing(scan_key)

        if len(frame_times) - len(behavior_clock) != 0:
            assert abs(len(frame_times) - len(behavior_clock)) < 2, 'Difference bigger than 2 time points'
            l = min(len(frame_times), len(behavior_clock))
            log.warning('Frametimes and stimulus.BehaviorSync differ in length! Shortening it.')
            frame_times = frame_times[:l]
            behavior_clock = behavior_clock[:l]

        fr2beh = NaNSpline(frame_times, behavior_clock, k=1, ext=3)
        duration, offset = map(float, (Preprocessing() & scan_key).fetch1('duration', 'offset'))
        sample_point = offset + duration / 2

        log.info('Downsampling treadmill signal to {}Hz'.format(1 / duration))

        h_tread = self.get_filter(duration, np.nanmedian(np.diff(treadmill_time)), 'hamming', warning=True)
        treadmill_spline = NaNSpline(treadmill_time, np.abs(np.convolve(v, h_tread, mode='same')), k=1, ext=0)

        flip_times = (InputResponse.Input * Frame * stimulus.Trial & scan_key).fetch('flip_times',
                                                                                     order_by='row_id ASC')

        flip_times = [ft.squeeze() for ft in flip_times]

        # If no Frames are present, skip this scan
        if len(flip_times) == 0:
            log.warning('No static frames were present to be processed for {}'.format(scan_key))
            return

        stimulus_onset = InputResponse.stimulus_onset(flip_times, duration)
        tm = treadmill_spline(fr2beh(stimulus_onset + sample_point))
        valid = ~np.isnan(tm)
        if not np.all(valid):
            log.warning('Found {} NaN trials. Setting to -1'.format((~valid).sum()))
            tm[~valid] = -1

        self.insert1(dict(scan_key, treadmill=tm, valid=valid))
    

# Patch job for the hardcoding mess that was StaticMultiDataset.fill()
# Instead of editing the code each time, the user will enter they scan with the desire group_id into here then call StaticMultiDataset.fill()
@schema
class StaticMultiDatasetGroupAssignment(dj.Manual):
    definition = """
    group_id : int unsigned
    -> InputResponse
    ---
    description = '' : varchar(1024)
    """

@schema
class StaticMultiDataset(dj.Manual):
    definition = """
    # defines a group of datasets
    -> ImageNetSubset
    group_id    : smallint  # index of group
    ---
    description : varchar(255) # short description of the data
    """

    class Member(dj.Part):
        definition = """
        -> master
        -> InputResponse
        ---
        name                    : varchar(50) unique # string description to be used for training
        """

    @staticmethod
    def fill(subset_key):
        _template = 'group{group_id:03d}-{animal_id}-{session}-{scan_idx}-{preproc_id}-{subset_id}'
        if not len(StaticMultiDataset & {'group_id': subset_key['group_id']}) > 0:
            existing_scan = StaticMultiDataset.fetch('group_id')
            try:
                gid = np.max(existing_scan) + 1
            except ValueError:
                gid = 1
        else:
            gid = subset_key['group_id']

        scans = (StaticMultiDatasetGroupAssignment & 'group_id={}'.format(gid)).fetch(as_dict=True)
        for scan in scans:
            subsets = (dj.U('subset_id') & (ConditionTier & scan & subset_key)).fetch(as_dict=True, order_by='subset_id')

            for i, s in enumerate(subsets):
                # scan['group_id'] = gid + i
                scan = dict(scan, subset_id=s['subset_id'])
                # Check if the scan has been added to StaticMultiDataset.Member, if not then do it
                if len(StaticMultiDataset & scan) == 0:
                    # Group id has not been added into StaticMultiDataset, thus add it
                    StaticMultiDataset.insert1(scan, ignore_extra_fields=1, skip_duplicates=True)
        
                # Handle instertion into Member table
                # if len(StaticMultiDataset.Member() & scan) == 0:
                StaticMultiDataset.Member().insert1(dict(scan, name = _template.format(**scan)), ignore_extra_fields=True, skip_duplicates=True)

    def fetch_data(self, key, key_order=None):
        assert len(self & key) == 1, 'Key must refer to exactly one multi dataset'
        ret = OrderedDict()
        log.info('Fetching data for ' +  repr(key))
        for mkey in (self.Member() & key).fetch(dj.key,
                                                order_by='animal_id ASC, session ASC, scan_idx ASC, preproc_id ASC'):
            name = (self.Member() & mkey).fetch1('name')
            include_behavior = bool(Eye().proj() * Treadmill().proj() & mkey)
            data_names = ['images', 'responses'] if not include_behavior \
                else ['images',
                      'behavior',
                      'pupil_center',
                      'responses']
            log.info('Data will be ({})'.format(','.join(data_names)))

            h5filename = InputResponse().get_filename(mkey)
            log.info('Loading dataset {} --> {}'.format(name, h5filename))
            ret[name] = datasets.StaticImageSet(h5filename, *data_names)
        if key_order is not None:
            log.info('Reordering datasets according to given key order {}'.format(', '.join(key_order)))
            ret = OrderedDict([
                (k, ret[k]) for k in key_order
            ])
        return ret


@schema
class MultipleDatasets(dj.Computed):
    definition = """
    -> StaticMultiDataset
    -> neuro_configs.CellMatchParameters
    -> loop.LoopGroup
    """
    
    @property
    def key_source(self):
        return loop.LoopGroup * StaticMultiDataset * neuro_configs.CellMatchParameters  # need to restrict by loop_group, group_id, match_params
   
    class MatchedCells(dj.Part):
        definition = """ # Matched cells (in the same order) in each member scan of the multi dataset
        -> master
        -> StaticMultiDataset.Member
        ---
        name:            varchar(50)   # string description of the member scan to be used for training
        matched_cells:   longblob      # array of matched cells in each member scan of the group, unit_ids at the same indices for each member scan represent the matched cells
        """
    
    def make(self, key):
        match_params = (neuro_configs.CellMatchParameters & key).fetch1()
        
        keys, d1, d2, corr_matrix = (loop.PairScanCellMatchOracleCorr & key).fetch('KEY', 'src_units', 'target_units', 'corr_matrix')
        keep = []
        all_diags = []
        all_target_keys = []
        all_target_units = []

        for k, d1_units, d2_units, corr in zip(keys, d1, d2, corr_matrix):
            diag = np.diag(corr)
            d1_key = (StaticMultiDataset.Member & key & {'animal_id': k['animal_id'], 'session': k['src_session'], 'scan_idx': k['src_scan_idx'], 'loop_group': k['loop_group']}).fetch1()
            d2_key = (StaticMultiDataset.Member & key & {'animal_id': k['animal_id'], 'session': k['target_session'], 'scan_idx': k['target_scan_idx'], 'loop_group': k['loop_group']}).fetch1()

            # Compute mean distance between matched cells
            if (loop.ClosedLoopScan & d1_key).fetch1('mei_source'): # when src key is has mei_source=1 (day 1 scan)
                src_key = d1_key.copy()
                src_units = d1_units.copy()
                all_target_keys.append(d2_key)
                all_target_units.append(d2_units)
                all_diags.append(diag)
                src_rest = InputResponse.ResponseKeys.proj(src_session='session', src_scan_idx='scan_idx', src_unit_id='unit_id')
                distance = (loop.TempBestProximityCellMatch2 & src_rest & d2_key & {'src_session': d1_key['session'], 'src_scan_idx': d1_key['scan_idx']}).fetch('mean_distance', order_by='src_unit_id')

            else: # when src key is has mei_source!=1 (a scan after day 1)
                distance = []
                for d1_u, d2_u in zip(d1_units, d2_units):
                    d1x, d1y, d1z = (meso.StackCoordinates.UnitInfo() & d1_key & 'segmentation_method = 6' & {'unit_id': d1_u}).fetch('stack_x', 'stack_y', 'stack_z', order_by=('stack_session', 'stack_idx'))
                    d2x, d2y, d2z = (meso.StackCoordinates.UnitInfo() & d2_key & 'segmentation_method = 6' & {'unit_id': d2_u}).fetch('stack_x', 'stack_y', 'stack_z', order_by=('stack_session', 'stack_idx'))
                    dist = []
                    for x1, y1, z1, x2, y2, z2 in zip(d1x, d1y, d1z, d2x, d2y, d2z):
                        dist.append(np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2))
                    distance.append(np.mean(dist))

            # Select matched pairs that satisfy thresholds on oracle correlation and max distance 
            oraclecorr_thre = match_params['oraclecorr_thre']
            distance_thre = match_params['distance_thre']
            keep.append((diag > oraclecorr_thre) & (np.array(distance) < distance_thre))

        # Select satisfied matched pairs across all pairs of datasets
        all_keep = keep[0]
        for i in range(len(keep)):
            all_keep *= keep[i]

        # # OLD METHOD
        # old_keep_src_units = src_units[all_keep]
        # old_keep_target_units = all_target_units[0][all_keep]

        # # Iteratively take the most correlated pairs for units matched to the same target unit
        # final_src_units = []
        # final_target_units = []
        # for src, target, corr in sorted(zip(old_keep_src_units, old_keep_target_units, diag[all_keep]), key=lambda x: x[-1], reverse=True):
        #     if target not in final_target_units:
        #         final_src_units.append(src)
        #         final_target_units.append(target)

        # CURRENT METHOD
        current_keep_src_units = src_units[all_keep]
        current_keep_target_units = np.stack(all_target_units)[:, all_keep]
        all_diags = np.stack(all_diags)[:, all_keep] # BUG: for group 188-193, forgot to add this line, so some of selected unique src_units are not the ones with highest correlation with the target_units

        # Keep only one unique unit in each dataset
        if match_params['unique_match_per_pair']:
            all_keep_idx = []
            for units, corrs in zip(current_keep_target_units, all_diags): # for each target dataset
                keep_units, keep_idx = [], []
                for (i, u), corr in sorted(zip(enumerate(units), corrs), key=lambda x: x[-1], reverse=True): # iteratively select the idx with highest correlation to keep
                    if u not in keep_units:
                        keep_units.append(u)
                        keep_idx.append(i)
                all_keep_idx.append(keep_idx)
            final_keep_idx = list(set.intersection(*map(set, all_keep_idx)))

            keep_src_units = current_keep_src_units[final_keep_idx]
            keep_target_units = current_keep_target_units[:, final_keep_idx]
            
            for units in keep_target_units:
                assert len(units) == len(np.unique(units)), 'Matched units not unique in target datasets!'

        self.insert1(key)
        self.MatchedCells.insert1({**key, **src_key, 'matched_cells': keep_src_units})
        for target_key, target_units in zip(all_target_keys, keep_target_units):
            self.MatchedCells.insert1({**key, **target_key, 'matched_cells': target_units})


from tqdm import tqdm
import json
import numpy as np
import pandas as pd
from scipy import stats
import torch

from neuro_data.utils.measures import corr

from .. import logger as log

from staticnet_experiments.utils import correlation_closure, compute_predictions, compute_scores
configs = dj.create_virtual_module('neurostatic_configs', 'neurostatic_configs')

@schema
class Oracle(dj.Computed):
    definition = """
    # oracle computation for static images

    -> StaticMultiDataset.proj(dummy_subset_id='subset_id')
    -> configs.DataConfig
    ---
    """

    @property
    def key_source(self):
        return StaticMultiDataset.proj(dummy_subset_id='subset_id') * configs.DataConfig()

    class Scores(dj.Part):
        definition = """
        -> master
        -> StaticMultiDataset.Member.proj(dummy_subset_id='subset_id')
        ---
        pearson           : float     # mean pearson correlation
        spearman          : float     # mean spearnab correlation
        """

    class UnitScores(dj.Part):
        definition = """
        -> master.Scores
        -> StaticScan.Unit
        ---
        pearson           : float     # unit pearson correlation
        spearman          : float     # unit spearman correlation
        """

    def make(self, key):
        key['subset_id'] = key['dummy_subset_id']

        # --- load data
        testsets, testloaders = configs.DataConfig().load_data(key, tier='test', oracle=True)
        self.insert1(dict(key), ignore_extra_fields=True)
        for readout_key, loader in testloaders.items():
            log.info('Computing oracle for ' + readout_key)
            oracles, data = [], []
            for inputs, *_, outputs in loader:
                inputs = inputs.numpy()
                outputs = outputs.numpy()
                assert np.all(np.abs(np.diff(inputs, axis=0)) == 0), \
                    'Images of oracle trials does not match'
                r, n = outputs.shape  # responses X neurons
                log.info('\t    {} responses for {} neurons'.format(r, n))
                assert r > 4, 'need more than 4 trials for oracle computation'
                mu = outputs.mean(axis=0, keepdims=True)
                oracle = (mu - outputs / r) * r / (r - 1)
                oracles.append(oracle)
                data.append(outputs)
            if len(data) == 0:
                log.error('Found no oracle trials! Skipping ...')
                return

            # Pearson correlation
            pearson = corr(np.vstack(data), np.vstack(oracles), axis=0)

            # Spearman correlation
            data_rank = np.empty(np.vstack(data).shape)
            oracles_rank = np.empty(np.vstack(oracles).shape)

            for i in range(np.vstack(data).shape[1]):
                data_rank[:, i] = np.argsort(np.argsort(np.vstack(data)[:, i]))
                oracles_rank[:, i] = np.argsort(np.argsort(np.vstack(oracles)[:, i]))
            spearman = corr(data_rank, oracles_rank, axis=0)

            member_key = (StaticMultiDataset.Member() & key &
                          dict(name=readout_key)).fetch1(dj.key)
            member_key = dict(member_key, **key)
            self.Scores().insert1(dict(member_key, pearson=np.mean(pearson), spearman=np.mean(spearman)), ignore_extra_fields=True)
            unit_ids = testsets[readout_key].neurons.unit_ids
            assert len(unit_ids) == len(
                pearson) == len(spearman) == outputs.shape[-1], 'Neuron numbers do not add up'
            self.UnitScores().insert(
                [dict(member_key, pearson=c, spearman=s, unit_id=u)
                 for u, c, s in tqdm(zip(unit_ids, pearson, spearman), total=len(unit_ids))],
                ignore_extra_fields=True)

########################
# @schema
# class ClassImageNetSplit(dj.Lookup):
#     definition = """ # split frames in each image_class (or mixed) of each scan into train, validation, and test sets (one sub test set for each image class of oracle images used)
#     -> StaticScan
#     subset_id:          int          # 
#     train_class:        varchar(45)  # description of image classes in the training set one specific image class or mixed)
#     image_class:        varchar(45)  # one specific image class in the current scan
#     ---
#     n_unique_images:    int  # total number of unique images in this scan
#     """
#     class Image(dj.Part):
#         definition = """ 
#         -> master
#         -> stimulus.StaticImage.Image
#         ---
#         -> Tier
#         """
#     def fill(self, scan_key, validation='like_train', include_mixed_classes=False):
#         """ Assign each frame in the specific image class in the current scan to train/test/validation set.
#         Arguments:
#             scan_key: An scan (animal_id, session, scan_idx) that has stimulus.Trials
#                 created. Usually one where the stimulus was presented.
#             include_mixed_classes: boolean, whether to add a mixed image_class
#         """
        
#         key = (StaticScan & scan_key).fetch1('KEY')
        
#         # Find out whether we are using the old pipeline (grayscale only) or the new version
#         if stimulus.Frame & (stimulus.Trial & scan_key):
#             frame_table = stimulus.Frame
#         elif stimulus.ColorFrameProjector & (stimulus.Trial & scan_key):
#             frame_table = stimulus.ColorFrameProjector
#         else:
#             print('Static images were not shown for this scan')

#         # Get all image ids in this scan
#         all_frames = frame_table * stimulus.Trial & scan_key & IMAGE_CLASSES
#         classes = (dj.U('image_class') & all_frames).fetch('image_class')
#         all_classes = classes

#         if include_mixed_classes:
#             all_classes = np.append(classes, 'mixed')

#         for im_class in all_classes:
#             if im_class != 'mixed':
#                 unique_frames = dj.U('image_id', 'image_class').aggr(all_frames, repeats='COUNT(*)')
#                 num_frames = len(unique_frames)
#                 assert num_frames != 0, 'unique_frames == 0'
                
#                 # Insert test set images (one sub test set for each image_class)
#                 n = int(np.median(unique_frames.fetch('repeats')))  # HACK
#                 oracle_rel = unique_frames & 'repeats > {}'.format(n)
#                 non_oracle_rel = (unique_frames & {'image_class': im_class}).proj() - oracle_rel.proj()
#                 self.insert1({**key, 'train_class': im_class, 'image_class': im_class, 'n_unique_images': len(non_oracle_rel)+len(oracle_rel)}, skip_duplicates=True)

#                 num_oracles = len(oracle_rel)  
#                 if num_oracles == 0:
#                     raise ValueError('Could not find repeated frames to use for oracle.')
#                 oracle_classes = (dj.U('image_class') & oracle_rel).fetch('image_class')
#                 for ora_class in oracle_classes:
#                     oracle_ids = (oracle_rel & {'image_class': ora_class}).fetch('image_id')
#                     if ora_class == 'imagenet':
#                         self.Image.insert([{**key, 'train_class': im_class, 'image_id': iid, 'image_class': ora_class, 'tier': 'test'} for iid in oracle_ids], skip_duplicates=True)
#                     else: 
#                         self.Image.insert([{**key, 'train_class': im_class, 'image_id': iid, 'image_class': ora_class, 'tier': '{}_test'.format(ora_class)} for iid in oracle_ids], skip_duplicates=True)

#                 # Fetch non-oracle images
#                 image_ids = non_oracle_rel.fetch('image_id', order_by='repeats DESC')
#                 # * NOTE: this fetches all oracle images first and the rest in a "random" order;
#                 # we use that random order to make the validation/training division below.

#                 # Compute number of validation examples
#                 num_validation = int(np.ceil(len(image_ids) * 0.1))  # 10% validation examples

#                 # Insert
#                 self.Image.insert([{**key, 'train_class': im_class, 'image_id': iid, 'image_class': im_class, 'tier': 'validation'} for iid in image_ids[:num_validation]])
#                 self.Image.insert([{**key, 'train_class': im_class, 'image_id': iid, 'image_class': im_class, 'tier': 'train'} for iid in image_ids[num_validation:]])
            
#             else:
#                 unique_frames = dj.U('image_id').aggr(all_frames & [{'image_class': ic} for ic in classes], repeats='COUNT(*)')
#                 num_frames = len(unique_frames)
#                 assert num_frames != 0, 'unique_frames == 0'
                
#                 # Insert test set images (one sub test set for each image_class)
#                 n = int(np.median(unique_frames.fetch('repeats')))  # HACK
#                 oracle_rel = unique_frames & 'repeats > {}'.format(n)
#                 num_oracles = len(oracle_rel)  # repeats
#                 if num_oracles == 0:
#                     raise ValueError('Could not find repeated frames to use for oracle.')
#                 oracle_classes = (dj.U('image_class') & oracle_rel).fetch('image_class')
#                 for ora_class in oracle_classes:
#                     oracle_ids = (oracle_rel & {'image_class': ora_class}).fetch('image_id')
#                     if ora_class == 'imagenet':
#                         self.Image.insert([{**key, 'image_id': iid, 'image_class': ora_class, 'tier': 'test'} for iid in oracle_ids], skip_duplicates=True)
#                     else: 
#                         self.Image.insert([{**key, 'image_id': iid, 'image_class': ora_class, 'tier': '{}_test'.format(ora_class)} for iid in oracle_ids], skip_duplicates=True)
                
#                 selected = []
#                 non_oracle_rel = (unique_frames.proj() - oracle_rel.proj())
#                 num_images_per_class = int(np.floor(len(non_oracle_rel)/len(classes)))
#                 for i, ic in enumerate(classes):
#                     # Fetch non-oracle images of a specific image class
#                     image_ids = (non_oracle_rel & {'image_class': ic}).fetch('image_id', order_by='image_id')
#                     assert len(image_ids) != num_images_per_class, 'Unequal number of images in each image class!'
#                     if i == 0:
#                         selected_image_ids = np.random.choice(image_ids, num_images_per_class)
#                     else:
#                         rest = np.array(set(image_ids) - set(selected))
#                         selected_image_ids = np.random.choice(rest, num_images_per_class)
#                     selected.append(selected_image_ids)

#                     # Compute number of validation examples
#                     num_validation = int(np.ceil(len(selected_image_ids) * 0.1))  # 10% validation examples

#                     # Insert
#                     self.Image.insert([{**key, 'image_id': iid, 'image_class': ic, 'tier': 'validation'} for iid in selected_image_ids[:num_validation]])
#                     self.Image.insert([{**key, 'image_id': iid, 'image_class': ic, 'tier': 'train'} for iid in selected_image_ids[num_validation:]])

# def oracle_score(key, tier):
#     # --- load data
#     testsets, testloaders = configs.DataConfig().load_data(key=key, tier=tier, oracle=True)
    
#     for readout_key, loader in testloaders.items():
#         log.info('Computing oracle for ' + readout_key)
#         oracles, data = [], []
#         for inputs, *_, outputs in loader:
#             inputs = inputs.numpy()
#             outputs = outputs.numpy()
#             assert np.all(np.abs(np.diff(inputs, axis=0)) == 0), \
#                 'Images of oracle trials does not match'
#             r, n = outputs.shape  # responses X neurons
#             log.info('\t    {} responses for {} neurons'.format(r, n))
#             assert r > 4, 'need more than 4 trials for oracle computation'
#             mu = outputs.mean(axis=0, keepdims=True)
#             oracle = (mu - outputs / r) * r / (r - 1)
#             oracles.append(oracle)
#             data.append(outputs)
#         if len(data) == 0:
#             log.error('Found no oracle trials! Skipping ...')
#             return
#         pearson = corr(np.vstack(data), np.vstack(oracles), axis=0)

#         member_key = (StaticMultiDataset.Member() & key &
#                       dict(name=readout_key)).fetch1(dj.key)
#         member_key = dict(member_key, **key)

#         unit_ids = testsets[readout_key].neurons.unit_ids
#         assert len(unit_ids) == len(
#             pearson) == outputs.shape[-1], 'Neuron numbers do not add up'

#         return member_key, pearson, unit_ids


# @schema
# class Oracle(dj.Computed):
#     definition = """
#     # oracle computation for static images

#     -> StaticMultiDataset
#     -> configs.DataConfig
#     ---
#     """

#     @property
#     def key_source(self):
#         return StaticMultiDataset() * configs.DataConfig()

#     class Scores(dj.Part):
#         definition = """
#         -> master
#         -> StaticMultiDataset.Member
#         ---
#         ff_pearson           : float     # mean test correlation using full-field oracles
#         mask_pearson         : float     # mean test correlation using masked oracles
#         avg_pearson          : float     # average of ff_pearson and mask_pearson
#         """

#     class UnitScores(dj.Part):
#         definition = """
#         -> master.Scores
#         -> StaticScan.Unit
#         ---
#         unit_ff_pearson           : float     # unit test correlation using full-field oracles
#         unit_masked_pearson       : float     # unit test correlation using masked oracles
#         unit_avg_pearson          : float     # average of unit_ff_pearson and unit_masked_pearson 
#         """

#     def make(self, key):

#         self.insert1(dict(key))
#         member_key, ff_pearson, ff_unit_ids = oracle_score(key, 'test')
#         _, masked_pearson, masked_unit_ids = oracle_score(key, 'test_masked')
#         # import pdb; pdb.set_trace()
#         avg_pearson = (ff_pearson + masked_pearson) / 2
#         assert ((ff_unit_ids == masked_unit_ids).all()), "Unit ids for two types of oracles do not match!"

#         self.Scores().insert1(dict(member_key, ff_pearson=np.mean(ff_pearson), mask_pearson=np.mean(masked_pearson), avg_pearson=np.mean(avg_pearson)), ignore_extra_fields=True)
#         self.UnitScores().insert(
#             [dict(member_key, unit_ff_pearson=f, unit_masked_pearson=m, unit_avg_pearson=a, unit_id=u)
#              for f, m, a, u in tqdm(zip(ff_pearson, masked_pearson, avg_pearson, ff_unit_ids), total=len(ff_unit_ids))],
#             ignore_extra_fields=True)

# @schema
# class NewOracle(dj.Computed):
#     definition = """
#     # oracle computation for static images

#     -> StaticMultiDataset
#     -> configs.DataConfig
#     ---
#     """

#     @property
#     def key_source(self):
#         return StaticMultiDataset() * configs.DataConfig()

#     class Scores(dj.Part):
#         definition = """
#         -> master
#         -> StaticMultiDataset.Member
#         ---
#         leave_one_out_pearson           : float     # mean test pearson correlation using leave-one-out 
#         leave_one_out_spearman          : float     # mean test spearmon correlation using leave-one-out
#         split_half_pearson              : float     # mean test pearson correlation using split_half
#         split_half_spearman             : float     # mean test spearman correlation using split_half
#         """

#     class UnitScores(dj.Part):
#         definition = """
#         -> master.Scores
#         -> StaticScan.Unit
#         ---
#         leave_one_out_pearson           : float     
#         leave_one_out_spearman          : float    
#         split_half_pearson              : float # mean pearson correlation over 1000 random splits
#         split_half_spearman             : float # mean spearman correlation over 1000 random splits
#         split_half_pearson_pval         : float # p value for t-test comparison of 1000 corr distribution from data and shuffled data
#         split_half_spearman_pval        : float # p value for t-test comparison of 1000 corr distribution from data and shuffled data
#         """

#     def make(self, key):
#         # --- load data
#         testsets, testloaders = configs.DataConfig().load_data(key, tier='test', oracle=True)

#         self.insert1(dict(key))
#         for readout_key, loader in testloaders.items():
#             log.info('Computing oracle for ' + readout_key)
#             oracles, data, filled_data = [], [], []
#             for inputs, *_, outputs in loader:
#                 inputs = inputs.numpy()
#                 outputs = outputs.numpy()
#                 assert np.all(np.abs(np.diff(inputs, axis=0)) == 0), \
#                     'Images of oracle trials does not match'
#                 r, n = outputs.shape  # responses X neurons
#                 log.info('\t    {} responses for {} neurons'.format(r, n))
#                 assert r > 4, 'need more than 4 trials for oracle computation'
#                 mu = outputs.mean(axis=0, keepdims=True)
#                 oracle = (mu - outputs / r) * r / (r - 1)
#                 oracles.append(oracle)
#                 data.append(outputs)
#                 filled = outputs.copy()
#                 if r != 10:
#                     filled = np.concatenate([(np.nan * np.ones(n)).reshape(1, -1), filled])
#                 filled_data.append(filled)
#             if len(data) == 0:
#                 log.error('Found no oracle trials! Skipping ...')
#                 return
#             # leave-one-out pearson
#             lou_pearson = corr(np.vstack(data), np.vstack(oracles), axis=0)
#             # leave-one-out spearman
#             data_rank = np.empty(np.vstack(data).shape)
#             oracles_rank = np.empty(np.vstack(oracles).shape)

#             for i in range(np.vstack(data).shape[1]):
#                 data_rank[:, i] = np.argsort(np.argsort(np.vstack(data)[:, i]))
#                 oracles_rank[:, i] = np.argsort(np.argsort(np.vstack(oracles)[:, i]))
#             lou_spearman = corr(data_rank, oracles_rank, axis=0)

#             # split-half
#             filled_data = np.stack(filled_data)
#             original = filled_data.copy()
#             shuffled = np.empty(filled_data.shape)

#             # Completely shuffle image * trial matrix for each neuron
#             np.random.seed(0)
#             for i in range(filled_data.shape[2]):
#                 a = filled_data[:, :, i].ravel()
#                 np.random.shuffle(a)
#                 shuffled[:, :, i] = a.reshape(filled_data.shape[0], filled_data.shape[1])

#             import scipy
#             all_pearson, all_spearman, all_s_pearson, all_s_spearman, pearson_pval, spearman_pval = [], [], [], [], [], []
#             for i in range(filled_data.shape[2]):
#                 o = original[:, :, i]
#                 s = shuffled[:, :, i]
#                 pearson, spearman, s_pearson, s_spearman = [], [], [], []
#                 for j in range(1000):
#                     np.random.seed(j)
#                     idx1 = np.random.choice(filled_data.shape[1], int(filled_data.shape[1]/2), replace=False)
#                     idx2 = list(set(range(filled_data.shape[1])) - set(idx1))
#                     pearson.append(scipy.stats.pearsonr(np.nanmean(o[:, idx1], axis=1), np.nanmean(o[:, idx2], axis=1))[0])
#                     spearman.append(scipy.stats.spearmanr(np.nanmean(o[:, idx1], axis=1), np.nanmean(o[:, idx2], axis=1))[0])
#                     s_pearson.append(scipy.stats.pearsonr(np.nanmean(s[:, idx1], axis=1), np.nanmean(s[:, idx2], axis=1))[0])
#                     s_spearman.append(scipy.stats.spearmanr(np.nanmean(s[:, idx1], axis=1), np.nanmean(s[:, idx2], axis=1))[0])

#                 all_pearson.append(np.stack(pearson))
#                 all_spearman.append(np.stack(spearman))
#                 all_s_pearson.append(np.stack(s_pearson))
#                 all_s_spearman.append(np.stack(s_spearman))
#                 pearson_pval.append(scipy.stats.ttest_ind(np.stack(pearson), np.stack(s_pearson))[1])
#                 spearman_pval.append(scipy.stats.ttest_ind(np.stack(spearman), np.stack(s_spearman))[1])

#             split_pearson = np.mean(np.stack(all_pearson), 1)
#             split_spearman = np.mean(np.stack(all_spearman), 1)

#             member_key = (StaticMultiDataset.Member() & key &
#                           dict(name=readout_key)).fetch1(dj.key)
#             member_key = dict(member_key, **key)
#             self.Scores().insert1(dict(member_key, leave_one_out_pearson=np.mean(lou_pearson),
#                                                    leave_one_out_spearman=np.mean(lou_spearman),
#                                                    split_half_pearson=np.mean(split_pearson),
#                                                    split_half_spearman=np.mean(split_spearman)), ignore_extra_fields=True)
#             unit_ids = testsets[readout_key].neurons.unit_ids
#             assert len(unit_ids) == len(lou_pearson) == len(lou_spearman) == len(split_pearson) == len(split_spearman) == outputs.shape[-1], 'Neuron numbers do not add up'
#             self.UnitScores().insert(
#                 [dict(member_key, unit_id=u, leave_one_out_pearson=lp, leave_one_out_spearman=ls, split_half_pearson=sp, split_half_spearman=ss, split_half_pearson_pval=ppval, split_half_spearman_pval=spval)
#                  for u, lp, ls, sp, ss, ppval, spval in tqdm(zip(unit_ids, lou_pearson, lou_spearman, split_pearson, split_spearman, pearson_pval, spearman_pval), total=len(unit_ids))],
#                 ignore_extra_fields=True)


# @schema
# class Model(dj.Computed):
#     definition = """
#     -> models.Model
#     ---
#     val_corr     : float     
#     """

#     class TestScores(dj.Part):
#         definition = """
#         -> master
#         -> models.Model.TestScores
#         ---
#         neurons                  : int         # number of neurons
#         ff_pearson               : float       # test correlation on full-field oracle single trial responses
#         masked_pearson           : float       # test correlation on masked oracle single trial responses
#         avg_pearson              : float       # average of ff_pearson and masked_pearson
#         """

#     class UnitTestScores(dj.Part):
#         definition = """
#         -> master.TestScores
#         -> models.Model.UnitTestScores
#         ---
#         unit_ff_pearson                  : float       # single unit test correlation on full-field oracle single trial responses
#         unit_masked_pearson              : float       # single unit test correlation on masked oracle single trial responses
#         unit_avg_pearson                 : float       # average of ff_pearson and masked_pearson
#         """
        
#     def make(self, key):
#         member_key, ff_unit_ids, ff_pearson, unit_ff_pearson, masked_pearson, unit_masked_pearson = self.evaluate(key)
#         avg_pearson = (ff_pearson + masked_pearson) / 2
#         unit_avg_pearson = (unit_ff_pearson + unit_masked_pearson) / 2
#         updated_key = (models.Model & key).proj('val_corr').fetch1()
#         self.insert1(updated_key)
#         self.TestScores.insert1({**member_key, 'neurons': len(ff_unit_ids), 'ff_pearson': ff_pearson, 'masked_pearson': masked_pearson, 'avg_pearson': avg_pearson})
#         self.UnitTestScores.insert([{**member_key, 'unit_id': u, 'unit_ff_pearson': f, 'unit_masked_pearson': m, 'unit_avg_pearson': a} for u, f, m, a in zip(ff_unit_ids, unit_ff_pearson, unit_masked_pearson, unit_avg_pearson)])
    
#     def load_network(self, key=None, trainsets=None):
#         if key is None:
#             key = self.fetch1(dj.key)
#         model = configs.NetworkConfig().build_network(key, trainsets=trainsets)
#         state_dict = (models.Model & key).fetch1('model')
#         state_dict = {k: torch.as_tensor(state_dict[k][0].copy()) for k in state_dict.dtype.names}
#         mod_state_dict = model.state_dict()
#         for k in set(mod_state_dict) - set(state_dict):
#             log.warning('Could not find paramater {} setting to initialization value'.format(repr(k)))
#             state_dict[k] = mod_state_dict[k]
#         model.load_state_dict(state_dict)
#         return model

        
#     def evaluate(self, key=None):
#         if key is None:
#             key = self.fetch1('KEY')

#         model = self.load_network(key)
#         model.eval()
#         model.cuda()

#         # get network configuration information
#         net_key = configs.NetworkConfig().net_key(key)
#         train_key = configs.TrainConfig().train_key(net_key)
        
#         def compute_test_corr(net_key, tier, train_key):
#             testsets, testloaders = configs.DataConfig().load_data(net_key, tier=tier, cuda=True, **train_key)

#             scores, unit_scores = [], []
#             for readout_key, testloader in testloaders.items():
#                 log.info('Computing test scores for ' + readout_key)

#                 y, y_hat = compute_predictions(testloader, model, readout_key)
#                 perf_scores = compute_scores(y, y_hat)

#                 member_key = (StaticMultiDataset.Member() & key & dict(name=readout_key)).fetch1(dj.key)
#                 member_key.update(key)

#                 unit_ids = testloader.dataset.neurons.unit_ids
#                 pearson = perf_scores.pearson.mean()

#             return member_key, unit_ids, pearson, perf_scores.pearson
#         # import pdb; pdb.set_trace()
#         member_key, ff_unit_ids, ff_pearson, unit_ff_pearson = compute_test_corr(net_key, 'test', train_key)
#         _, masked_unit_ids, masked_pearson, unit_masked_pearson = compute_test_corr(net_key, 'test_masked', train_key)
#         assert ((ff_unit_ids == masked_unit_ids).all()), "Unit ids for two types of oracles do not match!"
        
#         return member_key, ff_unit_ids, ff_pearson, unit_ff_pearson, masked_pearson, unit_masked_pearson