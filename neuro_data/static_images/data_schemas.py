from collections import OrderedDict
from functools import partial
from itertools import compress
from pprint import pformat

import datajoint as dj
import numpy as np
import pandas as pd

from neuro_data import logger as log
from neuro_data.utils.data import h5cached, SplineCurve, FilterMixin, fill_nans, NaNSpline
from neuro_data.static_images import datasets

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

schema = dj.schema('neurodata_static')

# set of attributes that uniquely identifies the frame content
UNIQUE_FRAME = {
    'stimulus.Frame': ('image_id', 'image_class'),
    'stimulus.MonetFrame': ('rng_seed', 'orientation'),
    'stimulus.TrippyFrame': ('rng_seed',),
    'stimulus.ColorFrameProjector': ('image_id', 'image_class'),
}

IMAGE_CLASSES = 'image_class in ("imagenet", "masked_oracle", "masked_single", "diverse_mei", "searched_nat", "mei2", "imagenet_v2_gray", "imagenet_v2_rgb", "gaudy_imagenet2", "exciting_imagenet")' # all valid natural image classes
ORACLE_CLASSES = 'image_class in ("imagenet", "masked_oracle", "gaudy_imagenet2", "exciting_imagenet", "mei2")'
TRAINING_CLASSES = 'image_class in ("imagenet", "masked_single", "searched_nat", "diverse_mei", "mei2", "gaudy_imagenet2")'
MASKED_CLASSES = ['diverse_mei', 'mei2', 'masked_single']
FF_CLASSES = ['imagenet', 'searched_nat', 'gaudy_imagenet2', 'exciting_imagenet']
    
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
        yield from zip(["train", "validation", "test", "test_masked", "test_exciting", 'test_ff', 'test_mei'])


@schema
class ExcludedTrial(dj.Manual):
    definition = """
    # trials to be excluded from analysis
    -> stimulus.Trial
    ---
    exclusion_comment='': varchar(64)   # reasons for exclusion
    """

# based on mesonet.MesoNetSplit
@schema
class ImageNetSplit(dj.Lookup):
    definition = """ # split imagenet frames into train, test, validation

    -> stimulus.StaticImage.Image
    ---
    -> Tier
    """
    def fill(self, scan_key):
        """ Assign each imagenet frame in the current scan to train/test/validation set.

        Arguments:
            scan_key: An scan (animal_id, session, scan_idx) that has stimulus.Trials
                created. Usually one where the stimulus was presented.

        Note:
            Each image is assigned to one set and that holds true for all our scans and
            collections. Once an image has been assigned (and models have been trained
            with that split), it cannot be changed in the future (this is problematic if
            images are reused as those from collection 2 or collection 3 with a different
            purpose).

            The exact split assigned will depend on the scans used in fill and the order
            that this table was filled. Not ideal.
        """
        # Find out whether we are using the old pipeline (grayscale only) or the new version
        if stimulus.Frame & (stimulus.Trial & scan_key):
            frame_table = stimulus.Frame
        elif stimulus.ColorFrameProjector & (stimulus.Trial & scan_key):
            frame_table = stimulus.ColorFrameProjector
        else:
            print('Static images were not shown for this scan')
        
        # Get all image ids in this scan
        all_frames = frame_table * stimulus.Trial & scan_key & IMAGE_CLASSES
        unique_frames = dj.U('image_id', 'image_class').aggr(all_frames, repeats='COUNT(*)')
        image_ids, image_classes = unique_frames.fetch('image_id', 'image_class', order_by='repeats DESC')
        num_frames = len(image_ids)
        # * NOTE: this fetches all oracle images first and the rest in a "random" order;
        # we use that random order to make the validation/training division below.

        # Get number of repeated frames
        assert len(unique_frames) != 0, 'unique_frames == 0'

        n = int(np.median(unique_frames.fetch('repeats')))  # HACK
        num_oracles = len(unique_frames & 'repeats > {}'.format(n))  # repeats
        if num_oracles == 0:
            raise ValueError('Could not find repeated frames to use for oracle.')

        # Compute number of validation examples
        num_validation = int(np.ceil((num_frames - num_oracles) * 0.1))  # 10% validation examples
        
        # Insert
        self.insert([{'image_id': iid, 'image_class': ic, 'tier': 'test'} for iid, ic in
                     zip(image_ids[:num_oracles], image_classes[:num_oracles])],
                    skip_duplicates=True)
        self.insert([{'image_id': iid, 'image_class': ic, 'tier': 'validation'} for
                     iid, ic in zip(image_ids[num_oracles: num_oracles + num_validation],
                                    image_classes[num_oracles: num_oracles + num_validation])])
        self.insert([{'image_id': iid, 'image_class': ic, 'tier': 'train'} for iid, ic in
                     zip(image_ids[num_oracles + num_validation:],
                         image_classes[num_oracles + num_validation:])])
        

@schema
class TempImageNetSplit(dj.Lookup):
    definition = """ # split imagenet frames into train, test, validation, and test_masked (if masked oracles are used)
    -> StaticScan
    ---
    n_unique_images:    int  # total number of unique images in this scan
    """
    class Image(dj.Part):
        definition = """ 
        -> master
        -> stimulus.StaticImage.Image
        ---
        -> Tier
        """
    def fill(self, scan_key):
        """ Assign each imagenet frame in the current scan to train/test/validation set.

        Arguments:
            scan_key: An scan (animal_id, session, scan_idx) that has stimulus.Trials
                created. Usually one where the stimulus was presented.

        Note:
            Each image is assigned to one set and that holds true for all our scans and
            collections. Once an image has been assigned (and models have been trained
            with that split), it cannot be changed in the future (this is problematic if
            images are reused as those from collection 2 or collection 3 with a different
            purpose).

            The exact split assigned will depend on the scans used in fill and the order
            that this table was filled. Not ideal.
        """
        
        key = (StaticScan & scan_key).fetch1('KEY')
        
        # Find out whether we are using the old pipeline (grayscale only) or the new version
        if stimulus.Frame & (stimulus.Trial & scan_key):
            frame_table = stimulus.Frame
        elif stimulus.ColorFrameProjector & (stimulus.Trial & scan_key):
            frame_table = stimulus.ColorFrameProjector
        else:
            print('Static images were not shown for this scan')

        # Get all image ids in this scan
        all_frames = frame_table * stimulus.Trial & scan_key & IMAGE_CLASSES
        unique_frames = dj.U('image_id', 'image_class').aggr(all_frames, repeats='COUNT(*)')
        image_ids, image_classes = unique_frames.fetch('image_id', 'image_class', order_by='repeats DESC')
        num_frames = len(unique_frames)
        self.insert1({**key, 'n_unique_images': num_frames})
        
        # Get number of repeated frames
        assert len(unique_frames) != 0, 'unique_frames == 0'

        n = int(np.median(unique_frames.fetch('repeats')))  # HACK
        oracle_rel = unique_frames & 'repeats > {}'.format(n)
        num_oracles = len(oracle_rel) 
        if num_oracles == 0:
            raise ValueError('Could not find repeated frames to use for oracle.')
            
        if len(dj.U('image_class') & unique_frames) == 1 or len(dj.U('image_class') & unique_frames) == 2 and len(dj.U('image_class') & oracle_rel) == 1:
            # Compute number of validation examples
            num_validation = int(np.ceil((num_frames - num_oracles) * 0.1))  # 10% validation examples

            # Fetch images in specific image classes
            image_ids, image_classes = (unique_frames & IMAGE_CLASSES).fetch('image_id', 'image_class', order_by='repeats DESC')
            # * NOTE: this fetches all oracle images first and the rest in a "random" order;
            # we use that random order to make the validation/training division below.

            self.Image.insert([{**key, 'image_id': iid, 'image_class': ic, 'tier': 'test'} for iid, ic in
                         zip(image_ids[:num_oracles], image_classes[:num_oracles])], skip_duplicates=True)
            self.Image.insert([{**key, 'image_id': iid, 'image_class': ic, 'tier': 'validation'} for iid, ic in 
                               zip(image_ids[num_oracles: num_oracles + num_validation], image_classes[num_oracles: num_oracles + num_validation])])
            self.Image.insert([{**key, 'image_id': iid, 'image_class': ic, 'tier': 'train'} for iid, ic in
                             zip(image_ids[num_oracles + num_validation:], image_classes[num_oracles + num_validation:])])
        
        elif len(dj.U('image_class') & unique_frames) == 2: # fewer than two image classes, this is a bit of hack, not ideal
            # Compute number of validation examples
            num_validation = int(np.ceil((num_frames - num_oracles) * 0.1))  # 10% validation examples

            # Fetch images in specific image classes
#             masked_oracle_ids, masked_oracle_classes = (unique_frames & 'image_class = "masked_oracle"').fetch('image_id', 'image_class', order_by='repeats DESC')
            mei_image_ids, mei_image_classes = (unique_frames & 'image_class = "mei2"').fetch('image_id', 'image_class', order_by='repeats DESC')
            nat_image_ids, nat_image_classes = (unique_frames & 'image_class = "imagenet"').fetch('image_id', 'image_class', order_by='repeats DESC')
            
            if len(mei_image_ids) > 0 and len(nat_image_ids) > 0:
                num_oracles = int(num_oracles / 2)
            if len(mei_image_ids) > len(nat_image_ids):
                image_ids = mei_image_ids
                image_classes = mei_image_classes
                self.Image.insert([{**key, 'image_id': iid, 'image_class': ic, 'tier': 'test_ff'} for iid, ic in
                         zip(nat_image_ids, nat_image_classes)], skip_duplicates=True)
            else:
                image_ids = nat_image_ids
                image_classes = nat_image_classes
                self.Image.insert([{**key, 'image_id': iid, 'image_class': ic, 'tier': 'test_mei'} for iid, ic in
                         zip(mei_image_ids, mei_image_classes)], skip_duplicates=True)
                
            self.Image.insert([{**key, 'image_id': iid, 'image_class': ic, 'tier': 'test'} for iid, ic in
                         zip(image_ids[:num_oracles], image_classes[:num_oracles])], skip_duplicates=True)
            self.Image.insert([{**key, 'image_id': iid, 'image_class': ic, 'tier': 'validation'} for
                             iid, ic in zip(image_ids[num_oracles: num_oracles + num_validation],
                                            image_classes[num_oracles: num_oracles + num_validation])])
            self.Image.insert([{**key, 'image_id': iid, 'image_class': ic, 'tier': 'train'} for iid, ic in
                             zip(image_ids[num_oracles + num_validation:],
                                 image_classes[num_oracles + num_validation:])])
                
            # * NOTE: this fetches all oracle images first and the rest in a "random" order;
            # we use that random order to make the validation/training division below.

            # Insert
#             if len(masked_oracle_ids) > 0:
#                 # Get number of full-field oracles
#                 num_oracles = int(num_oracles / 2)
#                 self.Image.insert([{**key, 'image_id': iid, 'image_class': ic, 'tier': 'test_masked'} for iid, ic in
#                          zip(masked_oracle_ids, masked_oracle_classes)], skip_duplicates=True)
                
#             if len(other_oracle_ids) > 0:
#                 # Get number of full-field oracles
#                 num_oracles = int(num_oracles / 2)
#                 self.Image.insert([{**key, 'image_id': iid, 'image_class': ic, 'tier': 'test'} for iid, ic in
#                          zip(exciting_oracle_ids, exciting_oracle_classes)], skip_duplicates=True)
                
#             self.Image.insert([{**key, 'image_id': iid, 'image_class': ic, 'tier': 'test'} for iid, ic in
#                          zip(image_ids[:num_oracles], image_classes[:num_oracles])], skip_duplicates=True)
#             self.Image.insert([{**key, 'image_id': iid, 'image_class': ic, 'tier': 'validation'} for
#                          iid, ic in zip(image_ids[num_oracles: num_oracles + num_validation],
#                                         image_classes[num_oracles: num_oracles + num_validation])])
#             self.Image.insert([{**key, 'image_id': iid, 'image_class': ic, 'tier': 'train'} for iid, ic in
#                          zip(image_ids[num_oracles + num_validation:],
#                              image_classes[num_oracles + num_validation:])])
        
        else: # when there are more than two image classes
            train_rels = (dj.U('image_class') & unique_frames & TRAINING_CLASSES).fetch()
            oracle_rels = (dj.U('image_class') & unique_frames & ORACLE_CLASSES).fetch()

            for rel in oracle_rels:
                image_ids, image_classes = (unique_frames & 'repeats > {}'.format(n) & rel).fetch('image_id', 'image_class', order_by='repeats DESC')
                if len(image_classes) > 0 and image_classes[0] == "imagenet":
                    self.Image.insert([{**key, 'image_id': iid, 'image_class': ic, 'tier': 'test'} for iid, ic in zip(image_ids, image_classes)])
                elif len(image_classes) > 0 and image_classes[0] == "masked_oracle":
                    self.Image.insert([{**key, 'image_id': iid, 'image_class': ic, 'tier': 'test_masked'} for iid, ic in zip(image_ids, image_classes)])
                else:
                    raise KeyError('Cannot find matching tier for this oracle image class!')

            for rel in train_rels:
                image_ids, image_classes = (unique_frames & 'repeats = 1' & rel).fetch('image_id', 'image_class', order_by='repeats DESC')
                if len(image_classes) > 0 :
                    # Compute number of validation examples
                    num_validation = int(np.ceil(len(image_ids) * 0.1))  # 10% validation examples
                    self.Image.insert([{**key, 'image_id': iid, 'image_class': ic, 'tier': 'validation'} for
                                 iid, ic in zip(image_ids[:num_validation], image_classes[:num_validation])])
                    self.Image.insert([{**key, 'image_id': iid, 'image_class': ic, 'tier': 'train'} for iid, ic in
                                 zip(image_ids[num_validation:], image_classes[num_validation:])])

        

@schema
class ConditionTier(dj.Computed):
    definition = """
    # split into train, test, validation, and test_masked (if masked oracles are used)

    -> stimulus.Condition
    -> StaticScan
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
        return StaticScan() & stimulus.Trial()

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
                targets = StaticScan * frame_table * TempImageNetSplit.Image & (stimulus.Trial & key) & IMAGE_CLASSES
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
        {'preproc_id': 5, 'offset': 0.05, 'duration': 0.5, 'row': 36, 'col': 64,
         'filter': 'hamming', 'gamma': True, 'trainstats_per_image': 1, 'linear_mon': 0}, 
        {'preproc_id': 6, 'offset': 0.05, 'duration': 0.5, 'row': 36, 'col': 64,
         'filter': 'hamming', 'gamma': True, 'trainstats_per_image': 1, 'linear_mon': 1}, 
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


@schema
class TrainClass(dj.Lookup):
    definition = """
    class_id   :  int
    image_class:  varchar(16)    # image class name
    ---
    table      :  varchar(1024)       # table query to fetch the stimuli for a certain image_class
    """
    contents = [(1, 'diverse_mei', 'stimulus.StaticImage.DiverseMEI'),
                (2, 'mei2', 'stimulus.StaticImage.MEICollection')]

    def get_stim_table(self):
        """ Return the stimulus tables (datajoint user tables) for a single training image class"""
        table_strs = self.fetch('table')
        tables = [eval(table_str) for table_str in table_strs]
        return tables


@h5cached('/external/cache/', mode='array', transfer_to_tmp=False,
          file_format='static{animal_id}-{session}-{scan_idx}-preproc{preproc_id}.h5')
@schema
class InputResponse(dj.Computed, FilterMixin):
    definition = """
    # responses of one neuron to the stimulus

    -> StaticScan
    -> Preprocessing
    ---
    """

    key_source = StaticScan() * Preprocessing() & Frame()

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
        hashes, trial_idxs, tiers, types, images = trials.fetch('condition_hash', 'trial_idx', 'tier',
                                                                'stimulus_type', 'frame', order_by='row_id')
        images = np.stack(images)
        if len(images.shape) == 3:
            log.info('Adding channel dimension')
            images = images[:, None, ...]
        elif len(images.shape) == 4:
            images = images.transpose(0, 3, 1, 2)
        hashes = hashes.astype(str)
        types = types.astype(str)
        
        # gamma correction
        if (Preprocessing & key).fetch1('gamma'):
#             log.info('Gamma correcting images.')
#             from staticnet_analyses import multi_mei

#             if len(multi_mei.ClosestCalibration & key) == 0:
#                 raise ValueError('No ClosestMonitorCalibration for this scan.')
#             f, f_inv = (multi_mei.ClosestCalibration & key).get_fs()

            from scipy.optimize import curve_fit
            from scipy import interpolate
            def func(x, a, b, m):
                return a + b * (x**m)

            def get_gamma_function(moncalib_key, pdcalib_key):
                # get the median pd values from the monitor calibration scan
                pixel_values, mean_pd, median_pd = (experiment.MonitorCalibration & moncalib_key).fetch1('pixel_value', 'pd_mean', 'pd_median')

                # get the most recent pd calibration trial
                pixels, lums, pds = (experiment.PhotodiodeCalibration() & pdcalib_key).fetch('pixel_value', 'luminance', 'pd_voltage', order_by='pixel_value')

                # fit a function of pd voltages and luminance 
                pd2lum_popt, _ = curve_fit(func, pds, lums)

                # fit a function of pixel values and the luminance
                px2lum_popt, _ = curve_fit(func, pixel_values, func(median_pd, *pd2lum_popt))
                px2lum_interp = interpolate.interp1d(pixel_values, func(median_pd, *pd2lum_popt))
                inv_px2lum_interp = interpolate.interp1d(func(median_pd, *pd2lum_popt), pixel_values)

                return pixel_values, px2lum_popt, px2lum_interp, inv_px2lum_interp

            moncalib_on_key = dict(animal_id=24620, session=4, scan_idx=18)
            pdcalib_on_key = dict(rig="2p4", trial=8)
            _, _, f, f_inv = get_gamma_function(moncalib_on_key, pdcalib_on_key)
            
            images = f(images)

        # get actual displayed pixels when monitor is linearized
        if (Preprocessing & key).fetch1('linear_mon'):
            lum_min = 0.01787
            lum_max = 10.2795
            lum_values = np.linspace(lum_min, lum_max, 256)
            lut = np.power((lum_values + 0.09643) /0.00027224, 1/1.905)
            from scipy import interpolate
            lin_f = interpolate.interp1d(np.linspace(0, 255, 256), lut, kind='cubic')
            images = lin_f(images)
            
        # Compute training set mean and std (now only implemented when the only one unique type is stimulus.Frame)
        def compute_train_stats(key, trials):
            train_cond_rel = stimulus.Frame * trials & 'tier = "train"'
            train_classes = (dj.U('image_class') & train_cond_rel).fetch('image_class')

            train_masks = []
            train_frames = []
            for c in train_classes:
                if c in MASKED_CLASSES: 
                    stim_tables = (TrainClass & {'image_class': c}).get_stim_table()
                    useful_tables = 0
                    for table in stim_tables:
                        if len(table * (stimulus.Frame * trials & {'image_class': c})) > 0:
                            useful_tables += 1
                            if c == 'masked_single':  # fixed mask for all images 
                                continue
#                                 mask_params = (StimulusMaskParameters & (table * train_cond_rel & {'image_class': c})).fetch1()
                            else: # mask dependent on specific image
                                masks, frames = (base.MEIMask * table * train_cond_rel & {'image_class': c}).fetch('mask', 'frame')
                    assert (useful_tables == 1), 'The current training image class exist in none or multiple tables defined in TrainClass!'

                elif c in FF_CLASSES:
                    frames = (stimulus.Frame * trials & {'image_class': c}).fetch('frame')
                    masks = [np.ones(frames[0].shape)] * len(frames)
                train_masks.append(np.stack(masks))
                train_frames.append(np.stack(frames))
                
            train_masks = np.vstack(train_masks)
            train_frames = np.vstack(train_frames)
            
            # gamma correction
            if (Preprocessing & key).fetch1('gamma'):
                log.info('Gamma correcting images.')
#                 from staticnet_analyses import multi_mei

#                 if len(multi_mei.ClosestCalibration & key) == 0:
#                     raise ValueError('No ClosestMonitorCalibration for this scan.')
#                 f, f_inv = (multi_mei.ClosestCalibration & key).get_fs()

                from scipy.optimize import curve_fit
                from scipy import interpolate
                def func(x, a, b, m):
                    return a + b * (x**m)

                def get_gamma_function(moncalib_key, pdcalib_key):
                    # get the median pd values from the monitor calibration scan
                    pixel_values, mean_pd, median_pd = (experiment.MonitorCalibration & moncalib_key).fetch1('pixel_value', 'pd_mean', 'pd_median')

                    # get the most recent pd calibration trial
                    pixels, lums, pds = (experiment.PhotodiodeCalibration() & pdcalib_key).fetch('pixel_value', 'luminance', 'pd_voltage', order_by='pixel_value')

                    # fit a function of pd voltages and luminance 
                    pd2lum_popt, _ = curve_fit(func, pds, lums)

                    # fit a function of pixel values and the luminance
                    px2lum_popt, _ = curve_fit(func, pixel_values, func(median_pd, *pd2lum_popt))
                    px2lum_interp = interpolate.interp1d(pixel_values, func(median_pd, *pd2lum_popt))
                    inv_px2lum_interp = interpolate.interp1d(func(median_pd, *pd2lum_popt), pixel_values)

                    return pixel_values, px2lum_popt, px2lum_interp, inv_px2lum_interp

                moncalib_on_key = dict(animal_id=24620, session=4, scan_idx=18)
                pdcalib_on_key = dict(rig="2p4", trial=8)
                _, _, f, f_inv = get_gamma_function(moncalib_on_key, pdcalib_on_key)
            

                train_frames = f(train_frames)

            # get actual displayed pixels when monitor is linearized
            if (Preprocessing & key).fetch1('linear_mon'):
                lum_min = 0.01787
                lum_max = 10.2795
                lum_values = np.linspace(lum_min, lum_max, 256)
                lut = np.power((lum_values + 0.09643) /0.00027224, 1/1.905)
                from scipy import interpolate
                lin_f = interpolate.interp1d(np.linspace(0, 255, 256), lut, kind='cubic')
                train_frames = lin_f(train_frames)

            # Not sure if this is ideal: compute mean and std of within the mask for each image, and then average across images
            im_mean = []
            im_std = []
            for m, i in zip(train_masks, train_frames):
                mean = (i * m).sum(axis=(-1, -2), keepdims=True) / m.sum()
                im_mean.append(mean)
                im_std.append(np.sqrt(np.sum(((i - mean) ** 2) * m, axis=(-1, -2), keepdims=True) / m.sum()))
            train_mean = np.mean(im_mean).astype(np.float32)
            train_std = np.mean(im_std).astype(np.float32)

            return train_mean, train_std
        train_mean, train_std = compute_train_stats(key, trials)

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
            dfs[t] = df.rename(str, mapping)
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
            pupil_center = pupil_center.T
            treadmill, valid_treadmill = (Treadmill & key).fetch1('treadmill', 'valid')
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

        def run_input_stats(selector, types, ix, axis=None):
            
            if len(np.unique(types)) == 1 and np.unique(types)[0] == 'stimulus.Frame':
                log.info('Computation of stats compatible with training set containing masked images is used')
                print('train_mean = {}, train_std = {}'.format(train_mean, train_std))
                
                ret = {}
                for t in np.unique(types):
                    if not np.any(ix & (types == t)):
                        continue
                    data = selector(ix & (types == t))

                    ret[t] = dict(
                        mean=train_mean,
                        std=train_std,
                        min=data.min(axis=axis).astype(np.float32),
                        max=data.max(axis=axis).astype(np.float32),
                        median=np.median(data, axis=axis).astype(np.float32)
                    )
                data = selector(ix)
                ret['all'] = dict(
                    mean=train_mean,
                    std=train_std,
                    min=data.min(axis=axis).astype(np.float32),
                    max=data.max(axis=axis).astype(np.float32),
                    median=np.median(data, axis=axis).astype(np.float32)
                )
                return ret
            
            else:
                log.warning('More than one stimulus type and computation of stats for training set containing masked images is bypassed!')
                ret = {}
                for t in np.unique(types):
                    if not np.any(ix & (types == t)):
                        continue
                    data = selector(ix & (types == t))

                    ret[t] = dict(
                        mean=data.mean(axis=axis).astype(np.float32),
                        std=data.std(axis=axis, ddof=1).astype(np.float32),
                        min=data.min(axis=axis).astype(np.float32),
                        max=data.max(axis=axis).astype(np.float32),
                        median=np.median(data, axis=axis).astype(np.float32)
                    )
                data = selector(ix)
                ret['all'] = dict(
                    mean=data.mean(axis=axis).astype(np.float32),
                    std=data.std(axis=axis, ddof=1).astype(np.float32),
                    min=data.min(axis=axis).astype(np.float32),
                    max=data.max(axis=axis).astype(np.float32),
                    median=np.median(data, axis=axis).astype(np.float32)
                )
                return ret
            
        def run_stats(selector, types, ix, axis=None):

            ret = {}
            for t in np.unique(types):
                if not np.any(ix & (types == t)):
                    continue
                data = selector(ix & (types == t))

                ret[t] = dict(
                    mean=data.mean(axis=axis).astype(np.float32),
                    std=data.std(axis=axis, ddof=1).astype(np.float32),
                    min=data.min(axis=axis).astype(np.float32),
                    max=data.max(axis=axis).astype(np.float32),
                    median=np.median(data, axis=axis).astype(np.float32)
                )
            data = selector(ix)
            ret['all'] = dict(
                mean=data.mean(axis=axis).astype(np.float32),
                std=data.std(axis=axis, ddof=1).astype(np.float32),
                min=data.min(axis=axis).astype(np.float32),
                max=data.max(axis=axis).astype(np.float32),
                median=np.median(data, axis=axis).astype(np.float32)
            )
            return ret

        # --- compute statistics
        log.info('Computing statistics on training dataset')
        response_statistics = run_stats(lambda ix: responses[ix], types, tiers == 'train', axis=0)
        if (Preprocessing & key).fetch1('trainstats_per_image'):
            log.info('Computing training input statistics by averaging across the values computed for individual images')
            input_statistics = run_input_stats(lambda ix: images[ix], types, tiers == 'train')
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
    def fill():
        _template = 'group{group_id:03d}-{animal_id}-{session}-{scan_idx}-{preproc_id}'
        for scan in StaticMultiDatasetGroupAssignment.fetch(as_dict=True):
            # Check if the scan has been added to StaticMultiDataset.Member, if not then do it
            if len(StaticMultiDataset & dict(group_id = scan['group_id'])) == 0:
                # Group id has not been added into StaticMultiDataset, thus add it
                StaticMultiDataset.insert1(dict(group_id = scan['group_id'], description = scan['description']))

            # Handle instertion into Member table
            if len(StaticMultiDataset.Member() & scan) == 0:
                StaticMultiDataset.Member().insert1(dict(scan, name = _template.format(**scan)), ignore_extra_fields=True)

    def fetch_data(self, key, key_order=None):
        assert len(self & key) == 1, 'Key must refer to exactly one multi dataset'
        ret = OrderedDict()
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
    
imagenet = dj.create_virtual_module('pipeline_imagenet', 'pipeline_imagenet')
@schema
class FakeImagenetStimuli(dj.Computed):
    definition = """
    -> imagenet.Album
    -> Preprocessing
    image_id:   int  # same image_id as in stimulus.StaticImage.Image
    row_id:     int  # oracle image repeats will have the same image_ids but unique row_ids
    ---
    image:      longblob # preprocessed image
    """
    
    def make(self, key):
        single_keys, single_images = (stimulus.StaticImage.Image * imagenet.Album.Single & key).fetch('KEY', 'image')
        oracle_keys, oracle_images = (stimulus.StaticImage.Image * imagenet.Album.Oracle & key).fetch('KEY', 'image')
        image_keys = np.concatenate([single_keys, np.tile(oracle_keys, 10)])
        images = np.concatenate([single_images, np.tile(oracle_images, 10)])
        for i, (k, im) in enumerate(zip(image_keys, images)):
            k['preproc_id'] = key['preproc_id']
            im = process_frame(k, im)
            self.insert1({**k, 'row_id': i, 'image': im})