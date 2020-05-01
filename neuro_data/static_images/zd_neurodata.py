import datajoint as dj
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
from scipy import stats
import torch

from neuro_data.utils.measures import corr
from .configs import DataConfig

from .data_schemas import StaticMultiDataset, StaticScan
from . import data_schemas
from .. import logger as log

from staticnet_experiments import models
from staticnet_experiments.configs import NetworkConfig, Seed, TrainConfig
from staticnet_experiments.utils import correlation_closure, compute_predictions, compute_scores

schema = dj.schema('zhiwei_neuro_data')

def oracle_score(key, tier):
    # --- load data
    testsets, testloaders = DataConfig().load_data(key=key, tier=tier, oracle=True)
    
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
        pearson = corr(np.vstack(data), np.vstack(oracles), axis=0)

        member_key = (StaticMultiDataset.Member() & key &
                      dict(name=readout_key)).fetch1(dj.key)
        member_key = dict(member_key, **key)

        unit_ids = testsets[readout_key].neurons.unit_ids
        assert len(unit_ids) == len(
            pearson) == outputs.shape[-1], 'Neuron numbers do not add up'

        return member_key, pearson, unit_ids


@schema
class Oracle(dj.Computed):
    definition = """
    # oracle computation for static images

    -> StaticMultiDataset
    -> DataConfig
    ---
    """

    @property
    def key_source(self):
        return StaticMultiDataset() * DataConfig()

    class Scores(dj.Part):
        definition = """
        -> master
        -> StaticMultiDataset.Member
        ---
        ff_pearson           : float     # mean test correlation using full-field oracles
        mask_pearson         : float     # mean test correlation using masked oracles
        avg_pearson          : float     # average of ff_pearson and mask_pearson
        """

    class UnitScores(dj.Part):
        definition = """
        -> master.Scores
        -> StaticScan.Unit
        ---
        unit_ff_pearson           : float     # unit test correlation using full-field oracles
        unit_masked_pearson       : float     # unit test correlation using masked oracles
        unit_avg_pearson          : float     # average of unit_ff_pearson and unit_masked_pearson 
        """

    def make(self, key):

        self.insert1(dict(key))
        member_key, ff_pearson, ff_unit_ids = oracle_score(key, 'test')
        _, masked_pearson, masked_unit_ids = oracle_score(key, 'test_masked')
        # import pdb; pdb.set_trace()
        avg_pearson = (ff_pearson + masked_pearson) / 2
        assert ((ff_unit_ids == masked_unit_ids).all()), "Unit ids for two types of oracles do not match!"

        self.Scores().insert1(dict(member_key, ff_pearson=np.mean(ff_pearson), mask_pearson=np.mean(masked_pearson), avg_pearson=np.mean(avg_pearson)), ignore_extra_fields=True)
        self.UnitScores().insert(
            [dict(member_key, unit_ff_pearson=f, unit_masked_pearson=m, unit_avg_pearson=a, unit_id=u)
             for f, m, a, u in tqdm(zip(ff_pearson, masked_pearson, avg_pearson, ff_unit_ids), total=len(ff_unit_ids))],
            ignore_extra_fields=True)

@schema
class NewOracle(dj.Computed):
    definition = """
    # oracle computation for static images

    -> StaticMultiDataset
    -> DataConfig
    ---
    """

    @property
    def key_source(self):
        return StaticMultiDataset() * DataConfig()

    class Scores(dj.Part):
        definition = """
        -> master
        -> StaticMultiDataset.Member
        ---
        leave_one_out_pearson           : float     # mean test pearson correlation using leave-one-out 
        leave_one_out_spearman          : float     # mean test spearmon correlation using leave-one-out
        split_half_pearson              : float     # mean test pearson correlation using split_half
        split_half_spearman             : float     # mean test spearman correlation using split_half
        """

    class UnitScores(dj.Part):
        definition = """
        -> master.Scores
        -> StaticScan.Unit
        ---
        leave_one_out_pearson           : float     
        leave_one_out_spearman          : float    
        split_half_pearson              : float # mean pearson correlation over 1000 random splits
        split_half_spearman             : float # mean spearman correlation over 1000 random splits
        split_half_pearson_pval         : float # p value for t-test comparison of 1000 corr distribution from data and shuffled data
        split_half_spearman_pval        : float # p value for t-test comparison of 1000 corr distribution from data and shuffled data
        """

    def make(self, key):
        # --- load data
        testsets, testloaders = DataConfig().load_data(key, tier='test', oracle=True)

        self.insert1(dict(key))
        for readout_key, loader in testloaders.items():
            log.info('Computing oracle for ' + readout_key)
            oracles, data, filled_data = [], [], []
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
                filled = outputs.copy()
                if r != 10:
                    filled = np.concatenate([(np.nan * np.ones(n)).reshape(1, -1), filled])
                filled_data.append(filled)
            if len(data) == 0:
                log.error('Found no oracle trials! Skipping ...')
                return
            # leave-one-out pearson
            lou_pearson = corr(np.vstack(data), np.vstack(oracles), axis=0)
            # leave-one-out spearman
            data_rank = np.empty(np.vstack(data).shape)
            oracles_rank = np.empty(np.vstack(oracles).shape)

            for i in range(np.vstack(data).shape[1]):
                data_rank[:, i] = np.argsort(np.argsort(np.vstack(data)[:, i]))
                oracles_rank[:, i] = np.argsort(np.argsort(np.vstack(oracles)[:, i]))
            lou_spearman = corr(data_rank, oracles_rank, axis=0)

            # split-half
            filled_data = np.stack(filled_data)
            original = filled_data.copy()
            shuffled = np.empty(filled_data.shape)

            # Completely shuffle image * trial matrix for each neuron
            np.random.seed(0)
            for i in range(filled_data.shape[2]):
                a = filled_data[:, :, i].ravel()
                np.random.shuffle(a)
                shuffled[:, :, i] = a.reshape(filled_data.shape[0], filled_data.shape[1])

            import scipy
            all_pearson, all_spearman, all_s_pearson, all_s_spearman, pearson_pval, spearman_pval = [], [], [], [], [], []
            for i in range(filled_data.shape[2]):
                o = original[:, :, i]
                s = shuffled[:, :, i]
                pearson, spearman, s_pearson, s_spearman = [], [], [], []
                for j in range(1000):
                    np.random.seed(j)
                    idx1 = np.random.choice(filled_data.shape[1], int(filled_data.shape[1]/2), replace=False)
                    idx2 = list(set(range(filled_data.shape[1])) - set(idx1))
                    pearson.append(scipy.stats.pearsonr(np.nanmean(o[:, idx1], axis=1), np.nanmean(o[:, idx2], axis=1))[0])
                    spearman.append(scipy.stats.spearmanr(np.nanmean(o[:, idx1], axis=1), np.nanmean(o[:, idx2], axis=1))[0])
                    s_pearson.append(scipy.stats.pearsonr(np.nanmean(s[:, idx1], axis=1), np.nanmean(s[:, idx2], axis=1))[0])
                    s_spearman.append(scipy.stats.spearmanr(np.nanmean(s[:, idx1], axis=1), np.nanmean(s[:, idx2], axis=1))[0])

                all_pearson.append(np.stack(pearson))
                all_spearman.append(np.stack(spearman))
                all_s_pearson.append(np.stack(s_pearson))
                all_s_spearman.append(np.stack(s_spearman))
                pearson_pval.append(scipy.stats.ttest_ind(np.stack(pearson), np.stack(s_pearson))[1])
                spearman_pval.append(scipy.stats.ttest_ind(np.stack(spearman), np.stack(s_spearman))[1])

            split_pearson = np.mean(np.stack(all_pearson), 1)
            split_spearman = np.mean(np.stack(all_spearman), 1)

            member_key = (StaticMultiDataset.Member() & key &
                          dict(name=readout_key)).fetch1(dj.key)
            member_key = dict(member_key, **key)
            self.Scores().insert1(dict(member_key, leave_one_out_pearson=np.mean(lou_pearson),
                                                   leave_one_out_spearman=np.mean(lou_spearman),
                                                   split_half_pearson=np.mean(split_pearson),
                                                   split_half_spearman=np.mean(split_spearman)), ignore_extra_fields=True)
            unit_ids = testsets[readout_key].neurons.unit_ids
            assert len(unit_ids) == len(lou_pearson) == len(lou_spearman) == len(split_pearson) == len(split_spearman) == outputs.shape[-1], 'Neuron numbers do not add up'
            self.UnitScores().insert(
                [dict(member_key, unit_id=u, leave_one_out_pearson=lp, leave_one_out_spearman=ls, split_half_pearson=sp, split_half_spearman=ss, split_half_pearson_pval=ppval, split_half_spearman_pval=spval)
                 for u, lp, ls, sp, ss, ppval, spval in tqdm(zip(unit_ids, lou_pearson, lou_spearman, split_pearson, split_spearman, pearson_pval, spearman_pval), total=len(unit_ids))],
                ignore_extra_fields=True)


@schema
class Model(dj.Computed):
    definition = """
    -> models.Model
    ---
    val_corr     : float     
    """

    class TestScores(dj.Part):
        definition = """
        -> master
        -> models.Model.TestScores
        ---
        neurons                  : int         # number of neurons
        ff_pearson               : float       # test correlation on full-field oracle single trial responses
        masked_pearson           : float       # test correlation on masked oracle single trial responses
        avg_pearson              : float       # average of ff_pearson and masked_pearson
        """

    class UnitTestScores(dj.Part):
        definition = """
        -> master.TestScores
        -> models.Model.UnitTestScores
        ---
        unit_ff_pearson                  : float       # single unit test correlation on full-field oracle single trial responses
        unit_masked_pearson              : float       # single unit test correlation on masked oracle single trial responses
        unit_avg_pearson                 : float       # average of ff_pearson and masked_pearson
        """
        
    def make(self, key):
        member_key, ff_unit_ids, ff_pearson, unit_ff_pearson, masked_pearson, unit_masked_pearson = self.evaluate(key)
        avg_pearson = (ff_pearson + masked_pearson) / 2
        unit_avg_pearson = (unit_ff_pearson + unit_masked_pearson) / 2
        updated_key = (models.Model & key).proj('val_corr').fetch1()
        self.insert1(updated_key)
        self.TestScores.insert1({**member_key, 'neurons': len(ff_unit_ids), 'ff_pearson': ff_pearson, 'masked_pearson': masked_pearson, 'avg_pearson': avg_pearson})
        self.UnitTestScores.insert([{**member_key, 'unit_id': u, 'unit_ff_pearson': f, 'unit_masked_pearson': m, 'unit_avg_pearson': a} for u, f, m, a in zip(ff_unit_ids, unit_ff_pearson, unit_masked_pearson, unit_avg_pearson)])
    
    def load_network(self, key=None, trainsets=None):
        if key is None:
            key = self.fetch1(dj.key)
        model = NetworkConfig().build_network(key, trainsets=trainsets)
        state_dict = (models.Model & key).fetch1('model')
        state_dict = {k: torch.as_tensor(state_dict[k][0].copy()) for k in state_dict.dtype.names}
        mod_state_dict = model.state_dict()
        for k in set(mod_state_dict) - set(state_dict):
            log.warning('Could not find paramater {} setting to initialization value'.format(repr(k)))
            state_dict[k] = mod_state_dict[k]
        model.load_state_dict(state_dict)
        return model

        
    def evaluate(self, key=None):
        if key is None:
            key = self.fetch1('KEY')

        model = self.load_network(key)
        model.eval()
        model.cuda()

        # get network configuration information
        net_key = NetworkConfig().net_key(key)
        train_key = TrainConfig().train_key(net_key)
        
        def compute_test_corr(net_key, tier, train_key):
            testsets, testloaders = DataConfig().load_data(net_key, tier=tier, cuda=True, **train_key)

            scores, unit_scores = [], []
            for readout_key, testloader in testloaders.items():
                log.info('Computing test scores for ' + readout_key)

                y, y_hat = compute_predictions(testloader, model, readout_key)
                perf_scores = compute_scores(y, y_hat)

                member_key = (StaticMultiDataset.Member() & key & dict(name=readout_key)).fetch1(dj.key)
                member_key.update(key)

                unit_ids = testloader.dataset.neurons.unit_ids
                pearson = perf_scores.pearson.mean()

            return member_key, unit_ids, pearson, perf_scores.pearson
        # import pdb; pdb.set_trace()
        member_key, ff_unit_ids, ff_pearson, unit_ff_pearson = compute_test_corr(net_key, 'test', train_key)
        _, masked_unit_ids, masked_pearson, unit_masked_pearson = compute_test_corr(net_key, 'test_masked', train_key)
        assert ((ff_unit_ids == masked_unit_ids).all()), "Unit ids for two types of oracles do not match!"
        
        return member_key, ff_unit_ids, ff_pearson, unit_ff_pearson, masked_pearson, unit_masked_pearson