from typing import List

from pandas.core.indexes import numeric
from evaluation.klx_gmm import calc_kl_from_data
import os
import numpy as np
import torch as tc
import pandas as pd
from glob import glob

import utils
from evaluation import mse
from evaluation import klx
from bptt import models
from evaluation.pse import power_spectrum_error, power_spectrum_error_per_dim

EPOCH = None
SKIP_TRANS = 0
PRINT = True
SMOOTHING = 20
CUTOFF = 20000
INITCONDS = 100

def printf(x):
    if PRINT:
        print(x)


class Evaluator(object):
    def __init__(self, init_data):
        model_ids, data, save_path = init_data
        self.model_ids = model_ids
        self.save_path = save_path

        self.data = tc.tensor(data[SKIP_TRANS:], dtype=tc.float)

        self.name = NotImplementedError
        self.dataframe_columns = NotImplementedError

    def metric(self, model):
        return NotImplementedError

    def evaluate_metric(self):
        metric_dict = dict()
        assert self.model_ids is not None
        for model_id in self.model_ids:
            model = self.load_model(model_id)
            metric_dict[model_id] = self.metric(model)
        self.save_dict(metric_dict)

    def load_model(self, model_id):
        model = models.Model(data_set = self.data)
        model.init_from_model_path(model_id, EPOCH)
        model.eval()
        print(f"# params: {model.get_num_trainable()}")
        return model

    def save_dict(self, metric_dict):
        df = pd.DataFrame.from_dict(data=metric_dict, orient='index')
        df.columns = self.dataframe_columns
        utils.make_dir(self.save_path)
        df.to_csv('{}/{}.csv'.format(self.save_path, self.name), sep='\t')


class EvaluateKLx(Evaluator):
    def __init__(self, init_data):
        super(EvaluateKLx, self).__init__(init_data)
        self.name = 'klx'
        self.dataframe_columns = ('klx',)

    def metric(self, model):
        self.data = self.data.to(model.device)

        global INITCONDS

        # EEG during training
        if self.data.shape[0] <= 10000:
            INITCONDS = 10

        # generate traj
        T, N = self.data.shape
        data_reshaped = self.data.reshape(INITCONDS, -1, N)
        T_sub = data_reshaped.shape[1]
        trajs = []
        for traj in data_reshaped:
            X, Z = model.generate_free_trajectory(traj, T_sub)
            trajs.append(X)
        data_gen = tc.stack(trajs).reshape(T, -1)
         
        # compute D_stsp
        if N < 5:
            print("Computing KLx-BIN")
            klx_value = klx.klx_metric(data_gen.reshape(-1, N), self.data, n_bins=30).cpu()
        else:
            print("Computing KLx-GMM")
            klx_value = calc_kl_from_data(data_gen.cpu().reshape(-1, N), self.data.cpu())

        printf('\tKLx {}'.format(klx_value.item()))
        return [np.array(klx_value.numpy())]

    @staticmethod
    def pca(x_gen, x_true):
        '''
        perform pca for to make KLx-Bin feasible for
        high dimensional data. Computes the first 5 principal
        components.
        '''
        U, S, V = tc.pca_lowrank(x_true, q=5, center=False, niter=10)
        x_pca = x_true @ V[:, :5]
        x_gen_pca = x_gen @ V[:, :5]
        return x_gen_pca, x_pca


class EvaluateMSE(Evaluator):
    def __init__(self, init_data):
        super(EvaluateMSE, self).__init__(init_data)
        self.name = 'mse'
        self.n_steps = 20
        self.dataframe_columns = tuple(['{}'.format(i) for i in range(1, 1 + self.n_steps)])

    def metric(self, model):
        mse_results = mse.n_steps_ahead_pred_mse(model, self.data, n_steps=self.n_steps)
        for step in [1, 5, 10, 20]:
            printf('\tMSE-{} {}'.format(step, mse_results[step-1]))
        return mse_results


class EvaluatePSE(Evaluator):
    def __init__(self, init_data):
        super(EvaluatePSE, self).__init__(init_data)
        self.name = 'pse'
        n_dim = self.data.shape[1]
        self.dataframe_columns = tuple(['mean'] + ['dim_{}'.format(dim) for dim in range(n_dim)])

    def metric(self, model):
        self.data = self.data.to(model.device)
        #data_gen = get_generated_data(model, self.data)
        data_gen, _ = model.generate_free_trajectory(self.data, len(self.data))

        x_gen = data_gen.cpu().unsqueeze(0).numpy()
        x_true = self.data.cpu().unsqueeze(0).numpy()
        print(SMOOTHING, CUTOFF)
        pse_per_dim = power_spectrum_error_per_dim(x_gen=x_gen, x_true=x_true, smoothing=SMOOTHING, cutoff=CUTOFF)
        pse = np.mean(pse_per_dim)

        printf('\tPSE {}'.format(pse))
        printf('\tPSE per dim {}'.format(pse_per_dim))
        return [pse] + pse_per_dim


class SaveArgs(Evaluator):
    def __init__(self, init_data):
        super(SaveArgs, self).__init__(init_data)
        self.name = 'args'
        self.dataframe_columns = ('dim_x', 'dim_z', 'n_bases')

    def metric(self, model):
        args = model.args
        return [args['dim_x'], args['dim_z'], args['n_bases']]


def gather_eval_results(eval_dir='save', save_path='save_eval', metrics=None):
    """Pre-calculated metrics in individual model directories are gathered in one csv file"""
    if metrics is None:
        metrics = ['klx', 'pse']
    metrics.append('args')
    model_ids = get_model_ids(eval_dir)
    for metric in metrics:
        paths = [os.path.join(model_id, '{}.csv'.format(metric)) for model_id in model_ids]
        data_frames = []
        for path in paths:
            try:
                data_frames.append(pd.read_csv(path, sep='\t', index_col=0))
            except:
                print('Warning: Missing model at path: {}'.format(path))
        data_gathered = pd.concat(data_frames)
        utils.make_dir(save_path)
        metric_save_path = '{}/{}.csv'.format(save_path, metric)
        data_gathered.to_csv(metric_save_path, sep='\t')


def choose_evaluator_from_metric(metric_name, init_data):
    if metric_name == 'mse':
        EvaluateMetric = EvaluateMSE(init_data)
    elif metric_name == 'klx':
        EvaluateMetric = EvaluateKLx(init_data)
    elif metric_name == 'pse':
        EvaluateMetric = EvaluatePSE(init_data)
    else:
        raise NotImplementedError
    return EvaluateMetric


def eval_model_on_data_with_metric(model, data, metric):
    init_data = (None, data, None)
    EvaluateMetric = choose_evaluator_from_metric(metric, init_data)
    #EvaluateMetric.data = data
    metric_value = EvaluateMetric.metric(model)
    return metric_value


def is_model_id(path):
    """Check if path ends with a three digit, e.g. save/test/001 """
    run_nr = path.split('/')[-1]
    three_digit_numbers = {str(digit).zfill(3) for digit in set(range(1000))}
    return run_nr in three_digit_numbers


def get_model_ids(path):
    """
    Get model ids from a directory by recursively searching all subdirectories for files ending with a number
    """
    assert os.path.exists(path)
    if is_model_id(path):
        model_ids = [path]
    else:
        all_subfolders = glob(os.path.join(path, '**/*'), recursive=True)
        model_ids = [file for file in all_subfolders if is_model_id(file)]
    assert model_ids, 'could not load from path: {}'.format(path)
    return model_ids


def eval_model(args):
    save_path = args.load_model_path
    evaluate_model_path(args, model_path=save_path, metrics=args.metrics)


def evaluate_model_path(data_path, model_path=None, metrics=None):
    """Evaluate a single model in directory model_path w.r.t. metrics and save results in csv file in model_path"""
    model_ids = [model_path]
    data = utils.read_data(data_path)
    init_data = (model_ids, data, model_path)
    Save = SaveArgs(init_data)
    Save.evaluate_metric()

    for metric_name in metrics:
        EvaluateMetric = choose_evaluator_from_metric(metric_name=metric_name, init_data=(model_ids, data, model_path))
        EvaluateMetric.evaluate_metric()


def evaluate_all_models(eval_dir, data_path, metrics):
    model_paths = get_model_ids(eval_dir)
    n_models = len(model_paths)
    print('Evaluating {} models'.format(n_models))
    for i, model_path in enumerate(model_paths):
        print('{} of {}'.format(i+1, n_models))
        # try:
        evaluate_model_path(data_path=data_path, model_path=model_path, metrics=metrics)
        # except:
        #     print('Error in model evaluation {}'.format(model_path))
    return

def print_metric_stats(results_path: str, save_path: str, metrics: List):

    d_ = {}
    # MSE
    if "mse" in metrics:
        path = os.path.join(results_path, 'mse.csv')
        df = pd.read_csv(path, delimiter='\t')
        mse5 = (df.mean(0, numeric_only=True)['5'], df.std(numeric_only=True)['5'])
        mse10 = (df.mean(0, numeric_only=True)['10'], df.std(numeric_only=True)['10'])
        mse20 = (df.mean(0, numeric_only=True)['20'], df.std(numeric_only=True)['20'])
        d_['5-MSE'] = mse5
        d_['10-MSE'] = mse10
        d_['20-MSE'] = mse20


    #PSE
    if "pse" in metrics:
        path = os.path.join(results_path, 'pse.csv')
        df = pd.read_csv(path, delimiter='\t')
        pse = (df.mean(0, numeric_only=True)['mean'], df.std(numeric_only=True)['mean'])
        d_['PSC'] = pse

    # Dstsp
    if "klx" in metrics:
        path = os.path.join(results_path, 'klx.csv')
        df = pd.read_csv(path, delimiter='\t')
        df_sub = df['klx']
        df_sub = df_sub[df_sub > 0]
        klx = (df_sub.mean(0), df_sub.std(0))
        d_['D_stsp'] = klx

    new_df = pd.DataFrame(d_)
    new_df.to_csv(os.path.join(save_path, 'eval_stats.csv'), sep='\t')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", "-p", type=str) # e.g. "results/BurstingNeuron/M26B47tau05T500"
    args = parser.parse_args()

    eval_dir = args.results_path

    # PSE SETTINGS
    if "Lorenz63" in eval_dir:
        SMOOTHING = 20
        CUTOFF = 20000
        INITCONDS = 100
        save_path = "Experiments/Table1/Lorenz63"
        data_path = "Experiments/Table1/Lorenz63/data/lorenz63_test.npy"
    if "Lorenz63_lowdata" in eval_dir:
        SMOOTHING = 20
        CUTOFF = 20000
        INITCONDS = 100
        save_path = "Experiments/Table1/Lorenz63_lowdata"
        data_path = "Experiments/Table1/Lorenz63_lowdata/data/lorenz63_lowdata_test.npy"
    if "Lorenz63_noisy" in eval_dir:
        SMOOTHING = 20
        CUTOFF = 20000
        INITCONDS = 100
        save_path = "Experiments/Table1/Lorenz63_noisy"
        data_path = "Experiments/Table1/Lorenz63_noisy/data/lorenz63_noisy_test.npy"
    if "Lorenz63_TDE" in eval_dir:
        SMOOTHING = 20
        CUTOFF = 20000
        INITCONDS = 100
        save_path = "Experiments/Table1/Lorenz63_TDE"
        data_path = "Experiments/Table1/Lorenz63_TDE/data/lorenz63_TDE_test.npy"
    if "BurstingNeuron" in eval_dir:
        SMOOTHING = 20
        CUTOFF = 20000
        INITCONDS = 100
        save_path = "Experiments/Table1/BurstingNeuron"
        data_path = "Experiments/Table1/BurstingNeuron/data/burstingneuron_test.npy" 
    if "NeuralPopulation" in eval_dir:
        SMOOTHING = 20
        CUTOFF = 20000
        INITCONDS = 100
        save_path = "Experiments/Table1/NeuralPopulation"
        data_path = "Experiments/Table1/NeuralPopulation/data/neuralpopulation_test.npy" 
    if "Lorenz96" in eval_dir:
        SMOOTHING = 10
        CUTOFF = 20000
        INITCONDS = 100
        save_path = "Experiments/Table1/Lorenz96"
        data_path = "Experiments/Table1/Lorenz96/data/lorenz96_test.npy"
    if "EEG" in eval_dir:
        SMOOTHING = 1
        CUTOFF = 1500
        INITCONDS = 10
        save_path = "Experiments/Table1/EEG"
        # EEG_test == EEG_train
        data_path = "Experiments/Table1/EEG/data/EEG_test.npy" 
    if "ECG" in eval_dir:
        SMOOTHING = 100
        CUTOFF = 10000
        INITCONDS = 100
        save_path = "Experiments/Table1/ECG"
        data_path = "Experiments/Table1/ECG/data/ECG_test.npy" 
    
    metrics = ['pse', 'mse', 'klx']
    
    evaluate_all_models(eval_dir=eval_dir, data_path=data_path, metrics=metrics)
    gather_eval_results(eval_dir=eval_dir, save_path=eval_dir, metrics=metrics)

    print(print_metric_stats(eval_dir, save_path, metrics))


