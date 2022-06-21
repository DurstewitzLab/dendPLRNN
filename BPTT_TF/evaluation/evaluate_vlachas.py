import glob
import pickle
import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch as tc

import utils
from evaluation.pse import power_spectrum_error
from evaluation.klx import klx_metric


class Metric:
    def __init__(self, name, columns):
        self.name = name
        self.columns = columns


Metric_mse = Metric('mse', ['1', '5', '25'])
Metric_klx = Metric('klx', ['klx'])
Metric_pse = Metric('pse', ['pse'])


class EvaluateVlachas:
    def __init__(self, paths):
        self.path_to_eval_data, self.path_to_dataset, self.save_path = paths
        self.mse_dict = dict()
        self.klx_dict = dict()
        self.pse_dict = dict()

    def evaluate_all_models(self):
        for model_path in sorted(glob.glob(os.path.join(self.path_to_eval_data, '*'))):
            self.evaluate_model(model_path)

    def evaluate_model(self, model_path):
        model_results = load_from_pickle(os.path.join(model_path, 'results.pickle'))
        model_id = id_from_path(model_path)
        self.mse_dict[model_id] = evaluate_mse(model_results)
        self.klx_dict[model_id] = evaluate_klx(model_results)
        self.pse_dict[model_id] = evaluate_pse(model_results)
        self.save_dicts()

    def save_dicts(self):
        for metric, metric_dict in [(Metric_mse, self.mse_dict), (Metric_klx, self.klx_dict), (Metric_pse, self.pse_dict)]:
            self.save_dict(metric, metric_dict)

    def save_dict(self, metric, metric_dict):
        df = pd.DataFrame.from_dict(data=metric_dict, orient='index')
        df.columns = metric.columns
        utils.make_dir(self.save_path)
        df.to_csv('{}/{}.csv'.format(self.save_path, metric.name), sep='\t')


def plot_lorenz_true_vs_gen(x_true, x_gen):
    plt.scatter(x_true[:, 0], x_true[:, 2], s=1, label='ground truth')
    plt.scatter(x_gen[:, 0], x_gen[:, 2], s=1, label='generated')
    plt.legend()
    plt.show()


def load_from_pickle(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def concat(x):
    return np.concatenate(x, axis=0)


def get_x_true():
    dataset_path = '../datasets/lorenz/lorenz_data.npy'
    assert os.path.exists(dataset_path)
    return np.load(dataset_path)


def get_x_gen(model_results):
    trajectories = model_results['predictions_all_TEST']
    return trajectories[:, :, :]


def evaluate_mse(model_results):
    mse_results = model_results['mse_all_TEST']
    return [mse_results[:, n - 1].mean(0) for n in [1, 5, 25]]


def evaluate_klx(model_results):
    x_true = concat(get_x_true())
    x_gen = concat(get_x_gen(model_results))
    # plot_lorenz_true_vs_gen(x_true, x_gen)
    x_true = tc.Tensor(x_true)
    x_gen = tc.Tensor(x_gen)
    return klx_metric(x_gen=x_gen, x_true=x_true)


def id_from_path(model_path):
    return model_path.split('/')[-1]  # e.g. ESN_001


def evaluate_pse(model_results):
    x_gen = get_x_gen(model_results)
    x_true = get_x_true()
    return power_spectrum_error(x_gen, x_true)


def print_means(save_path):
    for metric in [Metric_mse, Metric_klx, Metric_pse]:
        path = os.path.join(save_path, '{}.csv'.format(metric.name))
        df = pd.read_csv(path, sep='\t')
        for column in metric.columns:
            data = df['{}'.format(column)]
            print('{} {}'.format(metric.name, column))
            print('{} ± {}'.format(data.mean(), data.std() / np.sqrt(len(data))))


def save_dict(metric, metric_dict):
    df = pd.DataFrame.from_dict(data=metric_dict, orient='index')
    df.columns = metric.columns
    utils.make_dir(save_path)
    df.to_csv('{}/{}.csv'.format(save_path, metric.name), sep='\t')


if __name__ == '__main__':
    # import collections
    path_to_eval_data = '../../svae_comparison/vlachas-rnn-rc/Results/Lorenz/Evaluation_Data/'
    path_to_dataset = 'datasets/lorenz/lorenz_pn.01_on.01_test'
    save_path = 'save_vlachas_rc'
    # paths = collections.namedtuple('paths', ['path_to_eval_data', 'path_to_dataset', 'save_path'])
    EvaluateVlachas(paths=(path_to_eval_data, path_to_dataset, save_path)).evaluate_all_models()
    print_means(save_path)
