import os
import numpy as np
import torch as tc
import pandas as pd
from glob import glob

import argparse
import utils
from evaluation import mse
from evaluation import klx
from evaluation import klz
from sgvb import sgvb_model
from evaluation.pse import power_spectrum_error, power_spectrum_error_per_dim

DATA_GENERATED = None
PRINT = True
MSE_TIME_STEPS = 1000


def get_generated_data(model):
    """
    Use global variable as a way to draw trajectories only once for evaluating several metrics, for speed.
    :param model:
    :return:
    """
    global DATA_GENERATED
    if DATA_GENERATED is None:
        DATA_GENERATED = model.gen_model.get_observed_time_series_repeat(time_steps=1000, n_repeat=100).detach()
    return DATA_GENERATED


def printf(x):
    if PRINT:
        print(x)


class Evaluator(object):
    def __init__(self, init_data):
        model_ids, data, args = init_data
        self.model_ids = model_ids
       # self.save_path = 'results/test/001'
        self.save_path = args.model_path
        self.data = data
        self.args = args

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
        model = sgvb_model.Model()
        model.init_from_model_path(model_id)
        model.eval()
        return model

    def save_dict(self, metric_dict):
        df = pd.DataFrame.from_dict(data=metric_dict, orient='index')
        df.columns = self.dataframe_columns
        utils.make_dir(self.save_path)
        df.to_csv('{}/{}.csv'.format(self.save_path, self.name), sep='\t')


class EvaluateKLx_gmm(Evaluator):
    def __init__(self, init_data):
        super(EvaluateKLx_gmm, self).__init__(init_data)
        self.name = 'klx_gmm'
        self.dataframe_columns = ('klx_gmm',)

    def metric(self, model):
        data_gen = get_generated_data(model)
        klx_value = klz.calc_kl_from_data(mu_gen=data_gen, x_true=self.data)
        klx_value = float(klx_value.detach().numpy())
        printf('\tKLx {}'.format(klx_value))
        return [np.array(klx_value)]


class EvaluateKLx(Evaluator):
    def __init__(self, init_data):
        super(EvaluateKLx, self).__init__(init_data)
        self.name = 'klx'
        self.dataframe_columns = ('klx',)

    def metric(self, model):
        data_gen = get_generated_data(model)
        klx_value = klx.klx_metric(x_gen=data_gen, x_true=self.data)
        printf('\tKLx {}'.format(klx_value))
        return [np.array(klx_value)]


class EvaluateKLz(Evaluator):
    def __init__(self, init_data):
        super(EvaluateKLz, self).__init__(init_data)
        self.name = 'klz'
        self.dataframe_columns = ('klz_mc',)

    def metric(self, model):
        # klz_mc = klz.calc_kl_with_covariance_approximation(model, self.data)
        klz_mc = klz.calc_kl_with_unit_covariance(model, self.data)
        klz_mc = float(klz_mc.detach().numpy())
        printf('\tKLz mc {}'.format(klz_mc))
        return [klz_mc]


class EvaluateMSE(Evaluator):
    def __init__(self, init_data):
        super(EvaluateMSE, self).__init__(init_data)
        self.name = 'mse'
        self.n_steps = 25
        self.dataframe_columns = tuple(['{}'.format(i) for i in range(1, 1 + self.n_steps)])

    def metric(self, model):
        data = self.data[:MSE_TIME_STEPS]
        mse_results = mse.n_steps_ahead_pred_mse(model, data, n_steps=self.n_steps)
        for step in [1, 5, 10, 25]:
            printf('\tMSE-{} {}'.format(step, mse_results[step - 1]))
        return mse_results


class EvaluatePSE(Evaluator):
    def __init__(self, init_data):
        super(EvaluatePSE, self).__init__(init_data)
        self.name = 'pse'
        n_dim = self.data.shape[1]
        self.dataframe_columns = tuple(['mean'] + ['dim_{}'.format(dim) for dim in range(n_dim)])

    def metric(self, model):
        time_steps, dim_x = self.data.shape
        x_true = tc.reshape(self.data, shape=(1, time_steps, dim_x))

        data_gen = model.gen_model.get_observed_time_series(time_steps=time_steps).detach()
        x_gen = tc.reshape(data_gen, shape=x_true.shape)

        # dataset_name = get_dataset_name_from_path(args.data_path)
        # plot_save_dir = 'save_eval/{}/figures'.format(dataset_name)
        # if not os.path.exists(plot_save_dir):
        #     os.makedirs(plot_save_dir)
        # plot_save_name = '{}/{}_pse.pdf'.format(plot_save_dir, model_id)
        pse_per_dim = power_spectrum_error_per_dim(x_gen=x_gen, x_true=x_true)
        pse = power_spectrum_error(x_gen=x_gen, x_true=x_true)

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
        metrics = ['mse', 'klz', 'klx', 'pse']
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
    elif metric_name == 'klz':
        EvaluateMetric = EvaluateKLz(init_data)
    elif metric_name == 'klx':
        EvaluateMetric = EvaluateKLx(init_data)
    elif metric_name == 'klx_gmm':
        EvaluateMetric = EvaluateKLx_gmm(init_data)
    elif metric_name == 'pse':
        EvaluateMetric = EvaluatePSE(init_data)
    else:
        raise NotImplementedError
    return EvaluateMetric


def eval_model_on_data_with_metric(model, data, metric):
    init_data = (None, data, None)
    EvaluateMetric = choose_evaluator_from_metric(metric, init_data)
   # EvaluateMetric.data = data
    metric_value = EvaluateMetric.metric(model)
  #  metric_value = EvaluateMetric.metric(model, model_id)
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


def evaluate_model_path(args):
    """Evaluate a single model in directory model_path w.r.t. metrics and save results in csv file in model_path"""
    # data_path, model_path, metrics= args
    model_ids = [args.model_path]
    data = tc.cat(utils.read_data(args.data_path))

    init_data = (model_ids, data, args)
    Save = SaveArgs(init_data)
    Save.evaluate_metric()

    global DATA_GENERATED
    DATA_GENERATED = None

    EvaluateMetric = choose_evaluator_from_metric(metric_name=args.metric, init_data=(model_ids, data, args))
    EvaluateMetric.evaluate_metric()


def evaluate_all_models(eval_dir, data_path, metrics):
    model_paths = get_model_ids(eval_dir)
    n_models = len(model_paths)
    print('Evaluating {} models'.format(n_models))
    for i, model_path in enumerate(model_paths):
        print('{} of {}'.format(i + 1, n_models))
        # evaluate_model_path(data_path=data_path, model_path=model_path, metrics=metrics)
        for metric in metrics:
            args = get_args()
            args.model_path = model_path
            args.eval_dir = eval_dir
            args.data_path = data_path
            args.metric = metric
            evaluate_model_path(args)
    return


def get_dataset_name_from_path(data_path):
    return data_path.split('/')[1]


def get_parser():
    parser = argparse.ArgumentParser(description="Evaluate Trained Models")
    parser.add_argument('--name', type=str, default='results/Lorenz')
    parser.add_argument('--eval_dir', type=str, default='results')
    parser.add_argument('--metric', type=str, default='mse')
    parser.add_argument('--data_path', type=str, default='datasets/BurstingNeuron/lorenz_test.npy')
    parser.add_argument('--model_path', type=str, default='results/Lorenz')
    parser.add_argument('--plot_metric', type=bool, default=False)
    return parser


def get_args():
    parser = get_parser()
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    # available metrics: klx, klx_gmm, klz, mse, pse
    # introduce metric plot
    evaluate_model_path(args)
