import os
import glob
import pickle

import numpy as np
import pandas as pd
import torch as tc
from tensorboardX import SummaryWriter

from sgvb import dataset


def get_runs(trial_path):
    run_nrs = list(os.walk(trial_path))[0][1]
    run_nrs = [r[0:3] for r in run_nrs]
    return run_nrs


def create_run_path(trial_path):
    """increase by one each run, if none exists start at '000' """
    run_nrs = get_runs(trial_path)
    if not run_nrs:
        run_nrs = ['000']
    run = str(int(max(run_nrs)) + 1).zfill(3)
    run_dir = os.path.join(trial_path, run)
    return run_dir


def create_savepath(args):
    save_path = os.path.join('results', args.experiment)
    make_dir(save_path)
    trial_path = os.path.join(save_path, args.name)
    make_dir(trial_path)
    if args.run is None:
        run_path = create_run_path(trial_path)
    else:
        run_path = os.path.join(trial_path, str(args.run).zfill(3))
    make_dir(run_path)
    return run_path


def init_writer(args):
    trial_path = create_savepath(args)
    writer = None
    if args.use_tb:
        if not args.no_printing:
            print('initialize tensorboard writer at {}'.format(trial_path))
        writer = SummaryWriter(trial_path)
    return writer, trial_path


def read_data(data_path):
    if data_path is None:
        data_list = None
    else:
        assert os.path.exists(data_path)
        data = np.load(data_path, allow_pickle=True)
        assert len(data.shape) == 3, "Required data shape is (N, T, dim_x)."
        data_list = [tc.FloatTensor(data[i]) for i in range(len(data))]
    return data_list


def load_dataset(args):
    data = read_data(args.data_path)
    
    inputs = read_data(args.inputs_path)
    data_set = dataset.Dataset(data, inputs)
    args.dim_x = data[0].shape[1]
    if inputs is not None:
        args.dim_s = inputs[0].shape[1]
    else:
        args.dim_s = None
    return args, data_set


def make_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            pass


def make_attribute_dict(args):
    """"make args a dict whose elements can be accessed like: dict.element"""
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self
    args = AttrDict(args)
    return args


def save_args(args, save_path, writer):
    """ add hyperparameters to txt file """
    d = args.__dict__
    txt = ''
    for k in d.keys():
        txt += ('{} {}\n'.format(k, d[k]))
    if writer is not None:
        writer.add_text(tag="""hypers""", text_string=str(txt), global_step=None)
    filename = '{}/hypers.txt'.format(save_path)
    with open(filename, 'w') as f:
        f.write(txt)
    filename = '{}/hypers.pkl'.format(save_path)
    with open(filename, 'wb') as f:
        pickle.dump(d, f)
    if not args.no_printing:
        print(txt)


def check_args(args):
    def assert_positive_or_none(arg):
        if arg is not None:
            assert arg > 0

    assert args.data_path[-4:] == '.npy'
    assert args.data_path is not None
    assert_positive_or_none(args.clip_range)
    assert args.dim_z > 0
    if args.n_bases is not None:
        assert args.n_bases >= 0
    assert args.learning_rate > 0
    assert args.n_epochs > 0
    assert args.annealing in [None, 'None', 'lin', 'quad']
    assert args.annealing_step_size > 0, ''
    # assert args.alpha_reg >= 0
    # assert args.n_states_reg >= 0
    # list entries are tuples of floats
    # first entry is between 0 and 1 and sum of all is not higher than one
    # but higher than 0
    # second entry is > 0


def save_files(save_path):
    curdir = os.path.abspath('.')
    from distutils.dir_util import copy_tree
    save_path = os.path.join(save_path, 'python_files')
    copy_tree(curdir, save_path)


def read_csv_to_df(path):
    with open(path, 'rb') as f:
        df = pd.read_csv(f, sep='\t')
    return df


def save_to_pickle(variable, file_name):
    filename = '{}.pkl'.format(file_name)
    with open(filename, 'wb') as f:
        pickle.dump(variable, f)


def load_from_pickle(file_name):
    filename = '{}.pkl'.format(file_name)
    with open(filename, 'rb') as f:
        variable = pickle.load(f)
    return variable


