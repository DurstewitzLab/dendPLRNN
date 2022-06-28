from experiments.helpers.multitasking import Argument, create_tasks_from_arguments, run_settings

import os
import glob


def is_train_path(data_path):
    return data_path.split('_')[-1] == 'train.npy'


def get_train_path(dataset_name):
    data_paths = glob.glob(os.path.join('Data', dataset_name, '*'))
    test_data_path = [data_path for data_path in data_paths if is_train_path(data_path)][0]
    return test_data_path


def get_lorenz_hyperparameters(n_runs):
    args = []
    args.append(Argument('n_bases', [0, 2, 5, 10], add_to_name_as='b'))
    args.append(Argument('dim_z', [4, 6, 8, 10, 12], add_to_name_as='z'))
    args.append(Argument('use_tb', [True]))
    args.append(Argument('rec_model', ['dc']))
    args.append(Argument('n_epochs', [300]))
    args.append(Argument('clip_range', [None]))
    args.append(Argument('annealing', [None]))
    args.append(Argument('reg_ratios', [1.]))
    args.append(Argument('reg_alphas', [1.]))
    args.append(Argument('run', list(range(1, 1 + n_runs))))
    return args


def get_lorenz96_hyperparameters(n_runs):
    args = []
    args.append(Argument('n_bases', [0, 5, 10, 20, 50], add_to_name_as='b'))
    args.append(Argument('dim_z', [20, 25, 30, 50], add_to_name_as='z'))
    args.append(Argument('use_tb', [True]))
    args.append(Argument('rec_model', ['dc']))
    args.append(Argument('n_epochs', [2000]))
    args.append(Argument('clip_range', [10.]))
    args.append(Argument('annealing', [None]))
    args.append(Argument('reg_ratios', [1.]))
    args.append(Argument('reg_alphas', [1.]))
    args.append(Argument('run', list(range(1, 1 + n_runs))))
    return args


def get_wilsoncowan_hyperparameters(n_runs):
    args = []
    args.append(Argument('n_bases', [0, 5, 10, 20, 50], add_to_name_as='b'))
    args.append(Argument('dim_z', [15, 25, 50], add_to_name_as='z'))
    args.append(Argument('use_tb', [True]))
    args.append(Argument('rec_model', ['dc']))
    args.append(Argument('n_epochs', [1000]))
    args.append(Argument('clip_range', [10.]))
    args.append(Argument('annealing', [None]))
    args.append(Argument('reg_ratios', [.9]))
    args.append(Argument('reg_alphas', [.3]))
    args.append(Argument('run', list(range(1, 1 + n_runs))))
    return args


def append_specific_args(args, dataset_name):
    if dataset_name == 'Lorenz96':
        args = get_lorenz96_hyperparameters(args)
    elif dataset_name == 'WilsonCowan':
        args = get_wilsoncowan_hyperparameters(args)
    else:
        args = get_lorenz_hyperparameters(args)
    return args


def get_args(dataset_name, n_runs):
    args = []
    args.append(Argument('name', [dataset_name], add_to_name_as=''))
    args.append(Argument('data_path', [get_train_path(dataset_name)]))
    args = append_specific_args(args, dataset_name)
    args.append(Argument('run', list(range(1, 1 + n_runs))))
    return args


if __name__ == '__main__':
    n_runs = 20
    n_cpu = 80

    datasets = ['Lorenz']
    tasks = []
    for dataset in datasets:
        tasks.extend(create_tasks_from_arguments(get_args(dataset, n_runs)))
    run_settings(tasks, n_cpu)
