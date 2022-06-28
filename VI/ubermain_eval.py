from main_eval import get_model_ids, gather_eval_results
from experiments.helpers.multitasking import Argument, create_tasks_from_arguments, run_settings

"""
Parallelized evaluation of models from eval_dir
"""


def ubermain(eval_dir, data_path, metrics):

    args = []
    args.append(Argument('eval_dir', [eval_dir]))
    args.append(Argument('data_path', [data_path]))
    args.append(Argument('metric', metrics, add_to_name_as=''))
    model_paths = get_model_ids(eval_dir)
    args.append(Argument('model_path', model_paths, add_to_name_as=''))
    return args


if __name__ == '__main__':
    n_cpu = 5
    eval_dir = '/home/manuel.brenner/Code/svae_dendPLRNN/results/lorenz'
    data_path = "/home/manuel.brenner/Code/svae_dendPLRNN/datasets/lorenz_test.npy"
    metrics = ['klx', 'mse', 'pse']

    args = ubermain(eval_dir=eval_dir, data_path=data_path, metrics=metrics)
    tasks = create_tasks_from_arguments(args, run_file='main_eval.py', python_version='python')
    run_settings(tasks, n_cpu)

    gather_eval_results(eval_dir=eval_dir, metrics=metrics)

