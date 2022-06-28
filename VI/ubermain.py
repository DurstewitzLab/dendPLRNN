from multitasking import Argument, create_tasks_from_arguments, run_settings


def ubermain(n_runs):
    """
    Specify the argument choices you want to be tested here in list format:
    e.g. args.append(Argument('dim_z', [5, 6], add_to_name_as='z'))
    will test for dimensions 5 and 6 and save experiments under z5 and z6
    """
    args = []
    args.append(Argument('experiment', ['lorenz']))
    args.append(Argument('data_path', ["/home/manuel.brenner/Code/svae_dendPLRNN/datasets/lorenz_train.npy"]))
    args.append(Argument('n_bases', [20], add_to_name_as='b'))
    args.append(Argument('dim_z', [15, 20, 25], add_to_name_as='z'))
    args.append(Argument('use_tb', [True]))
    args.append(Argument('rec_model', ['dc']))
    args.append(Argument('n_epochs', [1000]))
    args.append(Argument('clip_range', [10.]))
    args.append(Argument('annealing', [None]))
    args.append(Argument('reg_ratios', [1.]))
    args.append(Argument('reg_alphas', [1.]))
    args.append(Argument('run', list(range(1, 1 + n_runs))))
    return args


if __name__ == '__main__':
    #choose number of runs per setting
    n_runs = 5
    #choose number of cpu's for parallel processing
    n_cpu = 2
    args = ubermain(n_runs)
    run_settings(create_tasks_from_arguments(args), n_cpu)
