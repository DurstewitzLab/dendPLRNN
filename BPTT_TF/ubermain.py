from multitasking import *


def ubermain(n_runs):
    """
    Specify the argument choices you want to be tested here in list format:
    e.g. args.append(Argument('dim_z', [5, 6], add_to_name_as='z'))
    will test for dimensions 5 and 6 and save experiments under z5 and z6.
    Possible Arguments can be found in main.py
    
    When using GPU for training (i.e. Argument 'use_gpu 1')  it is generally
    not necessary to specify device ids, tasks will be distributed automatically.
    """
    args = []
    args.append(Argument('experiment', ['ubermain_example']))
    args.append(Argument('data_path', ['examples/Lorenz63/lorenz.npy']))
    args.append(Argument('dim_z', [15]))
    args.append(Argument('fix_obs_model', [1]))
    args.append(Argument('layer_norm', [1]))
    args.append(Argument('learn_z0', [1]))
    args.append(Argument('n_epochs', [1000]))
    args.append(Argument('n_interleave', [15, 30], add_to_name_as="tau"))
    args.append(Argument('seq_len', [200]))
    args.append(Argument('latent_model', ['PLRNN']))
    args.append(Argument('learning_rate', [1e-3]))
    args.append(Argument('run', list(range(1, 1 + n_runs))))
    return args


if __name__ == '__main__':
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # number of runs for each experiment
    n_runs =1
    # number of runs to run in parallel
    n_cpu = 1
    # number of processes run parallel on a single GPU
    n_proc_per_gpu = 1

    args = ubermain(n_runs)
    run_settings(*create_tasks_from_arguments(args, n_proc_per_gpu, n_cpu))
