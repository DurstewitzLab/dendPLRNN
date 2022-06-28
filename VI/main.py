import argparse
import torch as tc
import utils

from sgvb import sgvb_algorithm

tc.set_num_threads(1)


def get_parser():
    parser = argparse.ArgumentParser(description="Estimate Dynamical System")
    parser.add_argument('--experiment', type=str, default='test')
    parser.add_argument('--name', type=str, default='lorenz')
    parser.add_argument('--run', type=int, default=None)

    # general settings
    parser.add_argument('--no_printing', type=bool, default=True)
    parser.add_argument('--use_tb', type=bool, default=True)
    parser.add_argument('--metrics', type=list, default=['mse, klz, pse'])

    # dataset
    parser.add_argument('--data_path', type=str, default="C:/Users/manub/Code/Datasets/Lorenz/lorenz_pn.01_on.01_train.npy")
    parser.add_argument('--inputs_path', type=str, default=None)
    parser.add_argument('--load_model_path', type=str, default=None)

    # model
    parser.add_argument('--fix_observation_model', type=bool, default=False)
    parser.add_argument('--rec_model', type=str, choices=['dc', 'cnn'], default='dc')
    parser.add_argument('--dim_z', type=int, default=20)
    parser.add_argument('--n_bases', '-nb', type=int, default=35)
    parser.add_argument('--clip_range', '-clip', type=float, default=10)

    # optimization
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--n_epochs', '-n', type=int, default=200)
    parser.add_argument('--annealing', '-a', type=str, default=None)
    parser.add_argument('--annealing_step_size', '-ass', type=int, default=1)
    parser.add_argument('--gradient_clipping', '-gc', type=float, default=1.)

    # regularization
    parser.add_argument('--reg_ratios', '-rr', nargs='*', type=float, default=[1.])
    parser.add_argument('--reg_alphas', '-ra', nargs='*', type=float, default=[.1])
    return parser


def get_args():
    parser = get_parser()
    return parser.parse_args()


def train(args):
    writer, save_path = utils.init_writer(args)
    args, data_set = utils.load_dataset(args)

    utils.check_args(args)
    utils.save_args(args, save_path, writer)

    training_algorithm = sgvb_algorithm.SGVB(args, data_set, writer, save_path)
    training_algorithm.train()
    return save_path


def main(args):
    print("Start Training")
    train(args)


if __name__ == '__main__':
    main(get_args())

