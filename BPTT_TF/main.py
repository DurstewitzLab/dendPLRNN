import argparse
import torch as tc
import utils

from bptt import bptt_algorithm

from bptt.PLRNN_model import PLRNN

tc.set_num_threads(1)

def get_parser():
    parser = argparse.ArgumentParser(description="TF RNN Training")
    parser.add_argument('--experiment', type=str, default="example")
    parser.add_argument('--name', type=str, default='Lorenz63')
    parser.add_argument('--run', type=int, default=None)

    # gpu
    parser.add_argument('--use_gpu',
        type=int,
        choices=[0, 1],
        help="If set to 1 (True), use GPU for training.",
        default=0
    )
    # cuda:0, cuda:1 etc.
    parser.add_argument('--device_id',
        type=int,
        help="Set the GPU device id (as determined by ordering in nvidia-smi) when use_gpu==1.",
        default=0
    )

    # general settings
    parser.add_argument('--no_printing', type=bool, default=True)
    parser.add_argument('--use_tb', type=bool, default=True)
    parser.add_argument('--metrics',
        type=str,
        nargs='+',
        help="Set metrics that are regularly computed (MSE, PSE, KLx).",
        default=['mse', 'pse', 'klx']
    )

    # dataset
    parser.add_argument('--data_path',
        type=str,
        help="Path to the .npy file that is used as training data.",
        default='examples/Lorenz63/lorenz.npy'
    )
    # TODO: inputs not implemented yet
    parser.add_argument('--inputs_path', type=str, default=None)
    # resume from a model checkpoint
    parser.add_argument('--load_model_path', type=str, default=None)
    # epoch is inferred if None
    parser.add_argument('--resume_epoch', type=int, default=None)

    # model
    parser.add_argument('--fix_obs_model', '-fo',
        type=int,
        help="Fix B to be of shape used for Id. TF.",
        default=1
    )
    parser.add_argument('--dim_z',
        type=int,
        help="Dimension of latent state vector of the model", 
        default=15
    )
    parser.add_argument('--n_bases', '-nb',
        type=int,
        help="Number of bases to use in dendr-PLRNN and clipped-PLRNN latent models.",
        default=5
    )
    parser.add_argument('--clip_range', '-clip',
        type=float,
        help="Clips the latent state vector to the support of [-clip_range, clip_range].",
        default=10
    )
    # currently no special function, ought to change when LSTM etc. are implemented...
    parser.add_argument('--model', '-m',
        type=str,
        default='PLRNN'
    )
    # specifiy which latent model to choose (only affects model PLRNN)
    parser.add_argument('--latent_model', '-ml',
        type=str,
        help="Latent steps to use, i.e. vanilla PLRNN, dendr-PLRNN, clipped-PLRNN.",
        choices=PLRNN.LATENT_MODELS,
        default='PLRNN'
    )
    # This has to be changed to mean-centering at some point, but keeping the name for now
    # due to compatibility reasons...
    parser.add_argument('--mean_centering', '-mc',
        type=int,
        help="Mean Centering for vanilla PLRNN to train." \
             " Re-centers latent state vector at each time step.",
        default=1
    )
    parser.add_argument('--learn_z0', '-z0',
        type=int,
        help="If set to 1 (True), jointly learn a mapping from data" \
             "to initial cond. for remaining latent states.",
        default=1
    )

    # BPTT
    parser.add_argument('--use_inv_tf', '-itf',
        type=int,
        help="'Invert' the B matrix to force latent states. (Inv. TF).",
        default=0
    )
    parser.add_argument('--estimate_forcing', '-ef',
        type=int,
        help="If set to 1 (True), estimate a forcing interval" \
             "using Autocorrelation & Mutual Information.",
        default=0
    )
    parser.add_argument('--noise_level', '-nl',
        type=float,
        help="Gaussian noise added to the teacher signals.",
        default=0.05
    )
    # maybe rename to tau someday..
    parser.add_argument('--teacher_forcing_interval', '-tfi',
        type=int,
        help="Teacher forcing interval (tau).",
        default=15
    )
    parser.add_argument('--batch_size', '-bs',
        type=int,
        help="Sequences are gathered as batches of this size (computed in parallel).",
        default=16
    )
    parser.add_argument('--batches_per_epoch', '-bpi',
        type=int,
        help="Amount of sampled batches that correspond to 1 epoch of training.",
        default=50
    )
    parser.add_argument('--seq_len', '-sl',
        type=int,
        help="Sequence length sampled from the total pool of the data.",
        default=200
    )
    parser.add_argument('--save_step', '-ss',
        type=int,
        help="Interval of computing and saving metrics to be stored to TB.",
        default=25
    )
    parser.add_argument('--save_img_step', '-si',
        type=int,
        help="Interval of saving images to TB and model parameters to storage.",
        default=25
    )

    # optimization
    parser.add_argument('--learning_rate', '-lr',
        type=float,
        help="Global Learning Rate.",
        default=1e-3
    )
    parser.add_argument('--n_epochs', '-n', type=int, default=5000)
    parser.add_argument('--gradient_clipping', '-gc',
        type=int,
        help="Gradient norm clip value for gradient clipping (GC). Value of 0 corresponds to no GC."
             "A value of 1 corresponds to enabling GC, where the optimal clipping value is estimated. This takes some time (training for 15 epochs)."
             "Specifying a value > 1 manually sets the GC value.",
        default=10
    )

    # regularization
    parser.add_argument('--use_reg', '-r',
        type=int,
        help="If set to 1 (True), use Manifold Attractor Regularization (MAR).",
        default=0
    )
    parser.add_argument('--reg_ratios', '-rr',
        nargs='*',
        help="Ratio of states to regularize. A value of 0.5 corresponds to 50% of dim_z regularized.",
        type=float,
        default=[0.5]
    )
    parser.add_argument('--reg_alphas', '-ra',
        nargs='*',
        help="Regularization weighting, determines the strength of regularization.",
        type=float,
        default=[1e-2]
    )
    parser.add_argument('--reg_norm', '-rn',
        type=str,
        help="Regularization norm. L2 -> standard, L1 -> sparsification of W, h (and pulls A stricter to 1).",
        choices=['l2', 'l1'],
        default='l2'
    )
    return parser


def get_args():
    parser = get_parser()
    return parser.parse_args()


def train(args):
    # prepare training device
    device = utils.prepare_device(args)

    writer, save_path = utils.init_writer(args)
    args, data_set = utils.load_dataset(args)

    utils.check_args(args)
    utils.save_args(args, save_path, writer)

    training_algorithm = bptt_algorithm.BPTT(args, data_set, writer, save_path,
                                             device)
    training_algorithm.train()
    return save_path


def main(args):
    train(args)


if __name__ == '__main__':
    main(get_args())

