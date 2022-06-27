# SVAE

Sequential Variational Autoencoder (SVAE) to infer parameters for the dendritic Piece-wise Linear Recurrent Neural Network (dendPLRNN) model by [Brenner, Hess, 2022].
Data input needs to be a numpy file with shape (N, T, dim_x), where N is the number of trials/batches, and T the number of time steps per sequence.


## Arguments:
- data_path: path to data. Format is .npy as a numpy array of numpy arrays, each contained array is one batch of data.
- input_path: path to external input data. Format is same as for data from data_path.
- load_model_path: path to saved model folder, to be loaded as an initialization with a pre-trained model.

- use_tb (bool):  write loss and plots to tensorboard
- eval (bool): evaluate models in terms of ahead prediction MSE, KLx, and KLz
- eval_dir: directory in which recursively all models will be searched and eval results gathered 

- dim_z: dimension of latent space
- n_bases: number of bases for dendritic basis expansion, if it is *None*,  defaults to standard PLRNN formulation
- clip_range: clip the latent states to a specified maximum and minimum value
- rec_model: recognition model type
  dc: (linear layer) diagonal covariance, cnn: convolutional neural network

- annealing ['lin', 'quad', 'exp', 'const']
    arguments to annealing should be given as list:
    e.g. ['lin'] (linear) or ['quad'] (quadratic)
    for exponential (exp) and constant (const) are with a factor:
    e.g. ['const', 0.8] wich means a constant annealing alpha of 0.8,
    or  ['exp', 10] which is the exponential decay constant
- annealing_step_size
    step function of annealing, if 1,  annealing increase every epoch, if 10 every 10 epochs

- reg (line attractor regularization) 
    list of tuples: tuple[0]: ratio of states to be regularized, tuple[1]: respective regularization constant
    e.g. if [[0.5, 1.], [0.3, 0.1]] and the total number of states is 10,
    then 5 states are regularized with strength 1.0 and 3 states with strength 0.1 and 2 are not regularized

## ubermain

- The ubermain file allows parallel scanning over ranges of hyperparameters. Specify the argument choices to be tested in list format:
e.g. args.append(Argument('dim_z', [5, 6], add_to_name_as='z')) will test for dimensions 5 and 6 and save experiments under z5 and z6.
- n_runs gives the number of runs per settings.
- n_cpu selects the number of cpu's used.

## Requirements 
conda install pytorch
conda install pandas
conda install matplotlib
conda install -c conda-forge tensorboardx
conda install tensorboard
conda install Pillow
