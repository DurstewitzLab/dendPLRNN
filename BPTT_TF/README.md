# **DendPLRNN: BPTT+TF Training**

## Setup
Install your anaconda distribution of choice, e.g. miniconda via the bash
script ```miniconda.sh```:
```
$ ./miniconda.sh
```
Create the local environment `BPTT_TF`:
```
$ conda env create -f environment.yml
```
Activate the environment and install the package
```
$ conda activate BPTT_TF
$ pip install -e .
```

## Running the code
### <u>Reproducing Table 1</u>
The folder `Experiments` contains ready-to-run examples for reproducing results found in Table 1 of the paper. That is, to run trainings on the Lorenz63 data set using the dendritic PLRNN, run the ubermain.py file in the corresponding directory `Experiments/Table1/Lorenz63`:
```
$ python Experiments/Table1/Lorenz63/ubermain.py
```
In the `ubermain.py` file of each subfolder, adjustments regarding the hyperparameter can be performed (all parser arguments can be found in `main.py`), as well as running multiple parallel runs from different initial paramter configurations, by setting `n_runs = x`, where `x` is the number of runs. Setting `n_cpu = x` will ensure that `x` processes are spawned to handle all the runs.

Running `ubermain.py` will create a `results` folder, in which models and tensorboard information will be stored. To track training or inspect trained models of e.g. the Lorenz63 runs produced by running the template `ubermain.py` file mentioned above using tensorboard, call 
```
$ tensorboard --logdir results/Lorenz63/M22B20tau25T200
```

To evaluate metrics on the test set, call `main_eval.py` with the corresponding results subfolder passed to the parser argument `-p`:
```
$ python main_eval.py -p results/Lorenz63/M22B20tau25T200
```

Finally, the jupyter notebook `example.ipynb` contains code that loads a trained model and plots trajectories, where `model_path` and `data_path` have to be set by the user.

### <u>Reproducing Table S2</u>
...



## Software Versions
* Python 3.9
* PyTorch 1.11 + cudatoolkit v11.3
