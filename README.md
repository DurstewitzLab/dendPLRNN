# **Tractable Dendritic RNNs for Reconstructing Nonlinear Dynamical Systems** [ICML 2022 Spotlight]
![alt text for screen readers](images/dendrites.png "Augmenting RNN units with dendrites to increase computational power. Image credit goes to Manuel Brenner & Darshana Kalita.")
## About

This repository provides the code to the paper **Tractable Dendritic RNNs for Reconstructing Nonlinear Dynamical Systems** as accepted at the [ICML 2022](https://icml.cc/Conferences/2022). This work augments RNN units with elements from dendritic computation to increase their computational capabilites in turn allowing for low dimensional RNNs. We apply these models to the field of dynamical systems reconstruction, where low-dimensional representations of the underlying system are very much desired. 

The repository is split into two codebases providing different approaches to the estimation of parameters of the dendritic, piecewise linear recurrent neural network (dendPLRNN). The folder `BPTT_TF` contains the codebase using backpropagation through time (BPTT) based training paired with sparse teacher forcing (TF), whereas `VI` embeds the dendPLRNN in a variational inference (VI) framework in the form of a sequential variational autoencoder (SVAE). All code is written in Python using [PyTorch](https://pytorch.org/) as the main deep learning framework.

## Citation
If you find the repository and/or paper helpful for your own research, please cite [our work](https://proceedings.mlr.press/v162/brenner22a.html):
```

@InProceedings{pmlr-v162-brenner22a,
  title = 	 {Tractable Dendritic {RNN}s for Reconstructing Nonlinear Dynamical Systems},
  author =       {Brenner, Manuel and Hess, Florian and Mikhaeil, Jonas M and Bereska, Leonard F and Monfared, Zahra and Kuo, Po-Chen and Durstewitz, Daniel},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {2292--2320},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/brenner22a/brenner22a.pdf},
  url = 	 {https://proceedings.mlr.press/v162/brenner22a.html},
  abstract = 	 {In many scientific disciplines, we are interested in inferring the nonlinear dynamical system underlying a set of observed time series, a challenging task in the face of chaotic behavior and noise. Previous deep learning approaches toward this goal often suffered from a lack of interpretability and tractability. In particular, the high-dimensional latent spaces often required for a faithful embedding, even when the underlying dynamics lives on a lower-dimensional manifold, can hamper theoretical analysis. Motivated by the emerging principles of dendritic computation, we augment a dynamically interpretable and mathematically tractable piecewise-linear (PL) recurrent neural network (RNN) by a linear spline basis expansion. We show that this approach retains all the theoretically appealing properties of the simple PLRNN, yet boosts its capacity for approximating arbitrary nonlinear dynamical systems in comparatively low dimensions. We employ two frameworks for training the system, one combining BPTT with teacher forcing, and another based on fast and scalable variational inference. We show that the dendritically expanded PLRNN achieves better reconstructions with fewer parameters and dimensions on various dynamical systems benchmarks and compares favorably to other methods, while retaining a tractable and interpretable structure.}
}

```

## Acknowledgements
This work was funded by the German Research Foundation (DFG) within Germany’s Excellence Strategy – EXC-2181 – 390900948 (’Structures’), by DFG grant Du354/10-1 to DD, and the European Union Horizon-2020 consortium SC1-DTH-13-2020 ('IMMERSE').
