# Frame Averaging for Invariant and Equivariant Network Design
This repository contains the offical implementation of the paper "Frame Averaging for Invariant and Equivariant Network Design" (ICLR 2022).

## Abstract
Many machine learning tasks involve learning functions that are known to be
invariant or equivariant to certain symmetries of the input data. However, it is often
challenging to design neural network architectures that respect these symmetries
while being expressive and computationally efficient. For example, Euclidean
motion invariant/equivariant graph or point cloud neural networks.
We introduce Frame Averaging (FA), a general purpose and systematic framework
for adapting known (backbone) architectures to become invariant or equivariant to
new symmetry types. Our framework builds on the well known group averaging
operator that guarantees invariance or equivariance but is intractable. In contrast,
we observe that for many important classes of symmetries, this operator can be
replaced with an averaging operator over a small subset of the group elements,
called a frame. We show that averaging over a frame guarantees exact invariance
or equivariance while often being much simpler to compute than averaging over
the entire group. Furthermore, we prove that FA-based models have maximal
expressive power in a broad setting and in general preserve the expressive power
of their backbone architectures. Using frame averaging, we propose a new class
of universal Graph Neural Networks (GNNs), universal Euclidean motion invariant point cloud networks, and Euclidean motion invariant Message Passing (MP)
GNNs. We demonstrate the practical effectiveness of FA on several applications including point cloud normal estimation, beyond 2-WL graph separation, and n-body
dynamics prediction, achieving state-of-the-art results in all of these benchmarks.

For more details, see: [https://arxiv.org/abs/2110.03336](https://arxiv.org/abs/2110.03336).

## Installation Requirmenets
The code is compatible with python 3.8 and pytorch 1.7. Conda environment file is provided at ``env.yml``

## Usage

The repository contains implementation to the experiments from the paper: normal estimation for point clouds, the n-body experiment and graph separation tasks. To run on of the experiments go to the relevant folder, for example for the n-body experiment:
```
cd nbody
```
and follow the instrouctions in the folder's ``.md`` file.

## Citation 
```
@inproceedings{
puny2022frame,
title={Frame Averaging for Invariant and Equivariant Network Design},
author={Omri Puny and Matan Atzmon and Edward J. Smith and Ishan Misra and Aditya Grover and Heli Ben-Hamu and Yaron Lipman},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=zIUyj55nXR}
}
```
