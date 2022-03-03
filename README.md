# TensEarth
This repository contains a series of attempts of teaching various machine learning algorithms to simulate landscape dynamics due to erosion, tectonic forcing and various other geological phenomina. 

Each attempt consists of two notebooks: 

1. The first notebook in each attempt uses [GOSPL](https://gospl.readthedocs.io/en/latest/) to generate training data, and I reccomend running those using [this docker container](https://hub.docker.com/r/geodels/gospl).
2. The second notebook in each attempt uses [TensorFlow](https://www.tensorflow.org/) to build and train various neural network models. To run these, you will need any jupyter environment with Tensorflow installed, but you don't need GOSPL. It is desirable that your tensorflow installation has access to a GPU to significantly speed up the run time of the training algorithms. I am personnaly running this on a NVIDIA RTX 2080 with 8 Gb of dedicated graphics memory. If you have less GPU memory, you may not be able to run some of the training algorithms, or they may run slowly.

Also note that some of the training data and tensorflow models used in this project have very large file sizes and could therefore not be uploaded to GitHub. However, the code to generate the relevant training data is all contained within the notebooks, and links to download relevant training data will also be provided.
