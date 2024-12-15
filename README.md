# Topic
Artificial neural network classifier trained by backpropagation for face recognition (Famous48 dataset).

# Technology used
- Python 3.11
- Tensorflow, Keras - python machine learning library


# Face Recognition with Famous48 Dataset

## Overview
This repository contains code for a face recognition system using the Famous48 dataset. The project demonstrates training, evaluation, and inference of a deep learning model to classify faces of celebrities in the dataset. It is intended as a starting point for exploring face recognition techniques and neural network architectures.

## Works done
- We first load the dataset from raw text files [load_dataset.py](utils/load_dataset.py)
- We tried to train the model on version of AlexNet adapted to size of images (24x24) [alexnet.ipynb](alexnet.ipynb)
- We experimented with LeNet-5 architecture which is designed for small images [lenet.ipynb](lenet.ipynb)
- We decided to adjust hyperparameters with optuna framework [optuna_hyperparams_tunning.ipynb](optuna_hyperparams_tunning.ipynb)
- Then, we tested the LeNet-5 architecture with found parameters from previous step [after_optuna.ipynb](after_optuna.ipynb)

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Pawcharz/face-recognition-famous48.git
cd face-recognition-famous48
```

## Contact
For questions or contributions, please contact:
- GitHub: [Pawcharz](https://github.com/Pawcharz)

---
Enjoy exploring face recognition!
