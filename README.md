# NA21B075-DA6401-Assignment2
This repository contains code for training a Convolutional Neural Network (CNN) using PyTorch and optimizing its performance using Weights and Biases (wandb) for hyperparameter sweeps, logging, and visualization.

**Part A**
This folder includes the following files:

**Q1,2,3.ipynb**

This script implements a CNN model with 5 convolutional layers using PyTorch. It uses wandb sweep functionality to automatically search for the best hyperparameters such as learning rate, batch size, number of epochs, optimizer, etc. The model is trained on the dataset, and metrics like training and validation accuracy are logged to the wandb dashboard.

**Q4,5.py/.ipynb**

This script is focused on visualizing the results of the best model obtained from the hyperparameter sweep. It includes two key components:
A 10 x 3 grid of sample images from the test set along with the predictions made by the best model.
Visualization of all filters from the first convolutional layer for a randomly selected test image to help understand what kind of features the model has learned.

**Part B**
This folder contains code for fine-tuning pretrained ResNet50 models using transfer learning on the iNaturalist dataset. It includes three fine-tuning strategies:
Freezing all layers except the final classification layer.
Unfreezing only the last two blocks of ResNet (layer3 and layer4).
Unfreezing all layers and training the entire model.

The code integrates with wandb to perform a sweep over different hyperparameter configurations including learning rate, batch size, optimizer type, number of epochs, and fine-tuning strategy. It logs performance metrics and helps visualize how each hyperparameter affects validation accuracy.

The sweep results can be viewed on the wandb dashboard, and help in selecting the best performing model configuration.
