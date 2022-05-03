import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters for our network

# Build a feed-forward network
def make_model(input_size, hidden_sizes):
    model = nn.Sequential(
                          nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Flatten(),
                          nn.Linear(hidden_sizes[0], 10)
                          
                      )
    return model
                     