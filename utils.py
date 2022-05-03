from random import randrange
import torch
import numpy as np
import matplotlib.pyplot as plt

## Pacing function

def data_pacing_function(data_set, batch_increase, increment, starting_percentage, curr_batch, current_pace, batch_size=100):

    X_train, y_train = data_set
    #print(len(y_train))
    X_train = torch.unsqueeze(X_train, 1)
    #y_train = torch.unsqueeze(y_train, 0)
    #print("SHAPE", X_train.shape)
    
    data_size = X_train.shape[0]
    pace = current_pace

    if curr_batch % batch_increase == 0:
      if curr_batch == 0:
        pace = starting_percentage
      else:
        pace = min(pace*increment, 1)

    pacing_idx = int(np.ceil(pace * data_size))
    #print(pacing_idx)

    new_X_train = X_train[:pacing_idx, :,:,:]
    new_y_train = y_train[:pacing_idx]

    return new_X_train, new_y_train, pace

def naive_pacing_function(data_set, epoch, bin, increment):
    new_data = new_labels = []
    for x,y in data_set:
        X_train = x
        y_train = y
    
        inds = np.where(y_train == bin)
        #print(inds, bin)
        labels = y_train[inds]
        data = X_train[inds]
        #print(data.shape)
        new_data.append(data)
        new_labels.append(labels)
    #print(len(new_data))
    idx = randrange(len(new_data))
    return new_data[idx], new_labels[idx]

def generate_random_batch(x, y, batch_size):
    size_data = x.shape[0]
    cur_batch_idxs = np.random.choice(size_data, batch_size, replace=False)
    return x[cur_batch_idxs, :, :, :], torch.Tensor(y[cur_batch_idxs].reshape(-1, 1))

def get_class(dataset, epoch, interval):
    l = 0
    x = y = []
    if epoch % interval != 0:
      x,y = dataset[l]

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def plot_metrics(results):
  for metrics in results:
    train_acc = metrics["train_loss"]
    test_acc = metrics["test_loss"]
    batch = metrics["batch"]
    print(len(test_acc))
    plt.figure(2,figsize=(8, 8))
    plt.plot(batch, train_acc, label = metrics["curriculum"])
    
    plt.legend(loc='best')
    plt.ylabel('Test Loss')
    plt.xlabel('Global rounds')
  plt.show