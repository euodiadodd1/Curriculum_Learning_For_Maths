from glob import glob
import pathlib
from random import sample
import torch
import argparse
import torch.nn as  nn
from load_KS_data import KSDataset, get_sized_datasets, process_all, process_files
from models.linear_model import make_model
from models.resnet50 import ResNet50
import numpy as np
from models.simple_cnn import Net
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

import tempfile


from train import train_model
from utils import plot_metrics

parser = argparse.ArgumentParser()

parser.add_argument('--order', type=str, default='random',
                    help='name of learning strategy (curriculum, anti, random)')
parser.add_argument('--model', type=str, default='cnn',
                    help='model to train with')
parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--num_bins', type=int, default=10, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--lr', type=float, default=1e-6,
                    help='initial learning rate (default: 1e-6)')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='number of epochs (default: 2)')
parser.add_argument('--batch_increase', type=int, default=10,
                    help='number of batches before increasing perc of data used')
parser.add_argument('--interval', type=int, default=10,
                    help='log ever n batches')
parser.add_argument('--increment', type=float, default=1.05,
                    help='pace * increment = perc data used')
parser.add_argument('--starting_percentage', type=float, default=0.004,
                    help='perc data used to begin with')

args = parser.parse_args()

np.random.seed(1)
hyp_params = args
hyp_params.lr = args.lr
hyp_params.num_epochs = args.num_epochs
hyp_params.criterion = nn.CrossEntropyLoss()
hyp_params.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.model == "resnet50":
    model = ResNet50(args.num_bins, channels = 1)
elif args.model == "linear":
    model = make_model(5,[64])
elif args.model == "cnn":
    model = Net()

results = []

for odr in ["curriculum", "anti", "random"]:

    data_paths = [i for i in glob("*_dataset")]
    label_paths = [i for i in glob("*_labels")]

    data_paths.sort()
    label_paths.sort()
    
    # print(split)
    split = round(0.8*len(data_paths))
    train_data_paths = data_paths[:split]
    test_data_paths = data_paths[split:]
    print(train_data_paths)
    print(test_data_paths)
    train_label_paths = label_paths[:split]
    test_label_paths = label_paths[split:]
    # ## Sort by label
    train_set = KSDataset(train_data_paths, train_label_paths, "anti")
    test_set = KSDataset(test_data_paths, test_label_paths, "anti")
    
    print(train_set[0][1])
    # # train_set = SortedDataset(train_set, hyp_params.num_bins)
    # # test_set = SortedDataset(test_set, hyp_params.num_bins)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=hyp_params.batch_size,
                                            shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=hyp_params.batch_size,
                                            shuffle=True, num_workers=0)

    #metrics = train_model(hyp_params, train_set, test_set, model)
    # results.append(metrics)

plot_metrics(results)