import math
from multiprocessing import freeze_support
import pandas as pd
import numpy as np
import ast
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
import gzip
import shutil
from glob import glob
import random
from time import perf_counter
import subprocess
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

import tempfile
import pathlib

# from multiprocessing import cpu_count
# n_cores = cpu_count()

def get_hodge_numbers(df):
    h_labels = df[6]
    c_labels = df[7]

    h11, h21 = h_labels.split(':')[1].split(',')
    chi = int(c_labels[1:-1])

    return chi, int(h11), int(h21)

## PROCESS KREUSER SKARKE DATA

          

def process_all(side = "positive",  predict = "chi",num_bins = 10):
    for file in glob("*.gz"):
        strt = file.split(".")[0]
        if strt in ["v05","v06"]:
            if glob(strt+".csv"):
                t = perf_counter()
                process_files(strt+".csv", side, num_bins, predict = predict)
                print(f'Finish time: {round(perf_counter() - t, 3)}')
            else:
                t = perf_counter()
                process_files(file, side, num_bins, predict = predict)
                print(f'Finish time: {round(perf_counter() - t, 3)}')
        
    #     full_set.append(d)
    #     full_labels.append(l)
    # return full_set, full_labels

def process_files1(file, side = "positive", num_bins = 10, predict = "chi"):
    #sample = pd.read_csv(file, delimiter=' ', header=None, nrows=1, skipinitialspace=True)
    strt = file.split(".")[0]
    print(file)
    metadata = pd.read_csv(file, delimiter=' ', skiprows=lambda x: x % 5 != 0, header=None,
                    skipinitialspace=True, low_memory=True, comment='#', on_bad_lines='warn') #, nrows=1000)
    data = pd.read_csv(file, delimiter=' ', skiprows=lambda x: x % 5 == 0, header=None,
                   skipinitialspace=True, low_memory=True,
                   comment='#', on_bad_lines='warn').dropna().astype(np.float32).iloc[:,:-1]
    x = data.values
    # Get shape of individual array (4,8) from manual inspection :(
    x = np.reshape(x, (-1, 4, x.shape[1]))

    chi, h11, h21 = map(list, zip(*[get_hodge_numbers(metadata.iloc[x]) for x in range(metadata.shape[0])]))

    if predict == "chi":
        labels = np.array(chi)
        if side == "positive" :
            pos_half = np.where(labels >= 0)[0]
        elif side == "negative" and chi < 0:
            pos_half = np.where(labels < 0)[0]
        labels = labels[pos_half]
        x =  x[pos_half]
    elif predict == "h11":
        labels = np.array(h11)
    elif predict == "h21":
        labels = np.array(h21)


    bins = [i for i in range(0, 960, 960//num_bins)]
    label_bins = np.digitize(labels,bins)

    idx = np.random.choice(x.shape[0], round(0.2*x.shape[0]), replace=False)  
    x =  x[idx]
    label_bins = label_bins[idx]

    new_data = []
    new_labels = []
    for bin in range(num_bins):
        inds = np.where(label_bins == bin)
        labels = label_bins[inds]
        data = x[inds]
        print(data.shape)
        new_data.append(data)
        new_labels.append(labels)

    data = pd.DataFrame(np.reshape(np.concatenate(new_data), (-1, x.shape[2])))
    labels = pd.DataFrame(np.concatenate(new_labels).reshape(-1,1))

    base = pathlib.Path("D:\curriculum_learning_for_maths")
    (base / f"{strt}_dataset").mkdir(exist_ok=True)
    (base / f"{strt}_labels").mkdir(exist_ok=True)
    n = 100000
    table = pa.table(data[:n])
    label_table = pa.table(labels[:n])
    # Write to scratch
    chunksize = 10000
    for i in range(math.ceil(table.shape[0]/chunksize)):
        pq.write_table(table.slice(i*chunksize, (i+1)*chunksize), base / f"{strt}_dataset/data{i}.parquet")
        pq.write_table(label_table.slice(i*chunksize, (i+1)*chunksize), base / f"{strt}_labels/data{i}.parquet")
        

def process_files(file, side = "positive", num_bins = 10, predict = "chi"):
    SAMPLE_FRAC = 0.2
    strt = file.split(".")[0]
    labels = None
    rows_proc = 0
    print(file)
    if file.endswith(".gz"):
        with gzip.open(file) as f_in:
            with open(strt + '.csv', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    data_chunks = pd.read_csv((strt + '.csv'), header=None, sep='delimiter', chunksize=5000)
    i = 0
    for data in data_chunks:
        rows_proc += data.shape[0]
        if rows_proc % 1000 == 0:
            print(rows_proc)

        if data.shape[0] < 5000:
            continue
        df = data[0].str.split(expand=True)

        chi, h11, h21 = map(list, zip(*[get_hodge_numbers(df.iloc[x]) for x in range(0,df.shape[0], 5)]))
        df = df[1:]

        df = df[df.index % 5 !=0]
        num_cols = int(strt[1:])-1
        df2 = df[df.columns[:num_cols]]

        data = df2.reset_index(drop=True)
    
        arr = np.asarray(data)

        split = np.split(arr, arr.shape[0]//4)
        arr = np.asarray([np.asarray(x).astype(int) for x in split])

        if predict == "chi":
            labels = np.array(chi)
            if side == "positive" :
                pos_half = np.where(labels >= 0)[0]
            elif side == "negative" and chi < 0:
                pos_half = np.where(labels < 0)[0]
            labels = labels[pos_half]
            arr =  arr[pos_half]
        elif predict == "h11":
            labels = np.array(h11)
        elif predict == "h21":
            labels = np.array(h21)
        
        bins = [i for i in range(0, 960, 960//num_bins)]
        label_bins = np.digitize(labels,bins)

        idx = np.random.choice(arr.shape[0], round(0.2*arr.shape[0]), replace=False)  
        arr =  arr[idx]
        label_bins = label_bins[idx]

        new_data = []
        new_labels = []
        for bin in range(num_bins):
            inds = np.where(label_bins == bin)
            labels = label_bins[inds]
            data = arr[inds]
            #print(data.shape)
            new_data.append(data)
            new_labels.append(labels)
        
        arr = pd.DataFrame(np.reshape(np.concatenate(new_data), (-1, arr.shape[2])))
        labels = pd.DataFrame(np.concatenate(new_labels).reshape(-1,1))

        num = random.random()
        if num < SAMPLE_FRAC:
            base = pathlib.Path("/home/ed581//curriculum_learning")
            (base / f"{strt}_dataset").mkdir(exist_ok=True)
            (base / f"{strt}_labels").mkdir(exist_ok=True)
            n = 100000
            table = pa.table(arr)
            label_table = pa.table(labels)
            # Write to scratch
            #chunksize = 10000
            #for i in range(math.ceil(table.shape[0]/chunksize)):
            pq.write_table(table, base / f"{strt}_dataset/data{i}.parquet")
            pq.write_table(label_table, base / f"{strt}_labels/data{i}.parquet") 
            i+=1   



            # print("writing")
            # with open(strt+"_data_samples.csv", 'a', newline='') as f:
            #     pd.DataFrame(arr).to_csv(f, header=f.tell()==0)
            # with open(strt+ "_" + predict + "_label_samples.csv", 'a', newline='') as f:
            #     pd.DataFrame(np.array(labels).reshape(-1,1)).to_csv(f, header=f.tell()==0)

class KSDataset(Dataset):
    def __init__(self, data_paths, labels_paths, order):
        self.data_paths = data_paths
        self.labels_paths = labels_paths
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.order = order

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        hist = []
        dataset = ds.dataset(self.labels_paths[idx], format="parquet")
        batch_size = 500  # batch size is not necessarily chunksize
        for batch in dataset.to_batches(batch_size=batch_size):
            t = batch.to_pandas().values
            hist.append(t)
        l = np.concatenate(hist)

        hist = []
        dataset = ds.dataset(self.data_paths[idx], format="parquet")
        batch_size = 500  # batch size is not necessarily chunksize
        for batch in dataset.to_batches(batch_size=batch_size):
            batch_x = batch.to_pandas().values
            # batch_x is a batch of matrices
            #print(batch_x.shape)
            if batch_x.shape[0] > 0:
                x = batch_x.reshape(batch_x.shape[0]//4, -1, batch_x.shape[1])  # [B//4, 4, 8]
                hist.append(x)
        
        #x = x.reshape(x.shape[0]//4, -1, x.shape[1]) 
        x = np.concatenate(hist)
        if self.order == "curriculum":
            sort = np.argsort(l, axis=0)
            l = l[sort]
            x = x[sort]
        elif self.order == "anti":
            l =  np.flip(l, axis=0)
            x = np.flip(x, axis=0)
        else:
            np.random.shuffle(x)
            np.random.shuffle(l)

        return self.transform(x.copy()), l

class DatasetMaker(Dataset):
    def __init__(self, datasets):
        """
        datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
        """
        self.datasets = datasets
        self.lengths = [len(d) for d in self.datasets]
        #self.transformFunc = transformFunc

    def __getitem__(self, i):
        class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)
        img = self.datasets[class_label][index_wrt_class]
        #img = self.transformFunc(img)
        return img, class_label

    def __len__(self):
        return sum(self.lengths)

    def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
        """
        Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
        """
        # Which class/bin does i fall into?
        accum = np.add.accumulate(bin_sizes)
        if verbose:
            print("accum =", accum)
        bin_index = len(np.argwhere(accum <= absolute_index))
        if verbose:
            print("class_label =", bin_index)
        # Which element of the fallent class/bin does i correspond to?
        index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
        if verbose:
            print("index_wrt_class =", index_wrt_class)

        return bin_index, index_wrt_class

def get_class_i(x, y, i):
    """
    x: trainset.train_data or testset.test_data
    y: trainset.train_labels or testset.test_labels
    i: class label, a number between 0 to 9
    return: x_i
    """
    # Convert to a numpy array
    y = np.array(y)
    # Locate position of labels that equal to i
    pos_i = np.argwhere(y == i)
    #print(pos_i)
    # Convert the result into a 1-D list
    pos_i = list(pos_i[:, 0])
    # Collect all data that match the desired label
    #print(len(pos_i))
    x_i = [x[j] for j in pos_i]


    return x_i

def get_sized_datasets(x_train, x_test, sizes_train, sizes_test, order):    
    train = []
    test = []

    matrix_sizes = order

    for s in matrix_sizes:
        print(s)
        train.append(get_class_i(x_train, sizes_train, s))
        test.append(get_class_i(x_test, sizes_test, s))

    trainset = DatasetMaker(train)
    testset = DatasetMaker(test)

    # Create datasetLoaders from trainset and testset
    trainloader = DataLoader(
        trainset, batch_size=len(trainset), shuffle=False)
    testloader = DataLoader(
        testset, batch_size=len(testset), shuffle=False)

    return trainloader, testloader


# print(m)

process_all()
# base = pathlib.Path("D:\curriculum_learning_for_maths")
# dataset = ds.dataset(base / "v05_dataset", format="parquet")
# print(dataset.files)
