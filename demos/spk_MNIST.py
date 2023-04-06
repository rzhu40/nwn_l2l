import numpy as np
import matplotlib.pyplot as plt
import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from nwnTorch.jn_models import *
from nwnTorch.nwn import *
from nwnTorch.generate_adj import *
from nwnTorch.misc import *

import dask
from dask.distributed import Client, LocalCluster

import os

os.environ['OMP_NUM_THREADS'] = '7'
os.environ['MKL_NUM_THREADS'] = '7'

torch.set_default_tensor_type('torch.FloatTensor')

def run_batch(net, node_shuffle, 
              batch_data):
    electrodes   = node_shuffle[:784+1]
    e_out        = node_shuffle[-100:]
    num_steps    = 25
    batch_size   = batch_data.shape[0]

    # spk_x   = spikegen.rate(data, 25, 0.5).swapaxes(0,1)
    readout = torch.zeros(batch_size, num_steps, len(e_out))

    for i in tqdm(range(batch_size)):
        # sample      = spk_x[i]
        sig_in      = torch.zeros(785)
        for t in range(num_steps):
            # sig_in[:-1] = sample[t].reshape(-1)
            sig_in[:-1] = batch_data[i,0].reshape(-1) * 2
            net.sim(sig_in.reshape(1,-1), electrodes)
            readout[i,t,:] = net.V[e_out]

    return readout

def main():
    batch_size  = 128
    data_path   = "/home/rzhu/data_access/data/mnist"
    num_classes = 10

    transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])
            
    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test  = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    
    con     = pkl_load("/home/rzhu/data_access/l2l_data/volterra_data/con0.pkl")
    adj     = torch.tensor(con["adj_matrix"])
    net     = NWN(adj, "sydney")

    net                     = NWN(adj, "sydney")
    E                       = net.number_of_junctions
    net.params["Ron"]       = 1e4
    net.params["grow"]      = 5
    net.params["decay"]     = 10
    net.params["precision"] = True
    net.params["collapse"]  = False
    net.params["dt"]        = 1e-3
    net.junction_state.L    = torch.rand(E) * 0.3 - 0.15


    cluster = LocalCluster(
                    n_workers = 2,
                    threads_per_worker = 1,
                    scheduler_port = 12121,
                    dashboard_address = 'localhost:11113',
                    )

    client = Client(cluster)
    client

    job_pool   = []
    label_pool = []
    node_shuffle = torch.randperm(1024)

    train_loader = DataLoader(mnist_train, batch_size=batch_size)
    train_batch = iter(train_loader)
    # data, label = next(train_batch)
    
    count = 0
    for batch_data, batch_label in train_batch:
        eval = dask.delayed(run_batch)(net, node_shuffle, batch_data)
        job_pool.append(eval)   
        label_pool.append(batch_label)
        # count += 1
        # if count >= 5: 
        #     break


    from_dask = client.compute(job_pool)
    collected = client.gather(from_dask)    
    print(len(collected))
    client.close()

if __name__ == "__main__":
    main()