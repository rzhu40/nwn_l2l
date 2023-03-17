import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append("/home/ruomin_zhu/nwn_l2l/nwnTorch/")
from nwn import *
from jn_models import *
from misc import *

data_path = "/home/ruomin_zhu/snn_data/"
# volterra_path = "/home/ruomin_zhu/old/volterra_data/"
# volterra_path = "/project/NASN/rzhu/l2l_data/volterra_data/"

def learn_snn(net, hyper_params):
    tensor_dict = {}
    dtype_here  = torch.get_default_dtype()
    for key in hyper_params.keys():
        tensor_dict[key] = torch.tensor(hyper_params[key], dtype=dtype_here)


    data_dict   = pkl_load(data_path+"snn_mem.pkl")
    lambda_dict = pkl_load(data_path+"lambda_data.pkl")
    waves       = data_dict["waves"]
    mems        = data_dict["mems"]
    n_neurons   = mems.shape[-1]
    num_inputs  = waves.shape[1]
    num_steps   = waves.shape[0]
    num_read    = 64

    torch.manual_seed(0)
    shuffled = torch.randperm(1024)
    e_in     = shuffled[:num_inputs]
    e_read   = shuffled[-64:]
    readout  = torch.zeros(num_steps, 64)
    
    W_in = tensor_dict["W_in"]
    b_in = tensor_dict["b_in"]
    net.junction_state.L = lambda_dict["lambda"]\
                [int(tensor_dict["init_time"] * 1000)]


    out_dict = {}
    for i in tqdm(range(num_steps)):
        sig_in = W_in * waves[i] + b_in
        net.sim(sig_in.reshape(1,-1), e_in)
        readout[i,:] = net.V[e_read]

    result = torch.zeros(n_neurons)
    # fig, ax = plt.subplots(n_neurons,1,)
    for i in range(n_neurons):
        lhs = readout[:,:]
        # rhs = spk2_rec[:,0,1].detach()
        rhs = mems[:,0,i]

        weight, result[i], rcond = best_regress(lhs, rhs)
        predict = weight @ lhs.T

        # ax[4-i].plot(rhs)
        # ax[4-i].plot(predict, ".", ms = 3)
        # ax[4-i].set_ylabel(f"neuron {i}")
        # ax[i].set_ylabel(f"{result.item():.4}")
    out_dict["e_in"] = e_in
    out_dict["e_read"] = e_read
    out_dict["params"] = tensor_dict
    # ax[4].set_ylabel("neuron 0")
    # fig.supxlabel('steps')
    # fig.supylabel('membrane potential (V)')

    print(result)
    print(out_dict)
    return result, out_dict
def prepare_network(index = 0):
    adj = torch.tensor(pkl_load(data_path + "con0.pkl")["adj_matrix"])
    net = NWN(adj, "sydney")

    net.params["Ron"]       = 1e4
    net.params["grow"]      = 5
    net.params["decay"]     = 10
    net.params["precision"] = True
    return net

if __name__ == "__main__":

    net   = prepare_network()
    hyper = {
        "W_in"     : torch.rand(1,20) * 3,
        "b_in"     : torch.rand(1,20),
        "init_time": 0.5,
    }


    fitness = learn_snn(net, hyper)