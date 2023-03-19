import torch
from tqdm import tqdm
# import matplotlib.pyplot as plt

from nwn import *
from jn_models import *
from misc import *
from utils import *
# data_path = "/home/ruomin_zhu/snn_data/"
# volterra_path = "/home/ruomin_zhu/old/volterra_data/"
# volterra_path = "/project/NASN/rzhu/l2l_data/volterra_data/"

def learn_snn(net, 
              hyper_params,
              _test = False):
    from utils import data_path
    tensor_dict = {}
    dtype_here  = torch.get_default_dtype()
    for key in hyper_params.keys():
        tensor_dict[key] = torch.tensor(hyper_params[key], dtype=dtype_here)

    idx         = torch.randint(100, size = (1,1)).item()
    data_dict   = pkl_load(snn_data_path+f"snn_mem_{idx}.pkl")
    lambda_dict = pkl_load(snn_data_path+"lambda_data.pkl")
    waves       = data_dict["waves"]
    mems        = data_dict["mems"]
    n_neurons   = mems.shape[-1]
    num_inputs  = waves.shape[1]
    num_steps   = waves.shape[0]
    num_read    = 64

    torch.manual_seed(0)
    shuffled = torch.randperm(1024)
    e_in     = shuffled[:num_inputs]
    e_read   = shuffled[-num_read:]
    readout  = torch.zeros(num_steps, num_read)
    
    W_in = tensor_dict["W_in"] * 3
    b_in = tensor_dict["b_in"]
    net.junction_state.L = lambda_dict["lambda"][2500]
    # net.junction_state.L = lambda_dict["lambda"]\
                # [int(tensor_dict["init_time"] * 1000)]

    for i in tqdm(range(num_steps)):
        sig_in = W_in * waves[i] + b_in
        net.sim(sig_in.reshape(1,-1), e_in)
        readout[i,:] = net.V[e_read]

    result = torch.zeros(n_neurons)
    mse    = torch.zeros(n_neurons)
    
    out_dict = {}
    if _test:
        out_dict["target"] = mems
        out_dict["predict"] = torch.zeros(mems.shape)
    for i in range(n_neurons):
        lhs = readout[:,:]
        rhs = mems[:,i]
        weight, mse[i], rcond = best_regress(lhs, rhs)
        predict = weight @ lhs.T
        if _test:
            out_dict["predict"][:,i] = predict
        result[i] = get_RNMSE(predict, rhs)

    out_dict["set"]    = idx 
    out_dict["mse"]    = mse
    out_dict["rnmse"]  = result
    out_dict["e_in"]   = e_in
    out_dict["e_read"] = e_read
    out_dict["params"] = tensor_dict

    print(result)
    # print(out_dict)
    return result.mean(), out_dict

if __name__ == "__main__":

    net   = prepare_network()
    hyper = {
        "W_in"     : torch.rand(1,20) * 3,
        "b_in"     : torch.rand(1,20),
        "init_time": 0.5,
    }

    fitness = learn_snn(net, hyper)