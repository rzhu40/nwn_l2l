import torch
from tqdm import tqdm
# import matplotlib.pyplot as plt

# from nwn import *
# from jn_models import *
# from misc import *
from nwnTorch.nwn import *
from nwnTorch.jn_models import *
from nwnTorch.misc import *
from l2l_scripts.utils import *
# volterra_path = "/home/ruomin_zhu/old/volterra_data/"
# volterra_path = "/project/NASN/rzhu/l2l_data/volterra_data/"

def volterra_test(net,
                  hyper_params,
                  fit_steps = 2000):
    
    from l2l_scripts.utils import volterra_path
    X,Y        = pkl_load(volterra_path+f"pair_0.pkl")
    steps      = len(X)
    n_in       = 1
    n_out      = 64
    # torch.manual_seed(0)
    readout    = torch.zeros(steps, n_out)
    
    lambda_dict          = pkl_load(volterra_path+"lambda_data.pkl")
    electrodes           = lambda_dict["electodes"]

    np.random.seed(0)
    elec_out  = np.random.choice(1024, size = (1,n_out), replace=None)
    np.random.seed()

    tensor_dict = {}
    dtype_here  = torch.get_default_dtype()
    for key in hyper_params.keys():
        tensor_dict[key] = torch.tensor(hyper_params[key], dtype=dtype_here)

    # net.junction_state.L = torch.normal(mean = tensor_dict["lambda_mean"] * net.params["Lmax"], 
    #                                     std  = tensor_dict["lambda_std"] * net.params["Lmax"],
    #                                     size = (1, net.number_of_junctions))
    if "W_in_mean" in tensor_dict.keys():
        W_in = tensor_dict["W_in_mean"] * 3
    else: 
        W_in = 0.06978787 
    if "b_in_mean" in tensor_dict.keys():
        b_in = tensor_dict["b_in_mean"]
    else: 
        b_in = 0.5022748
    if "init_time" in tensor_dict.keys():
        net.junction_state.L = lambda_dict["lambda"][int(tensor_dict["init_time"] * 10000)]
        
    # net.junction_state.L = tensor_dict["lam"]
    # weight               = tensor_dict["W_out"]

    netG  = torch.zeros(steps)
    # for t in range(steps):
    for t in tqdm(range(steps)):
    # for t in tqdm(range(30)):
        sig_in = torch.zeros(len(electrodes))
        sig_in[0] = X[t] * W_in + b_in
        net.sim(sig_in.reshape(1,-1), electrodes)
        readout[t,:] = net.V[elec_out]
        netG[t]      = net.I[-1] / sig_in[0]

    # lhs     = torch.hstack((X[-fit_steps:].reshape(-1,1), readout[-fit_steps:]))
    lhs    = torch.hstack((torch.ones(fit_steps,1), readout[-fit_steps:]))
    result = torch.zeros(5)

    out_dict = {}
    out_dict["tests"] = torch.zeros(5,2)
    print(f'----- W_in = {W_in:.4}, b_in = {b_in:.4}, init_time = {int(tensor_dict["init_time"] * 10000)} -----')
    for i in range(5):
        index   = np.random.randint(100)
        _,Y     = pkl_load(volterra_path + f"pair_{index}.pkl")
        rhs     = Y[-fit_steps:]
        weight, result[i], rcond = best_regress(lhs, rhs)
        out_dict["tests"][i,0] = index
        out_dict["tests"][i,1] = result[i]

    out_dict["length"] = steps
    out_dict["runned"] = torch.sum(netG != 0)
    out_dict["netG"]   = netG
    out_dict["weight"] = weight
    out_dict["rcond"]  = rcond
    out_dict["params"] = hyper_params
    print(out_dict["tests"].T)
    return result.mean(), out_dict

# def prepare_network(index = 0):
#     adj = torch.tensor(pkl_load(volterra_path + "con0.pkl")["adj_matrix"])
#     net = NWN(adj, "sydney")

#     net.params["Ron"]       = 1e4
#     net.params["grow"]      = 5
#     net.params["decay"]     = 10
#     net.params["precision"] = True
#     return net

if __name__ == "__main__":    
    net = prepare_network(1)

    hyper = {
         "W_in_mean": 2.9,
         "b_in_mean": 0.06,
         "init_time": 2, 
                }
    
    mse = volterra_test(net, hyper)
    print(mse)
