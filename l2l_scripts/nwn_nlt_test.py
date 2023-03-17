import torch
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/ruomin_zhu/nwn_l2l/nwnTorch/")
# sys.path.append("/home/ruomin_zhu/nwn_l2l")

from nwn import *
from jn_models import *
from misc import *
volterra_path = "/home/ruomin_zhu/old/volterra_data/"
# volterra_path = "/project/NASN/rzhu/l2l_data/volterra_data/"

def non_lin_trans_test(net,
                  hyper_params,
                  fit_steps = 2000):
    Tmax       = 3
    X          = generate_signal(signal_type="Triangular", onAmp= 1, f = 2, Tmax = Tmax)
    Y          = generate_signal(signal_type="AC", onAmp= 1, f = 2, Tmax = Tmax)
    steps      = len(X)
    n_in       = 1
    n_out      = 100
    readout    = torch.zeros(steps, n_out)
    electrodes = torch.tensor([118, 211])

    # elec_out   = torch.randint(low = 0, high = 1024, size = (1,n_out))
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
    
    # W_in = torch.normal(mean = tensor_dict["W_in_mean"], 
    #                     std  = tensor_dict["W_in_std"],
    #                     size = (1, n_in))
    
    # b_in = torch.normal(mean = tensor_dict["b_in_mean"], 
    #                     std  = tensor_dict["b_in_std"],
    #                     size = (1, n_in))
    W_in                 = tensor_dict["W_in_mean"]
    b_in                 = tensor_dict["b_in_mean"]
    net.junction_state.L = tensor_dict["lam"]
    # weight               = tensor_dict["W_out"]
    
    for t in range(steps):
    # for t in tqdm(range(steps)):
        sig_in = torch.zeros(len(electrodes))
        sig_in[0] = X[t] * W_in + b_in
        net.sim(sig_in.reshape(1,-1), electrodes)
        readout[t,:] = net.V[elec_out]

    # lhs = torch.hstack((X[-fit_steps:].reshape(-1,1), readout[-fit_steps:]))

    lhs     = torch.hstack((torch.ones(fit_steps,1), readout[-fit_steps:]))
    rhs     = Y[-fit_steps:]
    weight  = torch.linalg.lstsq(lhs, rhs, rcond=None).solution
    predict = weight @ lhs.T
    result  = get_RNMSE(predict, rhs)

    return result

def prepare_network(index = 0):
    # adj = torch.tensor(pkl_load("/home/ruomin_zhu/nwn_l2l/volterra_data/con0.pkl")["adj_matrix"])
    adj = torch.tensor(pkl_load(volterra_path + "con0.pkl")["adj_matrix"])
    net = NWN(adj, "sydney")

    net.params["Ron"]       = 1e4
    net.params["grow"]      = 5
    net.params["decay"]     = 10
    net.params["precision"] = True
    return net

if __name__ == "__main__":    
    net = prepare_network(1)
    # mkl.set_num_threads(8)
    # mkl.set_num_threads(8)
    hyper = {
         "W_in_mean": np.random.random(),
         "b_in_mean": np.random.random()*2-1,
         "lam"      : np.random.random(6877) * 0.3 - 0.15,
        #  "W_out"    : np.random.random(101) * 2 - 1,
                }
    mse = non_lin_trans_test(net, hyper)
    
    print(mse)