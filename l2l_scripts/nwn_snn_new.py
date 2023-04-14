import torch
from tqdm import tqdm
import numpy as np
# import matplotlib.pyplot as plt

from nwnTorch.nwn import *
from nwnTorch.jn_models import *
from nwnTorch.misc import *
from l2l_scripts.utils import *
# data_path = "/home/ruomin_zhu/snn_data/"
# volterra_path = "/home/ruomin_zhu/old/volterra_data/"
# volterra_path = "/project/NASN/rzhu/l2l_data/volterra_data/"
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFromModel

def learn_snn_new(net, 
              hyper_params,
              _test = False,
              test_idx = 0):
    from l2l_scripts.utils import snn_data_path
    tensor_dict = {}
    dtype_here  = torch.get_default_dtype()
    for key in hyper_params.keys():
        tensor_dict[key] = torch.tensor(hyper_params[key], dtype=dtype_here)

    # idx         = torch.randint(100, size = (1,1)).item()
    if _test:
        idx = test_idx
    else:
        idx = np.random.randint(100)
    # print(idx)
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
    e_read   = shuffled
    readout  = torch.zeros(num_steps, 1024)
    
    net.params["Ron"]       = 1e1
    net.params["Roff"]      = 1e4
    # W_in = tensor_dict["W_in"] 
    W_in = tensor_dict["W_in"] * 3
    b_in = tensor_dict["b_in"]
    # net.junction_state.L = lambda_dict["lambda"][2500]
    if "init_time" in tensor_dict.keys():
        net.junction_state.L = lambda_dict["lambda"]\
                [int(tensor_dict["init_time"] * 10000)]
    else: 
        net.junction_state.L = lambda_dict["lambda"][2000]

    for i in tqdm(range(num_steps)):
        sig_in = W_in * waves[i] + b_in
        net.sim(sig_in.reshape(1,-1), e_in)
        readout[i,:] = net.V[e_read]

    coefs = torch.zeros(20,1024)

    for i in range(20):
        for j in range(1024):
            coefs[i,j] = abs(torch.corrcoef(torch.stack((waves[:,i], readout[:,j])))[0,1])

    e_outs      = torch.argsort(coefs.mean(axis=0))[:num_read]

    result = torch.zeros(n_neurons)
    test   = torch.zeros(n_neurons)
    mse    = torch.zeros(n_neurons)
    
    train_range = torch.arange(2000,8000)
    test_range  = torch.arange(8000,9000)
    alphas   = [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1]

    out_dict = {}
    if _test:
        out_dict["target"] = mems
        out_dict["predict"] = torch.zeros(mems.shape)

    for i in range(n_neurons):
        # lhs = readout[:,e_outs]
        lhs = torch.nn.functional.softmax(readout[:,e_outs], 0)
        rhs = mems[:,i]
        
        model      = linear_model.RidgeCV(alphas = alphas, cv=KFold(5))
        var_filter = SelectFromModel(model)
        reg        = make_pipeline(
                        StandardScaler(),
                        var_filter,
                        model)
        
        reg.fit(lhs[train_range], rhs[train_range])
        predict = reg.predict(lhs)

        # weight, mse[i], rcond = best_regress(lhs, rhs)
        # predict = weight @ lhs.T
        if _test:
            out_dict["predict"][:,i] = torch.tensor(predict)
        # result[i] = get_RNMSE(predict, rhs)
        # result[i]=mean_squared_error(predict[test_range], rhs[test_range])

        result[i] = get_RNMSE(
                            torch.tensor(predict[train_range]), 
                            rhs[train_range])
        
        test[i]   = torch.tensor(
                        mean_squared_error(
                            predict[test_range], 
                            rhs[test_range]
                        ))
        
        mse[i]    = torch.tensor(
                        mean_squared_error(
                            predict[train_range], 
                            rhs[train_range]
                        ))
        
    out_dict["set"]    = idx
    out_dict["mse"]    = mse
    out_dict["rnmse"]  = result
    out_dict["e_in"]   = e_in
    out_dict["e_read"] = e_read
    out_dict["params"] = tensor_dict
    out_dict["test"]   = test

    print(idx, result.mean())
    print(mse.mean(), test.mean())
    # print(waves)
    # print(out_dict)
    return result.mean(), out_dict

if __name__ == "__main__":

    net   = prepare_network()
    hyper = {
        "W_in"     : torch.rand(1,20) * 3,
        "b_in"     : torch.rand(1,20),
        "init_time": 0.5,
    }

    fitness = learn_snn_new(net, hyper)
    