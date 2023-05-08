import torch
from tqdm import tqdm
from nwnTorch.nwn import *
from nwnTorch.jn_models import *
from nwnTorch.misc import *
from l2l_scripts.utils import *

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFromModel

def volterra_new(net,
                  hyper_params,
                  fit_steps = 2000):
    
    from l2l_scripts.utils import volterra_path
    X,Y        = pkl_load(volterra_path+f"pair_0")
    steps      = len(X)
    readout    = torch.zeros(steps, 1024)
    
    lambda_dict = pkl_load(volterra_path+"lambda_data.pkl")
    electrodes  = lambda_dict["electodes"]

    tensor_dict = {}
    dtype_here  = torch.get_default_dtype()
    for key in hyper_params.keys():
        tensor_dict[key] = torch.tensor(hyper_params[key], dtype=dtype_here)

    net.params["Ron"]       = 1e1
    net.params["Roff"]      = 1e4
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
        

    netG  = torch.zeros(steps)
    for t in tqdm(range(steps)):
        sig_in = torch.zeros(len(electrodes))
        sig_in[0] = X[t] * W_in + b_in
        net.sim(sig_in.reshape(1,-1), electrodes)
        readout[t,:] = net.V
        netG[t]      = net.I[-1] / sig_in[0]

    num_read = 64
    coefs    = torch.zeros(1024)
    for i in range(1024):
        coefs[i] = abs(torch.corrcoef(torch.stack((X, readout[:,i])))[0,1])

    e_outs = torch.argsort(coefs)[:num_read]
    lhs    = readout[:,e_outs]


    train_range = torch.arange(2000,8000)
    test_range  = torch.arange(8000,10000)
    result      = torch.zeros(5)
    out_dict    = {}

    out_dict["tests"] = torch.zeros(5,3)
    print(f'----- W_in = {W_in:.4}, b_in = {b_in:.4}, init_time = {int(tensor_dict["init_time"] * 10000)} -----')
    for i in range(5):
        index   = np.random.randint(100)
        _,Y     = pkl_load(volterra_path + f"pair_{index}")
        # rhs     = Y[-fit_steps:]
        # weight, result[i], rcond = best_regress(lhs, rhs)
        rhs = Y
        
        alphas = [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1]
        model  = linear_model.RidgeCV(alphas = alphas, cv = KFold(5))
        var_filter = SelectFromModel(model)
        reg    = make_pipeline(
                    StandardScaler(),
                    var_filter, 
                    model
                    )
        reg.fit(lhs[train_range], rhs[train_range])
        predict = reg.predict(lhs)

        # result[i] = mean_squared_error(
        #                 predict[train_range], 
        #                 rhs[train_range]
        #                 )
        
        result[i] = get_RNMSE(torch.tensor(predict[train_range]), 
                              rhs[train_range])
        mse = mean_squared_error(
                        predict[test_range], 
                        rhs[test_range]
                        )
        
        # nrmse = np.sqrt(mse) / (rhs[test_range].max() - rhs[test_range].min())

        out_dict["tests"][i,0] = index
        out_dict["tests"][i,1] = result[i]
        out_dict["tests"][i,2] = mse

    out_dict["length"] = steps
    out_dict["runned"] = torch.sum(netG != 0)
    out_dict["netG"]   = netG
    # out_dict["weight"] = weight
    # out_dict["rcond"]  = rcond
    out_dict["params"] = hyper_params
    print(out_dict["tests"].T)
    return result.mean(), out_dict

if __name__ == "__main__":    
    net = prepare_network(1)

    hyper = {
         "W_in_mean": 2.9,
         "b_in_mean": 0.06,
         "init_time": .2, 
                }
    
    mse = volterra_new(net, hyper)
    print(mse)
