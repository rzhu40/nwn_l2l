import torch
import numpy as np

from nwnTorch.jn_models import *
from nwnTorch.nwn import *
from nwnTorch.generate_adj import *
from nwnTorch.misc import *
from scipy.io import loadmat

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFromModel
from tqdm import tqdm

dtype = torch.float

def run_MG(onAmp, b):
    con = pkl_load("/home/rzhu/data_access/l2l_data/volterra_data/con0.pkl")
    adj = torch.tensor(con["adj_matrix"])
    # net = NWN(adj, "sydney")
    lambda_dict = pkl_load("/home/rzhu/data_access/l2l_data/volterra_data/lambda_data.pkl")
    
    maxG   = lambda_dict["maxG"]
    minG   = lambda_dict["minG"]
    netG   = lambda_dict["netG"]
    readL0 = lambda_dict["lambda"]

    MGdata = torch.tensor(loadmat("mackeyglass.mat")["datain"])

    net                     = NWN(adj, "sydney")
    E                       = net.number_of_junctions
    N                       = net.number_of_nodes
    net.params["Ron"]       = 1e4
    net.params["grow"]      = 5
    net.params["decay"]     = 10
    net.params["precision"] = True
    net.params["collapse"]  = True
    net.params["dt"]        = 1e-3

    T0                   = 2000
    electrodes           = lambda_dict["electodes"]
    net.junction_state.L = readL0[T0]
    net.junction_state.updateG()

    Tmax  = 5
    # onAmp, b = 1.55, 0.2001 # * NLT non-ergodic
    # onAmp, b = 0.05, 0.0001 # * NLT ergodic

    # onAmp, b = 5, 2
    # onAmp, b = 1.2, 0.001
    f     = 2
    T     = torch.arange(0,Tmax,1e-3)
    T_l   = len(T)

    sig   = MGdata * onAmp + b

    readI = torch.zeros(T_l, 1)
    readG = torch.zeros(T_l, E)
    # readL = torch.zeros(T_l, E)
    readV = torch.zeros(T_l, N)
    netG2 = torch.zeros(T_l)

    for t in tqdm(range(T_l)):
        sig_in = torch.zeros(len(electrodes))
        sig_in[0] = sig[t]

        net.sim(sig_in.reshape(1,-1), electrodes)
        readI[t,:] = net.I[-1:]
        # readG[t,:] = net.test_conductance(electrodes)
        readG[t,:] = net.junction_state.G[:]
        # readL[t,:] = net.junction_state.L[:]
        readV[t,:] = net.V
        netG2[t] = net.I[-1] / sig[t]

    pred_steps = 5
    torch.manual_seed(0)
    elec_out = torch.randint(N, (1,100))

    lhs = readV[:,elec_out.reshape(-1)]
    # rhs = sig[pred_steps:5000+pred_steps].type(torch.float32).reshape(-1)
    rhs = MGdata[pred_steps:5000+pred_steps].type(torch.float32).reshape(-1)

    train_range = np.arange(500,4000)
    test_range  = np.arange(4000,5000)
    whole_range = np.concatenate((train_range, test_range))

    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    # model      = linear_model.RidgeCV(alphas=alphas)
    model = linear_model.Ridge(alpha = 1e-2)
    var_filter = SelectFromModel(model)
    reg        = make_pipeline(
                    StandardScaler(),
                    var_filter, 
                    model
                    )

    reg.fit(lhs[train_range], rhs[train_range])

    predict = torch.tensor(reg.predict(lhs))
    score = reg.score(lhs[test_range], rhs[test_range])

    savedict = {
        "amp"   : onAmp,
        "b"     : b,
        "readG" : readG,
        "netG"  : netG2,
        "score" : score,
    }
    print(score)
    pkl_save(savedict, f"results/amp_{onAmp:.3}_b_{b:.3}.pkl")
if __name__ == "__main__":
    run_MG(0.5, 0.1)