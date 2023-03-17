import torch
from nwn import *
from jn_models import *
from misc import *

data_path = "/home/ruomin_zhu/snn_data/"
volterra_path = "/home/ruomin_zhu/old/volterra_data/"

def prepare_network(index = 0):
    adj = torch.tensor(pkl_load(data_path + "con0.pkl")["adj_matrix"])
    net = NWN(adj, "sydney")

    net.params["Ron"]       = 1e4
    net.params["grow"]      = 5
    net.params["decay"]     = 10
    net.params["precision"] = True
    return net