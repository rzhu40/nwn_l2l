# import torch
# from nwn import *
# from jn_models import *
# from misc import *
import torch
from nwnTorch.nwn import *
from nwnTorch.jn_models import *
from nwnTorch.misc import *


# data_path = "/home/ruomin_zhu/snn_data/"
# volterra_path = "/home/ruomin_zhu/old/volterra_data/"

# snn_data_path = "/home/ruomin_zhu/l2l_data/snn_data_new/"
# volterra_path = "/home/ruomin_zhu/l2l_data/volterra_data/"

snn_data_path = "/home/rzhu/data_access/l2l_data/snn_data_long/"
volterra_path = "/home/rzhu/data_access/l2l_data/volterra_data/"
# volterra_path = "/home/rzhu/data_access/l2l_data/volterra_data_new/"

def prepare_network(index = 0):
    adj = torch.tensor(pkl_load(snn_data_path + "con0.pkl")["adj_matrix"])
    net = NWN(adj, "sydney")

    net.params["Ron"]       = 1e4
    net.params["grow"]      = 5
    net.params["decay"]     = 10
    net.params["precision"] = True
    return net