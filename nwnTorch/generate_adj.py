import torch
import wires 
import numpy as np

def generate_fully_connected(shape = (128,128)):
    """Generate fully connected 2-layer network.
    Effectivly the same as a cross-bar.

    :param shape: shape of the network
    :type shape: tuple, (int, int)
    """
    L1 = shape[0]
    L2 = shape[1]
    LN  = L1 + L2
    adj = torch.zeros(LN, LN)

    adj[:L1, L1:LN] = 1
    adj = adj + adj.T
    return adj

def generate_network(
    numOfWires = 100, 
    mean_length = 100, 
    dispersion = 10,
    this_seed = None, 
    iterations = 0, 
    max_iters = 10, 
    L = 3e3):

    if this_seed is None:
        np.random.seed()
        this_seed = np.random.randint(100000)
    wires_dict = wires.generate_wires_distribution(number_of_wires = numOfWires,
                                            wire_av_length = mean_length,
                                            wire_dispersion = dispersion,
                                            Lx = L,
                                            Ly = L,
                                            this_seed = this_seed)

    wires_dict = wires.detect_junctions(wires_dict)
    # wires_dict = wires.generate_adj_matrix(wires_dict)
    wires_dict = wires.generate_graph(wires_dict)
    if wires.check_connectedness(wires_dict):
        print(f'The returned network has {wires_dict["number_of_junctions"]} junctions.')
        return wires_dict
    elif iterations < max_iters: 
        return generate_network(numOfWires, mean_length, dispersion, None, iterations+1, max_iters, L)
    else:
        print('No network is generated.')
        return None