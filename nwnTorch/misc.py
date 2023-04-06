import torch
import copy
import numpy as np
import l2l
# import pickle

def AC_sweep(
        net,
        electrodes, 
        min_amp = 1,
        max_amp = 3,
        n_cycle = 5
        ):
    f = 1
    dt = 1e-3
    T = torch.arange(0,n_cycle/f,dt)
    
    sig = (torch.sin(2*torch.pi*f*T).reshape(n_cycle,-1) * \
            torch.linspace(min_amp,max_amp, n_cycle).reshape(-1,1)).reshape(-1)

    # sig = torch.ones(len(T)) * 1e-10
    test_net = copy.deepcopy(net)
    n_step = len(sig)
    I = torch.zeros(n_step)
    for step in range(n_step):
        input = torch.tensor([sig[step], 0])
        sol = test_net.sim_step(electrodes, input)
        I[step] = sol[-1]

    return I, sig

def graphical_distance(adjMat):
    from networkx import from_numpy_array, floyd_warshall_numpy
    adjMat = np.array(adjMat)
    G = from_numpy_array(adjMat)
    distMat = np.array(floyd_warshall_numpy(G))
    return distMat

def generate_incidence_matrix(adjMat):
    from networkx import from_numpy_array, incidence_matrix
    G = from_numpy_array(np.array(adjMat))
    return incidence_matrix(G).todense()

def get_RNMSE(predict, real):
    return torch.sqrt(torch.sum((predict-real)**2) / torch.sum(real**2))

def get_MSE(predict, real):
    return ((predict - real)**2).mean()

def generate_signal(signal_type = "DC", 
                    onAmp = 1, 
                    f = 1,
                    Tmax = 1,
                    dt = 1e-3):
    TimeVector = torch.arange(0,Tmax, dt)
    period     = 1/f

    if signal_type == "DC":
        signal = onAmp * torch.ones(len(TimeVector))
    elif signal_type == 'AC':
        signal = onAmp*torch.sin(2*torch.pi*f*TimeVector)
    elif signal_type == 'Square':
        signal = onAmp * (-torch.sign(TimeVector % period - period/2))
    elif signal_type == 'Triangular':
        signal = 4*onAmp/period * abs((TimeVector-period/4) % period - period/2) - onAmp
    elif signal_type == 'Sawtooth':
        signal = onAmp/period * (TimeVector % period)
    return signal

def pkl_save(obj, filename = None, NASN = False, user = 'rzhu'):
    if filename is None:
        import time
        filename = time.strftime("%Y-%m-%d-%H%M%S") + '_tmp_file.pkl'
    if NASN:
        filename = f'/project/NASN/{user}/'+filename
    
    try: 
        from pickle5 import dump, HIGHEST_PROTOCOL
    except:
        from pickle import dump, HIGHEST_PROTOCOL

    with open(filename, 'wb') as handle:
        dump(obj, handle, protocol = HIGHEST_PROTOCOL)

def pkl_load(filename, NASN = False, user = 'rzhu'):
    try:
        from pickle5 import load as pklload
    except:
        from pickle import load as pklload
    if NASN:
        filename = f'/project/NASN/{user}/'+filename
    with open(filename, 'rb') as handle:
        return pklload(handle)
    
def draw_wires(wires, ax = None):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots()

    xa, xb = wires["xa"], wires["xb"]
    ya, yb = wires["ya"], wires["yb"]
    Lx, Ly = wires["length_x"], wires["length_y"]

    ax.plot([xa,xb], [ya,yb], c = 'k', alpha = 0.2)

# plt.plot([xa[read], xb[read]], [ya[read],yb[read]], c = 'b', alpha = 0.5)
    ax.set_xlim(-0.1*Lx, 1.1*Lx)
    ax.set_ylim(-0.1*Ly, 1.1*Ly)

    plt.gca().invert_yaxis()
    return ax

def best_regress(lhs, rhs, search_min = -15):
    searches = torch.arange(-1, search_min, -1)
    n_search = len(searches) + 1
    results = torch.zeros(n_search)
    Ws = torch.zeros(n_search, lhs.shape[1], dtype = lhs.dtype)
    for i in range(n_search):
        if i == 0:
            rcond = None
        else:
            rcond = torch.float_power(10, searches[i-1])

        Ws[i] = torch.linalg.lstsq(lhs, rhs, rcond = rcond).solution
        predict = Ws[i] @ lhs.T
        results[i] = get_MSE(predict, rhs)

    best = torch.argmin(results)
    return Ws[best], results[best], -best

# def best_regress(lhs, rhs, search_min = -15):
#     """
#     lhs: [n_samples, n_input_features]
#     rhs: [n_samples, n_target_features]
#     """
#     searches = torch.arange(-1, search_min, -1)
#     n_search = len(searches) + 1
#     results = torch.zeros(n_search)
#     # Ws = torch.zeros(n_search, lhs.shape[1], dtype = lhs.dtype)
#     Ws = []
#     for i in range(n_search):
#         if i == 0:
#             rcond = None
#         else:
#             rcond = torch.float_power(10, searches[i-1])

#         # Ws[i] = torch.linalg.lstsq(lhs, rhs, rcond = rcond).solution
#         temp = torch.linalg.lstsq(lhs, rhs, rcond = rcond).solution
#         # predict = temp @ lhs.T
#         predict = lhs @ temp
#         results[i] = get_MSE(predict, rhs)
#         Ws.append(temp)

#     best = torch.argmin(results)
#     return Ws[best], results[best], -best

def error_shade(mean, err,
                xdata = None,
                ax = None,
                c = 'r',
                **kwargs):
    import matplotlib.pyplot as plt
    # mean = np.mean(data, axis = axis)
    # err = np.std(data, axis = axis)/np.sqrt(data.shape[axis])
    if xdata is None:
        xdata = np.arange(len(mean))
    if ax is None:
        ax = plt
    line, = ax.plot(xdata, mean, c = c, **kwargs)
    ax.fill_between(xdata, mean-err, mean+err, color = c, alpha = 0.2)
    return line