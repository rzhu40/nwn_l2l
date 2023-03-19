import torch
from .jn_models import *
from copy import deepcopy

class NWN:
    def __init__(
        self, 
        adjMat,
        junction_mode = "sydney"):
        """
        Only able to deal with symmetric matrices!!!
        Change Params before anything else!!!

        :param adjMat: adjacency matrix of the network
        :type  adjMat: 2d tensor

        :param junction_mode: type of the junction, defaults to ``sydney``
            currently available : 
                ``sydney`` ;
                ``miranda`` ;
                ``linear``
        :type junction_mode: string
        
        """
        self.adjMat  = adjMat.clone().detach()
        self.jn_mode = junction_mode
        self.steps   = 0
        self.params  = {
                        "dt"      : 1e-3,
                        "Vset"    : 1e-2,
                        "Vreset"  : 1e-3,
                        "Ron"     : 1e4,
                        "Roff"    : 1e7,
        }
        self.create_junctions()
    
    def create_junctions(self):
        self.number_of_nodes     = self.adjMat.shape[0]
        self.number_of_junctions = int(torch.sum(self.adjMat)/2)
        a, b                     = torch.where(torch.triu(self.adjMat == 1))
        self.junction_list       = torch.vstack([a,b]).T
        # self.junction_list       = torch.tensor(torch.where(torch.triu(self.adjMat) == 1)).T

        jn_function         = junction_dict[self.jn_mode]
        self.junction_state = jn_function(self.number_of_junctions, self.params)

    def sim_step(self, electrodes, signal_in):
        self.junction_state.updateG()
        junctionG     = self.junction_state.G
        junction_list = self.junction_list
        signal_in     = signal_in.reshape(-1)

        N   = self.number_of_nodes
        E   = len(electrodes)
        # lhs = torch.zeros((N+E, N+E), dtype = torch.float64)
        # rhs = torch.zeros(N+E, dtype = torch.float64)
        lhs = torch.zeros((N+E, N+E))
        rhs = torch.zeros(N+E)


        lhs[junction_list[:,0], junction_list[:,1]] = -junctionG
        lhs[junction_list[:,1], junction_list[:,0]] = -junctionG
        lhs[range(N), range(N)]                     = -torch.sum(lhs,axis = 0)[:N]

        lhs[range(N,N+E), electrodes] = 1
        lhs[electrodes, range(N,N+E)] = 1
        rhs[N:]                       = signal_in

        sol                   = torch.linalg.solve(lhs, rhs)
        self.I                = sol[self.number_of_nodes:]
        self.V                = sol[:self.number_of_nodes]
        self.junction_state.V = sol[junction_list[:,0]] - sol[junction_list[:,1]]
        
        self.junction_state.updateL()
        return sol

    @classmethod
    def data_check(
        cls,
        data_in, 
        electrodes, 
        num_steps,
        fit_data = False):
        """
        Check the shape of input data.
        """

        T0, E0 = data_in.shape
        T1     = num_steps
        E1     = len(electrodes)
        flag   = False
        msg1   = ""
        
        if T0 != T1:
            flag = True
            msg1 += f"The length of input data is {T0}, expect to run {num_steps} steps of simulation! \n"

        if E0 != E1:
            flag = True
            msg1 += f"Signals to {E0} electrodes are provided, while {E1} electrodes are specified. \n"
        if len(msg1)>0:
            print(msg1)

        if flag & fit_data:
            Tmax = max(T0, T1)
            Emax = max(E0, E1)

            temp = torch.zeros((Tmax, Emax))
            temp[:T0,:E0] = data_in
            data_out = temp[:T1,:E1]
            msg = f"Data fitted, current input shape (T={num_steps}, E={E1})."
            print(msg)

        elif (T0 < T1) or (E0 < E1):
                raise ValueError(msg + "Not enough data. \n")
        else:
            data_out = data_in
        
        return data_out

    def sim(
        self, 
        data_in,
        electrodes,
        num_steps = -1,
        data_check = True,
        fit_data = True,
        ):
        """
        Simulate the network with given data and electrodes.

        :param data: input data stream to each electrode
        :type  data: 2D tensor of shape (T, E) 
            -- T, number of steps;
            -- E, number of electrodes.

        :param electrodes: the eletrodes used for input data
            electrodes with input of 0s is effectively a drain
        :type  electrodes: int, array 

        :param num_steps: number of simulation steps, 
            defaults to -1, simulate for the length of the data stream
        :type num_steps : int
        """

        if num_steps == -1:
            num_steps = data_in.shape[0]

        self.junction_state.updateG()
        if data_check:
            data = self.data_check(data_in, electrodes, num_steps, fit_data)

        for step in range(num_steps):
            self.steps += 1
            sol = self.sim_step(electrodes, data[step])
            
        self.sol = sol

    def test_conductance(
        self, 
        electrodes,
        amp = 1e-7):

        test_net = deepcopy(self)
        input = torch.tensor([amp, 0], device="cpu")
        sol = test_net.sim_step(electrodes, input)
        return sol[-1]/amp