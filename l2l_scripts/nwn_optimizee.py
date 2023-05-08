import numpy as np
import torch
from nwn_volterra_test import *
from nwn_learn_snn import *
from nwn_snn_new import *
from nwn_volterra_new import *
# from nwn_nlt_test import *
from l2l.optimizees.optimizee import Optimizee
from os import path
from nwnTorch.misc import *
# import dask

task_dict = {
            "volterra"      : volterra_test,
            "volterra_new"  : volterra_new,
            "learn_snn"     : learn_snn,
            "learn_snn_new" : learn_snn_new,
            # "nlt"     : non_lin_trans_test
            }

class NWN_Optimizee(Optimizee): 
    """
    Implements NWN optimizee. 
    NOTE: Make sure the optimizee_fitness_weights is set to (-1,) to minimize the value of the function
    """

    def __init__(self, traj = None,  
                 benchmark = "volterra", 
                #  params_to_learn = ["W_in_mean"],
                 learn_dict = {"W_in_mean" : 1},
                 output_path = ""):
        self.benchmark = benchmark
        self.output_path = output_path
        super().__init__(traj)
        # super(NWN_Optimizee, self).__init__(traj)
        
        self.params_to_learn = list(learn_dict.keys())

        self.params_pool = {
            "W_in_mean"  : np.random.random() * 2 - 1,
            "W_in_std"   : np.float64(np.random.random()),
            "b_in_mean"  : np.random.random() * 2 - 1,
            "b_in_std"   : np.float64(np.random.random()),
            "init_time"  : 0.,
            "lam"        : np.random.normal(0, 0.05, size = 6877),
            "W_out"      : np.random.random(101) * 2 - 1,
            "lambda_mean": np.float64(np.random.random()),
            "lambda_std" : np.float64(np.random.random()),
        }

        self.learn_dict = learn_dict
        
        self.bound_dict = {
            "W_in"       : (-1, 1),
            "b_in"       : (-1, 1),
            "W_in_mean"  : (-1, 1),
            "W_in_std"   : (0, 1),
            "b_in_mean"  : (-1, 1),
            "b_in_std"   : (0, 1),
            "init_time"  : (0, 0.5),
            "lam"        : (-0.15, 0.15),
            "W_out"      : (-1, 1),
            "lambda_mean": (0, 1),
            "lambda_std" : (0, 1),
        }

        indiv_dict = self.create_individual()
        for key, val in indiv_dict.items():
            traj.individual.f_add_parameter(key, val)

    def create_individual(self):
        """
        Creates a random value of parameter within given bounds
        """
        
        # init_dict = {}
        # for key in self.params_to_learn:
        #     init_dict[key] = self.params_pool[key]
        init_dict = {}
        for key in self.learn_dict:
            if self.learn_dict[key] is None:
                init_dict[key] = np.float64(self.params_pool[key])
            else:
                init_dict[key] = np.float64(self.learn_dict[key])
        return init_dict

    def bounding_func(self, individual):
        """
        Bounds the individual within the required bounds via coordinate clipping
        """
        bound_dict = {}
        for key in self.params_to_learn:
            bound_dict[key] = np.clip(individual[key], a_min = self.bound_dict[key][0], a_max = self.bound_dict[key][1])
        return bound_dict

    def get_params(self):
        """
        Get the important parameters of the optimizee. This is used by :class:`ltl.recorder`
        for recording the optimizee parameters.
        :return: a :class:`dict`
        """
        return None
    
    # @dask.delayed
    # def simulate_dask(self, work_traj_path):
    #     return self.simulate(work_traj_path)
    
    # @dask.delayed
    def simulate(self, work_traj_path):
        """
        Returns the value of the function chosen during initialization
        :param ~pypet.trajectory.Trajectory traj: Trajectory
        :return: a single element :obj:`tuple` containing the value of the chosen function
        """
        # traj = pkl_load(f"_traj_gen_{gen_idx:04d}_ind_{ind_idx:04d}.bin")
        # gen_idx = traj.individual.__dict__["generation"]
        # ind_idx = traj.individual.__dict__["ind_idx"]

        traj = pkl_load(work_traj_path)
        gen_idx = int(work_traj_path.split("_")[-3])
        ind_idx = int(work_traj_path.split("_")[-1].split(".")[-2])
        
        hyper_params = {}
        for key in self.params_to_learn:
            # hyper_params[key] = np.array(traj.individual.__dict__["params"][f"individual.{key}"])
            hyper_params[key] = traj.individual.__dict__["params"][f"individual.{key}"]
            # hyper_params[key] = np.array(traj[f"individual.{key}"])

        index = np.random.randint(100)
        net   = prepare_network(index)
        task  = task_dict[self.benchmark]

        fitness, out_dict = task(net, hyper_params)
        fname = path.join(self.output_path, f"gen_{gen_idx:04d}_ind_{ind_idx:04d}.pkl")
        pkl_save(out_dict, fname)

        return (fitness,)
    


        

