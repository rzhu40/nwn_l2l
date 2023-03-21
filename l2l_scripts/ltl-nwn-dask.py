import logging.config
import os
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np 
from time import strftime
from dask.distributed import Client, LocalCluster
from dask_jobqueue import PBSCluster

file_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, file_dir)

from l2l.utils.environment import Environment
# from l2l.optimizers.simulatedannealing.optimizer import SimulatedAnnealingParameters, SimulatedAnnealingOptimizer, AvailableCoolingSchedules
from l2l.paths import Paths
from l2l.logging_tools import create_shared_logger_data, configure_loggers

from prepare_optimizer import *
from nwn_optimizee import NWN_Optimizee
logger = logging.getLogger('bin.ltl-fun-sa')

def main():
    usage = "\
            Call this script from command line.  \n\
            Make sure to do some fine-tuning before running. High parallelization doesn't mean faster simulations. \n\
            Examples for good performances are: \n\
            python l2l_scripts/ltl-nwn-dask.py --nworkers 2 --tpw 1 --ngen 50 --nind 16 --optimizer ES --task learn_snn \n\
            python l2l_scripts/ltl-nwn-dask.py --cluster_mode PBS --njobs 8 --ncores 4 --nworkers 1 --mem 4 --walltime 20 --ngen 50 --nind 16 --optimizer SA --task learn_snn \
            "
    parser = ArgumentParser(
                description=usage,
                formatter_class=RawTextHelpFormatter
                )
    # NOTE These are for dask
    # Just using one node per script
    # parser.add_argument("--nnodes", type=int, required=True,
    #                     help="Number of nodes to request.")
#     parser.add_argument("--scheduler", type = int,
#                         required=False, default = 9999,
#                         help="IP address of scheduler, e.g. 127.0.0.1:9999.")
    parser.add_argument("--cluster_mode", type=str, 
                        required=False, default = "Local",
                        help="Cluster type, Local| PBS. Defaults to Local.")
    parser.add_argument("--nworkers", type=int, 
                        required=False, default = 1,
                        help="Number of workers job in dask.")
    parser.add_argument("--tpw", type=int, 
                        required=False, default = 1,
                        help="Number of threads per worker. Local only!")
    # NOTE following are for PBS clusters.
    parser.add_argument("--njobs", type=int, 
                        required=False, default = 1,
                        help="Number of PBS jobs to request. PBS only!")
    parser.add_argument("--ncores", type=int, 
                        required=False, default = 1,
                        help="Number of cores to request. PBS only!")
    # parser.add_argument("--nprocs", type=int, 
    #                     required=False, default = 1,
    #                     help="Number of processes per job in dask.")
    parser.add_argument("--mem", type=int, 
                        required=False, default = 1,
                        help="Memory, GB. PBS only!")
    parser.add_argument("--walltime", type=int, 
                        required=False, default = 1,
                        help="Wall time, hr. PBS only!")

    # NOTE These are for l2l
    parser.add_argument("--label", type=str,required=False, 
                        default=strftime("%Y-%m-%d-%H%M%S"),
                        help="Label of current run, defaults to time")
    parser.add_argument("--task", type=str, required=False, 
                        default="volterra", 
                        help="Benchmark task for the NWN.")
    parser.add_argument("--optimizer", type=str, required=False, 
                        default="SA", 
                        help="Optimizer used for L2L.")
    parser.add_argument("--ngens", type=int,
                        required=False, default=50,
                        help="Number of generations of l2l.")
    parser.add_argument("--ninds", type=int,
                        required=False, default=16,
                        help="Number of individuals per gen.")
    parser.add_argument("--T0", type=float, required=False, default = 0, 
                        help="Pre-initialization time for the task, in unit of second. ")
    parser.add_argument("--W0", type=float,required=False,
                        help="Starting input weight.")
    parser.add_argument("--b0", type=float,required=False,
                        help="Starting input bias.")
    
    
    args = parser.parse_args()
    name = f'LTL-NWN-{args.task}-{args.optimizer}'
    # args.label = "debug_wrap"
    
    # NOTE learn_dict: specify the parameters to learn as keys
    # values are the initial states of eacha parameter, set to "None" for random state.

    # for volterra
    if args.task == "volterra":
        learn_dict = {
            "W_in_mean": args.W0,
            "b_in_mean": args.b0,
            "init_time": args.T0
            }
        
    # for learn snn
    elif args.task == "learn_snn":
        learn_dict = {
            # NOTE initializing as 2D arrays somehow leads to bugs
            # TODO check if np.array() wrapping is necessary in optimizee
            "W_in"     : np.random.rand(20),
            "b_in"     : np.random.rand(20),
            # "init_time": args.T0
            }    
    
    if args.cluster_mode == "Local":
        cluster = LocalCluster(
                    n_workers = args.nworkers,
                    threads_per_worker = args.tpw,
                    scheduler_port = 12121,
                    dashboard_address = 'localhost:11113',
                    )
        
    elif args.cluster_mode == "PBS":
        dask_log_path = f"/project/NASN/rzhu/l2l_data/LTL_logs/pbs_logs/{args.label}"
        os.makedirs(dask_log_path, exist_ok=True)
        
        # TODO directly use IP address for client.
        cluster = PBSCluster(
                        name = f"dask_{args.label}",
                        cores = args.ncores, 
                        n_workers = args.nworkers, 
    #                     processes = args.nprocs,
                        shebang='#!/bin/bash',
                        memory = f'{args.mem}GB',
                        walltime = f'{args.walltime}:{1}:00',
                        log_directory = dask_log_path,
                        env_extra = [f'cd /home/rzhu0837/nwn_l2l/',
                                "module unload python magma openmpi-gcc sqlite",
                                "module load python/3.8.2 magma/2.5.3 openmpi-gcc/3.1.3",
                                "echo $PATH",
                                ],
                        job_extra = ['-P NASN'],
                        scheduler_port = 12121,
                        dashboard_address=':11113',
    #                  scheduler_options = {'dashboard_address' : 'localhost:11113',
    #                                       'host' : ':12121'}
                    )
        cluster.scale(jobs = args.njobs)
        print(cluster.job_script())

    client = Client(cluster)

    file_path = os.path.dirname(__file__)
    try:
        with open(os.path.join(file_path, 'bin/path.conf')) as f:
            root_dir_path = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            "You have not set the root path to store your results."
            " Write the path to a path.conf text file in the bin directory"
            " before running the simulation"
        )
    paths = Paths(name, dict(run_no=args.label), root_dir_path=root_dir_path)

    print("All output logs can be found in directory ", paths.logs_path)

    traj_file = os.path.join(paths.output_dir_path, 'data.h5')

    # Create an environment that handles running our simulation
    # This initializes an environment
    env = Environment(trajectory=name, 
                      filename=traj_file, 
                      dask_client = client,
                      file_title='{} data'.format(name),
                      comment='{} data'.format(name),
                      add_time=True,
                      multiprocessing=True, # NOTE: set to False for debugging
                      automatic_storing=True,
                      log_stdout=False,  # Sends stdout to logs
                      )
    
    create_shared_logger_data(logger_names=['bin', 'optimizers', 'utils'],
                              log_levels=['INFO', 'INFO', 'INFO'],
                              log_to_consoles=[True, True, True],
                              sim_name=name,
                              log_directory=paths.logs_path)
    configure_loggers()
    # Get the trajectory from the environment
    traj = env.trajectory

    logger.info(
            "dask scheduler address:" 
            + client.scheduler.address)
    logger.info(
            "dask dashboard address (need port forwarding):" 
            + client.dashboard_link)
    
    # NOTE: Benchmark function
    optimizee = NWN_Optimizee(traj, args.task, learn_dict, 
                              os.path.abspath(paths.results_path))
    
    optimizer = prepare_optimizer(
                    optimizee, traj, 
                    optimizer_type=args.optimizer,
                    n_individual = args.ninds,
                    n_generation = args.ngens, 
                    stop_criterion = -1e-5
                    )

    # Add post processing
    env.add_postprocessing(optimizer.post_process)

    # Run the simulation with all parameter combinations
    env.run(optimizee.simulate)
    # env.run(optimizee.simulate_dask)

    # NOTE: Outerloop optimizer end
    optimizer.end(traj)
    # recorder.end()
    
    # Finally disable logging and close all log-files
    env.disable_logging()

if __name__ == '__main__':
    main()