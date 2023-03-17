import logging.config
import os
import argparse
import numpy as np 
from time import strftime
from dask.distributed import Client, LocalCluster

from l2l.utils.environment import Environment
from l2l.optimizers.simulatedannealing.optimizer import SimulatedAnnealingParameters, SimulatedAnnealingOptimizer, AvailableCoolingSchedules
from l2l.paths import Paths
from l2l.logging_tools import create_shared_logger_data, configure_loggers

from nwn_optimizee import NWN_Optimizee
logger = logging.getLogger('bin.ltl-fun-sa')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nprocs", type=int, required=True,
                        help="Number of processes in parallel.")
    parser.add_argument("--task", type=str, required=False, 
                        default="volterra", 
                        help="Benchmark task for the NWN.")
    parser.add_argument("--label", type=str,required=False, 
                        default=strftime("%Y-%m-%d-%H%M%S"),
                        help="Label of current run, defaults to time")
    parser.add_argument("--W0", type=float,required=False,
                        help="Starting input weight.")
    parser.add_argument("--b0", type=float,required=False,
                        help="Starting input bias.")
    parser.add_argument("--T0", type=float, required=False, default = 0, 
                        help="Pre-initialization time for the task, in unit of second. ")
    
    args = parser.parse_args()
    task = args.task
    name = f'LTL-NWN-{task}-SA'
    # args.label = "debug02"
    
    # NOTE learn_dict: specify the parameters to learn as keys
    # values are the initial states of eacha parameter, set to "None" for random state.
    learn_dict = {"W_in_mean" : args.W0,
                  "b_in_mean" : args.b0,
                  "init_time" : args.T0}

    parameters = SimulatedAnnealingParameters(
                    n_parallel_runs=24, n_iteration=50,
                    noisy_step=.1, temp_decay=.99, 
                    stop_criterion=-1e-5, seed=21343, 
                    cooling_schedule=AvailableCoolingSchedules.QUADRATIC_ADDAPTIVE)
    
    cluster = LocalCluster(
                n_workers = args.nprocs,
                # threads_per_worker = 1,
                scheduler_port = 8789,
                dashboard_address = 'localhost:8787',
                )
    client = Client(cluster)
    
    try:
        with open('bin/path.conf') as f:
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
                      multiprocessing=True,
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
    
    # NOTE: Benchmark function
    optimizee = NWN_Optimizee(traj, task, learn_dict, 
                              os.path.abspath(paths.results_path))

    optimizer = SimulatedAnnealingOptimizer(
                    traj, 
                    optimizee_create_individual=optimizee.create_individual,
                    optimizee_fitness_weights=(-1.,),
                    parameters=parameters,
                    optimizee_bounding_func=optimizee.bounding_func)

    # Add post processing
    env.add_postprocessing(optimizer.post_process)

    logger.info("Initial conditions are:")
    logger.info(learn_dict)
    logger.info(f"Using {args.nprocs} workers, each has 1 thread.")
    logger.info(
            "dask scheduler address:" 
            + client.scheduler.address)
    logger.info(
            "dask dashboard address (need port forwarding):" 
            + client.dashboard_link)
    
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