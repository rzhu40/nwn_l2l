import logging.config
import os
import numpy as np 
from time import strftime

from l2l.utils.environment import Environment
from l2l.optimizers.gradientdescent.optimizer import GradientDescentOptimizer
from l2l.optimizers.gradientdescent.optimizer import ClassicGDParameters
from l2l.optimizers.gradientdescent.optimizer import StochasticGDParameters
from l2l.optimizers.gradientdescent.optimizer import AdamParameters
from l2l.optimizers.gradientdescent.optimizer import RMSPropParameters
from l2l.paths import Paths
from l2l.logging_tools import create_shared_logger_data, configure_loggers

from nwn_optimizee import NWN_Optimizee
import l2l.utils.JUBE_runner as jube

logger = logging.getLogger('bin.ltl-fun-gd')


def main():
    task = "volterra"
    name = f'LTL-NWN-{task}-GD'
    params_to_learn = ["W_in_mean", "b_in_mean", "init_time"]

    try:
        with open('bin/path.conf') as f:
            root_dir_path = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            "You have not set the root path to store your results."
            " Write the path to a path.conf text file in the bin directory"
            " before running the simulation"
        )
    paths = Paths(name, dict(run_no=strftime("%Y-%m-%d-%H%M%S")), root_dir_path=root_dir_path)

    print("All output logs can be found in directory ", paths.logs_path)

    traj_file = os.path.join(paths.output_dir_path, 'data.h5')

    # Create an environment that handles running our simulation
    # This initializes an environment
    env = Environment(trajectory=name, filename=traj_file, file_title='{} data'.format(name),
                      comment='{} data'.format(name),
                      add_time=True,
                      multiproc=True,
                      automatic_storing=True,
                      log_stdout=False,  # Sends stdout to logs
                      )
    create_shared_logger_data(logger_names=['bin', 'optimizers'],
                              log_levels=['INFO', 'INFO'],
                              log_to_consoles=[True, True],
                              sim_name=name,
                              log_directory=paths.logs_path)
    configure_loggers()
    # Get the trajectory from the environment
    traj = env.trajectory

        # Set JUBE params
    traj.f_add_parameter_group("JUBE_params", "Contains JUBE parameters")

    # Scheduler parameters
    # These parameters need to be filled in when using a scheduler
    # Name of the scheduler
    # traj.f_add_parameter_to_group("JUBE_params", "scheduler", "Slurm")
    # Command to submit jobs to the schedulers
    traj.f_add_parameter_to_group("JUBE_params", "submit_cmd", "sbatch")
    # Template file for the particular scheduler
    traj.f_add_parameter_to_group("JUBE_params", "job_file", "job.run")
    # Number of nodes to request for each run
    traj.f_add_parameter_to_group("JUBE_params", "nodes", "4")
    # Requested time for the compute resources
    traj.f_add_parameter_to_group("JUBE_params", "walltime", "00:30:00")
    # MPI Processes per node
    traj.f_add_parameter_to_group("JUBE_params", "ppn", "4")
    # CPU cores per MPI process
    traj.f_add_parameter_to_group("JUBE_params", "cpu_pp", "4")
    # Threads per process
    traj.f_add_parameter_to_group("JUBE_params", "threads_pp", "4")
    # JUBE parameters for multiprocessing. Relevant even without scheduler.
    # MPI Processes per job
    traj.f_add_parameter_to_group("JUBE_params", "tasks_per_job", "4")
    # Type of emails to be sent from the scheduler
    traj.f_add_parameter_to_group("JUBE_params", "mail_mode", "ALL")
    # Email to notify events from the scheduler
    traj.f_add_parameter_to_group("JUBE_params", "mail_address", "ruomin.zhu@gmail.com")

    # These parameters need to be filled in always because JUBE takes care of exploring all the required
    # parameters from the optimizer
    # Error file for the job
    traj.f_add_parameter_to_group("JUBE_params", "err_file", "stderr")
    # Output file for the job
    traj.f_add_parameter_to_group("JUBE_params", "out_file", "stdout")
    
    traj.f_add_parameter_to_group("JUBE_params", "para_procs", "2")
    
    # The execution command

    # abs_path = os.path.abspath(root_dir_path)
    abs_path = os.path.abspath(paths.output_dir_path)
    traj.f_add_parameter_to_group("JUBE_params", "exec", "mpirun python3 " + abs_path +
                                  "/run_files/run_optimizee.py")
    # Ready file for a generation
    traj.f_add_parameter_to_group("JUBE_params", "ready_file", abs_path + "/readyfiles/ready_w_")
    # Path where the job will be executed
    traj.f_add_parameter_to_group("JUBE_params", "work_path", abs_path)

    # NOTE: Benchmark function
    optimizee = NWN_Optimizee(traj, task, params_to_learn)

    jube.prepare_optimizee(optimizee, abs_path)

    # NOTE: Outerloop optimizer initialization
    # TODO: Change the optimizer to the appropriate Optimizer class
    # parameters = ClassicGDParameters(learning_rate=0.005, exploration_step_size=0.01,
    #                                   n_random_steps=20, n_iteration=10,
    #                                   stop_criterion=np.Inf, seed=1234)
    parameters = AdamParameters(learning_rate=0.001, exploration_step_size=0.001, 
                                n_random_steps=8,  n_iteration=2, 
                                first_order_decay=0.8, second_order_decay=0.8,
                                stop_criterion=np.Inf, seed=99)
    # parameters = StochasticGDParameters(learning_rate=0.01, stochastic_deviation=1, stochastic_decay=0.99,
    #                                     exploration_step_size=0.01, n_random_steps=5, n_iteration=100,
    #                                     stop_criterion=np.Inf)
    #parameters = RMSPropParameters(learning_rate=0.01, exploration_step_size=0.01,
    #                               n_random_steps=5, momentum_decay=0.5,
    #                               n_iteration=100, stop_criterion=np.Inf, seed=99)

    optimizer = GradientDescentOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                         optimizee_fitness_weights=(-1.,),
                                         parameters=parameters,
                                         optimizee_bounding_func=optimizee.bounding_func,
                                        #  base_point_evaluations=10
                                        )

    # Add post processing
    env.add_postprocessing(optimizer.post_process)

    # Run the simulation with all parameter combinations
    env.run(optimizee.simulate)

    # NOTE: Outerloop optimizer end
    optimizer.end(traj)
    # recorder.end()
    
    # Finally disable logging and close all log-files
    env.disable_logging()

if __name__ == '__main__':
    main()