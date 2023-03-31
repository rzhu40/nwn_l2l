from l2l.optimizers.simulatedannealing.optimizer import SimulatedAnnealingParameters, \
            SimulatedAnnealingOptimizer, AvailableCoolingSchedules
from l2l.optimizers.gradientdescent.optimizer import GradientDescentOptimizer,\
            AdamParameters
from l2l.optimizers.crossentropy.distribution import NoisyGaussian
from l2l.optimizers.crossentropy.optimizer import CrossEntropyOptimizer, CrossEntropyParameters
from l2l.optimizers.evolutionstrategies.optimizer import EvolutionStrategiesParameters, \
            EvolutionStrategiesOptimizer

optimzier_dict = {
    "SA" : SimulatedAnnealingOptimizer,
    "GD" : GradientDescentOptimizer,
    "CE" : CrossEntropyOptimizer,
    "ES" : EvolutionStrategiesOptimizer,
}


def prepare_optimizer(optimizee,
                      trajectory,
                      optimizer_type = "SA", 
                      n_individual = 10,
                      n_generation = 10, 
                      stop_criterion = -1e-5, 
                      seed = 1): 

    call_string = ""
    if optimizer_type == "SA" or optimizer_type == "simulated_annealing":
        call_string = "SA"
        parameters = SimulatedAnnealingParameters(
                        n_parallel_runs=n_individual, n_iteration=n_generation,
                        noisy_step=.05, temp_decay=.995, 
                        # noisy_step=.1, temp_decay=.995, 
                        stop_criterion=stop_criterion, seed=21343, 
                        cooling_schedule=AvailableCoolingSchedules.QUADRATIC_ADDAPTIVE)
        
    elif optimizer_type == "GD" or optimizer_type == "gradient_descent":
        call_string = "GD"
        parameters = AdamParameters(
                        learning_rate=0.01, exploration_step_size=0.01, 
                        n_random_steps=n_individual,  n_iteration=n_generation, 
                        first_order_decay=0.8, second_order_decay=0.8,
                        stop_criterion=stop_criterion, seed=99)
        
    elif optimizer_type == "CE" or optimizer_type == "cross_entropy":
        call_string = "CE"
        parameters = CrossEntropyParameters(
                        pop_size=n_individual, n_iteration=n_generation,
                        rho=0.5, smoothing=0.0, temp_decay=0, 
                        distribution=NoisyGaussian(noise_magnitude=0.01,
                                                    noise_decay=0.95),
                        # distribution=NoisyGaussian(noise_magnitude=0.1,
                        #                             noise_decay=0.95),
                        stop_criterion=stop_criterion, seed=1)

    elif optimizer_type == "ES" or optimizer_type == "evolution_strategy":
        call_string = "ES"
        parameters = EvolutionStrategiesParameters(
                        pop_size=n_individual,
                        n_iteration=n_generation,
                        learning_rate=0.05,
                        # learning_rate_decay=0.95,
                        noise_std=0.1,
                        mirrored_sampling_enabled=True,
                        fitness_shaping_enabled=True,
                        stop_criterion=stop_criterion,
                        seed=1)

    optimizer = optimzier_dict[call_string](
                    trajectory, 
                    optimizee_create_individual=optimizee.create_individual,
                    optimizee_fitness_weights=(-1.,),
                    parameters=parameters,
                    optimizee_bounding_func=optimizee.bounding_func)

    return optimizer
