import hydra
import jax.numpy as jnp
import jax
import logging
import os
import pandas as pd
import time
import visu_brax
import wandb

from brax_step_functions import play_mo_step_fn
from dataclasses import dataclass
from functools import partial
from typing import Tuple
from omegaconf import OmegaConf
from plotting_functions import plotting_function, pf_plotting_function
from qdax import environments
from qdax.baselines.nsga2 import NSGA2
from qdax.baselines.spea2 import SPEA2
from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.neuroevolution.mdp_utils import scoring_function
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.emitters.pga_me_emitter import PGAMEEmitter, PGAMEConfig
from qdax.core.emitters.mutation_operators import (
    isoline_variation,
    polynomial_mutation
)
from qdax.utils.metrics import (
    default_moqd_metrics,
    default_ga_metrics,
    default_qd_metrics,
    moqd_metrics_3d
)


@dataclass
class ExperimentConfig:
    """Configuration from this experiment script"""

    # Env config
    seed: int
    env_name: str
    algo_name: str
    fixed_init_state: bool 
    episode_length: int

    # MOO parameters
    pareto_front_max_length: int

    # Initialisation parameters
    num_evaluations: int
    num_init_cvt_samples: int
    num_centroids: int

    # Policy parameters
    policy_hidden_layer_sizes: Tuple[int,...]

    # Emitter parameters
    iso_sigma: float
    line_sigma: float 
    total_batch_size: int

    # TD3 params
    replay_buffer_size: int
    critic_hidden_layer_size: Tuple[int,...]
    critic_learning_rate: float
    greedy_learning_rate: float
    policy_learning_rate: float
    noise_clip: float 
    policy_noise: float 
    discount: float 
    transitions_batch_size: int 
    soft_tau_update: float 
    policy_delay: int
    num_critic_training_steps: int 
    num_pg_training_steps: int 

   # Logging parameters
    metrics_log_period: int
    plot_repertoire_period: int
    checkpoint_period: int
    num_save_visualisations: int
    
    metrics_list: Tuple[str,...]



@hydra.main(config_path="configs/", config_name="brax")
def main(config: ExperimentConfig) -> None:

   # Save params to weights and biases 
    wandb.init(
        # set the wandb project where this run will be logged
        project=f"PC-MOME-PGX",
        name=f"{config.algo_name}",
        # track hyperparameters and run metadata
        config=OmegaConf.to_container(config, resolve=True),
    )

    # Init environment
    env = environments.create(config.env_name, 
        episode_length=config.episode_length, 
        fixed_init_state=config.fixed_init_state)
    
    reference_point = jnp.array(config.env.reference_point)

    # Scale reference point to episode length
    reference_point *= config.episode_length/1000

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init policy network
    policy_layer_sizes = config.policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )
        
    # Init population of controllers
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=config.total_batch_size)
    fake_batch = jnp.zeros(shape=(config.total_batch_size, env.observation_size))
    init_genotypes = jax.vmap(policy_network.init)(keys, fake_batch)

    # Create the initial environment states (same initial state for each individual in env_batch)
    random_key, subkey = jax.random.split(random_key)
    keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=config.total_batch_size, axis=0)
    reset_fn = jax.jit(jax.vmap(env.reset))
    init_states = reset_fn(keys)

    # TO DO: save init_state  
    play_step_fn = partial(
        play_mo_step_fn,
        policy_network=policy_network,
        env=env,
    )  

    # Define a metrics function
    if config.env.num_objective_functions == 2:
        metrics_fn = partial(
            default_moqd_metrics,
            reference_point=jnp.array(reference_point),
            min_fitnesses=jnp.array(config.env.reference_point),
            max_fitnesses=jnp.array(config.env.max_fitnesses),
        )
    else:
        metrics_fn = partial(
            moqd_metrics_3d,
            reference_point=jnp.array(reference_point),
            min_fitnesses=jnp.array(config.env.reference_point),
            max_fitnesses=jnp.array(config.env.max_fitnesses),
        )

    metrics_list = config.wandb_metrics_keys
    if config.env.standardise_rewards:
        for obj_num in range(config.env.num_objective_functions):
            metrics_list.append(f"running_reward_mean_{obj_num+1}")
            metrics_list.append(f"running_reward_var_{obj_num+1}")


    # Prepare the scoring function
    bd_extraction_fn = environments.behavior_descriptor_extractor[config.env_name]
    scoring_fn = partial(
        scoring_function,
        init_states=init_states,
        episode_length=config.episode_length,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
        num_objective_functions=config.env.num_objective_functions,
        normalise_rewards=config.env.normalise_rewards,
        standardise_rewards=config.env.standardise_rewards,
        min_rewards=jnp.array(config.env.min_rewards),
        max_rewards=jnp.array(config.env.max_rewards),
    )

    # Get the GA emitter
    ga_variation_function = partial(
        isoline_variation, 
        iso_sigma=config.iso_sigma, 
        line_sigma=config.line_sigma
    )

    # Set up logging functions  
    num_iterations = config.num_evaluations // config.total_batch_size
    num_loops = int(num_iterations/config.metrics_log_period)
    
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().handlers[0].setLevel(logging.INFO)
    logger = logging.getLogger(f"{__name__}")
    output_dir = "./" 

    # Name save directories
    _repertoire_plots_save_dir = os.path.join(output_dir, "plots",)
    _metrics_dir = os.path.join(output_dir, "metrics")
    _repertoire_dir = os.path.join(output_dir, "repertoire")
    _visualisation_dir = os.path.join(output_dir, "visualisations")

    # Create save directories
    os.makedirs(_repertoire_plots_save_dir, exist_ok=True)
    os.makedirs(_metrics_dir, exist_ok=True)
    os.makedirs(_repertoire_dir, exist_ok=True)
    os.makedirs(_visualisation_dir, exist_ok=True)

    if config.algo_name == "nsga2" or config.algo_name == "spea2":
    
        # mutation function
        mutation_function = partial(
            polynomial_mutation,
            eta=config.algo.eta,
            minval=config.env.min_bd,
            maxval=config.env.max_bd,
            proportion_to_mutate=config.algo.proportion_to_mutate
        )

        # Define emitter
        emitter = MixingEmitter(
            mutation_fn=mutation_function, 
            variation_fn=ga_variation_function, 
            variation_percentage=config.algo.crossover_percentage, 
            batch_size=config.total_batch_size
        )
        
        if config.algo_name=="nsga2":

            algo_main = NSGA2(
                scoring_function=scoring_fn,
                emitter=emitter,
                moqd_metrics_function=metrics_fn,
                ga_metrics_function=default_ga_metrics,
            )
        
        elif config.algo_name=="spea2":
            algo_main = SPEA2(
                scoring_function=scoring_fn,
                emitter=emitter,
                moqd_metrics_function=metrics_fn,
                ga_metrics_function=default_ga_metrics,
            )
        
    elif config.algo_name == "pga":
        
        pga_metrics_fn = partial(
            default_qd_metrics,
            qd_offset=0. * config.episode_length,
        )
        
        pga_emitter_config = PGAMEConfig(
            num_objective_functions=config.env.num_objective_functions,
            mutation_ga_batch_size=config.algo.mutation_ga_batch_size,
            mutation_qpg_batch_size=config.algo.mutation_qpg_batch_size,        
            batch_size=config.transitions_batch_size,
            critic_hidden_layer_size=config.critic_hidden_layer_size,
            critic_learning_rate=config.critic_learning_rate,
            greedy_learning_rate=config.greedy_learning_rate,
            policy_learning_rate=config.policy_learning_rate,
            noise_clip=config.noise_clip,
            policy_noise=config.policy_noise,
            discount=config.discount,
            reward_scaling=[1.0],
            replay_buffer_size=config.replay_buffer_size,
            soft_tau_update=config.soft_tau_update,
            num_critic_training_steps=config.num_critic_training_steps,
            num_pg_training_steps=config.num_pg_training_steps
        )


        pg_emitter = PGAMEEmitter(
            config=pga_emitter_config,
            policy_network=policy_network,
            env=env,
            variation_fn=ga_variation_function,
        )
        
        algo_main = MAPElites(
            scoring_function=scoring_fn,
            emitter=pg_emitter,
            metrics_function=pga_metrics_fn,
            moqd_metrics_function=metrics_fn,
        )


    # Compute the centroids
    logger.warning("--- Computing the CVT centroids ---")

    # Start timing the algorithm
    init_time = time.time()
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=config.env.num_descriptor_dimensions, 
        num_init_cvt_samples=config.num_init_cvt_samples, 
        num_centroids=config.num_centroids, 
        minval=config.env.min_bd, 
        maxval=config.env.max_bd,
        random_key=random_key,
    )
    
    if config.algo_name=="pga":
        num_pga_centroids = config.num_centroids * config.pareto_front_max_length

        pga_centroids, random_key = compute_cvt_centroids(
            num_descriptors=config.env.num_descriptor_dimensions,
            num_init_cvt_samples=config.num_init_cvt_samples,
            num_centroids=num_pga_centroids,
            minval=config.env.min_bd,
            maxval=config.env.max_bd,
            random_key=random_key,
        )


    centroids_init_time = time.time() - init_time
    logger.warning(f"--- Duration for CVT centroids computation : {centroids_init_time:.2f}s ---")

    logger.warning("--- Algorithm initialisation ---")
    total_algorithm_duration = 0.0
    algorithm_start_time = time.time()

    # Initialize repertoire and emitter state
    
    if config.algo_name=="nsga2": 
        population_size = config.num_centroids * config.pareto_front_max_length
               
        repertoire, moqd_passive_repertoire, emitter_state, init_metrics, runnning_stats, random_key = algo_main.init(
            init_genotypes=init_genotypes,
            population_size=population_size,
            random_key=random_key,
            centroids=centroids,
            pareto_front_max_length=config.pareto_front_max_length,
            num_objective_functions=config.env.num_objective_functions,
        )
                
    elif config.algo_name=="spea2":
        population_size = config.num_centroids * config.pareto_front_max_length

        repertoire, moqd_passive_repertoire, emitter_state, init_metrics, runnning_stats, random_key = algo_main.init(
            init_genotypes=init_genotypes,
            population_size=population_size,
            num_neighbours=config.algo.num_neighbours,
            random_key=random_key,
            centroids=centroids,
            pareto_front_max_length=config.pareto_front_max_length,
            num_objective_functions=config.env.num_objective_functions,
        )

    elif config.algo_name=="pga":
        repertoire, moqd_passive_repertoire, emitter_state, init_metrics, runnning_stats, random_key = algo_main.init(
            init_genotypes, 
            pga_centroids, 
            centroids,
            config.pareto_front_max_length, 
            random_key,
            num_objective_functions=config.env.num_objective_functions,
        )

    initial_repertoire_time = time.time() - algorithm_start_time
    total_algorithm_duration += initial_repertoire_time
    logger.warning("--- Initialised initial repertoire ---")


    # Log initial metrics with wandb
    evaluations_done = config.total_batch_size
    logged_metrics = {"evaluations": evaluations_done,  "time": initial_repertoire_time}

    for key in metrics_list:
        # take last value
        logged_metrics[key] = init_metrics[key]

    wandb.log(logged_metrics)
    
    # Create full metrics history dict
    metrics_history = init_metrics.copy()
    for k, v in metrics_history.items():
        metrics_history[k] = jnp.expand_dims(jnp.array(v), axis=0)

    logger.warning(f"------ Initial Repertoire Metrics ------")
    logger.warning(f"--- MOQD Score: {init_metrics['moqd_score']:.2f}")
    logger.warning(f"--- Coverage: {init_metrics['coverage']:.2f}%")
    logger.warning("--- Max Fitnesses:" +  str(init_metrics['max_scores']))

    logger_header = [k for k,_ in metrics_history.items()]
    logger_header.append("time")
        
    if config.env.num_descriptor_dimensions == 2:
        plt = plotting_function(
            config,
            centroids,
            metrics_history,
            moqd_passive_repertoire,
            _repertoire_plots_save_dir,
            "init",
            config.env.num_objective_functions,
        )
        plt.close()
    
    plt = pf_plotting_function(
        moqd_passive_repertoire,
        _repertoire_plots_save_dir,
        "init",
        config.env.num_objective_functions,
    )
    plt.close()
    
    algo_main_scan_fn = algo_main.scan_update
    
    # Run the algorithm
    for iteration in range(num_loops):
        start_time = time.time()

        # 'Log period' number of QD itertions
        (repertoire, moqd_passive_repertoire, emitter_state, runnning_stats, random_key,), metrics = jax.lax.scan(
            algo_main_scan_fn,
            (repertoire, moqd_passive_repertoire, emitter_state, runnning_stats, random_key),
            (),
            length=config.metrics_log_period,
        )

        timelapse = time.time() - start_time
        total_algorithm_duration += timelapse

        # log metrics
        metrics_history = {key: jnp.concatenate((metrics_history[key], metrics[key]), axis=0) for key in metrics}
        evaluations_done += config.metrics_log_period * config.total_batch_size
        logged_metrics = {"evaluations": evaluations_done,  "time": timelapse}

        for key in metrics_list:
            # take last value
            logged_metrics[key] = metrics[key][-1]

        # Print metrics
        logger.warning(f"------ Evaluations: {evaluations_done} out of {config.num_evaluations} ------")
        logger.warning(f"--- MOQD Score: {metrics['moqd_score'][-1]:.2f}")
        logger.warning(f"--- Coverage: {metrics['coverage'][-1]:.2f}%")
        logger.warning("--- Max Fitnesses:" +  str(metrics['max_scores'][-1]))
        wandb.log(logged_metrics)

        # Save plot of repertoire every plot_repertoire_period
        if (iteration+1)*config.metrics_log_period % config.plot_repertoire_period == 0:
            if config.env.num_descriptor_dimensions == 2:
                plt = plotting_function(
                    config,
                    centroids,
                    metrics,
                    moqd_passive_repertoire,
                    _repertoire_plots_save_dir,
                    str(evaluations_done),
                    config.env.num_objective_functions,
                )
                plt.close()
                
            plt = pf_plotting_function(
                moqd_passive_repertoire,
                _repertoire_plots_save_dir,
                str(evaluations_done),
                config.env.num_objective_functions,
            )
            plt.close()
    
        # Save latest repertoire and metrics every 'checkpoint_period'
        if (iteration+1)*config.metrics_log_period  % config.checkpoint_period == 0:
            moqd_passive_repertoire.save(path=_repertoire_dir)
            metrics_history_df = pd.DataFrame.from_dict(metrics_history,orient='index').transpose()
            metrics_history_df.to_csv(os.path.join(_metrics_dir, "metrics_history.csv"), index=False)

    total_duration = time.time() - init_time

    #Calculate minimum and maximum observed rewards
    min_observed_rewards = jnp.min(metrics_history["min_rewards"], axis=0)
    max_observed_rewards = jnp.max(metrics_history["max_rewards"], axis=0)

    logger.warning("--- FINAL METRICS ---")
    logger.warning(f"Total duration: {total_duration:.2f}s")
    logger.warning(f"Main algorithm duration: {total_algorithm_duration:.2f}s")
    logger.warning(f"MOQD Score: {metrics['moqd_score'][-1]:.2f}")
    logger.warning(f"Coverage: {metrics['coverage'][-1]:.2f}%")
    logger.warning("Max Fitnesses:" + str(metrics['max_scores'][-1]))
    logger.warning("Min Observed Rewards:" +  str(min_observed_rewards))
    logger.warning("Max Observed Rewards:" +  str(max_observed_rewards))

    # Save metrics

    metrics_history_df = pd.DataFrame.from_dict(metrics_history,orient='index').transpose()
    metrics_history_df.to_csv(os.path.join(_metrics_dir, "metrics_history.csv"), index=False)

    metrics_df = pd.DataFrame.from_dict(metrics,orient='index').transpose()
    metrics_df.to_csv(os.path.join(_metrics_dir, "final_metrics.csv"), index=False)

    # Save final repertoire
    moqd_passive_repertoire.save(path=_repertoire_dir)

    # Save visualisation of best repertoire
    random_key, subkey = jax.random.split(random_key)
    
    visu_brax.save_mo_samples(
        env,                       
        policy_network,
        subkey,
        moqd_passive_repertoire, 
        config.num_save_visualisations,
        save_dir=_visualisation_dir,
    )

    # Save final plots
    if config.env.num_descriptor_dimensions == 2:

        plt = plotting_function(
            config,
            centroids,
            metrics,
            moqd_passive_repertoire,
            _repertoire_plots_save_dir,
            "final",
            config.env.num_objective_functions,
        )
                        
        wandb.log({"Final Repertoire": wandb.Image(plt)})
        plt.close()
        
    plt = pf_plotting_function(
        moqd_passive_repertoire,
        _repertoire_plots_save_dir,
        "final",
        config.env.num_objective_functions,
    )
                    
    wandb.log({"Final Global PF": wandb.Image(plt)})
    plt.close()
    
    return moqd_passive_repertoire, centroids, random_key, metrics, metrics_history

if __name__ == '__main__':
    main()

 
 

