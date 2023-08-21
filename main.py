import hydra
import jax.numpy as jnp
import jax
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import visu_brax
import wandb

from brax_step_functions import play_mo_step_fn, play_pc_mo_step_fn
from dataclasses import dataclass
from functools import partial
from typing import Tuple
from omegaconf import OmegaConf
from plotting_functions import plotting_function
from qdax import environments
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.emitters.pga_me_emitter import PGAMEConfig, MOPGAEmitter
from qdax.core.emitters.pc_mome_pgx_emitter import PCMOPGAEmitter
from qdax.core.mome import MOME
from qdax.core.neuroevolution.mdp_utils import scoring_function, preference_conditioned_scoring_function
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.emitters.preference_sampling.naive_samplers import (
    KeepPreferencesSampler,
    NaiveSamplingConfig,
    OneHotPreferenceSampler,
    RandomPreferenceSampler,
)
from qdax.core.emitters.preference_sampling.hyperbolic_model import HyperbolicModelConfig, HyperbolicPredictionGuidedSampler
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.emitters.mutation_operators import (
    isoline_variation,
    polynomial_mutation
)

from qdax.utils.metrics import default_moqd_metrics, pc_actor_metrics
from qdax.utils.pareto_front import uniform_preference_sampling


preference_conditioned_algos = [
    "pc-mome-pgx",
    "one-hot-pc-mome-pgx",
    "pg-pc-mome-pgx",
    "random-pc-mome-pgx"
]

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
    reward_scaling: Tuple[float, ...]
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

    if config.algo_name in preference_conditioned_algos:
        actor_keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=config.algo.num_actor_active_samples, axis=0)
        actor_init_states = reset_fn(actor_keys)

    # TO DO: save init_state  
    play_step_fn = partial(
        play_mo_step_fn,
        policy_network=policy_network,
        env=env,
        min_rewards=jnp.array(config.env.min_rewards),
        max_rewards=jnp.array(config.env.max_rewards),
    )  

    # Define a metrics function
    metrics_fn = partial(
        default_moqd_metrics,
        reference_point=jnp.array(reference_point),
    )

    # Prepare the scoring function
    bd_extraction_fn = environments.behavior_descriptor_extractor[config.env_name]
    scoring_fn = partial(
        scoring_function,
        init_states=init_states,
        episode_length=config.episode_length,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
        num_objective_functions=config.env.num_objective_functions,
        min_rewards=jnp.array(config.env.min_rewards),
        max_rewards=jnp.array(config.env.max_rewards),
    )


    # Get the GA emitter
    ga_variation_function = partial(
        isoline_variation, 
        iso_sigma=config.iso_sigma, 
        line_sigma=config.line_sigma
    )


    if config.algo_name == "mome-pgx" or config.algo_name in preference_conditioned_algos:
        # Define the PG-emitter config
        pg_emitter_config = PGAMEConfig(
            num_objective_functions=config.env.num_objective_functions,
            mutation_ga_batch_size=config.algo.mutation_ga_batch_size,
            mutation_qpg_batch_size=config.algo.mutation_qpg_batch_size,
            critic_hidden_layer_size=config.critic_hidden_layer_size,
            critic_learning_rate=config.critic_learning_rate,
            greedy_learning_rate=config.greedy_learning_rate,
            policy_learning_rate=config.policy_learning_rate,
            noise_clip=config.noise_clip,
            policy_noise=config.policy_noise,
            discount=config.discount,
            reward_scaling=config.reward_scaling,
            replay_buffer_size=config.replay_buffer_size,
            soft_tau_update=config.soft_tau_update,
            policy_delay=config.policy_delay,
            num_critic_training_steps=config.num_critic_training_steps,
            num_pg_training_steps=config.num_pg_training_steps
        )

    if config.algo_name == "mome":

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

    if config.algo_name == "mome-pgx":

        emitter = MOPGAEmitter(
            config=pg_emitter_config,
            policy_network=policy_network,
            env=env,
            variation_fn=ga_variation_function,
        )

    if config.algo_name in preference_conditioned_algos:
        pc_actor_layer_sizes = config.algo.pc_actor_hidden_layer_sizes + (env.action_size,)
        pc_actor_network = MLP(
            layer_sizes=pc_actor_layer_sizes,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            final_activation=jnp.tanh,
        )
    
        play_pc_mo_step_function = partial(
            play_pc_mo_step_fn,
            policy_network=pc_actor_network,
            env=env,
            min_rewards=jnp.array(config.env.min_rewards),
            max_rewards=jnp.array(config.env.max_rewards),
        )  
        
        if config.algo.pc_actor_uniform_preference_sampling:
            actor_sampling_fn = partial(uniform_preference_sampling,
                batch_size=config.algo.num_actor_active_samples,
                num_objectives=config.env.num_objective_functions
            )

            preference_conditioned_scoring_fn = partial(preference_conditioned_scoring_function,
                init_states=actor_init_states,
                episode_length=config.episode_length,
                pc_play_step_fn=play_pc_mo_step_function,
                behavior_descriptor_extractor=bd_extraction_fn,
                num_objective_functions=config.env.num_objective_functions,
                min_rewards=jnp.array(config.env.min_rewards),
                max_rewards=jnp.array(config.env.max_rewards),
            )

        sampling_config = NaiveSamplingConfig(
            num_objectives=config.env.num_objective_functions,
            emitter_batch_size=config.algo.mutation_qpg_batch_size,
        )

        if config.algo.sampler == "random":
            sampler = RandomPreferenceSampler(
                config=sampling_config,
            )
        elif config.algo.sampler == "keep":
            sampler = KeepPreferencesSampler(
                config=sampling_config,
            )
        elif config.algo.sampler == "one-hot":
            sampler = OneHotPreferenceSampler(
                config=sampling_config,
            )

        elif config.algo.sampler == "prediction_guided":
            
            assert(config.algo.num_neighbours < config.algo.mutation_qpg_batch_size)
            
            sampling_config = HyperbolicModelConfig(
                num_objectives=config.env.num_objective_functions,
                reference_point=jnp.array(config.env.reference_point),
                emitter_batch_size=config.algo.mutation_qpg_batch_size,
                buffer_size=config.algo.hyperbolic_buffer_size,
                num_weight_candidates=config.algo.num_weight_candidates,
                num_neighbours=config.algo.num_neighbours,
                scaling_sigma=config.algo.scaling_sigma,
            )

            sampler = HyperbolicPredictionGuidedSampler(
                config=sampling_config,
            )

        emitter = PCMOPGAEmitter(
            config=pg_emitter_config,
            policy_network=policy_network,
            pc_actor_network=pc_actor_network,
            pc_actor_scoring_function=preference_conditioned_scoring_fn,
            pc_actor_metrics_function=pc_actor_metrics,
            pc_actor_preferences_sample_fn=actor_sampling_fn,
            num_actor_active_samples=config.algo.num_actor_active_samples,
            sampler=sampler,
            env=env,
            variation_fn=ga_variation_function,
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

    # Instantiate MOME
    mome = MOME(
        scoring_function=scoring_fn,
        emitter=emitter,
        metrics_function=metrics_fn,
        bias_sampling=config.algo.bias_sampling,
        preference_conditioned=config.algo.preference_conditioned,
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


    centroids_init_time = time.time() - init_time
    logger.warning(f"--- Duration for CVT centroids computation : {centroids_init_time:.2f}s ---")

    logger.warning("--- Algorithm initialisation ---")
    total_algorithm_duration = 0.0
    algorithm_start_time = time.time()

    # Initialize repertoire and emitter state
    repertoire, init_metrics, emitter_state, random_key = mome.init(
        init_genotypes,
        centroids,
        config.pareto_front_max_length,
        random_key
    )

    initial_repertoire_time = time.time() - algorithm_start_time
    total_algorithm_duration += initial_repertoire_time
    logger.warning("--- Initialised initial repertoire ---")


    # Store initial repertoire metrics and convert to jnp.arrays
    metrics_history = init_metrics.copy()
    for k, v in metrics_history.items():
        metrics_history[k] = jnp.expand_dims(jnp.array(v), axis=0)

    logger.warning(f"------ Initial Repertoire Metrics ------")
    logger.warning(f"--- MOQD Score: {init_metrics['moqd_score']:.2f}")
    logger.warning(f"--- Coverage: {init_metrics['coverage']:.2f}%")
    logger.warning("--- Max Fitnesses:" +  str(init_metrics['max_scores']))

    logger_header = [k for k,_ in metrics_history.items()]
    logger_header.insert(0, "iteration")
    logger_header.append("time")
    
    mome_scan_fn = mome.scan_update
    
    if config.env.num_descriptor_dimensions == 2:
        plt = plotting_function(
            config,
            centroids,
            metrics_history,
            repertoire,
            _repertoire_plots_save_dir,
            "init",
        )
        plt.close()
    
    # Run the algorithm
    for iteration in range(num_loops):
        start_time = time.time()

        # 'Log period' number of QD itertions
        (repertoire, emitter_state, random_key,), metrics = jax.lax.scan(
            mome_scan_fn,
            (repertoire, emitter_state, random_key),
            (),
            length=config.metrics_log_period,
        )

        timelapse = time.time() - start_time
        total_algorithm_duration += timelapse

        # log metrics
        metrics_history = {key: jnp.concatenate((metrics_history[key], metrics[key]), axis=0) for key in metrics}
        logged_metrics = {"iteration": (iteration+ 1)*config.metrics_log_period,  "time": timelapse}

        wandb_metrics_keys = [
            "moqd_score",
            "max_hypervolume",
            "max_sum_scores",
            "coverage",
            "global_hypervolume",
            "total_num_solutions"]

        for key in wandb_metrics_keys:
            # take last value
            logged_metrics[key] = metrics[key][-1]

        # Print metrics
        logger.warning(f"------ Iteration: {(iteration+1)*config.metrics_log_period} out of {num_iterations} ------")
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
                    repertoire,
                    _repertoire_plots_save_dir,
                    str((iteration+1)*config.metrics_log_period)
                )
                
                plt.close()

        # Save latest repertoire and metrics every 'checkpoint_period'
        if (iteration+1)*config.metrics_log_period  % config.checkpoint_period == 0:
            repertoire.save(path=_repertoire_dir)
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
    repertoire.save(path=_repertoire_dir)

    # Save visualisation of best repertoire
    random_key, subkey = jax.random.split(random_key)
    
    visu_brax.save_mo_samples(
        env,                       
        policy_network,
        subkey,
        repertoire, 
        config.num_save_visualisations,
        save_dir=_visualisation_dir,
    )

    # Save final plots
    if config.env.num_descriptor_dimensions == 2:

        plt = plotting_function(
            config,
            centroids,
            metrics,
            repertoire,
            _repertoire_plots_save_dir,
            "final",
        )
                        
        wandb.log({"Final Repertoire": wandb.Image(plt)})
        plt.savefig(f"{_repertoire_plots_save_dir}/repertoire_final")
        plt.close()

    return repertoire, centroids, random_key, metrics, metrics_history

if __name__ == '__main__':
    main()

 
 

