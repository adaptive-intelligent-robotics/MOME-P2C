from typing import Tuple
from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore


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
    if config.algo.name in ["mome", "mome-pgx", "mome-p2c-keep-prefs", "mome-p2c", "mome-p2c-actor-random-sampler", "mome-p2c-no-actor", "mome-p2c-no-qpg", "mome-p2c-no-crowding", "mome-p2c-one-hot"]:
        import main_mome as main
    elif config.algo.name in ["pga", "nsga2", "spea2"]:
        import main_other as main
    else:
        raise NotImplementedError

    main.main(config)


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="main", node=ExperimentConfig)
    main()
