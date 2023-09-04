from dataclasses import dataclass
from typing import Callable, Tuple

import flax.linen as nn
from functools import partial
import jax
import jax.numpy as jnp

from qdax.core.containers.mome_repertoire import MOMERepertoire
from qdax.core.emitters.preference_sampling.preference_sampler import PreferenceSampler
from qdax.core.emitters.multi_emitter import MultiEmitter, MultiEmitterState
from qdax.core.emitters.pc_qpg_emitter import PCQualityPGConfig, PCQualityPGEmitter
from qdax.core.emitters.qpg_emitter import QualityPGConfig, QualityPGEmitter, QualityPGEmitterState
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.environments.base_wrappers import QDEnv
from qdax.types import Descriptor, ExtraScores, Genotype, Fitness, Params, Preference, RNGKey
from qdax.utils.pareto_front import uniform_preference_sampling


@dataclass
class PGAMEConfig:
    """Configuration for PGAME Algorithm"""
    
    num_objective_functions: int = 1 
    mutation_ga_batch_size: int = 128
    mutation_qpg_batch_size: int = 128
    num_critic_training_steps: int = 300
    num_pg_training_steps: int = 100

    # TD3 params
    replay_buffer_size: int = 1000000
    critic_hidden_layer_size: Tuple[int, ...] = (256, 256)
    critic_learning_rate: float = 3e-4
    greedy_learning_rate: float = 3e-4
    policy_learning_rate: float = 1e-3
    noise_clip: float = 0.5
    policy_noise: float = 0.2
    discount: float = 0.99
    reward_scaling: float = 1.0
    batch_size: int = 256
    soft_tau_update: float = 0.005
    policy_delay: int = 2

class MOPGAEmitter(MultiEmitter):
    def __init__(
        self,
        config: PGAMEConfig,
        policy_network: nn.Module,
        env: QDEnv,
        variation_fn: Callable[[Params, Params, RNGKey], Tuple[Params, RNGKey]],
    ) -> None:

        self._config = config
        self._policy_network = policy_network
        self._env = env
        self._variation_fn = variation_fn

        ga_batch_size = config.mutation_ga_batch_size
        qpg_batch_size = config.mutation_qpg_batch_size
        
        batch_size = config.mutation_qpg_batch_size // config.num_objective_functions
        batch_sizes = jnp.ones(config.num_objective_functions) * batch_size
        batch_sizes = batch_sizes.at[-1].set(config.mutation_qpg_batch_size - (config.num_objective_functions - 1) * batch_size)

        emitters = []

        for objective_index in range(config.num_objective_functions):

            qpg_config = QualityPGConfig(
                num_objective_functions=config.num_objective_functions,
                objective_function_index=objective_index,
                env_batch_size=int(batch_sizes.at[objective_index].get()),
                num_critic_training_steps=config.num_critic_training_steps,
                num_pg_training_steps=config.num_pg_training_steps,
                replay_buffer_size=config.replay_buffer_size,
                critic_hidden_layer_size=config.critic_hidden_layer_size,
                critic_learning_rate=config.critic_learning_rate,
                actor_learning_rate=config.greedy_learning_rate,
                policy_learning_rate=config.policy_learning_rate,
                noise_clip=config.noise_clip,
                policy_noise=config.policy_noise,
                discount=config.discount,
                reward_scaling=config.reward_scaling,
                batch_size=config.batch_size,
                soft_tau_update=config.soft_tau_update,
                policy_delay=config.policy_delay,
            )

            # define the quality emitter
            q_emitter = QualityPGEmitter(
                config=qpg_config, policy_network=policy_network, env=env
            )

            emitters.append(q_emitter)

        # define the GA emitter
        ga_emitter = MixingEmitter(
            mutation_fn=lambda x, r: (x, r),
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=ga_batch_size,
        )

        emitters.append(ga_emitter)

        super().__init__(emitters=tuple(emitters))


class PGAMEEmitter(MultiEmitter):
    def __init__(
        self,
        config: PGAMEConfig,
        policy_network: nn.Module,
        env: QDEnv,
        variation_fn: Callable[[Params, Params, RNGKey], Tuple[Params, RNGKey]],
    ) -> None:

        self._config = config
        self._policy_network = policy_network
        self._env = env
        self._variation_fn = variation_fn

        ga_batch_size = config.mutation_ga_batch_size
        qpg_batch_size = config.mutation_qpg_batch_size

        emitters = []

        qpg_config = QualityPGConfig(
            num_objective_functions=config.num_objective_functions,
            objective_function_index=0,
            env_batch_size=qpg_batch_size,
            num_critic_training_steps=config.num_critic_training_steps,
            num_pg_training_steps=config.num_pg_training_steps,
            replay_buffer_size=config.replay_buffer_size,
            critic_hidden_layer_size=config.critic_hidden_layer_size,
            critic_learning_rate=config.critic_learning_rate,
            actor_learning_rate=config.greedy_learning_rate,
            policy_learning_rate=config.policy_learning_rate,
            noise_clip=config.noise_clip,
            policy_noise=config.policy_noise,
            discount=config.discount,
            reward_scaling=config.reward_scaling,
            batch_size=config.batch_size,
            soft_tau_update=config.soft_tau_update,
            policy_delay=config.policy_delay,
        )

        # define the quality emitter
        q_emitter = QualityPGEmitter(
            config=qpg_config, policy_network=policy_network, env=env
        )

        emitters.append(q_emitter)

        # define the GA emitter
        ga_emitter = MixingEmitter(
            mutation_fn=lambda x, r: (x, r),
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=ga_batch_size,
        )

        emitters.append(ga_emitter)

        super().__init__(emitters=tuple(emitters))


