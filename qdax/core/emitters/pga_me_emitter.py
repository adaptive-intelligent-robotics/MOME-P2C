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
    mutation_qpg_batch_size: int = 64
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

        emitters = []

        for objective_index in range(config.num_objective_functions):

            qpg_config = QualityPGConfig(
                num_objective_functions=config.num_objective_functions,
                objective_function_index=objective_index,
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


class PCMOPGAEmitter(MultiEmitter):
    def __init__(
        self,
        config: PGAMEConfig,
        policy_network: nn.Module,
        pc_actor_network: nn.Module,
        pc_actor_scoring_function: Callable[[Params, Preference, RNGKey], Tuple[Fitness, Descriptor, Preference, ExtraScores, RNGKey]],
        pc_actor_preferences_sample_fn: Callable[[MOMERepertoire, RNGKey], jnp.ndarray],
        num_actor_active_samples: int,
        sampler: PreferenceSampler,
        env: QDEnv,
        variation_fn: Callable[[Params, Params, RNGKey], Tuple[Params, RNGKey]],
    ) -> None:

        self._config = config
        self._policy_network = policy_network
        self._pc_actor_network = pc_actor_network
        self._pc_actor_scoring_function = pc_actor_scoring_function
        self._pc_actor_preferences_sample_fn = pc_actor_preferences_sample_fn
        self._num_actor_active_samples = num_actor_active_samples
        self._env = env
        self._variation_fn = variation_fn

        ga_batch_size = config.mutation_ga_batch_size
        qpg_batch_size = config.mutation_qpg_batch_size

        emitters = []

        qpg_config = PCQualityPGConfig(
            num_objective_functions=config.num_objective_functions,
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
        pc_q_emitter = PCQualityPGEmitter(
            config=qpg_config,
            policy_network=policy_network,
            pc_actor_network=pc_actor_network,
            sampler=sampler,
            env=env
        )

        emitters.append(pc_q_emitter)

        # define the GA emitter
        ga_emitter = MixingEmitter(
            mutation_fn=lambda x, r: (x, r),
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=ga_batch_size,
        )

        emitters.append(ga_emitter)


        super().__init__(emitters=tuple(emitters))
    


    @partial(jax.jit, static_argnames=("self",))
    def evaluate_preference_conditioned_actor(
        self,
        repertoire: MOMERepertoire,
        emitter_state: MultiEmitterState,
        random_key: RNGKey,
    )-> Tuple[MultiEmitterState, RNGKey]:
        """Evaluates the preference conditioned actor on given preferences in the environment
        and adds the transitions to the replay buffer.
        """

        pg_emitter_state = emitter_state.emitter_states[0]
        ga_emitter_state = emitter_state.emitter_states[1]
                
        actor_eval_params = jax.tree_map(
            lambda x: jnp.repeat(
                jnp.expand_dims(x, axis=0), self._num_actor_active_samples, axis=0
            ),
            pg_emitter_state.actor_params
        )
        
        sampled_preferences, random_key = self._pc_actor_preferences_sample_fn(
            random_key,
        )

        _, _, _, extra_scores, random_key = self._pc_actor_scoring_function(
            actor_eval_params,
            sampled_preferences,
            random_key
        )

        transitions = extra_scores["transitions"]

        # add transitions in the replay buffer
        replay_buffer = pg_emitter_state.replay_buffer.insert(transitions)
        new_pg_emitter_state = pg_emitter_state.replace(replay_buffer=replay_buffer)

        new_emitter_state = MultiEmitterState(tuple([new_pg_emitter_state, ga_emitter_state]))

        return new_emitter_state, random_key

    def init_random_pg(
        self,
        emitter_state: MultiEmitterState,
        genotypes: Genotype,
        random_key: RNGKey,
    )-> MultiEmitterState:
        """Evaluates the preference conditioned actor on given preferences in the environment
        and adds the transitions to the replay buffer.
        """

        # get emitter state
        pg_emitter_state = emitter_state.emitter_states[0]
        pg_emitter = self.emitters[0]
        
        # generate random weights
        random_key, subkey = jax.random.split(random_key)
        random_weights, _ = uniform_preference_sampling(
            random_key = subkey,
            batch_size = self._config.mutation_qpg_batch_size + self._config.mutation_ga_batch_size,
            num_objectives = self._config.num_objective_functions,
        )
                
        # get new genotypes from pg variation on random weights
        new_genotypes = pg_emitter.emit_pg(
            emitter_state = pg_emitter_state,
            parents = genotypes,
            preferences = random_weights,
        )

        return new_genotypes, random_weights, random_key
    
    
    def init_sampler_state_update(
        self,
        emitter_state: MultiEmitterState,
        old_fitnesses: Fitness,
        weights: Preference,
    )-> MultiEmitterState:
        """Evaluates the preference conditioned actor on given preferences in the environment
        and adds the transitions to the replay buffer.
        """

        # get emitter state
        pg_emitter_state = emitter_state.emitter_states[0]
        ga_emitter_state = emitter_state.emitter_states[1]
        pg_emitter = self.emitters[0]
        

        # pg_fitnesses = fitnesses.at[:self._config.mutation_qpg_batch_size].get()
        # pg_preferences = fitnesses.at[:self._config.mutation_qpg_batch_size].get()

        new_sampling_state = pg_emitter._sampler.init_state_update(
            sampling_state=pg_emitter_state.sampling_state,
            batch_init_fitnesses=old_fitnesses,
            batch_init_preferences=weights,
        )

        new_pg_emitter_state = pg_emitter_state.replace(
            sampling_state=new_sampling_state
        )

        new_emitter_state = emitter_state.replace(
            emitter_states=tuple([new_pg_emitter_state, ga_emitter_state])
        )

        return new_emitter_state