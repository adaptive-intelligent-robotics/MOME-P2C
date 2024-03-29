from typing import Callable, Tuple, List

import flax.linen as nn
from functools import partial
import jax
import jax.numpy as jnp

from qdax.core.containers.mome_repertoire import MOMERepertoire
from qdax.core.emitters.preference_sampling.preference_sampler import PreferenceSampler
from qdax.core.emitters.multi_emitter import MultiEmitter, MultiEmitterState
from qdax.core.emitters.pc_qpg_emitter import PCQualityPGConfig, PCQualityPGEmitter
from qdax.core.emitters.pga_me_emitter import PGAMEConfig
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.environments.base_wrappers import QDEnv
from qdax.types import Descriptor, ExtraScores, Genotype, Fitness, Metrics, Params, Preference, RNGKey
from qdax.utils.pareto_front import uniform_preference_sampling



class PCMOPGAEmitter(MultiEmitter):
    def __init__(
        self,
        config: PGAMEConfig,
        policy_network: nn.Module,
        pc_actor_network: nn.Module,
        pc_actor_metrics_function: Callable[[Preference, Preference], Metrics],
        inject_actor_preferences_sample_fn: Callable[[MOMERepertoire, RNGKey], jnp.ndarray],
        train_pc_networks_preferences_sample_fn: Callable[[MOMERepertoire, RNGKey], jnp.ndarray],
        pg_sampler: PreferenceSampler,
        env: QDEnv,
        variation_fn: Callable[[Params, Params, RNGKey], Tuple[Params, RNGKey]],
        inject_actor_batch_size: int,
        qpg_emit_batch_size: int,
    ) -> None:

        self._config = config
        self._policy_network = policy_network
        self._pc_actor_network = pc_actor_network
        
        inject_actor_preferences_sample_fn =  partial(inject_actor_preferences_sample_fn,
                batch_size=inject_actor_batch_size,
                num_objectives=config.num_objective_functions
        )
        
        train_pc_networks_preferences_sample_fn =  partial(train_pc_networks_preferences_sample_fn,
                batch_size=config.batch_size,
                num_objectives=config.num_objective_functions
        )
        
        self._env = env
        self._variation_fn = variation_fn

        ga_batch_size = config.mutation_ga_batch_size
        assert(config.mutation_qpg_batch_size == (inject_actor_batch_size + qpg_emit_batch_size))
        
        self._qpg_batch_size = qpg_emit_batch_size
        self._inject_actor_batch_size = inject_actor_batch_size

        emitters = []

        qpg_config = PCQualityPGConfig(
            num_objective_functions=config.num_objective_functions,
            num_critic_training_steps=config.num_critic_training_steps,
            num_pg_training_steps=config.num_pg_training_steps,
            qpg_batch_size=self._qpg_batch_size,
            inject_actor_batch_size=self._inject_actor_batch_size,
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
            pc_actor_metrics_function=pc_actor_metrics_function,
            inject_actor_preferences_sample_fn=inject_actor_preferences_sample_fn,
            train_pc_networks_preferences_sample_fn=train_pc_networks_preferences_sample_fn,
            pg_sampler=pg_sampler,
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

    @partial(jax.jit, static_argnames=("self",))   
    def update_added_counts(
        self,
        container_addition_metrics: List,
        metrics: Metrics,
    ):

        n_pg = self._qpg_batch_size
        n_inject = self._inject_actor_batch_size

        added_list = container_addition_metrics[0]
        removed_list = container_addition_metrics[1]

        metrics["removed_count"] = jnp.sum(removed_list)

        pg_added_list = added_list[:n_pg]
        inject_added_list = added_list[n_pg: n_pg+n_inject]
        mutation_added_list = added_list[n_pg+n_inject:]

        metrics[f'emitter_pg_count'] = jnp.sum(pg_added_list)
        metrics[f'emitter_actor_inject_count'] = jnp.sum(inject_added_list)
        metrics[f'emitter_ga_count'] = jnp.sum(mutation_added_list)

        return metrics