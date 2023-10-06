import flax.linen as nn
from flax.core.frozen_dict import freeze, unfreeze
import jax
import optax
from jax import numpy as jnp

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, Tuple

from qdax.core.emitters.emitter import EmitterState
from qdax.core.emitters.preference_sampling.preference_sampler import (
    PreferenceSampler,
    PreferenceSamplingState,
)
from qdax.core.containers.repertoire import Repertoire
from qdax.core.containers.mome_repertoire import MOMERepertoire
from qdax.core.emitters.emitter import Emitter
from qdax.core.neuroevolution.buffers.buffer import QDTransition, ReplayBuffer
from qdax.core.neuroevolution.losses.td3_loss import make_pc_td3_loss_fn
from qdax.core.neuroevolution.networks.networks import MOQModule
from qdax.environments.base_wrappers import QDEnv
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, Params, Preference, Metrics, RNGKey
from qdax.utils.pareto_front import uniform_preference_sampling


@dataclass
class PCQualityPGConfig:
    """Configuration for PGAME Algorithm"""
    
    num_objective_functions: int = 1 
    num_critic_training_steps: int = 300
    num_pg_training_steps: int = 100
    qpg_batch_size: int = 64
    inject_actor_batch_size: int = 64

    # TD3 params
    replay_buffer_size: int = 1000000
    critic_hidden_layer_size: Tuple[int, ...] = (256, 256)
    critic_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    policy_learning_rate: float = 1e-3
    noise_clip: float = 0.5
    policy_noise: float = 0.2
    discount: float = 0.99
    reward_scaling: float = 1.0
    batch_size: int = 256
    soft_tau_update: float = 0.005
    policy_delay: int = 2


class PCQualityPGEmitterState(EmitterState):
    """Contains training state for the learner."""

    critic_params: Params
    critic_optimizer_state: optax.OptState
    actor_params: Params
    actor_opt_state: optax.OptState
    target_critic_params: Params
    target_actor_params: Params
    replay_buffer: ReplayBuffer
    actor_preferences: Preference
    pc_actor_metrics: Metrics
    random_key: RNGKey
    steps: jnp.ndarray

    sampling_state: PreferenceSamplingState


class PCQualityPGEmitter(Emitter):
    """
    A preference-conditioned policy gradient emitter 
    """

    def __init__(
        self,
        config: PCQualityPGConfig,
        policy_network: nn.Module,
        pc_actor_network: nn.Module,
        pc_actor_metrics_function: Callable[[Preference, Preference], Metrics],
        inject_actor_preferences_sample_fn: Callable[[MOMERepertoire, RNGKey], jnp.ndarray],
        train_pc_networks_preferences_sample_fn: Callable[[MOMERepertoire, RNGKey], jnp.ndarray],
        pg_sampler: PreferenceSampler,
        env: QDEnv,
    ) -> None:
        self._config = config
        self._env = env
        self._policy_network = policy_network
        self._pc_actor_network = pc_actor_network
        self._pc_actor_metrics_function = pc_actor_metrics_function
        self._inject_actor_preferences_sample_fn = inject_actor_preferences_sample_fn
        self._train_pc_networks_preferences_sample_fn = train_pc_networks_preferences_sample_fn

        # Init Critics
        pc_critic_network = MOQModule(
            n_critics=2, 
            n_objectives = config.num_objective_functions,
            hidden_layer_sizes=self._config.critic_hidden_layer_size
        )
        self._pc_critic_network = pc_critic_network

        # Set up the losses and optimizers - return the opt states        
        self._policy_loss_fn, self._pc_policy_loss_fn, self._critic_loss_fn = make_pc_td3_loss_fn(
            policy_fn=policy_network.apply,
            pc_actor_policy_fn=pc_actor_network.apply,
            train_pc_networks_preferences_sample_fn=train_pc_networks_preferences_sample_fn,
            pc_critic_fn=pc_critic_network.apply,
            reward_scaling=self._config.reward_scaling,
            discount=self._config.discount,
            noise_clip=self._config.noise_clip,
            policy_noise=self._config.policy_noise,
        )

        # Init optimizers
        self._actor_optimizer = optax.adam(
            learning_rate=self._config.actor_learning_rate
        )
        self._critic_optimizer = optax.adam(
            learning_rate=self._config.critic_learning_rate
        )
        self._policies_optimizer = optax.adam(
            learning_rate=self._config.policy_learning_rate
        )

        # Set up the preference sampling function
        self._pg_sampler = pg_sampler

    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return self._config.inject_actor_batch_size + self._config.qpg_batch_size

    @property
    def use_all_data(self) -> bool:
        """Whether to use all data or not when used along other emitters.
        QualityPGEmitter uses the transitions from the genotypes that were generated
        by other emitters.
        """
        return True

    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[PCQualityPGEmitterState, RNGKey]:
        """Initializes the emitter state.
        Args:
            init_genotypes: The initial population.
            random_key: A random key.
        Returns:
            The initial state of the PGAMEEmitter, a new random key.
        """

        observation_size = self._env.observation_size
        action_size = self._env.action_size
        descriptor_size = self._env.state_descriptor_length

        # Initialise critic, greedy actor and population
        random_key, subkey = jax.random.split(random_key)
        fake_obs = jnp.zeros(shape=(observation_size,))
        fake_action = jnp.zeros(shape=(action_size,))
        fake_preference = jnp.zeros(shape=(self._config.num_objective_functions,))

        pc_critic_params = self._pc_critic_network.init(
            subkey,
            obs=fake_obs, 
            actions=fake_action,
            preferences=fake_preference,
        )
        target_pc_critic_params = jax.tree_util.tree_map(lambda x: x, pc_critic_params)

        random_key, subkey = jax.random.split(random_key)
        pc_actor_params = self._pc_actor_network.init(
            subkey, 
            jnp.concatenate([fake_obs, fake_preference], axis=-1),
        )

        target_pc_actor_params = jax.tree_util.tree_map(lambda x: x, pc_actor_params)

        # Prepare init optimizer states
        critic_optimizer_state = self._critic_optimizer.init(pc_critic_params)
        actor_optimizer_state = self._actor_optimizer.init(pc_actor_params)

        # Initialize replay buffer
        dummy_transition = QDTransition.init_dummy(
            observation_dim=observation_size,
            action_dim=action_size,
            descriptor_dim=descriptor_size,
            reward_dim=self._config.num_objective_functions,
        )

        replay_buffer = ReplayBuffer.init(
            buffer_size=self._config.replay_buffer_size, transition=dummy_transition
        )

        random_key, subkey = jax.random.split(random_key)
        sampling_state, random_key = self._pg_sampler.init(
            init_genotypes=init_genotypes,
            random_key=subkey,
        )
        
        # Init dummy actor preferences
        dummy_preferences, random_key = self._inject_actor_preferences_sample_fn(
            random_key = random_key,
        )
        dummy_metrics = self._pc_actor_metrics_function(dummy_preferences, dummy_preferences)
        
        # Initial training state
        random_key, subkey = jax.random.split(random_key)
        emitter_state = PCQualityPGEmitterState(
            critic_params=pc_critic_params,
            critic_optimizer_state=critic_optimizer_state,
            actor_params=pc_actor_params,
            actor_opt_state=actor_optimizer_state,
            target_critic_params=target_pc_critic_params,
            target_actor_params=target_pc_actor_params,
            random_key=subkey,
            steps=jnp.array(0),
            replay_buffer=replay_buffer,
            actor_preferences=dummy_preferences,
            pc_actor_metrics=dummy_metrics,
            sampling_state=sampling_state,
        )

        return emitter_state, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit(
        self,
        repertoire: Repertoire,
        emitter_state: PCQualityPGEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, PCQualityPGEmitterState, RNGKey]:
        """Do a step of PG emission.
        Args:
            repertoire: the current repertoire of genotypes
            emitter_state: the state of the emitter used
            random_key: a random key
        Returns:
            A batch of offspring, the new emitter state and a new key.
        """

        # sample parents
        parents, preferences, sampling_state = self._pg_sampler.sample(
            repertoire=repertoire,
            sampling_state=emitter_state.sampling_state,
        )

        emitter_state = emitter_state.replace(sampling_state=sampling_state)
        
        all_offspring = []
        
        # apply the pg mutation
        pg_genotypes = self.emit_pg(emitter_state, parents, preferences)
        all_offspring.append(pg_genotypes)
        
        # reshape actor for injection
        if self._config.inject_actor_batch_size > 0:
            random_key, subkey = jax.random.split(random_key)
            actor_genotypes, emitter_state = self.emit_actor(emitter_state, subkey)
            all_offspring.append(actor_genotypes)
        
        # concatenate offsprings together
        genotypes = jax.tree_util.tree_map(
            lambda *x: jnp.concatenate(x, axis=0), *all_offspring
        )
        
        return genotypes, emitter_state, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit_pg(
        self, 
        emitter_state: PCQualityPGEmitterState, 
        parents: Genotype,
        preferences: jnp.ndarray,
    ) -> Genotype:
        """Emit the offsprings generated through pg mutation.
        Args:
            emitter_state: current emitter state, contains critic and
                replay buffer.
            parents: the parents selected to be applied gradients in order
                to mutate towards better performance.
        Returns:
            A new set of offsprings.
        """
        mutation_fn = partial(
            self._mutation_function_pg,
            emitter_state=emitter_state,
        )
        offsprings = jax.vmap(mutation_fn)(parents, preferences)

        return offsprings
    
    
    @partial(jax.jit, static_argnames=("self",))
    def emit_actor(
        self,
        emitter_state: PCQualityPGEmitterState,
        random_key: RNGKey,
    )-> Genotype:
        
        # generate random weights
        random_weights, _ = self._inject_actor_preferences_sample_fn(
            random_key = random_key,
        )
        
        emitter_state = emitter_state.replace(actor_preferences=random_weights)
        
        # reshape to be same shape as repertoire genotypes
        partial_reshape_fun = partial(self.reshape_actor_params, actor_params=emitter_state.actor_params)
        genotypes = jax.vmap(partial_reshape_fun)(
            random_weights
        )    
        
        return genotypes, emitter_state
    
    
    @partial(jax.jit, static_argnames=("self",))
    def reshape_actor_params(
        self,
        preferences: Preference,
        actor_params: Genotype,
    )-> Genotype:
        
        # Get old kernel and bias of first layer
        kernel = actor_params["params"]["Dense_0"]["kernel"]
        bias = actor_params["params"]["Dense_0"]["bias"]
        
        # find reshape kernel and bias
        new_kernel = kernel[:-preferences.shape[0], :]
        new_bias = bias + jnp.dot(preferences, kernel[-preferences.shape[0]:])
    
        # replace kernel and bias in actor_params
        actor_params = unfreeze(actor_params)
        actor_params["params"]["Dense_0"]["kernel"] = new_kernel
        actor_params["params"]["Dense_0"]["bias"] = new_bias
        actor_params = freeze(actor_params)
        
        return actor_params       

    @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        emitter_state: PCQualityPGEmitterState,
        repertoire: Optional[Repertoire],
        genotypes: Optional[Genotype],
        fitnesses: Optional[Fitness],
        descriptors: Optional[Descriptor],
        extra_scores: ExtraScores,
    ) -> PCQualityPGEmitterState:
        """This function gives an opportunity to update the emitter state
        after the genotypes have been scored.
        Here it is used to fill the Replay Buffer with the transitions
        from the scoring of the genotypes, and then the training of the
        critic/actor happens. Hence the params of critic/actor are updated,
        as well as their optimizer states.
        Args:
            emitter_state: current emitter state.
            repertoire: the current genotypes repertoire
            genotypes: unused here - but compulsory in the signature.
            es: unused here - but compulsory in the signature.
            descriptors: unused here - but compulsory in the signature.
            extra_scores: extra information coming from the scoring function,
                this contains the transitions added to the replay buffer.
        Returns:
            New emitter state where the replay buffer has been filled with
            the new experienced transitions.
        """
        # get the transitions out of the dictionary
        assert "transitions" in extra_scores.keys(), "Missing transitions or wrong key"
        transitions = extra_scores["transitions"]
        
        actor_input_preferences = emitter_state.actor_preferences
        
        if actor_input_preferences != None:
            achieved_preferences = extra_scores["achieved_preferences"]
            actor_achieved_preferences = achieved_preferences[self._config.qpg_batch_size:self.batch_size,:]

            # calculate actor metrics:
            pc_actor_metrics = self._pc_actor_metrics_function(
                actor_achieved_preferences,
                actor_input_preferences,
            )
            
        else:
            pc_actor_metrics = {}
            
        # add transitions in the replay buffer
        replay_buffer = emitter_state.replay_buffer.insert(transitions)
        emitter_state = emitter_state.replace(replay_buffer=replay_buffer)

        # update the sampling state
        pg_fitnesses = fitnesses.at[:self._config.qpg_batch_size].get()
        sampling_state = self._pg_sampler.state_update(
            sampling_state=emitter_state.sampling_state,
            batch_new_fitnesses=pg_fitnesses,
        )
        emitter_state = emitter_state.replace(
            sampling_state=sampling_state,
            pc_actor_metrics=pc_actor_metrics,
        )

        def scan_train_critics(
            carry: PCQualityPGEmitterState, unused: Any
        ) -> Tuple[PCQualityPGEmitterState, Any]:
            emitter_state = carry
            new_emitter_state = self._train_critics(emitter_state)
            return new_emitter_state, ()

        # Train critics and greedy actor
        emitter_state, _ = jax.lax.scan(
            scan_train_critics,
            emitter_state,
            (),
            length=self._config.num_critic_training_steps,
        )

        return emitter_state  # type: ignore

    @partial(jax.jit, static_argnames=("self",))
    def _train_critics(
        self, emitter_state: PCQualityPGEmitterState
    ) -> PCQualityPGEmitterState:
        """Apply one gradient step to critics and to the greedy actor
        (contained in carry in training_state), then soft update target critics
        and target actor.
        Those updates are very similar to those made in TD3.
        Args:
            emitter_state: actual emitter state
        Returns:
            New emitter state where the critic and the greedy actor have been
            updated. Optimizer states have also been updated in the process.
        """

        # Sample a batch of transitions in the buffer
        random_key = emitter_state.random_key
        replay_buffer = emitter_state.replay_buffer
        transitions, random_key = replay_buffer.sample(
            random_key, sample_size=self._config.batch_size
        )

        # Update Critic
        (
            critic_optimizer_state,
            critic_params,
            target_critic_params,
            random_key,
        ) = self._update_critic(
            critic_params=emitter_state.critic_params,
            target_critic_params=emitter_state.target_critic_params,
            target_actor_params=emitter_state.target_actor_params,
            critic_optimizer_state=emitter_state.critic_optimizer_state,
            transitions=transitions,
            random_key=random_key,
        )

        # Update greedy actor
        (actor_optimizer_state, actor_params, target_actor_params, random_key) = jax.lax.cond(
            emitter_state.steps % self._config.policy_delay == 0,
            lambda x: self._update_actor(*x),
            lambda _: (
                emitter_state.actor_opt_state,
                emitter_state.actor_params,
                emitter_state.target_actor_params,
                emitter_state.random_key,
            ),
            operand=(
                emitter_state.actor_params,
                emitter_state.actor_opt_state,
                emitter_state.target_actor_params,
                emitter_state.critic_params,
                transitions,
                emitter_state.random_key
            ),
        )

        # Create new training state
        new_emitter_state = emitter_state.replace(
            critic_params=critic_params,
            critic_optimizer_state=critic_optimizer_state,
            actor_params=actor_params,
            actor_opt_state=actor_optimizer_state,
            target_critic_params=target_critic_params,
            target_actor_params=target_actor_params,
            random_key=random_key,
            steps=emitter_state.steps + 1,
            replay_buffer=replay_buffer,
        )

        return new_emitter_state  # type: ignore

    @partial(jax.jit, static_argnames=("self",))
    def _update_critic(
        self,
        critic_params: Params,
        target_critic_params: Params,
        target_actor_params: Params,
        critic_optimizer_state: Params,
        transitions: QDTransition,
        random_key: RNGKey,
    ) -> Tuple[Params, Params, Params, RNGKey]:

        # compute loss and gradients
        random_key, subkey = jax.random.split(random_key)
        critic_loss, critic_gradient = jax.value_and_grad(self._critic_loss_fn)(
            critic_params,
            target_actor_params,
            target_critic_params,
            transitions,
            subkey,
        )
        critic_updates, critic_optimizer_state = self._critic_optimizer.update(
            critic_gradient, critic_optimizer_state
        )

        # update critic
        critic_params = optax.apply_updates(critic_params, critic_updates)

        # Soft update of target critic network
        target_critic_params = jax.tree_map(
            lambda x1, x2: (1.0 - self._config.soft_tau_update) * x1
            + self._config.soft_tau_update * x2,
            target_critic_params,
            critic_params,
        )

        return critic_optimizer_state, critic_params, target_critic_params, random_key

    @partial(jax.jit, static_argnames=("self",))
    def _update_actor(
        self,
        actor_params: Params,
        actor_opt_state: optax.OptState,
        target_actor_params: Params,
        critic_params: Params,
        transitions: QDTransition,
        random_key: RNGKey,
    ) -> Tuple[optax.OptState, Params, Params]:

        random_key, subkey = jax.random.split(random_key)

        # Update greedy actor
        policy_loss, policy_gradient = jax.value_and_grad(self._pc_policy_loss_fn)(
            actor_params,
            critic_params,
            transitions,
            subkey,
        )
        (
            policy_updates,
            actor_optimizer_state,
        ) = self._actor_optimizer.update(policy_gradient, actor_opt_state)
        actor_params = optax.apply_updates(actor_params, policy_updates)

        # Soft update of target greedy actor
        target_actor_params = jax.tree_map(
            lambda x1, x2: (1.0 - self._config.soft_tau_update) * x1
            + self._config.soft_tau_update * x2,
            target_actor_params,
            actor_params,
        )

        return (
            actor_optimizer_state,
            actor_params,
            target_actor_params,
            random_key
        )

    @partial(jax.jit, static_argnames=("self",))
    def _mutation_function_pg(
        self,
        policy_params: Genotype,
        policy_preferences: jnp.ndarray,
        emitter_state: PCQualityPGEmitterState,
    ) -> Genotype:
        """Apply pg mutation to a policy via multiple steps of gradient descent.
        First, update the rewards to be diversity rewards, then apply the gradient
        steps.
        Args:
            policy_params: a policy, supposed to be a differentiable neural
                network.
            emitter_state: the current state of the emitter, containing among others,
                the replay buffer, the critic.
        Returns:
            The updated params of the neural network.
        """

        # Define new policy optimizer state
        policy_optimizer_state = self._policies_optimizer.init(policy_params)

        def scan_train_policy(
            carry: Tuple[PCQualityPGEmitterState, Genotype, optax.OptState],
            unused: Any,
        ) -> Tuple[Tuple[PCQualityPGEmitterState, Genotype, optax.OptState], Any]:
            emitter_state, policy_params, policy_preferences, policy_optimizer_state = carry
            (
                new_emitter_state,
                new_policy_params,
                new_policy_optimizer_state,
            ) = self._train_policy(
                emitter_state,
                policy_params,
                policy_preferences,
                policy_optimizer_state,
            )
            return (
                new_emitter_state,
                new_policy_params,
                policy_preferences,
                new_policy_optimizer_state,
            ), ()

        (emitter_state, policy_params, policy_preferences, policy_optimizer_state,), _ = jax.lax.scan(
            scan_train_policy,
            (emitter_state, policy_params, policy_preferences, policy_optimizer_state),
            (),
            length=self._config.num_pg_training_steps,
        )

        return policy_params

    @partial(jax.jit, static_argnames=("self",))
    def _train_policy(
        self,
        emitter_state: PCQualityPGEmitterState,
        policy_params: Params,
        policy_preferences: jnp.ndarray,
        policy_optimizer_state: optax.OptState,
    ) -> Tuple[PCQualityPGEmitterState, Params, optax.OptState]:
        """Apply one gradient step to a policy (called policy_params).
        Args:
            emitter_state: current state of the emitter.
            policy_params: parameters corresponding to the weights and bias of
                the neural network that defines the policy.
        Returns:
            The new emitter state and new params of the NN.
        """

        # Sample a batch of transitions in the buffer
        random_key = emitter_state.random_key
        replay_buffer = emitter_state.replay_buffer
        transitions, random_key = replay_buffer.sample(
            random_key, sample_size=self._config.batch_size
        )

        tiled_preferences = jnp.tile(policy_preferences, (self._config.batch_size, 1))

        # update policy
        policy_optimizer_state, policy_params = self._update_policy(
            critic_params=emitter_state.critic_params,
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            transitions=transitions,
            preferences=tiled_preferences,
        )

        # Create new training state
        new_emitter_state = emitter_state.replace(
            random_key=random_key,
            replay_buffer=replay_buffer,
        )

        return new_emitter_state, policy_params, policy_optimizer_state

    @partial(jax.jit, static_argnames=("self",))
    def _update_policy(
        self,
        critic_params: Params,
        policy_optimizer_state: optax.OptState,
        policy_params: Params,
        transitions: QDTransition,
        preferences: jnp.ndarray,
    ) -> Tuple[optax.OptState, Params]:

        # compute loss
        _policy_loss, policy_gradient = jax.value_and_grad(self._policy_loss_fn)(
            policy_params,
            critic_params,
            transitions,
            preferences,
        )
        # Compute gradient and update policies
        (
            policy_updates,
            policy_optimizer_state,
        ) = self._policies_optimizer.update(policy_gradient, policy_optimizer_state)
        policy_params = optax.apply_updates(policy_params, policy_updates)

        return policy_optimizer_state, policy_params
    