from functools import partial
from typing import Any, Callable, Tuple

import brax
import jax
import jax.numpy as jnp
from brax.envs import State as EnvState
from flax.struct import PyTreeNode

from qdax.core.neuroevolution.buffers.buffer import (
    QDTransition,
    ReplayBuffer,
    Transition,
)
from qdax.types import (
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    Params,
    Preference,
    Reward,
    RNGKey,
)


class TrainingState(PyTreeNode):
    """The state of a training process. Can be used to store anything
    that is useful for a training process. This object is used in the
    package to store all stateful object necessary for training an agent
    that learns how to act in an MDP.
    """

    pass


@partial(
    jax.jit,
    static_argnames=(
        "num_warmstart_steps",
        "play_step_fn",
        "env_batch_size",
    ),
)
def warmstart_buffer(
    replay_buffer: ReplayBuffer,
    policy_params: Params,
    random_key: RNGKey,
    env_state: EnvState,
    play_step_fn: Callable[
        [EnvState, Params, RNGKey],
        Tuple[
            EnvState,
            Params,
            RNGKey,
            Transition,
        ],
    ],
    num_warmstart_steps: int,
    env_batch_size: int,
) -> Tuple[ReplayBuffer, EnvState]:
    """Pre-populates the buffer with transitions. Returns the warmstarted buffer
    and the new state of the environment.
    """

    def _scan_play_step_fn(
        carry: Tuple[EnvState, Params, RNGKey], unused_arg: Any
    ) -> Tuple[Tuple[EnvState, Params, RNGKey], Transition]:
        env_state, policy_params, random_key, transitions = play_step_fn(*carry)
        return (env_state, policy_params, random_key), transitions

    random_key, subkey = jax.random.split(random_key)
    (state, _, _), transitions = jax.lax.scan(
        _scan_play_step_fn,
        (env_state, policy_params, subkey),
        (),
        length=num_warmstart_steps // env_batch_size,
    )
    replay_buffer = replay_buffer.insert(transitions)

    return replay_buffer, env_state


@partial(jax.jit, static_argnames=("play_step_fn", "episode_length"))
def generate_unroll(
    init_state: EnvState,
    policy_params: Params,
    random_key: RNGKey,
    episode_length: int,
    play_step_fn: Callable[
        [EnvState, Params, RNGKey],
        Tuple[
            EnvState,
            Params,
            RNGKey,
            Transition,
        ],
    ],
) -> Tuple[EnvState, Transition]:
    """Generates an episode according to the agent's policy, returns the final state of
    the episode and the transitions of the episode.

    Args:
        init_state: first state of the rollout.
        policy_params: params of the individual.
        random_key: random key for stochasiticity handling.
        episode_length: length of the rollout.
        play_step_fn: function describing how a step need to be taken.

    Returns:
        A new state, the experienced transition.
    """

    def _scan_play_step_fn(
        carry: Tuple[EnvState, Params, RNGKey], unused_arg: Any
    ) -> Tuple[Tuple[EnvState, Params, RNGKey], Transition]:
        env_state, policy_params, random_key, transitions = play_step_fn(*carry)
        return (env_state, policy_params, random_key), transitions

    (state, _, _), transitions = jax.lax.scan(
        _scan_play_step_fn,
        (init_state, policy_params, random_key),
        (),
        length=episode_length,
    )
    return state, transitions


@partial(jax.jit, static_argnames=("pc_play_step_fn", "episode_length"))
def generate_pc_unroll(
    init_state: EnvState,
    pc_actor_policy_params: Params,
    preference: Preference,
    random_key: RNGKey,
    episode_length: int,
    pc_play_step_fn: Callable[
        [EnvState, Params, RNGKey],
        Tuple[
            EnvState,
            Params,
            RNGKey,
            Transition,
        ],
    ],
) -> Tuple[EnvState, Transition]:
    """Generates an episode according to the agent's policy conditioned on a preference.
    Returns the final state of the episode and the transitions of the episode.

    Args:
        init_state: first state of the rollout.
        pc_policy_params: params of the individual.
        preference: preference of the individual.
        random_key: random key for stochasiticity handling.
        episode_length: length of the rollout.
        play_step_fn: function describing how a step need to be taken.

    Returns:
        A new state, the experienced transition.
    """

    def _scan_play_pc_step_fn(
        carry: Tuple[EnvState, Params, RNGKey], unused_arg: Any
    ) -> Tuple[Tuple[EnvState, Params, RNGKey], Transition]:
        _, _, preference, _ = carry

        env_state, policy_params, random_key, transitions = pc_play_step_fn(*carry)
        return (env_state, policy_params, preference, random_key), transitions

    (state, _, _, _), transitions = jax.lax.scan(
        _scan_play_pc_step_fn,
        (init_state, pc_actor_policy_params, preference, random_key),
        (),
        length=episode_length,
    )
    return state, transitions


@partial(
    jax.jit,
    static_argnames=(
        "episode_length",
        "play_step_fn",
        "behavior_descriptor_extractor",
        "num_objective_functions",
        "normalise_rewards",
        "standardise_rewards",
    ),
)
def scoring_function(
    policies_params: Genotype,
    running_reward_mean: jnp.ndarray,
    running_reward_var: jnp.ndarray,
    running_reward_count: int,
    random_key: RNGKey,
    init_states: brax.envs.State,
    episode_length: int,
    play_step_fn: Callable[
        [EnvState, Params, RNGKey, brax.envs.Env],
        Tuple[EnvState, Params, RNGKey, QDTransition],
    ],
    behavior_descriptor_extractor: Callable[[QDTransition, jnp.ndarray], Descriptor],
    num_objective_functions: int,
    normalise_rewards: bool,
    standardise_rewards: bool,
    min_rewards: Reward,
    max_rewards: Reward,
) -> Tuple[Fitness, Descriptor, Preference, ExtraScores, RNGKey]:
    """Evaluates policies contained in policies_params in parallel in
    deterministic or pseudo-deterministic environments.

    This rollout is only deterministic when all the init states are the same.
    If the init states are fixed but different, as a policy is not necessarly
    evaluated with the same environment everytime, this won't be determinist.
    When the init states are different, this is not purely stochastic.
    """

    # Perform rollouts with each policy
    random_key, subkey = jax.random.split(random_key)
    unroll_fn = partial(
        generate_unroll,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        random_key=subkey,
    )

    _final_state, data = jax.vmap(unroll_fn)(init_states, policies_params)

    # create a mask to extract data properly
    is_done = jnp.clip(jnp.cumsum(data.dones, axis=1), 0, 1)
    mask = jnp.roll(is_done, 1, axis=1)
    mask = mask.at[:, 0].set(0)
    fitnesses_mask = jnp.repeat(jnp.expand_dims(mask, axis=-1), repeats=num_objective_functions, axis=-1)

    # Scores - add offset to ensure positive fitness (through positive rewards)
    fitnesses = jnp.sum(data.rewards * (1.0 - fitnesses_mask), axis=1)
    descriptors = behavior_descriptor_extractor(data, mask)

    # set running reward mean and standard deviation to None
    new_running_reward_mean = None
    new_running_reward_std = None

    if normalise_rewards:
                
        # Calculate normalised fitnesses
        normalised_rewards = data.rewards - min_rewards / (max_rewards - min_rewards)
        normalised_fitnesses = jnp.sum(normalised_rewards * (1.0 - fitnesses_mask), axis=1)

        # Calculate achieved preferences
        preferences = normalised_fitnesses / jnp.transpose(jnp.expand_dims(jnp.sum(normalised_fitnesses, axis=-1), axis=0))

        data = data.replace(rewards=normalised_rewards)
        
    elif standardise_rewards:
                
        # calculate mean and batch size of rewards
        nan_rewards = jnp.where(fitnesses_mask, jnp.nan, data.rewards)
        all_rewards = jnp.concatenate(nan_rewards, axis=0)
        rewards_batch_count = jnp.sum(mask)
        rewards_batch_mean = jnp.nanmean(all_rewards, axis=0)
        rewards_batch_var = jnp.nanvar(all_rewards, axis=0)
        
        # update runnning mean and standard deviation
        def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
            delta = batch_mean - mean
            tot_count = count + batch_count

            new_mean = mean + delta * batch_count / tot_count
            m_a = var * count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
            new_var = M2 / tot_count
            new_count = tot_count

            return new_mean, new_var, new_count

        new_rm, new_rv, new_rc = update_mean_var_count_from_moments(
            running_reward_mean,
            running_reward_var,
            running_reward_count,
            rewards_batch_mean,
            rewards_batch_var,
            rewards_batch_count,
        )
        
        # Calculate standardised fitnesses
        standardised_rewards = (data.rewards - new_rm) / jnp.sqrt(new_rv)
        standardised_fitnesses = jnp.sum(standardised_rewards * (1.0 - fitnesses_mask), axis=1)
        
        # Calculate achieved preferences
        preferences = standardised_fitnesses / jnp.transpose(jnp.expand_dims(jnp.sum(standardised_fitnesses, axis=-1), axis=0))

        data = data.replace(rewards=standardised_rewards)
        
    # Calculate preferences based on fitnesses:
    preferences = jnp.clip(preferences, 0.0, 1.0)

    # Add preferences to transitions
    tiled_preferences = jnp.repeat(
            preferences[:, jnp.newaxis, :], episode_length, axis=1)
    
    data = data.replace(preference=tiled_preferences,
                        input_preference=tiled_preferences
    )

    return (
        fitnesses,
        descriptors,
        preferences,
        {
            "transitions": data,
            "min_rewards": jnp.min(data.rewards, axis=(0,1)),
            "max_rewards": jnp.max(data.rewards, axis=(0,1)),
            "running_reward_mean": new_rm,
            "running_reward_var": new_rv,
            "running_reward_count": new_rc,
            
        },
        random_key,
    )


@partial(
    jax.jit,
    static_argnames=(
        "episode_length",
        "pc_play_step_fn",
        "behavior_descriptor_extractor",
        "num_objective_functions",
        "normalise_rewards",
        "standardise_rewards",
    ),
)
def preference_conditioned_scoring_function(
    pc_actor_params: Genotype,
    input_preferences: Preference,
    running_reward_mean: jnp.ndarray,
    running_reward_var: jnp.ndarray,
    running_reward_count: int,
    random_key: RNGKey,
    init_states: brax.envs.State,
    episode_length: int,
    pc_play_step_fn: Callable[
        [EnvState, Params, RNGKey, brax.envs.Env],
        Tuple[EnvState, Params, RNGKey, QDTransition],
    ],
    behavior_descriptor_extractor: Callable[[QDTransition, jnp.ndarray], Descriptor],
    num_objective_functions: int,
    normalise_rewards: bool,
    standardise_rewards: bool,
    min_rewards: jnp.ndarray,
    max_rewards: jnp.ndarray,
) -> Tuple[Fitness, Descriptor, Preference, ExtraScores, RNGKey]:
    """Evaluates policies contained in preference conditioned policies_params in parallel in
    deterministic or pseudo-deterministic environments.

    This rollout is only deterministic when all the init states are the same.
    If the init states are fixed but different, as a policy is not necessarly
    evaluated with the same environment everytime, this won't be determinist.
    When the init states are different, this is not purely stochastic.
    """

    # Perform rollouts with each policy
    random_key, subkey = jax.random.split(random_key)
    unroll_fn = partial(
        generate_pc_unroll,
        random_key=subkey,
        episode_length=episode_length,
        pc_play_step_fn=pc_play_step_fn,
    )

    _final_state, data = jax.vmap(unroll_fn)(init_states, pc_actor_params, input_preferences)

    # create a mask to extract data properly
    is_done = jnp.clip(jnp.cumsum(data.dones, axis=1), 0, 1)
    mask = jnp.roll(is_done, 1, axis=1)
    mask = mask.at[:, 0].set(0)
    fitnesses_mask = jnp.repeat(jnp.expand_dims(mask, axis=-1), repeats=num_objective_functions, axis=-1)

    # Scores - add offset to ensure positive fitness (through positive rewards)
    fitnesses = jnp.sum(data.rewards * (1.0 - fitnesses_mask), axis=1)
    descriptors = behavior_descriptor_extractor(data, mask)

    # set running reward mean and standard deviation to None
    new_running_reward_mean = None
    new_running_reward_std = None
    
    if normalise_rewards:
                
        # Calculate normalised fitnesses
        normalised_rewards = data.rewards - min_rewards / (max_rewards - min_rewards)
        normalised_fitnesses = jnp.sum(normalised_rewards * (1.0 - fitnesses_mask), axis=1)

        # Calculate achieved preferences
        achieved_preferences = normalised_fitnesses / jnp.transpose(jnp.expand_dims(jnp.sum(normalised_fitnesses, axis=-1), axis=0))

        data = data.replace(rewards=normalised_rewards)
        
    elif standardise_rewards:
        
        # calculate mean and batch size of rewards
        nan_rewards = jnp.where(fitnesses_mask, jnp.nan, data.rewards)
        all_rewards = jnp.concatenate(nan_rewards, axis=0)
        rewards_batch_count = jnp.sum(mask)
        rewards_batch_mean = jnp.nanmean(all_rewards, axis=0)
        rewards_batch_var = jnp.nanvar(all_rewards, axis=0)
        
        # update runnning mean and standard deviation
        def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
            delta = batch_mean - mean
            tot_count = count + batch_count

            new_mean = mean + delta * batch_count / tot_count
            m_a = var * count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
            new_var = M2 / tot_count
            new_count = tot_count

            return new_mean, new_var, new_count

        new_rm, new_rv, new_rc = update_mean_var_count_from_moments(
            running_reward_mean,
            running_reward_var,
            running_reward_count,
            rewards_batch_mean,
            rewards_batch_var,
            rewards_batch_count,
        )
        
        # Calculate standardised fitnesses
        standardised_rewards = (data.rewards - new_rm) / jnp.sqrt(new_rv)
        standardised_fitnesses = jnp.sum(standardised_rewards * (1.0 - fitnesses_mask), axis=1)
        
        # Calculate achieved preferences
        achieved_preferences = standardised_fitnesses / jnp.transpose(jnp.expand_dims(jnp.sum(standardised_fitnesses, axis=-1), axis=0))

        data = data.replace(rewards=standardised_rewards)

        
    # Add preferences to transitions
    achieved_preferences = jnp.clip(achieved_preferences, 0.0, 1.0)

    tiled_input_preferences = jnp.repeat(
            input_preferences[:, jnp.newaxis, :], episode_length, axis=1)

    tiled_achieved_preferences = jnp.repeat(
            achieved_preferences[:, jnp.newaxis, :], episode_length, axis=1)


    data = data.replace(preference=tiled_achieved_preferences,
                        input_preference=tiled_input_preferences
    )
    
    
    return (
        fitnesses,
        descriptors,
        achieved_preferences,
        {
            "transitions": data,
            "running_reward_mean": new_rm,
            "running_reward_var": new_rv,
            "running_reward_count": new_rc,
        },
        random_key,
    )


@partial(
    jax.jit,
    static_argnames=(
        "episode_length",
        "play_reset_fn",
        "play_step_fn",
        "behavior_descriptor_extractor",
    ),
)
def reset_based_scoring_function(
    policies_params: Genotype,
    random_key: RNGKey,
    episode_length: int,
    play_reset_fn: Callable[[RNGKey], brax.envs.State],
    play_step_fn: Callable[
        [brax.envs.State, Params, RNGKey, brax.envs.Env],
        Tuple[brax.envs.State, Params, RNGKey, QDTransition],
    ],
    behavior_descriptor_extractor: Callable[[QDTransition, jnp.ndarray], Descriptor],
) -> Tuple[Fitness, Descriptor, Preference, ExtraScores, RNGKey]:
    """Evaluates policies contained in policies_params in parallel.
    The play_reset_fn function allows for a more general scoring_function that can be
    called with different batch-size and not only with a batch-size of the same
    dimension as init_states.

    To define purely stochastic environments, using the reset function from the
    environment, use "play_reset_fn = env.reset".

    To define purely deterministic environments, as in "scoring_function", generate
    a single init_state using "init_state = env.reset(random_key)", then use
    "play_reset_fn = lambda random_key: init_state".
    """

    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(
        subkey, jax.tree_util.tree_leaves(policies_params)[0].shape[0]
    )
    reset_fn = jax.vmap(play_reset_fn)
    init_states = reset_fn(keys)

    fitnesses, descriptors, preferences, extra_scores, random_key = scoring_function(
        policies_params=policies_params,
        random_key=random_key,
        init_states=init_states,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=behavior_descriptor_extractor,
    )

    return (fitnesses, descriptors, preferences, extra_scores, random_key)


@partial(
    jax.jit,
    static_argnames=(
        "episode_length",
        "play_reset_fn",
        "play_step_fn",
        "behavior_descriptor_extractor",
    ),
)
def reset_based_preference_conditioned_scoring_function(
    pc_actor_params: Genotype,
    input_preferences: Preference,
    random_key: RNGKey,
    episode_length: int,
    play_reset_fn: Callable[[RNGKey], brax.envs.State],
    play_step_fn: Callable[
        [brax.envs.State, Params, RNGKey, brax.envs.Env],
        Tuple[brax.envs.State, Params, RNGKey, QDTransition],
    ],
    behavior_descriptor_extractor: Callable[[QDTransition, jnp.ndarray], Descriptor],
) -> Tuple[Fitness, Descriptor, Preference, ExtraScores, RNGKey]:
    """Evaluates policies contained in policies_params in parallel.
    The play_reset_fn function allows for a more general scoring_function that can be
    called with different batch-size and not only with a batch-size of the same
    dimension as init_states.

    To define purely stochastic environments, using the reset function from the
    environment, use "play_reset_fn = env.reset".

    To define purely deterministic environments, as in "scoring_function", generate
    a single init_state using "init_state = env.reset(random_key)", then use
    "play_reset_fn = lambda random_key: init_state".
    """

    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(
        subkey, jax.tree_util.tree_leaves(pc_actor_params)[0].shape[0]
    )
    reset_fn = jax.vmap(play_reset_fn)
    init_states = reset_fn(keys)

    fitnesses, descriptors, preferences, extra_scores, random_key = preference_conditioned_scoring_function(
        policies_params=pc_actor_params,
        input_preferences=input_preferences,
        random_key=random_key,
        init_states=init_states,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=behavior_descriptor_extractor,
    )

    return (fitnesses, descriptors, preferences, extra_scores, random_key)



@partial(
    jax.jit,
    static_argnames=(
        "env_batch_size",
        "grad_updates_per_step",
        "play_step_fn",
        "update_fn",
    ),
)
def do_iteration_fn(
    training_state: TrainingState,
    env_state: EnvState,
    replay_buffer: ReplayBuffer,
    env_batch_size: int,
    grad_updates_per_step: float,
    play_step_fn: Callable[
        [EnvState, Params, RNGKey],
        Tuple[
            EnvState,
            Params,
            RNGKey,
            Transition,
        ],
    ],
    update_fn: Callable[
        [TrainingState, ReplayBuffer],
        Tuple[
            TrainingState,
            ReplayBuffer,
            Metrics,
        ],
    ],
) -> Tuple[TrainingState, EnvState, ReplayBuffer, Metrics]:
    """Performs one environment step (over all env simultaneously) followed by one
    training step. The number of updates is controlled by the parameter
    `grad_updates_per_step` (0 means no update while 1 means `env_batch_size`
    updates). Returns the updated states, the updated buffer and the aggregated
    metrics.
    """

    def _scan_update_fn(
        carry: Tuple[TrainingState, ReplayBuffer], unused_arg: Any
    ) -> Tuple[Tuple[TrainingState, ReplayBuffer], Metrics]:
        training_state, replay_buffer, metrics = update_fn(*carry)
        return (training_state, replay_buffer), metrics

    # play steps in the environment
    random_key = training_state.random_key
    env_state, _, random_key, transitions = play_step_fn(
        env_state,
        training_state.policy_params,
        random_key,
    )

    # insert transitions in replay buffer
    replay_buffer = replay_buffer.insert(transitions)
    num_updates = int(grad_updates_per_step * env_batch_size)

    (training_state, replay_buffer), metrics = jax.lax.scan(
        _scan_update_fn,
        (training_state, replay_buffer),
        (),
        length=num_updates,
    )

    return training_state, env_state, replay_buffer, metrics


@jax.jit
def get_first_episode(transition: Transition) -> Transition:
    """Extracts the first episode from a batch of transitions, returns the batch of
    transitions that is masked with nans except for the first episode.
    """

    dones = jnp.roll(transition.dones, 1, axis=0).at[0].set(0)
    mask = 1 - jnp.clip(jnp.cumsum(dones, axis=0), 0, 1)

    def mask_episodes(x: jnp.ndarray) -> jnp.ndarray:
        # the double transpose trick is here to allow easy broadcasting
        return jnp.where(mask.T, x.T, jnp.nan * jnp.ones_like(x).T).T

    return jax.tree_util.tree_map(mask_episodes, transition)  # type: ignore


