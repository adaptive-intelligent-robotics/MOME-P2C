""" Implements a function to create critic and actor losses for the TD3 algorithm."""

from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from qdax.core.neuroevolution.buffers.buffer import Transition
from qdax.types import Action, Observation, Params, RNGKey


def make_td3_loss_fn(
    policy_fn: Callable[[Params, Observation], jnp.ndarray],
    critic_fn: Callable[[Params, Observation, Action], jnp.ndarray],
    reward_scaling: Tuple[float, ...],
    discount: float,
    noise_clip: float,
    policy_noise: float,
    objective_function_index: int,
) -> Tuple[
    Callable[[Params, Params, Transition], jnp.ndarray],
    Callable[[Params, Params, Params, Transition, RNGKey], jnp.ndarray],
]:
    """Creates the loss functions for TD3.

    Args:
        policy_fn: forward pass through the neural network defining the policy.
        critic_fn: forward pass through the neural network defining the critic.
        reward_scaling: value to multiply the reward given by the environment.
        discount: discount factor.
        noise_clip: value that clips the noise to avoid extreme values.
        policy_noise: noise applied to smooth the bootstrapping.

    Returns:
        Return the loss functions used to train the policy and the critic in TD3.
    """

    @jax.jit
    def _policy_loss_fn(
        policy_params: Params,
        critic_params: Params,
        transitions: Transition,
    ) -> jnp.ndarray:
        """Policy loss function for TD3 agent"""

        action = policy_fn(policy_params, transitions.obs)
        q_value = critic_fn(
            critic_params, obs=transitions.obs, actions=action  # type: ignore
        )
        q1_action = jnp.take(q_value, jnp.asarray([0]), axis=-1)
        policy_loss = -jnp.mean(q1_action)
        return policy_loss

    @jax.jit
    def _critic_loss_fn(
        critic_params: Params,
        target_policy_params: Params,
        target_critic_params: Params,
        transitions: Transition,
        random_key: RNGKey,
    ) -> jnp.ndarray:
        """Critics loss function for TD3 agent"""
        noise = (
            jax.random.normal(random_key, shape=transitions.actions.shape)
            * policy_noise
        ).clip(-noise_clip, noise_clip)

        next_action = (
            policy_fn(target_policy_params, transitions.next_obs) + noise
        ).clip(-1.0, 1.0)
        next_q = critic_fn(  # type: ignore
            target_critic_params, obs=transitions.next_obs, actions=next_action
        )
        next_v = jnp.min(next_q, axis=-1)
        target_q = jax.lax.stop_gradient(
            transitions.rewards[:, objective_function_index] * reward_scaling
            + (1.0 - transitions.dones) * discount * next_v
        )
        q_old_action = critic_fn(  # type: ignore
            critic_params,
            obs=transitions.obs,
            actions=transitions.actions,
        )
        q_error = q_old_action - jnp.expand_dims(target_q, -1)

        # Better bootstrapping for truncated episodes.
        q_error = q_error * jnp.expand_dims(1.0 - transitions.truncations, -1)

        # compute the loss
        q_losses = jnp.mean(jnp.square(q_error), axis=-2)
        q_loss = jnp.sum(q_losses, axis=-1)

        return q_loss

    return _policy_loss_fn, _critic_loss_fn


def make_pc_td3_loss_fn(
    policy_fn: Callable[[Params, Observation], jnp.ndarray],
    pc_actor_policy_fn: Callable[[Params, Observation], jnp.ndarray],
    pc_critic_fn: Callable[[Params, Observation, Action], jnp.ndarray],
    reward_scaling: Tuple[float, ...],
    discount: float,
    noise_clip: float,
    policy_noise: float,
) -> Tuple[
    Callable[[Params, Params, Transition], jnp.ndarray],
    Callable[[Params, Params, Params, Transition, RNGKey], jnp.ndarray],
]:
    """Creates the loss functions for TD3.

    Args:

        policy_fn: forward pass through the neural network defining the policy.
        pc_actor_fn: forward pass through the neural network defining the preference-conditioned actor.
        pc_critic_fn: forward pass through the neural network defining the preference-conditioned critic.
        reward_scaling: value to multiply the reward given by the environment.
        discount: discount factor.
        noise_clip: value that clips the noise to avoid extreme values.
        policy_noise: noise applied to smooth the bootstrapping.

    Returns:
        Return the loss functions used to train non-preference conditioned policies and the pc-actor and the pc-critic in TD3.
    """

    @jax.jit
    def _policy_loss_fn(
        policy_params: Params,
        pc_critic_params: Params,
        transitions: Transition,
        preferences: jnp.ndarray,
    ) -> jnp.ndarray:
        
        """Policy loss function for non-preference conditioned policies"""

        # Action: B x action_dim
        action = policy_fn(policy_params, transitions.obs)

        # Q-value: n_critics x B x n_objectives
        q_value = pc_critic_fn(
            pc_critic_params,
            obs=transitions.obs,
            actions=action,  # type: ignore
            preferences=preferences,
            )

        # Vector q1 action: 1 x B x n_objectives
        vector_q1_action = jnp.take(q_value, jnp.asarray([0]), axis=0)

        # Scalarised q1 action: B 
        q1_action = jnp.sum(preferences * vector_q1_action, axis=-1)

        # Policy loss: scalar
        policy_loss = -jnp.mean(q1_action)

        return policy_loss

    @jax.jit
    def _pc_actor_policy_loss_fn(
        pc_actor_params: Params,
        pc_critic_params: Params,
        transitions: Transition,
    ) -> jnp.ndarray:
        """Policy loss function for preference conditioned TD3 agent"""

        # Action: B x action_dim
        action = pc_actor_policy_fn(pc_actor_params,
                                    jnp.concatenate([transitions.obs, transitions.input_preference], axis=-1)
        )

        # Q-value: n_critics x B x n_objectives
        q_value = pc_critic_fn(
            pc_critic_params,
            obs=transitions.obs,
            actions=action,  # type: ignore,
            preferences=transitions.input_preference,
        )

        # Vector q1 action: 1 x B x n_objectives
        vector_q1_action = jnp.take(q_value, jnp.asarray([0]), axis=0)

        # Q1 action: B
        q1_action = jnp.sum(transitions.input_preference * vector_q1_action, axis=-1)

        # Policy loss: scalar
        policy_loss = -jnp.mean(q1_action)

        return policy_loss
    
    @jax.jit
    def _pc_critic_loss_fn(
        pc_critic_params: Params,
        target_pc_actor_policy_params: Params,
        target_pc_critic_params: Params,
        transitions: Transition,
        random_key: RNGKey,
    ) -> jnp.ndarray:
        """Preference conditioned critics loss function for TD3 agent"""
        # Preferences: batch_size x n_objectives

        # Noise: B x action_dim
        noise = (
            jax.random.normal(random_key, shape=transitions.actions.shape)
            * policy_noise
        ).clip(-noise_clip, noise_clip)

        # Action: B x action_dim
        next_action = (
            pc_actor_policy_fn(target_pc_actor_policy_params, jnp.concatenate([transitions.next_obs, transitions.input_preference], axis=-1)
                               ) + noise
        ).clip(-1.0, 1.0)

        # Vector_next_q: n_critics x batch_size x n_objectives
        vector_next_q = pc_critic_fn(  # type: ignore
            target_pc_critic_params, 
            obs=transitions.next_obs, 
            actions=next_action,
            preferences=transitions.input_preference
        )

        # Next Q: n_critics x batch_size
        next_q = jnp.sum(vector_next_q*transitions.input_preference, axis=-1)
        # Next V:  batch_size
        next_v = jnp.min(next_q, axis=0)
        
        # Target Q: batch_size
        target_q = jax.lax.stop_gradient(
            jnp.sum(transitions.input_preference * transitions.rewards, axis=1)
            + (1.0 - transitions.dones) * discount * next_v
        )

        # Vector_q_old_action: n_critics x batch_size x n_objectives    
        vector_q_old_action = pc_critic_fn(  # type: ignore
            pc_critic_params,
            obs=transitions.obs,
            actions=transitions.actions,
            preferences=transitions.input_preference,
        )

        # Q old action: n_critics x batch_size
        q_old_action = jnp.sum(vector_q_old_action*transitions.input_preference, axis=-1)

        # Q-error: n_critics x batch_size
        q_error = q_old_action - jnp.expand_dims(target_q, 0)

        # Better bootstrapping for truncated episodes.
        # Q-error: n_critics x batch_size
        q_error = q_error * jnp.expand_dims(1.0 - transitions.truncations, 0)

        # compute the loss
        # Q-losses: n_critics 
        q_losses = jnp.mean(jnp.square(q_error), axis=-1)

        # Q-loss: scalar
        q_loss = jnp.sum(q_losses, axis=-1)

        return q_loss

    return _policy_loss_fn, _pc_actor_policy_loss_fn, _pc_critic_loss_fn