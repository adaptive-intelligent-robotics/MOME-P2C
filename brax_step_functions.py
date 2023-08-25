import jax.numpy as jnp
import jax
from qdax.core.neuroevolution.buffers.buffer import QDTransition


def play_mo_step_fn(
        env_state,
        policy_params,
        random_key,
        policy_network,
        env,
    ):
        """
        Play an environment step and return the updated state and the transition.

        Rewards at each timestep are normalised by the maximum and minimum rewards
        """

        actions = policy_network.apply(policy_params, env_state.obs)
        
        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, actions)
        
        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            actions=actions,
            truncations=next_state.info["truncation"],
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
            preference=jnp.zeros(next_state.reward.shape) * jnp.nan,
            input_preference=jnp.zeros(next_state.reward.shape) * jnp.nan,
        )

        return next_state, policy_params, random_key, transition


def play_pc_mo_step_fn(
        env_state,
        policy_params,
        preference,
        random_key,
        policy_network,
        env,
    ):
        """
        Step function for preference-conditioned agent.
        Play an environment step and return the updated state and the transition
        """

        actions = policy_network.apply(policy_params, jnp.concatenate([env_state.obs, preference], axis=-1))
       
        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, actions)

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            actions=actions,
            truncations=next_state.info["truncation"],
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
            preference=jnp.zeros(next_state.reward.shape) * jnp.nan,
            input_preference=jnp.zeros(next_state.reward.shape) * jnp.nan,
        )

        return next_state, policy_params, random_key, transition

def play_step_fn(
        env_state,
        policy_params,
        random_key,
        policy_network,
        env,
    ):
        """
        Play an environment step and return the updated state and the transition.
        """

        actions = policy_network.apply(policy_params, env_state.obs)
        
        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, actions)

        reward = jnp.expand_dims(next_state.reward, axis=-1)

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=reward,
            dones=next_state.done,
            actions=actions,
            truncations=next_state.info["truncation"],
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
        )

        return next_state, policy_params, random_key, transition