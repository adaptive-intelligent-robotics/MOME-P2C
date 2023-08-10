import jax.numpy as jnp
import jax

from dataclasses import dataclass
from functools import partial
from qdax.core.containers.mome_repertoire import MOMERepertoire
from qdax.core.emitters.preference_sampling.preference_sampler import (
    PreferenceSamplingState,
    PreferenceSampler
)

from qdax.utils.pareto_front import (
    uniform_preference_sampling,
)

@dataclass
class NaiveSamplingConfig:
    """Configuration for PGAME Algorithm"""
    
    # Env params
    num_objectives: int = 1 

    # Emitter params
    emitter_batch_size: int = 256



class RandomPreferenceSampler(PreferenceSampler):

    """
    Train genotypyes with policy gradient using random preferences.
    """

    def __init__(
        self,
        config: NaiveSamplingConfig,
    ):

        # Model parameters
        self._config = config

    @partial(jax.jit, static_argnames=("self",))
    def sample(
        self,
        repertoire: MOMERepertoire,
        sampling_state: PreferenceSamplingState,
    ):

        random_key = sampling_state.random_key
        random_key, subkey = jax.random.split(random_key)
        
        genotypes, _, _, _ = repertoire.sample_batch(
            random_key = subkey,
            num_samples = self._config.emitter_batch_size
        )

        # Find best weights for sampled genotypes
        weights, _ = uniform_preference_sampling(
            random_key = random_key,
            batch_size = self._config.emitter_batch_size,
            num_objectives = self._config.num_objectives,
        )
    
        sampling_state = sampling_state.replace(
            random_key = random_key
        )

        return genotypes, weights, sampling_state



class OneHotPreferenceSampler(PreferenceSampler):
    """
    Train genotypyes with policy gradient using one-hot random preferences.
    """
              
    def __init__(
        self,
        config: NaiveSamplingConfig,
    ):

        # Model parameters
        self._config = config

    @partial(jax.jit, static_argnames=("self",))
    def sample(
        self,
        repertoire: MOMERepertoire,
        sampling_state: PreferenceSamplingState,
    ):

        random_key = sampling_state.random_key
        random_key, subkey = jax.random.split(random_key)
        
        genotypes, _, _, _ = repertoire.sample_batch(
            random_key = subkey,
            num_samples = self._config.emitter_batch_size
        )

        # Generate random weights for sampled genotypes
        random_weights, _ = uniform_preference_sampling(
            random_key = random_key,
            batch_size = self._config.emitter_batch_size,
            num_objectives = self._config.num_objectives,
        )
        
        # Turn weights into one-hot vectors using argmax
        idx = jnp.argmax(weights, axis=1)
        one_hot = jnp.zeros_like(random_weights)
        weights = one_hot.at[jnp.arange(weights.shape[0]), idx].set(1.0)

        sampling_state = sampling_state.replace(
            random_key = random_key
        )

        return genotypes, weights, sampling_state


class KeepPreferencesSampler(PreferenceSampler):
    """
    Train genotypyes with policy gradient using their existing preferences.
    """
              
    def __init__(
        self,
        config: NaiveSamplingConfig,
    ):

        # Model parameters
        self._config = config

    @partial(jax.jit, static_argnames=("self",))
    def sample(
        self,
        repertoire: MOMERepertoire,
        sampling_state: PreferenceSamplingState,
    ):

        random_key = sampling_state.random_key
        random_key, subkey = jax.random.split(random_key)
        
        genotypes, _, preferences, _ = repertoire.sample_batch(
            random_key = subkey,
            num_samples = self._config.emitter_batch_size
        )

        sampling_state = sampling_state.replace(
            random_key = random_key
        )
        
        return genotypes, preferences, sampling_state