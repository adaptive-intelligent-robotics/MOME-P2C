import jax
import jax.numpy as jnp

from abc import ABC, abstractmethod
from flax.struct import PyTreeNode
from functools import partial
from qdax.core.containers.repertoire import Repertoire
from qdax.types import (
    Descriptor, 
    ExtraScores, 
    Fitness, 
    Genotype, 
    Preference,
    RNGKey
)
from typing import Optional, Tuple


class PreferenceSamplingState(PyTreeNode):
    """The state of an preference sampler."""

    random_key: RNGKey


class PreferenceSampler(ABC):
    def init(
        self,
        init_genotypes: Genotype,
        random_key: RNGKey
    ) -> Tuple[PreferenceSamplingState, RNGKey]:
        
        """
        Initialise the state of the sampler."""

        # Initialise random key
        random_key, subkey = jax.random.split(random_key)

        return PreferenceSamplingState(random_key = random_key), subkey

    @abstractmethod
    def sample(
        self,
        repertoire: Optional[Repertoire],
        sampling_state: Optional[PreferenceSamplingState],
    ) -> Tuple[Genotype, Preference, PreferenceSamplingState]:
        """
        Args:
            repertoire: a repertoire of genotypes.
            sampling_state: the state of the sampler.
            random_key: a random key to handle random operations.
        Returns:
            A batch of offspring, the corresponding preference to 
            train on and a new random key.
        """
        pass


    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def init_state_update(
        self,
        sampling_state: Optional[PreferenceSamplingState],
        batch_init_fitnesses: Optional[Fitness],
        batch_init_preferences: Optional[Preference]
    ) -> Optional[PreferenceSamplingState]:
        """        
        Returns:
            The modified sampling state.
        """
        return sampling_state
    

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def state_update(
        self,
        sampling_state: Optional[PreferenceSamplingState],
        batch_new_fitnesses: Optional[Fitness],
    ) -> Optional[PreferenceSamplingState]:
        """        
        Returns:
            The modified sampling state.
        """
        return sampling_state

