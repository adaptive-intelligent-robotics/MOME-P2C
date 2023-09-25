"""Core components of the MAP-Elites algorithm."""
from __future__ import annotations

from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.containers.mome_repertoire import MOMERepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)


class MAPElites:
    """Core elements of the MAP-Elites algorithm.

    Note: Although very similar to the GeneticAlgorithm, we decided to keep the
    MAPElites class independant of the GeneticAlgorithm class at the moment to keep
    elements explicit.

    Args:
        scoring_function: a function that takes a batch of genotypes and compute
            their fitnesses and descriptors
        emitter: an emitter is used to suggest offsprings given a MAPELites
            repertoire. It has two compulsory functions. A function that takes
            emits a new population, and a function that update the internal state
            of the emitter.
        metrics_function: a function that takes a MAP-Elites repertoire and compute
            any useful metric to track its evolution
    """

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        emitter: Emitter,
        metrics_function: Callable[[MapElitesRepertoire], Metrics],
        moqd_metrics_function: Callable[[MOMERepertoire], Metrics],
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._moqd_metrics_function = moqd_metrics_function

    def init(
        self,
        init_genotypes: Genotype,
        centroids: Centroid,
        moqd_passive_centroids: Centroid,
        pareto_front_max_length: int,
        random_key: RNGKey,
        num_objective_functions: int=2,
        epsilon: float=1e-8,
    ) -> Tuple[MapElitesRepertoire, Optional[MOMERepertoire], Optional[EmitterState], RNGKey]:
        """
        Initialize a Map-Elites repertoire with an initial population of genotypes.
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Args:
            init_genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            centroids: tesselation centroids of shape (batch_size, num_descriptors)
            random_key: a random key used for stochastic operations.

        Returns:
            An initialized MAP-Elite repertoire with the initial state of the emitter,
            and a random key.
        """
        running_reward_mean = jnp.zeros(num_objective_functions, dtype=jnp.float32)
        running_reward_var = jnp.zeros(num_objective_functions, dtype=jnp.float32)
        running_reward_count = epsilon

        # score initial genotypes
        fitnesses, descriptors, preferences, extra_scores, random_key = self._scoring_function(
            init_genotypes,
            running_reward_mean,
            running_reward_var,
            running_reward_count,
            random_key
        )

        # Update running statistics
        running_reward_mean = extra_scores["running_reward_mean"]
        running_reward_var = extra_scores["running_reward_var"]
        running_reward_count = extra_scores["running_reward_count"]
        

        mono_objective_fitnesses = jnp.sum(fitnesses, axis=-1)

        # init the repertoire
        repertoire = MapElitesRepertoire.init(
            genotypes=init_genotypes,
            fitnesses=mono_objective_fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            mo_fitnesses = fitnesses,
            preferences=preferences,
        )


        # then readd the passive archive
        moqd_passive_repertoire, container_addition_metrics = MOMERepertoire.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=moqd_passive_centroids,
            preferences=preferences,
            pareto_front_max_length=pareto_front_max_length,
        )

        # get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            init_genotypes=init_genotypes, random_key=random_key
        )

        # update emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=init_genotypes,
            fitnesses=mono_objective_fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        metrics = self._metrics_function(repertoire)
        metrics = self._emitter.update_added_counts(container_addition_metrics, metrics)

        moqd_metrics = self._moqd_metrics_function(moqd_passive_repertoire)
        moqd_metrics = self._emitter.update_added_counts(container_addition_metrics, moqd_metrics)

        # store empirically observed min and max rewards
        moqd_metrics["min_rewards"] = extra_scores["min_rewards"]
        moqd_metrics["max_rewards"] = extra_scores["max_rewards"]
        
        # Store running reward statistics
        num_rewards = running_reward_mean.shape[0]
        for m in range(num_rewards):
            moqd_metrics[f"running_reward_mean_{m+1}"] = running_reward_mean[m]
            moqd_metrics[f"running_reward_var_{m+1}"] = running_reward_var[m]
        moqd_metrics["running_reward_count"] = running_reward_count
        
        metrics  = {**moqd_metrics,  **metrics}

        running_stats = (running_reward_mean, running_reward_var, running_reward_count)

        return repertoire, moqd_passive_repertoire, emitter_state, metrics, running_stats, random_key

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        repertoire: MapElitesRepertoire,
        moqd_passive_repertoire: Optional[MOMERepertoire],
        emitter_state: Optional[EmitterState],
        running_stats: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        random_key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, Optional[MOMERepertoire], Optional[EmitterState], Metrics, RNGKey]:
        """
        Performs one iteration of the MAP-Elites algorithm.
        1. A batch of genotypes is sampled in the repertoire and the genotypes
            are copied.
        2. The copies are mutated and crossed-over
        3. The obtained offsprings are scored and then added to the repertoire.


        Args:
            repertoire: the MAP-Elites repertoire
            emitter_state: state of the emitter
            random_key: a jax PRNG random key

        Returns:
            the updated MAP-Elites repertoire
            the updated (if needed) emitter state
            metrics about the updated repertoire
            a new jax PRNG key
        """
        running_reward_mean, running_reward_var, running_reward_count = running_stats

        # generate offsprings with the emitter
        genotypes, emitter_state, random_key = self._emitter.emit(
            repertoire, emitter_state, random_key
        )
        
        # scores the offsprings
        fitnesses, descriptors, preferences, extra_scores, random_key = self._scoring_function(
            genotypes,
            running_reward_mean,
            running_reward_var,
            running_reward_count,
            random_key
        )

        # Update running statistics
        running_reward_mean = extra_scores["running_reward_mean"]
        running_reward_var = extra_scores["running_reward_var"]
        running_reward_count = extra_scores["running_reward_count"]

        mono_objective_fitnesses = jnp.sum(fitnesses, axis=-1)

        # add genotypes in the repertoire
        repertoire = repertoire.add(
            genotypes,
            descriptors,
            mono_objective_fitnesses,
            fitnesses,
            preferences)
        
        # first empty the passive repertoire
        moqd_passive_repertoire = moqd_passive_repertoire.empty()
        moqd_passive_repertoire, container_addition_metrics = moqd_passive_repertoire.add(
            repertoire.genotypes,
            repertoire.descriptors,
            repertoire.mo_fitnesses,
            repertoire.preferences,
        )

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=mono_objective_fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # update the metrics
        metrics = self._metrics_function(repertoire)
        metrics = self._emitter.update_added_counts(container_addition_metrics, metrics)

        moqd_metrics = self._moqd_metrics_function(moqd_passive_repertoire)
        moqd_metrics = self._emitter.update_added_counts(container_addition_metrics, moqd_metrics)
  
        # store empirically observed min and max rewards
        moqd_metrics["min_rewards"] = extra_scores["min_rewards"]
        moqd_metrics["max_rewards"] = extra_scores["max_rewards"]
              
        # Store running reward statistics
        num_rewards = running_reward_mean.shape[0]
        for m in range(num_rewards):
            moqd_metrics[f"running_reward_mean_{m+1}"] = running_reward_mean[m]
            moqd_metrics[f"running_reward_var_{m+1}"] = running_reward_var[m]
        moqd_metrics["running_reward_count"] = running_reward_count
        
        metrics  = {**moqd_metrics,  **metrics}
        
        running_stats = (running_reward_mean, running_reward_var, running_reward_count)

        return repertoire, moqd_passive_repertoire, emitter_state, metrics, running_stats, random_key

    @partial(jax.jit, static_argnames=("self",))
    def scan_update(
        self,
        carry: Tuple[MapElitesRepertoire, Optional[MOMERepertoire], Optional[EmitterState], RNGKey],
        unused: Any,
    ) -> Tuple[Tuple[MapElitesRepertoire, Optional[MOMERepertoire],  Optional[EmitterState], RNGKey], Metrics]:
        """Rewrites the update function in a way that makes it compatible with the
        jax.lax.scan primitive.

        Args:
            carry: a tuple containing the repertoire, the emitter state and a
                random key.
            unused: unused element, necessary to respect jax.lax.scan API.

        Returns:
            The updated repertoire and emitter state, with a new random key and metrics.
        """
        repertoire, moqd_passive_repertoire, emitter_state, running_stats, random_key = carry

        repertoire, moqd_passive_repertoire, emitter_state, metrics, running_stats, random_key = self.update(
            repertoire, moqd_passive_repertoire, emitter_state, running_stats, random_key
        )

        return (repertoire, moqd_passive_repertoire, emitter_state, running_stats, random_key), metrics