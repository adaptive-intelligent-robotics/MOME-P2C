"""Core components of a basic genetic algorithm."""
from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.core.containers.mome_repertoire import MOMERepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import ExtraScores, Fitness, Genotype, Metrics, RNGKey, Centroid


class GeneticAlgorithm:
    """Core class of a genetic algorithm.

    This class implements default methods to run a simple
    genetic algorithm with a simple repertoire.

    Args:
        scoring_function: a function that takes a batch of genotypes and compute
            their fitnesses
        emitter: an emitter is used to suggest offsprings given a repertoire. It has
            two compulsory functions. A function that takes emits a new population, and
            a function that update the internal state of the emitter
        metrics_function: a function that takes a repertoire and compute any useful
            metric to track its evolution
    """

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, ExtraScores, RNGKey]
        ],
        emitter: Emitter,
        moqd_metrics_function: Callable[[MOMERepertoire], Metrics],
        ga_metrics_function: Callable[[GARepertoire], Metrics],

    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._moqd_metrics_function = moqd_metrics_function
        self._ga_metrics_function = ga_metrics_function

    @partial(jax.jit, static_argnames=("self", "population_size"))
    def init(
        self, 
        init_genotypes: Genotype, 
        population_size: int, 
        centroids: Centroid,
        pareto_front_max_length: int,
        random_key: RNGKey,
        num_objective_functions: int=2,
        epsilon: float=1e-8,
    ) -> Tuple[GARepertoire, Optional[MOMERepertoire], Optional[EmitterState], RNGKey]:
        """Initialize a GARepertoire with an initial population of genotypes.

        Args:
            init_genotypes: the initial population of genotypes
            population_size: the maximal size of the repertoire
            random_key: a random key to handle stochastic operations

        Returns:
            The initial repertoire, an initial emitter state and a new random key.
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

        # init the repertoire
        repertoire = GARepertoire.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            preferences=preferences,
            population_size=population_size,
        )

        # init the passive MOQD repertoire for comparison
        moqd_passive_repertoire, container_addition_metrics = MOMERepertoire.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
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
            fitnesses=fitnesses,
            descriptors=None,
            extra_scores=extra_scores,
        )

        moqd_metrics = self._moqd_metrics_function(self.moqd_passive_repertoire)
        moqd_metrics = self._emitter.update_added_counts(container_addition_metrics, moqd_metrics)
        ga_metrics = self._ga_metrics_function(repertoire)
        
        # Store running reward statistics
        num_rewards = running_reward_mean.shape[0]
        for m in range(num_rewards):
            moqd_metrics[f"running_reward_mean_{m+1}"] = running_reward_mean[m]
            moqd_metrics[f"running_reward_var_{m+1}"] = running_reward_var[m]
        moqd_metrics["running_reward_count"] = running_reward_count

        metrics  = {**moqd_metrics,  **ga_metrics}
        
        running_stats = (running_reward_mean, running_reward_var, running_reward_count)

        return repertoire, moqd_passive_repertoire, emitter_state, metrics, running_stats, random_key

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        repertoire: GARepertoire,
        moqd_passive_repertoire: Optional[MOMERepertoire],
        emitter_state: Optional[EmitterState],
        running_stats: Tuple[jnp.ndarray, jnp.ndarray, int],
        random_key: RNGKey,
    ) -> Tuple[GARepertoire, Optional[MOMERepertoire], Optional[EmitterState], Metrics, RNGKey]:
        """
        Performs one iteration of a Genetic algorithm.
        1. A batch of genotypes is sampled in the repertoire and the genotypes
            are copied.
        2. The copies are mutated and crossed-over
        3. The obtained offsprings are scored and then added to the repertoire.

        Args:
            repertoire: a repertoire
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
        
        # update the repertoire
        repertoire = repertoire.add(genotypes, fitnesses, descriptors, preferences)

        # update the passive repertoire
        moqd_passive_repertoire = moqd_passive_repertoire.empty()
        moqd_passive_repertoire, container_addition_metrics = moqd_passive_repertoire.add(
            repertoire.genotypes,
            repertoire.descriptors,
            repertoire.fitnesses,
            repertoire.preferences,
        )

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # update the metrics
        moqd_metrics = self._moqd_metrics_function(moqd_passive_repertoire)
        moqd_metrics = self._emitter.update_added_counts(container_addition_metrics, moqd_metrics)
        ga_metrics = self._ga_metrics_function(repertoire)
        
        # Store running reward statistics
        num_rewards = running_reward_mean.shape[0]
        for m in range(num_rewards):
            moqd_metrics[f"running_reward_mean_{m+1}"] = running_reward_mean[m]
            moqd_metrics[f"running_reward_var_{m+1}"] = running_reward_var[m]
        moqd_metrics["running_reward_count"] = running_reward_count
        
        metrics  = {**moqd_metrics,  **ga_metrics}
        
        running_stats = (running_reward_mean, running_reward_var, running_reward_count)

        return repertoire, moqd_passive_repertoire, emitter_state, metrics, running_stats, random_key

    @partial(jax.jit, static_argnames=("self",))
    def scan_update(
        self,
        carry: Tuple[GARepertoire, Optional[MOMERepertoire], Optional[EmitterState], RNGKey],
        unused: Any,
    ) -> Tuple[Tuple[GARepertoire, Optional[MOMERepertoire], Optional[EmitterState], RNGKey], Metrics]:
        """Rewrites the update function in a way that makes it compatible with the
        jax.lax.scan primitive.

        Args:
            carry: a tuple containing the repertoire, the emitter state and a
                random key.
            unused: unused element, necessary to respect jax.lax.scan API.

        Returns:
            The updated repertoire and emitter state, with a new random key and metrics.
        """
        # iterate over grid
        repertoire, moqd_passive_repertoire, emitter_state, running_stats, random_key = carry

        repertoire, moqd_passive_repertoire, emitter_state, metrics, running_stats, random_key = self.update(
            repertoire, moqd_passive_repertoire, emitter_state, running_stats, random_key
        )

        return (repertoire, moqd_passive_repertoire, emitter_state, running_stats, random_key), metrics
