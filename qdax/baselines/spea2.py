"""Core components of the SPEA2 algorithm.

Link to paper: "https://www.semanticscholar.org/paper/SPEA2%3A
-Improving-the-strength-pareto-evolutionary-Zitzler-Laumanns/
b13724cb54ae4171916f3f969d304b9e9752a57f"
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.baselines.genetic_algorithm import GeneticAlgorithm
from qdax.core.containers.spea2_repertoire import SPEA2Repertoire
from qdax.core.containers.mome_repertoire import MOMERepertoire
from qdax.core.emitters.emitter import EmitterState
from qdax.types import Centroid, Genotype, RNGKey


class SPEA2(GeneticAlgorithm):
    """Implements main functions of the SPEA2 algorithm.

    This class inherits most functions from GeneticAlgorithm.
    The init function is overwritten in order to precise the type
    of repertoire used in SPEA2.

    Link to paper: "https://www.semanticscholar.org/paper/SPEA2%3A-
    Improving-the-strength-pareto-evolutionary-Zitzler-Laumanns/
    b13724cb54ae4171916f3f969d304b9e9752a57f"
    """

    @partial(
        jax.jit,
        static_argnames=(
            "self",
            "population_size",
            "num_neighbours",
            "pareto_front_max_length"
        ),
    )
    def init(
        self,
        init_genotypes: Genotype,
        population_size: int,
        num_neighbours: int,
        random_key: RNGKey,
        centroids: Centroid,
        pareto_front_max_length: int,
        num_objective_functions: int=2,
        epsilon: float=1e-8
    ) -> Tuple[SPEA2Repertoire, Optional[MOMERepertoire], Optional[EmitterState], RNGKey]:

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
        repertoire = SPEA2Repertoire.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            population_size=population_size,
            num_neighbours=num_neighbours,
            descriptors=descriptors,
            preferences=preferences,
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
            extra_scores=extra_scores,
        )

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
