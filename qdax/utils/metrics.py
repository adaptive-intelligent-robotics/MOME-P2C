"""Defines functions to retrieve metrics from training processes."""

from __future__ import annotations

import csv
import time
from functools import partial
from typing import Dict, List

import jax
from jax import numpy as jnp

from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.containers.mome_repertoire import MOMERepertoire
from qdax.types import Metrics, Preference
from qdax.utils.pareto_front import compute_hypervolume


class CSVLogger:
    """Logger to save metrics of an experiment in a csv file
    during the training process.
    """

    def __init__(self, filename: str, header: List) -> None:
        """Create the csv logger, create a file and write the
        header.

        Args:
            filename: path to which the file will be saved.
            header: header of the csv file.
        """
        self._filename = filename
        self._header = header
        with open(self._filename, "w") as file:
            writer = csv.DictWriter(file, fieldnames=self._header)
            # write the header
            writer.writeheader()

    def log(self, metrics: Dict[str, float]) -> None:
        """Log new metrics to the csv file.

        Args:
            metrics: A dictionary containing the metrics that
                need to be saved.
        """
        with open(self._filename, "a") as file:
            writer = csv.DictWriter(file, fieldnames=self._header)
            # write new metrics in a raw
            writer.writerow(metrics)


def default_ga_metrics(
    repertoire: GARepertoire,
) -> Metrics:
    """Compute the usual GA metrics that one can retrieve
    from a GA repertoire.

    Args:
        repertoire: a GA repertoire

    Returns:
        a dictionary containing the max fitness of the
            repertoire.
    """

    # get metrics
    max_fitness = jnp.max(repertoire.fitnesses, axis=0)
    population_size = repertoire.size

    return {
        "ga_max_fitness": max_fitness,
        "ga_population_size": population_size,
    }


def default_qd_metrics(repertoire: MapElitesRepertoire, qd_offset: float) -> Metrics:
    """Compute the usual QD metrics that one can retrieve
    from a MAP Elites repertoire.

    Args:
        repertoire: a MAP-Elites repertoire
        qd_offset: an offset used to ensure that the QD score
            will be positive and increasing with the number
            of individuals.

    Returns:
        a dictionary containing the QD score (sum of fitnesses
            modified to be all positive), the max fitness of the
            repertoire, the coverage (number of niche filled in
            the repertoire).
    """

    # get metrics
    repertoire_empty = repertoire.fitnesses == -jnp.inf
    qd_score = jnp.sum(repertoire.fitnesses, where=~repertoire_empty)
    qd_score += qd_offset * jnp.sum(1.0 - repertoire_empty)
    coverage = 100 * jnp.mean(1.0 - repertoire_empty)
    max_fitness = jnp.max(repertoire.fitnesses)

    return {"qd_score": qd_score, "max_fitness": max_fitness, "map_elites_coverage": coverage}


def default_moqd_metrics(
    repertoire: MOMERepertoire, 
    reference_point: jnp.ndarray,
) -> Metrics:
    """Compute the MOQD metric given a MOME repertoire and a reference point.

    Args:
        repertoire: a MOME repertoire.
        reference_point: the hypervolume of a pareto front has to be computed
            relatively to a reference point.

    Returns:
        A dictionary containing all the computed metrics.
    """
    # Calculating coverage
    repertoire_empty = repertoire.fitnesses == -jnp.inf # num centroids x pareto-front length x num criteria
    min_scores = jnp.min(~repertoire_empty * repertoire.fitnesses, axis=(0, 1))
    repertoire_empty = jnp.all(repertoire_empty, axis=-1) # num centroids x pareto-front length
    repertoire_not_empty = ~repertoire_empty # num centroids x pareto-front length
    num_solutions = jnp.sum(repertoire_not_empty, axis=-1)
    total_num_solutions = jnp.sum(num_solutions)
    repertoire_not_empty = jnp.any(repertoire_not_empty, axis=-1) # num centroids
    coverage = 100 * jnp.mean(repertoire_not_empty)

    #### Calculate unnormalised hypervolume metrics
    hypervolume_function = partial(compute_hypervolume, reference_point=reference_point)
    hypervolumes = jax.vmap(hypervolume_function)(repertoire.fitnesses)  # num centroids
    hypervolumes = jnp.where(repertoire_not_empty, hypervolumes, -jnp.inf)
    moqd_score = jnp.sum(repertoire_not_empty * hypervolumes)
    max_hypervolume = jnp.max(hypervolumes)

    max_scores = jnp.max(repertoire.fitnesses, axis=(0, 1))
    max_sum_scores = jnp.max(jnp.sum(repertoire.fitnesses, axis=-1), axis=(0, 1))

    (
        pareto_front,
        _,
    ) = repertoire.compute_global_pareto_front()

    global_hypervolume = compute_hypervolume(
        pareto_front, reference_point=reference_point
    )

    metrics = {
        "hypervolumes": hypervolumes,
        "moqd_score": moqd_score,
        "max_hypervolume": max_hypervolume,
        "max_scores": max_scores,
        "min_scores": min_scores,
        "max_sum_scores": max_sum_scores,
        "coverage": coverage,
        "num_solutions": num_solutions,
        "total_num_solutions": total_num_solutions,
        "global_hypervolume": global_hypervolume,
    }

    return metrics


def pc_actor_metrics(
    achieved_preferences: Preference,
    sampled_preferences: Preference,
)-> Metrics:
    
    
    errors = jnp.abs(achieved_preferences - sampled_preferences)
        
    # Calculate total error
    pc_actor_total_error = jnp.mean((
        jnp.sum(errors, axis=1)
    ))
    
    # Calculate whether there is correlation between error and preference sampled
    f1_errors = errors[:,0]
    f1_preferences = sampled_preferences[:,0]
    f1_errors_correlation = jnp.corrcoef(f1_errors, f1_preferences)[0,1]

    pc_actor_metrics = {
        "pc_actor_total_error": pc_actor_total_error,
        "pc_actor_f1_errors_correlation": f1_errors_correlation,
    }
    
    
    return pc_actor_metrics