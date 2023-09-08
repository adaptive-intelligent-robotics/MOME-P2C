from __future__ import annotations

from functools import partial
from typing import (
    Any,
    Callable,
    Optional,
    Tuple
)

import jax
import jax.numpy as jnp

from qdax.core.emitters.emitter import Emitter
from qdax.core.containers.mome_repertoire import MOMERepertoire
from qdax.core.containers.biased_sampling_mome_repertoire import BiasedSamplingMOMERepertoire
from qdax.core.emitters.emitter import EmitterState
from qdax.types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey
)


class MOME:
    """Implements Multi-Objectives MAP Elites.

    Note: most functions are inherited from MAPElites. The only function
    that had to be overwritten is the init function as it has to take
    into account the specificities of the the Multi Objective repertoire.
    """

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        emitter: Emitter,
        metrics_function: Callable[[MOMERepertoire], Metrics],
        bias_sampling: bool=False,
        preference_conditioned: bool=False,
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._bias_sampling = bias_sampling
        self._preference_conditioned = preference_conditioned

    @partial(jax.jit, static_argnames=("self", "pareto_front_max_length", "num_objective_functions"))
    def init(
        self,
        init_genotypes: jnp.ndarray,
        centroids: Centroid,
        pareto_front_max_length: int,
        random_key: RNGKey,
        num_objective_functions: int=2,
        epsilon: float=1e-8,
    ) -> Tuple[MOMERepertoire, Optional[EmitterState], RNGKey]:
        """Initialize a MOME grid with an initial population of genotypes. Requires
        the definition of centroids that can be computed with any method such as
        CVT or Euclidean mapping.

        Args:
            init_genotypes: genotypes of the initial population.
            centroids: centroids of the repertoire.
            pareto_front_max_length: maximum size of the pareto front. This is
                necessary to respect jax.jit fixed shape size constraint.
            random_key: a random key to handle stochasticity.

        Returns:
            The initial repertoire and emitter state, and a new random key.
        """
        running_reward_mean = jnp.zeros(num_objective_functions, dtype=jnp.float32)
        running_reward_var = jnp.zeros(num_objective_functions, dtype=jnp.float32)
        running_reward_count = epsilon

        # first score
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
        if self._bias_sampling:
            repertoire, container_addition_metrics = BiasedSamplingMOMERepertoire.init(
                genotypes=init_genotypes,
                fitnesses=fitnesses,
                descriptors=descriptors,
                centroids=centroids,
                preferences=preferences,
                pareto_front_max_length=pareto_front_max_length,
            )

        else:
            repertoire, container_addition_metrics = MOMERepertoire.init(
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
        
        pc_actor_metrics = {}
        
        # Evaluate preference conditioned actor and add samples to replay buffer
        if self._preference_conditioned:
            
            # pg_batch_size = self._emitter.emitters[0].batch_size
            # ga_batch_size = self._emitter.emitters[0].
            
            # pg_genotypes = jax.tree_util.tree_map(
            #     lambda x : x[:pg_batch_size],
            #     init_genotypes
            # )
            
            emitter_state, actor_extra_scores, pc_actor_metrics, random_key = self._emitter.evaluate_preference_conditioned_actor(
                repertoire=repertoire,
                emitter_state=emitter_state,
                running_reward_mean=running_reward_mean,
                running_reward_std=running_reward_var,
                running_reward_count=running_reward_count,
                random_key=random_key,
            )
            
            # Update running statistics
            running_reward_mean = actor_extra_scores["running_reward_mean"]
            running_reward_var = actor_extra_scores["running_reward_var"]
            running_reward_count = actor_extra_scores["running_reward_count"]
                       
            # Do some policy gradient on random weights to initialise buffers
            new_genotypes, random_weights, random_key = self._emitter.init_random_pg(
                emitter_state = emitter_state,
                genotypes = init_genotypes,
                random_key = random_key,
            )
            emitter_state = self._emitter.init_sampler_state_update(
                emitter_state=emitter_state,
                old_fitnesses=fitnesses,
                weights=random_weights,
            )
            
            # score new fitnesses to record delta fitness from weights 
            fitnesses, _, _, extra_scores, random_key = self._scoring_function(
                new_genotypes,
                running_reward_mean,
                running_reward_var,
                running_reward_count,
                random_key
            )
            
            # Update running statistics
            running_reward_mean = extra_scores["running_reward_mean"]
            running_reward_var = extra_scores["running_reward_var"]
            running_reward_count = extra_scores["running_reward_count"]
            
        # update emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )
        
        # update the metrics
        metrics = self._metrics_function(repertoire)
        metrics = self._emitter.update_added_counts(container_addition_metrics, metrics)

        # store empirically observed min and max rewards
        metrics["min_rewards"] = extra_scores["min_rewards"]
        metrics["max_rewards"] = extra_scores["max_rewards"]

        # Store running reward statistics
        num_rewards = running_reward_mean.shape[0]
        for m in range(num_rewards):
            metrics[f"running_reward_mean_{m+1}"] = running_reward_mean[m]
            metrics[f"running_reward_var_{m+1}"] = running_reward_var[m]
        metrics["running_reward_count"] = running_reward_count
        
        metrics = {**metrics, **pc_actor_metrics}
        
        running_stats = (running_reward_mean, running_reward_var, running_reward_count)

        return repertoire, metrics, emitter_state, running_stats, random_key


    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        repertoire: MOMERepertoire,
        emitter_state: Optional[EmitterState],
        running_stats: Tuple[jnp.ndarray, jnp.ndarray, int],
        random_key: RNGKey,
    ) -> Tuple[MOMERepertoire, Optional[EmitterState], Metrics, RNGKey]:
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
        
        # add genotypes in the repertoire
        repertoire, container_addition_metrics = repertoire.add(genotypes, descriptors, fitnesses, preferences)

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )
        # # jax.debug.print("OLD:{}", emitter_state.emitter_states[0].sampling_state.old_fitnesses.data)
        # jax.debug.print("NEW:{}", emitter_state.emitter_states[0].sampling_state.new_fitnesses.data)
        # jax.debug.print("WEIGHTS:{}", emitter_state.emitter_states[0].sampling_state.weights_history.data)
        pc_actor_metrics = {}
            
        # Evaluate preference conditioned actor and add samples to replay buffer
        if self._preference_conditioned:
            emitter_state, actor_extra_scores, pc_actor_metrics, random_key = self._emitter.evaluate_preference_conditioned_actor(
                repertoire=repertoire,
                emitter_state=emitter_state,
                running_reward_mean=running_reward_mean,
                running_reward_std=running_reward_var,
                running_reward_count=running_reward_count,
                random_key=random_key,
            )
            
            running_reward_mean = actor_extra_scores["running_reward_mean"]
            running_reward_var = actor_extra_scores["running_reward_var"]
            running_reward_count = actor_extra_scores["running_reward_count"]

        # update the metrics
        metrics = self._metrics_function(repertoire)
        metrics = self._emitter.update_added_counts(container_addition_metrics, metrics)

        # store empirically observed min and max rewards
        metrics["min_rewards"] = extra_scores["min_rewards"]
        metrics["max_rewards"] = extra_scores["max_rewards"]

        # Store running reward statistics
        num_rewards = running_reward_mean.shape[0]
        for m in range(num_rewards):
            metrics[f"running_reward_mean_{m+1}"] = running_reward_mean[m]
            metrics[f"running_reward_var_{m+1}"] = running_reward_var[m]
        metrics["running_reward_count"] = running_reward_count

        metrics = {**metrics, **pc_actor_metrics}

        running_stats = (running_reward_mean, running_reward_var, running_reward_count)

        return repertoire, emitter_state, metrics, running_stats, random_key

    @partial(jax.jit, static_argnames=("self",))
    def scan_update(
        self,
        carry: Tuple[MOMERepertoire, Optional[EmitterState], RNGKey],
        unused: Any,
    ) -> Tuple[Tuple[MOMERepertoire, Optional[EmitterState], RNGKey], Metrics]:
        """Rewrites the update function in a way that makes it compatible with the
        jax.lax.scan primitive.

        Args:
            carry: a tuple containing the repertoire, the emitter state and a
                random key.
            unused: unused element, necessary to respect jax.lax.scan API.

        Returns:
            The updated repertoire and emitter state, with a new random key and metrics.
        """
        repertoire, emitter_state, running_stats, random_key = carry

        repertoire, emitter_state, metrics, running_stats, random_key = self.update(
            repertoire, emitter_state, running_stats, random_key
        )

        return (repertoire, emitter_state, running_stats, random_key), metrics
