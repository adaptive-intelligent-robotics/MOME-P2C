"""This file contains the class to define the repertoire used to
store individuals in the Multi-Objective MAP-Elites algorithm as
well as several variants."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Tuple, List

import flax
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from qdax.core.containers.mapelites_repertoire import (
    get_cells_indices,
)
from qdax.types import (
    Centroid,
    Descriptor,
    Fitness,
    Genotype,
    Mask,
    Preference,
    ParetoFront,
    RNGKey,
)
from qdax.utils.pareto_front import compute_masked_pareto_front


class MOMERepertoire(flax.struct.PyTreeNode):
    """Class for the repertoire in Multi Objective Map Elites

    This class inherits from MAPElitesRepertoire. The stored data
    is the same: genotypes, fitnesses, descriptors, centroids.

    The shape of genotypes is (in the case where it's an array):
    (num_centroids, pareto_front_length, genotype_dim).
    When the genotypes is a PyTree, the two first dimensions are the same
    but the third will depend on the leafs.

    The shape of fitnesses and preferences are: 
    (num_centroids, pareto_front_length, num_criteria)

    The shape of descriptors and centroids are:
    (num_centroids, num_descriptors, pareto_front_length).

    Inherited functions: save and load.
    """
    genotypes: Genotype
    fitnesses: Fitness
    descriptors: Descriptor
    centroids: Centroid
    preferences: Preference

    def save(self, path: str = "./") -> None:
        """Saves the repertoire on disk in the form of .npy files.

        Flattens the genotypes to store it with .npy format. Supposes that
        a user will have access to the reconstruction function when loading
        the genotypes.

        Args:
            path: Path where the data will be saved. Defaults to "./".
        """

        def flatten_genotype(genotype: Genotype) -> jnp.ndarray:
            flatten_genotype, _ = ravel_pytree(genotype)
            return flatten_genotype

        # flatten all the genotypes
        flat_genotypes = jax.vmap(flatten_genotype)(self.genotypes)

        # save data
        jnp.save(path + "genotypes.npy", flat_genotypes)
        jnp.save(path + "fitnesses.npy", self.fitnesses)
        jnp.save(path + "descriptors.npy", self.descriptors)
        jnp.save(path + "centroids.npy", self.centroids)
        jnp.save(path + "preferences.npy", self.preferences)

    @classmethod
    def load(cls, reconstruction_fn: Callable, path: str = "./") -> MOMERepertoire:
        """Loads a MAP Elites Repertoire.

        Args:
            reconstruction_fn: Function to reconstruct a PyTree
                from a flat array.
            path: Path where the data is saved. Defaults to "./".

        Returns:
            A MAP Elites Repertoire.
        """

        flat_genotypes = jnp.load(path + "genotypes.npy")
        genotypes = jax.vmap(reconstruction_fn)(flat_genotypes)

        fitnesses = jnp.load(path + "fitnesses.npy")
        descriptors = jnp.load(path + "descriptors.npy")
        centroids = jnp.load(path + "centroids.npy")
        preferences = jnp.load(path + "preferences.npy")

        return cls(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            preferences=preferences,
        )

    @property
    def repertoire_capacity(self) -> int:
        """Returns the maximum number of solutions the repertoire can
        contain which corresponds to the number of cells times the
        maximum pareto front length.

        Returns:
            The repertoire capacity.
        """
        first_leaf = jax.tree_util.tree_leaves(self.genotypes)[0]
        return int(first_leaf.shape[0] * first_leaf.shape[1])

    @jax.jit
    def _sample_in_masked_pareto_front(
        self,
        pareto_front_genotypes: ParetoFront[Genotype],
        pareto_front_preferences: ParetoFront[Preference],
        mask: Mask,
        random_key: RNGKey,
    ) -> Genotype:
        """Sample one single genotype and its associated preference in masked pareto front.

        Note: do not retrieve a random key because this function
        is to be vmapped. The public method that uses this function
        will return a random key

        Args:
            pareto_front_genotypes: the genotypes of a pareto front
            pareto_front_preferences: the preferences of a pareto front
            mask: a mask associated to the front
            random_key: a random key to handle stochastic operations

        Returns:
            A single genotype among the pareto front.
        """
        p = (1.0 - mask) / jnp.sum(1.0 - mask)

        genotype_sample = jax.tree_util.tree_map(
            lambda x: jax.random.choice(random_key, x, shape=(1,), p=p),
            pareto_front_genotypes,
        )
    
        # Use same random key to ensure preferences correspond to the sampled genotype
        preference_sample = jax.tree_util.tree_map(
            lambda x: jax.random.choice(random_key, x, shape=(1,), p=p),
            pareto_front_preferences,
        )

        return genotype_sample, preference_sample

    @partial(jax.jit, static_argnames=("num_samples",))
    def sample(self, random_key: RNGKey, num_samples: int) -> Tuple[Genotype, RNGKey]:
        """Sample elements in the repertoire.

        This method sample a non-empty pareto front, and then sample
        genotypes from this pareto front.

        Args:
            random_key: a random key to handle stochasticity.
            num_samples: number of samples to retrieve from the repertoire.

        Returns:
            A sample of genotypes and a new random key.
        """

        # create sampling probability for the cells
        repertoire_empty = jnp.any(self.fitnesses == -jnp.inf, axis=-1)
        occupied_cells = jnp.any(~repertoire_empty, axis=-1)

        p = occupied_cells / jnp.sum(occupied_cells)

        # possible indices - num cells
        indices = jnp.arange(start=0, stop=repertoire_empty.shape[0])

        # choose idx - among indices of cells that are not empty
        random_key, subkey = jax.random.split(random_key)
        cells_idx = jax.random.choice(subkey, indices, shape=(num_samples,), p=p)

        # get genotypes (front) from the chosen indices
        pareto_front_genotypes = jax.tree_util.tree_map(
            lambda x: x[cells_idx], self.genotypes
        )

        pareto_front_preferences = jax.tree_util.tree_map(
            lambda x: x[cells_idx], self.preferences
        )

        # prepare second sampling function
        sample_in_fronts = jax.vmap(self._sample_in_masked_pareto_front)

        # sample genotypes from the pareto front
        random_key, subkey = jax.random.split(random_key)
        subkeys = jax.random.split(subkey, num=num_samples)
        sampled_genotypes, _ = sample_in_fronts(  # type: ignore
            pareto_front_genotypes=pareto_front_genotypes,
            pareto_front_preferences=pareto_front_preferences,
            mask=repertoire_empty[cells_idx],
            random_key=subkeys,
        )

        # remove the dim coming from pareto front
        sampled_genotypes = jax.tree_util.tree_map(
            lambda x: x.squeeze(axis=1), sampled_genotypes
        )

        return sampled_genotypes, random_key

    @partial(jax.jit, static_argnames=("num_samples",))
    def sample_parents_and_preferences(self, random_key: RNGKey, num_samples: int) -> Tuple[Genotype, RNGKey]:
        """Sample elements and associated preferences in the repertoire.

        This method sample a non-empty pareto front, and then sample
        genotypes from this pareto front.

        Args:
            random_key: a random key to handle stochasticity.
            num_samples: number of samples to retrieve from the repertoire.

        Returns:
            A sample of genotypes and a new random key.
        """

        # create sampling probability for the cells
        repertoire_empty = jnp.any(self.fitnesses == -jnp.inf, axis=-1)
        occupied_cells = jnp.any(~repertoire_empty, axis=-1)

        p = occupied_cells / jnp.sum(occupied_cells)

        # possible indices - num cells
        indices = jnp.arange(start=0, stop=repertoire_empty.shape[0])

        # choose idx - among indices of cells that are not empty
        random_key, subkey = jax.random.split(random_key)
        cells_idx = jax.random.choice(subkey, indices, shape=(num_samples,), p=p)

        # get genotypes (front) from the chosen indices
        pareto_front_genotypes = jax.tree_util.tree_map(
            lambda x: x[cells_idx], self.genotypes
        )

        # get preferences (front) from the same chosen indices
        pareto_front_preferences = jax.tree_util.tree_map(
            lambda x: x[cells_idx], self.preferences
        )

        # prepare second sampling function
        sample_in_fronts = jax.vmap(self._sample_in_masked_pareto_front)

        # sample genotypes from the pareto front
        random_key, subkey = jax.random.split(random_key)
        subkeys = jax.random.split(subkey, num=num_samples)
        sampled_genotypes, sampled_preferences = sample_in_fronts(  # type: ignore
            pareto_front_genotypes=pareto_front_genotypes,
            pareto_front_preferences=pareto_front_preferences,
            mask=repertoire_empty[cells_idx],
            random_key=subkeys,
        )

        # remove the dim coming from pareto front
        sampled_genotypes = jax.tree_util.tree_map(
            lambda x: x.squeeze(axis=1), sampled_genotypes
        )
        sampled_preferences = jax.tree_util.tree_map(
            lambda x: x.squeeze(axis=1), sampled_preferences
        )
        
        return sampled_genotypes, sampled_preferences, random_key

    @jax.jit
    def _update_masked_pareto_front(
        self,
        pareto_front_fitnesses: ParetoFront[Fitness],
        pareto_front_genotypes: ParetoFront[Genotype],
        pareto_front_descriptors: ParetoFront[Descriptor],
        pareto_front_preferences: ParetoFront[Preference],
        mask: Mask,
        new_batch_of_fitnesses: Fitness,
        new_batch_of_genotypes: Genotype,
        new_batch_of_descriptors: Descriptor,
        new_batch_of_preferences: Preference,
        new_mask: Mask,
    ) -> Tuple[
        ParetoFront[Fitness], ParetoFront[Genotype], ParetoFront[Descriptor], Mask
    ]:
        """Takes a fixed size pareto front, its mask and new points to add.
        Returns updated front and mask.

        Args:
            pareto_front_fitnesses: fitness of the pareto front
            pareto_front_genotypes: corresponding genotypes
            pareto_front_descriptors: corresponding descriptors
            pareto_front_preferences: corresponding preferences
            mask: mask of the front, to hide void parts
            new_batch_of_fitnesses: new batch of fitness that is considered
                to be added to the pareto front
            new_batch_of_genotypes: corresponding genotypes
            new_batch_of_descriptors: corresponding descriptors
            new_batch_of_preferences: corresponding preferences
            new_mask: corresponding mask (no one is masked)

        Returns:
            The updated pareto front.
        """
        # get dimensions
        batch_size = new_batch_of_fitnesses.shape[0]
        num_criteria = new_batch_of_fitnesses.shape[1]

        pareto_front_len = pareto_front_fitnesses.shape[0]  # type: ignore
        descriptors_dim = new_batch_of_descriptors.shape[1]

        # gather all data
        cat_mask = jnp.concatenate([mask, new_mask], axis=-1)
        cat_fitnesses = jnp.concatenate(
            [pareto_front_fitnesses, new_batch_of_fitnesses], axis=0
        )
        cat_genotypes = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            pareto_front_genotypes,
            new_batch_of_genotypes,
        )
        cat_descriptors = jnp.concatenate(
            [pareto_front_descriptors, new_batch_of_descriptors], axis=0
        )

        cat_preferences = jnp.concatenate(
            [pareto_front_preferences, new_batch_of_preferences], axis=0
        )

        # get new front
        cat_bool_front = compute_masked_pareto_front(
            batch_of_criteria=cat_fitnesses, mask=cat_mask
        )

        # Gather container addition metrics
        added_bool = jnp.take(cat_bool_front, -1) # Count if new genotype is now on cell PF
        removed_count = jnp.sum(~mask) - jnp.sum(cat_bool_front[:-1]) # Count how many 

        # get corresponding indicesß
        indices = (
            jnp.arange(start=0, stop=pareto_front_len + batch_size) * cat_bool_front
        )
        indices = indices + ~cat_bool_front * (batch_size + pareto_front_len - 1)
        indices = jnp.sort(indices)

        # get new fitness, genotypes and descriptors
        new_front_fitness = jnp.take(cat_fitnesses, indices, axis=0)
        new_front_genotypes = jax.tree_util.tree_map(
            lambda x: jnp.take(x, indices, axis=0), cat_genotypes
        )
        new_front_descriptors = jnp.take(cat_descriptors, indices, axis=0)
        new_front_preferences = jnp.take(cat_preferences, indices, axis=0)

        # compute new mask
        num_front_elements = jnp.sum(cat_bool_front)
        new_mask_indices = jnp.arange(start=0, stop=batch_size + pareto_front_len)
        new_mask_indices = (num_front_elements - new_mask_indices) > 0

        new_mask = jnp.where(
            new_mask_indices,
            jnp.ones(shape=batch_size + pareto_front_len, dtype=bool),
            jnp.zeros(shape=batch_size + pareto_front_len, dtype=bool),
        )

        fitness_mask = jnp.repeat(
            jnp.expand_dims(new_mask, axis=-1), num_criteria, axis=-1
        )
        new_front_fitness = new_front_fitness * fitness_mask

        front_size = len(pareto_front_fitnesses)  # type: ignore
        new_front_fitness = new_front_fitness[:front_size, :]

        new_front_preferences = new_front_preferences * fitness_mask
        new_front_preferences = new_front_preferences[:front_size, :]

        new_front_genotypes = jax.tree_util.tree_map(
            lambda x: x * new_mask_indices[0], new_front_genotypes
        )
        new_front_genotypes = jax.tree_util.tree_map(
            lambda x: x[:front_size], new_front_genotypes
        )

        descriptors_mask = jnp.repeat(
            jnp.expand_dims(new_mask, axis=-1), descriptors_dim, axis=-1
        )
        new_front_descriptors = new_front_descriptors * descriptors_mask
        new_front_descriptors = new_front_descriptors[:front_size, :]

        new_mask = ~new_mask[:front_size]

        return new_front_fitness, new_front_genotypes, new_front_descriptors, new_front_preferences, new_mask, added_bool, removed_count

    @jax.jit
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_preferences: Preference,
    ) -> MOMERepertoire:
        """Insert a batch of elements in the repertoire.

        Shape of the batch_of_genotypes (if an array):
        (batch_size, genotypes_dim)
        Shape of the batch_of_descriptors: (batch_size, num_descriptors)
        Shape of the batch_of_fitnesses: (batch_size, num_criteria)
        Shape of the batch_of_preferences: (batch_size, num_criteria)

        Args:
            batch_of_genotypes: a batch of genotypes that we are trying to
                insert into the repertoire.
            batch_of_descriptors: the descriptors of the genotypes we are
                trying to add to the repertoire.
            batch_of_fitnesses: the fitnesses of the genotypes we are trying
                to add to the repertoire.
            batch_of_preferences: the preferences of the genotypes we are
                trying to add to the repertoire.

        Returns:
            The updated repertoire with potential new individuals.
        """
        # get the indices that corresponds to the descriptors in the repertoire
        batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)
        batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1)

        def _add_one(
            carry: MOMERepertoire,
            data: Tuple[Genotype, Descriptor, Fitness, Preference, jnp.ndarray],
        ) -> Tuple[MOMERepertoire, Any]:
            # unwrap data
            genotype, descriptors, fitness, preferences, index = data

            index = index.astype(jnp.int32)

            # get current repertoire cell data
            cell_genotype = jax.tree_util.tree_map(lambda x: x[index][0], carry.genotypes)
            cell_fitness = carry.fitnesses[index][0]
            cell_preferences = carry.preferences[index][0]
            cell_descriptor = carry.descriptors[index][0]
            cell_mask = jnp.any(cell_fitness == -jnp.inf, axis=-1)

            new_genotypes = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), genotype)

            # update pareto front
            (
                cell_fitness,
                cell_genotype, # new pf for cell 
                cell_descriptor,
                cell_preferences,
                cell_mask,
                added_bool,
                removed_count,
            ) = self._update_masked_pareto_front(
                pareto_front_fitnesses=cell_fitness,
                pareto_front_genotypes=cell_genotype,
                pareto_front_descriptors=cell_descriptor,
                pareto_front_preferences=cell_preferences,
                mask=cell_mask,
                new_batch_of_fitnesses=jnp.expand_dims(fitness, axis=0),
                new_batch_of_genotypes=new_genotypes,
                new_batch_of_descriptors=jnp.expand_dims(descriptors, axis=0),
                new_batch_of_preferences=jnp.expand_dims(preferences, axis=0),
                new_mask=jnp.zeros(shape=(1,), dtype=bool),
            )

            # update cell fitness
            cell_fitness = cell_fitness - jnp.inf * jnp.expand_dims(cell_mask, axis=-1)

            # update grid
            new_genotypes = jax.tree_util.tree_map(
                lambda x, y: x.at[index].set(y), carry.genotypes, cell_genotype
            )
            new_fitnesses = carry.fitnesses.at[index].set(cell_fitness)
            new_descriptors = carry.descriptors.at[index].set(cell_descriptor)
            new_preferences = carry.preferences.at[index].set(cell_preferences)
            carry = carry.replace(  # type: ignore
                genotypes=new_genotypes,
                descriptors=new_descriptors,
                fitnesses=new_fitnesses,
                preferences=new_preferences,
            )

            # return new grid
            return carry, [added_bool, removed_count]

        # scan the addition operation for all the data
        self, container_addition_metrics = jax.lax.scan(
            _add_one,
            self,
            (
                batch_of_genotypes,
                batch_of_descriptors,
                batch_of_fitnesses,
                batch_of_preferences,
                batch_of_indices,
            ),
        )

        return self, container_addition_metrics

    @classmethod
    def init(  # type: ignore
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        centroids: Centroid,
        preferences: Preference,
        pareto_front_max_length: int,
    ) -> Tuple[MOMERepertoire, List]:
        """
        Initialize a Multi Objective Map-Elites repertoire with an initial population
        of genotypes. Requires the definition of centroids that can be computed with
        any method such as CVT or Euclidean mapping.

        Note: this function has been kept outside of the object MapElites, so it can
        be called easily called from other modules.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            fitnesses: fitness of the initial genotypes of shape:
                (batch_size, num_criteria)
            descriptors: descriptors of the initial genotypes
                of shape (batch_size, num_descriptors)
            centroids: tesselation centroids of shape (batch_size, num_descriptors)
            preferences: preferences of initial genotypes of shape:
                (batch_size, num_criteria)
            pareto_front_max_length: maximum size of the pareto fronts

        Returns:
            An initialized MAP-Elite repertoire
        """

        # get dimensions
        num_criteria = fitnesses.shape[1]
        num_descriptors = descriptors.shape[1]
        num_centroids = centroids.shape[0]

        # create default values
        default_fitnesses = -jnp.inf * jnp.ones(
            shape=(num_centroids, pareto_front_max_length, num_criteria)
        )
        default_genotypes = jax.tree_util.tree_map(
            lambda x: jnp.zeros(
                shape=(
                    num_centroids,
                    pareto_front_max_length,
                )
                + x.shape[1:] # Use [1:] as first dimension is batch size (so ignore this)
            ),
            genotypes,
        )
        default_descriptors = jnp.zeros(
            shape=(num_centroids, pareto_front_max_length, num_descriptors)
        )

        default_preferences = -jnp.inf * jnp.ones(
            shape=(num_centroids, pareto_front_max_length, num_criteria)
        )

        # create repertoire with default values
        repertoire = MOMERepertoire(  # type: ignore
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            descriptors=default_descriptors,
            centroids=centroids,
            preferences=default_preferences,
        )

        # add first batch of individuals in the repertoire
        new_repertoire, container_addition_metrics = repertoire.add(genotypes, descriptors, fitnesses, preferences)

        return new_repertoire, container_addition_metrics  # type: ignore

    @jax.jit
    def compute_global_pareto_front(
        self,
    ) -> Tuple[ParetoFront[Fitness], Mask]:
        """Merge all the pareto fronts of the MOME repertoire into a single one
        called global pareto front.

        Returns:
            The pareto front and its mask.
        """
        fitnesses = jnp.concatenate(self.fitnesses, axis=0)
        mask = jnp.any(fitnesses == -jnp.inf, axis=-1)
        pareto_mask = compute_masked_pareto_front(fitnesses, mask)
        pareto_front = fitnesses - jnp.inf * (~jnp.array([pareto_mask, pareto_mask]).T)

        return pareto_front, pareto_mask

    @jax.jit
    def get_best_individuals(self) -> Tuple[Genotype, Fitness, jnp.ndarray]:
        """ Return individuals with highest fitness in each objective function
        
        Returns:
            Individuals and corresponding fitness values
        
        """
        fitnesses = jnp.concatenate(self.fitnesses, axis=0)
        best_individuals = jnp.argmax(fitnesses, axis=0)
        best_fitnesses = jnp.take(fitnesses, best_individuals, axis=0)
        genotypes_list = jax.tree_util.tree_map(
            lambda x: jnp.concatenate(x, axis=0), self.genotypes
        )
        best_genotypes = []

        for individual in best_individuals:
            genotype = jax.tree_util.tree_map(
                lambda x: x[individual], genotypes_list
            )
            best_genotypes.append(genotype)
        
        return (best_genotypes, best_fitnesses)


    def sample_global_pf(self, random_key: RNGKey, num_samples: int) -> Tuple[Genotype, RNGKey]:
        """Sample elements from global pareto front.

        Args:
            random_key: a random key to handle stochasticity.
            num_samples: number of samples to retrieve from the repertoire.

        Returns:
            A sample of genotypes and a new random key.
        """

        pareto_front, pareto_mask = self.compute_global_pareto_front
        
        random_key, subkey = jax.random.split(random_key)
        subkeys = jax.random.split(subkey, num=num_samples)
        sampled_genotypes = sample_in_fronts(  # type: ignore
            pareto_front_genotypes=pareto_front,
            mask=pareto_mask,
            random_key=subkeys,
        )

        return sampled_genotypes, random_key
    
    @jax.jit
    def empty(self) -> MOMERepertoire:
        """
        Empty the grid from all existing individuals.

        Returns:
            An empty MapElitesRepertoire
        """

        new_fitnesses = jnp.full_like(self.fitnesses, -jnp.inf)
        new_descriptors = jnp.zeros_like(self.descriptors)
        new_genotypes = jax.tree_map(lambda x: jnp.zeros_like(x), self.genotypes)
        new_preferences = jnp.full_like(self.preferences, -jnp.inf)

        return MOMERepertoire(
            genotypes=new_genotypes,
            fitnesses=new_fitnesses,
            descriptors=new_descriptors,
            centroids=self.centroids,
            preferences=new_preferences,
        )

        # create default values


