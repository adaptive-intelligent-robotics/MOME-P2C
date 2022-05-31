import jax.numpy as jnp
import pytest

from qdax.core.containers.repertoire import (
    MapElitesRepertoire,
    compute_euclidean_centroids,
)


def test_repertoire() -> None:

    batch_size = 2
    genotype_size = 12
    num_centroids = 4
    num_descriptors = 2

    centroids = compute_euclidean_centroids(
        num_descriptors=num_descriptors,
        num_centroids=num_centroids,
        minval=0.0,
        maxval=1.0,
    )

    expected_centroids = jnp.array(
        [
            [0.25, 0.25],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.75, 0.75],
        ]
    )

    pytest.assume(jnp.allclose(centroids, expected_centroids, atol=1e-6))

    # create an instance
    repertoire = MapElitesRepertoire(
        genotypes=jnp.zeros(shape=(num_centroids, genotype_size)),
        fitnesses=jnp.ones(shape=(num_centroids,)) * (-jnp.inf),
        descriptors=jnp.zeros(shape=(num_centroids, num_descriptors)),
        centroids=centroids,
    )

    # create fake genotypes and scores to add
    fake_genotypes = jnp.ones(shape=(batch_size, genotype_size))
    fake_fitnesses = jnp.zeros(shape=(batch_size,))
    fake_descriptors = jnp.array([[0.1, 0.1], [0.9, 0.9]])

    # do an addition
    repertoire = repertoire.add(fake_genotypes, fake_descriptors, fake_fitnesses)

    # check that the repertoire looks like expected
    expected_genotypes = jnp.array(
        [
            [1.0 for _ in range(genotype_size)],
            [0.0 for _ in range(genotype_size)],
            [0.0 for _ in range(genotype_size)],
            [1.0 for _ in range(genotype_size)],
        ]
    )
    expected_fitnesses = jnp.array([0.0, -jnp.inf, -jnp.inf, 0.0])
    expected_descriptors = jnp.array(
        [
            [0.1, 0.1],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.9, 0.9],
        ]
    )

    # check values
    pytest.assume(jnp.allclose(repertoire.genotypes, expected_genotypes, atol=1e-6))
    pytest.assume(jnp.allclose(repertoire.fitnesses, expected_fitnesses, atol=1e-6))
    pytest.assume(jnp.allclose(repertoire.descriptors, expected_descriptors, atol=1e-6))