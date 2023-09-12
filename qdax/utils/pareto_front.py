"""Utils to handle pareto fronts."""

import chex
import jax
import jax.numpy as jnp
import numpy as np

from functools import partial
from typing import Tuple
from pymoo.indicators.hv import HV
from qdax.types import (
    Mask,
    ParetoFront,
    Preference,
    RNGKey
)


def compute_pareto_dominance(
    criteria_point: jnp.ndarray, batch_of_criteria: jnp.ndarray
) -> jnp.ndarray:
    """Returns if a point is pareto dominated given a set of points or not.
    We use maximization convention here.

    criteria_point has shape (num_criteria,)
    batch_of_criteria has shape (num_points, num_criteria)

    Args:
        criteria_point: a vector of values.
        batch_of_criteria: a batch of vector of values.

    Returns:
        Return booleans when the vector is dominated by the batch.
    """
    diff = jnp.subtract(batch_of_criteria, criteria_point)
    return jnp.any(jnp.all(diff > 0, axis=-1))


def compute_pareto_front(batch_of_criteria: jnp.ndarray) -> jnp.ndarray:
    """Returns an array of boolean that states for each element if it is
    in the pareto front or not.

    Args:
        batch_of_criteria: a batch of points of shape (num_points, num_criteria)

    Returns:
        An array of boolean with the boolean stating if each point is on the
        front or not.
    """
    func = jax.vmap(lambda x: ~compute_pareto_dominance(x, batch_of_criteria))
    return func(batch_of_criteria)


def compute_masked_pareto_dominance(
    criteria_point: jnp.ndarray, batch_of_criteria: jnp.ndarray, mask: Mask
) -> jnp.ndarray:
    """Returns if a point is pareto dominated given a set of points or not.
    We use maximization convention here.

    This function is to be used with constant size batches of criteria,
    thus a mask is used to know which values are padded.

    Args:
        criteria_point: values to be evaluated, of shape (num_criteria,)
        batch_of_criteria: set of points to compare with,
            of shape (batch_size, num_criteria)
        mask: mask of shape (batch_size,), 1.0 where there is not element,
            0 otherwise

    Returns:
        Boolean assessing if the point is dominated or not.
    """

    diff = jnp.subtract(batch_of_criteria, criteria_point)
    neutral_values = -jnp.ones_like(diff)
    diff = jax.vmap(lambda x1, x2: jnp.where(mask, x1, x2), in_axes=(1, 1), out_axes=1)(
        neutral_values, diff
    )
    return jnp.any(jnp.all(diff > 0, axis=-1))


def compute_masked_pareto_front(
    batch_of_criteria: jnp.ndarray, mask: Mask
) -> jnp.ndarray:
    """Returns an array of boolean that states for each element if it is to be
    considered or not. This function is to be used with batches of constant size
    criteria, thus a mask is used to know which values are padded.

    Args:
        batch_of_criteria: data points considered
        mask: mask to hide several points

    Returns:
        An array of boolean stating the points to consider.
    """
    func = jax.vmap(
        lambda x: ~compute_masked_pareto_dominance(x, batch_of_criteria, mask)
    )
    return func(batch_of_criteria) * ~mask


def compute_hypervolume(
    pareto_front: ParetoFront[jnp.ndarray], reference_point: jnp.ndarray
) -> jnp.ndarray:
    """Compute hypervolume of a pareto front.

    Args:
        pareto_front: a pareto front, shape (pareto_size, num_objectives)
        reference_point: a reference point to compute the volume, of shape
            (num_objectives,)

    Returns:
        The hypervolume of the pareto front.
    """
    # check the number of objectives
    custom_message = (
        "Hypervolume calculation for more than" " 2 objectives not yet supported."
    )
    chex.assert_axis_dimension(
        tensor=pareto_front,
        axis=1,
        expected=2,
        custom_message=custom_message,
    )

    # concatenate the reference point to prepare for the area computation
    pareto_front = jnp.concatenate(  # type: ignore
        (pareto_front, jnp.expand_dims(reference_point, axis=0)), axis=0
    )
    # get ordered indices for the first objective
    idx = jnp.argsort(pareto_front[:, 0])
    # Note: this orders it in inversely for the second objective

    # create the mask - hide fake elements (those having -inf fitness)
    mask = pareto_front[idx, :] != -jnp.inf

    # sort the front and offset it with the ref point
    sorted_front = (pareto_front[idx, :] - reference_point) * mask

    # compute area rectangles between successive points
    sumdiff = (sorted_front[1:, 0] - sorted_front[:-1, 0]) * sorted_front[1:, 1]
    #jax.debug.print("SAME {}:",jnp.all((((sorted_front[1:, 0] - sorted_front[:-1, 0])) * sorted_front[1:, 1])==sumdiff))

    # remove the irrelevant values - where a mask was applied
    sumdiff = sumdiff * mask[:-1, 0]

    # get the hypervolume by summing the succcessives areas
    hypervolume = jnp.sum(sumdiff)

    return hypervolume


def compute_hypervolume_3d(
    pareto_front: ParetoFront[jnp.ndarray],
    reference_point: jnp.ndarray
) -> jnp.ndarray:

    mask = pareto_front == -jnp.inf
    _pf = jnp.where(mask, reference_point, pareto_front)
    _ref_point = reference_point

    def _hv(ref_point, pf):
        return HV(ref_point=ref_point * -1)(pf * -1).astype(np.float32)

    _hv_shape = jax.core.ShapedArray((), jnp.float32)
    hv = jax.pure_callback(_hv, _hv_shape, _ref_point, _pf)

    return hv

def compute_sparsity(
    pareto_front: jnp.ndarray,
    min_fitnesses: jnp.ndarray,
    max_fitnesses: jnp.ndarray,
)-> float:
    
    # sort by first objective
    num_objectives = pareto_front.shape[-1]
    len_front = jnp.sum(pareto_front != -jnp.inf)/num_objectives
    
    # scale pareto front so sparsity is not affected by scale
    scaled_front = pareto_front/(max_fitnesses - min_fitnesses)
    
    # compute sparsity for front with more than one solution
    def true_fun(pareto_front, num_objectives, len_front):
        sparsity = 0.0
        for objective in range(num_objectives):
            sorted_vals = jnp.sort(pareto_front.at[:, objective].get())
            rolled_vals =  jnp.roll(sorted_vals, -1)
            squared_diffs = jnp.square(rolled_vals - sorted_vals)
            mask = squared_diffs != jnp.inf
            sparsity += jnp.nansum(squared_diffs * mask)
        
        sparsity /= len_front - 1
    
        return sparsity
    
    # handle the case where there is only one solution on the front
    def false_fun(pareto_front):
        return 0.0
    
    partial_true_fun = partial(true_fun, num_objectives=num_objectives, len_front=len_front)
    
    return jax.lax.cond(len_front>1, partial_true_fun, false_fun, scaled_front)

@partial(jax.jit, static_argnames=("batch_size", "num_objectives"))
def uniform_preference_sampling(
    random_key: RNGKey,
    batch_size: int,
    num_objectives: int,
) -> Tuple[Preference, RNGKey]:
    """Sample random preferences to evalute and train actor with."""

    random_key, subkey = jax.random.split(random_key)

    first_cols_sampled_preferences = jax.random.uniform(
        random_key, shape=(batch_size, num_objectives-1), minval=0.0, maxval=1.0
    )

    sum_first_cols_sampled_preferences = jnp.sum(first_cols_sampled_preferences, axis=1)
    
    # Need to make sure preferences sum to 1
    last_col_sampled = jnp.ones(batch_size) - sum_first_cols_sampled_preferences
    sampled_preferences = jnp.hstack((first_cols_sampled_preferences, jnp.expand_dims(last_col_sampled, axis=1)))

    return sampled_preferences, random_key
