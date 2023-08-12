import jax.numpy as jnp
import jax
import numpy as np

from dataclasses import dataclass
from functools import partial
from scipy.optimize import least_squares
from typing import Tuple
from qdax.core.containers.mome_repertoire import MOMERepertoire
from qdax.core.emitters.preference_sampling.preference_sampler import (
    PreferenceSamplingState,
    PreferenceSampler
)
from qdax.core.neuroevolution.buffers.scores_buffer import ScoresBuffer
from qdax.utils.pareto_front import (
    compute_hypervolume,
    compute_masked_pareto_front, 
    uniform_preference_sampling,
)

from qdax.types import (
    Fitness,
    Genotype,
    Mask,
    ParetoFront,
    Preference,
    RNGKey
)



@dataclass
class HyperbolicModelConfig:
    """Configuration for PGAME Algorithm"""
    
    # Env params
    num_objectives: int = 1 
    reference_point: Tuple[float, ...] = (0.0, 0.0)

    # Emitter params
    emitter_batch_size: int = 256

    # Hyperbolic Model Params
    buffer_size: int = 256000
    num_weight_candidates: int = 10
    scaling_sigma: float = 0.1

    # Neighbourhood params
    num_neighbours: int = 16 


class HyperbolicModelState(PreferenceSamplingState):
    """The state of a Hyperbolic Model preference sampler."""
    old_fitnesses: ScoresBuffer
    new_fitnesses: ScoresBuffer
    weights_history: ScoresBuffer
    
    prev_iter_fitnesses: Fitness
    prev_iter_weights: Preference
    random_key: RNGKey


class HyperbolicPredictionGuidedSampler(PreferenceSampler):
    
    def __init__(
        self,
        config: HyperbolicModelConfig,
    ):
        """Hyperbolic Model for Multi-Objective Optimisation.
        """

        # Model parameters
        self._config = config


    def init(
        self,
        init_genotypes: Genotype,
        random_key: RNGKey
    ) -> Tuple[PreferenceSamplingState, RNGKey]:
        
        """
        Initialise the state of the sampler."""

        old_fitnesses_buffer = ScoresBuffer.init(
            buffer_size=self._config.buffer_size,
            num_objectives=self._config.num_objectives,
        )
        
        prev_iter_fitnesses = jnp.zeros(
            (self._config.emitter_batch_size,
            self._config.num_objectives)
        )

        new_fitnesses_buffer = ScoresBuffer.init(
            buffer_size=self._config.buffer_size,
            num_objectives=self._config.num_objectives,
        )

        weights_history = ScoresBuffer.init(
            buffer_size=self._config.buffer_size,
            num_objectives=self._config.num_objectives,
        )
        
        prev_iter_weights = jnp.zeros(
            (self._config.emitter_batch_size,
            self._config.num_objectives)
        )

        random_key, subkey = jax.random.split(random_key)

        sampling_state = HyperbolicModelState(
            old_fitnesses = old_fitnesses_buffer,
            new_fitnesses = new_fitnesses_buffer,
            weights_history = weights_history,
            prev_iter_fitnesses = prev_iter_fitnesses,
            prev_iter_weights = prev_iter_weights,
            random_key = subkey,
        )

        return sampling_state, random_key

    
    def init_state_update(
        self,
        sampling_state: HyperbolicModelState,
        batch_init_fitnesses: Fitness,
        batch_init_preferences: Preference
    ) -> HyperbolicModelState:
        """        
        Returns:
            The modified sampling state.
        """
        # just take batch size amount to avoid errors with jax.lax.scan in mome main
               
        sampling_state = sampling_state.replace(
            prev_iter_weights=batch_init_preferences[:self._config.emitter_batch_size],
            prev_iter_fitnesses=batch_init_fitnesses[:self._config.emitter_batch_size],
        )

        return sampling_state
    
    @partial(jax.jit, static_argnames=("self",))
    def sample(
        self,
        repertoire: MOMERepertoire,
        sampling_state: HyperbolicModelState,
    ):
        """
        Sample a batch of genotypes and find their corresponding
        weights to train with poloicy gradient.

        Args:
            repertoire: MOMERepertoire
            sampling_state: current state of the sampler
        
        Returns:
            sampling_state: updated state of the sampler
            genotypes: sampled genotypes
            weights: weights to use PG variation with
        """
        random_key = sampling_state.random_key
        random_key, subkey = jax.random.split(random_key)
        
        genotypes, fitnesses, _, pfs = repertoire.sample_batch(
            random_key = subkey,
            num_samples = self._config.emitter_batch_size
        )

        # Find best weights for sampled genotypes

        random_key = sampling_state.random_key
        random_key, subkey = jax.random.split(random_key)
        subkeys = jax.random.split(subkey, num=self._config.emitter_batch_size)

        partial_find_weight_fn = partial(self._find_best_predicted_weights, sampling_state=sampling_state)
        
        # weights = jax.vmap(partial_find_weight_fn)(
        #     fitnesses,
        #     pfs,
        #     subkeys,
        # )
        
        weights = jnp.zeros((self._config.emitter_batch_size, self._config.num_objectives))
        
        for i in range(self._config.emitter_batch_size):
            fitness = fitnesses[i]
            pf = pfs[i]
            subkey = subkeys[i]

            best_predicted_weight = partial_find_weight_fn(
                fitness,
                pf,
                subkey,
            )
            weights = weights.at[i].set(best_predicted_weight)
        
        # jax.debug.print("prev iter fitnesses shape {}", fitnesses.shape)
        sampling_state = sampling_state.replace(
            prev_iter_fitnesses=fitnesses,
            prev_iter_weights=weights,
            random_key=random_key,
        )

        return genotypes, weights, sampling_state
    
    @partial(jax.jit, static_argnames=("self",))
    def _find_best_predicted_weights(
        self,
        fitness,
        front,
        random_key,
        sampling_state,
    )-> Preference:

        neighbours_weights, neighbours_deltas, neighbour_scales = self._find_neighbours(
            solution_fitness=fitness,
            sampling_state=sampling_state,
        )      
                
        model_params = self._build_model(
            neighbours_weights,
            neighbours_deltas,
            neighbour_scales
        )

        weight_candidates, _ = uniform_preference_sampling(
            random_key=random_key,
            batch_size=self._config.num_weight_candidates,
            num_objectives=self._config.num_objectives,
        )

        predicted_deltas = self._predict_performance(
            weight_candidates=weight_candidates,
            model_params=model_params, 
        )
        
        best_predicted_weight = self._choose_weight_candidate(
            current_fitness=fitness,
            current_pf=front,
            predicted_deltas=predicted_deltas,
            weight_candidates=weight_candidates,
        )

        return best_predicted_weight
        

    @partial(jax.jit, static_argnames=("self",))
    def _find_neighbours(
        self,
        solution_fitness,
        sampling_state,
    )-> Mask:
        """Find neighbours of the current solution.
        Args:
            solution_fitness: The fitness of the current solution [num_objectives].
            weights_history: The weights of the previous solutions [num_evaluations, num_objectives].
            old_fitnesses: The fitnesses of the previous solutions [num_evaluations, num_objectives].
            init_threshold: threshold value for neighbours [int].
            init_sigma: sigma value for neighbours [int].
        """

        # Find distance between solution and old fitnesses in buffer
        difference_magnitudes = sampling_state.old_fitnesses.calculate_magnitude_difference(solution_fitness)
        
        # Find the samples with closest distance
        idx = jnp.argsort(difference_magnitudes)
        neighbours_weights_sorted = sampling_state.weights_history.get_indexed_data(idx)
        new_fitnesses_sorted = sampling_state.new_fitnesses.get_indexed_data(idx)
        old_fitnesses_sorted = sampling_state.old_fitnesses.get_indexed_data(idx)
        # jax.debug.print("old_fitnesses_sorted {}", old_fitnesses_sorted[:20])
        
        neighbours_deltas_sorted = new_fitnesses_sorted - old_fitnesses_sorted
        
        # Keep KNN
        knn_neighbours_weights = neighbours_weights_sorted.at[:self._config.num_neighbours].get()
        knn_neighbours_deltas = neighbours_deltas_sorted.at[:self._config.num_neighbours].get()
        
        # Find scaling
        knn_new_fitnesses = new_fitnesses_sorted.at[:self._config.num_neighbours].get()
        new_fitness_diffs = jnp.linalg.norm(
            jnp.abs(knn_new_fitnesses - solution_fitness)/jnp.abs(solution_fitness), 
            axis=-1
        )
        
        knn_scales = jnp.exp(-(new_fitness_diffs/self._config.scaling_sigma)**2/2.0)

        return knn_neighbours_weights, knn_neighbours_deltas, knn_scales
    

  
    @partial(jax.jit, static_argnames=("self",))
    def _build_model(
        self,
        neighbours_weights: Preference,
        neighbours_deltas: jnp.ndarray,
        scales: jnp.ndarray,
    )-> jnp.ndarray:
        """ Build a model for choosing next weights, given weights and deltas of neighbours.
        
        Args:
            weights: The weights of the neighbours [num_neighbours, num_objectives].
            deltas: The deltas of the neighbours [num_neighbours, num_objectives].
        
        Returns:
            model_params: The parameters of the model [num_objectives, 4].
        """
        
        model_params = jnp.zeros((self._config.num_objectives, 4))
                        
        def _fun(params, x, y, scales, num_neighbours):
            """ Error Function """
            # f = A * (exp(a(x - b)) - 1) / (exp(a(x - b)) + 1) + c
            
            # return (params[0] * (np.exp(params[1] * (x - params[2])) - 1.) / (np.exp(params[1] * (x - params[2])) + 1) + params[3] - y)
            
            error_fun = lambda params, x_train, y_train, scale, num_neighbours: (
                params[0] * (np.exp(params[1] * (x_train - params[2])) - 1.) / (np.exp(params[1] * (x_train - params[2])) + 1) 
                + params[3]
                - y_train
            ) * scale
            
            fun_shape = jax.core.ShapedArray((num_neighbours,), 'float32')
        
            return jax.pure_callback(error_fun, fun_shape, params, x, y, scales, num_neighbours, vectorized=True)

        def _jac(params, x, y, scales, num_neighbours):
            """ Jacobian of Error Function """
            
            def jac_callback(params, x_train, y_train, scale, num_neighbours):

                A, a, b, c = params[0], params[1], params[2], params[3]

                J = np.zeros([4, num_neighbours])

                # df_dA = (exp(a(x - b)) - 1) / (exp(a(x - b)) + 1)
                J[0] = (np.exp(a * (x_train - b)) - 1) / (np.exp(a * (x_train - b)) + 1) * scale

                # df_da = A(x - b)(2exp(a(x-b)))/(exp(a(x-b)) + 1)^2
                J[1] = (A * (x_train - b) * (2. * np.exp(a * (x_train - b))) / ((np.exp(a * (x_train - b)) + 1) ** 2)) * scale

                # df_db = A(-a)(2exp(a(x-b)))/(exp(a(x-b)) + 1)^2
                J[2] = (A * (-a) * (2. * np.exp(a * (x_train - b))) / ((np.exp(a * (x_train - b)) + 1) ** 2)) * scale

                # df_dc = 1
                J[3] = 1 * scale

                return np.transpose(J)

            jac_shape = jax.core.ShapedArray((num_neighbours, 4), 'float32')

            return jax.pure_callback(jac_callback, jac_shape, params, x, y, scales, num_neighbours, vectorized=True)
    
        def _fit_model(
            x_train,
            y_train,
            scales,
            num_neighbours,
            clip
        ):
                
            _scipy_least_squares = lambda x, y, s, n, clip: least_squares(
                fun = _fun,
                x0 = np.ones(4),
                loss = "soft_l1",
                f_scale = 20.,
                args = (x, y, s, n),
                jac = _jac,
                bounds = ([0, 0.1, -5., -500.], [clip, 2., 5., 500.])
            ).x.astype("float32")
            
            shape = jax.core.ShapedArray((4,), 'float32')

            return jax.pure_callback(_scipy_least_squares, shape, x_train, y_train, scales, num_neighbours, clip, vectorized=True)

            # _scipy_least_squares = lambda x, y, z: least_squares(
            #     fun = _fun,
            #     x0 = np.ones(4),
            #     loss = "soft_l1",
            #     f_scale=20.,
            #     args = (x, y, z),
            #     jac = _jac,
            # ).x.astype("float32")
            
            # shape = jax.core.ShapedArray((4,), 'float32')

            
            # return jax.pure_callback(_scipy_least_squares, shape, x_train, y_train, num_neighbours, vectorized=True)


        for dim in range(self._config.num_objectives):

            train_x = neighbours_weights.at[:, dim].get()
            train_y = neighbours_deltas.at[:, dim].get()
            
            scale_coeff_upperbound = jnp.clip(jnp.max(train_x) - jnp.min(train_y), 1.0, 500.0)

            params = _fit_model(
                train_x,
                train_y,
                scales,
                self._config.num_neighbours,
                scale_coeff_upperbound,
            )

            model_params = model_params.at[dim].set(params)

        return model_params

    @partial(jax.jit, static_argnames=("self",))
    def _predict_performance(
        self,
        weight_candidates: Preference,
        model_params: jnp.ndarray,
    )-> jnp.ndarray:
        """
        Predict new fitness of solution for different weight values, given hyperbolic model params.

        Args:
            weight_candidates: Weight candidates [num_candidates, num_objectives].
            model_params: Parameters of hyperbolic model [num_objectives, 4].

        Returns:
            Predictions: Predicted change in fitnesses [num_candidates, num_objectives].
        """
        
        scale_coeff = jnp.transpose(model_params.at[:, 0].get())
        a = jnp.transpose(model_params.at[:, 1].get())
        b = jnp.transpose(model_params.at[:, 2].get())
        c = jnp.transpose(model_params.at[:, 3].get())
        
        return scale_coeff * (jnp.exp(a * (weight_candidates - b)) - 1) / (jnp.exp(a * (weight_candidates - b)) + 1) + c
    
    @partial(jax.jit, static_argnames=("self",))
    def _choose_weight_candidate(
        self,
        current_fitness: Fitness,
        current_pf: ParetoFront,
        predicted_deltas: jnp.ndarray,
        weight_candidates: jnp.ndarray,
    ):
    
        """
        Args:
            current_fitness: fitness of selected solution [num_objectives].
            current_pf: pareto front of current population [max_pareto_front_length, num_objectives].
            predicted_deltas: predicted change in fitnesses [num_candidates, num_objectives].
            weight_candidates: weight candidates [num_candidates, num_objectives].
        
        Returns:
            best_predicted_weight: best weight for training solution [num_objectives].
        
        """

        def _add_to_pf(
            candidate_fitness,
            current_pf,
            current_pf_mask,        
        ):
            """
            Calculate new pf by adding candidate to current pf.
            """

            # gather dimensions
            num_objectives = current_pf.shape[-1]
            new_front_length = current_pf.shape[0] + 1


            # gather all data
            new_mask = jnp.array([0], dtype=bool)
            candidate_fitness = jnp.expand_dims(candidate_fitness, axis=0)

            cat_mask = jnp.concatenate(
                [current_pf_mask, new_mask], axis=0
            )
            cat_fitnesses = jnp.concatenate(
                [current_pf, candidate_fitness], axis=0
            )

            # get new front
            cat_bool_front = compute_masked_pareto_front(
                batch_of_criteria=cat_fitnesses, mask=cat_mask
            )

            indices = (
                jnp.arange(start=0, stop=new_front_length) * cat_bool_front
            )
            indices = indices + ~cat_bool_front * (new_front_length - 1)
            indices = jnp.sort(indices)

            new_front_fitness = jnp.take(cat_fitnesses, indices, axis=0)

            num_front_elements = jnp.sum(cat_bool_front)
            new_mask_indices = jnp.arange(start=0, stop=new_front_length)
            new_mask_indices = (num_front_elements - new_mask_indices) > 0

            new_mask = jnp.where(
                new_mask_indices,
                jnp.ones(shape=new_front_length, dtype=bool),
                jnp.zeros(shape=new_front_length, dtype=bool),
            )

            fitness_mask = jnp.repeat(
                jnp.expand_dims(new_mask, axis=-1), num_objectives, axis=-1
            )
            new_pf = new_front_fitness * fitness_mask
            
            return new_pf

        predicted_next_fitnesses = current_fitness + predicted_deltas

        mask = jnp.any(current_pf == -jnp.inf, axis=-1)

        _new_fronts_function = partial(
            _add_to_pf, 
            current_pf=current_pf, 
            current_pf_mask=mask
        )
        
        # New predicted fronts = [num_candidates, max_pareto_front_length + 1, num_objectives]
        new_fronts = jax.vmap(_new_fronts_function)(predicted_next_fitnesses)

        # Compute hypervolume of predicted fronts
        hypervolume_function = partial(compute_hypervolume, reference_point=self._config.reference_point)
        hypervolumes = jax.vmap(hypervolume_function)(new_fronts) 

        # Choose weight which is predicted to lead to best hypervolume
        best_predicted_weight = weight_candidates[jnp.argmax(hypervolumes)]

        return best_predicted_weight
    

    @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        sampling_state: HyperbolicModelState,
        batch_new_fitnesses: Fitness,
    )-> HyperbolicModelState:

        fitnesses_buffer = sampling_state.new_fitnesses.insert(batch_new_fitnesses)
        old_fitnesses_buffer = sampling_state.old_fitnesses.insert(sampling_state.prev_iter_fitnesses)
        weights_history_buffer = sampling_state.weights_history.insert(sampling_state.prev_iter_weights)
        
        new_sampling_state = sampling_state.replace(
            new_fitnesses=fitnesses_buffer,
            old_fitnesses=old_fitnesses_buffer,
            weights_history=weights_history_buffer,
        )

        return new_sampling_state
