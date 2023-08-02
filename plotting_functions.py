
import matplotlib.pyplot as plt
import numpy as np
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.types import (
    Centroid,
    Metrics,
)
from qdax.utils.plotting import ( 
    plot_2d_map_elites_repertoire, 
    plot_mome_pareto_fronts, 
)
from typing import Dict


def plotting_function(
    config: Dict,
    centroids: Centroid,
    metrics: Metrics,
    repertoire: MapElitesRepertoire,
    save_dir: str,
    save_name: str,
):
    
    fig = plt.figure()
    axes = fig.add_subplot(111) 

    fig, axes = plot_2d_map_elites_repertoire(
        centroids=centroids,
        repertoire_fitnesses=metrics["num_solutions"][-1],
        minval=config.env.min_bd,
        maxval=config.env.max_bd,
        vmin=0,
        vmax=config.pareto_front_max_length,
        ax=axes
    )
    plt.savefig(f"{save_dir}/num_solutions_{save_name}")
    plt.close()

    fig, axes = plt.subplots(figsize=(18, 6), ncols=3)

    # plot pareto fronts
    axes = plot_mome_pareto_fronts(
        centroids,
        repertoire,
        minval=config.env.min_bd,
        maxval=config.env.max_bd,
        color_style='spectral',
        axes=axes,
        with_global=True
    )

    # add map elites plot on last axes
    fig, axes = plot_2d_map_elites_repertoire(
        centroids=centroids,
        repertoire_fitnesses=metrics["hypervolumes"][-1],
        minval=config.env.min_bd,
        maxval=config.env.max_bd,
        ax=axes[2],
        vmax=config.env.max_hypervolume,
    )
    plt.savefig(f"{save_dir}/repertoire_{save_name}")

    return plt

