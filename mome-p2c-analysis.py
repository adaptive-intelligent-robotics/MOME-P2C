from analysis_functions.run_analysis import MOQD_Analysis


# Parent directory of results
parent_dirname = "results/"


# Directory names of experiments
experiment_names = [
    # main
    "mome-p2c",

    # baselines
    "mome-pgx",
    "mome",
    "pga",
    "nsga2",
    "spea2",

]

# Directory names of environments
env_names=[
    "ant_multi",
    "ant_tri",
    "halfcheetah_multi",
    "hopper_multi",
    "hopper_tri",
    "walker2d_multi",
]


experiment_dicts = {
    
    ## MAIN
    
    "mome-p2c": {
        "label": "MOME-P2C",
        "emitter_names": ["emitter_actor_inject_count", "emitter_ga_count", "emitter_pg_count"],
        "emitter_labels": ["Actor Injection Emitter", "GA Emitter", "PG Emitter"],
        "grid_plot_linestyle": "solid",
    },
    
    ## BASELINES
    
    "mome-pgx": {
        "label": "MOME-PGX",
        "emitter_names": ["emitter_f1_count", "emitter_f2_count", "emitter_f3_count", "emitter_ga_count"],
        "emitter_labels": ["F1 Emitter", "F2 Emitter", "F3 Emitter", "GA Emitter"],
        "grid_plot_linestyle": "dashed", 
    },
    "mome": {
        "label": "MOME",
        "emitter_names": ["emitter_count"],
        "emitter_labels": ["GA Emitter"],
        "grid_plot_linestyle": (0, (3, 1, 1, 1, 1, 1)),
    },
    "pga": {
        "label": "PGA",
        "emitter_names": ["emitter_1_count", "emitter_2_count"],
        "emitter_labels": ["Gradient Emitter", "GA Emitter"],
        "grid_plot_linestyle": "dotted"
    },
    "nsga2": {
        "label": "NSGA-II",
        "emitter_names": ["emitter_count"],
        "emitter_labels": ["GA Emitter"],
        "grid_plot_linestyle": (5, (10, 3)),
    },
    "spea2": {
        "label": "SPEA2",
        "emitter_names": ["emitter_count"],
        "emitter_labels": ["GA Emitter"],
        "grid_plot_linestyle": "dashdot",
    },
    
    ## ABLATIONS
    
    "mome-p2c-inject-actor": {
        "label": "Inject-Actor",
        "emitter_names": ["emitter_actor_inject_count", "emitter_ga_count", "emitter_pg_count"],
        "emitter_labels": ["Actor Injection Emitter", "GA Emitter", "PG Emitter"],
        "grid_plot_linestyle": "dashed",
    },   
    "mome-p2c-no-qpg": {
        "label": "No-QPG",
        "emitter_names": ["emitter_actor_inject_count", "emitter_ga_count", "emitter_pg_count"],
        "emitter_labels": ["Actor Injection Emitter", "GA Emitter", "PG Emitter"],
        "grid_plot_linestyle": (0, (3, 1, 1, 1, 1, 1)),
    },
    "mome-p2c-no-crowding": {
        "label": "No-Crowding",
        "emitter_names": ["emitter_actor_inject_count", "emitter_ga_count", "emitter_pg_count"],
        "emitter_labels": ["Actor Injection Emitter", "GA Emitter", "PG Emitter"],
        "grid_plot_linestyle": "dotted",
    },
    "mome-p2c-keep-prefs": {
        "label": "Keep-Prefs",
        "emitter_names": ["emitter_actor_inject_count", "emitter_ga_count", "emitter_pg_count"],
        "emitter_labels": ["Actor Injection Emitter", "GA Emitter", "PG Emitter"],
        "grid_plot_linestyle": (5, (10, 3)),
    },
   "mome-p2c-actor-random-sampler": {
        "label": "Actor Random Sampler",
        "emitter_names": ["emitter_actor_inject_count", "emitter_ga_count", "emitter_pg_count"],
        "emitter_labels": ["Actor Injection Emitter", "GA Emitter", "PG Emitter"],
        "grid_plot_linestyle": "dashed",
    },   

   "mome-p2c-one-hot": {
        "label": "One-hot-Prefs",
        "emitter_names": ["emitter_actor_inject_count", "emitter_ga_count", "emitter_pg_count"],
        "emitter_labels": ["Actor Injection Emitter", "GA Emitter", "PG Emitter"],
        "grid_plot_linestyle": "dashed",
    },   
   
}



env_dicts = {
    "ant_multi": {
        "label": "Ant-2",
        "reward_labels": ["Forward", "Energy"],
        "action_size":8,
        "observation_size":87,
        "reference_point": [-350, -4500],
        "exceptions": [],
    },
    "ant_tri": {
        "label": "Ant-3",
        "reward_labels": ["X-Velocity", "Y-Velocity", "Energy"],
        "action_size":8,
        "observation_size":87,
        "reference_point": [-1200, -1200, -4500],
        "exceptions": ["pga", "nsga2", "spea2"],
    },
    "halfcheetah_multi": {
        "label": "HalfCheetah-2",
        "reward_labels": ["Forward", "Energy"],
        "action_size":6,
        "observation_size":18,
        "reference_point": [-2000, -800],
        "exceptions": [],
    },
    "hopper_multi": {
        "label": "Hopper-2",
        "reward_labels": ["Forward", "Energy"],
        "action_size":3,
        "observation_size":11,
        "reference_point":[-50, -3],
        "exceptions": [],
    },
    "hopper_tri": {
        "label": "Hopper-3",
        "reward_labels": ["Forward", "Energy", "Torso Height"],
        "action_size":3,
        "observation_size":11,
        "reference_point": [-750, -3, 0],
        "exceptions": ["pga", "nsga2", "spea2"],
    },
    "walker2d_multi": {
        "label": "Walker-2",
        "reward_labels": ["Forward", "Energy"],
        "action_size":6,
        "observation_size":17,
        "reference_point": [-210, -15],
        "exceptions": [],
    },
}



# Metrics to plot in grid plot
grid_plot_metrics_list = [
    "moqd_score", 
    "global_hypervolume", 
    # "max_sum_scores",
]

grid_plot_metrics_labels = {
    "moqd_score": "MOQD Score", 
    "global_hypervolume": "Global Hypervolume", 
    # "max_sum_scores": "Max Sum Scores",
}


# List of metrics to calculate p-values for
p_value_metrics_list = [
    "global_hypervolume", 
    "max_sum_scores", 
    "moqd_score",
]

# Which algorithms to compare data-efficiency and which metric for comparison
data_efficiency_params={
    "test_algs": ["mome_pgx"],
    "baseline_algs": ["mome"],
    "metrics": ["moqd_score"],
}


if __name__ == "__main__":
    
    
    analysis_helper = MOQD_Analysis(
        parent_dirname=parent_dirname,
        env_names=env_names,
        env_dicts=env_dicts,
        experiment_names=experiment_names,
        experiment_dicts=experiment_dicts,
        num_replications=20,
        num_iterations=4000,
        episode_length=1000,
        batch_size=256
    )
    
    analysis_helper.plot_grid(
        grid_plot_metrics_list,
        grid_plot_metrics_labels,
    )

    analysis_helper.calculate_wilcoxon(
        p_value_metrics_list
    )
    
    analysis_helper.sparsity_analysis()

    analysis_helper.plot_final_pfs()