## Intro
This repository contains all code used in [Preference-Conditioned Gradient Variations for Multi-Objective Quality-Diversity]() paper. This  builds on top of the [QDax framework](https://github.com/adaptive-intelligent-robotics/QDax) and includes the newly introduced _Multi-Objective Map-Elites with Preference-Conditioned Policy-Gradient and Crowding Mechanisms_ ([MOME-P2C]).


MOME-P2C is a Multi-Objective Quality-Diversity algorithm which is builds on the [MOME algorithm](https://arxiv.org/abs/2202.03057) and the [MOME-PGX algorithm](https://arxiv.org/abs/2302.12668). In particular, MOME-P2C improves upon these algorithms by using preference-conditioned actor-critic networks. These are used to (1) provide preference-conditioned policy gradient updates on solutions that are already in the grid and (2) inject the actor into the main population at each iteration.

<img width="1092" alt="mome-p2c" src="https://github.com/adaptive-intelligent-robotics/MOME-P2C/assets/49594227/2a53afd0-0d76-44cf-9a71-e9b2cd4032c1">

MOME-P2C is evaluated on six tasks from the [Brax Suite](https://pypi.org/project/brax/), summarised in the table below:

<img width="961" alt="tasks" src="https://github.com/adaptive-intelligent-robotics/MOME-P2C/assets/49594227/2c5bc13b-6d7a-44a6-8ba5-3b86c2499dcd">


## Installation

To run this code, you need to install the necessary libraries as listed in `requirements.txt` via:

```bash
pip install -r requirements.txt
```

However, we recommend using a containerised environment such as Docker, Singularity or conda  to use the repository. Further details are provided in the last section. 

## Basic API Usage

To run the MOME-P2C algorithm, or any other baseline algorithm mentioned in the paper, you just need to run the `main.py` script and specify the  algorithm you wish to run. For example, to run the 
```bash
python3 main.py -—algo=mome-p2c
```

Or to run the MOME algorithm:
```bash
python3 main.py -—algo=mome
```

The hyperparameters of the algorithms can be modified by changing their values in the `configs` directory of the repository. Alternatively, they can be modified directly in the command line. For example, to decrease the `pareto_front_max_length` parameter from 50 to 20 in MOME-P2C, you can run:

```bash
python3 main.py --algo=mome-p2c pareto_front_max_length=20
```

Running each algorithm automatically saves metrics, visualisations and plots of performance into a `results` directory. However, you can compare performance between algorithms once they have been run using the `analysis.py` script. To do this, you need to edit the list of the algorithms and environments you wish to compare and the metrics you wish to compute (at the bottom of `analysis.py`). Then, the relevant plots and performance metrics will be computed by running:

```bash
python3 analysis.py
```

## Singularity Usage

To build a final container (an executable file) using Singularity make sure you are in the root of the repository and then run:

```bash
singularity build --fakeroot --force singularity/[FINAL CONTAINER NAME].sif singularity/singularity.def
```

where you can replace '[FINAL CONTAINER NAME]' by your desired file name. When you get the final image, you can execute it via:

```bash
singularity -d run --app [APPNAME] --cleanenv --containall --no-home --nv [FINAL CONTAINER NAME].sif [EXTRA ARGUMENTS]
```

where 
- [FINAL CONTAINER NAME].sif is the final image built
- [APPNAME] is the name of the experiment you want to run, as specified by `%apprun` in the `singularity/singularity.def` file. There is a specific `%apprun` for each of the algorithms, ablations and baselines mentioned in the paper.
- [EXTRA ARGUMENTS] is a list of any futher arguments that you want to add. For example, you may want to change the random seed or Brax environment.

