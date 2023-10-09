# HPC utils

For more information about the HPC, you can check the [website](https://www.imperial.ac.uk/computational-methods/hpc/) or the [wiki](https://wiki.imperial.ac.uk/display/HPC/High+Performance+Computing).

`hpc.sh` is a script designed to simplify:
- the generation of jobscripts.
- running jobscripts on the HPC.

## How to use `hpc.sh`?

### Prerequisites

Before using `hpc.sh`, ensure you have the following prerequisites in place:
- You are at the root of a project.
- You have created a container with [`launch_container.sh`](https://gitlab.doc.ic.ac.uk/AIRL/airl_tools/singularity_scripts).
- There is a file named `apptainer/hpc.yaml` containing the parameters of the jobscripts you want to generate and run on the HPC.

### Usage

```bash
hpc.sh <container_path>
```

For example,
```bash
hpc.sh apptainer/container_2023-10-04_174826_a04b5a55a7e22f715d4b0eb1f35447cd20f86dd3.sif
```

The script will:
1. set up a directory for the experiment on the HPC, called `~/project/<project_name>/<container_name>/`.
2. send the container to the HPC.
3. send a file containing the output of the git log command to the HPC.
4. generate jobscripts according to the configuration supplied in `apptainer/hpc.yaml` and using `template.job`.
5. send and submit the jobs to the HPC.
