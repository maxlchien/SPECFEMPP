# Wave propagation through homogeneous medium with no interfaces

This example reproduces the results from Fig 9 of [Tromp et al. 2005](https://doi.org/10.1111/j.1365-246X.2004.02453.x). For a step-by-step guide on this example, please refer to the [documentation](https://specfem2d-kokkos.readthedocs.io/en/latest/cookbooks/dim2/kernels-example-tromp-2005/index.html).

## Running the examples

To run the examples, you first need to install uv following these
[instructions](https://docs.astral.sh/uv/getting-started/installation). Once you've done
so, you can install the dependencies for the examples by running the following
command in the current directory:

```bash
# verify uv is installed
uv --version

# install dependencies
uv sync --group examples

```

After installing the dependencies, you can run the examples by running the following command within the example directory you want to run:

```bash

cd <example directory>

# run the example
uv run snakemake -j 1

# or to run the example on a slurm cluster
uv run snakemake --executor slurm -j 1

```

## Cleaning up

To clean up the example directory, you can run the following command in the directory of the example you want to clean up:

```bash

# clean up the example
uv run snakemake clean -j 1

```
