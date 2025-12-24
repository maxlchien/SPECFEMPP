# Marmousi2 Benchmark

This benchmark uses the Marmousi2 model with an externally generated mesh from CUBIT.
The mesh files are included in the `MESH-default` directory.

The Marmousi2 model is a complex synthetic velocity model commonly used for testing
seismic wave propagation codes. This benchmark uses the default CUBIT-generated mesh.

## Running the benchmark

To run the benchmark, you first need to install uv following these
[instructions](https://docs.astral.sh/uv/getting-started/installation). Once you've done
so, you can install the dependencies for the benchmarks by running the following
command in the current directory:

```bash
# verify uv is installed
uv --version

# install dependencies
uv sync --group examples

```

After installing the dependencies, you can run the benchmark by running the
following command within the benchmark directory:

```bash

# run the benchmark
uv run snakemake -j 1

# or to run the benchmark on a slurm cluster
uv run snakemake --executor slurm -j 1

```

## Cleaning up

To clean up the benchmark directory, you can run the following command:

```bash

# clean up the benchmark
uv run snakemake clean -j 1

```

## Details

- **Mesh**: External mesh from CUBIT (4-node quadrilateral elements)
- **Source**: Ricker wavelet force source at (5000.0, 3450.0) meters with f0=5.0 Hz
- **Receivers**: 11 receivers distributed along a horizontal line from x=1000m to x=11000m at z=3450m
- **Time stepping**: Newmark scheme with dt=5.0e-6s for 100000 steps (0.5s total)
- **Boundary conditions**: Stacey absorbing conditions on bottom, left, and right edges

## Additional Resources

This benchmark includes several directories for mesh generation and alternative meshes:

- **CUBIT_meshing/** - Scripts and files for generating the mesh using CUBIT/Trelis
  - Python scripts for creating, meshing, smoothing, and exporting the mesh
  - CUBIT journal files
  - Pre-generated CUBIT mesh files (.cub format)

- **MESH-default/** - Default mesh used by the benchmark (included in build)
  - Contains all mesh files needed to run the simulation

- **DATA/** - Parameter files for different mesh configurations
  - `Par_file` - Default parameter file
  - `Par_file.mesh-improved` - Parameter file for improved meshes
  - `SOURCE` and `SOURCE.mesh-improved` - Source files for different configurations

- **improved_mesh_from_Hom_Nath_Gharti_2016_using_CUBIT_higher_resolution/** - Higher resolution improved mesh from 2016

- **improved_mesh_from_Hom_Nath_Gharti_2016_using_CUBIT_lower_resolution/** - Lower resolution improved mesh from 2016

- **improved_mesh_from_Hom_Nath_Gharti_2023_using_CUBIT_higher_resolution/** - Latest higher resolution improved mesh from 2023

- **original_mesh_from_Yann_Capdeville_2009_using_Gmsh/** - Original mesh generated using Gmsh

These directories are provided for reference and manual mesh generation but are not required to run the benchmark.
