.. _tests:

Continuous Integration (CI)
===========================

As part of our Continuous Integration (CI) process, we run a series of automated
tests to ensure the stability and reliability of the codebase, as well as
automated documentation builds. The primary goal of these tests are to ensure
that the code can be compiled and run accurately on various supported platforms,
and that new changes do not introduce regressions or break existing
functionality. We use a combination of Github Actions and Jenkins to run these
tests, the details of which are described below.

Read the Docs
-------------

.. |rdt_proj| replace:: ``specfem2d_kokkos``

We host documentation on Read the Docs under the project name |rdt_proj|_ . The
documentation is built using the configuration file
:repo-file:`.readthedocs.yml`. The documentation is automatically built on every
push to the repository and on every pull request, to easily identify
documentation issues.

.. _rdt_proj: https://app.readthedocs.org/projects/specfem2d-kokkos/?utm_source=specfem2d-kokkos&utm_content=flyout

Github Actions
--------------

Partial compilation checks and unit tests
+++++++++++++++++++++++++++++++++++++++++

On Github actions, there are two types of tests that are run on every pull
request or push to the repository:

1. **CPU Compilation checks**: The goal is to ensure current push doesn't break
   the compilation. These tests would run on forks of this repository.
   Ultimately, the hope is that end developer commits their changes to local
   fork at regualar intervals which would reduce compilation errors during
   development process.
   The cpu compilation checks are defined in :repo-file:`.github/workflows/compilation.yml`

2. **CPU unit tests**: The tests are run in a serial mode using GNU compilers.
   The goal is to ensure current push doesn't break the unit tests. These tests
   would run on forks of this repository. Ultimately, the hope is that end
   developer commits their changes to local fork at regular intervals which
   would reduce unit test errors during development process.
   The cpu unit tests are defined in :repo-file:`.github/workflows/unittests.yml`

Docker
++++++

We use Docker to give users the ability to easily build and run the code with
extensive configuration options. The Docker configuration file is in the root
directory of the repository as :repo-file:`Dockerfile`. We host two types builds through
github release builds that have a version number and builds based on the
``devel`` branch. The github workflow configuration file is in
:repo-file:`.github/workflows/docker.yml`.


Jenkins - Complete compilation and unit tests
---------------------------------------------

We also have a Jenkins server that runs more exhaustive tests on every
maintainer pull request to the repository and on request from contributors from
forks. If an external contributor would like to run these tests on their pull
request, then a maintainer will have to comment ``please this this`` or ``retest
this please`` on the pull request, to launch these tests. Pull requests can only
be merged if these tests pass (and maintainers approve the pull request).

We run a matrix of compilation and unit tests on various supported compilers and
and options for both CPU and GPU, which are summarized below.

CPU
+++

**Gnu Compiler Collection (GCC)**

This is defined in :repo-file:`.jenkins/gnu_compiler_checks.gvy`

- GCC Versions: 11.5.0, 14.2.1
- Compilation modes: Serial, OpenMP
- SIMD options: Enabled, Disabled

The resulting test combinations are:

.. list-table::
   :header-rows: 1
   :widths: 20 25 20

   * - GCC Version
     - Compilation Mode
     - SIMD Option
   * - 11.5.0
     - Serial
     - Enabled
   * - 11.5.0
     - Serial
     - Disabled
   * - 11.5.0
     - OpenMP
     - Enabled
   * - 11.5.0
     - OpenMP
     - Disabled
   * - 14.2.1
     - Serial
     - Enabled
   * - 14.2.1
     - Serial
     - Disabled
   * - 14.2.1
     - OpenMP
     - Enabled
   * - 14.2.1
     - OpenMP
     - Disabled

**Intel OneAPI Compiler**

This is defined in :repo-file:`.jenkins/intel_compiler_checks.gvy`

- Intel compiler versions: 2024.2.0
- Compilation modes: Serial, OpenMP
- SIMD options: Enabled, Disabled

The resulting test combinations are:

.. list-table::
   :header-rows: 1
   :widths: 20 25 20

   * - Intel Version
     - Compilation Mode
     - SIMD Option
   * - 2024.2.0
     - Serial
     - Enabled
   * - 2024.2.0
     - Serial
     - Disabled
   * - 2024.2.0
     - OpenMP
     - Enabled
   * - 2024.2.0
     - OpenMP
     - Disabled

GPU
+++

**NVIDIA CUDA Compiler (NVCC)**

This is defined in :repo-file:`.jenkins/cuda_compiler_checks.gvy`. Currently the only
architecture that is tested is NVIDIA Ampere (A100).

- CPU Compiler: GNU 11.5.0
- CUDA: :repo-file:`cudatoolkit/11.8`, :repo-file:`cudatoolkit/12.8`
- Compilation modes: Serial, OpenMP
- SIMD options: Enabled, Disabled

The resulting test combinations are:

.. list-table::
   :header-rows: 1
   :widths: 20 25 20

   * - CUDA Version
     - Compilation Mode
     - SIMD Option
   * - 11.8
     - Serial
     - Enabled
   * - 11.8
     - Serial
     - Disabled
   * - 11.8
     - OpenMP
     - Enabled
   * - 11.8
     - OpenMP
     - Disabled
   * - 12.8
     - Serial
     - Enabled
   * - 12.8
     - Serial
     - Disabled
   * - 12.8
     - OpenMP
     - Enabled
   * - 12.8
     - OpenMP
     - Disabled
