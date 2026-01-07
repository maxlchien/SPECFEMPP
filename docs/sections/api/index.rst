.. _api_funcs:

API documentation
=================

The documentation is under development while we are restructuring the codebase.
Parts the are being moved to the new structure are directly below this line.

.. note:: For reference, we are keeping a pinned issue on GitHub describing the
          restructuring process and the reasoning behind it.
          See :issue:`190` for more details.

New structure
-------------

.. toctree::
    :maxdepth: 2
    :glob:

    specfem/assembly/index
    specfem/chunk_element/index
    specfem/io/index
    specfem/mpi/index
    specfem/macros/index
    specfem/periodic_tasks/index
    specfem/point/index
    specfem/receivers/index
    specfem/sources/index
    specfem/data_access/index
    specfem/source_time_functions/index
    specfem/chunk_edge/index


Old structure
-------------

.. toctree::
    :maxdepth: 2
    :glob:

    algorithms/index
    enumerations/index
    quadrature/index
    material/index
    mesh/index
    datatypes/index
    execution/index
    operators/index
    medium/index
    kokkos_kernels/index
    coupling_physics/coupled_interface
    timescheme/index
    solver/index
    setup_parameters/index
    macros/index
    parallel_configuration/index
