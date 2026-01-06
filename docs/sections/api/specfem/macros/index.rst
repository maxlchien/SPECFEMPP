.. _specfem_macros:


Macros
======

The macros described in this section are available defined in folder ``core/specfem/macros`` and can be accessed by ``#include "specfem/macros.hpp"``.

These macros provide utilities for the SPECFEMPP, including:

*   :doc:`Material and Interface Definitions <material_definitions>`: Tags and iterators used for template metaprogramming and defining physical properties.
*   :doc:`Data Containers <data_container>`: Helper macros for defining kernel / property data container.
*   :doc:`Point Containers <point_container>`: Helper macros for defining kernel / property of a GLL point.
*   :doc:`Warning Suppression <suppress_warnings>`: Cross-platform macros to suppress specific compiler warnings.
*   :doc:`Configuration <config_strings>`: Macros for handling configuration strings.

.. toctree::
    :maxdepth: 1

    material_definitions.rst
    suppress_warnings.rst
    point_container
    data_container
    config_strings
