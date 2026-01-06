.. _specfem_shape_function:

``specfem::shape_function``
===========================

.. doxygennamespace:: specfem::shape_function
    :desc-only:

``specfem::shape_function::shape_function``
-------------------------------------------

2D Overload
^^^^^^^^^^^

.. doxygenfunction:: specfem::shape_function::shape_function(const T xi, const T gamma, const int ngod)

3D Overload
^^^^^^^^^^^

.. doxygenfunction:: specfem::shape_function::shape_function(const T xi, const T eta, const T zeta, const int ngod)

``specfem::shape_function::shape_function_derivatives``
-------------------------------------------------------

2D Overload
^^^^^^^^^^^

.. doxygenfunction:: specfem::shape_function::shape_function_derivatives(const T xi, const T gamma, const int ngod)

3D Overload
^^^^^^^^^^^

.. doxygenfunction:: specfem::shape_function::shape_function_derivatives(const T xi, const T eta, const T zeta, const int ngod)
