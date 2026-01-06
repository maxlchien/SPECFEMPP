
``specfem::time_scheme::newmark``
=================================

.. doxygenclass:: specfem::time_scheme::newmark
    :members:


Predictor and Corrector Phase Implementations
---------------------------------------------

The two functions below implement the predictor and corrector phases of the
Newmark time integration scheme. They are called internally by the
:cpp:class:`specfem::time_scheme::newmark` class, and not intended to be used
directly by users, but are documented here for mathematical clarity.

.. doxygenfunction:: specfem::time_scheme::newmark_impl::predictor_phase_impl

.. doxygenfunction:: specfem::time_scheme::newmark_impl::corrector_phase_impl


Implementation Details
----------------------

.. toctree::
    :maxdepth: 1

    newmark_forward
    newmark_combined
