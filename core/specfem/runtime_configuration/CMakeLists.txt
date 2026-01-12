file(GLOB_RECURSE RUNTIME_CONFIGURATION_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(runtime_configuration ${RUNTIME_CONFIGURATION_SOURCE_FILES})
add_library(specfem::runtime_configuration ALIAS runtime_configuration)

target_link_libraries(
        runtime_configuration
        quadrature
        specfem::timescheme
        specfem::receivers
        yaml-cpp
        specfem::periodic_tasks
        writer
        reader
        kokkos_kernels
        specfem::solver
        ${BOOST_LIBS}
)
