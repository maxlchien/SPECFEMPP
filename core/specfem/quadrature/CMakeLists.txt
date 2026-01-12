file(GLOB_RECURSE QUADRATURE_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(quadrature ${QUADRATURE_SOURCE_FILES})
add_library(specfem::quadrature ALIAS quadrature)

target_link_libraries(
        quadrature
        Kokkos::kokkos
)
