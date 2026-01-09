file(GLOB_RECURSE ALGORITHMS_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(
        algorithms
        ${ALGORITHMS_SOURCE_FILES}
)
add_library(specfem::algorithms ALIAS algorithms)

target_link_libraries(
        algorithms
        Kokkos::kokkos
        jacobian
        point
        assembly
        ${BOOST_LIBS}
)
