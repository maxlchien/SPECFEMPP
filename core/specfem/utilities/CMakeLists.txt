file(GLOB_RECURSE UTILITIES_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(utilities ${UTILITIES_SOURCE_FILES})
add_library(specfem::utilities ALIAS utilities)

target_link_libraries(
        utilities
        Kokkos::kokkos
        ${BOOST_LIBS}
)
