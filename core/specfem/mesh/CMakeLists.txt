file(GLOB_RECURSE MESH_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(mesh ${MESH_SOURCE_FILES})
add_library(specfem::mesh ALIAS mesh)

target_link_libraries(
        mesh
        enumerations
        Kokkos::kokkos
        specfem::macros
        # material_class
        yaml-cpp
        ${BOOST_LIBS}
)
