file(GLOB_RECURSE KOKKOS_KERNELS_SOURCE_FILES "./*.cpp")

add_library(kokkos_kernels ${KOKKOS_KERNELS_SOURCE_FILES})
add_library(specfem::kokkos_kernels ALIAS kokkos_kernels)

set_target_properties(kokkos_kernels PROPERTIES
        UNITY_BUILD $<NOT:$<BOOL:${__APPLE__}>>
        UNITY_BUILD_BATCH_SIZE $<IF:$<BOOL:${__APPLE__}>,0,4>
)

target_link_libraries(
        kokkos_kernels
        PRIVATE
        assembly
        Kokkos::kokkos
)
