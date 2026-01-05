# Static MPI wrapper library for SPECFEM++
# Provides context-managed MPI initialization and static access to rank/size
add_library(specfem_mpi_new STATIC mpi.cpp)
add_library(specfem::mpi ALIAS specfem_mpi_new)

target_include_directories(specfem_mpi_new
    PUBLIC
        ${CMAKE_SOURCE_DIR}/core
)

# Link MPI if available
if(TARGET MPI::MPI_CXX)
    target_link_libraries(specfem_mpi_new PUBLIC MPI::MPI_CXX)
endif()

# Set MPI_PARALLEL define if MPI is enabled
if(MPI_PARALLEL)
    target_compile_definitions(specfem_mpi_new PUBLIC MPI_PARALLEL)
endif()
