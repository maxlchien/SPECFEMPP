
CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES
    ${SPECFEM_NIGHTLY_BENCHMARKS_BUILD_DIR}
)

if (SPECFEM_CPU_BUILD)
    set(SPECFEM_NIGHTLY_BENCHMARKS_REMOTE_SERVER "/home/TROMP/SPECFEMPP-benchmarks/nightly_benchmarks/data/benchmarks/cpu")
endif()
if (SPECFEM_GPU_BUILD)
    set(SPECFEM_NIGHTLY_BENCHMARKS_REMOTE_SERVER "/home/TROMP/SPECFEMPP-benchmarks/nightly_benchmarks/data/benchmarks/gpu")
endif()

configure_file(CMakeFiles/nightly_benchmarks.smk.in ${SPECFEM_NIGHTLY_BENCHMARKS_BUILD_DIR}/nightly_benchmarks.smk @ONLY)
configure_file(CMakeFiles/nightly_config.yaml.in ${SPECFEM_NIGHTLY_BENCHMARKS_BUILD_DIR}/nightly_config.yaml @ONLY)
## Copy entire nightly_benchmarks directory
file(COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/nightly_benchmarks
    ${CMAKE_CURRENT_SOURCE_DIR}/run_parser.py
    DESTINATION ${SPECFEM_NIGHTLY_BENCHMARKS_BUILD_DIR}
)

add_subdirectory(dim2/homogeneous-medium-flat-topography)
