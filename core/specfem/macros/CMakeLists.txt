
add_library(
        specfem_macros INTERFACE
)

target_link_libraries(
        specfem_macros
        INTERFACE
        ${BOOST_LIBS}
)

add_library(specfem::macros ALIAS specfem_macros)

target_compile_definitions(
        specfem_macros
        INTERFACE
        $<$<BOOL:${__GNUC__}>:SPECFEMPP_GNU_BUILD>
        $<$<BOOL:${__APPLE__}>:SPECFEMPP_APPLE_BUILD>
        $<$<BOOL:${__MSVC__}>:SPECFEMPP_MSVC_BUILD>
        $<$<BOOL:${__INTEL_LLVM__}>:SPECFEMPP_INTEL_LLVM_BUILD>
)
