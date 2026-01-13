file(GLOB_RECURSE SOURCE_FILES "./*.cpp" "**/*.cpp")

add_library(shape_functions ${SOURCE_FILES})
add_library(specfem::shape_function ALIAS shape_functions)

target_link_libraries(
        shape_functions
)
