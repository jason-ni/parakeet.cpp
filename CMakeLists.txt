cmake_minimum_required(VERSION 3.26)
project(parakeet_cpp)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# define a variable to hold the list of source files
file(GLOB SOURCE_FILES "src/*.cpp")

set(INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src
)

if(APPLE)
    set(GGML_DEPENDENCIES ggml ggml-base ggml-cpu ggml-blas ggml-metal)
endif()
if(WIN32)
    set(GGML_DEPENDENCIES ggml ggml-base ggml-cpu ggml-cuda)
endif()



add_subdirectory(ggml)
add_subdirectory(src)
add_subdirectory(bin)
add_subdirectory(perf)


