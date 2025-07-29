cmake_minimum_required(VERSION 3.28)
project(parakeet_cpp)

set(CMAKE_CXX_STANDARD 17)

add_executable(parakeet_cpp main.cpp
        framework.h
        framework.cpp
        framework_def.h
        framework_model.h
        framework_model.cpp
        framework_common.h
        framework_common.cpp
        framework_nn.h
        framework_nn.cpp
        parakeet.cpp
        parakeet.h)

target_include_directories(parakeet_cpp PUBLIC
        /Users/jason/prj/ggml/include
        /Users/jason/prj/ggml/src
        /Users/jason/prj/include
)
target_link_directories(parakeet_cpp PUBLIC /Users/jason/prj/libs/ggml_debug)
target_link_libraries(parakeet_cpp PUBLIC ggml ggml-base ggml-cpu ggml-blas ggml-metal)

