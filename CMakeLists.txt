cmake_minimum_required(VERSION 3.28)
project(parakeet_cpp)

set(CMAKE_CXX_STANDARD 17)

add_executable(parakeet_cpp main.cpp
        framework.h
        framework.cpp)

target_include_directories(parakeet_cpp PUBLIC
        /Users/jason/prj/whisper.cpp/ggml/include
        /Users/jason/prj/whisper.cpp/ggml/src
        /Users/jason/prj/whisper.cpp/include
)
target_link_directories(parakeet_cpp PUBLIC /Users/jason/prj/libs/ggml_debug)
target_link_libraries(parakeet_cpp PUBLIC ggml ggml-base ggml-cpu ggml-blas ggml-metal)

