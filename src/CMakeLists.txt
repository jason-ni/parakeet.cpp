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
        ../ggml/include
        ../ggml/src
)
target_link_libraries(parakeet_cpp PUBLIC ggml ggml-base ggml-cpu ggml-blas ggml-metal)
