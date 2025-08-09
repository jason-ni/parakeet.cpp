add_executable(parakeet_cpp
        ${SOURCE_FILES} main.cpp)

target_include_directories(parakeet_cpp PRIVATE ${INCLUDE_DIRS})
target_link_libraries(parakeet_cpp PRIVATE ${GGML_DEPENDENCIES})
