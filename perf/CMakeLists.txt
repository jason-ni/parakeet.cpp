add_executable(perf_conv2d
        ${SOURCE_FILES}
        perf_conv2d.cpp)

target_include_directories(perf_conv2d PRIVATE ${INCLUDE_DIRS})
target_link_libraries(perf_conv2d PRIVATE ${GGML_DEPENDENCIES})