add_library(ggmlf ${SOURCE_FILES})

target_include_directories(ggmlf PUBLIC ${INCLUDE_DIRS})

target_link_libraries(ggmlf ${GGML_DEPENDENCIES})
