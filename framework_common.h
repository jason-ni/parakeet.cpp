//
// Created by jason on 2025/7/27.
//
#pragma once

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <vector>
#include "ggml.h"

#ifdef __GNUC__
#ifdef __MINGW32__
#define PARAKEET_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define PARAKEET_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define PARAKEET_ATTRIBUTE_FORMAT(...)
#endif

//
// logging
//

PARAKEET_ATTRIBUTE_FORMAT(5, 6)
void parakeet_log_internal        (ggml_log_level level, const char * file, int line, const char * func, const char * format, ...);
void parakeet_log_callback_default(ggml_log_level level, const char * text, void * user_data);
std::string format(const char * fmt, ...);

struct parakeet_global;

#define GGMLF_LOG_ERROR(...) parakeet_log_internal(GGML_LOG_LEVEL_ERROR, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define GGMLF_LOG_WARN(...)  parakeet_log_internal(GGML_LOG_LEVEL_WARN , __FILE__, __LINE__, __func__, __VA_ARGS__)
#define GGMLF_LOG_INFO(...)  parakeet_log_internal(GGML_LOG_LEVEL_INFO , __FILE__, __LINE__, __func__, __VA_ARGS__)

#define GGMLF_LOG_DATA(TENSOR, TENSOR_DATA)  tensor_data_logging(__FILE__, __LINE__, __func__, TENSOR, TENSOR_DATA)

void tensor_data_logging(const char* file, int line, const char* func, const struct ggml_tensor* tensor, const void* tensor_data);
