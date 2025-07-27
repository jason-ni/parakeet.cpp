//
// Created by jason on 2025/7/27.
//
#include "framework_common.h"

void parakeet_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

struct parakeet_global {
    // We save the log callback globally
    ggml_log_callback log_callback = parakeet_log_callback_default;
    void * log_callback_user_data = nullptr;
};

static parakeet_global g_state;

GGML_ATTRIBUTE_FORMAT(5, 6)
void parakeet_log_internal(
    ggml_log_level level,
    const char * file,
    int line,
    const char * func,
    const char * format, ...) {
    va_list args;
    va_start(args, format);
    char buffer[1024];
    int len = vsnprintf(buffer, 1024, format, args);
    if (len < 1024) {
        char formatted_buffer[2048];
        snprintf(formatted_buffer, sizeof(formatted_buffer),
                 "%s:%d:<%s> %s", file, line, func, buffer);
        g_state.log_callback(level, formatted_buffer, g_state.log_callback_user_data);
    } else {
        char* buffer2 = new char[len+1];
        vsnprintf(buffer2, len+1, format, args);
        buffer2[len] = 0;
        char formatted_buffer[4096];
        snprintf(formatted_buffer, sizeof(formatted_buffer),
                 "%s:%d:<%s> %s", file, line, func, buffer2);
        g_state.log_callback(level, formatted_buffer, g_state.log_callback_user_data);
        delete[] buffer2;
    }
    va_end(args);
}

std::string format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

// Helper to safely append to the buffer
int append_str(char* buf, size_t size, int* offset, const char* fmt, ...) {
    if (*offset >= size) return 0; // Don't write if buffer is already full
    va_list args;
    va_start(args, fmt);
    int written = vsnprintf(buf + *offset, size - *offset, fmt, args);
    va_end(args);
    if (written > 0) {
        *offset += written;
    }
    return written;
}

// Main function to print a ggml_tensor to a string buffer
void ggml_tensor_to_str_summarized(char* buf, size_t size, const struct ggml_tensor* tensor, const void* tensor_data) {
    int offset = 0;
    // Reset the buffer
    if (size > 0) {
        buf[0] = '\0';
    }
    const int n_dims = ggml_n_dims(tensor);

    // This implementation only supports FP32 tensors for simplicity.
    if (tensor->type != GGML_TYPE_F32) {
        append_str(buf, size, &offset, "tensor(unsupported type: %s)", ggml_type_name(tensor->type));
        return;
    }

    append_str(buf, size, &offset, "tensor(\n");

    // Recursive lambda function to print dimensions
    auto print_dim = [&](auto& self, int dim_idx, int* indices) -> void {
        const int MAX_EDGE_ITEMS = 3;
        const int current_dim_size = tensor->ne[dim_idx];
        const bool summarize = current_dim_size > (2 * MAX_EDGE_ITEMS);

        auto print_element_at = [&](int i) {
            indices[dim_idx] = i;
            if (dim_idx > 0) {
                // Recurse to the next dimension
                self(self, dim_idx - 1, indices);
            } else {
                // We are at the innermost dimension, print the float value
                size_t elem_offset = 0;
                for (int d = 0; d < n_dims; ++d) {
                    elem_offset += indices[d] * tensor->nb[d];
                }
                float* data = (float*)((char*)tensor_data + elem_offset);
                append_str(buf, size, &offset, "%7.4f", *data);
            }
        };

        for (int i = 0; i < n_dims - dim_idx - 1; ++i)
        {
            append_str(buf, size, &offset, " ");
        }
        append_str(buf, size, &offset, "[");
        if (dim_idx > 0)
        {
            append_str(buf, size, &offset, "\n");
        }
        if (summarize) {
            // Print first 3 elements
            for (int i = 0; i < MAX_EDGE_ITEMS; ++i) {
                print_element_at(i);
                append_str(buf, size, &offset, ", ");
                if (dim_idx > 0) append_str(buf, size, &offset, "\n");
            }

            append_str(buf, size, &offset, "...");

            // Print last 3 elements
            for (int i = current_dim_size - MAX_EDGE_ITEMS; i < current_dim_size; ++i) {
                append_str(buf, size, &offset, ", ");
                if (dim_idx > 0) append_str(buf, size, &offset, "\n");
                print_element_at(i);
            }
        } else {
            // Print all elements
            for (int i = 0; i < current_dim_size; ++i) {
                print_element_at(i);
                if (i < current_dim_size - 1) {
                    append_str(buf, size, &offset, ", ");
                    if (dim_idx > 0) append_str(buf, size, &offset, "\n");
                    for (int j = 0; j < n_dims - dim_idx; ++j)
                    {
                        append_str(buf, size, &offset, " ");
                    }
                }
            }
        }
        if (dim_idx > 0)
        {
            append_str(buf, size, &offset, "\n");
            for (int i = 0; i < n_dims - dim_idx; ++i)
            {
                append_str(buf, size, &offset, " ");
            }
        }
        append_str(buf, size, &offset, "]");
    };

    if (n_dims > 0) {
        int indices[GGML_MAX_DIMS] = {0};
        // The dimensions in ggml are stored in reverse order of how they are typically interpreted.
        // We start printing from the highest dimension (n_dims - 1).
        print_dim(print_dim, n_dims - 1, indices);
    } else {
        append_str(buf, size, &offset, "[]"); // Scalar or empty
    }

    // Print shape and type
    append_str(buf, size, &offset, ",\n shape: (");
    for (int i = n_dims - 1; i >= 0; --i) {
        append_str(buf, size, &offset, "%d", (int)tensor->ne[i]);
        if (i > 0) {
            append_str(buf, size, &offset, ", ");
        }
    }
    append_str(buf, size, &offset, "), type: %s)", ggml_type_name(tensor->type));
}

void tensor_data_logging(const char* file, int line, const char* func, const struct ggml_tensor* tensor, const void* tensor_data)
{
    char buffer[4096];
    ggml_tensor_to_str_summarized(buffer, sizeof(buffer), tensor, tensor_data);
    parakeet_log_internal(GGML_LOG_LEVEL_INFO, file, line, func, "tensor data: %s\n", buffer);
}