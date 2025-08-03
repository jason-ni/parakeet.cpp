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

struct tensor_view
{
    enum ggml_type type;
    const void* data;
    int64_t ne[GGML_MAX_DIMS];
    size_t nb[GGML_MAX_DIMS];
    size_t dims;
};

struct tensor_view tensor_sub_slice(const struct tensor_view* view, int64_t last_dim_idx)
{
    size_t dims = view->dims;
    struct tensor_view sub_view{};

    sub_view.type = view->type;
    sub_view.dims = view->dims - 1;

    sub_view.data = (char*)view->data + view->nb[dims-1] * last_dim_idx;
    for (int j=0; j<GGML_MAX_DIMS; j++)
    {
        if (j < dims)
        {
            sub_view.ne[j] = view->ne[j];
        } else
        {
            sub_view.ne[j] = 1;
        }
        sub_view.nb[j] = view->nb[j];
    }

    return sub_view;

}

void print_tensor(char* buf, size_t buf_size, int* buf_offset, const struct tensor_view t_view, int left_blanks)
{
    size_t element_bytes = sizeof(float);
    // TODO: currently only support f32 and f16
    if (t_view.type == GGML_TYPE_F16)
    {
        element_bytes = element_bytes / 2;
    }

    for (int i = 0; i < left_blanks; i++)
    {
        append_str(buf, buf_size, buf_offset, " ");
    }
    append_str(buf, buf_size, buf_offset, "[");
    if (t_view.dims == 1)
    {
        if (t_view.ne[0] > 6)
        {
            for (int i=0; i<3; i++)
            {
                if (t_view.type == GGML_TYPE_F32)
                {
                    append_str(buf, buf_size, buf_offset, "%9.6f, ", *(float*)((char*)t_view.data + i*element_bytes));
                } else
                {
                    append_str(buf, buf_size, buf_offset, "%9.4f, ", ggml_fp16_to_fp32(((uint16_t*)((char*)t_view.data))[i*element_bytes]));
                }
            }
            append_str(buf, buf_size, buf_offset, "... ");
            for (int i=t_view.ne[0]-3; i<t_view.ne[0]; i++)
            {
                if (t_view.type == GGML_TYPE_F32)
                {
                    append_str(buf, buf_size, buf_offset, "%9.6f, ", *(float*)((char*)t_view.data + i*element_bytes));
                } else
                {
                    append_str(buf, buf_size, buf_offset, "%9.4f", ggml_fp16_to_fp32(((uint16_t*)((char*)t_view.data))[i*element_bytes]));
                }
            }
        } else
        {
            for (int i=0; i<t_view.ne[0]; i++)
            {
                if (t_view.type == GGML_TYPE_F32)
                {
                    append_str(buf, buf_size, buf_offset, "%9.6f", *(float*)((char*)t_view.data + i*element_bytes));
                } else
                {
                    append_str(buf, buf_size, buf_offset, "%9.4f", ggml_fp16_to_fp32(((uint16_t*)((char*)t_view.data))[i*element_bytes]));
                }
            }
        }
        append_str(buf, buf_size, buf_offset, "]\n");
    } else
    {
        append_str(buf, buf_size, buf_offset, "\n");
        int64_t last_dim_len = t_view.ne[t_view.dims-1];
        if (last_dim_len > 6)
        {
            for(int i=0; i<3; i++)
            {
                struct tensor_view sub_view = tensor_sub_slice(&t_view, i);
                print_tensor(buf, buf_size, buf_offset, sub_view, left_blanks + 1);
            }
            for(int i=0; i<left_blanks; i++)
            {
                append_str(buf, buf_size, buf_offset, " ");
            }
            append_str(buf, buf_size, buf_offset, "...\n");
            for(int i=last_dim_len-3; i<last_dim_len; i++)
            {
                struct tensor_view sub_view = tensor_sub_slice(&t_view, i);
                print_tensor(buf, buf_size, buf_offset, sub_view, left_blanks + 1);
            }
        } else
        {
            for(int i=0; i<last_dim_len; i++)
            {
                struct tensor_view sub_view = tensor_sub_slice(&t_view, i);
                print_tensor(buf, buf_size, buf_offset, sub_view, left_blanks + 1);
            }
        }
        for(int i=0; i<left_blanks; i++)
        {
            append_str(buf, buf_size, buf_offset, " ");
        }
        append_str(buf, buf_size, buf_offset, "],\n");
    }
}

void ggml_tensor_to_str_summarized(char* buf, size_t size, const struct ggml_tensor* tensor, const void* tensor_data)
{
    int buf_offset = 0;
    // Reset the buffer
    if (size > 0) {
        buf[0] = '\0';
    }
    struct tensor_view t_view{};
    t_view.type = tensor->type;
    t_view.data = tensor_data;
    t_view.dims = ggml_n_dims(tensor);
    for (int i = 0; i < t_view.dims; i++)
    {
        t_view.ne[i] = tensor->ne[i];
        t_view.nb[i] = tensor->nb[i];
    }
    print_tensor(buf, size, &buf_offset, t_view, 0);
    append_str(buf , size, &buf_offset, "shape: [%lld, %lld, %lld, %lld]\n", tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
}

void tensor_data_logging(const char* file, int line, const char* func, const struct ggml_tensor* tensor, const void* tensor_data)
{
    char buffer[4096];
    ggml_tensor_to_str_summarized(buffer, sizeof(buffer), tensor, tensor_data);
    parakeet_log_internal(GGML_LOG_LEVEL_INFO, file, line, func, "tensor data: %s\n", buffer);
}