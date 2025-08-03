//
// Created by jason on 2025/7/27.
//
#pragma once

#include <ggml-cpp.h>
#include <string>
#include <vector>
#include <map>
#include <tuple>
#include "ggml.h"

struct llama_file;

namespace ggml_runtime
{
    class GGUFLoader
    {
    public:
        explicit GGUFLoader(const std::string& path);
        ~GGUFLoader() = default;
        const char* get_tensor_file_data(const std::string& tensor_name, size_t size);
        ggml_type get_tensor_type(const std::string& tensor_name);

    private:
        std::string m_path;
        gguf_context_ptr m_context;
        llama_file* m_file;
        std::map<std::string, std::tuple<ggml_type, uint64_t>> m_tensor_infos;
        std::vector<char> m_tensor_buffer;
    };

}