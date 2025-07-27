//
// Created by jason on 2025/7/27.
//
#pragma once

#include "framework.h"

namespace ggml_runtime
{
    class Conv2D : public Module {
    public:
        Conv2D(
            const std::string& name,
            int in_channels,
            int out_channels,
            int kernel_size,
            int stride = 1,
            int padding = 0,
            int dilation = 1):
        name(name), in_channels(in_channels), out_channels(out_channels),
        kernel_size(kernel_size), stride(stride), padding(padding), dilation(dilation)
        {
            weight_name = this->name + ".weight";
            bias_name = this->name + ".bias";
        }
        ~Conv2D() = default;

        int tensor_count() override;

        void define_tensors(Session* session) override;

        TensorBag build_graph(Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container) override;

        void set_data(Session* session) override;

    private:
        std::string name;
        std::string weight_name;
        std::string bias_name;
        int in_channels;
        int out_channels;
        int kernel_size;
        int stride;
        int padding;
        int dilation;

        ggml_bf_tensor weight = ggml_bf_tensor(nullptr, nullptr);
        ggml_bf_tensor bias = ggml_bf_tensor(nullptr, nullptr);
    };

}
