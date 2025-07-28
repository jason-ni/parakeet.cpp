//
// Created by jason on 2025/7/27.
//
#pragma once

#include <utility>

#include "framework.h"
#include "framework_nn.h"

class SubSampling : public ggml_runtime::Module
{
    public:
        SubSampling(std::string  name): name(std::move(name))
        {
            m_conv2d = new ggml_runtime::Conv2D(
                this->name + ".0",
                1,
                256,
                3,
                2,
                1);
            m_relu = new ggml_runtime::ReLU(
                this->name + ".1");
        };
        ~SubSampling()
        {
            delete m_conv2d;
            delete m_relu;
        }

    int tensor_count() override;

    void define_tensors(ggml_runtime::Session *session) override;

    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session *session,
        ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer *session_tensor_container) override;

    void set_data(ggml_runtime::Session *session) override;

    private:
        std::string name;
        ggml_runtime::Conv2D* m_conv2d;
        ggml_runtime::ReLU* m_relu;
};

