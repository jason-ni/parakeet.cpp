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
            // TODO: support general subsampling logic
            auto sampling_num = 2;
            auto conv_channels = 256;
            auto kernel_size = 3;
            auto stride = 2;
            auto padding = 1;
            int layer_index = 0;
            int feature_out = 1024;
            int out_length = 16;

            conv = new ggml_runtime::SequenceModule(this->name + ".conv");

            conv->modules.push_back(new ggml_runtime::Conv2D(
                this->name + ".conv." + std::to_string(layer_index),
                1,
                conv_channels,
                kernel_size,
                stride,
                padding));

            layer_index++;
            conv->modules.push_back(new ggml_runtime::ReLU(
                this->name + ".conv." + std::to_string(layer_index)));

            for (int i = 0; i < sampling_num; i++)
            {
                layer_index++;
                conv->modules.push_back(new ggml_runtime::Conv2DDW(
                    this->name + ".conv." + std::to_string(layer_index),
                    conv_channels,
                    conv_channels,
                    kernel_size,
                    stride,
                    padding));

                layer_index++;
                conv->modules.push_back(new ggml_runtime::Conv2D(
                    this->name + ".conv." + std::to_string(layer_index),
                    conv_channels,
                    conv_channels,
                    1,
                    1,
                    0));

                layer_index++;
                conv->modules.push_back(new ggml_runtime::ReLU(
                    this->name + ".conv." + std::to_string(layer_index)));
            }

            out = new ggml_runtime::Linear(
                this->name + ".out",
                conv_channels * out_length,
                feature_out);
        };
        ~SubSampling()
        {
            delete conv;
            delete out;
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
        ggml_runtime::SequenceModule* conv;
        ggml_runtime::Linear* out;
};

class ConformerFeedForward: public ggml_runtime::Module
{
public:
    ConformerFeedForward(
        const std::string& name,
        int d_model,
        int d_ff,
        float dropout_p,
        bool use_bias=true): name(name), d_model(d_model), d_ff(d_ff), dropout_p(dropout_p), use_bias(use_bias)
    {
        name_linear1 = name + ".linear1";
        name_linear2 = name + ".linear2";
        linear1 = new ggml_runtime::Linear(
            name_linear1,
            d_model,
            d_ff,
            use_bias);
        linear2 = new ggml_runtime::Linear(
            name_linear2,
            d_ff,
            d_model,
            use_bias);
    }
    ~ConformerFeedForward()
    {
        delete linear1;
        delete linear2;
    }

    int tensor_count() override;
    void define_tensors(ggml_runtime::Session* session) override;
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session,
        ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* session_tensor_container) override;
    void set_data(ggml_runtime::Session* session) override;

private:
    std::string name;
    std::string name_linear1;
    std::string name_linear2;
    int d_model;
    int d_ff;
    float dropout_p;
    bool use_bias = true;
    ggml_runtime::Linear* linear1;
    ggml_runtime::Linear* linear2;
};

class ConformerConvolution: public ggml_runtime::Module
{
public:
    ConformerConvolution(
        const std::string& name,
        int d_model,
        int kernel_size,
        bool use_bias=true):
    name(name), d_model(d_model), kernel_size(kernel_size), use_bias(use_bias)
    {
        pointwise_conv1 = new ggml_runtime::Conv1D(
            name + ".pointwise_conv1",
            d_model,
            d_model*2,
            1,
            1,
            0,
            1,
            false);
        depthwise_conv = new ggml_runtime::Conv1D(
            name + ".depthwise_conv",
            d_model,
            d_model,
            9,
            1,
            4,
            1,
            false,
            true);

    };
    ~ConformerConvolution()
    {
        delete pointwise_conv1;
        delete depthwise_conv;
    };

    int tensor_count() override;
    void define_tensors(ggml_runtime::Session* session) override;
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session,
        ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* session_tensor_container) override;
    void set_data(ggml_runtime::Session* session) override;

private:
    std::string name;
    int d_model;
    int kernel_size;
    bool use_bias = true;
    ggml_runtime::Conv1D* pointwise_conv1;
    ggml_runtime::Conv1D* depthwise_conv;
};


class ConFormerLayer: public ggml_runtime::Module
{
public:
    ConFormerLayer(const std::string& name, int d_model, bool use_bias=true):
    name(name), d_model(d_model), use_bias(use_bias)
    {
        int64_t input_shape[4] = {d_model, 1, 1, 1};
        norm_feed_forward1 = new ggml_runtime::LayerNorm(
            name + ".norm_feed_forward1",
            input_shape);
        feed_forward1 = new ConformerFeedForward(
            name + ".feed_forward1",
            d_model,
            4096,
            0.1,
            use_bias);
        norm_self_attn = new ggml_runtime::LayerNorm(
            name + ".norm_self_att",
            input_shape);
        self_attn = new ggml_runtime::RelPositionMultiHeadAttention(
            name + ".self_attn",
            8,
            d_model,
            false);
        norm_conv = new ggml_runtime::LayerNorm(
            name + ".norm_conv",
            input_shape);
        conv = new ConformerConvolution(
            name + ".conv",
            d_model,
            9,
            false);
    };
    ~ConFormerLayer()
    {
        delete norm_feed_forward1;
        delete feed_forward1;
        delete norm_self_attn;
        delete self_attn;
        delete norm_conv;
        delete conv;
    };

    int tensor_count() override;
    void define_tensors(ggml_runtime::Session *session) override;
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session *session,
        ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer *session_tensor_container) override;
    void set_data(ggml_runtime::Session *session) override;

private:
    std::string name;
    int d_model;
    bool use_bias = true;
    ggml_runtime::LayerNorm* norm_feed_forward1;
    ConformerFeedForward* feed_forward1;
    ggml_runtime::LayerNorm* norm_self_attn;
    ggml_runtime::RelPositionMultiHeadAttention* self_attn;
    ggml_runtime::LayerNorm* norm_conv;
    ConformerConvolution* conv;
};

class ConFormer: public ggml_runtime::Module
{
public:
    ConFormer(std::string name): name(std::move(name))
    {
        d_model = 1024;
        bool use_bias = false;
        pre_encode = new SubSampling(this->name + ".pre_encode");
        pos_enc = new ggml_runtime::RelPositionalEncoding(this->name + ".pos_enc", 1024, 5000);
        layers_0 = new ConFormerLayer(this->name + ".layers.0", d_model, use_bias);
    };
    ~ConFormer()
    {
        delete pre_encode;
        delete pos_enc;
        delete layers_0;
    };

    int tensor_count() override;
    void define_tensors(ggml_runtime::Session *session) override;
    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session *session,
        ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer *session_tensor_container) override;
    void set_data(ggml_runtime::Session *session) override;

private:
    std::string name;
    int d_model;
    SubSampling* pre_encode;
    ggml_runtime::RelPositionalEncoding* pos_enc;
    ConFormerLayer* layers_0;
};
