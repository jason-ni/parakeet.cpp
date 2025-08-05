//
// Created by jason on 2025/7/27.
//
#pragma once

#include <utility>

#include "framework.h"

namespace ggml_runtime
{
    class Conv1D : public Module {
    public:
        Conv1D(
            const std::string& name,
            int in_channels,
            int out_channels,
            int kernel_size,
            int stride = 1,
            int padding = 0,
            int dilation = 1,
            bool use_bias = true,
            bool is_dw = false):
        name(name), in_channels(in_channels), out_channels(out_channels),
        kernel_size(kernel_size), stride(stride), padding(padding), dilation(dilation), use_bias(use_bias), is_dw(is_dw)
        {
            weight_name = this->name + ".weight";
            bias_name = this->name + ".bias";
        }
        ~Conv1D() = default;

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
        bool use_bias;
        bool is_dw;

        ggml_bf_tensor weight = ggml_bf_tensor(nullptr, nullptr);
        ggml_bf_tensor bias = ggml_bf_tensor(nullptr, nullptr);
    };

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

    class ReLU : public Module {
    public:
        ReLU(const std::string& name): name(name) {}
        ~ReLU() = default;

        int tensor_count() override;

        void define_tensors(Session* session) override;

        TensorBag build_graph(Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container) override;

        void set_data(Session* session) override;

    private:
        std::string name;
    };


    class Conv2DDW : public Module {
    public:
        Conv2DDW(
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
        ~Conv2DDW() = default;

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

    class SequenceModule : public Module {
    public:
        SequenceModule(
            const std::string& name): name(name)
        {
            modules = std::vector<Module*>();
        };
        ~SequenceModule()
        {
            for (auto module : modules) {
                delete module;
            }
        }

        std::vector<Module*> modules;

        int tensor_count() override;

        void define_tensors(Session* session) override;
        TensorBag build_graph(Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container) override;

        void set_data(Session* session) override;
    private:
        std::string name;
    };

    class Linear : public Module {
    public:
        Linear(
            const std::string& name,
            int in_features,
            int out_features,
            bool use_bias = true):
        name(name), in_features(in_features), out_features(out_features), use_bias(use_bias)
        {
            weight_name = this->name + ".weight";
            bias_name = this->name + ".bias";
        };
        ~Linear() = default;

        int tensor_count() override;

        void define_tensors(Session* session) override;

        TensorBag build_graph(Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container) override;

        void set_data(Session* session) override;

    private:
        std::string name;
        std::string weight_name;
        std::string bias_name;
        int in_features;
        int out_features;
        bool use_bias = true;

        ggml_bf_tensor weight = ggml_bf_tensor(nullptr, nullptr);
        ggml_bf_tensor bias = ggml_bf_tensor(nullptr, nullptr);
    };

    class RelPositionalEncoding: public Module
    {
    public:
        RelPositionalEncoding(
            const std::string& name,
            int d_model,
            int max_len):
        name(name), d_model(d_model), max_len(max_len)
        {
            pe_name = this->name + ".pe";
        };
        ~RelPositionalEncoding() = default;

        int tensor_count() override;

        void define_tensors(Session* session) override;

        TensorBag build_graph(Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container) override;

        void set_data(Session* session) override;

    private:
        std::string name;
        std::string pe_name;
        int d_model;
        int max_len;

        ggml_bf_tensor get_pe_tensor(Session* session, TensorContainer* session_tensor_container);
    };

    class LayerNorm : public Module
    {
    public:
        LayerNorm(
            const std::string& name,
            const int64_t (&input_shape)[GGML_MAX_DIMS]): name(name)
        {
            for (int i = 0; i < GGML_MAX_DIMS; i++)
            {
                this->input_shape[i] = input_shape[i];
            }
            weight_name = this->name + ".weight";
            bias_name = this->name + ".bias";
        };
        ~LayerNorm() = default;

        int tensor_count() override;

        void define_tensors(Session* session) override;

        TensorBag build_graph(Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container) override;

        void set_data(Session* session) override;

    private:
        std::string name;
        std::string weight_name;
        std::string bias_name;
        int64_t input_shape[GGML_MAX_DIMS] = {0};
    };

    class BatchNorm1d : public Module
    {
    public:
        BatchNorm1d(
            const std::string& name,
            int num_features,
            bool affine = true):
        name(name), num_features(num_features), affine(affine)
        {
            weight_name = this->name + ".weight";
            bias_name = this->name + ".bias";
            running_mean_name = this->name + ".running_mean";
            running_var_name = this->name + ".running_var";
            eps_name = this->name + ".eps";
        };
        ~BatchNorm1d() = default;

        int tensor_count() override;

        void define_tensors(Session* session) override;

        TensorBag build_graph(Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container) override;

        void set_data(Session* session) override;

    private:
        std::string name;
        std::string weight_name;
        std::string bias_name;
        std::string running_mean_name;
        std::string running_var_name;
        std::string eps_name;
        int num_features;
        bool affine;
    };

    class RelPositionMultiHeadAttention : public Module
    {
    public:
        RelPositionMultiHeadAttention(
            const std::string& name,
            int n_head,
            int n_feat,
            bool use_bias = true):
        name(name), n_head(n_head), n_feat(n_feat), use_bias(use_bias)
        {
            d_k = n_feat / n_head;
            pos_bias_u_name = this->name + ".pos_bias_u";
            pos_bias_v_name = this->name + ".pos_bias_v";

            linear_q = new Linear(this->name + ".linear_q", n_feat, n_feat, use_bias);
            linear_k = new Linear(this->name + ".linear_k", n_feat, n_feat, use_bias);
            linear_v = new Linear(this->name + ".linear_v", n_feat, n_feat, use_bias);
            linear_out = new Linear(this->name + ".linear_out", n_feat, n_feat, use_bias);
            linear_pos = new Linear(this->name + ".linear_pos", n_feat, n_feat, use_bias);
        };
        ~RelPositionMultiHeadAttention()
        {
            delete linear_q;
            delete linear_k;
            delete linear_v;
            delete linear_out;
            delete linear_pos;
        };

        int tensor_count()override;

        void define_tensors(Session* session) override;

        TensorBag build_graph(Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container) override;

        void set_data(Session* session) override;

    private:
        std::string name;
        std::string pos_bias_u_name;
        std::string pos_bias_v_name;
        int n_head;
        int n_feat;
        int d_k;
        bool use_bias;
        Linear* linear_q;
        Linear* linear_k;
        Linear* linear_v;
        Linear* linear_out;
        Linear* linear_pos;
    };
}
