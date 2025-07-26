//
// Created by jason on 2025/7/23.
//
#pragma once

#include <cassert>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <iostream>
#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "framework_def.h"

namespace ggml_runtime
{
    using buft_list_t = std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>;
    using buft_ctx_map_t = std::map<ggml_backend_buffer_type_t, ggml_context*>;

    class TensorBag
    {
        public:
            TensorBag();
            ~TensorBag() = default;

            void add_tensor(ggml_tensor* tensor);

            ggml_tensor* get_first() const;

            ggml_tensor* get_tensor(size_t index) const;

            size_t tensor_count() const;
        private:
            std::vector<ggml_tensor*> tensors;
    };

    class Session;
    class TensorContainer;

    class Module {
    public:

        virtual int tensor_count() = 0;

        // Defines the tensors (weights) for this module.
        virtual void define_tensors(Session* session) = 0;

        // Builds the computation graph for this module.
        virtual TensorBag build_graph(Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container) = 0;

        // Sets the data for the module's tensors.
        virtual void set_data(Session* session) = 0;

    };

    class Conv2D : public Module {
    public:
        Conv2D(const std::string& name, int input_channels, int output_channels, int kernel_size, int stride, int padding);
        ~Conv2D() = default;

        int tensor_count() override;

        void define_tensors(Session* session) override;

        TensorBag build_graph(Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container) override;

        void set_data(Session* session) override;

    private:
        int input_channels;
        int output_channels;
        int kernel_size;
        int stride;
        int padding;

        ggml_tensor* weights;
        ggml_tensor* bias;
        ggml_tensor* output;
    };

    struct Params
    {
        bool use_gpu = false;
        int gpu_device_idx = 0;
    };

    class BackendManager
    {
        public:
            static BackendManager& get_instance(Params params);
            explicit BackendManager(Params params);
            ~BackendManager() = default;
            void init_backends();
            buft_list_t get_buft_list();
            std::vector<ggml_backend_t> get_backends();

        private:
            Params params;
            std::vector<ggml_backend_t> backends;
            ggml_backend_t gpu_backend;
            buft_list_t buft_list;
    };

    class TensorContainer
    {
    public:
        explicit TensorContainer(buft_list_t buft_list, size_t max_n_tensors);
        ~TensorContainer()
        {
            for (auto & p : ctx_map)
            {
                ggml_free(p.second);
            }
            for (auto & b : backend_buffers)
            {
                if (b)
                {
                    ggml_backend_buffer_free(b);
                }
            }
        }

        ggml_context* get_temp_ctx();
        ggml_context* get_ctx_of_buffer_type(ggml_backend_buffer_type_t buft);
        void allocate_tensors_on_backend_buffers();
        void free_temp_ctx();
        ggml_tensor* get_tensor_by_name(const std::string& name);

        ggml_tensor* create_tensor_1d(
            std::string name,
            ggmlf_tensor tensor_type,
            enum ggml_type data_type,
            int64_t ne0);

    private:
        size_t max_n_tensors;
        ggml_context* temp_ctx;
        buft_list_t buft_list;
        buft_ctx_map_t ctx_map;
        std::vector<ggml_backend_buffer_t> backend_buffers; // used to free the buffers when the tensor container is destroyed.
        std::map<std::string, ggml_tensor*> tensor_lookup;
    };


    class Session
    {
        public:
            explicit Session(Params params, Module* module);
            ~Session() = default;

            int setup();

            // session run expect one closure function that takes a ggml_context* and returns a TensorBag,
            // and another closure function that takes a ggml_context* and a TensorBag and returns void.
            // The first closure function is used to build the input tensorbag, and the second closure function is used
            // to return the output in any way you want.
            void run(
                std::function<TensorBag(Session*, TensorContainer*)> define_input_tensors,
                std::function<void(Session*, TensorContainer*)> set_input_data,
                std::function<void(Session*, TensorBag)> return_output
                );

            ggml_context* get_cpu_ctx();
            ggml_context* get_gpu_ctx();

            ggml_context* input_ctx;
            std::unique_ptr<TensorContainer> model_tensor_container;

        private:
            void init_schedule();
            void build_graph(TensorBag input_tensors);
            void run_schedule(TensorBag input_tensors);

            Params params;
            Module* root_module;
            size_t n_tensors;
            buft_list_t buft_list;
            std::vector<ggml_backend_t> backends;

            ggml_backend_sched_t sched;
            std::vector<uint8_t> sched_meta;
            ggml_cgraph * gf;
            TensorBag output_tensors;
    };

    class ModelLoader
    {
        public:
            explicit ModelLoader(std::string model_path);
            ~ModelLoader();

            int load_model(ggml_context* ctx);


        protected:
            std::string model_path;

    };
}

