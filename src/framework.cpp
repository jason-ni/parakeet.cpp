//
// Created by jason on 2025/7/23.
//
#include "ggml.h"
#include "ggml-backend-impl.h"
#include "framework.h"
#include "framework_common.h"

namespace ggml_runtime
{
    // checks if the weight tensor can be used with the specified buffer type and device
    static bool weight_buft_supported(ggml_tensor * w, ggml_op op, ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev) {
        GGML_ASSERT(w != nullptr);

        if (op == GGML_OP_NONE) {
            return true;
        }

        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead()*8,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr ctx_ptr { ggml_init(params) };
        if (!ctx_ptr) {
            throw std::runtime_error("failed to create ggml context");
        }
        ggml_context * ctx = ctx_ptr.get();

        ggml_tensor * op_tensor = nullptr;

        switch (op) {
            case GGML_OP_GET_ROWS:
                {
                    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
                    op_tensor = ggml_get_rows(ctx, w, b);
                } break;
            case GGML_OP_MUL_MAT:
                {
                    ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], 512, w->ne[2], w->ne[3]);
                    op_tensor = ggml_mul_mat(ctx, w, b);
                } break;
            /* TODO: don't know how hparams is used here
            case GGML_OP_MUL_MAT_ID:
                {
                    int n_expert_used = hparams.n_expert_used;
                    ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w->ne[0], n_expert_used, 512);
                    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, 512);
                    op_tensor = ggml_mul_mat_id(ctx, w, b, ids);
                } break;
            */
            case GGML_OP_ADD:
                {
                    ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], w->ne[1], w->ne[2], w->ne[3]);
                    op_tensor = ggml_add(ctx, a, w);
                } break;
            case GGML_OP_MUL:
                {
                    ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], w->ne[1], w->ne[2], w->ne[3]);
                    op_tensor = ggml_mul(ctx, a, w);
                } break;
            case GGML_OP_DIV:
                {
                    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, w->ne[0]);
                    op_tensor = ggml_div(ctx, a, w);
                } break;
            /*
            case GGML_OP_ROPE:
                {
                    int n_embd_head = hparams.n_embd_head_v;
                    int n_head = hparams.n_head();
                    ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_head, n_head, 512);
                    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
                    op_tensor = ggml_rope_ext(
                        ctx, a, b, w,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0
                    );

                } break;
            */
            case GGML_OP_SSM_CONV:
                {
                    // FIXME
                    ggml_tensor * conv_x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 12345, w->ne[1], 6789);
                    op_tensor = ggml_ssm_conv(ctx, conv_x, w);
                } break;
            case GGML_OP_SSM_SCAN:
                {
                    // FIXME
                    const int64_t d_state      = w->ne[0];
                    const int64_t d_inner      = w->ne[1];
                    const int64_t n_seq_tokens = 512;
                    const int64_t n_seqs       = 3;
                    ggml_tensor * s  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_state, d_inner, n_seqs);
                    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_seq_tokens, n_seqs);
                    ggml_tensor * dt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_seq_tokens, n_seqs);
                    ggml_tensor * B = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_state, n_seq_tokens, n_seqs);
                    ggml_tensor * C = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_state, n_seq_tokens, n_seqs);
                    ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs);
                    op_tensor = ggml_ssm_scan(ctx, s, x, dt, w, B, C, ids);
                } break;
            case GGML_OP_RWKV_WKV6:
                {
                    // FIXME
                    const int64_t S = 123;
                    const int64_t H = 123;
                    const int64_t n_tokens = 123;
                    const int64_t n_seqs = 123;
                    ggml_tensor  * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, 1, H, n_tokens);
                    ggml_tensor  * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, S, H, n_tokens);
                    ggml_tensor  * r = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, S, H, n_tokens);
                    ggml_tensor  * tf = w;
                    ggml_tensor  * td = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, S, H, n_tokens);
                    ggml_tensor  * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, n_seqs, S, H);
                    op_tensor = ggml_rwkv_wkv6(ctx, k, v, r, tf, td, state);
                } break;
            case GGML_OP_IM2COL:
                {
                    // FIXME: for general usage, how we supposed to set the im2col parameters?
                    //const int n_embd = hparams.n_embd;
                    ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1024, w->ne[1], 1, 1);
                    op_tensor = ggml_im2col(ctx, w, b, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F16);
                } break;
            default:
                GGML_ABORT("%s: missing test for op %s for tensor %s", __func__, ggml_op_name(op), w->name);
        }

        // create a temporary dummy buffer for the weight so that supports_op can check the buffer type
        GGML_ASSERT(w->buffer == nullptr);
        w->buffer = ggml_backend_buft_alloc_buffer(buft, 0);
        bool op_supported = ggml_backend_dev_supports_op(dev, op_tensor);
        ggml_backend_buffer_free(w->buffer);
        w->buffer = nullptr;

        return op_supported;
    }

    // find the first buffer type in the list that can use the tensor
    static ggml_backend_buffer_type_t select_weight_buft(ggml_tensor * tensor, ggml_op op, const buft_list_t & buft_list) {
        GGML_ASSERT(!buft_list.empty());
        for (const auto & cur : buft_list) {
            ggml_backend_dev_t cur_dev = cur.first;
            ggml_backend_buffer_type_t cur_buft = cur.second;
            if (weight_buft_supported(tensor, op, cur_buft, cur_dev)) {
                return cur_buft;
            }
        }
        return nullptr;
    }

    TensorBag::TensorBag()
    {
        tensors = std::vector<ggml_bf_tensor>();
    }

    void TensorBag::add_tensor(ggml_bf_tensor tensor)
    {
        tensors.emplace_back(tensor);
    }

    ggml_bf_tensor TensorBag::get_tensor(const size_t index) const
    {
        GGML_ASSERT(index < tensors.size());
        return tensors[index];
    };

    size_t TensorBag::tensor_count() const
    {
        return tensors.size();
    }

    void TensorBag::set_first_tensor(ggml_bf_tensor tensor)
    {
        if (tensors.empty())
        {
            tensors.emplace_back(tensor);
        }
        else
        {
            tensors[0] = tensor;
        }
    }

    BackendManager::BackendManager(Params params)
    {
        this->params = params;
    }
    BackendManager& BackendManager::get_instance(Params params)
    {
        static BackendManager* instance = nullptr;
        static std::once_flag flag;
        std::call_once(flag, [&]() {
            try {
                instance = new BackendManager(params);
                instance->init_backends();
            } catch (const std::exception& e) {
                GGMLF_LOG_ERROR("Failed to create BackendManager instance: %s\n", e.what());
                throw;
            }
        });
        return *instance;
    }


    void BackendManager::init_backends()
    {
        //TODO: make a global variable for the backend manager

        ggml_time_init();
        //ggml_log_set(g_state.log_callback, g_state.log_callback_user_data);

        // NOTE: copied from llama.cpp, don't know if it's necessary
        // needed to initialize f16 tables
        {
            struct ggml_init_params params = { 0, NULL, false };
            struct ggml_context * ctx = ggml_init(params);
            ggml_free(ctx);
        }

        auto dev_count = ggml_backend_dev_count();
        GGMLF_LOG_INFO("Found %zu devices.\n", dev_count);

        ggml_backend_dev_t dev = nullptr;
        // gpu backend
        if (params.use_gpu)
        {
            int idx = 0;
            for (int i = 0; i < dev_count; i++) {
                ggml_backend_dev_t dev_cur = ggml_backend_dev_get(i);
                GGMLF_LOG_INFO("Device %d: %s\n", i, ggml_backend_dev_name(dev_cur));
                if (ggml_backend_dev_type(dev_cur) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                    if (idx == 0 || idx == params.gpu_device_idx) {
                        dev = dev_cur;
                        auto * buft = ggml_backend_dev_buffer_type(dev);
                        if (buft)
                        {
                            buft_list.emplace_back(dev, buft);
                        }
                    }

                    if (++idx > params.gpu_device_idx) {
                        break;
                    }
                }
            }
            if (dev != nullptr) {
                GGMLF_LOG_INFO("Using GPU backend: %s\n", ggml_backend_dev_name(dev));
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    GGMLF_LOG_ERROR("Failed to initialize GPU backend.\n");
                } else
                {
                    backends.push_back(backend);
                    gpu_backend = backend;
                }
            }

        }

        // ACCEL backends
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                GGMLF_LOG_INFO("Using %s backend\n", ggml_backend_dev_name(dev));
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (!backend) {
                    GGMLF_LOG_INFO("failed to initialize %s backend\n", ggml_backend_dev_name(dev));
                    continue;
                }
                backends.push_back(backend);
            }
        }

        // cpu backend
        ggml_backend_t backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        GGMLF_LOG_INFO("Using CPU backend\n");
        backends.push_back(backend_cpu);

        // CPU Extra
        auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
        auto get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");
        if (get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = get_extra_bufts_fn(cpu_dev);
            while (extra_bufts && *extra_bufts) {
                buft_list.emplace_back(cpu_dev, *extra_bufts);
                ++extra_bufts;
            }
        }
        // CPU
        buft_list.emplace_back(cpu_dev, ggml_backend_cpu_buffer_type());

        for (const auto &buft : buft_list)
        {
            ggml_backend_dev_t dev = buft.first;
            ggml_backend_buffer_type_t buft_type = buft.second;
            GGMLF_LOG_INFO("Buffer type: %s\n", buft_type->iface.get_name(buft_type));
        }

    }

    std::vector<ggml_backend_t> BackendManager::get_backends()
    {
        return backends;
    }

    buft_list_t BackendManager::get_buft_list()
    {
        return buft_list;
    }

    TensorContainer::TensorContainer(buft_list_t buft_list, size_t max_n_tensors)
    {
        this->buft_list = buft_list;
        this->max_n_tensors = max_n_tensors;
        this->temp_ctx = nullptr;
    }

    ggml_context* TensorContainer::get_temp_ctx()
    {
        if (temp_ctx == nullptr)
        {
            ggml_init_params params = {
                /*.mem_size   =*/ max_n_tensors * ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };

            temp_ctx = ggml_init(params);
            if (!temp_ctx) {
                throw std::runtime_error(format("failed to create ggml context"));
            }
        }
        return temp_ctx;
    }

    void TensorContainer::free_temp_ctx()
    {
        if (temp_ctx)
        {
            ggml_free(temp_ctx);
            temp_ctx = nullptr;
        }
    }
    ggml_bf_tensor TensorContainer::get_tensor_by_name(const std::string& name)
    {
        auto it = tensor_lookup.find(name);
        if (it == tensor_lookup.end()) {
            throw std::runtime_error(format("tensor %s not found", name.c_str()));
        }
        return it->second;
    }

    bool TensorContainer::has_tensor_by_name(const std::string& name)
    {
        return tensor_lookup.find(name)!= tensor_lookup.end();
    }

    ggml_bf_tensor TensorContainer::m_create_tensor(
        ggml_tensor* meta,
        ggmlf_tensor tensor_type,
        ggml_op op,
        std::string& name)
    {
        ggml_backend_buffer_type_t buft = select_weight_buft(meta, op, buft_list);
        if (!buft) {
            throw std::runtime_error(format(
                "failed to find a compatible buffer type for tensor %s",
                ggmlf_tensor_info_mapping.at(tensor_type)));
        }
        /*
        GGMLF_LOG_INFO("Using %s buffer for tensor %s\n",
            buft->iface.get_name(buft),
            name.c_str());
            */

        ggml_bf_context bf_ctx = get_ctx_of_buffer_type(buft);
        ggml_tensor * tensor = ggml_dup_tensor(bf_ctx.ctx, meta);
        ggml_set_name(tensor, name.c_str());
        auto bf_tensor = ggml_bf_tensor(tensor, buft);
        tensor_lookup.insert(std::make_pair(name, bf_tensor));
        return bf_tensor;
    }

    void TensorContainer::cache_tensor(std::string name, ggml_bf_tensor tensor)
    {
        tensor_lookup.insert(std::make_pair(name, tensor));
    }

    ggml_bf_tensor TensorContainer::create_tensor_1d(
        std::string name,
        ggmlf_tensor tensor_type,
        ggml_type data_type,
        int64_t ne0)
    {
        ggml_op op = ggmlf_tensor_info_mapping.at(tensor_type);
        ggml_tensor* meta = ggml_new_tensor_1d(get_temp_ctx(), data_type, ne0);
        return m_create_tensor(meta, tensor_type, op, name);
    }

    ggml_bf_tensor TensorContainer::create_tensor_2d(
        std::string name,
        ggmlf_tensor tensor_type,
        ggml_type data_type,
        int64_t ne0,
        int64_t ne1)
    {
        ggml_op op = ggmlf_tensor_info_mapping.at(tensor_type);
        ggml_tensor* meta = ggml_new_tensor_2d(get_temp_ctx(), data_type, ne0, ne1);
        return m_create_tensor(meta, tensor_type, op, name);
    }

    ggml_bf_tensor TensorContainer::create_tensor_3d(
        std::string name,
        ggmlf_tensor tensor_type,
        ggml_type data_type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2)
    {
        ggml_op op = ggmlf_tensor_info_mapping.at(tensor_type);
        ggml_tensor* meta = ggml_new_tensor_3d(get_temp_ctx(), data_type, ne0, ne1, ne2);
        return m_create_tensor(meta, tensor_type, op, name);
    }

    ggml_bf_tensor TensorContainer::create_tensor_4d(
        std::string name,
        ggmlf_tensor tensor_type,
        ggml_type data_type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3)
    {
        ggml_op op = ggmlf_tensor_info_mapping.at(tensor_type);
        ggml_tensor* meta = ggml_new_tensor_4d(get_temp_ctx(), data_type, ne0, ne1, ne2, ne3);
        return m_create_tensor(meta, tensor_type, op, name);
    }

    ggml_bf_context TensorContainer::get_ctx_of_buffer_type(ggml_backend_buffer_type_t buft)
    {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ max_n_tensors * ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                throw std::runtime_error(format("failed to create ggml context"));
            }

            ggml_bf_context bf_ctx = ggml_bf_context(ctx, buft);
            ctx_map.insert(std::make_pair(buft, bf_ctx));

            return bf_ctx;
        }

        return it->second;
    }

    void TensorContainer::allocate_tensors_on_backend_buffers()
    {
        for (auto & p : ctx_map) {
            ggml_backend_buffer_type_t buft = p.first;
            ggml_bf_context bf_ctx = p.second;
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(bf_ctx.ctx, buft);
            if (buf) {
                backend_buffers.emplace_back(buf);

                //size_t size_main = ggml_backend_buffer_get_size(buf);
                //GGMLF_LOG_INFO("%12s total size = %8.2f MB\n", ggml_backend_buffer_name(buf), size_main / 1e6);
            }
        }
    }

    void TensorContainer::dump_tensors()
    {
        for (auto &p : ctx_map)
        {
            ggml_backend_buffer_type_t buft = p.first;
            ggml_bf_context bf_ctx = p.second;
            auto iter_tensor = ggml_get_first_tensor(bf_ctx.ctx);
            while (iter_tensor)
            {
                auto tensor_name = ggml_get_name(iter_tensor);
                if (tensor_name == nullptr)
                {
                    tensor_name = "unknown";
                }
                auto tensor_backend_buf_name = iter_tensor->buffer->buft->iface.get_name(iter_tensor->buffer->buft);
                GGMLF_LOG_INFO("Tensor: %s, %s\n", tensor_name, tensor_backend_buf_name);
                iter_tensor = ggml_get_next_tensor(bf_ctx.ctx, iter_tensor);
            }
        }
    }

    Session::Session(Params params, Module* module, GGUFLoader* gguf_loader)
    {
        this->params = params;
        this->root_module = module;
        this->gguf_loader = gguf_loader;
    }

    static bool ggml_graph_compute_helper(
      ggml_backend_sched_t   sched,
        struct ggml_cgraph * graph,
                       int   n_threads) {

        for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
            ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;

            auto * fn_set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (fn_set_n_threads) {
                fn_set_n_threads(backend, n_threads);
            }
        }

        bool t = ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS;
        ggml_backend_sched_reset(sched);
        return t;
    }

    void Session::init_schedule()
    {
        auto & sched = this->sched;
        auto & meta = sched_meta;

        auto n_tensors = root_module->tensor_count() + 64;
        sched = ggml_backend_sched_new(backends.data(), nullptr, backends.size(), n_tensors, false, false);
        meta.resize(ggml_tensor_overhead() * n_tensors + ggml_graph_overhead());


    }

    void Session::build_graph(TensorBag input_tensors)
    {
        auto & meta = sched_meta;
        struct ggml_init_params params = {
            /*.mem_size   =*/ meta.size(),
            /*.mem_buffer =*/ meta.data(),
            /*.no_alloc   =*/ true,
        };

        struct ggml_context * ctx = ggml_init(params);
        gf = ggml_new_graph_custom(ctx, 4096, false);

        for (size_t i = 0; i < input_tensors.tensor_count(); ++i)
        {
            ggml_bf_tensor bf_tensor = input_tensors.get_tensor(i);
            ggml_set_input(bf_tensor.tensor);
        }

        for (size_t i = 0; i < output_tensors.tensor_count(); ++i)
        {
            ggml_bf_tensor bf_tensor = output_tensors.get_tensor(i);
            ggml_set_output(bf_tensor.tensor);
            ggml_build_forward_expand(gf, bf_tensor.tensor);
        }

        ggml_free(ctx);
        // end of builiding graph
    }


    void Session::run_schedule(TensorBag input_tensors)
    {
        build_graph(input_tensors);

        // allocate graph in the backend
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            // should never happen as we pre-allocate the memory
            GGMLF_LOG_ERROR("Failed to allocate graph\n");
            throw std::runtime_error("failed to allocate graph");
        }

        if (!ggml_graph_compute_helper(sched, gf, 2))
        {
            GGMLF_LOG_ERROR("Failed to compute graph\n");
            throw std::runtime_error("failed to compute graph");
        }

        // reset the scheduler for the next run
        ggml_backend_sched_reset(sched);
    }


    int Session::setup()
    {
        auto bm = BackendManager::get_instance(params);
        buft_list = bm.get_buft_list();
        backends = bm.get_backends();

        model_tensor_container = std::make_unique<TensorContainer>(buft_list, root_module->tensor_count());
        root_module->define_tensors(this);
        model_tensor_container->free_temp_ctx();
        model_tensor_container->allocate_tensors_on_backend_buffers();
        root_module->set_data(this);
        init_schedule();
        return 0;
    }

    void Session::run(
                std::function<TensorBag(Session*, TensorContainer*)> define_input_tensors,
                std::function<void(Session*, TensorContainer*)> set_input_data,
                std::function<void(Session*, TensorBag, TensorContainer*)> return_output)
    {
        auto tensor_max_approx_size = root_module->tensor_count();
        std::unique_ptr<TensorContainer>  session_tensor_container = std::make_unique<TensorContainer>(
            buft_list, tensor_max_approx_size);
        auto input_tensors = define_input_tensors(this, session_tensor_container.get());
        output_tensors = root_module->build_graph(this, input_tensors, session_tensor_container.get());
        session_tensor_container->allocate_tensors_on_backend_buffers();
        session_tensor_container->free_temp_ctx();
        set_input_data(this, session_tensor_container.get());
        auto start_time = std::chrono::high_resolution_clock::now();
        run_schedule(input_tensors);
        auto after_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(after_time - start_time).count();
        GGMLF_LOG_INFO("run_schedule took %lld microseconds\n", duration);
        return_output(this, output_tensors, session_tensor_container.get());
    }


}