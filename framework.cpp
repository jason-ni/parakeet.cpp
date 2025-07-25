//
// Created by jason on 2025/7/23.
//
#include "ggml.h"
#include "ggml-backend-impl.h"
#include "framework.h"

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
static void parakeet_log_internal        (ggml_log_level level, const char * file, int line, const char * func, const char * format, ...);
static void parakeet_log_callback_default(ggml_log_level level, const char * text, void * user_data);

#define PARAKEET_LOG_ERROR(...) parakeet_log_internal(GGML_LOG_LEVEL_ERROR, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define PARAKEET_LOG_WARN(...)  parakeet_log_internal(GGML_LOG_LEVEL_WARN , __FILE__, __LINE__, __func__, __VA_ARGS__)
#define PARAKEET_LOG_INFO(...)  parakeet_log_internal(GGML_LOG_LEVEL_INFO , __FILE__, __LINE__, __func__, __VA_ARGS__)

static void parakeet_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
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
static void parakeet_log_internal(
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
        char formatted_buffer[2048];
        snprintf(formatted_buffer, sizeof(formatted_buffer),
                 "%s:%d:<%s> %s", file, line, func, buffer2);
        g_state.log_callback(level, formatted_buffer, g_state.log_callback_user_data);
        delete[] buffer2;
    }
    va_end(args);
}

namespace ggml_runtime
{
    TensorBag::TensorBag()
    {
        tensors = std::vector<ggml_tensor*>();
    }

    void TensorBag::add_tensor(ggml_tensor* tensor)
    {
        tensors.push_back(tensor);
    }

    ggml_tensor* TensorBag::get_tensor(const size_t index) const
    {
        assert(index < tensors.size());
        return tensors[index];
    };

    size_t TensorBag::tensor_count() const
    {
        return tensors.size();
    }

    BackendManager::BackendManager(Params params)
    {
        this->params = params;
    }
    BackendManager& BackendManager::get_instance(Params params)
    {
        static BackendManager* instance = nullptr;
        if (!instance) {
            try {
                instance = new BackendManager(params);
            } catch (const std::exception& e) {
                PARAKEET_LOG_ERROR("Failed to create BackendManager instance: %s\n", e.what());
                throw;
            }
        }
        return *instance;
    }


    void BackendManager::init_backends()
    {
        //TODO: make a global variable for the backend manager

        ggml_time_init();
        ggml_log_set(g_state.log_callback, g_state.log_callback_user_data);

        // NOTE: copied from llama.cpp, don't know if it's necessary
        // needed to initialize f16 tables
        {
            struct ggml_init_params params = { 0, NULL, false };
            struct ggml_context * ctx = ggml_init(params);
            ggml_free(ctx);
        }

        auto dev_count = ggml_backend_dev_count();
        PARAKEET_LOG_INFO("Found %zu devices.\n", dev_count);

        ggml_backend_dev_t dev = nullptr;
        // gpu backend
        if (params.use_gpu)
        {
            int idx = 0;
            for (int i = 0; i < dev_count; i++) {
                ggml_backend_dev_t dev_cur = ggml_backend_dev_get(i);
                PARAKEET_LOG_INFO("Device %d: %s\n", i, ggml_backend_dev_name(dev_cur));
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
                PARAKEET_LOG_INFO("Using GPU backend: %s\n", ggml_backend_dev_name(dev));
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    PARAKEET_LOG_ERROR("Failed to initialize GPU backend.\n");
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
                PARAKEET_LOG_INFO("Using %s backend\n", ggml_backend_dev_name(dev));
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (!backend) {
                    PARAKEET_LOG_INFO("failed to initialize %s backend\n", ggml_backend_dev_name(dev));
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
        PARAKEET_LOG_INFO("Using CPU backend\n");
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
            PARAKEET_LOG_INFO("Buffer type: %s\n", buft_type->iface.get_name(buft_type));
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

    int Session::init_runtime()
    {
        auto bm = BackendManager::get_instance(params);
        bm.init_backends();
        buft_list = bm.get_buft_list();
        backends = bm.get_backends();
        return 0;
    }

    ggml_context* Session::get_ctx_of_buffer_type(ggml_backend_buffer_type_t buft)
    {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            if (n_tensors == 0)
            {
                n_tensors = root_module->tensor_count() + 64;
            }
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors * ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                throw std::runtime_error("failed to create ggml context");
            }

            ctx_map[buft] = ctx;

            return ctx;
        }

        return it->second;
    }

    Session::Session(Params params, Module* module)
    {
        this->params = params;
        this->root_module = module;
    }

    ggml_context* Session::get_cpu_ctx()
    {
        auto cpu_buft = buft_list.back().second;
        return get_ctx_of_buffer_type(cpu_buft);
    }

    ggml_context* Session::get_gpu_ctx()
    {
        auto gpu_buft = buft_list.begin()->second;
        return get_ctx_of_buffer_type(gpu_buft);
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

    void Session::init_schedule(TensorBag input_tensors, TensorBag output_tensors)
    {
        auto & sched = this->sched;
        auto & meta = sched_meta;

        sched = ggml_backend_sched_new(backends.data(), nullptr, backends.size(), root_module->tensor_count(), false);
        meta.resize(ggml_tensor_overhead() * (root_module->tensor_count() + input_tensors.tensor_count() + output_tensors.tensor_count()) + ggml_graph_overhead());

        struct ggml_init_params params = {
            /*.mem_size   =*/ meta.size(),
            /*.mem_buffer =*/ meta.data(),
            /*.no_alloc   =*/ true,
        };

        struct ggml_context * ctx = ggml_init(params);
        ggml_cgraph * gf = ggml_new_graph(ctx);

        for (size_t i = 0; i < input_tensors.tensor_count(); ++i)
        {
            ggml_tensor * tensor = input_tensors.get_tensor(i);
            ggml_set_input(tensor);
        }

        for (size_t i = 0; i < output_tensors.tensor_count(); ++i)
        {
            ggml_tensor * tensor = output_tensors.get_tensor(i);
            ggml_set_output(tensor);
            ggml_build_forward_expand(gf, tensor);
        }

        ggml_free(ctx);
        // end of builiding graph

        // allocate graph in the backend
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            // should never happen as we pre-allocate the memory
            PARAKEET_LOG_ERROR("Failed to allocate graph\n");
            throw std::runtime_error("failed to allocate graph");
        }

        if (!ggml_graph_compute_helper(sched, gf, 1))
        {
            PARAKEET_LOG_ERROR("Failed to compute graph\n");
            throw std::runtime_error("failed to compute graph");
        }


    }


    void Session::run(
                std::function<TensorBag(Session*)> define_input_tensors,
                std::function<void(Session*)> set_input_data,
                std::function<void(Session*, TensorBag)> return_output
                )
    {
        auto input_tensors = define_input_tensors(this);
        root_module->define_tensors(this);
        // allocate tensors in the backend buffers
        for (auto & p : ctx_map) {
            ggml_backend_buffer_type_t buft = p.first;
            ggml_context * ctx = p.second;
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (buf) {
                backend_buffers.emplace_back(buf);

                size_t size_main = ggml_backend_buffer_get_size(buf);
                PARAKEET_LOG_INFO("%12s total size = %8.2f MB\n", ggml_backend_buffer_name(buf), size_main / 1e6);
            }
        }
        set_input_data(this);
        root_module->set_data(this);
        auto output_tensors = root_module->build_graph(this, input_tensors);
        init_schedule(input_tensors, output_tensors);
        return_output(this, output_tensors);
    }

    ModelLoader::ModelLoader(std::string model_path)
    {
        this->model_path = std::move(model_path);
    }

    int ModelLoader::load_model(ggml_context* ctx)
    {
        return 0;
    }

    ModelLoader::~ModelLoader()
    = default;

}