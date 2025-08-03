#include <iostream>
#include <thread>
#include <chrono>
#include <ggml-backend-impl.h>

#include "ggml.h"
#include "framework.h"
#include "framework_common.h"
#include "framework_nn.h"
#include "parakeet.h"

class MyModule: public ggml_runtime::Module
{
public:
    explicit MyModule(const std::string name) : name(name)  {};
    ~MyModule() = default;

    int tensor_count() override { return 3; }
    void define_tensors(ggml_runtime::Session* session) override
    {
        session->model_tensor_container->create_tensor_1d("mymodule.t1", GGMLF_TENSOR_BIAS,  GGML_TYPE_F32, 1);
        session->model_tensor_container->create_tensor_1d("mymodule.t2", GGMLF_TENSOR_BIAS, GGML_TYPE_F32, 1);
    }

    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session,
        ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* session_tensor_container) override
    {
        auto input = input_tensors.get_tensor(0);
        auto t1 = session->model_tensor_container->get_tensor_by_name("mymodule.t1");

        //auto bias = ggml_repeat(ctx, t1, input);
        printf("buld graph, input size: %lld\n", input.tensor->ne[0]);
        auto session_ctx = session_tensor_container->get_ctx_of_buffer_type(t1.buft);
        ggml_tensor* out = ggml_add(session_ctx.ctx, input.tensor, t1.tensor);

        ggml_runtime::TensorBag ret = {};
        ret.add_tensor(ggml_runtime::ggml_bf_tensor(out, t1.buft));
        return ret;
    };

    void set_data(ggml_runtime::Session* session) override
    {
        float data = 1.0f;
        auto t1 = session->model_tensor_container->get_tensor_by_name("mymodule.t1");
        ggml_backend_tensor_set(t1.tensor, &data, 0, sizeof(float));
    };
private:
    std::string name;
};

void test_case_1_simple_infer()
{
    auto params = ggml_runtime::Params();
    params.use_gpu = true;
    params.gpu_device_idx = 0;

    auto root_module = MyModule("mymodule");
    auto session = ggml_runtime::Session(params, &root_module, nullptr);

    session.setup();

    int64_t input_size = 10;
    std::function<ggml_runtime::TensorBag(ggml_runtime::Session*, ggml_runtime::TensorContainer*)> input_fn = [&](
        ggml_runtime::Session* session,
        ggml_runtime::TensorContainer* session_tensor_container)
    {
        ggml_runtime::TensorBag input_tensors = {};
        printf("input_size: %lld\n", input_size);
        auto input = session_tensor_container->create_tensor_1d("mymodule.input", GGMLF_TENSOR_BIAS, GGML_TYPE_F32, input_size);
        input_tensors.add_tensor(input);
        return input_tensors;
    };

    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    std::function<void(ggml_runtime::Session*, ggml_runtime::TensorContainer*)> set_data_fn = [&](
        ggml_runtime::Session* session,
        ggml_runtime::TensorContainer* session_tensor_container)
    {
        auto input = session_tensor_container->get_tensor_by_name("mymodule.input");
        ggml_backend_tensor_set(input.tensor, data, 0, sizeof(float) * input.tensor->ne[0]);
    };


    std::function<void(ggml_runtime::Session*, ggml_runtime::TensorBag, ggml_runtime::TensorContainer*)> output_fn = [&](
        ggml_runtime::Session* session, ggml_runtime::TensorBag output_tensors, ggml_runtime::TensorContainer* session_tensor_container)
    {
        auto print_array = [](float* arr, int size)
        {
            for (int i = 0; i < size; i++)
            {
                std::cout << arr[i] << " ";
            }
            std::cout << std::endl;
        };
        auto output = output_tensors.get_tensor(0);
        ggml_backend_tensor_get(output.tensor, data, 0, sizeof(float) * input_size);
        std::cout << "Output: ";
        print_array(data, input_size);
    };

    session.run(
        input_fn,
        set_data_fn,
        output_fn
        );

    for (int i = 0; i < 100; i++)
    {
        input_size = i % 10 + 1;
        for (int j = 0; j < 10; j++)
        {
            data[j] += 1.0f;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        session.run(
            input_fn,
            set_data_fn,
            output_fn
            );
    }
}

std::vector<float> load_audio_input(const std::string& file_path)
{
    std::vector<float> input_data;
    auto file = ggml_fopen(file_path.c_str(), "rb");
    if (file == nullptr)
    {
        printf("Failed to open file: %s\n", file_path.c_str());
        return input_data;
    }
    // seek to the end of the file to get the size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    // resize the vector to the size of the file
    input_data.resize(file_size / sizeof(float));
    // seek back to the beginning of the file
    fseek(file, 0, SEEK_SET);
    // read the data into the vector
    fread(input_data.data(), sizeof(float), input_data.size(), file);
    // close the file
    fclose(file);

    return input_data;
}

int main()
{

    auto gguf_loader = ggml_runtime::GGUFLoader("/Users/jason/models/parakeet-tdt-0.6b-v2-float32.gguf");

    auto params = ggml_runtime::Params();
    params.use_gpu = true;
    params.gpu_device_idx = 0;

    auto root_module = ConFormer("encoder");
    //auto root_module = ggml_runtime::RelPositionalEncoding("pos_enc", 1024, 5000);
    auto session = ggml_runtime::Session(params, &root_module, &gguf_loader);
    session.setup();

    auto input_audio_features = load_audio_input("/Users/jason/prj/parakeet/input.data");

    std::function<ggml_runtime::TensorBag(ggml_runtime::Session*, ggml_runtime::TensorContainer*)> input_fn = [&](
        ggml_runtime::Session* session,
        ggml_runtime::TensorContainer* session_tensor_container)
    {
        ggml_runtime::TensorBag input_tensors = {};
        auto input_fp32 = session_tensor_container->create_tensor_4d(
            "subsampling.input.fp32", GGMLF_TENSOR_BIAS, GGML_TYPE_F32,
            128, 4775, 1, 1);
        //auto input_16 = session_tensor_container->create_tensor_4d(
        //    "subsampling.input.fp16", GGMLF_TENSOR_BIAS, GGML_TYPE_F16,
        //    128, 4775, 1, 1);
        input_tensors.add_tensor(input_fp32);
        //input_tensors.add_tensor(input_16);
        return input_tensors;
    };

    std::function<void(ggml_runtime::Session*, ggml_runtime::TensorContainer*)> set_data_fn = [&](
        ggml_runtime::Session* session,
        ggml_runtime::TensorContainer* session_tensor_container)
    {
        auto input = session_tensor_container->get_tensor_by_name("subsampling.input.fp32");

        ggml_backend_tensor_set(
            input.tensor,
            input_audio_features.data(),
            0,
            sizeof(float) * input_audio_features.size());

        GGMLF_LOG_DATA(input.tensor, input_audio_features.data());
    };

    std::function<void(ggml_runtime::Session*, ggml_runtime::TensorBag, ggml_runtime::TensorContainer*)> output_fn = [&](
        ggml_runtime::Session* session, ggml_runtime::TensorBag output_tensors, ggml_runtime::TensorContainer* session_tensor_container)
    {
        for (int i = 0; i < output_tensors.tensor_count(); i++)
        {
            auto output = output_tensors.get_tensor(i);
            auto output_bytes = ggml_nbytes(output.tensor);
            auto buffer = std::vector<char>(output_bytes);
            ggml_backend_tensor_get(output.tensor, buffer.data(), 0, output_bytes);
            GGMLF_LOG_DATA(output.tensor, buffer.data());
            GGMLF_LOG_INFO("output tensor buft: %s\n", output.buft->iface.get_name(output.buft));
        }
    };

    for (int i = 0; i < 1; i++)
    {
        auto before_time = std::chrono::high_resolution_clock::now();
        session.run(
            input_fn,
            set_data_fn,
            output_fn
            );
        auto after_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(after_time - before_time).count();
        printf("duration: %lld\n", duration);
        //std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    return 0;
}
