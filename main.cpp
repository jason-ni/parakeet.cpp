#include <iostream>
#include <thread>
#include <chrono>
#include <ggml-backend-impl.h>

#include "ggml.h"
#include "whisper.h"
#include "framework.h"

class MyModule: public ggml_runtime::Module
{
public:
    explicit MyModule(const std::string name) : name(name)  {};
    ~MyModule() = default;

    int tensor_count() override { return 3; }
    void define_tensors(ggml_runtime::Session* session) override
    {
        ggml_tensor* t1 = session->model_tensor_container->create_tensor_1d("mymodule.t1", GGMLF_TENSOR_BIAS,  GGML_TYPE_F32, 1);
        ggml_tensor* t2 = session->model_tensor_container->create_tensor_1d("mymodule.t2", GGMLF_TENSOR_BIAS, GGML_TYPE_F32, 1);
    }

    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session,
        ggml_runtime::TensorBag input_tensors,
        ggml_runtime::TensorContainer* session_tensor_container) override
    {
        auto input = input_tensors.get_tensor(0);
        ggml_tensor* t1 = session->model_tensor_container->get_tensor_by_name("mymodule.t1");

        //auto bias = ggml_repeat(ctx, t1, input);
        printf("buld graph, input size: %ld\n", input->ne[0]);
        auto session_ctx = session_tensor_container->get_ctx_of_buffer_type(t1->buffer->buft);
        ggml_tensor* out = ggml_add(session_ctx, input, t1);

        ggml_runtime::TensorBag ret = {};
        ret.add_tensor(out);
        return ret;
    };

    void set_data(ggml_runtime::Session* session) override
    {
        float data = 1.0f;
        ggml_tensor* t1 = session->model_tensor_container->get_tensor_by_name("mymodule.t1");
        ggml_backend_tensor_set(t1, &data, 0, sizeof(float));
    };
private:
    std::string name;
};


int main()
{
    auto params = ggml_runtime::Params();
    params.use_gpu = true;
    params.gpu_device_idx = 0;

    auto model_loader = ggml_runtime::ModelLoader("./model.ggml");
    auto root_module = MyModule("mymodule");
    auto session = ggml_runtime::Session(params, &root_module);

    int64_t input_size = 10;
    std::function<ggml_runtime::TensorBag(ggml_runtime::Session*, ggml_runtime::TensorContainer*)> input_fn = [&](
        ggml_runtime::Session* session,
        ggml_runtime::TensorContainer* session_tensor_container)
    {
        ggml_runtime::TensorBag input_tensors = {};
        printf("input_size: %ld\n", input_size);
        ggml_tensor* input = session_tensor_container->create_tensor_1d("mymodule.input", GGMLF_TENSOR_BIAS, GGML_TYPE_F32, input_size);
        input_tensors.add_tensor(input);
        return input_tensors;
    };
    session.setup();

    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    std::function<void(ggml_runtime::Session*, ggml_runtime::TensorContainer*)> set_data_fn = [&](
        ggml_runtime::Session* session,
        ggml_runtime::TensorContainer* session_tensor_container)
    {
        ggml_tensor* input = session_tensor_container->get_tensor_by_name("mymodule.input");
        ggml_backend_tensor_set(input, data, 0, sizeof(float) * input->ne[0]);
    };


    std::function<void(ggml_runtime::Session*, ggml_runtime::TensorBag)> output_fn = [&](ggml_runtime::Session* session, ggml_runtime::TensorBag output_tensors)
    {
        auto print_array = [](float* arr, int size)
        {
            for (int i = 0; i < size; i++)
            {
                std::cout << arr[i] << " ";
            }
            std::cout << std::endl;
        };
        ggml_tensor* output = output_tensors.get_tensor(0);
        ggml_backend_tensor_get(output, data, 0, sizeof(float) * input_size);
        std::cout << "Output: ";
        print_array(data, input_size);
    };

    session.run(
        input_fn,
        set_data_fn,
        output_fn
        );

    // how to sleep
    for (int i = 0; i < 100000; i++)
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

    return 0;
}
