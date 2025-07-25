#include <iostream>
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
        auto ctx = session->get_gpu_ctx();
        ggml_tensor* t1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_set_name(t1, "mymodule.t1");
        ggml_tensor* t2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_set_name(t2, "mymodule.t2");
    }

    ggml_runtime::TensorBag build_graph(ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors) override
    {
        auto input = input_tensors.get_tensor(0);
        auto ctx = session->get_gpu_ctx();
        ggml_tensor* t1 = ggml_get_tensor(ctx, "mymodule.t1");

        ggml_tensor* out = ggml_add(ctx, input, t1);

        ggml_runtime::TensorBag ret = {};
        ret.add_tensor(out);
        return ret;
    };

    void set_data(ggml_runtime::Session* session) override
    {
        float data = 1.0f;
        ggml_tensor* t1 = ggml_get_tensor(session->get_gpu_ctx(), "mymodule.t1");
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
    session.init_runtime();

    std::function<ggml_runtime::TensorBag(ggml_runtime::Session*)> input_fn = [](ggml_runtime::Session* session)
    {
        ggml_runtime::TensorBag input_tensors = {};
        auto ctx = session->get_cpu_ctx();
        ggml_tensor* input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_set_name(input, "mymodule.input");
        input_tensors.add_tensor(input);
        return input_tensors;
    };

    std::function<void(ggml_runtime::Session*)> set_data_fn = [](ggml_runtime::Session* session)
    {
        auto ctx = session->get_cpu_ctx();
        float data = 3.0f;
        ggml_tensor* input = ggml_get_tensor(ctx, "mymodule.input");
        ggml_backend_tensor_set(input, &data, 0, sizeof(float));
    };

    std::function<void(ggml_runtime::Session*, ggml_runtime::TensorBag)> output_fn = [](ggml_runtime::Session* session, ggml_runtime::TensorBag output_tensors)
    {
        auto ctx = session->get_cpu_ctx();
        ggml_tensor* input = ggml_get_tensor(ctx, "mymodule.input");
        float data;
        ggml_backend_tensor_get(input, &data, 0, sizeof(float));
        std::cout << "Input: " << data << std::endl;

        ggml_tensor* output = output_tensors.get_tensor(0);
        float out_data;
        ggml_backend_tensor_get(output, &out_data, 0, sizeof(float));
        std::cout << "Output: " << out_data << std::endl;
    };

    session.run(
        input_fn,
        set_data_fn,
        output_fn
        );
    return 0;
}
