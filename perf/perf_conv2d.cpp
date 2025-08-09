//
// Created by jason on 2025/8/9.
//
#include <iostream>
#include <thread>
#include <chrono>
#include <ggml-backend-impl.h>

#include "ggml.h"
#include "framework.h"
#include "framework_common.h"
#include "framework_nn.h"
#include "parakeet.h"

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

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        printf("Usage: %s <gguf_file> <input.data>\n", argv[0]);
        return -1;
    }

    auto model_path = std::string(argv[1]);

    auto gguf_loader = ggml_runtime::GGUFLoader(model_path);
    auto preprocessed_audio_features_path = std::string(argv[2]);

    auto params = ggml_runtime::Params();
    params.use_gpu = true;
    params.gpu_device_idx = 0;
    params.pe_bin_path = "";

    auto root_module = SubSampling("encoder.pre_encode");
    auto session = ggml_runtime::Session(params, &root_module, &gguf_loader);
    session.setup();

    auto input_audio_features = load_audio_input(preprocessed_audio_features_path);

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
            ggml_nbytes(input.tensor));

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
            // write to file
            auto file = ggml_fopen(("output_" + std::to_string(i) + ".bin").c_str(), "wb");
            if (file == nullptr)
            {
                printf("Failed to open file: output_%d.bin\n", i);
                return;
            }
            fwrite(buffer.data(), 1, output_bytes, file);
            fclose(file);
        }
    };

    session.run(input_fn, set_data_fn, output_fn);

    return 0;

}