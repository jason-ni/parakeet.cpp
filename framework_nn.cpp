//
// Created by jason on 2025/7/27.
//

#include "framework_nn.h"

#include <ggml-backend-impl.h>

#include "framework_common.h"

namespace ggml_runtime
{
    int Conv2D::tensor_count()
    {
        return 3;
    }

    void Conv2D::define_tensors(Session* session)
    {

        this->weight = session->model_tensor_container->create_tensor_4d(
            weight_name,
            GGMLF_TENSOR_BIAS,
            GGML_TYPE_F32,
            3, 3, 1, 256);
        this->bias = session->model_tensor_container->create_tensor_4d(
            bias_name,
            GGMLF_TENSOR_BIAS,
            GGML_TYPE_F32,
            1, 1, 256, 1);
    }

    TensorBag Conv2D::build_graph(Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container)
    {
        auto input_tensor = input_tensors.get_tensor(0);
        ggml_bf_tensor weight_tensor = session->model_tensor_container->get_tensor_by_name(weight_name);
        ggml_bf_tensor bias_tensor = session->model_tensor_container->get_tensor_by_name(bias_name);
        ggml_bf_context bf_ctx = session_tensor_container->get_ctx_of_buffer_type(weight_tensor.buft);
        auto conv2d_ret = ggml_conv_2d(
            bf_ctx.ctx,
            weight_tensor.tensor,
            input_tensor.tensor,
            stride,
            stride,
            padding,
            padding,
            dilation,
            dilation);
        GGMLF_LOG_INFO("conv2 output size: %lld, %lld, %lld, %lld\n",
            conv2d_ret->ne[0], conv2d_ret->ne[1], conv2d_ret->ne[2], conv2d_ret->ne[3]);

        ggml_tensor* output_tensor = ggml_add(bf_ctx.ctx, conv2d_ret, bias_tensor.tensor);
        auto output_tensor_bag = TensorBag();
        output_tensor_bag.add_tensor(ggml_bf_tensor(output_tensor, bf_ctx.buft));
        return output_tensor_bag;
    }

    void Conv2D::set_data(Session* session)
    {
        ggml_bf_tensor weight_tensor = session->model_tensor_container->get_tensor_by_name(weight_name);
        ggml_bf_tensor bias_tensor = session->model_tensor_container->get_tensor_by_name(bias_name);
        auto weight_data_size = ggml_nbytes(weight_tensor.tensor);
        auto tensor_data = session->gguf_loader->get_tensor_file_data(weight_name, weight_data_size);
        ggml_backend_tensor_set(weight_tensor.tensor, tensor_data, 0, weight_data_size);

        GGMLF_LOG_DATA(weight_tensor.tensor, tensor_data);

        auto bias_data_size = ggml_nbytes(bias_tensor.tensor);
        tensor_data = session->gguf_loader->get_tensor_file_data(bias_name, bias_data_size);
        ggml_backend_tensor_set(bias_tensor.tensor, tensor_data, 0, bias_data_size);
        GGMLF_LOG_DATA(bias_tensor.tensor, tensor_data);
    }

    int ReLU::tensor_count()
    {
        return 1;
    }

    void ReLU::define_tensors(Session* session){}

    TensorBag ReLU::build_graph(Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container)
    {
        auto input_tensor = input_tensors.get_tensor(0);
        ggml_bf_context bf_ctx = session_tensor_container->get_ctx_of_buffer_type(input_tensor.buft);
        ggml_tensor* relu_tensor = ggml_relu_inplace(bf_ctx.ctx, input_tensor.tensor);
        auto output_tensor_bag = TensorBag();
        output_tensor_bag.add_tensor(ggml_bf_tensor(relu_tensor, bf_ctx.buft));
        return output_tensor_bag;
    }

    void ReLU::set_data(Session* session){}

}
