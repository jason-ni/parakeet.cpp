//
// Created by jason on 2025/7/27.
//
#include "parakeet.h"
#include <ggml-backend-impl.h>

int SubSampling::tensor_count()
{
    return m_conv2d->tensor_count() + 8 + m_conv2d->tensor_count();
}

void SubSampling::define_tensors(ggml_runtime::Session* session)
{
    m_conv2d->define_tensors(session);
}

ggml_runtime::TensorBag SubSampling::build_graph(ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors, ggml_runtime::TensorContainer* session_tensor_container)
{
    /*
    auto input_tensor_fp32 = input_tensors.get_tensor(0);
    auto input_tensor_f16 = input_tensors.get_tensor(1);
    auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(input_tensor_fp32.buft);
    auto input_f16 = ggml_cpy(bf_ctx.ctx, input_tensor_f16.tensor, input_tensor_fp32.tensor);

    ggml_runtime::TensorBag output_tensors = ggml_runtime::TensorBag();
    output_tensors.add_tensor(ggml_runtime::ggml_bf_tensor(input_f16, bf_ctx.buft));
    */

    auto conv2d_output_tensors = m_conv2d->build_graph(session, input_tensors, session_tensor_container);
    //return conv2d_output_tensors;
    auto relu_output_tensors = m_relu->build_graph(session, conv2d_output_tensors, session_tensor_container);
    return relu_output_tensors;
}

void SubSampling::set_data(ggml_runtime::Session* session)
{
    m_conv2d->set_data(session);
}

