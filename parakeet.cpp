//
// Created by jason on 2025/7/27.
//
#include "parakeet.h"
#include <ggml-backend-impl.h>

#include "framework_common.h"

int SubSampling::tensor_count()
{
    return conv->tensor_count() + out->tensor_count() + 4;
}

void SubSampling::define_tensors(ggml_runtime::Session* session)
{
    conv->define_tensors(session);
    out->define_tensors(session);
}

ggml_runtime::TensorBag SubSampling::build_graph(ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors, ggml_runtime::TensorContainer* session_tensor_container)
{
    auto conv_output = conv->build_graph(session, input_tensors, session_tensor_container);
    auto conv_ret_tensor = conv_output.get_tensor(0);
    auto buft_ctx = session_tensor_container->get_ctx_of_buffer_type(conv_ret_tensor.buft);
    auto transpose_1_2 = ggml_permute(buft_ctx.ctx, conv_ret_tensor.tensor, 0, 2, 1, 3);
    auto const_tensor = ggml_cont(buft_ctx.ctx, transpose_1_2);
    auto reshape_tensor = ggml_reshape_4d(
        buft_ctx.ctx,
        const_tensor,
        const_tensor->ne[0] * const_tensor->ne[1],
        const_tensor->ne[2],
        const_tensor->ne[3],
        1
    );
    GGMLF_LOG_INFO("reshape_tensor shape: %lld %lld %lld %lld\n",
        reshape_tensor->ne[0], reshape_tensor->ne[1], reshape_tensor->ne[2], reshape_tensor->ne[3]);
    ggml_runtime::TensorBag output_tensors;
    output_tensors.add_tensor(ggml_runtime::ggml_bf_tensor(reshape_tensor, buft_ctx.buft));
    return out->build_graph(session, output_tensors, session_tensor_container);
}

void SubSampling::set_data(ggml_runtime::Session* session)
{
    conv->set_data(session);
    out->set_data(session);
}

int ConFormer::tensor_count()
{
    return pre_encode->tensor_count() + 4 + pos_enc->tensor_count();
}

void ConFormer::define_tensors(ggml_runtime::Session* session)
{
    pre_encode->define_tensors(session);
    pos_enc->define_tensors(session);
    layers_0->define_tensors(session);
}

ggml_runtime::TensorBag ConFormer::build_graph(ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors, ggml_runtime::TensorContainer* session_tensor_container)
{
    auto pre_output = pre_encode->build_graph(session, input_tensors, session_tensor_container);
    auto pos_enc_output = pos_enc->build_graph(session, pre_output, session_tensor_container);
    auto output_tensors = layers_0->build_graph(session, pos_enc_output, session_tensor_container);
    return output_tensors;
}

void ConFormer::set_data(ggml_runtime::Session* session)
{
    pre_encode->set_data(session);
    pos_enc->set_data(session);
    layers_0->set_data(session);
}

int ConformerFeedForward::tensor_count()
{
    return 16;
}

void ConformerFeedForward::define_tensors(ggml_runtime::Session* session)
{
    linear1->define_tensors(session);
    linear2->define_tensors(session);
}

ggml_runtime::TensorBag ConformerFeedForward::build_graph(ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors, ggml_runtime::TensorContainer* session_tensor_container)
{
    auto input_tensor = input_tensors.get_tensor(0);
    auto output_tensors = linear1->build_graph(session, input_tensors, session_tensor_container);
    auto linear1_ret_tensor = output_tensors.get_tensor(0);
    auto buft_ctx = session_tensor_container->get_ctx_of_buffer_type(linear1_ret_tensor.buft);
    auto silu_tensor = ggml_silu(buft_ctx.ctx, linear1_ret_tensor.tensor);
    output_tensors.set_first_tensor(ggml_runtime::ggml_bf_tensor(silu_tensor, buft_ctx.buft));
    output_tensors = linear2->build_graph(session, output_tensors, session_tensor_container);
    return output_tensors;
}

void ConformerFeedForward::set_data(ggml_runtime::Session* session)
{
    linear1->set_data(session);
    linear2->set_data(session);
}

int ConFormerLayer::tensor_count()
{
    return norm_feed_forward1->tensor_count() +
        feed_forward1->tensor_count() +
        norm_self_attn->tensor_count() +
        self_attn->tensor_count() + 4;
}

void ConFormerLayer::define_tensors(ggml_runtime::Session* session)
{
    norm_feed_forward1->define_tensors(session);
    feed_forward1->define_tensors(session);
    norm_self_attn->define_tensors(session);
    self_attn->define_tensors(session);
}

ggml_runtime::TensorBag ConFormerLayer::build_graph(ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors, ggml_runtime::TensorContainer* session_tensor_container)
{
    auto input_tensor = input_tensors.get_tensor(0);
    auto pos_emb_tensor = input_tensors.get_tensor(1);
    auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(input_tensor.buft);
    ggml_tensor* x_copy = ggml_dup(bf_ctx.ctx, input_tensor.tensor);
    auto output_tensors = norm_feed_forward1->build_graph(session, input_tensors, session_tensor_container);
    output_tensors = feed_forward1->build_graph(session, output_tensors, session_tensor_container);
    auto feed_forward1_ret_tensor = output_tensors.get_tensor(0);
    auto attn_input_tensor = ggml_add(
        bf_ctx.ctx,
        x_copy,
        ggml_scale(
            bf_ctx.ctx,
            feed_forward1_ret_tensor.tensor,
            0.5));
    output_tensors.set_first_tensor(ggml_runtime::ggml_bf_tensor(attn_input_tensor, input_tensor.buft));
    output_tensors = norm_self_attn->build_graph(session, output_tensors, session_tensor_container);
    output_tensors.add_tensor(pos_emb_tensor);
    output_tensors = self_attn->build_graph(session, output_tensors, session_tensor_container);
    return output_tensors;
}

void ConFormerLayer::set_data(ggml_runtime::Session* session)
{
    norm_feed_forward1->set_data(session);
    feed_forward1->set_data(session);
    norm_self_attn->set_data(session);
    self_attn->set_data(session);
}


