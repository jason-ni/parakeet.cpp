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
    //GGMLF_LOG_INFO("reshape_tensor shape: %lld %lld %lld %lld\n",
    //    reshape_tensor->ne[0], reshape_tensor->ne[1], reshape_tensor->ne[2], reshape_tensor->ne[3]);
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
    return pre_encode->tensor_count() + 4 + pos_enc->tensor_count() + layers->tensor_count();
}

void ConFormer::define_tensors(ggml_runtime::Session* session)
{
    pre_encode->define_tensors(session);
    pos_enc->define_tensors(session);
    layers->define_tensors(session);
}

ggml_runtime::TensorBag ConFormer::build_graph(ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors, ggml_runtime::TensorContainer* session_tensor_container)
{
    auto pre_output = pre_encode->build_graph(session, input_tensors, session_tensor_container);
    auto pos_enc_output = pos_enc->build_graph(session, pre_output, session_tensor_container);
    auto output_tensors = layers->build_graph(session, pos_enc_output, session_tensor_container);
    auto x = output_tensors.get_tensor(0);
    auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(x.buft);
    x.tensor = ggml_cont(bf_ctx.ctx, ggml_permute(bf_ctx.ctx, x.tensor, 1, 0, 2, 3));
    output_tensors.set_first_tensor(x);
    return output_tensors;
}

void ConFormer::set_data(ggml_runtime::Session* session)
{
    pre_encode->set_data(session);
    pos_enc->set_data(session);
    layers->set_data(session);
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

int ConformerConvolution::tensor_count()
{
    return pointwise_conv1->tensor_count() + depthwise_conv->tensor_count() + batch_norm->tensor_count() + pointwise_conv2->tensor_count() + 4;
}

void ConformerConvolution::define_tensors(ggml_runtime::Session* session)
{
    pointwise_conv1->define_tensors(session);
    depthwise_conv->define_tensors(session);
    batch_norm->define_tensors(session);
    pointwise_conv2->define_tensors(session);
}

ggml_runtime::TensorBag ConformerConvolution::build_graph(ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors, ggml_runtime::TensorContainer* session_tensor_container)
{
    auto input_tensor = input_tensors.get_tensor(0);
    auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(input_tensor.buft);
    auto x_transpose = ggml_cont(bf_ctx.ctx,
        ggml_permute(bf_ctx.ctx, input_tensor.tensor, 1, 0, 2, 3));
    input_tensors.set_first_tensor(ggml_runtime::ggml_bf_tensor(x_transpose, input_tensor.buft));
    auto output_tensors = pointwise_conv1->build_graph(session, input_tensors, session_tensor_container);
    auto x = output_tensors.get_tensor(0);

    // manually inmplement glu
    auto part_a = ggml_view_4d(
        bf_ctx.ctx,
        x.tensor,
        x.tensor->ne[0], x.tensor->ne[1] / 2, x.tensor->ne[2], x.tensor->ne[3],
        x.tensor->nb[1], x.tensor->nb[2], x.tensor->nb[3], 0);
    auto part_b = ggml_view_4d(
        bf_ctx.ctx,
        x.tensor,
        x.tensor->ne[0], x.tensor->ne[1] / 2, x.tensor->ne[2], x.tensor->ne[3],
        x.tensor->nb[1], x.tensor->nb[2], x.tensor->nb[3], x.tensor->nb[1] * (x.tensor->ne[1] / 2));
    auto part_b_sigmoid = ggml_sigmoid(bf_ctx.ctx, part_b);
    auto x_glu = ggml_mul(bf_ctx.ctx, part_a, part_b_sigmoid);
    output_tensors.set_first_tensor(ggml_runtime::ggml_bf_tensor(x_glu, input_tensor.buft));
    output_tensors = depthwise_conv->build_graph(session, output_tensors, session_tensor_container);
    output_tensors = batch_norm->build_graph(session, output_tensors, session_tensor_container);
    x = output_tensors.get_tensor(0);
    x.tensor = ggml_silu(bf_ctx.ctx, x.tensor);
    output_tensors.set_first_tensor(x);
    output_tensors = pointwise_conv2->build_graph(session, output_tensors, session_tensor_container);
    x = output_tensors.get_tensor(0);
    x.tensor = ggml_cont(bf_ctx.ctx, ggml_permute(bf_ctx.ctx, x.tensor, 1, 0, 2, 3));
    output_tensors.set_first_tensor(x);
    return output_tensors;
}

void ConformerConvolution::set_data(ggml_runtime::Session* session)
{
    pointwise_conv1->set_data(session);
    depthwise_conv->set_data(session);
    batch_norm->set_data(session);
    pointwise_conv2->set_data(session);
}

int ConFormerLayer::tensor_count()
{
    return norm_feed_forward1->tensor_count() +
        feed_forward1->tensor_count() +
        norm_self_attn->tensor_count() +
        self_attn->tensor_count() +
        norm_conv->tensor_count() +
        conv->tensor_count() +
        norm_feed_forward2->tensor_count() +
        feed_forward2->tensor_count() +
        norm_out->tensor_count() + 4;
}

void ConFormerLayer::define_tensors(ggml_runtime::Session* session)
{
    norm_feed_forward1->define_tensors(session);
    feed_forward1->define_tensors(session);
    norm_self_attn->define_tensors(session);
    self_attn->define_tensors(session);
    norm_conv->define_tensors(session);
    conv->define_tensors(session);
    norm_feed_forward2->define_tensors(session);
    feed_forward2->define_tensors(session);
    norm_out->define_tensors(session);
}

ggml_runtime::TensorBag ConFormerLayer::build_graph(ggml_runtime::Session* session, ggml_runtime::TensorBag input_tensors, ggml_runtime::TensorContainer* session_tensor_container)
{
    auto input_tensor = input_tensors.get_tensor(0);
    auto pos_emb_tensor = input_tensors.get_tensor(1);
    auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(input_tensor.buft);
    ggml_tensor* x_copy = ggml_dup(bf_ctx.ctx, input_tensor.tensor);
    auto ret_tensors = norm_feed_forward1->build_graph(session, input_tensors, session_tensor_container);
    ret_tensors = feed_forward1->build_graph(session, ret_tensors, session_tensor_container);
    auto feed_forward1_ret_tensor = ret_tensors.get_tensor(0);
    auto attn_input_tensor = ggml_add(
        bf_ctx.ctx,
        x_copy,
        ggml_scale(
            bf_ctx.ctx,
            feed_forward1_ret_tensor.tensor,
            0.5));
    ret_tensors.set_first_tensor(ggml_runtime::ggml_bf_tensor(attn_input_tensor, input_tensor.buft));
    ret_tensors = norm_self_attn->build_graph(session, ret_tensors, session_tensor_container);
    ret_tensors.add_tensor(pos_emb_tensor);
    ret_tensors = self_attn->build_graph(session, ret_tensors, session_tensor_container);
    auto residual = ggml_add(bf_ctx.ctx, attn_input_tensor, ret_tensors.get_tensor(0).tensor);
    ret_tensors.set_first_tensor(ggml_runtime::ggml_bf_tensor(residual, input_tensor.buft));
    ret_tensors = norm_conv->build_graph(session, ret_tensors, session_tensor_container);
    ret_tensors.add_tensor(ggml_runtime::ggml_bf_tensor(residual, input_tensor.buft));
    ret_tensors = conv->build_graph(session, ret_tensors, session_tensor_container);
    auto x = ret_tensors.get_tensor(0);
    residual = ggml_add(bf_ctx.ctx, residual, x.tensor);
    ret_tensors.set_first_tensor(ggml_runtime::ggml_bf_tensor(residual, input_tensor.buft));
    ret_tensors = norm_feed_forward2->build_graph(session, ret_tensors, session_tensor_container);
    ret_tensors = feed_forward2->build_graph(session, ret_tensors, session_tensor_container);
    x = ret_tensors.get_tensor(0);
    residual = ggml_add(bf_ctx.ctx, residual, ggml_scale(bf_ctx.ctx, x.tensor, 0.5));
    ret_tensors.set_first_tensor(ggml_runtime::ggml_bf_tensor(residual, input_tensor.buft));
    ret_tensors = norm_out->build_graph(session, ret_tensors, session_tensor_container);
    auto output_tensors = ggml_runtime::TensorBag();
    output_tensors.add_tensor(ret_tensors.get_tensor(0));
    output_tensors.add_tensor(pos_emb_tensor);
    return output_tensors;
}

void ConFormerLayer::set_data(ggml_runtime::Session* session)
{
    norm_feed_forward1->set_data(session);
    feed_forward1->set_data(session);
    norm_self_attn->set_data(session);
    self_attn->set_data(session);
    norm_conv->set_data(session);
    conv->set_data(session);
    norm_feed_forward2->set_data(session);
    feed_forward2->set_data(session);
    norm_out->set_data(session);
}


