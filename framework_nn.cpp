//
// Created by jason on 2025/7/27.
//

#include <math.h>
#include "framework_nn.h"
#include <ggml-backend-impl.h>
#include "framework_common.h"

namespace ggml_runtime
{
    int Conv2D::tensor_count()
    {
        return 5;
    }

    void Conv2D::define_tensors(Session* session)
    {
        // it seems for most of cases, f16 is best option.
        // in metal backend, weight f16 is required for ggml_conv_2d_dw
        this->weight = session->model_tensor_container->create_tensor_4d(
            weight_name,
            GGMLF_TENSOR_BIAS,
            GGML_TYPE_F32,
            kernel_size, kernel_size, in_channels, out_channels);

        this->bias = session->model_tensor_container->create_tensor_4d(
            bias_name,
            GGMLF_TENSOR_BIAS,
            GGML_TYPE_F32,
            1, 1, out_channels, 1);
    }

    TensorBag Conv2D::build_graph(Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container)
    {
        auto input_tensor = input_tensors.get_tensor(0);
        ggml_bf_tensor weight_tensor = session->model_tensor_container->get_tensor_by_name(weight_name);
        ggml_bf_tensor bias_tensor = session->model_tensor_container->get_tensor_by_name(bias_name);
        ggml_bf_context bf_ctx = session_tensor_container->get_ctx_of_buffer_type(weight_tensor.buft);
        ggml_tensor* conv2d_ret = ggml_conv_2d(
            bf_ctx.ctx,
            weight_tensor.tensor,
            input_tensor.tensor,
            stride,
            stride,
            padding,
            padding,
            dilation,
            dilation);

        ggml_tensor* output_tensor = ggml_add(bf_ctx.ctx, conv2d_ret, bias_tensor.tensor);
        auto output_tensor_bag = TensorBag();
        output_tensor_bag.add_tensor(ggml_bf_tensor(output_tensor, bf_ctx.buft));
        return output_tensor_bag;
    }

    void Conv2D::set_data(Session* session)
    {
        ggml_bf_tensor weight_tensor = session->model_tensor_container->get_tensor_by_name(weight_name);
        ggml_bf_tensor bias_tensor = session->model_tensor_container->get_tensor_by_name(bias_name);
        ggml_type weight_tensor_type = session->gguf_loader->get_tensor_type(weight_name);
        ggml_type bias_tensor_type = session->gguf_loader->get_tensor_type(bias_name);
        if (weight_tensor_type != GGML_TYPE_F32 || bias_tensor_type != GGML_TYPE_F32)
        {
            GGMLF_LOG_ERROR("conv_2d(%s) parameter %s or %s type is not f32", name.c_str(), weight_name.c_str(), bias_name.c_str());
        }

        auto weight_data_size = ggml_nbytes(weight_tensor.tensor);
        auto tensor_data = session->gguf_loader->get_tensor_file_data(weight_name, weight_data_size);
        ggml_backend_tensor_set(weight_tensor.tensor, tensor_data, 0, weight_data_size);
        //GGMLF_LOG_DATA(weight_tensor.tensor, tensor_data);

        auto bias_data_size = ggml_nbytes(bias_tensor.tensor);
        tensor_data = session->gguf_loader->get_tensor_file_data(bias_name, bias_data_size);
        ggml_backend_tensor_set(bias_tensor.tensor, tensor_data, 0, bias_data_size);
        //GGMLF_LOG_DATA(bias_tensor.tensor, tensor_data);
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

    int Conv2DDW::tensor_count()
    {
        return 16;
    }

    void Conv2DDW::define_tensors(Session* session)
    {
        this->weight = session->model_tensor_container->create_tensor_4d(
            weight_name,
            GGMLF_TENSOR_BIAS,
            GGML_TYPE_F16,
            kernel_size, kernel_size, 1, out_channels);

        this->bias = session->model_tensor_container->create_tensor_4d(
            bias_name,
            GGMLF_TENSOR_BIAS,
            GGML_TYPE_F32,
            1, 1, out_channels, 1);
    }

    TensorBag Conv2DDW::build_graph(Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container)
    {
        auto input_tensor = input_tensors.get_tensor(0);
        ggml_bf_tensor weight_tensor = session->model_tensor_container->get_tensor_by_name(weight_name);
        ggml_bf_tensor bias_tensor = session->model_tensor_container->get_tensor_by_name(bias_name);
        ggml_bf_context bf_ctx = session_tensor_container->get_ctx_of_buffer_type(weight_tensor.buft);

        ggml_tensor* conv2d_ret = ggml_conv_2d_dw(
            bf_ctx.ctx,
            weight_tensor.tensor,
            input_tensor.tensor,
            stride,
            stride,
            padding,
            padding,
            dilation,
            dilation);

        ggml_tensor* output_tensor = ggml_add(bf_ctx.ctx, conv2d_ret, bias_tensor.tensor);
        auto output_tensor_bag = TensorBag();
        output_tensor_bag.add_tensor(ggml_bf_tensor(output_tensor, bf_ctx.buft));
        return output_tensor_bag;
    }

    void Conv2DDW::set_data(Session* session)
    {
        ggml_bf_tensor weight_tensor = session->model_tensor_container->get_tensor_by_name(weight_name);
        ggml_bf_tensor bias_tensor = session->model_tensor_container->get_tensor_by_name(bias_name);
        ggml_type weight_tensor_type = session->gguf_loader->get_tensor_type(weight_name);
        ggml_type bias_tensor_type = session->gguf_loader->get_tensor_type(bias_name);

        //TODO: we may convert weight to fp16 in model file for conv2d dw weight tensors.
        if (weight_tensor_type != GGML_TYPE_F32 || bias_tensor_type != GGML_TYPE_F32)
        {
            GGMLF_LOG_ERROR("conv_2d(%s) parameter %s or %s type is not f32", name.c_str(), weight_name.c_str(), bias_name.c_str());
        }

        auto weight_data_size = ggml_nbytes(weight_tensor.tensor);
        auto tensor_data = session->gguf_loader->get_tensor_file_data(weight_name, weight_data_size * 2);
        std::vector<char> tensor_fp16_data = std::vector<char>(weight_data_size);
        ggml_fp32_to_fp16_row((float*)tensor_data, (ggml_fp16_t*)tensor_fp16_data.data(), weight_data_size / 2);
        ggml_backend_tensor_set(weight_tensor.tensor, tensor_fp16_data.data(), 0, weight_data_size);
        //GGMLF_LOG_DATA(weight_tensor.tensor, tensor_data);

        auto bias_data_size = ggml_nbytes(bias_tensor.tensor);
        tensor_data = session->gguf_loader->get_tensor_file_data(bias_name, bias_data_size);
        ggml_backend_tensor_set(bias_tensor.tensor, tensor_data, 0, bias_data_size);
        //GGMLF_LOG_DATA(bias_tensor.tensor, tensor_data);
    }

    int SequenceModule::tensor_count()
    {
        int count = 0;
        for (auto& module : modules)
        {
            count += module->tensor_count();
        }
        return count;
    }

    void SequenceModule::define_tensors(Session* session)
    {
        for (auto& module : modules)
        {
            module->define_tensors(session);
        }
    }

    TensorBag SequenceModule::build_graph(Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container)
    {
        TensorBag output_tensors;
        for (auto& module : modules)
        {
            output_tensors = module->build_graph(session, input_tensors, session_tensor_container);
            input_tensors = output_tensors;
        }
        return output_tensors;
    }

    void SequenceModule::set_data(Session* session)
    {
        for (auto& module : modules)
        {
            module->set_data(session);
        }
    }

    int Linear::tensor_count()
    {
        return 4;
    }

    void Linear::define_tensors(Session* session)
    {
        this->weight = session->model_tensor_container->create_tensor_4d(
            weight_name,
            GGMLF_TENSOR_OUTPUT,
            GGML_TYPE_F32,
            in_features, out_features, 1, 1);
        if (use_bias)
        {
            this->bias = session->model_tensor_container->create_tensor_4d(
                bias_name,
                GGMLF_TENSOR_BIAS,
                GGML_TYPE_F32,
                out_features, 1, 1, 1);
        }
    }

    TensorBag Linear::build_graph(Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container)
    {
        auto input_tensor = input_tensors.get_tensor(0);
        ggml_bf_tensor weight_tensor = session->model_tensor_container->get_tensor_by_name(weight_name);
        ggml_bf_context bf_ctx = session_tensor_container->get_ctx_of_buffer_type(weight_tensor.buft);

        /*
        GGMLF_LOG_INFO("linear(%s) input tensor shape: %lld, %lld, %lld, %lld\n",
            name.c_str(),
            input_tensor.tensor->ne[0], input_tensor.tensor->ne[1], input_tensor.tensor->ne[2], input_tensor.tensor->ne[3]);
        GGMLF_LOG_INFO("linear(%s) weight tensor ne: %lld, %lld, %lld, %lld\n",
            name.c_str(),
            weight_tensor.tensor->ne[0], weight_tensor.tensor->ne[1], weight_tensor.tensor->ne[2], weight_tensor.tensor->ne[3]);
            */
        ggml_tensor* matmul_ret = ggml_mul_mat(
            bf_ctx.ctx,
            weight_tensor.tensor,
            input_tensor.tensor);

        ggml_tensor* output_tensor = nullptr;
        if (use_bias)
        {
            ggml_bf_tensor bias_tensor = session->model_tensor_container->get_tensor_by_name(bias_name);
            output_tensor = ggml_add(bf_ctx.ctx, matmul_ret, bias_tensor.tensor);
        } else
        {
            output_tensor = matmul_ret;
        }
        auto output_tensor_bag = TensorBag();
        output_tensor_bag.add_tensor(ggml_bf_tensor(output_tensor, bf_ctx.buft));
        return output_tensor_bag;
    }

    void Linear::set_data(Session* session)
    {
        ggml_bf_tensor weight_tensor = session->model_tensor_container->get_tensor_by_name(weight_name);
        ggml_type weight_tensor_type = session->gguf_loader->get_tensor_type(weight_name);
        if (weight_tensor_type != GGML_TYPE_F32)
        {
            GGMLF_LOG_ERROR("linear(%s) parameter %s or %s type is not f32", name.c_str(), weight_name.c_str(), bias_name.c_str());
        }

        auto weight_data_size = ggml_nbytes(weight_tensor.tensor);
        auto tensor_data = session->gguf_loader->get_tensor_file_data(weight_name, weight_data_size);
        ggml_backend_tensor_set(weight_tensor.tensor, tensor_data, 0, weight_data_size);
        //GGMLF_LOG_DATA(weight_tensor.tensor, tensor_data);

        if (use_bias)
        {
            ggml_bf_tensor bias_tensor = session->model_tensor_container->get_tensor_by_name(bias_name);
            ggml_type bias_tensor_type = session->gguf_loader->get_tensor_type(bias_name);
            if (bias_tensor_type != GGML_TYPE_F32)
            {
                GGMLF_LOG_ERROR("linear(%s) parameter %s or %s type is not f32", name.c_str(), weight_name.c_str(), bias_name.c_str());
            }

            auto bias_data_size = ggml_nbytes(bias_tensor.tensor);
            tensor_data = session->gguf_loader->get_tensor_file_data(bias_name, bias_data_size);
            ggml_backend_tensor_set(bias_tensor.tensor, tensor_data, 0, bias_data_size);
            //GGMLF_LOG_DATA(bias_tensor.tensor, tensor_data);
        }

    }

    int RelPositionalEncoding::tensor_count()
    {
        return 32;
    }

    void RelPositionalEncoding::define_tensors(Session* session)
    {
        session->model_tensor_container->create_tensor_1d(
            name + ".div_term",
            GGMLF_TENSOR_OUTPUT,
            GGML_TYPE_F32,
            d_model/2);
        session->model_tensor_container->create_tensor_1d(
            name + ".positions",
            GGMLF_TENSOR_OUTPUT,
            GGML_TYPE_F32,
            2*max_len-1);
        session->model_tensor_container->create_tensor_1d(
            name + ".interleave_1",
            GGMLF_TENSOR_OUTPUT,
            GGML_TYPE_F32,
            2);
        session->model_tensor_container->create_tensor_1d(
            name + ".interleave_2",
            GGMLF_TENSOR_OUTPUT,
            GGML_TYPE_F32,
            2);
    }

    ggml_bf_tensor RelPositionalEncoding::get_pe_tensor(Session* session, TensorContainer* session_tensor_container)
    {
        /*
        if (session->model_tensor_container->has_tensor_by_name(pe_name))
        {
            //GGMLF_LOG_INFO("pe(%s) tensor already exists, skip building\n", pe_name.c_str());
            return session->model_tensor_container->get_tensor_by_name(pe_name);
        } else
        */
        {
            //GGMLF_LOG_INFO("pe(%s) tensor not exists, build it\n", pe_name.c_str());
            ggml_bf_tensor div_term_tensor = session->model_tensor_container->get_tensor_by_name(name + ".div_term");
            ggml_bf_tensor positions_tensor = session->model_tensor_container->get_tensor_by_name(name + ".positions");
            ggml_bf_context bf_ctx = session_tensor_container->get_ctx_of_buffer_type(div_term_tensor.buft);
            auto grid_tensor = ggml_out_prod(
                bf_ctx.ctx,
                div_term_tensor.tensor,
                positions_tensor.tensor);

            auto cos_grid_tensor = ggml_cos(
                bf_ctx.ctx,
                grid_tensor);
            auto sin_grid_tensor = ggml_sin(
                bf_ctx.ctx,
                grid_tensor);

            auto interleave_1_tensor = session->model_tensor_container->get_tensor_by_name(name + ".interleave_1");
            auto interleave_2_tensor = session->model_tensor_container->get_tensor_by_name(name + ".interleave_2");

            auto cos_grid_unsqueeze_tensor = ggml_reshape_4d(
                bf_ctx.ctx,
                cos_grid_tensor,
                1,
                1,
                grid_tensor->ne[0],
                grid_tensor->ne[1]);

            auto sin_grid_unsqueeze_tensor = ggml_reshape_4d(
                bf_ctx.ctx,
                sin_grid_tensor,
                1,
                1,
                grid_tensor->ne[0],
                grid_tensor->ne[1]);

            auto cos_grid_interleave_1_tensor = ggml_out_prod(
                bf_ctx.ctx,
                interleave_1_tensor.tensor,
                cos_grid_unsqueeze_tensor);

            auto sin_grid_interleave_2_tensor = ggml_out_prod(
                bf_ctx.ctx,
                interleave_2_tensor.tensor,
                sin_grid_unsqueeze_tensor);

            auto cos_grid_interleave_1_compact_tensor = ggml_reshape_2d(
                bf_ctx.ctx,
                cos_grid_interleave_1_tensor,
                cos_grid_interleave_1_tensor->ne[0] * cos_grid_interleave_1_tensor->ne[1] * cos_grid_interleave_1_tensor->ne[2] ,
                cos_grid_interleave_1_tensor->ne[3]);

            auto sin_grid_interleave_2_compact_tensor = ggml_reshape_2d(
                bf_ctx.ctx,
                sin_grid_interleave_2_tensor,
                sin_grid_interleave_2_tensor->ne[0] * sin_grid_interleave_2_tensor->ne[1] * sin_grid_interleave_2_tensor->ne[2] ,
                sin_grid_interleave_2_tensor->ne[3]);

            auto pe_tensor = ggml_add_inplace(
                bf_ctx.ctx,
                cos_grid_interleave_1_compact_tensor,
                sin_grid_interleave_2_compact_tensor);

            auto pe_bf_tensor = ggml_bf_tensor(pe_tensor, bf_ctx.buft);
            session->model_tensor_container->cache_tensor(pe_name, pe_bf_tensor);
            return pe_bf_tensor;
        }

    }


    TensorBag RelPositionalEncoding::build_graph(Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container)
    {
        auto input_tensor = input_tensors.get_tensor(0);
        int feature_len = input_tensor.tensor->ne[1];

        auto pe_tensor = get_pe_tensor(session, session_tensor_container);
        auto bf_ctx = session->model_tensor_container->get_ctx_of_buffer_type(pe_tensor.buft);
        auto center_pos = max_len;
        auto start_pos = center_pos - feature_len;
        auto end_pos = center_pos + feature_len - 1;
        auto new_ne0 = pe_tensor.tensor->ne[0];
        auto new_ne1 = end_pos - start_pos;
        auto new_nb1 = input_tensor.tensor->nb[1];
        auto offset = start_pos * input_tensor.tensor->nb[1];
        //GGMLF_LOG_INFO("rel pos encoding(%s) pos_embd tensor shape: %lld, %d, %zu, %zu\n",
        //    name.c_str(), new_ne0, new_ne1, new_nb1, offset);
        auto pos_embd = ggml_view_2d(
            bf_ctx.ctx,
            pe_tensor.tensor,
            new_ne0,
            new_ne1,
            new_nb1,
            offset);
        input_tensors.add_tensor(ggml_bf_tensor(pos_embd, bf_ctx.buft));

        return input_tensors;
    }

    void RelPositionalEncoding::set_data(Session* session)
    {
        auto pe_data_size = d_model * (2*max_len-1) * sizeof(float);
        std::vector<char> buffer(pe_data_size);
        float* data = (float*)buffer.data();
        for (int i = 0; i < d_model / 2; i++)
        {
            int pos = i * 2;
            data[i] = std::exp((double)pos * (-(std::log(10000.0) / (double)d_model)));
        }
        auto div_term_tensor = session->model_tensor_container->get_tensor_by_name(name + ".div_term");
        ggml_backend_tensor_set(div_term_tensor.tensor, buffer.data(), 0, d_model/2 * sizeof(float));
        //GGMLF_LOG_DATA(div_term_tensor.tensor, buffer.data());

        auto positions_tensor = session->model_tensor_container->get_tensor_by_name(name + ".positions");
        for (int i = max_len-1; i > -max_len; i--)
        {
            data[max_len-1 - i] = (float)i;
        }
        ggml_backend_tensor_set(positions_tensor.tensor, buffer.data(), 0, (2*max_len-1) * sizeof(float));
        //GGMLF_LOG_DATA(positions_tensor.tensor, buffer.data());

        auto interleave_1_tensor = session->model_tensor_container->get_tensor_by_name(name + ".interleave_1");
        data[0] = 0.0f;
        data[1] = 1.0f;
        ggml_backend_tensor_set(interleave_1_tensor.tensor, buffer.data(), 0, 2 * sizeof(float));

        auto interleave_2_tensor = session->model_tensor_container->get_tensor_by_name(name + ".interleave_2");
        data[0] = 1.0f;
        data[1] = 0.0f;
        ggml_backend_tensor_set(interleave_2_tensor.tensor, buffer.data(), 0, 2 * sizeof(float));
    }

    int LayerNorm::tensor_count()
    {
        return 4;
    }

    void LayerNorm::define_tensors(Session* session)
    {
        session->model_tensor_container->create_tensor_4d(
            weight_name,
            GGMLF_TENSOR_OUTPUT,
            GGML_TYPE_F32,
            input_shape[0], input_shape[1], input_shape[2], 1);
        session->model_tensor_container->create_tensor_4d(
            bias_name,
            GGMLF_TENSOR_OUTPUT,
            GGML_TYPE_F32,
            input_shape[0], 1, 1, 1);
    }

    TensorBag LayerNorm::build_graph(Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container)
    {
        auto input_tensor = input_tensors.get_tensor(0);
        auto weight_tensor = session->model_tensor_container->get_tensor_by_name(weight_name);
        auto bias_tensor = session->model_tensor_container->get_tensor_by_name(bias_name);

        ggml_bf_context bf_ctx = session_tensor_container->get_ctx_of_buffer_type(weight_tensor.buft);
        auto norm_tensor = ggml_norm(
            bf_ctx.ctx,
            input_tensor.tensor,
            1e-5);
        auto scaled_tensor = ggml_mul(
            bf_ctx.ctx,
            norm_tensor,
            weight_tensor.tensor);
        auto output_tensor = ggml_add(
            bf_ctx.ctx,
            scaled_tensor,
            bias_tensor.tensor);
        auto output_tensor_bag = TensorBag();
        output_tensor_bag.add_tensor(ggml_bf_tensor(output_tensor, bf_ctx.buft));
        return output_tensor_bag;
    }

    void LayerNorm::set_data(Session* session)
    {
        ggml_bf_tensor weight_tensor = session->model_tensor_container->get_tensor_by_name(weight_name);
        ggml_bf_tensor bias_tensor = session->model_tensor_container->get_tensor_by_name(bias_name);
        ggml_type weight_tensor_type = session->gguf_loader->get_tensor_type(weight_name);
        ggml_type bias_tensor_type = session->gguf_loader->get_tensor_type(bias_name);
        if (weight_tensor_type != GGML_TYPE_F32 || bias_tensor_type != GGML_TYPE_F32)
        {
            GGMLF_LOG_ERROR("layer norm(%s) parameter %s or %s type is not f32", name.c_str(), weight_name.c_str(), bias_name.c_str());
        }

        auto weight_data_size = ggml_nbytes(weight_tensor.tensor);
        auto tensor_data = session->gguf_loader->get_tensor_file_data(weight_name, weight_data_size);
        ggml_backend_tensor_set(weight_tensor.tensor, tensor_data, 0, weight_data_size);
        //GGMLF_LOG_DATA(weight_tensor.tensor, tensor_data);

        auto bias_data_size = ggml_nbytes(bias_tensor.tensor);
        tensor_data = session->gguf_loader->get_tensor_file_data(bias_name, bias_data_size);
        ggml_backend_tensor_set(bias_tensor.tensor, tensor_data, 0, bias_data_size);
        //GGMLF_LOG_DATA(bias_tensor.tensor, tensor_data);
    }

    int RelPositionMultiHeadAttention::tensor_count()
    {
        return 16;
    }

    void RelPositionMultiHeadAttention::define_tensors(Session* session)
    {
        session->model_tensor_container->create_tensor_4d(
            pos_bias_u_name,
            GGMLF_TENSOR_OUTPUT,
            GGML_TYPE_F32,
            n_head, d_k, 1, 1);
        session->model_tensor_container->create_tensor_4d(
            pos_bias_v_name,
            GGMLF_TENSOR_OUTPUT,
            GGML_TYPE_F32,
            n_head, d_k, 1, 1);
        linear_q->define_tensors(session);
        linear_k->define_tensors(session);
        linear_v->define_tensors(session);
        linear_pos->define_tensors(session);
        linear_out->define_tensors(session);
    }

    TensorBag RelPositionMultiHeadAttention::build_graph(Session* session, TensorBag input_tensors, TensorContainer* session_tensor_container)
    {
        auto q_tensor = input_tensors.get_tensor(0);
        auto k_tensor = input_tensors.get_tensor(0);
        auto v_tensor = input_tensors.get_tensor(0);
        auto pos_emb_tensor = input_tensors.get_tensor(1);

        auto bf_ctx = session_tensor_container->get_ctx_of_buffer_type(q_tensor.buft);

        auto q_bag = TensorBag();
        q_bag.add_tensor(q_tensor);
        auto k_bag = TensorBag();
        k_bag.add_tensor(k_tensor);
        auto v_bag = TensorBag();
        v_bag.add_tensor(v_tensor);

        auto q_linear_out = linear_q->build_graph(session, q_bag, session_tensor_container);
        auto k_linear_out = linear_k->build_graph(session, k_bag, session_tensor_container);
        auto v_linear_out = linear_v->build_graph(session, v_bag, session_tensor_container);

        auto q_tensor_linear = q_linear_out.get_tensor(0);
        auto q_multi_head = ggml_reshape_4d(
            bf_ctx.ctx,
            q_tensor_linear.tensor,
            d_k,
            n_head,
            q_tensor_linear.tensor->ne[1],
            q_tensor_linear.tensor->ne[2]);
        auto k_tensor_linear = k_linear_out.get_tensor(0);
        auto k_multi_head = ggml_permute(bf_ctx.ctx,
            ggml_reshape_4d(
                bf_ctx.ctx,
                k_tensor_linear.tensor,
                d_k,
                n_head,
                k_tensor_linear.tensor->ne[1],
                k_tensor_linear.tensor->ne[2]),
                0, 2, 1, 3);
        auto v_tensor_linear = v_linear_out.get_tensor(0);
        auto v_multi_head = ggml_permute(bf_ctx.ctx,
            ggml_reshape_4d(
                bf_ctx.ctx,
                v_tensor_linear.tensor,
                d_k,
                n_head,
                v_tensor_linear.tensor->ne[1],
                v_tensor_linear.tensor->ne[2]),
                0, 2, 1, 3);

        auto linear_pos_input_bag = TensorBag();
        auto pos_emb_transpose = ggml_reshape_4d(
            bf_ctx.ctx,
            pos_emb_tensor.tensor,
            pos_emb_tensor.tensor->ne[0],
            1,
            pos_emb_tensor.tensor->ne[1],
            pos_emb_tensor.tensor->ne[2]);
        linear_pos_input_bag.add_tensor(ggml_bf_tensor(pos_emb_transpose, bf_ctx.buft));
        auto pos_linear_out = linear_pos->build_graph(session, linear_pos_input_bag, session_tensor_container);
        auto pos_linear_out_tensor = pos_linear_out.get_tensor(0);
        auto pos_linear_compact = ggml_reshape_3d(
            bf_ctx.ctx,
            pos_linear_out_tensor.tensor,
            pos_linear_out_tensor.tensor->ne[0],
            pos_linear_out_tensor.tensor->ne[2],
            pos_linear_out_tensor.tensor->ne[3]);

        auto out_bag = TensorBag();
        out_bag.add_tensor(ggml_bf_tensor(pos_linear_compact, bf_ctx.buft));

        return out_bag;
    }

    void RelPositionMultiHeadAttention::set_data(Session* session)
    {
        ggml_bf_tensor pos_bias_u_tensor = session->model_tensor_container->get_tensor_by_name(pos_bias_u_name);
        ggml_bf_tensor pos_bias_v_tensor = session->model_tensor_container->get_tensor_by_name(pos_bias_v_name);

        auto pos_bias_u_data_size = ggml_nbytes(pos_bias_u_tensor.tensor);
        auto tensor_data = session->gguf_loader->get_tensor_file_data(pos_bias_u_name, pos_bias_u_data_size);
        ggml_backend_tensor_set(pos_bias_u_tensor.tensor, tensor_data, 0, pos_bias_u_data_size);
        //GGMLF_LOG_DATA(pos_bias_u_tensor.tensor, tensor_data);

        auto pos_bias_v_data_size = ggml_nbytes(pos_bias_v_tensor.tensor);
        tensor_data = session->gguf_loader->get_tensor_file_data(pos_bias_v_name, pos_bias_v_data_size);
        ggml_backend_tensor_set(pos_bias_v_tensor.tensor, tensor_data, 0, pos_bias_v_data_size);
        //GGMLF_LOG_DATA(pos_bias_v_tensor.tensor, tensor_data);

        linear_q->set_data(session);
        linear_k->set_data(session);
        linear_v->set_data(session);
        linear_pos->set_data(session);
        linear_out->set_data(session);
    }
}
