#include "megdnn/oprs/nn.h"
#include "test/cuda/fixture.h"

#include "test/common/checker.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, MULTIHEADATTN_FORWARD) {
    using Param = MultiHeadAttnForward::Param;
    Param param;
    param.training = false;
    Checker<MultiHeadAttnForward> checker(handle_cuda(), false);

    auto run = [&](DType d) {
        auto get_weight_len = [&](size_t num_heads, size_t qsize, size_t ksize,
                                  size_t vsize, size_t qproj_size, size_t kproj_size,
                                  size_t vproj_size, size_t oproj_size, bool qbias,
                                  bool kbias, bool vbias, bool obias) {
            size_t weight_len = 0;
            if (qproj_size > 0)
                weight_len += qsize * qproj_size + (qbias ? qproj_size : 0);
            if (kproj_size > 0)
                weight_len += ksize * kproj_size + (kbias ? kproj_size : 0);
            if (vproj_size > 0)
                weight_len += vsize * vproj_size + (vbias ? vproj_size : 0);
            if (oproj_size > 0 and vproj_size > 0)
                weight_len += vproj_size * oproj_size + (obias ? oproj_size : 0);
            else if (oproj_size > 0 and vproj_size == 0)
                weight_len += num_heads * vsize * oproj_size + (obias ? oproj_size : 0);
            return weight_len;
        };

        auto get_smscalar = [&](size_t num_heads, size_t qsize, size_t qproj_size) {
            size_t head_dim = qproj_size ? qsize / num_heads : qsize;
            return 1 / std::sqrt(head_dim);
        };

        auto run_kernel = [&](size_t batch_size, size_t seq_qlen, size_t seq_klen,
                              size_t num_heads, size_t qsize, size_t ksize,
                              size_t vsize, size_t qproj_size, size_t kproj_size,
                              size_t vproj_size, size_t oproj_size, bool qbias,
                              bool kbias, bool vbias, bool obias, bool maybe_cudnn,
                              bool have_mask) {
            param.need_weights = !maybe_cudnn;
            param.num_heads = num_heads;
            param.embeding_size = qsize;
            param.k_size = ksize;
            param.v_size = vsize;
            param.qbias = qbias && qproj_size;
            param.kbias = kbias && kproj_size;
            param.vbias = vbias && vproj_size;
            param.obias = obias && oproj_size;
            param.qproj_size = qproj_size;
            param.kproj_size = kproj_size;
            param.vproj_size = vproj_size;
            param.oproj_size = oproj_size;
            if (param.qproj_size == 0 && param.kproj_size > 0)
                param.embeding_size = param.embeding_size / num_heads;
            if (param.qproj_size > 0 && param.kproj_size == 0)
                param.k_size = param.k_size / num_heads;
            param.sm_scaler = get_smscalar(
                    param.num_heads, param.embeding_size, param.qproj_size);
            size_t weight_len = get_weight_len(
                    param.num_heads, param.embeding_size, param.k_size, param.v_size,
                    param.qproj_size, param.kproj_size, param.vproj_size,
                    param.oproj_size, param.qbias, param.kbias, param.vbias,
                    param.obias);
            TensorShape attn_mask{};
            if (!maybe_cudnn) {
                // attn_weight, cudnn does not calculate attn_weight
                checker.set_bypass(8);
            }
            if (have_mask) {
                param.attn_mask_type =
                        param::MultiHeadAttn::AttnMaskType::USER_DEFINED_MASK;
                param.tensor_combination_type =
                        param::MultiHeadAttn::TensorCombinationType::ONLY_MASK;
                attn_mask = {batch_size * num_heads, seq_qlen, seq_klen};
                checker.set_dtype(4, d);
            } else {
                param.attn_mask_type = param::MultiHeadAttn::AttnMaskType::NO_MASK;
                param.tensor_combination_type =
                        param::MultiHeadAttn::TensorCombinationType::NONE;
            }
            TensorShape query{batch_size, seq_qlen, param.embeding_size};
            TensorShape key{batch_size, seq_klen, param.k_size};
            TensorShape value{batch_size, seq_klen, param.v_size};
            TensorShape weight{weight_len};

            checker.set_param(param).set_bypass(9).set_bypass(10);
            checker.set_dtype(0, d).set_dtype(1, d).set_dtype(2, d).set_dtype(3, d);
            checker.execs(
                    {query, key, value, weight, attn_mask, {}, {}, {}, {}, {}, {}});
        };
        {
            size_t embeding_size = 4;
            size_t k_size = 4;
            size_t v_size = 4;
            for (size_t batch_size : {5})
                for (size_t seq_qlen : {1, 11})
                    for (size_t seq_klen : {1, 12})
                        for (size_t num_heads : {1, 2, 4})
                            for (size_t qproj_size : {0, 4})
                                for (size_t kproj_size : {0, 4})
                                    for (size_t vproj_size : {0, 8})
                                        for (size_t oproj_size : {0, 12})
                                            for (bool maybe_cudnn : {false, true})
                                                for (bool have_mask : {false, true}) {
                                                    run_kernel(
                                                            batch_size, seq_qlen,
                                                            seq_klen, num_heads,
                                                            embeding_size, k_size,
                                                            v_size, qproj_size,
                                                            kproj_size, vproj_size,
                                                            oproj_size, false, false,
                                                            false, false, maybe_cudnn,
                                                            have_mask);
                                                }
        }
        {
            size_t embeding_size = 4;
            size_t k_size = 5;
            size_t v_size = 6;
            size_t qproj_size = 4;
            size_t kproj_size = 4;
            size_t vproj_size = 4;
            size_t oproj_size = 4;
            for (size_t batch_size : {1, 10})
                for (size_t seq_qlen : {1, 11})
                    for (size_t seq_klen : {1, 6, 12})
                        for (size_t num_heads : {1, 2, 4})
                            for (bool qbias : {false, true})
                                for (bool kbias : {false, true})
                                    for (bool vbias : {false, true})
                                        for (bool obias : {false, true}) {
                                            run_kernel(
                                                    batch_size, seq_qlen, seq_klen,
                                                    num_heads, embeding_size, k_size,
                                                    v_size, qproj_size, kproj_size,
                                                    vproj_size, oproj_size, qbias,
                                                    kbias, vbias, obias, false, false);
                                        }
        }
    };

    checker.set_epsilon(1e-4);
    run(dtype::Float32());
    checker.set_epsilon(1e-1);
    run(dtype::Float16());
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
