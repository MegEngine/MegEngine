/**
 * \file
 * dnn/test/common/dct_ref.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "test/common/dct_ref.h"
namespace megdnn {
namespace test {
struct FixCase {
    std::vector<int> mask_offset;
    std::vector<int> mask_val;
};
using Param = DctChannelSelectForward::Param;

static inline FixCase get_fix_mask(Param::FastImpl impl) {
    std::vector<int> fix_32_mask_offset{0, 16, 24, 32};
    std::vector<int> fix_32_mask_val{0,  1,  8,  16, 9, 2,  3, 10, 17, 24, 32,
                                     25, 18, 11, 4,  5, 0,  1, 8,  16, 9,  2,
                                     3,  10, 0,  1,  8, 16, 9, 2,  3,  10};
    megdnn_assert(impl == Param::FastImpl::FIX_32_MASK,
                  "only support gen FIX_32_MASK");
    return {fix_32_mask_offset, fix_32_mask_val};
}

CheckerHelper::TensorsConstriant gen_dct_constriant(
        const size_t /* n */, const size_t ic, const size_t ih, const size_t iw,
        const size_t oc, Param param) {
    auto constraint = [=](CheckerHelper::TensorValueArray& tensors_orig) {
        const size_t block = param.dct_block_size;
        const int block_c = param.format == Param::Format::NCHW4 ? 4 : 1;
        megdnn_assert(oc % block_c == 0, "oc mod block_c must == 0");
        std::shared_ptr<DctTestcase> test_case_ptr = DctTestcase::make();
        DctTestcase& test_case = *test_case_ptr.get();
        UniformIntRNG rng(0, 255);
        UniformIntRNG mask_rng(0, 64 / block_c - 1);
        const size_t no_mask_oc = ic * block * block;
        megdnn_assert(ih % block == 0, "%zu mod %zu == 0", ih, block);
        megdnn_assert(iw % block == 0, "%zu mod %zu == 0", iw, block);

        TensorND mask_offset;
        TensorND mask_val;
        std::vector<int>& mask_offset_vec = test_case.mask_offset_vec;
        std::vector<int>& mask_val_vec = test_case.mask_val_vec;
        UniformIntRNG rng_oc(0, oc);
        if (param.fastImpl == Param::FastImpl::FIX_32_MASK) {
            auto fix_32_mask = get_fix_mask(Param::FastImpl::FIX_32_MASK);
            mask_offset_vec = fix_32_mask.mask_offset;
            mask_val_vec = fix_32_mask.mask_val;
            megdnn_assert(oc == 32, "oc must eq 32");
        } else if (no_mask_oc > oc) {
            size_t remain_oc = oc;
            mask_offset_vec.resize(ic + 1);
            mask_val_vec.resize(oc);
            mask_offset_vec[0] = 0;
            for (size_t ic_idx = 0; ic_idx < ic; ++ic_idx) {
                size_t random_len = (int)rng_oc.gen_single_val() * block_c;
                size_t mask_len = (ic_idx == ic - 1) || (remain_oc == 0)
                                          ? remain_oc
                                          : random_len % remain_oc;
                megdnn_assert(mask_len % block_c == 0,
                              "mask_len mod block_c == 0, but %zu mod %d ",
                              mask_len, block_c);
                const size_t oc_idx = mask_offset_vec[ic_idx];
                remain_oc -= mask_len;
                mask_offset_vec[ic_idx + 1] = oc_idx + mask_len;
                for (size_t mask_idx = 0; mask_idx < mask_len; ++mask_idx) {
                    mask_val_vec[oc_idx + mask_idx] =
                            (int)mask_rng.gen_single_val();
                }
            }
        }
        mask_offset = TensorND(mask_offset_vec.data(),
                               {{mask_offset_vec.size()}, dtype::Int32()});
        mask_val = TensorND(mask_val_vec.data(),
                            {{mask_val_vec.size()}, dtype::Int32()});
        if (tensors_orig.size() > 1) {
            megdnn_assert(tensors_orig.size() == 4, "tensors_orig.size() == 4");
            megdnn_assert(mask_offset_vec.size() >= 2,
                          "mask_offset_vec.size() >= 2");
            megdnn_assert(tensors_orig[1].layout == mask_offset.layout,
                          "tensors_orig[1].layout == mask_offset.layout");
            megdnn_assert(tensors_orig[2].layout == mask_val.layout,
                          "tensors_orig[2].layout == mask_val.layout");
            auto naive_handle = create_cpu_handle(2, false);
            megdnn_memcpy_D2D(naive_handle.get(), tensors_orig[1].raw_ptr,
                              mask_offset.raw_ptr,
                              mask_offset.layout.span().dist_byte());
            megdnn_memcpy_D2D(naive_handle.get(), tensors_orig[2].raw_ptr,
                              mask_val.raw_ptr,
                              mask_val.layout.span().dist_byte());
        }
    };
    return constraint;
}

std::shared_ptr<DctTestcase> gen_dct_case(const size_t n, const size_t ic,
                                          const size_t ih, const size_t iw,
                                          const size_t oc, Param param,
                                          DType dst_dtype,
                                          bool correct_result) {
    const size_t block = param.dct_block_size;
    const int block_c = param.format == Param::Format::NCHW4 ? 4 : 1;
    megdnn_assert(oc % block_c == 0, "oc mod block_c must == 0");
    std::shared_ptr<DctTestcase> test_case_ptr = DctTestcase::make();
    DctTestcase& test_case = *test_case_ptr.get();
    UniformIntRNG rng(0, 255);
    UniformIntRNG mask_rng(0, 64 / block_c - 1);
    const size_t input_elements = n * ic * ih * iw;
    const size_t no_mask_oc = ic * block * block;
    megdnn_assert(ih % block == 0, "%zu mod %zu == 0", ih, block);
    megdnn_assert(iw % block == 0, "%zu mod %zu == 0", iw, block);
    std::vector<uint8_t>& inp_vec = test_case.inp_vec;
    inp_vec.resize(input_elements);
    TensorShape input_shape{n, ic, ih, iw};
    for (auto& elm : inp_vec) {
        elm = (uint8_t)rng.gen_single_val();
    }
    auto src = TensorND(inp_vec.data(), {input_shape, dtype::Uint8()});
    TensorND mask_offset;
    TensorND mask_val;
    std::vector<int>& mask_offset_vec = test_case.mask_offset_vec;
    std::vector<int>& mask_val_vec = test_case.mask_val_vec;
    UniformIntRNG rng_oc(0, oc);
    if (param.fastImpl == Param::FastImpl::FIX_32_MASK) {
        auto fix_32_mask = get_fix_mask(Param::FastImpl::FIX_32_MASK);
        mask_offset_vec = fix_32_mask.mask_offset;
        mask_val_vec = fix_32_mask.mask_val;
        megdnn_assert(oc == 32, "oc must eq 32");
    } else if (no_mask_oc > oc) {
        size_t remain_oc = oc;
        mask_offset_vec.resize(ic + 1);
        mask_val_vec.resize(oc);
        mask_offset_vec[0] = 0;
        for (size_t ic_idx = 0; ic_idx < ic; ++ic_idx) {
            size_t random_len = (int)rng_oc.gen_single_val() * block_c;
            size_t mask_len = (ic_idx == ic - 1) || (remain_oc == 0)
                                      ? remain_oc
                                      : random_len % remain_oc;
            megdnn_assert(mask_len % block_c == 0,
                          "mask_len mod block_c == 0, but %zu mod %d ",
                          mask_len, block_c);
            const size_t oc_idx = mask_offset_vec[ic_idx];
            remain_oc -= mask_len;
            mask_offset_vec[ic_idx + 1] = oc_idx + mask_len;
            for (size_t mask_idx = 0; mask_idx < mask_len; ++mask_idx) {
                mask_val_vec[oc_idx + mask_idx] =
                        (int)mask_rng.gen_single_val();
            }
        }
    }
    mask_offset = TensorND(mask_offset_vec.data(),
                           {{mask_offset_vec.size()}, dtype::Int32()});
    mask_val = TensorND(mask_val_vec.data(),
                        {{mask_val_vec.size()}, dtype::Int32()});
    if (mask_offset_vec.size() >= 2) {
        test_case.testcase_in = {
                src, mask_offset, mask_val, {nullptr, {{}, dst_dtype}}};
    } else {
        test_case.testcase_in = {src, {}, {}, {nullptr, {{}, dst_dtype}}};
    }

    auto naive_handle = create_cpu_handle(2, false);
    auto opr_naive = naive_handle->create_operator<DctChannelSelectForward>();
    opr_naive->param() = param;
    using Proxy = OprProxy<DctChannelSelectForward>;
    Proxy naive_proxy;
    TensorLayout temp_dst_layout;
    temp_dst_layout.dtype = dst_dtype;

    TensorLayoutArray layouts{src.layout, mask_offset.layout, mask_val.layout,
                              temp_dst_layout};
    naive_proxy.deduce_layout(opr_naive.get(), layouts);
    const size_t output_elements = layouts[3].total_nr_elems();
    std::vector<float>& output_vec = test_case.output_vec;
    output_vec.resize(output_elements);
    auto dst = TensorND(output_vec.data(), layouts[3]);
    DctTestcase::TensorValueArray testcase_naive;
    testcase_naive.emplace_back(test_case.testcase_in[0]);
    testcase_naive.emplace_back(test_case.testcase_in[1]);
    testcase_naive.emplace_back(test_case.testcase_in[2]);
    testcase_naive.emplace_back(dst);
    if (correct_result) {
        naive_proxy.exec(opr_naive.get(), testcase_naive);
    }
    test_case.testcase_out = {{}, {}, {}, dst};
    return test_case_ptr;
}

}  // namespace test
}  // namespace megdnn
   // vim: syntax=cpp.doxygen