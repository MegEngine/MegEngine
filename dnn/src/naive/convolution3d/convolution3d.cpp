/**
 * \file dnn/src/naive/convolution3d/convolution3d.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "./helper.h"
#include "./opr_impl.h"

#include "megdnn/dtype.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"

#include <cstring>

#include "midout.h"
MIDOUT_DECL(megdnn_naive_conv3d_fwd)

using namespace megdnn;
using namespace naive;

void Convolution3DForwardImpl::exec(_megdnn_tensor_in src,
                                    _megdnn_tensor_in filter,
                                    _megdnn_tensor_out dst,
                                    _megdnn_workspace workspace) {
    MIDOUT_BEGIN(megdnn_naive_conv3d_fwd) {
        auto filter_meta = check_exec(src.layout, filter.layout, dst.layout,
                                      workspace.size);
        switch (param().data_type) {
            case Param::DataType::FLOAT:
#define cb(dt)                                                            \
    do {                                                                  \
        if (src.layout.dtype == dt()) {                                   \
            using ctype = DTypeTrait<dt>::ctype;                          \
            MEGDNN_DISPATCH_CPU_KERN(                                     \
                    static_cast<HandleImpl*>(handle()),                   \
                    convolution3d::forward<                               \
                            ctype MEGDNN_COMMA ctype MEGDNN_COMMA ctype>( \
                            src, filter, dst, filter_meta););             \
            return;                                                       \
        }                                                                 \
    } while (0);
                MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb);
#undef cb
                break;
            case Param::DataType::FLOAT_IO16xC32:
                DNN_INC_FLOAT16(MEGDNN_DISPATCH_CPU_KERN(
                        static_cast<HandleImpl*>(handle()),
                        convolution3d::forward<
                                dt_float16 MEGDNN_COMMA dt_float16 MEGDNN_COMMA
                                        dt_float32>(src, filter, dst,
                                                    filter_meta);));
                return;
        }
        megdnn_assert_internal(0);
    }
    MIDOUT_END();
}

void Convolution3DBackwardDataImpl::exec(_megdnn_tensor_in filter,
                                         _megdnn_tensor_in diff,
                                         _megdnn_tensor_out grad,
                                         _megdnn_workspace workspace) {
    auto filter_meta =
            check_exec(filter.layout, diff.layout, grad.layout, workspace.size);
#define cb(dt)                                                            \
    do {                                                                  \
        if (filter.layout.dtype == dt()) {                                \
            using ctype = DTypeTrait<dt>::ctype;                          \
            MEGDNN_DISPATCH_CPU_KERN(                                     \
                    static_cast<HandleImpl*>(handle()),                   \
                    convolution3d::backward_data<                         \
                            ctype MEGDNN_COMMA ctype MEGDNN_COMMA ctype>( \
                            filter, diff, grad, filter_meta););           \
            return;                                                       \
        }                                                                 \
    } while (0);
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb);
#undef cb

    megdnn_assert_internal(0);
}
void Convolution3DBackwardFilterImpl::exec(_megdnn_tensor_in src,
                                           _megdnn_tensor_in diff,
                                           _megdnn_tensor_out grad,
                                           _megdnn_workspace workspace) {
    auto filter_meta =
            check_exec(src.layout, diff.layout, grad.layout, workspace.size);
#define cb(dt)                                                            \
    do {                                                                  \
        if (src.layout.dtype == dt()) {                                   \
            using ctype = DTypeTrait<dt>::ctype;                          \
            MEGDNN_DISPATCH_CPU_KERN(                                     \
                    static_cast<HandleImpl*>(handle()),                   \
                    convolution3d::backward_filter<                       \
                            ctype MEGDNN_COMMA ctype MEGDNN_COMMA ctype>( \
                            src, diff, grad, filter_meta););              \
            return;                                                       \
        }                                                                 \
    } while (0);
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb);
#undef cb

    megdnn_assert_internal(0);
}

std::vector<Convolution3DForward::Algorithm*>
Convolution3DForwardImpl::get_all_algorithms(const TensorLayout&,
                                             const TensorLayout&,
                                             const TensorLayout&) {
    return {static_cast<HandleImpl*>(handle())->default_conv3d_fwd_algo()};
}

Convolution3DForward::Algorithm*
Convolution3DForwardImpl::get_algorithm_heuristic(
        const TensorLayout& /* src */, const TensorLayout& /* filter */,
        const TensorLayout& /* dst */, size_t /* workspace_limit_in_bytes */,
        const AlgoAttribute& positive_attr,
        const AlgoAttribute& negative_attr) {
    auto algo = static_cast<HandleImpl*>(handle())->default_conv3d_fwd_algo();
    algo->check_attribute(positive_attr, negative_attr);
    return algo;
}

Convolution3DForward::Algorithm*
Convolution3DForwardImpl::get_algorithm_from_desc(
        const AlgorithmDesc& desc) {
    Algorithm* ret =
            static_cast<HandleImpl*>(handle())->default_conv3d_fwd_algo();
    megdnn_assert(desc == ret->info().desc);
    return ret;
}

std::vector<Convolution3DBackwardData::Algorithm*>
Convolution3DBackwardDataImpl::get_all_algorithms(const TensorLayout&,
                                                  const TensorLayout&,
                                                  const TensorLayout&) {
    return {static_cast<HandleImpl*>(handle())->default_conv3d_bwd_data_algo()};
}

Convolution3DBackwardData::Algorithm*
Convolution3DBackwardDataImpl::get_algorithm_heuristic(
        const TensorLayout& /* filter */, const TensorLayout& /* diff */,
        const TensorLayout& /* grad */, size_t /* workspace_limit_in_bytes */,
        const AlgoAttribute& positive_attr,
        const AlgoAttribute& negative_attr) {
    auto algo =
            static_cast<HandleImpl*>(handle())->default_conv3d_bwd_data_algo();
    algo->check_attribute(positive_attr, negative_attr);
    return algo;
}

Convolution3DBackwardData::Algorithm*
Convolution3DBackwardDataImpl::get_algorithm_from_desc(
        const AlgorithmDesc& desc) {
    Algorithm* ret =
            static_cast<HandleImpl*>(handle())->default_conv3d_bwd_data_algo();
    megdnn_assert(desc == ret->info().desc);
    return ret;
}

std::vector<Convolution3DBackwardFilter::Algorithm*>
Convolution3DBackwardFilterImpl::get_all_algorithms(const TensorLayout&,
                                                    const TensorLayout&,
                                                    const TensorLayout&) {
    return {static_cast<HandleImpl*>(handle())
                    ->default_conv3d_bwd_filter_algo()};
}

Convolution3DBackwardFilter::Algorithm*
Convolution3DBackwardFilterImpl::get_algorithm_heuristic(
        const TensorLayout& /* src */, const TensorLayout& /* diff */,
        const TensorLayout& /* grad */, size_t /* workspace_limit_in_bytes */
        ,
        const AlgoAttribute& positive_attr,
        const AlgoAttribute& negative_attr) {
    auto algo = static_cast<HandleImpl*>(handle())
                        ->default_conv3d_bwd_filter_algo();
    algo->check_attribute(positive_attr, negative_attr);
    return algo;
}

Convolution3DBackwardFilter::Algorithm*
Convolution3DBackwardFilterImpl::get_algorithm_from_desc(
        const AlgorithmDesc& desc) {
    Algorithm* ret = static_cast<HandleImpl*>(handle())
                             ->default_conv3d_bwd_filter_algo();
    megdnn_assert(desc == ret->info().desc);
    return ret;
}

const char* Convolution3DForwardImpl::get_algorithm_set_name() const {
    return "DEFAULT";
}

const char* Convolution3DBackwardDataImpl::get_algorithm_set_name() const {
    return "DEFAULT";
}

const char* Convolution3DBackwardFilterImpl::get_algorithm_set_name() const {
    return "DEFAULT";
}
// vim: syntax=cpp.doxygen
