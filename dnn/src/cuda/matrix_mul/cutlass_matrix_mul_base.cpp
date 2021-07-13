/**
 * \file dnn/src/cuda/matrix_mul/cutlass_matrix_mul_base.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/cuda/handle.h"
#include "src/cuda/matrix_mul/algos.h"
#include "src/cuda/matrix_mul/cutlass_matrix_mul_wrapper.cuh"
#include "src/cuda/utils.h"

#if CUDA_VERSION >= 9020
using namespace megdnn;
using namespace cuda;

std::string
MatrixMulForwardImpl::AlgoCutlassMatrixMulBase::AlgoParam::to_string() const {
    return ssprintf("%dX%dX%d_%dX%dX%d", threadblock_m, threadblock_n,
                    threadblock_k, warp_m, warp_n, warp_k);
}

std::pair<bool, TensorLayoutArray>
MatrixMulForwardImpl::AlgoCutlassMatrixMulBase::construct_aligned_layouts(
        const SizeArgs& args) const {
    int alignment = max_alignment(args);
    int min_alignment = min_alignment_requirement();
    bool aligned = alignment >= min_alignment;
    if (aligned)
        return std::make_pair(!aligned, TensorLayoutArray{{}});
    auto&& param = args.opr->param();
    int m = args.layout_c.shape[0], n = args.layout_c.shape[1],
        k = args.layout_a.shape[param.transposeA ? 0 : 1];
    size_t align_m = get_aligned_power2(m, min_alignment);
    size_t align_n = get_aligned_power2(n, min_alignment);
    size_t align_k = get_aligned_power2(k, min_alignment);
    TensorLayoutArray layouts;
    layouts.emplace_back(TensorLayout{{align_m, align_k}, args.layout_a.dtype});
    layouts.emplace_back(TensorLayout{{align_k, align_n}, args.layout_b.dtype});
    layouts.emplace_back(TensorLayout{{align_m, align_n}, args.layout_c.dtype});
    return std::make_pair(!aligned, std::move(layouts));
}

void MatrixMulForwardImpl::AlgoCutlassMatrixMulBase::exec(
        const ExecArgs& args) const {
    auto aligned = construct_aligned_layouts(args);
    if (!aligned.first)
        return do_exec(args);
    const auto& layouts = aligned.second;
    auto tensor_a = args.tensor_a;
    auto tensor_b = args.tensor_b;
    auto workspace = args.workspace;
    size_t copy_size = 0;
    for (const auto& ly : layouts)
        copy_size += ly.span().dist_byte();
    auto&& param = args.opr->param();
    auto&& stream = cuda_stream(args.opr->handle());

    cuda_check(cudaMemsetAsync(workspace.raw_ptr, 0, copy_size, stream));

    auto&& relayout = args.opr->handle()->create_operator<RelayoutForward>();

    auto copy_stride = [](const TensorLayout& src, TensorLayout& dst,
                          bool trans) {
        dst.stride[0] = src.stride[0], dst.stride[1] = src.stride[1];
        if (trans)
            std::swap(dst.stride[0], dst.stride[1]);
    };
    copy_stride(layouts[0], tensor_a.layout, param.transposeA);
    tensor_a.raw_ptr = workspace.raw_ptr;
    relayout->exec(args.tensor_a, tensor_a);
    workspace.raw_ptr += layouts[0].span().dist_byte();
    workspace.size -= layouts[0].span().dist_byte();

    copy_stride(layouts[1], tensor_b.layout, param.transposeB);
    tensor_b.raw_ptr = workspace.raw_ptr;
    relayout->exec(args.tensor_b, tensor_b);
    workspace.raw_ptr += layouts[1].span().dist_byte();
    workspace.size -= layouts[1].span().dist_byte();

    decltype(tensor_a) tensor_c{workspace.raw_ptr, layouts[2]};
    workspace.raw_ptr += layouts[2].span().dist_byte();
    workspace.size -= layouts[2].span().dist_byte();

    auto&& matmul = args.opr->handle()->create_operator<MatrixMulForward>();
    matmul->param().transposeA = false;
    matmul->param().transposeB = false;
    matmul->param().compute_mode = args.opr->param().compute_mode;

    tensor_a.layout = layouts[0];
    tensor_b.layout = layouts[1];
    ExecArgs args_{static_cast<MatrixMulForwardImpl*>(matmul.get()), tensor_a,
                   tensor_b, tensor_c, workspace};
    do_exec(args_);

    tensor_c.layout.TensorShape::operator=(args.layout_c);
    relayout->exec(tensor_c, args.tensor_c);
}

int MatrixMulForwardImpl::AlgoCutlassMatrixMulBase::max_alignment(
        const SizeArgs& args) const {
    auto&& dtype_a = args.layout_a.dtype;
    auto&& dtype_b = args.layout_b.dtype;
    auto&& dtype_c = args.layout_c.dtype;
    auto get_alignment = [](const DType& dt, int len) {
        int size_bits = dt.size(1) * 8;
        int align = 128;
        while (align > 1) {
            if ((len * size_bits) % align == 0)
                break;
            align = align / 2;
        }
        return align / size_bits;
    };
    int lda = args.layout_a.stride[0], ldb = args.layout_b.stride[0],
        ldc = args.layout_c.stride[0];
    auto&& param = args.opr->param();
    int m = args.layout_c.shape[0], n = args.layout_c.shape[1],
        k = args.layout_a.shape[param.transposeA ? 0 : 1];
    int max_align = get_alignment(dtype_a, lda);
    max_align = std::min(get_alignment(dtype_a, m), max_align);
    max_align = std::min(get_alignment(dtype_a, n), max_align);
    max_align = std::min(get_alignment(dtype_a, k), max_align);
    max_align = std::min(get_alignment(dtype_a, lda), max_align);
    max_align = std::min(get_alignment(dtype_b, ldb), max_align);
    max_align = std::min(get_alignment(dtype_c, ldc), max_align);
    return max_align;
}
#endif

// vim: syntax=cpp.doxygen
