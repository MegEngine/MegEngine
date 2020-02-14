/**
 * \file dnn/test/common/svd.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "test/common/svd.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"
#include "test/common/utils.h"
#include "test/common/workspace_wrapper.h"

using namespace megdnn;
using namespace test;

using Param = SVDForward::Param;

namespace {

template <typename T>
void fill_diag(const TensorND& v, TensorND& diag) {
    const auto& layout = diag.layout;
    megdnn_assert_contiguous(layout);
    megdnn_assert(layout.ndim >= 2);
    size_t block_cnt = 1;
    for (size_t i = 0; i < layout.ndim - 2; i++) {
        block_cnt *= layout[i];
    }
    size_t m = layout[layout.ndim - 2];
    size_t n = layout[layout.ndim - 1];
    size_t mn = std::min(m, n);
    auto v_ptr = v.ptr<T>();
    auto ptr = diag.ptr<T>();
    memset(ptr, 0, diag.layout.span().dist_byte());
    auto ld = layout.stride[layout.ndim - 2];
    for (size_t blk = 0; blk < block_cnt; blk++) {
        size_t offset(0), s_offset(0);
        if (block_cnt > 1) {
            offset = blk * layout.stride[layout.ndim - 3];
            s_offset = blk * v.layout.stride[v.layout.ndim - 2];
        }
        for (size_t i = 0; i < mn; i++) {
            ptr[offset + i * ld + i] = v_ptr[s_offset + i];
        }
    }
}

std::shared_ptr<Tensor<>> matmul(Handle* handle, const TensorND& A,
                                 const TensorND& B) {
    auto matmul_opr = handle->create_operator<BatchedMatrixMul>();

    TensorLayout result_layout;
    matmul_opr->deduce_layout(A.layout, B.layout, result_layout);
    std::shared_ptr<Tensor<>> result(new Tensor<>(handle, result_layout));
    WorkspaceWrapper ws(handle, matmul_opr->get_workspace_in_bytes(
                                        A.layout, B.layout, result->layout()));
    matmul_opr->exec(A, B, result->tensornd(), ws.workspace());
    return result;
}

}  // namespace

std::vector<SVDTestcase> SVDTestcase::make() {
    std::vector<SVDTestcase> ret;

    auto param = Param(false /* compute_uv */, false /* full_matrices */);
    auto add_shape = [&ret, &param](const TensorShape& shape) {
        ret.push_back({param, TensorLayout{shape, dtype::Float32()}});
    };

    add_shape({1, 7, 7});
    add_shape({1, 3, 7});
    add_shape({1, 7, 3});
    for (size_t rows : {2, 3, 5, 7, 10, 32, 100}) {
        for (size_t cols : {2, 3, 5, 7, 10, 32, 100}) {
            param.compute_uv = false;
            param.full_matrices = false;
            add_shape({3, rows, cols});

            param.compute_uv = true;
            add_shape({2, rows, cols});
            param.full_matrices = true;
            add_shape({3, rows, cols});
        }
    }

    NormalRNG data_rng;
    auto fill_data = [&](TensorND& data) {
        auto sz = data.layout.span().dist_byte(), szf = sz / sizeof(dt_float32);
        auto pf = static_cast<dt_float32*>(data.raw_ptr);
        data_rng.fill_fast_float32(pf, szf);
    };

    for (auto&& i : ret) {
        i.m_mem.reset(new dt_float32[i.m_mat.layout.span().dist_elem()]);
        i.m_mat.raw_ptr = i.m_mem.get();
        fill_data(i.m_mat);
    }

    return ret;
}

SVDTestcase::Result SVDTestcase::run(SVDForward* opr) {
    auto handle = opr->handle();
    auto src = make_tensor_h2d(handle, m_mat);

    // Deduce layout
    TensorLayout u_layout, s_layout, vt_layout;
    opr->param() = m_param;
    opr->deduce_layout(m_mat.layout, u_layout, s_layout, vt_layout);

    // Alloc tensor on device
    Tensor<> u{handle, u_layout}, s{handle, s_layout}, vt{handle, vt_layout};
    WorkspaceWrapper ws(handle,
                        opr->get_workspace_in_bytes(m_mat.layout, u_layout,
                                                    s_layout, vt_layout));

    opr->exec(*src, u.tensornd(), s.tensornd(), vt.tensornd(), ws.workspace());

    auto u_host = make_tensor_d2h(handle, u.tensornd());
    // Defined in wsdk8/Include/shared/inaddr.h Surprise! It's Windows.
    #undef s_host
    auto s_host = make_tensor_d2h(handle, s.tensornd());
    auto vt_host = make_tensor_d2h(handle, vt.tensornd());
    if (m_param.compute_uv) {
        // Copy back singular value, build diag(s)
        std::unique_ptr<dt_float32> diag_s_host_mem(
                new dt_float32[m_mat.layout.span().dist_elem()]);
        TensorLayout diag_layout = m_mat.layout;
        if (!m_param.full_matrices) {
            SmallVector<size_t> shape;
            for (int i = 0; i < (int)diag_layout.ndim - 2; i++) {
                shape.push_back(diag_layout[i]);
            }
            size_t x = std::min(diag_layout[diag_layout.ndim - 1],
                                diag_layout[diag_layout.ndim - 2]);
            shape.push_back(x);
            shape.push_back(x);
            diag_layout = {shape, diag_layout.dtype};
        }
        TensorND diag_s_host{diag_s_host_mem.get(), diag_layout};
        fill_diag<dt_float32>(*s_host, diag_s_host);

        // Try to recover original matrix by u * diag(s) * vt
        auto diag_s_dev = make_tensor_h2d(handle, diag_s_host);
        auto tmp = matmul(handle, u.tensornd(), *diag_s_dev);
        auto recovered = matmul(handle, tmp->tensornd(), vt.tensornd());
        return {u_host, s_host, vt_host,
                make_tensor_d2h(handle, recovered->tensornd())};
    }
    return {u_host, s_host, vt_host, nullptr};
}

// vim: syntax=cpp.doxygen
