/**
 * \file dnn/src/fallback/group_local/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/fallback/group_local/opr_impl.h"
#include "src/naive/local/opr_impl.h"

#include "src/common/opr_delegate.h"
#include "src/naive/handle.h"

using namespace megdnn;
using namespace fallback;

GroupLocalImpl::GroupLocalImpl(Handle *handle):
    GroupLocalForward(handle),
    m_local_opr(inplace_cpu_handle()->create_operator<Local>())
{
}

size_t GroupLocalImpl::get_workspace_in_bytes(const TensorLayout &src,
        const TensorLayout &filter,
        const TensorLayout &dst)
{
    auto N = src.shape[0],
         IC = src.shape[1], IH = src.shape[2], IW = src.shape[3];
    auto FH = filter.shape[4], FW = filter.shape[5];
    auto OC = dst.shape[1],
         OH = dst.shape[2], OW = dst.shape[3];
    auto nr_group = filter.shape[0];
    auto ICg = IC/nr_group, OCg = OC/nr_group;
    m_local_opr->param() = this->param();
    TensorLayout src2, filter2, dst2;
    src2 = TensorLayout({N, ICg, IH, IW}, src.dtype);
    filter2 = TensorLayout({OH, OW, ICg, FH, FW, OCg},
            filter.dtype);
    dst2 = TensorLayout({N, OCg, OH, OW}, dst.dtype);
    return m_local_opr->get_workspace_in_bytes(src2, filter2, dst2);
}

void GroupLocalImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_in filter,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
    // Implemented by regular local
    check_exec(src.layout, filter.layout, dst.layout, workspace.size);

    auto local_dense =
        static_cast<naive::LocalForwardImpl*>(m_local_opr.get());
    local_dense->param() = this->param();
    auto src_gly = src, flt_gly = filter, dst_gly = dst;
    src_gly.layout.shape[1] /= filter.layout[0];
    flt_gly.layout = flt_gly.layout.remove_axis(0);
    dst_gly.layout.shape[1] /= filter.layout[0];
    auto fp = local_dense->make_float_kern_param(
            src_gly, flt_gly, dst_gly, workspace);
    auto kptr = local_dense->dispatch_float_noncontig_batch(
            src_gly.layout, flt_gly.layout, dst_gly.layout);
    auto nr_group = filter.layout.shape[0];
    auto flt_gstride = filter.layout.stride[0];
    auto data_type_size_in_bytes = src.layout.dtype.size();

    auto kern = [fp, nr_group, kptr, flt_gstride, data_type_size_in_bytes]() {
        auto cur_fp = fp;
        rep(g, nr_group) {
            auto ic = g * fp.ic;
            auto oc = g * fp.oc;
            const int8_t *sptr_tmp = reinterpret_cast<const int8_t*>(fp.src);
            const int8_t *fptr_tmp = reinterpret_cast<const int8_t*>(fp.filter);
            int8_t *dptr_tmp = reinterpret_cast<int8_t*>(fp.dst);

            sptr_tmp = sptr_tmp + ic * fp.ih * fp.iw * data_type_size_in_bytes;
            fptr_tmp = fptr_tmp + g * flt_gstride * data_type_size_in_bytes;
            dptr_tmp = dptr_tmp + oc * fp.oh * fp.ow * data_type_size_in_bytes;
            cur_fp.src = static_cast<const void*>(sptr_tmp);
            cur_fp.filter = static_cast<const void*>(fptr_tmp);
            cur_fp.dst = static_cast<void*>(dptr_tmp);
            kptr(cur_fp);
        }
    };
    static_cast<naive::HandleImpl*>(handle())->dispatch_kern(kern);
}

// vim: syntax=cpp.doxygen
