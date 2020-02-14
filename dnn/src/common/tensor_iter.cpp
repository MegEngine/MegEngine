/**
 * \file dnn/src/common/tensor_iter.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/tensor_iter.h"
#include "src/common/utils.h"

using namespace megdnn;

////////////////////////// TypeRef ////////////////////
TypeRef<dt_quint4>::TypeRef(dt_quint4* _ptr, size_t _offset) {
    ptr = reinterpret_cast<uint8_t*>(_ptr);
    offset = _offset;
    uint8_t cur = ptr[offset >> 1];
    val = convert<uint8_t, dt_quint4>(cur, dt_quint4(cur), offset & 0x1)
                  .as_uint8();

}

void TypeRef<dt_quint4>::operator=(const uint8_t _) {
    uint8_t cur = ptr[offset >> 1];
    ptr[offset >> 1] =
            convert<dt_quint4, uint8_t>(dt_quint4(_), cur, offset & 0x1);
}

TypeRef<dt_qint4>::TypeRef(dt_qint4* _ptr, size_t _offset) {
    ptr = reinterpret_cast<int8_t*>(_ptr);
    offset = _offset;
    int8_t cur = ptr[offset >> 1];
    val = convert<int8_t, dt_qint4>(cur, dt_qint4(cur), offset & 0x1).as_int8();
}

void TypeRef<dt_qint4>::operator=(const int8_t _) {
    int8_t cur = ptr[offset >> 1];
    ptr[offset >> 1] =
            convert<dt_qint4, int8_t>(dt_qint4(_), cur, offset & 0x1);
}

////////////////////// TensorIter /////////////////////

template<typename ctype, bool valonly>
typename TensorIter<ctype, valonly>::Iter
TensorIter<ctype, valonly>::Iter::make(
        ctype *ptr, const TensorLayout &layout, size_t offset) {
    megdnn_assert(layout.ndim);
    Iter rst;
    rst.m_ptr = ptr;
    if (valonly)
        rst.m_layout = layout.collapse_contiguous();
    else
        rst.m_layout = layout;
    rst.m_logical_offset = offset;
    rst.m_tot_nr_elems = rst.m_layout.total_nr_elems();
    rst.m_offset = 0;
    megdnn_assert(offset <= rst.m_tot_nr_elems);
    for (int i = rst.m_layout.ndim - 1; i >= 0; i --) {
        auto shp = rst.m_layout.shape[i];
        auto stride = rst.m_layout.stride[i];
        if (!shp) {
            // empty iter for empty layout
            return {};
        }
        rst.m_axis_reset_stride[i] = stride * (shp - 1);
        rst.m_axis_offset[i] = offset % shp;
        rst.m_offset += rst.m_axis_offset[i] * stride;
        offset /= shp;
    }
    return rst;
}

template<typename ctype, bool valonly>
void TensorIter<ctype, valonly>::Iter::on_access_idx_valonly_true() const {
    megdnn_throw("can not access idx of TensorIter if valonly is true");
}

namespace megdnn {
#define cb(_dt) \
    template class TensorIter<DTypeTrait<dtype::_dt>::ctype, false>; \
    template class TensorIter<DTypeTrait<dtype::_dt>::ctype, true>;

    MEGDNN_FOREACH_DTYPE_NAME(cb)
    MEGDNN_FOREACH_PARAMETERIZED_DTYPE(cb)
#undef cb
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
