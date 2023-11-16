#include "src/cambricon/indexing_one_hot/opr_impl.h"
#include "src/cambricon/cnnl_wrapper/cnnl_tensor_descriptor.h"
#include "src/cambricon/utils.h"

using namespace megdnn;
using namespace cambricon;

void IndexingOneHotForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in index, _megdnn_tensor_out dst,
        _megdnn_workspace workspace) {
    check_exec(src.layout, index.layout, dst.layout, workspace.size);
    auto dtype_dest = dst.layout.dtype.enumv();
    megdnn_assert(
            check_dtype_float_ieee(dtype_dest) ||
                    dtype_dest == megdnn::DTypeEnum::Int32,
            "Cambricon unsupport IndexingOneHot with dtype:%d",
            static_cast<int>(dtype_dest));
    auto cnnl_handler = cnnl_handle(this->handle());
    auto axis = param().axis;
    TensorLayout index_mid_layout = index.layout;
    index_mid_layout.add_axis_cont_inplace(axis);
    CnnlTensorDescriptor src_desc, index_desc, dst_desc;
    src_desc.set(src.layout);
    index_desc.set(index_mid_layout);
    dst_desc.set(dst.layout);
    cnnl_check(cnnlGather(
            cnnl_handler, axis, src_desc.desc(), src.raw_ptr(), index_desc.desc(),
            index.raw_ptr(), dst_desc.desc(), dst.raw_ptr()));
}

void IndexingSetOneHotForwardImpl::exec(
        _megdnn_tensor_inout data, _megdnn_tensor_in index, _megdnn_tensor_in sub,
        _megdnn_workspace workspace) {
    check_exec(data.layout, index.layout, sub.layout, workspace.size);
    auto dtype_dest = data.layout.dtype.enumv();
    megdnn_assert(
            check_dtype_float_ieee(dtype_dest) ||
                    dtype_dest == megdnn::DTypeEnum::Int32,
            "Cambricon unsupport IndexingSetOneHot with dtype:%d",
            static_cast<int>(dtype_dest));
    auto cnnl_handler = cnnl_handle(this->handle());
    auto axis = param().axis;
    TensorLayout index_mid_layout = index.layout;
    index_mid_layout.add_axis_cont_inplace(axis);
    CnnlTensorDescriptor data_desc, index_desc, sub_desc;
    data_desc.set(data.layout);
    index_desc.set(index_mid_layout);
    sub_desc.set(sub.layout);
    cnnl_check(cnnlScatter(
            cnnl_handler, axis, data_desc.desc(), data.raw_ptr(), index_desc.desc(),
            index.raw_ptr(), sub_desc.desc(), sub.raw_ptr(), data_desc.desc(),
            data.raw_ptr(), cnnlScatterMode_t::CNNL_SCATTER));
}
