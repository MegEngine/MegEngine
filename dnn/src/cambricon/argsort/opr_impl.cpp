#include "src/cambricon/argsort/opr_impl.h"
#include "cnnl.h"
#include "src/cambricon/cnnl_wrapper/cnnl_tensor_descriptor.h"
#include "src/cambricon/utils.h"

#include <cmath>
#include <numeric>

namespace megdnn {
namespace cambricon {

using Order = ::megdnn::param::Argsort::Order;

void check_dtype(const TensorLayout& src) {
    auto dsrc = src.dtype;
    megdnn_assert(
            dsrc == dtype::Uint8() || dsrc == dtype::Int8() || dsrc == dtype::Int16() ||
            dsrc == dtype::Int32() || dsrc == dtype::Float32() ||
            dsrc == dtype::Float16());
}

struct ArgsortCnnlDescs {
    CnnlTensorDescriptor src_desc, output_desc, indices_desc;
    int k = 1;
    int sort_dim = 0;
    bool sorted = true;
    bool lower_index_first = true;
    bool largest = false;
    ArgsortCnnlDescs(
            const TensorLayout& src, const TensorLayout& dst,
            const TensorLayout& indices, Order order = Order::ASCENDING) {
        src_desc.set(src);
        output_desc.set(dst);
        indices_desc.set(indices, CNNL_DTYPE_INT32);
        largest = order == Order::DESCENDING;
        sort_dim = src.ndim - 1;
        k = src.shape[sort_dim];
        megdnn_assert(
                sort_dim * k <= 64 * pow(10, 12),
                "The size of dim dimension multiplied by k should be no larger than "
                "64*10^12");
    }
};

size_t ArgsortForwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& dst, const TensorLayout& indices) {
    auto handle = concrete_handle(this->handle());
    size_t topk_ws = 0;
    ArgsortCnnlDescs descs(src, dst, indices, param().order);
    cnnl_check(cnnlGetTopKTensorWorkspaceSize(
            /* handle         */ handle->cnnl_handle(),
            /* input_desc     */ descs.src_desc.desc(),
            /* k              */ descs.k,
            /* dim            */ descs.sort_dim,
            /* largest        */ descs.largest,
            /* output_desc    */ descs.output_desc.desc(),
            /* index_desc     */ descs.indices_desc.desc(),
            /* workspace_size */ &topk_ws));
    return topk_ws;
}

void ArgsortForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_tensor_out indices,
        _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, indices.layout, workspace.size);
    check_dtype(src.layout);
    auto _handle = cnnl_handle(this->handle());
    ArgsortCnnlDescs descs(src.layout, dst.layout, indices.layout, param().order);
    cnnl_check(cnnlTopKTensor_v3(
            /* handle            */ _handle,
            /* input_desc        */ descs.src_desc.desc(),
            /* input             */ src.raw_ptr(),
            /* k                 */ descs.k,
            /* dim               */ descs.sort_dim,
            /* largest           */ descs.largest,
            /* sorted            */ descs.sorted,
            /* lower_index_first */ descs.lower_index_first,
            /* workspace         */ workspace.raw_ptr,
            /* workspace_size    */ workspace.size,
            /* output_desc       */ descs.output_desc.desc(),
            /* output            */ dst.raw_ptr(),
            /* index_desc        */ descs.indices_desc.desc(),
            /* index             */ indices.ptr<int32_t>()));
}

void ArgsortBackwardImpl::exec(
        _megdnn_tensor_in diff, _megdnn_tensor_in indices, _megdnn_tensor_out grad,
        _megdnn_workspace workspace) {
    check_exec(diff.layout, indices.layout, grad.layout, workspace.size);
    check_dtype(diff.layout);
    auto handle = concrete_handle(this->handle());
    ArgsortCnnlDescs descs(diff.layout, grad.layout, indices.layout);
    cnnl_check(cnnlScatter(
            /* handle            */ handle->cnnl_handle(),
            /* dim               */ descs.sort_dim,
            /* input_desc        */ descs.src_desc.desc(),
            /* input             */ diff.raw_ptr(),
            /* index_desc        */ descs.indices_desc.desc(),
            /* index             */ indices.raw_ptr(),
            /* src_desc        */ descs.src_desc.desc(),
            /* src             */ diff.raw_ptr(),
            /* output_desc       */ descs.output_desc.desc(),
            /* output            */ grad.raw_ptr(),
            /* mode            */ CNNL_SCATTER));
}
}  // namespace cambricon
}  // namespace megdnn

// vim: syntax=cpp.doxygen
