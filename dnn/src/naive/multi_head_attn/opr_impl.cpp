#include "src/naive/multi_head_attn/opr_impl.h"
#include "megdnn/oprs/linalg.h"
#include "src/common/utils.cuh"

namespace megdnn {
namespace naive {

using Param = MultiHeadAttnBase::Param;

size_t MultiHeadAttnForwardImpl::get_workspace_in_bytes(
        const TensorLayout& queries, const TensorLayout& keys,
        const TensorLayout& values, const TensorLayout& wqkv, const TensorLayout& out,
        const TensorLayout& reserveSpace) {
    MEGDNN_MARK_USED_VAR(queries);
    MEGDNN_MARK_USED_VAR(keys);
    MEGDNN_MARK_USED_VAR(values);
    MEGDNN_MARK_USED_VAR(wqkv);
    MEGDNN_MARK_USED_VAR(out);
    MEGDNN_MARK_USED_VAR(reserveSpace);
    megdnn_throw("unsupported naive multiheadattn forward\n");
}

void MultiHeadAttnForwardImpl::exec(
        _megdnn_tensor_in queries, _megdnn_tensor_in keys, _megdnn_tensor_in values,
        _megdnn_tensor_in wqkv, _megdnn_tensor_out out, _megdnn_tensor_out reserveSpace,
        _megdnn_workspace workspace) {
    MEGDNN_MARK_USED_VAR(queries);
    MEGDNN_MARK_USED_VAR(keys);
    MEGDNN_MARK_USED_VAR(values);
    MEGDNN_MARK_USED_VAR(wqkv);
    MEGDNN_MARK_USED_VAR(out);
    MEGDNN_MARK_USED_VAR(reserveSpace);
    check_exec(
            queries.layout, keys.layout, values.layout, wqkv.layout, out.layout,
            reserveSpace.layout, workspace.size);

    megdnn_throw("unsupported naive multiheadattn forward\n");
}

void MultiHeadAttnBackwardImpl::exec(
        _megdnn_tensor_in diff, _megdnn_tensor_in queries, _megdnn_tensor_in keys,
        _megdnn_tensor_in values, _megdnn_tensor_in wqkv,
        _megdnn_tensor_in reserveSpace, _megdnn_tensor_out dqueries,
        _megdnn_tensor_out dkeys, _megdnn_tensor_out dvalues,
        _megdnn_tensor_out dweights, _megdnn_workspace workspace) {
    check_exec(
            diff.layout, queries.layout, keys.layout, values.layout, wqkv.layout,
            reserveSpace.layout, dqueries.layout, dkeys.layout, dvalues.layout,
            dweights.layout, workspace.size);

    megdnn_throw("unsupported naive multiheadattn backward\n");
}

}  // namespace naive
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
