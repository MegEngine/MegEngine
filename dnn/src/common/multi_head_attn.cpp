#include "megdnn/basic_types.h"
#include "megdnn/oprs.h"
#include "src/common/utils.cuh"
#include "unroll_macro.h"

#include "src/common/utils.h"

namespace megdnn {

using Param = MultiHeadAttnBase::Param;

void MultiHeadAttnForward::deduce_layout(
        const TensorLayout& queries, const TensorLayout& keys,
        const TensorLayout& values, const TensorLayout& wqkv, TensorLayout& out,
        TensorLayout& reserveSpace) {
    megdnn_assert(
            queries.ndim == 3,
            "queries.ndim should be 3[batch, sequence, embeding], but got %zu",
            queries.ndim);
    size_t size =
            get_reservespace_in_bytes(queries, keys, values, wqkv, out, reserveSpace);
    out = TensorLayout(
            {queries.shape[0], queries.shape[1], queries.shape[2]}, queries.dtype);
    reserveSpace = TensorLayout({size}, queries.dtype);
}

void MultiHeadAttnForward::check_exec(
        const TensorLayout& queries, const TensorLayout& keys,
        const TensorLayout& values, const TensorLayout& wqkv, const TensorLayout& out,
        const TensorLayout& reserveSpace, size_t workspace_in_bytes) {
    Param p = param();
    megdnn_assert_contiguous(queries);
    megdnn_assert_contiguous(keys);
    megdnn_assert_contiguous(values);
    megdnn_assert_contiguous(wqkv);
    megdnn_assert_contiguous(out);
    if (p.training)
        megdnn_assert_contiguous(reserveSpace);
    auto required_workspace_in_bytes =
            get_workspace_in_bytes(queries, keys, values, wqkv, out, reserveSpace);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);

    megdnn_assert(
            queries.ndim == 3,
            "queries.ndim should be 3[batch, sequence, embeding], but got %zu",
            queries.ndim);
    megdnn_assert(
            keys.ndim == 3,
            "keys.ndim should be 3[batch, sequence, embeding], but got %zu", keys.ndim);
    megdnn_assert(
            values.ndim == 3,
            "values.ndim should be 3[batch, sequence, embeding], but got %zu",
            values.ndim);

    auto errmsg = [&]() {
        return megdnn_layout_msg(queries) + ", " + megdnn_layout_msg(keys) + ", " +
               megdnn_layout_msg(values) + ", " + megdnn_layout_msg(wqkv) + ", " +
               megdnn_layout_msg(out) + ", " + megdnn_layout_msg(reserveSpace);
    };
    megdnn_assert(queries.shape[0] == out.shape[0], "%s", errmsg().c_str());
    megdnn_assert(keys.shape[0] == values.shape[0], "%s", errmsg().c_str());
    megdnn_assert(queries.shape[0] == keys.shape[0], "%s", errmsg().c_str());
    megdnn_assert(queries.shape[1] == out.shape[1], "%s", errmsg().c_str());
    megdnn_assert(keys.shape[1] == values.shape[1], "%s", errmsg().c_str());
    megdnn_assert(
            queries.shape[2] == keys.shape[2] and keys.shape[2] == values.shape[2] and
                    queries.shape[2] == out.shape[2],
            "%s", errmsg().c_str());
}

void MultiHeadAttnBackward::deduce_layout(
        const TensorLayout& diff, const TensorLayout& queries, const TensorLayout& keys,
        const TensorLayout& values, const TensorLayout& wqkv,
        const TensorLayout& reserveSpace, TensorLayout& dqueries, TensorLayout& dkeys,
        TensorLayout& dvalues, TensorLayout& dweights) {
    MEGDNN_MARK_USED_VAR(diff);
    MEGDNN_MARK_USED_VAR(reserveSpace);
    dqueries = queries;
    dkeys = keys;
    dvalues = values;
    dweights = wqkv;
}

void MultiHeadAttnBackward::check_exec(
        const TensorLayout& diff, const TensorLayout& queries, const TensorLayout& keys,
        const TensorLayout& values, const TensorLayout& wqkv,
        const TensorLayout& reserveSpace, const TensorLayout& dqueries,
        const TensorLayout& dkeys, const TensorLayout& dvalues,
        const TensorLayout& dweights, size_t workspace_in_bytes) {
    Param p = param();
    megdnn_assert(
            p.training,
            "When calling MultiHeadAttn backward, param().training must be true, "
            "but got false");
    megdnn_assert_contiguous(diff);
    megdnn_assert_contiguous(queries);
    megdnn_assert_contiguous(keys);
    megdnn_assert_contiguous(values);
    megdnn_assert_contiguous(wqkv);
    megdnn_assert_contiguous(dqueries);
    megdnn_assert_contiguous(dkeys);
    megdnn_assert_contiguous(dvalues);
    megdnn_assert_contiguous(dweights);
    if (p.training)
        megdnn_assert_contiguous(reserveSpace);
    auto required_workspace_in_bytes = get_workspace_in_bytes(
            diff, queries, keys, values, wqkv, reserveSpace, dqueries, dkeys, dvalues,
            dweights);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
    megdnn_assert(reserveSpace.total_nr_elems() > 0);

    megdnn_assert(
            queries.ndim == 3,
            "queries.ndim should be 3[batch, sequence, embeding], but got %zu",
            queries.ndim);
    megdnn_assert(
            keys.ndim == 3,
            "keys.ndim should be 3[batch, sequence, embeding], but got %zu", keys.ndim);
    megdnn_assert(
            values.ndim == 3,
            "values.ndim should be 3[batch, sequence, embeding], but got %zu",
            values.ndim);
    megdnn_assert(
            diff.ndim == 3,
            "diff.ndim should be 3[batch, sequence, embeding], but got %zu", diff.ndim);

    auto errmsg = [&]() {
        return megdnn_layout_msg(diff) + ", " + megdnn_layout_msg(queries) + ", " +
               megdnn_layout_msg(keys) + ", " + megdnn_layout_msg(values) + ", " +
               megdnn_layout_msg(wqkv) + ", " + megdnn_layout_msg(reserveSpace) + ", " +
               megdnn_layout_msg(dqueries) + ", " + megdnn_layout_msg(dkeys) + ", " +
               megdnn_layout_msg(dvalues) + ", " + megdnn_layout_msg(dweights);
    };

    auto equal_layout = [](const TensorLayout& lhs, const TensorLayout& rhs) -> bool {
        if (!(lhs.ndim == rhs.ndim && lhs.dtype == rhs.dtype &&
              lhs.format == rhs.format))
            return false;
        for (size_t i = 0; i < lhs.ndim; ++i) {
            if (lhs.shape[i] != rhs.shape[i] || lhs.stride[i] != rhs.stride[i]) {
                return false;
            }
        }
        return true;
    };

    megdnn_assert(equal_layout(queries, diff), "%s", errmsg().c_str());
    megdnn_assert(equal_layout(queries, dqueries), "%s", errmsg().c_str());
    megdnn_assert(equal_layout(keys, dkeys), "%s", errmsg().c_str());
    megdnn_assert(equal_layout(values, dvalues), "%s", errmsg().c_str());
    megdnn_assert(equal_layout(wqkv, dweights), "%s", errmsg().c_str());

    megdnn_assert(queries.shape[0] == diff.shape[0], "%s", errmsg().c_str());
    megdnn_assert(keys.shape[0] == values.shape[0], "%s", errmsg().c_str());
    megdnn_assert(queries.shape[0] == keys.shape[0], "%s", errmsg().c_str());
    megdnn_assert(queries.shape[1] == diff.shape[1], "%s", errmsg().c_str());
    megdnn_assert(keys.shape[1] == values.shape[1], "%s", errmsg().c_str());
    megdnn_assert(
            queries.shape[2] == keys.shape[2] and keys.shape[2] == values.shape[2] and
                    queries.shape[2] == diff.shape[2],
            "%s", errmsg().c_str());
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen
