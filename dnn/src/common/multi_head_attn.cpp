#include "megdnn/basic_types.h"
#include "megdnn/dtype.h"
#include "megdnn/oprs.h"
#include "src/common/utils.cuh"
#include "unroll_macro.h"

#include "src/common/utils.h"

namespace megdnn {

using Param = MultiHeadAttnBase::Param;
using InputType = Param::TensorCombinationType;
using MaskType = Param::AttnMaskType;

void MultiHeadAttnForward::check_exec(
        const TensorLayout& queries, const TensorLayout& keys,
        const TensorLayout& values, const TensorLayout& qkvo_weight_bias,
        const TensorLayout& attn_mask, const TensorLayout& bias_k,
        const TensorLayout& bias_v, const TensorLayout& out,
        const TensorLayout& attn_weight, const TensorLayout& mask_reservespace,
        const TensorLayout& othr_reservespace, size_t workspace_in_bytes) {
    Param p = param();
    // contiguous
    megdnn_assert_contiguous(queries);
    megdnn_assert_contiguous(keys);
    megdnn_assert_contiguous(values);
    megdnn_assert_contiguous(out);
    megdnn_assert_contiguous(attn_weight);
    if (p.training) {
        megdnn_assert_contiguous(othr_reservespace);
    }
    if (p.qproj_size or p.kproj_size or p.vproj_size or p.kproj_size)
        megdnn_assert_contiguous(qkvo_weight_bias);
    bool have_mask = false;
    bool have_biaskv = false;
    auto input_type = p.tensor_combination_type;
    if (input_type == InputType::ONLY_BIASKV or input_type == InputType::ALL) {
        have_biaskv = true;
        megdnn_assert_contiguous(bias_k);
        megdnn_assert_contiguous(bias_v);
    }
    if (input_type == InputType::ONLY_MASK or input_type == InputType::ALL) {
        have_mask = true;
        megdnn_assert_contiguous(attn_mask);
    }

    // misc
    size_t required_workspace_in_bytes = get_workspace_in_bytes(
            queries, keys, values, qkvo_weight_bias, attn_mask, bias_k, bias_v, out,
            attn_weight, mask_reservespace, othr_reservespace);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
    megdnn_assert(
            queries.ndim == 3, "queries.ndim should be 3, but got %zu", queries.ndim);
    megdnn_assert(keys.ndim == 3, "keys.ndim should be 3, but got %zu", keys.ndim);
    megdnn_assert(
            values.ndim == 3, "values.ndim should be 3, but got %zu", values.ndim);
    auto errmsg = [&]() {
        return megdnn_layout_msg(queries) + ", " + megdnn_layout_msg(keys) + ", " +
               megdnn_layout_msg(values) + ", " + megdnn_layout_msg(qkvo_weight_bias) +
               ", " + megdnn_layout_msg(attn_mask) + ", " + megdnn_layout_msg(bias_k) +
               ", " + megdnn_layout_msg(bias_v) + ", " + megdnn_layout_msg(out) + ", " +
               megdnn_layout_msg(attn_weight);
    };

    // batch match
    megdnn_assert(
            (queries.shape[0] == out.shape[0]) and
                    (keys.shape[0] == values.shape[0]) and
                    (queries.shape[0] == keys.shape[0]),
            "the batch of query(%zu), key(%zu), value(%zu) and output(%zu) do not "
            "match. details: %s",
            queries.shape[0], keys.shape[0], values.shape[0], out.shape[0],
            errmsg().c_str());
    // sequence length match
    megdnn_assert(
            queries.shape[1] == out.shape[1],
            "the sequence length of query(%zu) does not match the sequence length of "
            "output(%zu). details: %s",
            queries.shape[1], out.shape[1], errmsg().c_str());
    megdnn_assert(
            keys.shape[1] == values.shape[1],
            "the sequence length of key(%zu) does not match the sequence length of "
            "value(%zu). details: %s",
            keys.shape[1], values.shape[1], errmsg().c_str());
    // bias_k and bias_v layout check
    if (have_biaskv) {
        megdnn_assert(
                bias_k.ndim == 3 and bias_v.ndim == 3,
                "bias_k ndim should be 3, but got %zu, details: %s", bias_k.ndim,
                errmsg().c_str());
        megdnn_assert(
                (bias_k.shape[0] == 1) and (bias_k.shape[1] == 1) and
                        (bias_k.shape[2] == (p.kproj_size ? p.kproj_size : p.k_size)),
                "bias_k.shape should be [1, 1, %u], but got [%zu, "
                "%zu, %zu], details: %s",
                p.kproj_size ? p.kproj_size : p.k_size, bias_k.shape[0],
                bias_k.shape[1], bias_k.shape[2], errmsg().c_str());
        megdnn_assert(
                (bias_v.shape[0] == 1) and (bias_v.shape[1] == 1) and
                        (bias_v.shape[2] == (p.vproj_size ? p.vproj_size : p.v_size)),
                "bias_v.shape should be [1, 1, %u], but got [%zu, "
                "%zu, %zu], details: %s",
                p.vproj_size ? p.vproj_size : p.v_size, bias_v.shape[0],
                bias_v.shape[1], bias_v.shape[2], errmsg().c_str());
    }
    // attn mask layout check
    size_t attn_add = (have_biaskv ? 1 : 0) + (p.add_zero_attn ? 1 : 0);
    if (have_mask and attn_mask.ndim == 3) {
        megdnn_assert(
                (queries.shape[0] * p.num_heads == attn_mask.shape[0]) and
                        (queries.shape[1] == attn_mask.shape[1]) and
                        ((keys.shape[1] + attn_add) == attn_mask.shape[2]),
                "attn_mask.shape should be [%zu, %zu, %zu](attn_add=%zu), but got "
                "[%zu, %zu, %zu]. details: %s",
                queries.shape[0] * p.num_heads, queries.shape[1],
                keys.shape[1] + attn_add, attn_add, attn_mask.shape[0],
                attn_mask.shape[1], attn_mask.shape[2], errmsg().c_str());
    } else if (have_mask and attn_mask.ndim == 2) {
        megdnn_assert(
                (queries.shape[1] == attn_mask.shape[0]) and
                        ((keys.shape[1] + attn_add) == attn_mask.shape[1]),
                "attn_mask.shape should be [%zu, %zu](attn_add=%zu), but got "
                "[%zu, %zu]. details: %s",
                queries.shape[1], keys.shape[1] + attn_add, attn_add,
                attn_mask.shape[0], attn_mask.shape[1], errmsg().c_str());
    }
    // attn_weight layout check
    megdnn_assert(
            (attn_weight.shape[0] == queries.shape[0] * p.num_heads) and
                    (attn_weight.shape[1] == queries.shape[1]) and
                    (attn_weight.shape[2] == keys.shape[1] + attn_add),
            "attn_weight.shape should be [%zu, %zu, %zu](attn_add=%zu), but got [%zu, "
            "%zu, %zu]. details: %s",
            queries.shape[0] * p.num_heads, queries.shape[1], keys.shape[1] + attn_add,
            attn_add, attn_weight.shape[0], attn_weight.shape[1], attn_weight.shape[2],
            errmsg().c_str());

    // weigth and bias
#define TOSTRING(data) #data "=" + std::to_string(data)
    auto param_errmsg = [&]() {
        return TOSTRING(p.embeding_size) + ", " + TOSTRING(p.k_size) + ", " +
               TOSTRING(p.v_size) + ", " + TOSTRING(p.qproj_size) + ", " +
               TOSTRING(p.kproj_size) + ", " + TOSTRING(p.vproj_size) + ", " +
               TOSTRING(p.oproj_size) + ", " + TOSTRING(p.qbias) + ", " +
               TOSTRING(p.kbias) + ", " + TOSTRING(p.vbias) + ", " + TOSTRING(p.obias) +
               ", " + TOSTRING(p.num_heads) + ", " + TOSTRING(p.need_weights) + ", " +
               TOSTRING(p.add_zero_attn) + ", " + TOSTRING(int(p.attn_mask_type)) +
               ", " + TOSTRING(int(p.tensor_combination_type)) + ", " +
               TOSTRING(p.sm_scaler) + ", " + TOSTRING(p.training);
    };
#undef TOSTRING
    size_t weight_len = 0;
    size_t embeding_size = p.embeding_size;
    size_t ksize = p.k_size;
    size_t vsize = p.v_size;
    size_t qprojsize = p.qproj_size;
    size_t kprojsize = p.kproj_size;
    size_t vprojsize = p.vproj_size;
    size_t oprojsize = p.oproj_size;
    megdnn_assert(embeding_size == queries.shape[2], "%s", param_errmsg().c_str());
    megdnn_assert(ksize == keys.shape[2], "%s", param_errmsg().c_str());
    megdnn_assert(vsize == values.shape[2], "%s", param_errmsg().c_str());
    if (qprojsize == 0 and kprojsize == 0)
        megdnn_assert(embeding_size == ksize, "%s", param_errmsg().c_str());
    if (qprojsize == 0 and kprojsize != 0)
        megdnn_assert(
                embeding_size * p.num_heads == kprojsize, "%s", param_errmsg().c_str());
    if (qprojsize != 0 and kprojsize == 0)
        megdnn_assert(qprojsize == ksize * p.num_heads, "%s", param_errmsg().c_str());
    if (qprojsize != 0 and kprojsize != 0)
        megdnn_assert(qprojsize == kprojsize, "%s", param_errmsg().c_str());
    if (p.qbias)
        megdnn_assert(p.qproj_size > 0, "%s", param_errmsg().c_str());
    if (p.kbias)
        megdnn_assert(p.kproj_size > 0, "%s", param_errmsg().c_str());
    if (p.vbias)
        megdnn_assert(p.vproj_size > 0, "%s", param_errmsg().c_str());
    if (p.obias)
        megdnn_assert(p.oproj_size > 0, "%s", param_errmsg().c_str());
    if (p.qproj_size > 0)
        weight_len += embeding_size * qprojsize + (p.qbias ? qprojsize : 0);
    if (p.kproj_size > 0)
        weight_len += ksize * kprojsize + (p.kbias ? kprojsize : 0);
    if (p.vproj_size > 0)
        weight_len += vsize * vprojsize + (p.vbias ? vprojsize : 0);
    if (p.oproj_size > 0 and p.vproj_size > 0)
        weight_len += vprojsize * oprojsize + (p.obias ? oprojsize : 0);
    else if (p.oproj_size > 0 and p.vproj_size == 0)
        weight_len += p.num_heads * vsize * oprojsize + (p.obias ? oprojsize : 0);
    megdnn_assert(
            weight_len == qkvo_weight_bias.total_nr_elems(),
            "qkvo_weight_bias length should be %zu, but got %zu. details: %s",
            weight_len, qkvo_weight_bias.total_nr_elems(), param_errmsg().c_str());
}

void MultiHeadAttnBackward::deduce_layout(
        const TensorLayout& diff, const TensorLayout& queries, const TensorLayout& keys,
        const TensorLayout& values, const TensorLayout& qkvo_weight_bias,
        const TensorLayout& attn_mask, const TensorLayout& attn_weight,
        const TensorLayout& mask_reservespace, const TensorLayout& othr_reservespace,
        TensorLayout& dqueries, TensorLayout& dkeys, TensorLayout& dvalues,
        TensorLayout& dqkvo_weight_bias, TensorLayout& dbias_k, TensorLayout& dbias_v) {
    MEGDNN_MARK_USED_VAR(diff);
    MEGDNN_MARK_USED_VAR(attn_mask);
    MEGDNN_MARK_USED_VAR(attn_weight);
    MEGDNN_MARK_USED_VAR(mask_reservespace);
    MEGDNN_MARK_USED_VAR(othr_reservespace);
    dqueries = queries;
    dkeys = keys;
    dvalues = values;
    dqkvo_weight_bias = qkvo_weight_bias;
    auto input_type = param().tensor_combination_type;
    if (input_type == InputType::ONLY_BIASKV or input_type == InputType::ALL) {
        dbias_k = TensorLayout(
                {1, 1, param().kproj_size ? param().kproj_size : param().k_size},
                keys.dtype);
        dbias_v = TensorLayout(
                {1, 1, param().vproj_size ? param().vproj_size : param().v_size},
                values.dtype);
    } else {
        dbias_k = TensorLayout();
        dbias_v = TensorLayout();
    }
}

void MultiHeadAttnBackward::check_exec(
        const TensorLayout& diff, const TensorLayout& queries, const TensorLayout& keys,
        const TensorLayout& values, const TensorLayout& qkvo_weight_bias,
        const TensorLayout& attn_mask, const TensorLayout& attn_weight,
        const TensorLayout& mask_reservespace, const TensorLayout& othr_reservespace,
        const TensorLayout& dqueries, const TensorLayout& dkeys,
        const TensorLayout& dvalues, const TensorLayout& dqkvo_weight_bias,
        const TensorLayout& dbias_k, const TensorLayout& dbias_v,
        size_t workspace_in_bytes) {
    Param p = param();
    megdnn_assert(
            p.training,
            "When calling MultiHeadAttn backward, param().training must be true, "
            "but got false");
    // contiguous
    megdnn_assert_contiguous(diff);
    megdnn_assert_contiguous(queries);
    megdnn_assert_contiguous(keys);
    megdnn_assert_contiguous(values);
    megdnn_assert_contiguous(attn_weight);
    megdnn_assert_contiguous(dqueries);
    megdnn_assert_contiguous(dkeys);
    megdnn_assert_contiguous(dvalues);
    if (p.training) {
        megdnn_assert_contiguous(othr_reservespace);
    }
    if (p.qproj_size or p.kproj_size or p.vproj_size or p.oproj_size) {
        megdnn_assert_contiguous(qkvo_weight_bias);
        megdnn_assert_contiguous(dqkvo_weight_bias);
    }

    auto input_type = p.tensor_combination_type;
    bool have_mask = false;
    bool have_biaskv =
            input_type == InputType::ONLY_BIASKV or input_type == InputType::ALL;
    if (input_type == InputType::ONLY_MASK or input_type == InputType::ALL) {
        have_mask = true;
        megdnn_assert_contiguous(attn_mask);
    }

    // misc
    auto required_workspace_in_bytes = get_workspace_in_bytes(
            diff, queries, keys, values, qkvo_weight_bias, attn_mask, attn_weight,
            mask_reservespace, othr_reservespace, dqueries, dkeys, dvalues,
            dqkvo_weight_bias, dbias_k, dbias_v);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
    megdnn_assert(othr_reservespace.total_nr_elems() > 0);
    megdnn_assert(
            queries.ndim == 3, "queries.ndim should be 3, but got %zu", queries.ndim);
    megdnn_assert(keys.ndim == 3, "keys.ndim should be 3, but got %zu", keys.ndim);
    megdnn_assert(
            values.ndim == 3, "values.ndim should be 3, but got %zu", values.ndim);
    megdnn_assert(diff.ndim == 3, "diff.ndim should be 3, but got %zu", diff.ndim);
    auto errmsg = [&]() {
        return megdnn_layout_msg(diff) + ", " + megdnn_layout_msg(queries) + ", " +
               megdnn_layout_msg(keys) + ", " + megdnn_layout_msg(values) + ", " +
               megdnn_layout_msg(qkvo_weight_bias) + ", " +
               megdnn_layout_msg(attn_weight) + ", " + megdnn_layout_msg(dqueries) +
               ", " + megdnn_layout_msg(dkeys) + ", " + megdnn_layout_msg(dvalues) +
               ", " + megdnn_layout_msg(dqkvo_weight_bias);
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

    // layout check
    size_t osize = p.oproj_size != 0
                         ? p.oproj_size
                         : (p.vproj_size != 0 ? p.vproj_size : p.v_size * p.num_heads);
    TensorLayout diff_expect = TensorLayout(
            TensorShape{queries.shape[0], queries.shape[1], osize}, queries.dtype);
    megdnn_assert(equal_layout(diff, diff_expect), "%s", errmsg().c_str());
    megdnn_assert(equal_layout(queries, dqueries), "%s", errmsg().c_str());
    megdnn_assert(equal_layout(keys, dkeys), "%s", errmsg().c_str());
    megdnn_assert(equal_layout(values, dvalues), "%s", errmsg().c_str());
    megdnn_assert(
            equal_layout(qkvo_weight_bias, dqkvo_weight_bias), "%s", errmsg().c_str());

    // batch match
    megdnn_assert(
            (queries.shape[0] == diff.shape[0]) and
                    (keys.shape[0] == values.shape[0]) and
                    (queries.shape[0] == keys.shape[0]),
            "the batch of query(%zu), key(%zu), value(%zu) and diff(%zu) do not "
            "match. details: %s",
            queries.shape[0], keys.shape[0], values.shape[0], diff.shape[0],
            errmsg().c_str());
    // sequence length match
    megdnn_assert(
            queries.shape[1] == diff.shape[1],
            "the sequence length of query(%zu) does not match the sequence length of "
            "output(%zu). details: %s",
            queries.shape[1], diff.shape[1], errmsg().c_str());
    megdnn_assert(
            keys.shape[1] == values.shape[1],
            "the sequence length of key(%zu) does not match the sequence length of "
            "value(%zu). details: %s",
            keys.shape[1], values.shape[1], errmsg().c_str());

    size_t attn_add = (have_biaskv ? 1 : 0) + (p.add_zero_attn ? 1 : 0);
    // attn_weight layout check
    megdnn_assert(
            (attn_weight.shape[0] == queries.shape[0] * p.num_heads) and
                    (attn_weight.shape[1] == queries.shape[1]) and
                    (attn_weight.shape[2] == keys.shape[1] + attn_add),
            "attn_weight.shape should be [%zu, %zu, %zu](attn_add=%zu), but got [%zu, "
            "%zu, %zu]. details: %s",
            queries.shape[0] * p.num_heads, queries.shape[1], keys.shape[1] + attn_add,
            attn_add, attn_weight.shape[0], attn_weight.shape[1], attn_weight.shape[2],
            errmsg().c_str());
    // dbias_k, dbias_v layout check
    if (have_biaskv) {
        megdnn_assert(
                dbias_k.ndim == 3 and dbias_v.ndim == 3,
                "dbias_k ndim should be 3, but got %zu, details: %s", dbias_k.ndim,
                errmsg().c_str());
        megdnn_assert(
                (dbias_k.shape[0] == 1) and (dbias_k.shape[1] == 1) and
                        (dbias_k.shape[2] == (p.kproj_size ? p.kproj_size : p.k_size)),
                "dbias_k.shape should be [1, 1, %u], but got [%zu, "
                "%zu, %zu], details: %s",
                p.kproj_size ? p.kproj_size : p.k_size, dbias_k.shape[0],
                dbias_k.shape[1], dbias_k.shape[2], errmsg().c_str());
        megdnn_assert(
                (dbias_v.shape[0] == 1) and (dbias_v.shape[1] == 1) and
                        (dbias_v.shape[2] == (p.vproj_size ? p.vproj_size : p.v_size)),
                "dbias_v.shape should be [1, 1, %u], but got [%zu, "
                "%zu, %zu], details: %s",
                p.vproj_size ? p.vproj_size : p.v_size, dbias_v.shape[0],
                dbias_v.shape[1], dbias_v.shape[2], errmsg().c_str());
    }
    // attn mask layout check
    if (have_mask and attn_mask.ndim == 3) {
        megdnn_assert(
                (queries.shape[0] * p.num_heads == attn_mask.shape[0]) and
                        (queries.shape[1] == attn_mask.shape[1]) and
                        ((keys.shape[1] + attn_add) == attn_mask.shape[2]),
                "attn_mask.shape should be [%zu, %zu, %zu](attn_add=%zu), but got "
                "[%zu, %zu, %zu]. details: %s",
                queries.shape[0] * p.num_heads, queries.shape[1],
                keys.shape[1] + attn_add, attn_add, attn_mask.shape[0],
                attn_mask.shape[1], attn_mask.shape[2], errmsg().c_str());
    } else if (have_mask and attn_mask.ndim == 2) {
        megdnn_assert(
                (queries.shape[1] == attn_mask.shape[0]) and
                        ((keys.shape[1] + attn_add) == attn_mask.shape[1]),
                "attn_mask.shape should be [%zu, %zu](attn_add=%zu), but got "
                "[%zu, %zu]. details: %s",
                queries.shape[1], keys.shape[1] + attn_add, attn_add,
                attn_mask.shape[0], attn_mask.shape[1], errmsg().c_str());
    }

    // weigth and bias
#define TOSTRING(data) #data "=" + std::to_string(data)
    auto param_errmsg = [&]() {
        return TOSTRING(p.embeding_size) + ", " + TOSTRING(p.k_size) + ", " +
               TOSTRING(p.v_size) + ", " + TOSTRING(p.qproj_size) + ", " +
               TOSTRING(p.kproj_size) + ", " + TOSTRING(p.vproj_size) + ", " +
               TOSTRING(p.oproj_size) + ", " + TOSTRING(p.qbias) + ", " +
               TOSTRING(p.kbias) + ", " + TOSTRING(p.vbias) + ", " + TOSTRING(p.obias) +
               ", " + TOSTRING(p.num_heads) + ", " + TOSTRING(p.need_weights) + ", " +
               TOSTRING(p.add_zero_attn) + ", " + TOSTRING(int(p.attn_mask_type)) +
               ", " + TOSTRING(int(p.tensor_combination_type)) + ", " +
               TOSTRING(p.sm_scaler) + ", " + TOSTRING(p.training);
    };
#undef TOSTRING
    size_t weight_len = 0;
    size_t embeding_size = p.embeding_size;
    size_t ksize = p.k_size;
    size_t vsize = p.v_size;
    size_t qprojsize = p.qproj_size;
    size_t kprojsize = p.kproj_size;
    size_t vprojsize = p.vproj_size;
    size_t oprojsize = p.oproj_size;
    megdnn_assert(embeding_size == queries.shape[2], "%s", param_errmsg().c_str());
    megdnn_assert(ksize == keys.shape[2], "%s", param_errmsg().c_str());
    megdnn_assert(vsize == values.shape[2], "%s", param_errmsg().c_str());
    if (qprojsize == 0 and kprojsize == 0)
        megdnn_assert(embeding_size == ksize, "%s", param_errmsg().c_str());
    if (qprojsize == 0 and kprojsize != 0)
        megdnn_assert(
                embeding_size * p.num_heads == kprojsize, "%s", param_errmsg().c_str());
    if (qprojsize != 0 and kprojsize == 0)
        megdnn_assert(qprojsize == ksize * p.num_heads, "%s", param_errmsg().c_str());
    if (qprojsize != 0 and kprojsize != 0)
        megdnn_assert(qprojsize == kprojsize, "%s", param_errmsg().c_str());
    if (p.qbias)
        megdnn_assert(p.qproj_size > 0, "%s", param_errmsg().c_str());
    if (p.kbias)
        megdnn_assert(p.kproj_size > 0, "%s", param_errmsg().c_str());
    if (p.vbias)
        megdnn_assert(p.vproj_size > 0, "%s", param_errmsg().c_str());
    if (p.obias)
        megdnn_assert(p.oproj_size > 0, "%s", param_errmsg().c_str());
    if (p.qproj_size > 0)
        weight_len += embeding_size * qprojsize + (p.qbias ? qprojsize : 0);
    if (p.kproj_size > 0)
        weight_len += ksize * kprojsize + (p.kbias ? kprojsize : 0);
    if (p.vproj_size > 0)
        weight_len += vsize * vprojsize + (p.vbias ? vprojsize : 0);
    if (p.oproj_size > 0 and p.vproj_size > 0)
        weight_len += vprojsize * oprojsize + (p.obias ? oprojsize : 0);
    else if (p.oproj_size > 0 and p.vproj_size == 0)
        weight_len += p.num_heads * vsize * oprojsize + (p.obias ? oprojsize : 0);
    megdnn_assert(
            weight_len == qkvo_weight_bias.total_nr_elems(),
            "qkvo_weight_bias length should be %zu, but got %zu. details: %s",
            weight_len, qkvo_weight_bias.total_nr_elems(), param_errmsg().c_str());
    megdnn_assert(
            weight_len == dqkvo_weight_bias.total_nr_elems(),
            "dqkvo_weight_bias length should be %zu, but got %zu. details: %s",
            weight_len, dqkvo_weight_bias.total_nr_elems(), param_errmsg().c_str());
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen
