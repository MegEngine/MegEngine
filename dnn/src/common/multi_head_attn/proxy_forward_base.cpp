#include "megdnn/basic_types.h"
#include "megdnn/dtype.h"
#include "megdnn/oprs.h"
#include "src/common/utils.cuh"

#include "src/common/multi_head_attn/proxy_forward_base.h"
#include "src/common/utils.h"

namespace megdnn {

namespace multi_head_attn {

bool MHAForwardProxyBase::layout_ismatch(MHA_PROXY_FORWARD_LAYOUT_CONST_PARAM) {
    MEGDNN_MARK_USED_VAR(handle);
    MEGDNN_MARK_USED_VAR(qkvo_weight_bias);
    MEGDNN_MARK_USED_VAR(attn_mask);
    MEGDNN_MARK_USED_VAR(bias_k);
    MEGDNN_MARK_USED_VAR(bias_v);
    MEGDNN_MARK_USED_VAR(out);
    MEGDNN_MARK_USED_VAR(attn_weight);
    MEGDNN_MARK_USED_VAR(mask_reservespace);
    MEGDNN_MARK_USED_VAR(othr_reservespace);
    if (m_matmul_opr == nullptr or m_bmatmul_opr == nullptr or m_add_opr == nullptr or
        m_elem_opr == nullptr or m_softmax_opr == nullptr or m_dropout_opr == nullptr or
        m_relayout_opr == nullptr or m_repeat_opr == nullptr) {
        megdnn_assert(
                m_matmul_opr == nullptr and m_bmatmul_opr == nullptr and
                m_add_opr == nullptr and m_elem_opr == nullptr and m_softmax_opr == nullptr and 
                m_dropout_opr == nullptr and m_relayout_opr == nullptr and 
                m_repeat_opr == nullptr,
                "All the sub-opr are either not constructed or all constructed, but "
                "now only a part is constructed.");
        m_matmul_opr = handle->create_operator<MatrixMulForward>();
        m_bmatmul_opr = handle->create_operator<BatchedMatrixMul>();
        m_add_opr = handle->create_operator<AddUpdate>();
        m_elem_opr = handle->create_operator<Elemwise>();
        m_softmax_opr = handle->create_operator<Softmax>();
        m_dropout_opr = handle->create_operator<Dropout>();
        m_relayout_opr = handle->create_operator<Relayout>();
        m_repeat_opr = handle->create_operator<RepeatForward>();
    }

    auto matmul_layout = [](const TensorLayout& A, const TensorLayout& B,
                            const TensorLayout& C, bool enable) -> bool {
        if (!enable) {
            return true;
        }
        // [A0, A1, A2]@[B0, B1] = [C0, C1, C2]
        if (A[2] != B[0] || C[0] != A[0] || A[1] != C[1] || C[2] != B[1]) {
            return false;
        }
        return true;
    };

    auto ndim_valid = [&](const Param& param) -> bool {
        if (param.num_heads > 1 && param.training) {
            return m_q_layout.ndim != 0 && m_k_layout.ndim != 0 &&
                   m_v_layout.ndim != 0 && m_nq_layout.ndim != 0 &&
                   m_nk_layout.ndim != 0 && m_nv_layout.ndim != 0 &&
                   m_nx_layout.ndim != 0 && m_nz_layout.ndim != 0 &&
                   m_z_layout.ndim != 0 && m_out_layout.ndim != 0 &&
                   m_mask1_layout.ndim != 0 && m_mask2_layout.ndim != 0;
        } else if (param.num_heads > 1 && !param.training) {
            return m_q_layout.ndim != 0 && m_k_layout.ndim != 0 &&
                   m_v_layout.ndim != 0 && m_nq_layout.ndim != 0 &&
                   m_nk_layout.ndim != 0 && m_nv_layout.ndim != 0 &&
                   m_nx_layout.ndim != 0 && m_nz_layout.ndim != 0 &&
                   m_z_layout.ndim != 0 && m_out_layout.ndim != 0;
        } else if (param.num_heads == 1 && param.training) {
            return m_q_layout.ndim != 0 && m_k_layout.ndim != 0 &&
                   m_v_layout.ndim != 0 && m_nx_layout.ndim != 0 &&
                   m_z_layout.ndim != 0 && m_out_layout.ndim != 0 &&
                   m_mask1_layout.ndim != 0 && m_mask2_layout.ndim != 0;
        } else {
            return m_q_layout.ndim != 0 && m_k_layout.ndim != 0 &&
                   m_v_layout.ndim != 0 && m_nx_layout.ndim != 0 &&
                   m_z_layout.ndim != 0 && m_out_layout.ndim != 0;
        }
    };

    auto equal_metadata = [&](const Param& param) -> bool {
        return m_heads == param.num_heads && m_embed_size == param.embeding_size &&
               m_ksize == param.k_size && m_vsize == param.v_size &&
               m_qproj_size == param.qproj_size && m_kproj_size == param.kproj_size &&
               m_vproj_size == param.vproj_size && m_oproj_size == param.oproj_size &&
               m_qbias == param.qbias && m_kbias == param.kbias &&
               m_vbias == param.vbias && m_obias == param.obias;
    };

    return equal_metadata(param) && ndim_valid(param) &&
           m_datatype == queries.dtype.enumv() &&
           matmul_layout(queries, m_wq_layout, m_q_layout, param.qproj_size != 0) &&
           matmul_layout(keys, m_wk_layout, m_k_layout, param.kproj_size != 0) &&
           matmul_layout(values, m_wv_layout, m_v_layout, param.vproj_size != 0);
}

void MHAForwardProxyBase::layout_refill(MHA_PROXY_FORWARD_LAYOUT_CONST_PARAM) {
    MEGDNN_MARK_USED_VAR(handle);
    MEGDNN_MARK_USED_VAR(attn_mask);
    MEGDNN_MARK_USED_VAR(out);
    MEGDNN_MARK_USED_VAR(attn_weight);
    MEGDNN_MARK_USED_VAR(mask_reservespace);
    MEGDNN_MARK_USED_VAR(othr_reservespace);

    m_heads = param.num_heads;
    m_embed_size = param.embeding_size;
    m_ksize = param.k_size;
    m_vsize = param.v_size;
    m_qproj_size = param.qproj_size;
    m_kproj_size = param.kproj_size;
    m_vproj_size = param.vproj_size;
    m_oproj_size = param.oproj_size;
    m_qbias = param.qbias;
    m_kbias = param.kbias;
    m_vbias = param.vbias;
    m_obias = param.obias;
    // m_add_bias_kv = param.add_bias_kv;
    // m_add_zero_attn = param.add_zero_attn;
    // m_reslink = param.reslink;
    auto cal_type = qkvo_weight_bias.dtype;
    TensorLayout placeholder_layout;

    auto reflash_dtype = [&](DType dtype) {
        // m_nbias_k_layout.dtype = dtype;
        // m_nbias_v_layout.dtype = dtype;
        // m_zero_k_layout.dtype = dtype;
        // m_zero_v_layout.dtype = dtype;
        // m_added_bias_zero_k_layout.dtype = dtype;
        // m_added_bias_zero_v_layout.dtype = dtype;
        m_q_layout.dtype = dtype;
        m_k_layout.dtype = dtype;
        m_v_layout.dtype = dtype;
        m_nq_layout.dtype = dtype;
        m_nk_layout.dtype = dtype;
        m_nv_layout.dtype = dtype;
        m_nx_layout.dtype = dtype;
        m_mask1_layout.dtype = dtype;
        m_nz_layout.dtype = dtype;
        m_z_layout.dtype = dtype;
        m_out_layout.dtype = dtype;
        m_mask2_layout.dtype = dtype;
    };
    reflash_dtype(queries.dtype);
    m_datatype = queries.dtype.enumv();
#define cb(DType)                                             \
    if (queries.dtype.enumv() == DTypeTrait<DType>::enumv) {  \
        m_sizeof_datatype = sizeof(DTypeTrait<DType>::ctype); \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb

    // proxy opr
    m_matmul_opr->param().format = param::MatrixMul::Format::DEFAULT;
    m_bmatmul_opr->param().format = param::MatrixMul::Format::DEFAULT;
    m_softmax_opr->param().axis = -1;
    m_dropout_opr->param().seed = param.seed;

    // wq/wk/wv/wo
    m_wq_layout = TensorLayout{{m_embed_size, m_qproj_size}, cal_type};
    m_wk_layout = TensorLayout{{m_ksize, m_kproj_size}, cal_type};
    m_wv_layout = TensorLayout{{m_vsize, m_vproj_size}, cal_type};
    if (m_vproj_size > 0) {
        m_wo_layout = TensorLayout{{m_vproj_size, m_oproj_size}, cal_type};
    } else {
        m_wo_layout = TensorLayout{{m_vsize * m_heads, m_oproj_size}, cal_type};
    }
    // bq/bk/bv/bo
    m_bq_layout = TensorLayout{{m_qproj_size}, cal_type};
    m_bk_layout = TensorLayout{{m_kproj_size}, cal_type};
    m_bv_layout = TensorLayout{{m_vproj_size}, cal_type};
    m_bo_layout = TensorLayout{{m_oproj_size}, cal_type};

    // wq/wk/wv/wo/bq/bk/bv/bo offset
    size_t end = 0;
    m_wq_off = 0, m_wk_off = 0, m_wv_off = 0, m_wo_off = 0;
    m_bq_off = 0, m_bk_off = 0, m_bv_off = 0, m_bo_off = 0;
    if (param.qproj_size) {
        m_wq_off = end;
        end += m_wq_layout.total_nr_elems();
    }
    if (param.kproj_size) {
        m_wk_off = end;
        end += m_wk_layout.total_nr_elems();
    }
    if (param.vproj_size) {
        m_wv_off = end;
        end += m_wv_layout.total_nr_elems();
    }
    if (param.oproj_size) {
        m_wo_off = end;
        end += m_wo_layout.total_nr_elems();
    }

    if (param.qbias && param.qproj_size) {
        m_bq_off = end;
        end += m_bq_layout.total_nr_elems();
    }
    if (param.kbias && param.kproj_size) {
        m_bk_off = end;
        end += m_bk_layout.total_nr_elems();
    }
    if (param.vbias && param.vproj_size) {
        m_bv_off = end;
        end += m_bv_layout.total_nr_elems();
    }
    if (param.obias && param.oproj_size) {
        m_bo_off = end;
        end += m_bo_layout.total_nr_elems();
    }

    // size_t attn_add = (param.add_bias_kv ? 1 : 0) + (param.add_zero_attn ? 1 : 0);

    // q/k/v, nq/nk/nv
    auto head_repeat = [&](TensorLayout& m_q_layout, TensorLayout& m_nq_layout) {
        m_repeat_opr->param().times =
                TensorLayout({1, m_heads, 1, 1}, m_q_layout.dtype);
        return m_repeat_opr->get_workspace_in_bytes(
                {{m_q_layout[0], 1, m_q_layout[1], m_q_layout[2]}, m_q_layout.dtype},
                {{m_nq_layout[0] / m_heads, m_heads, m_nq_layout[1], m_nq_layout[2]},
                 m_nq_layout.dtype});
    };
    m_matmul_opr->param().transposeA = false;
    m_matmul_opr->param().transposeB = false;
    if (param.qproj_size) {
        matmul_deduce_layout(m_matmul_opr, queries, m_wq_layout, m_q_layout);
        m_nq_layout = TensorLayout{
                {m_q_layout.shape[0] * m_heads, m_q_layout.shape[1],
                 m_q_layout.shape[2] / m_heads},
                m_q_layout.dtype};
        m_q_head_repeat_workspacesize = 0;
    } else {
        m_q_layout = queries;
        m_nq_layout = TensorLayout{
                {m_q_layout[0] * m_heads, m_q_layout[1], m_q_layout[2]},
                m_q_layout.dtype};
        m_q_head_repeat_workspacesize = head_repeat(m_q_layout, m_nq_layout);
    }
    if (param.kproj_size) {
        matmul_deduce_layout(m_matmul_opr, keys, m_wk_layout, m_k_layout);
        m_nk_layout = TensorLayout{
                {m_k_layout.shape[0] * m_heads, m_k_layout.shape[1],
                 m_k_layout.shape[2] / m_heads},
                m_k_layout.dtype};
        m_k_head_repeat_workspacesize = 0;
    } else {
        m_k_layout = keys;
        m_nk_layout = TensorLayout{
                {m_k_layout[0] * m_heads, m_k_layout[1], m_k_layout[2]},
                m_k_layout.dtype};
        m_k_head_repeat_workspacesize = head_repeat(m_k_layout, m_nk_layout);
    }
    if (param.vproj_size) {
        matmul_deduce_layout(m_matmul_opr, values, m_wv_layout, m_v_layout);
        m_nv_layout = TensorLayout{
                {m_v_layout.shape[0] * m_heads, m_v_layout.shape[1],
                 m_v_layout.shape[2] / m_heads},
                m_v_layout.dtype};
        m_v_head_repeat_workspacesize = 0;
    } else {
        m_v_layout = values;
        m_nv_layout = TensorLayout{
                {m_v_layout[0] * m_heads, m_v_layout[1], m_v_layout[2]},
                m_v_layout.dtype};
        m_v_head_repeat_workspacesize = head_repeat(m_v_layout, m_nv_layout);
    }
    m_q_workspacesize = 0;
    m_k_workspacesize = 0;
    m_v_workspacesize = 0;

    // nx
    m_bmatmul_opr->param().transposeA = false;
    m_bmatmul_opr->param().transposeB = true;
    m_dropout_opr->param().drop_prob = param.attn_prob;
    m_bmatmul_opr->deduce_layout(m_nq_layout, m_nk_layout, m_nx_layout);
    m_dropout_opr->deduce_layout(m_nx_layout, placeholder_layout, m_mask1_layout);
    m_nx_workspacesize = m_bmatmul_opr->get_workspace_in_bytes(
            m_nq_layout, m_nk_layout, m_nx_layout);
    m_softmax_workspacesize = m_softmax_opr->get_workspace_in_bytes(m_nx_layout, {});
    m_dropout1_workspacesize = m_dropout_opr->get_workspace_in_bytes(
            m_nx_layout, placeholder_layout, m_mask1_layout);

    // nz
    m_bmatmul_opr->param().transposeA = false;
    m_bmatmul_opr->param().transposeB = false;
    m_bmatmul_opr->deduce_layout(m_nx_layout, m_nv_layout, m_nz_layout);
    m_nz_workspacesize = m_bmatmul_opr->get_workspace_in_bytes(
            m_nx_layout, m_nv_layout, m_nz_layout);

    // z
    m_z_layout = TensorLayout{
            {m_nz_layout.shape[0] / m_heads, m_nz_layout.shape[1],
             m_nz_layout.shape[2] * m_heads},
            m_nz_layout.dtype};

    // out
    m_dropout_opr->param().drop_prob = param.out_prob;
    if (param.oproj_size) {
        matmul_deduce_layout(m_matmul_opr, m_z_layout, m_wo_layout, m_out_layout);
    } else {
        m_out_layout = m_z_layout;
    }
    m_dropout_opr->deduce_layout(m_out_layout, placeholder_layout, m_mask2_layout);
    m_out_workspacesize = 0;
    m_dropout2_workspacesize = m_dropout_opr->get_workspace_in_bytes(
            m_out_layout, placeholder_layout, m_mask2_layout);
}

WorkspaceBundle MHAForwardProxyBase::get_mask_reservespace_bundle(
        MHA_PROXY_FORWARD_LAYOUT_CONST_PARAM, void* ptr) {
    if (!layout_ismatch(MHA_PROXY_FORWARD_CALL)) {
        layout_refill(MHA_PROXY_FORWARD_CALL);
    }
    return WorkspaceBundle(
            ptr,
            {param.training ? m_mask1_layout.span().dist_byte() : 0,
             param.training ? m_mask2_layout.span().dist_byte() : 0},
            4);
}

WorkspaceBundle MHAForwardProxyBase::get_othr_reservespace_bundle(
        MHA_PROXY_FORWARD_LAYOUT_CONST_PARAM, void* ptr) {
    if (!layout_ismatch(MHA_PROXY_FORWARD_CALL)) {
        layout_refill(MHA_PROXY_FORWARD_CALL);
    }
    return WorkspaceBundle(
            ptr,
            {param.num_heads > 1 or param.qproj_size ? m_nq_layout.span().dist_byte()
                                                     : 0,
             param.num_heads > 1 or param.kproj_size ? m_nk_layout.span().dist_byte()
                                                     : 0,
             param.num_heads > 1 or param.vproj_size ? m_nv_layout.span().dist_byte()
                                                     : 0,
             param.training ? m_nx_layout.span().dist_byte() : 0,
             param.training or param.oproj_size ? m_z_layout.span().dist_byte() : 0},
            queries.dtype.size());
}

WorkspaceBundle MHAForwardProxyBase::get_workspace_bundle(
        MHA_PROXY_FORWARD_LAYOUT_CONST_PARAM, void* ptr) {
    if (!layout_ismatch(MHA_PROXY_FORWARD_CALL)) {
        layout_refill(MHA_PROXY_FORWARD_CALL);
    }
    return WorkspaceBundle(
            ptr,
            {param.num_heads > 1 and param.qproj_size ? m_q_layout.span().dist_byte()
                                                      : 0,
             param.num_heads > 1 and param.qproj_size ? m_q_workspacesize : 0,
             param.num_heads > 1 and param.kproj_size ? m_k_layout.span().dist_byte()
                                                      : 0,
             param.num_heads > 1 and param.kproj_size ? m_k_workspacesize : 0,
             param.num_heads > 1 and param.vproj_size ? m_v_layout.span().dist_byte()
                                                      : 0,
             param.num_heads > 1 and param.vproj_size ? m_v_workspacesize : 0,
             param.num_heads > 1 and !param.qproj_size ? m_q_head_repeat_workspacesize
                                                       : 0,
             param.num_heads > 1 and !param.kproj_size ? m_k_head_repeat_workspacesize
                                                       : 0,
             param.num_heads > 1 and !param.vproj_size ? m_v_head_repeat_workspacesize
                                                       : 0,
             m_nx_layout.span().dist_byte(), m_nx_workspacesize, m_sizeof_datatype,
             m_softmax_workspacesize, param.training ? m_dropout1_workspacesize : 0,
             param.num_heads > 1 ? m_nz_layout.span().dist_byte() : 0,
             m_nz_workspacesize,
             (param.oproj_size and param.training) ? m_out_layout.span().dist_byte()
                                                   : 0,
             param.oproj_size ? m_out_workspacesize : 0,
             param.training ? m_dropout2_workspacesize : 0});
}

size_t MHAForwardProxyBase::get_mask_reservespace_in_bytes(
        MHA_PROXY_FORWARD_LAYOUT_CONST_PARAM) {
    auto bundle = get_mask_reservespace_bundle(MHA_PROXY_FORWARD_CALL);
    return bundle.total_size_in_bytes();
}

size_t MHAForwardProxyBase::get_othr_reservespace_in_bytes(
        MHA_PROXY_FORWARD_LAYOUT_CONST_PARAM) {
    auto bundle = get_othr_reservespace_bundle(MHA_PROXY_FORWARD_CALL);
    return bundle.total_size_in_bytes();
}

size_t MHAForwardProxyBase::get_workspace_in_bytes(
        MHA_PROXY_FORWARD_LAYOUT_CONST_PARAM) {
    auto bundle = get_workspace_bundle(MHA_PROXY_FORWARD_CALL);
    return bundle.total_size_in_bytes();
}

void MHAForwardProxyBase::deduce_layout(MHA_PROXY_FORWARD_LAYOUT_PARAM) {
    if (!layout_ismatch(MHA_PROXY_FORWARD_CALL)) {
        layout_refill(MHA_PROXY_FORWARD_CALL);
    }
    attn_weight = m_nx_layout;
    out = m_out_layout;
    size_t mask_size = get_mask_reservespace_in_bytes(MHA_PROXY_FORWARD_CALL);
    size_t othr_size = get_othr_reservespace_in_bytes(MHA_PROXY_FORWARD_CALL);
    mask_reservespace = TensorLayout{{mask_size}, dtype::Uint8()};
    othr_reservespace = TensorLayout{{othr_size / queries.dtype.size()}, queries.dtype};
}

void MHAForwardProxyBase::exec(MHA_PROXY_FORWARD_EXEC_PARAM) {
#define cb(DType)                                                   \
    if (queries.layout.dtype.enumv() == DTypeTrait<DType>::enumv) { \
        using ctype = typename DTypeTrait<DType>::ctype;            \
        exec_internal<ctype>(MHA_PROXY_FORWARD_CALL, workspace);    \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
}

template <typename T>
void MHAForwardProxyBase::exec_internal(MHA_PROXY_FORWARD_EXEC_PARAM) {
    auto wksp_bundle = get_workspace_bundle(
            MHA_PROXY_FORWARD_TENSOR_TO_LAYOUT_CALL, workspace.raw_ptr);
    auto mask_bundle = get_mask_reservespace_bundle(
            MHA_PROXY_FORWARD_TENSOR_TO_LAYOUT_CALL, mask_reservespace.raw_ptr());
    auto othr_bundle = get_othr_reservespace_bundle(
            MHA_PROXY_FORWARD_TENSOR_TO_LAYOUT_CALL, othr_reservespace.raw_ptr());

    m_matmul_opr->param().transposeA = false;
    m_matmul_opr->param().transposeB = false;
    TensorND q, k, v;
    if (param.qproj_size) {
        if (param.num_heads == 1) {
            q = TensorND{othr_bundle.get_workspace(0).raw_ptr, m_q_layout};
        } else {
            q = TensorND{wksp_bundle.get_workspace(0).raw_ptr, m_q_layout};
        }
        TensorND qweight{qkvo_weight_bias.ptr<T>() + m_wq_off, m_wq_layout};
        matmul_exec(m_matmul_opr, queries, qweight, q, wksp_bundle.get_workspace(1));
        if (param.qbias) {
            m_add_opr->exec(q, {qkvo_weight_bias.ptr<T>() + m_bq_off, m_bq_layout});
        }
    } else {
        q = TensorND{queries.raw_ptr(), queries.layout};
    }
    if (param.kproj_size) {
        if (param.num_heads == 1) {
            k = TensorND{othr_bundle.get_workspace(1).raw_ptr, m_k_layout};
        } else {
            k = TensorND{wksp_bundle.get_workspace(2).raw_ptr, m_k_layout};
        }
        TensorND kweight{qkvo_weight_bias.ptr<T>() + m_wk_off, m_wk_layout};
        matmul_exec(m_matmul_opr, keys, kweight, k, wksp_bundle.get_workspace(3));
        if (param.kbias) {
            m_add_opr->exec(k, {qkvo_weight_bias.ptr<T>() + m_bk_off, m_bk_layout});
        }
    } else {
        k = TensorND{keys.raw_ptr(), keys.layout};
    }
    if (param.vproj_size) {
        if (param.num_heads == 1) {
            v = TensorND{othr_bundle.get_workspace(2).raw_ptr, m_v_layout};
        } else {
            v = TensorND{wksp_bundle.get_workspace(4).raw_ptr, m_v_layout};
        }
        TensorND vweight{qkvo_weight_bias.ptr<T>() + m_wv_off, m_wv_layout};
        matmul_exec(m_matmul_opr, values, vweight, v, wksp_bundle.get_workspace(5));
        if (param.vbias) {
            m_add_opr->exec(v, {qkvo_weight_bias.ptr<T>() + m_bv_off, m_bv_layout});
        }
    } else {
        v = TensorND{values.raw_ptr(), values.layout};
    }

    // nq/nk/nv: norm to multihead
    auto relayout_to_multihead = [&](TensorND& q, TensorND& nq) {
        size_t batch = q.layout[0];
        size_t seq = q.layout[1];
        size_t embeding_size = q.layout[2];
        TensorLayout nlayout{
                {batch, seq, m_heads, embeding_size / m_heads}, q.layout.dtype};
        nlayout = nlayout.dimshuffle({0, 2, 1, 3});
        m_relayout_opr->exec({q.raw_ptr(), nlayout}, nq);
    };
    auto repeat_to_multihead = [&](TensorND& q, TensorND& nq, size_t wksp_idx) {
        q.layout = TensorLayout(
                {q.layout[0], 1, q.layout[1], q.layout[2]}, q.layout.dtype);
        nq.layout = TensorLayout(
                {nq.layout[0] / m_heads, m_heads, nq.layout[1], nq.layout[2]},
                nq.layout.dtype);
        m_repeat_opr->param().times = TensorLayout({1, m_heads, 1, 1}, q.layout.dtype);
        m_repeat_opr->exec(q, nq, wksp_bundle.get_workspace(wksp_idx));
        nq.layout = TensorLayout(
                {nq.layout[0] * nq.layout[1], nq.layout[2], nq.layout[3]},
                nq.layout.dtype);
    };
    TensorND nq = q, nk = k, nv = v;
    if (param.num_heads > 1) {
        nq = TensorND{othr_bundle.get_workspace(0).raw_ptr, m_nq_layout};
        if (param.qproj_size) {
            relayout_to_multihead(q, nq);
        } else {
            repeat_to_multihead(q, nq, 6);
        }
    }
    if (param.num_heads > 1) {
        nk = TensorND{othr_bundle.get_workspace(1).raw_ptr, m_nk_layout};
        if (param.kproj_size) {
            relayout_to_multihead(k, nk);
        } else {
            repeat_to_multihead(k, nk, 7);
        }
    }
    if (param.num_heads > 1) {
        nv = TensorND{othr_bundle.get_workspace(2).raw_ptr, m_nv_layout};
        if (param.vproj_size) {
            relayout_to_multihead(v, nv);
        } else {
            repeat_to_multihead(v, nv, 8);
        }
    }

    // nx
    TensorND nx{wksp_bundle.get_workspace(9).raw_ptr, m_nx_layout};
    TensorND ny{othr_bundle.get_workspace(3).raw_ptr, m_nx_layout};
    TensorND mask1{mask_bundle.get_workspace(0).raw_ptr, m_mask1_layout};
    m_bmatmul_opr->param().transposeA = false;
    m_bmatmul_opr->param().transposeB = true;
    m_bmatmul_opr->exec(nq, nk, nx, wksp_bundle.get_workspace(10));
    // scale
    auto d_scaler = wksp_bundle.get_workspace(11).ptr<T>();
    T param_scaler = static_cast<T>(param.sm_scaler);
    move_scaler_to_device(handle, d_scaler, &param_scaler);
    m_elem_opr->param().mode = Elemwise::Mode::MUL;
    m_elem_opr->exec({nx, TensorND{d_scaler, {{1}, queries.layout.dtype}}}, nx);
    // mask
    if (param.attn_mask_type == MaskType::DEFAULT_MASK or
        param.attn_mask_type == MaskType::USER_DEFINED_MASK) {
        m_elem_opr->param().mode = Elemwise::Mode::ADD;
        m_elem_opr->exec({nx, attn_mask}, nx);
    }
    if (param.training) {
        // softmax
        m_softmax_opr->exec(nx, ny, wksp_bundle.get_workspace(12));
        // dropout
        m_dropout_opr->param().drop_prob = param.attn_prob;
        m_dropout_opr->exec(ny, attn_weight, mask1, wksp_bundle.get_workspace(13));
    } else {
        m_softmax_opr->exec(nx, attn_weight, wksp_bundle.get_workspace(12));
    }
    // nz
    TensorND nz{wksp_bundle.get_workspace(14).raw_ptr, m_nz_layout};
    TensorND z{othr_bundle.get_workspace(4).raw_ptr, m_z_layout};
    m_bmatmul_opr->param().transposeA = false;
    m_bmatmul_opr->param().transposeB = false;
    if (param.num_heads > 1) {
        m_bmatmul_opr->exec(attn_weight, nv, nz, wksp_bundle.get_workspace(15));
        // z: multihead to norm
        auto relayout_from_multihead = [&](const TensorND& nq, const TensorND& q) {
            size_t batch = nq.layout[0];
            size_t seq = nq.layout[1];
            size_t embeding_size = nq.layout[2];
            TensorLayout layout{
                    {batch / m_heads, m_heads, seq, embeding_size}, nq.layout.dtype};
            layout = layout.dimshuffle({0, 2, 1, 3});
            m_relayout_opr->exec({nq.raw_ptr(), layout}, q);
        };
        if ((param.training == false) and (param.oproj_size == 0)) {
            relayout_from_multihead(nz, out);
        } else {
            relayout_from_multihead(nz, z);
        }
    } else if ((param.training == false) and (param.oproj_size == 0)) {
        m_bmatmul_opr->exec(attn_weight, nv, out, wksp_bundle.get_workspace(15));
    } else {
        m_bmatmul_opr->exec(attn_weight, nv, z, wksp_bundle.get_workspace(15));
    }

    // o
    TensorND o;
    TensorND mask2{mask_bundle.get_workspace(1).raw_ptr, m_mask2_layout};
    m_matmul_opr->param().transposeA = false;
    m_matmul_opr->param().transposeB = false;
    if (param.oproj_size) {
        if (param.training) {
            o = TensorND{wksp_bundle.get_workspace(16).raw_ptr, m_out_layout};
        } else {
            o = out;
        }
        TensorND oweight{qkvo_weight_bias.ptr<T>() + m_wo_off, m_wo_layout};
        matmul_exec(m_matmul_opr, z, oweight, o, wksp_bundle.get_workspace(17));
        if (param.obias) {
            m_add_opr->exec(o, {qkvo_weight_bias.ptr<T>() + m_bo_off, m_bo_layout});
        }
    } else {
        o = TensorND{z.raw_ptr(), m_z_layout};
    }
    if (param.training) {
        m_dropout_opr->param().drop_prob = param.out_prob;
        m_dropout_opr->exec(o, out, mask2, wksp_bundle.get_workspace(18));
    }
}
}  // namespace multi_head_attn
}  // namespace megdnn

// vim: syntax=cpp.doxygen
