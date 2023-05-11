#include "src/common/multi_head_attn/proxy_backward_base.h"
#include "megdnn/basic_types.h"
#include "megdnn/oprs/nn.h"

namespace megdnn {

namespace multi_head_attn {

bool MHABackwardProxyBase::layout_ismatch(MHA_PROXY_BACKWARD_LAYOUT_CONST_PARAM) {
    MEGDNN_MARK_USED_VAR(handle);
    MEGDNN_MARK_USED_VAR(diff);
    MEGDNN_MARK_USED_VAR(attn_mask);
    MEGDNN_MARK_USED_VAR(attn_weight);
    MEGDNN_MARK_USED_VAR(mask_reservespace);
    MEGDNN_MARK_USED_VAR(othr_reservespace);
    MEGDNN_MARK_USED_VAR(qkvo_weight_bias);
    MEGDNN_MARK_USED_VAR(dqueries);
    MEGDNN_MARK_USED_VAR(dkeys);
    MEGDNN_MARK_USED_VAR(dvalues);
    MEGDNN_MARK_USED_VAR(dqkvo_weight_bias);
    MEGDNN_MARK_USED_VAR(dbias_k);
    MEGDNN_MARK_USED_VAR(dbias_v);
    if (m_matmul_opr == nullptr or m_bmatmul_opr == nullptr or m_add_opr == nullptr or
        m_elem_opr == nullptr or m_reduce_opr == nullptr or
        m_softmaxbw_opr == nullptr or m_dropout_opr == nullptr or
        m_dropoutbw_opr == nullptr or m_relayout_opr == nullptr) {
        megdnn_assert(
                m_matmul_opr == nullptr and m_bmatmul_opr == nullptr and
                        m_add_opr == nullptr and m_elem_opr == nullptr and
                        m_reduce_opr == nullptr and m_softmaxbw_opr == nullptr and
                        m_dropout_opr == nullptr and m_dropoutbw_opr == nullptr and
                        m_relayout_opr == nullptr,
                "All the sub-opr are either not constructed or all constructed, but "
                "now only a part is constructed.");
        m_matmul_opr = handle->create_operator<MatrixMulForward>();
        m_bmatmul_opr = handle->create_operator<BatchedMatrixMul>();
        m_add_opr = handle->create_operator<AddUpdate>();
        m_elem_opr = handle->create_operator<Elemwise>();
        m_reduce_opr = handle->create_operator<Reduce>();
        m_softmaxbw_opr = handle->create_operator<SoftmaxBackward>();
        m_dropout_opr = handle->create_operator<Dropout>();
        m_dropoutbw_opr = handle->create_operator<DropoutBackward>();
        m_relayout_opr = handle->create_operator<Relayout>();
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

    auto equal_metadata = [&](const Param& param) -> bool {
        return m_head == param.num_heads && m_embed_size == param.embeding_size &&
               m_ksize == param.k_size && m_vsize == param.v_size &&
               m_qproj_size == param.qproj_size && m_kproj_size == param.kproj_size &&
               m_vproj_size == param.vproj_size && m_oproj_size == param.oproj_size &&
               m_qbias == param.qbias && m_kbias == param.kbias &&
               m_vbias == param.vbias && m_obias == param.obias;
    };

    return equal_metadata(param) && m_datatype == queries.dtype.enumv() &&
           matmul_layout(
                   queries, m_wq_layout, m_grad_q_layout, param.qproj_size != 0) &&
           matmul_layout(keys, m_wk_layout, m_grad_k_layout, param.kproj_size != 0) &&
           matmul_layout(values, m_wv_layout, m_grad_v_layout, param.vproj_size != 0) &&
           diff.eq_layout(m_grad_out_layout) && diff.eq_layout(m_grad_drop2_layout) &&
           dqueries.eq_layout(m_grad_qin_layout) &&
           dkeys.eq_layout(m_grad_kin_layout) && dvalues.eq_layout(m_grad_vin_layout);
}

void MHABackwardProxyBase::layout_refill(MHA_PROXY_BACKWARD_LAYOUT_CONST_PARAM) {
    MEGDNN_MARK_USED_VAR(attn_weight);
    MEGDNN_MARK_USED_VAR(mask_reservespace);
    MEGDNN_MARK_USED_VAR(handle);
    MEGDNN_MARK_USED_VAR(diff);
    MEGDNN_MARK_USED_VAR(attn_mask);
    MEGDNN_MARK_USED_VAR(othr_reservespace);
    MEGDNN_MARK_USED_VAR(dqueries);
    MEGDNN_MARK_USED_VAR(dkeys);
    MEGDNN_MARK_USED_VAR(dvalues);
    MEGDNN_MARK_USED_VAR(dqkvo_weight_bias);
    MEGDNN_MARK_USED_VAR(dbias_k);
    MEGDNN_MARK_USED_VAR(dbias_v);
    // proxy opr
    m_softmaxbw_opr->param().axis = -1;
    m_matmul_opr->param().format = param::MatrixMul::Format::DEFAULT;
    m_bmatmul_opr->param().format = param::MatrixMul::Format::DEFAULT;
    m_dropoutbw_opr->param().seed = param.seed;
    m_dropout_opr->param().seed = param.seed;
    m_reduce_opr->param().mode = param::Reduce::Mode::SUM;
    m_reduce_opr->param().data_type = param::Reduce::DataType::DEFAULT;

    m_head = param.num_heads;
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
    auto cal_type = qkvo_weight_bias.dtype;
    m_grad_qin_layout = queries;
    m_grad_kin_layout = keys;
    m_grad_vin_layout = values;

    auto reflash_dtype = [&](DType dtype) {
        m_grad_drop2_layout.dtype = dtype;
        m_grad_out_layout.dtype = dtype;
        m_grad_z_layout.dtype = dtype;
        m_grad_wo_layout.dtype = dtype;
        m_grad_bo_layout.dtype = dtype;
        m_grad_nz_layout.dtype = dtype;
        m_grad_nv_layout.dtype = dtype;
        m_grad_ny_layout.dtype = dtype;
        m_grad_drop1_layout.dtype = dtype;
        m_grad_nx_layout.dtype = dtype;
        m_grad_nq_layout.dtype = dtype;
        m_grad_nk_layout.dtype = dtype;
        m_grad_q_layout.dtype = dtype;
        m_grad_k_layout.dtype = dtype;
        m_grad_v_layout.dtype = dtype;
        m_grad_qin_layout.dtype = dtype;
        m_grad_wq_layout.dtype = dtype;
        m_grad_bq_layout.dtype = dtype;
        m_grad_kin_layout.dtype = dtype;
        m_grad_wk_layout.dtype = dtype;
        m_grad_bk_layout.dtype = dtype;
        m_grad_vin_layout.dtype = dtype;
        m_grad_wv_layout.dtype = dtype;
        m_grad_bv_layout.dtype = dtype;
    };
    reflash_dtype(queries.dtype);
    m_datatype = queries.dtype.enumv();
#define cb(DType)                                             \
    if (queries.dtype.enumv() == DTypeTrait<DType>::enumv) {  \
        m_sizeof_datatype = sizeof(DTypeTrait<DType>::ctype); \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb

    // weight/bias
    m_wq_layout = TensorLayout{{m_embed_size, m_qproj_size}, cal_type};
    m_wk_layout = TensorLayout{{m_ksize, m_kproj_size}, cal_type};
    m_wv_layout = TensorLayout{{m_vsize, m_vproj_size}, cal_type};
    if (m_vproj_size > 0) {
        m_wo_layout = TensorLayout{{m_vproj_size, m_oproj_size}, cal_type};
    } else {
        m_wo_layout = TensorLayout{{m_vsize * m_head, m_oproj_size}, cal_type};
    }
    m_bq_layout = TensorLayout{{1, 1, m_qproj_size}, cal_type};
    m_bk_layout = TensorLayout{{1, 1, m_kproj_size}, cal_type};
    m_bv_layout = TensorLayout{{1, 1, m_vproj_size}, cal_type};
    m_bo_layout = TensorLayout{{1, 1, m_oproj_size}, cal_type};

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

    // q/k/v
    m_matmul_opr->param().transposeA = false;
    m_matmul_opr->param().transposeB = false;
    if (param.qproj_size) {
        matmul_deduce_layout(m_matmul_opr, queries, m_wq_layout, m_grad_q_layout);
        m_grad_nq_layout = TensorLayout{
                {m_grad_q_layout.shape[0] * m_head, m_grad_q_layout.shape[1],
                 m_grad_q_layout.shape[2] / m_head},
                m_grad_q_layout.dtype};
    } else {
        m_grad_q_layout = queries;
        m_grad_nq_layout = TensorLayout{
                {m_grad_q_layout[0] * m_head, m_grad_q_layout[1], m_grad_q_layout[2]},
                m_grad_q_layout.dtype};
    }
    if (param.kproj_size) {
        matmul_deduce_layout(m_matmul_opr, keys, m_wk_layout, m_grad_k_layout);
        m_grad_nk_layout = TensorLayout{
                {m_grad_k_layout.shape[0] * m_head, m_grad_k_layout.shape[1],
                 m_grad_k_layout.shape[2] / m_head},
                m_grad_k_layout.dtype};
    } else {
        m_grad_k_layout = keys;
        m_grad_nk_layout = TensorLayout{
                {m_grad_k_layout[0] * m_head, m_grad_k_layout[1], m_grad_k_layout[2]},
                m_grad_k_layout.dtype};
    }
    if (param.vproj_size) {
        matmul_deduce_layout(m_matmul_opr, values, m_wv_layout, m_grad_v_layout);
        m_grad_nv_layout = TensorLayout{
                {m_grad_v_layout.shape[0] * m_head, m_grad_v_layout.shape[1],
                 m_grad_v_layout.shape[2] / m_head},
                m_grad_v_layout.dtype};
    } else {
        m_grad_v_layout = values;
        m_grad_nv_layout = TensorLayout{
                {m_grad_v_layout[0] * m_head, m_grad_v_layout[1], m_grad_v_layout[2]},
                m_grad_v_layout.dtype};
    }

    // nx
    m_bmatmul_opr->param().transposeA = false;
    m_bmatmul_opr->param().transposeB = true;
    m_bmatmul_opr->deduce_layout(m_grad_nq_layout, m_grad_nk_layout, m_grad_nx_layout);
    m_bmatmul_opr->param().transposeA = false;
    m_bmatmul_opr->param().transposeB = false;
    m_grad_nq_workspacesize = m_bmatmul_opr->get_workspace_in_bytes(
            m_grad_nx_layout, m_grad_nk_layout, m_grad_nq_layout);
    m_bmatmul_opr->param().transposeA = true;
    m_bmatmul_opr->param().transposeB = false;
    m_grad_nk_workspacesize = m_bmatmul_opr->get_workspace_in_bytes(
            m_grad_nx_layout, m_grad_nq_layout, m_grad_nk_layout);
    // softmax
    m_grad_ny_layout = m_grad_nx_layout;
    m_grad_nx_workspacesize = m_softmaxbw_opr->get_workspace_in_bytes(
            m_grad_nx_layout, m_grad_ny_layout, m_grad_nx_layout);
    // dropout
    m_dropout_opr->param().drop_prob = param.attn_prob;
    m_dropoutbw_opr->param().drop_prob = param.attn_prob;
    m_dropout_opr->deduce_layout(m_grad_ny_layout, m_grad_drop1_layout, m_mask1_layout);
    m_grad_drop1_workspacesize = m_dropoutbw_opr->get_workspace_in_bytes(
            m_grad_drop1_layout, m_mask1_layout, m_grad_ny_layout);

    // nz
    m_bmatmul_opr->param().transposeA = false;
    m_bmatmul_opr->param().transposeB = false;
    m_bmatmul_opr->deduce_layout(m_grad_ny_layout, m_grad_nv_layout, m_grad_nz_layout);
    m_bmatmul_opr->param().transposeA = false;
    m_bmatmul_opr->param().transposeB = true;
    m_grad_ny_workspacesize = m_bmatmul_opr->get_workspace_in_bytes(
            m_grad_nz_layout, m_grad_nv_layout, m_grad_ny_layout);
    m_bmatmul_opr->param().transposeA = true;
    m_bmatmul_opr->param().transposeB = false;
    m_grad_nv_workspacesize = m_bmatmul_opr->get_workspace_in_bytes(
            m_grad_ny_layout, m_grad_nz_layout, m_grad_nv_layout);

    // z
    m_grad_z_layout = TensorLayout{
            {m_grad_nz_layout.shape[0] / m_head, m_grad_nz_layout.shape[1],
             m_grad_nz_layout.shape[2] * m_head},
            m_grad_nz_layout.dtype};

    // out
    m_matmul_opr->param().transposeA = false;
    m_matmul_opr->param().transposeB = false;
    if (param.oproj_size) {
        matmul_deduce_layout(
                m_matmul_opr, m_grad_z_layout, m_wo_layout, m_grad_out_layout);
    } else {
        m_grad_out_layout = m_grad_z_layout;
    }

    // dropout
    m_dropout_opr->param().drop_prob = param.out_prob;
    m_dropoutbw_opr->param().drop_prob = param.out_prob;
    m_dropout_opr->deduce_layout(
            m_grad_out_layout, m_grad_drop2_layout, m_mask2_layout);
    m_grad_drop2_workspacesize = m_dropoutbw_opr->get_workspace_in_bytes(
            m_grad_drop2_layout, m_mask2_layout, m_grad_out_layout);

    // q = qin @ wq + bq
    // k = kin @ wk + bk
    // v = vin @ wv + bv
    m_matmul_opr->param().transposeA = false;
    m_matmul_opr->param().transposeB = true;
    m_grad_z_workspacesize = 0;
    m_grad_qin_workspacesize = 0;
    m_grad_kin_workspacesize = 0;
    m_grad_vin_workspacesize = 0;

    m_bmatmul_opr->param().transposeA = true;
    m_bmatmul_opr->param().transposeB = false;
    m_bmatmul_opr->deduce_layout(m_grad_z_layout, m_grad_out_layout, m_grad_wo_layout);
    m_grad_wo0_workspacesize = m_bmatmul_opr->get_workspace_in_bytes(
            m_grad_z_layout, m_grad_out_layout, m_grad_wo_layout);
    m_reduce_opr->param().axis = 0;
    m_grad_wo1_workspacesize =
            m_reduce_opr->get_workspace_in_bytes(m_grad_wo_layout, m_wo_layout);
    m_grad_bo_layout = m_grad_out_layout;
    m_grad_bo_layout.shape[0] = 1;
    m_grad_bo0_workspacesize =
            m_reduce_opr->get_workspace_in_bytes(m_grad_out_layout, m_grad_bo_layout);
    m_reduce_opr->param().axis = 1;
    m_grad_bo1_workspacesize =
            m_reduce_opr->get_workspace_in_bytes(m_grad_bo_layout, m_bo_layout);

    m_bmatmul_opr->deduce_layout(queries, m_grad_q_layout, m_grad_wq_layout);
    m_grad_wq0_workspacesize = m_bmatmul_opr->get_workspace_in_bytes(
            queries, m_grad_q_layout, m_grad_wq_layout);
    m_reduce_opr->param().axis = 0;
    m_grad_wq1_workspacesize =
            m_reduce_opr->get_workspace_in_bytes(m_grad_wq_layout, m_wq_layout);
    m_grad_bq_layout = m_grad_q_layout;
    m_grad_bq_layout.shape[0] = 1;
    m_grad_bq0_workspacesize =
            m_reduce_opr->get_workspace_in_bytes(m_grad_q_layout, m_grad_bq_layout);
    m_reduce_opr->param().axis = 1;
    m_grad_bq1_workspacesize =
            m_reduce_opr->get_workspace_in_bytes(m_grad_bq_layout, m_bq_layout);

    m_bmatmul_opr->deduce_layout(keys, m_grad_k_layout, m_grad_wk_layout);
    m_grad_wk0_workspacesize = m_bmatmul_opr->get_workspace_in_bytes(
            keys, m_grad_k_layout, m_grad_wk_layout);
    m_reduce_opr->param().axis = 0;
    m_grad_wk1_workspacesize =
            m_reduce_opr->get_workspace_in_bytes(m_grad_wk_layout, m_wk_layout);
    m_grad_bk_layout = m_grad_k_layout;
    m_grad_bk_layout.shape[0] = 1;
    m_grad_bk0_workspacesize =
            m_reduce_opr->get_workspace_in_bytes(m_grad_k_layout, m_grad_bk_layout);
    m_reduce_opr->param().axis = 1;
    m_grad_bk1_workspacesize =
            m_reduce_opr->get_workspace_in_bytes(m_grad_bk_layout, m_bk_layout);

    m_bmatmul_opr->deduce_layout(values, m_grad_v_layout, m_grad_wv_layout);
    m_grad_wv0_workspacesize = m_bmatmul_opr->get_workspace_in_bytes(
            values, m_grad_v_layout, m_grad_wv_layout);
    m_reduce_opr->param().axis = 0;
    m_grad_wv1_workspacesize =
            m_reduce_opr->get_workspace_in_bytes(m_grad_wv_layout, m_wv_layout);
    m_grad_bv_layout = m_grad_v_layout;
    m_grad_bv_layout.shape[0] = 1;
    m_grad_bv0_workspacesize =
            m_reduce_opr->get_workspace_in_bytes(m_grad_v_layout, m_grad_bv_layout);
    m_reduce_opr->param().axis = 1;
    m_grad_bv1_workspacesize =
            m_reduce_opr->get_workspace_in_bytes(m_grad_bv_layout, m_bv_layout);

    m_reduce_opr->param().axis = 1;
    m_grad_qin_reduce_workspacesize = m_reduce_opr->get_workspace_in_bytes(
            {{m_grad_nq_layout[0] / m_head, m_head, m_grad_nq_layout[1],
              m_grad_nq_layout[2]},
             m_grad_nq_layout.dtype},
            {{m_grad_nq_layout[0] / m_head, 1, m_grad_nq_layout[1],
              m_grad_nq_layout[2]},
             m_grad_nq_layout.dtype});
    m_grad_kin_reduce_workspacesize = m_reduce_opr->get_workspace_in_bytes(
            {{m_grad_nk_layout[0] / m_head, m_head, m_grad_nk_layout[1],
              m_grad_nk_layout[2]},
             m_grad_nk_layout.dtype},
            {{m_grad_nk_layout[0] / m_head, 1, m_grad_nk_layout[1],
              m_grad_nk_layout[2]},
             m_grad_nk_layout.dtype});
    m_grad_vin_reduce_workspacesize = m_reduce_opr->get_workspace_in_bytes(
            {{m_grad_nv_layout[0] / m_head, m_head, m_grad_nv_layout[1],
              m_grad_nv_layout[2]},
             m_grad_nv_layout.dtype},
            {{m_grad_nv_layout[0] / m_head, 1, m_grad_nv_layout[1],
              m_grad_nv_layout[2]},
             m_grad_nv_layout.dtype});
}

WorkspaceBundle MHABackwardProxyBase::get_mask_reservespace_bundle(
        MHA_PROXY_BACKWARD_LAYOUT_CONST_PARAM, void* ptr) {
    if (!layout_ismatch(MHA_PROXY_BACKWARD_CALL)) {
        layout_refill(MHA_PROXY_BACKWARD_CALL);
    }
    return WorkspaceBundle(
            ptr, {m_mask1_layout.span().dist_byte(), m_mask2_layout.span().dist_byte()},
            4);
}

WorkspaceBundle MHABackwardProxyBase::get_othr_reservespace_bundle(
        MHA_PROXY_BACKWARD_LAYOUT_CONST_PARAM, void* ptr) {
    if (!layout_ismatch(MHA_PROXY_BACKWARD_CALL)) {
        layout_refill(MHA_PROXY_BACKWARD_CALL);
    }
    return WorkspaceBundle(
            ptr,
            {param.num_heads > 1 or param.qproj_size
                     ? m_grad_nq_layout.span().dist_byte()
                     : 0,
             param.num_heads > 1 or param.kproj_size
                     ? m_grad_nk_layout.span().dist_byte()
                     : 0,
             param.num_heads > 1 or param.vproj_size
                     ? m_grad_nv_layout.span().dist_byte()
                     : 0,
             m_grad_nx_layout.span().dist_byte(),
             param.oproj_size ? m_grad_z_layout.span().dist_byte() : 0},
            queries.dtype.size());
}

WorkspaceBundle MHABackwardProxyBase::get_workspace_bundle(
        MHA_PROXY_BACKWARD_LAYOUT_CONST_PARAM, void* ptr) {
    if (!layout_ismatch(MHA_PROXY_BACKWARD_CALL)) {
        layout_refill(MHA_PROXY_BACKWARD_CALL);
    }
    return WorkspaceBundle(
            ptr,
            {m_grad_drop2_layout.span().dist_byte(),
             m_grad_drop2_workspacesize,
             param.oproj_size ? m_grad_z_layout.span().dist_byte() : 0,
             param.oproj_size ? m_grad_wo_layout.span().dist_byte() : 0,
             param.oproj_size ? m_grad_z_workspacesize : 0,
             param.oproj_size ? m_grad_wo0_workspacesize : 0,
             param.oproj_size ? m_grad_wo1_workspacesize : 0,
             (param.oproj_size and param.obias) ? m_grad_bo_layout.span().dist_byte()
                                                : 0,
             (param.oproj_size and param.obias) ? m_grad_bo0_workspacesize : 0,
             (param.oproj_size and param.obias) ? m_grad_bo1_workspacesize : 0,
             param.num_heads > 1 ? m_grad_nz_layout.span().dist_byte() : 0,
             m_grad_ny_layout.span().dist_byte(),
             m_grad_nv_layout.span().dist_byte(),
             m_grad_ny_workspacesize,
             m_grad_nv_workspacesize,
             m_grad_drop1_layout.span().dist_byte(),
             m_grad_drop1_workspacesize,
             m_grad_nx_layout.span().dist_byte(),
             m_grad_nx_workspacesize,
             m_sizeof_datatype,
             m_grad_nq_layout.span().dist_byte(),
             m_grad_nk_layout.span().dist_byte(),
             m_grad_nq_workspacesize,
             m_grad_nk_workspacesize,
             (param.qproj_size and param.num_heads > 1)
                     ? m_grad_q_layout.span().dist_byte()
                     : 0,
             param.qproj_size ? m_grad_wq_layout.span().dist_byte() : 0,
             param.qproj_size ? m_grad_bq_layout.span().dist_byte() : 0,
             param.qproj_size ? m_grad_qin_workspacesize : 0,
             param.qproj_size ? m_grad_wq0_workspacesize : 0,
             param.qproj_size ? m_grad_wq1_workspacesize : 0,
             (param.qproj_size and param.qbias) ? m_grad_bq0_workspacesize : 0,
             (param.qproj_size and param.qbias) ? m_grad_bq1_workspacesize : 0,
             param.qproj_size == 0 ? m_grad_qin_reduce_workspacesize : 0,
             (param.kproj_size and param.num_heads > 1)
                     ? m_grad_k_layout.span().dist_byte()
                     : 0,
             param.kproj_size ? m_grad_wk_layout.span().dist_byte() : 0,
             param.kproj_size ? m_grad_bk_layout.span().dist_byte() : 0,
             param.kproj_size ? m_grad_kin_workspacesize : 0,
             param.kproj_size ? m_grad_wk0_workspacesize : 0,
             param.kproj_size ? m_grad_wk1_workspacesize : 0,
             (param.kproj_size and param.kbias) ? m_grad_bk0_workspacesize : 0,
             (param.kproj_size and param.kbias) ? m_grad_bk1_workspacesize : 0,
             param.kproj_size == 0 ? m_grad_kin_reduce_workspacesize : 0,
             (param.vproj_size and param.num_heads > 1)
                     ? m_grad_v_layout.span().dist_byte()
                     : 0,
             param.vproj_size ? m_grad_wv_layout.span().dist_byte() : 0,
             param.vproj_size ? m_grad_bv_layout.span().dist_byte() : 0,
             param.vproj_size ? m_grad_vin_workspacesize : 0,
             param.vproj_size ? m_grad_wv0_workspacesize : 0,
             param.vproj_size ? m_grad_wv1_workspacesize : 0,
             (param.vproj_size and param.vbias) ? m_grad_bv0_workspacesize : 0,
             (param.vproj_size and param.vbias) ? m_grad_bv1_workspacesize : 0,
             param.vproj_size == 0 ? m_grad_vin_reduce_workspacesize : 0});
}

size_t MHABackwardProxyBase::get_mask_reservespace_in_bytes(
        MHA_PROXY_BACKWARD_LAYOUT_CONST_PARAM) {
    auto bundle = get_mask_reservespace_bundle(MHA_PROXY_BACKWARD_CALL);
    return bundle.total_size_in_bytes();
}

size_t MHABackwardProxyBase::get_othr_reservespace_in_bytes(
        MHA_PROXY_BACKWARD_LAYOUT_CONST_PARAM) {
    auto bundle = get_othr_reservespace_bundle(MHA_PROXY_BACKWARD_CALL);
    return bundle.total_size_in_bytes();
}

size_t MHABackwardProxyBase::get_workspace_in_bytes(
        MHA_PROXY_BACKWARD_LAYOUT_CONST_PARAM) {
    auto bundle = get_workspace_bundle(MHA_PROXY_BACKWARD_CALL);
    return bundle.total_size_in_bytes();
}

void MHABackwardProxyBase::exec(MHA_PROXY_BACKWARD_EXEC_PARAM) {
#define cb(DType)                                                   \
    if (queries.layout.dtype.enumv() == DTypeTrait<DType>::enumv) { \
        using ctype = typename DTypeTrait<DType>::ctype;            \
        exec_internal<ctype>(MHA_PROXY_BACKWARD_CALL, workspace);   \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
}

template <typename T>
void MHABackwardProxyBase::exec_internal(MHA_PROXY_BACKWARD_EXEC_PARAM) {
    auto wksp_bundle = get_workspace_bundle(
            MHA_PROXY_BACKWARD_TENSOR_TO_LAYOUT_CALL, workspace.raw_ptr);
    auto mask_bundle = get_mask_reservespace_bundle(
            MHA_PROXY_BACKWARD_TENSOR_TO_LAYOUT_CALL, mask_reservespace.raw_ptr());
    auto othr_bundle = get_othr_reservespace_bundle(
            MHA_PROXY_BACKWARD_TENSOR_TO_LAYOUT_CALL, othr_reservespace.raw_ptr());
    size_t head = param.num_heads;
    size_t one = 1;
    TensorND mask1{mask_bundle.get_workspace(0).raw_ptr, m_mask1_layout};
    TensorND mask2{mask_bundle.get_workspace(1).raw_ptr, m_mask2_layout};
    TensorND nq, nk, nv;
    if (param.qproj_size == 0 and param.num_heads == 1) {
        nq = queries;
    } else {
        nq = TensorND{othr_bundle.get_workspace(0).raw_ptr, m_grad_nq_layout};
    }
    if (param.kproj_size == 0 and param.num_heads == 1) {
        nk = keys;
    } else {
        nk = TensorND{othr_bundle.get_workspace(1).raw_ptr, m_grad_nk_layout};
    }
    if (param.vproj_size == 0 and param.num_heads == 1) {
        nv = values;
    } else {
        nv = TensorND{othr_bundle.get_workspace(2).raw_ptr, m_grad_nv_layout};
    }
    TensorND nx{othr_bundle.get_workspace(3).raw_ptr, m_grad_nx_layout};

    // out = dropout(out)
    TensorND grad_drop2{wksp_bundle.get_workspace(0).raw_ptr, m_grad_drop2_layout};
    m_dropoutbw_opr->param().drop_prob = param.out_prob;
    m_dropoutbw_opr->exec(diff, mask2, grad_drop2, wksp_bundle.get_workspace(1));

    // out = z @ wo + bo
    TensorND grad_z;
    if (param.oproj_size) {
        TensorND z{othr_bundle.get_workspace(4).raw_ptr, m_grad_z_layout};
        TensorND oweight{qkvo_weight_bias.ptr<T>() + m_wo_off, m_wo_layout};
        grad_z = TensorND{wksp_bundle.get_workspace(2).raw_ptr, m_grad_z_layout};
        TensorND grad_wo{wksp_bundle.get_workspace(3).raw_ptr, m_grad_wo_layout};
        m_matmul_opr->param().transposeA = false;
        m_matmul_opr->param().transposeB = true;
        matmul_exec(
                m_matmul_opr, grad_drop2, oweight, grad_z,
                wksp_bundle.get_workspace(4));
        m_bmatmul_opr->param().transposeA = true;
        m_bmatmul_opr->param().transposeB = false;
        m_bmatmul_opr->exec(z, grad_drop2, grad_wo, wksp_bundle.get_workspace(5));
        std::swap(m_grad_wo_layout.shape[0], one);
        TensorND doweight{dqkvo_weight_bias.ptr<T>() + m_wo_off, m_grad_wo_layout};
        std::swap(m_grad_wo_layout.shape[0], one);
        m_reduce_opr->param().axis = 0;
        m_reduce_opr->exec(grad_wo, doweight, wksp_bundle.get_workspace(6));
        if (param.obias) {
            TensorND dobias{dqkvo_weight_bias.ptr<T>() + m_bo_off, m_bo_layout};
            TensorND grad_bo{wksp_bundle.get_workspace(7).raw_ptr, m_grad_bo_layout};
            m_reduce_opr->exec(grad_drop2, grad_bo, wksp_bundle.get_workspace(8));
            m_reduce_opr->param().axis = 1;
            m_reduce_opr->exec(grad_bo, dobias, wksp_bundle.get_workspace(9));
        }
    } else {
        grad_z = grad_drop2;
    }

    // z = nz
    TensorND grad_nz;
    if (param.num_heads > 1) {
        grad_nz = TensorND{wksp_bundle.get_workspace(10).raw_ptr, m_grad_nz_layout};
        auto to_multihead_layout = [&](size_t head,
                                       const TensorLayout& layout) -> TensorLayout {
            size_t batch = layout.shape[0];
            size_t seq = layout.shape[1];
            size_t embeding_size = layout.shape[2];
            TensorLayout ret;
            ret = TensorLayout{{batch, seq, head, embeding_size / head}, layout.dtype};
            ret = ret.dimshuffle({0, 2, 1, 3});
            return ret;
        };
        m_relayout_opr->exec(
                {grad_z.raw_ptr(), to_multihead_layout(head, grad_z.layout)}, grad_nz);
    } else {
        grad_nz = grad_z;
    }

    // nz = ny @ nv
    TensorND grad_ny{wksp_bundle.get_workspace(11).raw_ptr, m_grad_ny_layout};
    TensorND grad_nv{wksp_bundle.get_workspace(12).raw_ptr, m_grad_nv_layout};
    m_bmatmul_opr->param().transposeA = false;
    m_bmatmul_opr->param().transposeB = true;
    m_bmatmul_opr->exec(grad_nz, nv, grad_ny, wksp_bundle.get_workspace(13));
    m_bmatmul_opr->param().transposeA = true;
    m_bmatmul_opr->param().transposeB = false;
    m_bmatmul_opr->exec(attn_weight, grad_nz, grad_nv, wksp_bundle.get_workspace(14));

    // ny = dropout(ny)
    TensorND grad_drop1{wksp_bundle.get_workspace(15).raw_ptr, m_grad_drop1_layout};
    m_dropoutbw_opr->param().drop_prob = param.attn_prob;
    m_dropoutbw_opr->exec(grad_ny, mask1, grad_drop1, wksp_bundle.get_workspace(16));
    // ny = softmax(nx)
    TensorND grad_nx{wksp_bundle.get_workspace(17).raw_ptr, m_grad_nx_layout};
    m_softmaxbw_opr->param().axis = -1;
    m_softmaxbw_opr->exec(nx, grad_drop1, grad_nx, wksp_bundle.get_workspace(18));
    // nx = nx * scaler
    T* d_scaler = wksp_bundle.get_workspace(19).ptr<T>();
    T param_scaler = static_cast<T>(param.sm_scaler);
    move_scaler_to_device(handle, d_scaler, &param_scaler);
    m_elem_opr->param().mode = Elemwise::Mode::MUL;
    m_elem_opr->exec(
            {grad_nx, TensorND{d_scaler, {{1}, queries.layout.dtype}}}, grad_nx);

    // nx = nq @ nk
    TensorND grad_nq{wksp_bundle.get_workspace(20).raw_ptr, m_grad_nq_layout};
    TensorND grad_nk{wksp_bundle.get_workspace(21).raw_ptr, m_grad_nk_layout};
    m_bmatmul_opr->param().transposeA = false;
    m_bmatmul_opr->param().transposeB = false;
    m_bmatmul_opr->exec(grad_nx, nk, grad_nq, wksp_bundle.get_workspace(22));
    m_bmatmul_opr->param().transposeA = true;
    m_bmatmul_opr->param().transposeB = false;
    m_bmatmul_opr->exec(grad_nx, nq, grad_nk, wksp_bundle.get_workspace(23));

    // nq, nk, nv = q, k, v
    auto from_multihead_layout = [&](size_t head,
                                     const TensorLayout& layout) -> TensorLayout {
        size_t batch = layout.shape[0];
        size_t seq = layout.shape[1];
        size_t embeding_size = layout.shape[2];
        TensorLayout ret;
        ret = TensorLayout{{batch / head, head, seq, embeding_size}, layout.dtype};
        ret = ret.dimshuffle({0, 2, 1, 3});
        return ret;
    };
    TensorND grad_k;
    TensorND grad_q;
    TensorND grad_v;

    // q = qin @ wq + bq
    if (param.qproj_size) {
        if (param.num_heads > 1) {
            grad_q = TensorND{wksp_bundle.get_workspace(24).raw_ptr, m_grad_q_layout};
            m_relayout_opr->exec(
                    {grad_nq.raw_ptr(), from_multihead_layout(head, m_grad_nq_layout)},
                    grad_q);
        } else {
            grad_q = grad_nq;
        }
        TensorND qweight{qkvo_weight_bias.ptr<T>() + m_wq_off, m_wq_layout};
        TensorND grad_wq{wksp_bundle.get_workspace(25).raw_ptr, m_grad_wq_layout};
        TensorND grad_bq{wksp_bundle.get_workspace(26).raw_ptr, m_grad_bq_layout};
        m_matmul_opr->param().transposeA = false;
        m_matmul_opr->param().transposeB = true;
        matmul_exec(
                m_matmul_opr, grad_q, qweight, dqueries, wksp_bundle.get_workspace(27));

        m_bmatmul_opr->param().transposeA = true;
        m_bmatmul_opr->param().transposeB = false;
        m_bmatmul_opr->exec(queries, grad_q, grad_wq, wksp_bundle.get_workspace(28));
        std::swap(m_grad_wq_layout.shape[0], one);
        TensorND dqweight{dqkvo_weight_bias.ptr<T>() + m_wq_off, m_grad_wq_layout};
        std::swap(m_grad_wq_layout.shape[0], one);
        m_reduce_opr->param().axis = 0;
        m_reduce_opr->exec(grad_wq, dqweight, wksp_bundle.get_workspace(29));
        if (param.qbias) {
            TensorND dqbias{dqkvo_weight_bias.ptr<T>() + m_bq_off, m_bq_layout};
            m_reduce_opr->exec(grad_q, grad_bq, wksp_bundle.get_workspace(30));
            m_reduce_opr->param().axis = 1;
            m_reduce_opr->exec(grad_bq, dqbias, wksp_bundle.get_workspace(31));
        }
    } else {
        m_reduce_opr->param().axis = 1;
        grad_nq.layout = TensorLayout{
                {grad_nq.layout[0] / head, head, grad_nq.layout[1], grad_nq.layout[2]},
                grad_nq.layout.dtype};
        m_reduce_opr->exec(
                grad_nq,
                {dqueries.raw_ptr(),
                 {{dqueries.layout[0], 1, dqueries.layout[1], dqueries.layout[2]},
                  dqueries.layout.dtype}},
                wksp_bundle.get_workspace(32));
    }

    // k = kin @ wk + bk
    if (param.kproj_size) {
        if (param.num_heads > 1) {
            grad_k = TensorND{wksp_bundle.get_workspace(33).raw_ptr, m_grad_k_layout};
            m_relayout_opr->exec(
                    {grad_nk.raw_ptr(), from_multihead_layout(head, m_grad_nk_layout)},
                    grad_k);
        } else {
            grad_k = grad_nk;
        }

        TensorND kweight{qkvo_weight_bias.ptr<T>() + m_wk_off, m_wk_layout};
        TensorND grad_wk{wksp_bundle.get_workspace(34).raw_ptr, m_grad_wk_layout};
        TensorND grad_bk{wksp_bundle.get_workspace(35).raw_ptr, m_grad_bk_layout};
        m_matmul_opr->param().transposeA = false;
        m_matmul_opr->param().transposeB = true;
        matmul_exec(
                m_matmul_opr, grad_k, kweight, dkeys, wksp_bundle.get_workspace(36));
        m_bmatmul_opr->param().transposeA = true;
        m_bmatmul_opr->param().transposeB = false;
        m_bmatmul_opr->exec(keys, grad_k, grad_wk, wksp_bundle.get_workspace(37));
        std::swap(m_grad_wk_layout.shape[0], one);
        TensorND dkweight{dqkvo_weight_bias.ptr<T>() + m_wk_off, m_grad_wk_layout};
        std::swap(m_grad_wk_layout.shape[0], one);
        m_reduce_opr->param().axis = 0;
        m_reduce_opr->exec(grad_wk, dkweight, wksp_bundle.get_workspace(38));
        if (param.kbias) {
            TensorND dkbias{dqkvo_weight_bias.ptr<T>() + m_bk_off, m_bk_layout};
            m_reduce_opr->exec(grad_k, grad_bk, wksp_bundle.get_workspace(39));
            m_reduce_opr->param().axis = 1;
            m_reduce_opr->exec(grad_bk, dkbias, wksp_bundle.get_workspace(40));
        }
    } else {
        m_reduce_opr->param().axis = 1;
        grad_nk.layout = TensorLayout{
                {grad_nk.layout[0] / head, head, grad_nk.layout[1], grad_nk.layout[2]},
                grad_nk.layout.dtype};
        m_reduce_opr->exec(
                grad_nk,
                {dkeys.raw_ptr(),
                 {{dkeys.layout[0], 1, dkeys.layout[1], dkeys.layout[2]},
                  dkeys.layout.dtype}},
                wksp_bundle.get_workspace(41));
    }

    // v = vin @ wv + bv
    if (param.vproj_size) {
        if (param.num_heads > 1) {
            grad_v = TensorND{wksp_bundle.get_workspace(42).raw_ptr, m_grad_v_layout};
            m_relayout_opr->exec(
                    {grad_nv.raw_ptr(), from_multihead_layout(head, m_grad_nv_layout)},
                    grad_v);
        } else {
            grad_v = grad_nv;
        }

        TensorND vweight{qkvo_weight_bias.ptr<T>() + m_wv_off, m_wv_layout};
        TensorND grad_wv{wksp_bundle.get_workspace(43).raw_ptr, m_grad_wv_layout};
        TensorND grad_bv{wksp_bundle.get_workspace(44).raw_ptr, m_grad_bv_layout};
        m_matmul_opr->param().transposeA = false;
        m_matmul_opr->param().transposeB = true;
        matmul_exec(
                m_matmul_opr, grad_v, vweight, dvalues, wksp_bundle.get_workspace(45));
        m_bmatmul_opr->param().transposeA = true;
        m_bmatmul_opr->param().transposeB = false;
        m_bmatmul_opr->exec(values, grad_v, grad_wv, wksp_bundle.get_workspace(46));
        std::swap(m_grad_wv_layout.shape[0], one);
        TensorND dvweight{dqkvo_weight_bias.ptr<T>() + m_wv_off, m_grad_wv_layout};
        std::swap(m_grad_wv_layout.shape[0], one);
        m_reduce_opr->param().axis = 0;
        m_reduce_opr->exec(grad_wv, dvweight, wksp_bundle.get_workspace(47));
        if (param.vbias) {
            TensorND dvbias{dqkvo_weight_bias.ptr<T>() + m_bv_off, m_bv_layout};
            m_reduce_opr->exec(grad_v, grad_bv, wksp_bundle.get_workspace(48));
            m_reduce_opr->param().axis = 1;
            m_reduce_opr->exec(grad_bv, dvbias, wksp_bundle.get_workspace(49));
        }
    } else {
        m_reduce_opr->param().axis = 1;
        grad_nv.layout = TensorLayout{
                {grad_nv.layout[0] / head, head, grad_nv.layout[1], grad_nv.layout[2]},
                grad_nv.layout.dtype};
        m_reduce_opr->exec(
                grad_nv,
                {dvalues.raw_ptr(),
                 {{dvalues.layout[0], 1, dvalues.layout[1], dvalues.layout[2]},
                  dvalues.layout.dtype}},
                wksp_bundle.get_workspace(50));
    }
}

}  // namespace multi_head_attn
}  // namespace megdnn

// vim: syntax=cpp.doxygen
