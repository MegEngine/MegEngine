/**
 * \file src/opr/impl/blas.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/blas.h"
#include "megbrain/common.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/graph/grad_impl.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/indexing.h"
#include "megbrain/opr/tensor_gen.h"
#include "megbrain/opr/tensor_manip.h"

#include "./internal/megdnn_opr_wrapper.inl"

using namespace mgb;
using namespace opr;

/* ================= MatrixMul =================  */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(MatrixMul);
MEGDNN_OPR_INIT2(MatrixMul, "matrix_mul")

void MatrixMul::init_output_dtype() {
    DType output_dtype = config().output_dtype();
    megdnn_opr()->deduce_dtype(input(0)->dtype(), input(1)->dtype(),
                               output_dtype);
    output(0)->dtype(output_dtype);
}

bool MatrixMul::check_layout(const TensorLayout& layout, int transpose) {
    mgb_assert(layout.ndim == 2, "input to MatrixMul must be 2-dim; got %s",
               layout.to_string().c_str());
    return layout.stride[0 ^ transpose] >=
                   static_cast<ptrdiff_t>(layout.shape[1 ^ transpose]) &&
           layout.stride[1 ^ transpose] == 1;
}

void MatrixMul::add_input_layout_constraint() {
    auto check = [](const TensorLayout& ly) {
        return check_layout(ly, 0) || check_layout(ly, 1);
    };
    input(0)->add_layout_constraint(check);
    input(1)->add_layout_constraint(check);
}

size_t MatrixMul::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    // we may change transepose param in the impl, so get the max possible
    // workspace by trying all cases
    // current implementation in megdnn guarantees that workspaces in different
    // cases are on the same order of magnitude
    auto mo = megdnn_opr();
    auto&& tparam = mo->param();
    size_t a, b, c, d;
    mgb_assert(input_shapes.size() == 2 && output_shapes.size() == 1);
    TensorLayout i0(input_shapes[0], input(0)->dtype()),
            i1(input_shapes[1], input(1)->dtype()),
            out(output_shapes[0], output(0)->dtype());

    auto transpose = [](TensorLayout& dst, bool& param) {
        std::swap(dst.shape[0], dst.shape[1]);
        dst.stride[0] = dst[1];
        param ^= 1;
    };
    MGB_TRY {
        a = mo->get_workspace_in_bytes(i0, i1, out);
        transpose(i0, tparam.transposeA);
        b = mo->get_workspace_in_bytes(i0, i1, out);
        transpose(i1, tparam.transposeB);
        c = mo->get_workspace_in_bytes(i0, i1, out);
        transpose(i0, tparam.transposeA);
        d = mo->get_workspace_in_bytes(i0, i1, out);
    }
    MGB_FINALLY({ tparam = this->param(); });
    return std::max(std::max(a, b), std::max(c, d));
}

void MatrixMul::scn_do_execute() {
    auto inp0 = input(0)->dev_tensor().as_megdnn(),
         inp1 = input(1)->dev_tensor().as_megdnn(),
         out = output(0)->dev_tensor().as_megdnn();
    auto transpose = [](TensorLayout& layout, bool& trans) {
        if (!check_layout(layout, 0)) {
            mgb_assert(check_layout(layout, 1));
            std::swap(layout.shape[0], layout.shape[1]);
            std::swap(layout.stride[0], layout.stride[1]);
            trans ^= 1;
        }
    };
    auto&& tparam = megdnn_opr()->param();
    MGB_TRY {
        transpose(inp0.layout, tparam.transposeA);
        transpose(inp1.layout, tparam.transposeB);
        megdnn_opr()->exec(inp0, inp1, out,
                           intl::get_megdnn_workspace_from_var(output(1)));
    }
    MGB_FINALLY({ tparam = this->param(); });
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(MatrixMul) {
    mgb_assert(opr.input(0)->dtype().category() == DTypeCategory::FLOAT,
               "only float data type supported for grad");
    SymbolVar grad, i0{opr.input(0)}, i1{opr.input(1)}, og{out_grad[0]};
    if (wrt_idx == 0) {
        // A * B = C, A' = C' * Bt
        if (opr.param().transposeA) {
            grad = MatrixMul::make(i1, og, {opr.param().transposeB, true});
        } else {
            grad = MatrixMul::make(og, i1, {false, !opr.param().transposeB});
        }
    } else {
        mgb_assert(wrt_idx == 1);
        // A * B = C, B' = At * C'
        if (opr.param().transposeB) {
            grad = MatrixMul::make(og, i0, {true, opr.param().transposeA});
        } else {
            grad = MatrixMul::make(i0, og, {!opr.param().transposeA, false});
        }
    }
    return grad.node();
}
#endif

/* ================= BatchedMatrixMul =================  */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(BatchedMatrixMul);
MEGDNN_OPR_INIT2(BatchedMatrixMul, "batched_matrix_mul")

void BatchedMatrixMul::add_input_layout_constraint() {
    auto check = [](const TensorLayout& ly) {
        mgb_assert(ly.ndim == 3,
                   "input to BatchedMatrixMul must be 3-dim; got %s",
                   ly.to_string().c_str());

        bool good_layout =
                ((ly.stride[0] >=
                  static_cast<ptrdiff_t>(ly.shape[1] * ly.stride[1])) &&
                 (ly.stride[0] >=
                  static_cast<ptrdiff_t>(ly.shape[2] * ly.stride[2])));

        bool ret = good_layout &&
                   (check_layout(ly, true) || check_layout(ly, false));
        return ret;
    };
    input(0)->add_layout_constraint(check);
    input(1)->add_layout_constraint(check);
}

void BatchedMatrixMul::init_output_dtype() {
    DType output_dtype = config().output_dtype();
    megdnn_opr()->deduce_dtype(input(0)->dtype(), input(1)->dtype(),
                               output_dtype);
    output(0)->dtype(output_dtype);
}

bool BatchedMatrixMul::check_layout(const TensorLayout& layout,
                                    bool transpose) {
    int lhs = (transpose) ? 2 : 1, rhs = (transpose) ? 1 : 2;
    return (layout.stride[lhs] >= static_cast<ptrdiff_t>(layout.shape[rhs])) &&
           (layout.stride[rhs] == 1);
}

size_t BatchedMatrixMul::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    // we may change transepose param in the impl, so get the max possible
    // workspace by trying all cases
    // current implementation in megdnn guarantees that workspaces in different
    // cases are on the same order of magnitude
    auto mo = megdnn_opr();
    auto&& tparam = mo->param();
    size_t a, b, c, d;
    mgb_assert(input_shapes.size() == 2 && output_shapes.size() == 1);
    TensorLayout i0(input_shapes[0], input(0)->dtype()),
            i1(input_shapes[1], input(1)->dtype()),
            out(output_shapes[0], output(0)->dtype());

    auto transpose = [](TensorLayout& dst, bool& param) {
        std::swap(dst.shape[1], dst.shape[2]);
        dst.stride[1] = dst[2];
        param ^= 1;
    };
    MGB_TRY {
        a = mo->get_workspace_in_bytes(i0, i1, out);
        transpose(i0, tparam.transposeA);
        b = mo->get_workspace_in_bytes(i0, i1, out);
        transpose(i1, tparam.transposeB);
        c = mo->get_workspace_in_bytes(i0, i1, out);
        transpose(i0, tparam.transposeA);
        d = mo->get_workspace_in_bytes(i0, i1, out);
    }
    MGB_FINALLY({ tparam = this->param(); });
    return std::max(std::max(a, b), std::max(c, d));
}

void BatchedMatrixMul::scn_do_execute() {
    auto inp0 = input(0)->dev_tensor().as_megdnn(),
         inp1 = input(1)->dev_tensor().as_megdnn(),
         out = output(0)->dev_tensor().as_megdnn();
    auto transpose = [](TensorLayout& layout, bool& trans) {
        if (!check_layout(layout, false)) {
            mgb_assert(check_layout(layout, true));
            std::swap(layout.shape[1], layout.shape[2]);
            std::swap(layout.stride[1], layout.stride[2]);
            mgb_assert(layout.stride[2] == 1);
            trans ^= 1;
        }
    };
    auto&& tparam = megdnn_opr()->param();
    MGB_TRY {
        transpose(inp0.layout, tparam.transposeA);
        transpose(inp1.layout, tparam.transposeB);
        megdnn_opr()->exec(inp0, inp1, out,
                           intl::get_megdnn_workspace_from_var(output(1)));
    }
    MGB_FINALLY({ tparam = this->param(); });
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(BatchedMatrixMul) {
    mgb_assert(opr.input(0)->dtype().category() == DTypeCategory::FLOAT,
            "only float data type supported for grad");
    mgb_assert(out_grad.size() == 2 && !out_grad[1]);
    SymbolVar grad, i0{opr.input(0)}, i1{opr.input(1)}, og{out_grad[0]};
    if (wrt_idx == 0) {
        // A * B = C, A' = C' * Bt
        if (opr.param().transposeA) {
            grad = BatchedMatrixMul::make(
                    i1, og, {opr.param().transposeB, true});
        } else {
            grad = BatchedMatrixMul::make(
                    og, i1, {false, !opr.param().transposeB});
        }
    } else {
        mgb_assert(wrt_idx == 1);
        // A * B = C, B' = At * C'
        if (opr.param().transposeB) {
            grad = BatchedMatrixMul::make(
                    og, i0, {true, opr.param().transposeA});
        } else {
            grad = BatchedMatrixMul::make(
                    i0, og, {!opr.param().transposeA, false});
        }
    }
    return grad.node();
}
#endif

/* ================= Dot =================  */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Dot);

Dot::Dot(VarNode *opr0, VarNode *opr1, const OperatorNodeConfig &config):
    Super{opr0->owner_graph(), config, "dot", {opr0, opr1}}
{
    init_megdnn_opr(*this, {});
    add_input({opr0, opr1}, AddInputSortType::CUR_ADDED);
    static_assert(std::is_empty<Param>::value, "Dot param should be empty");
    mgb_assert(opr0->dtype().category() != DTypeCategory::QUANTIZED &&
                       opr1->dtype().category() != DTypeCategory::QUANTIZED,
               "Dot does not support quantized input.");
}

void Dot::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto &&mgr = owner_graph()->static_infer_manager();
    auto infer_shp = [](TensorShape &dest, const InpVal &){
        dest = {1};
        return true;
    };
    auto infer_workspace = [this](TensorShape &dest, const InpVal &iv) {
        auto dtype = input(0)->dtype();
        TensorLayout ily(
                {std::max(
                        iv.val[0].shape().total_nr_elems(),
                        iv.val[1].shape().total_nr_elems())},
                dtype);
        dest.ndim = 1;
        dest.shape[0] = megdnn_opr()->get_workspace_in_bytes(
                ily, ily, {{1}, dtype});
        return true;
    };
    mgr.register_shape_infer(output(0), {SourceType::CONSTANT, {}, infer_shp});
    mgr.register_shape_infer(output(1),
            {SourceType::DEP,
            {{input(0), DepType::SHAPE}, {input(1), DepType::SHAPE}},
            infer_workspace});
}

void Dot::scn_do_execute() {
    auto i0 = input(0)->dev_tensor().as_megdnn(),
         i1 = input(1)->dev_tensor().as_megdnn();
    mgb_throw_if(i0.layout.ndim != 1 || i1.layout.ndim != 1, GraphError,
            "Invalid input shapes for Dot: %s",
            cg::dump_var_info(input()).c_str());
    if (i0.layout.shape[0] != i1.layout.shape[0]) {
        bool s0 = i0.layout.shape[0] == 1, s1 = i1.layout.shape[0] == 1;
        mgb_throw_if(!s0 && !s1, GraphError,
                "Invalid input shapes for Dot: %s",
                cg::dump_var_info(input()).c_str());
        if (s0) {
            i0.layout.shape[0] = i1.layout.shape[0];
            i0.layout.stride[0] = 0;
        }
        else {
            i1.layout.shape[0] = i0.layout.shape[0];
            i1.layout.stride[0] = 0;
        }
    }
    megdnn_opr()->exec(i0, i1, output(0)->dev_tensor().as_megdnn(),
            intl::get_megdnn_workspace_from_var(output(1)));
}

void Dot::add_input_layout_constraint() {
    auto check = [](const TensorLayout &ly) {
        mgb_throw_if(ly.ndim != 1, GraphError,
                "Dot input must be 1-dim; got %s", ly.to_string().c_str());
        return ly.stride[0] >= 0;
    };
    input(0)->add_layout_constraint(check);
    input(1)->add_layout_constraint(check);
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(Dot) {
    auto other_input = opr.input(wrt_idx == 0 ? 1 : 0);
    auto ishp0 = opr::GetVarShape::make(opr.input(0)),
         ishp1 = opr::GetVarShape::make(opr.input(1));
    auto max_ishp = opr::GetVarShape::make({opr.input(0), opr.input(1)});
    return reduce_sum(
            Broadcast::make(mul(out_grad[0], other_input), max_ishp),
            wrt_idx ? ishp1 : ishp0).node();
}
#endif

SymbolVar Dot::make(SymbolVar opr0, SymbolVar opr1,
         const OperatorNodeConfig &config) {
    return opr0.insert_single_output_opr<Dot>(opr0.node(), opr1.node(), config);
}

void Dot::record_execute_deps(ExecDependencyArray &deps) {
    record_megdnn_opr(deps);
}

/* ================= MatrixInverse =================  */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(MatrixInverse);
MEGDNN_OPR_INIT1(MatrixInverse, "matrix_inv")

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(MatrixInverse) {
    SymbolVar a = opr.output(0);
    // TODO: use unified MatrixMul interface when we have it
    auto n = opr::Subtensor::make(a.symshape(),
            {opr::Subtensor::AxisIndexer::make_index(0, a.make_scalar(-1))}),
         tshp = opr::Concat::make({a.make_scalar(0), n, n}, 0),
         // our hard disk is limited so derivation of the gradient is omitted:)
         a_bnn = opr::Dimshuffle::make(opr::Reshape::make(a, tshp, 0),
                 {0, 2, 1}),
         dy = opr::Reshape::make(out_grad.at(0), tshp, 0),
         da = - BatchedMatrixMul::make(BatchedMatrixMul::make(a_bnn, dy),
                 a_bnn);
    return da.reshape(a.symshape()).node();
}
#endif

/* ================= SVD =================  */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(SVD);

SVD::SVD(VarNode* src, const Param& param, const OperatorNodeConfig& config) :
        Super(OperatorNodeBaseCtorParam{src->owner_graph(),
                                        config, "svd", {src}}) {
    mgb_assert(src->dtype() == megdnn::dtype::Float32(),
               "Singular Value Decomposition on non-float32 tensors is "
               "not supoorted.");
    init_megdnn_opr(*this, param);
    add_input({src});

    if (!param.compute_uv) {
        output(0)->add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE)
                  .add_flag(VarNode::Flag::VOLATILE_CONTENT);
        output(2)->add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE)
                  .add_flag(VarNode::Flag::VOLATILE_CONTENT);
    }
}

#if MGB_ENABLE_GRAD
namespace {

/*!
 * \brief a wrapper similar to SymbolVar but can safely contain nullptr as zero
 *
 * Note: here we introduce a new class of SymbolVar representation, which allows
 * nullptr to represent zero values, and overload other C++ operators
 * accordingly.  Therefore we can avoid testing nullptr values everywhere in SVD
 * grad.
 *
 * This is a general approach. It can be moved to some header file if we
 * encounter another operator that also has complex gradient computation.
 */
class SafeSymbolVar {
    VarNode* m_node;

public:
    explicit SafeSymbolVar(VarNode* node) : m_node{node} {}

    SafeSymbolVar(SymbolVar x) : m_node{x.node()} {}

    SafeSymbolVar() : m_node{nullptr} {}

    VarNode* node() const { return m_node; }
    SymbolVar s() const { return m_node; }

#define FWD(name)                                                   \
    template <typename... Args>                                     \
    SafeSymbolVar name(Args&&... args) {                            \
        if (!m_node)                                                \
            return {};                                              \
        return SymbolVar{m_node}.name(std::forward<Args>(args)...); \
    }
    FWD(reshape)
    FWD(broadcast)
#undef FWD
};

SymbolVar unsafe(SymbolVar x) {
    return x;
}
SymbolVar unsafe(SafeSymbolVar x) {
    return x.s();
}

template <typename T>
T reshape_anybatch(T x, SymbolVar tshp) {
    if (!x.node())
        return x;
    return opr::Reshape::make(unsafe(x), tshp, 0);
}

template <typename T>
T trans(T x) {
    if (!x.node())
        return x;
    return opr::Dimshuffle::make(unsafe(x), {0, 2, 1});
}

template <typename T>
T matmul(T a, T b, const opr::BatchedMatrixMul::Param& param = {}) {
    if (!a.node() || !b.node())
        return {};
    return opr::BatchedMatrixMul::make(unsafe(a), unsafe(b), param);
}

SafeSymbolVar matmuls(SafeSymbolVar x, SafeSymbolVar y,
                      const opr::BatchedMatrixMul::Param& param = {}) {
    return matmul(x, y, param);
}

SafeSymbolVar operator-(SafeSymbolVar x) {
    if (x.node())
        return -x.s();
    return {};
}

#define OP(x, a_, b_)                                            \
    SafeSymbolVar operator x(SafeSymbolVar a, SafeSymbolVar b) { \
        if (!a.node())                                           \
            return a_;                                           \
        if (!b.node())                                           \
            return b_;                                           \
        return a.s() x b.s();                                    \
    }
OP(+, b, a)
OP(-, -b, a)
OP(*, {}, {})
#undef OP

}  // anonymous namespace
#endif

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(SVD) {
    /**
     * The formula is copied from
     * https://j-towns.github.io/papers/svd-derivative.pdf
     * It is hard to compare m, n here, so I do not refer this paper :
     * http://eprints.maths.ox.ac.uk/1079/1/NA-08-01.pdf
     */
    mgb_throw_if(!opr.param().compute_uv, MegBrainError,
                 "Singular value decomposition gradient computation depends "
                 "on U and V, please set compute_uv = True");
    SymbolVar a{opr.input(0)}, u_raw{opr.output(0)}, s_raw{opr.output(1)},
            vt_raw{opr.output(2)};
    SafeSymbolVar grad_u_raw{out_grad[0]}, grad_s_raw{out_grad[1]},
            grad_vt_raw{out_grad[2]};
    auto param10 = BatchedMatrixMul::Param{true, false},
         param00 = BatchedMatrixMul::Param{false, false},
         param01 = BatchedMatrixMul::Param{false, true};
    auto n = opr::Subtensor::make(a.symshape(),
                                  {opr::Subtensor::AxisIndexer::make_index(
                                          0, a.make_scalar(-1))}),
         m = opr::Subtensor::make(a.symshape(),
                                  {opr::Subtensor::AxisIndexer::make_index(
                                          0, a.make_scalar(-2))}),
         r = opr::Subtensor::make(s_raw.symshape(),
                                  {opr::Subtensor::AxisIndexer::make_index(
                                          0, s_raw.make_scalar(-1))});
    SymbolVar sshp = opr::Concat::make({a.make_scalar(0), r}, 0),
              ushp = opr::Concat::make({a.make_scalar(0), m, r}, 0),
              vtshp = opr::Concat::make({a.make_scalar(0), r, n}, 0),
              u = reshape_anybatch(u_raw, ushp),
              vt = reshape_anybatch(vt_raw, vtshp), v = trans(vt);
    SafeSymbolVar grad_u = reshape_anybatch(grad_u_raw, ushp),
                  grad_vt = reshape_anybatch(grad_vt_raw, vtshp),
                  grad_v = trans(grad_vt);
    auto batches = opr::Subtensor::make(
            u.symshape(),
            {opr::Subtensor::AxisIndexer::make_index(0, u.make_scalar(-3))});
    auto brr = opr::Concat::make({batches, r, r}, 0);
    auto I_r = opr::Eye::make(r, {0, DTypeEnum::Float32})
                       .reshape(opr::Concat::make({a.make_scalar(1), r, r}, 0))
                       .broadcast(brr),
         filter_matrix = 1 - I_r;
    auto sf = reshape_anybatch(s_raw, sshp)
                      .reshape(opr::Concat::make({batches, r, a.make_scalar(1)},
                                                 0))
                      .broadcast(brr);
    auto grad_sf = reshape_anybatch(grad_s_raw, sshp)
                           .reshape(opr::Concat::make(
                                   {batches, r, a.make_scalar(1)}, 0))
                           .broadcast(brr);
    auto s = I_r * sf;
    auto grad_s = I_r * grad_sf;
    auto s_inv = 1 / (s + filter_matrix) - filter_matrix;
    auto s_rhs = sf * sf, s_mid = trans(s_rhs) - s_rhs,
         s_avoid_nan = s_mid + I_r, f = filter_matrix / s_avoid_nan;
    auto I_m = opr::Eye::make(m, {0, DTypeEnum::Float32})
                       .reshape(opr::Concat::make({a.make_scalar(1), m, m}, 0))
                       .broadcast(opr::Concat::make({batches, m, m}, 0)),
         I_n = opr::Eye::make(n, {0, DTypeEnum::Float32})
                       .reshape(opr::Concat::make({a.make_scalar(1), n, n}, 0))
                       .broadcast(opr::Concat::make({batches, n, n}, 0));
    auto ut_du = matmuls(u, grad_u, param10),
         vt_dv = matmuls(v, grad_v, param10);
    auto ret =
            matmuls(matmuls(matmuls(u, f * (ut_du - trans(ut_du))), s,
                            param00) +
                            matmuls(matmuls(I_m - matmul(u, u, param01),
                                            grad_u),
                                    s_inv),
                    v, param01) +
            matmuls(matmuls(u, I_r * grad_s), v, param01) +
            matmuls(u, matmuls(matmuls(s, f * (vt_dv - trans(vt_dv)), param00),
                               v, param01) +
                               matmuls(matmuls(s_inv, grad_v, param01),
                                       I_n - matmul(v, v, param01)));
    return ret.reshape(a.symshape()).node();
}
#endif

SymbolVarArray SVD::make(const SymbolVar& src, const Param& param,
                         const OperatorNodeConfig& config) {
    auto&& out = src.node()
                         ->owner_graph()
                         ->insert_opr(std::make_unique<SVD>(src.node(), param,
                                                            config))
                         ->output();
    SymbolVarArray ret(out.size());
    for (size_t i = 0; i < ret.size(); i++) {
        ret[i] = out[i];
    }
    return ret;
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
