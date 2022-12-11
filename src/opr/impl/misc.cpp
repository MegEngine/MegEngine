#include "./internal/megdnn_opr_wrapper.inl"

#include "megbrain/graph/grad_impl.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/indexing.h"
#include "megbrain/opr/misc.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"

using namespace mgb;
using namespace opr;

namespace mgb {
namespace opr {
namespace intl {
template <>
struct MegDNNOprInitPostCtor<Argmax> {
    static void apply(cg::OperatorNodeBase& opr) {
        opr.output(0)->dtype(dtype::Int32()).add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
    }
};

template <>
struct MegDNNOprInitPostCtor<Argmin> : public MegDNNOprInitPostCtor<Argmax> {};

template <>
struct MegDNNOprInitPostCtor<ArgsortForward> {
    static void apply(cg::OperatorNodeBase& opr) {
        opr.output(0)->dtype(opr.input(0)->dtype());
        opr.output(1)->dtype(dtype::Int32());
    }
};
}  // namespace intl
}  // namespace opr
}  // namespace mgb

/* ================= Argmxx ================= */

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(Argmax) {
    MGB_MARK_USED_VAR(out_grad);
    MGB_MARK_USED_VAR(opr);
    mgb_assert(!wrt_idx);
    return nullptr;
}
#endif

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Argmax);
MEGDNN_OPR_INIT1(Argmax, "argmax")

void Argmax::scn_do_execute() {
    if (input(0)->dev_tensor().empty()) {
        mgb_assert(
                input(0)->dev_tensor().layout().shape[param().axis] != 0,
                "Argmax: expected reduction dim %d to be specified for empty input",
                param().axis);
        mgb_assert(output(0)->dev_tensor().empty());
        return;
    }
    mgb_assert(!output(0)->dev_tensor().empty());
    Super::scn_do_execute();
}

MAKE_NODE_PROP_WITH_ZERO_SHAPE_1(Argmax, 0)

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(Argmin) {
    MGB_MARK_USED_VAR(out_grad);
    MGB_MARK_USED_VAR(opr);
    mgb_assert(!wrt_idx);
    return nullptr;
}
#endif

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Argmin);
MEGDNN_OPR_INIT1(Argmin, "argmin")

void Argmin::scn_do_execute() {
    if (input(0)->dev_tensor().empty()) {
        mgb_assert(
                input(0)->dev_tensor().layout().shape[param().axis] != 0,
                "Argmin: expected reduction dim %d to be specified for empty input",
                param().axis);
        mgb_assert(output(0)->dev_tensor().empty());
        return;
    }
    mgb_assert(!output(0)->dev_tensor().empty());
    Super::scn_do_execute();
}

MAKE_NODE_PROP_WITH_ZERO_SHAPE_1(Argmin, 0)

/* ================= ArgsortForward =================  */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(ArgsortForward);
// MEGDNN_OPR_CTOR_INIT1(ArgsortForward, "argsort")

ArgsortForward::ArgsortForward(
        VarNode* i0, const Param& param, const OperatorNodeConfig& config)
        : Super(OperatorNodeBaseCtorParam{i0->owner_graph(), config, "argsort", {i0}}) {
    init_megdnn_opr(*this, param);
    add_input({i0});
    output(0)->add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);  // sorted value
    output(1)->add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);  // sorted index
    intl::MegDNNOprInitPostCtor<ArgsortForward>::apply(*this);
}

std::array<SymbolVar, 2> ArgsortForward::make(
        SymbolVar in_tensor, const Param& param, const OperatorNodeConfig& config) {
    auto node = in_tensor.node()->owner_graph()->insert_opr(
            std::make_unique<ArgsortForward>(in_tensor.node(), param, config));
    mgb_assert(node->output().size() == 3);
    return {node->output(0), node->output(1)};
}

void ArgsortForward::scn_do_execute() {
    if (input(0)->dev_tensor().empty()) {
        mgb_assert(output(0)->dev_tensor().empty() && output(1)->dev_tensor().empty());
        return;
    }
    mgb_assert(!output(0)->dev_tensor().empty() && !output(1)->dev_tensor().empty());
    Super::scn_do_execute();
}

void ArgsortForward::get_output_var_shape(
        const TensorShapeArray& inp_shape, TensorShapeArray& out_shape) const {
    mgb_assert(inp_shape.size() == 1 && out_shape.size() == 2);
    out_shape[0] = inp_shape[0];
    out_shape[1] = inp_shape[0];
}

ArgsortForward::NodeProp* ArgsortForward::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    ret->add_dep_type_existing_var(input(0), NodeProp::DepType::VALUE_ALLOW_EMPTY);
    return ret;
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(ArgsortForward) {
    mgb_assert(out_grad.size() == 3 && wrt_idx == 0 && !out_grad[2]);
    if (!out_grad[0])
        return nullptr;
    return ArgsortBackward::make(out_grad[0], opr.output(1)).node();
}
#endif

/* ================= ArgsortBackward =================  */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(ArgsortBackward);
MEGDNN_OPR_INIT3(ArgsortBackward, "argsort_bwd", 2, false)

/* ================= Cumprod =================  */

/* ================= Cumsum =================  */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Cumsum);

Cumsum::Cumsum(VarNode* opr, const Param& param, const OperatorNodeConfig& config)
        : Super{opr->owner_graph(), config, "Cumsum", {opr}} {
    init_megdnn_opr(*this, param);
    add_input({opr}, AddInputSortType::CUR_ADDED);
    output(0)->add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(Cumsum) {
    mgb_assert(out_grad[0] && !out_grad[1]);
    auto param = opr.param();
    param.reverse = !param.reverse;
    return Cumsum::make(out_grad[0], param).node();
}
#endif

SymbolVar Cumsum::make(
        SymbolVar opr, const Param& param, const OperatorNodeConfig& config) {
    return opr.insert_single_output_opr<Cumsum>(opr.node(), param, config);
}

void Cumsum::scn_do_execute() {
    if (input(0)->dev_tensor().empty()) {
        mgb_assert(output(0)->dev_tensor().empty());
        return;
    }
    megdnn_opr()->exec(
            input(0)->dev_tensor().as_megdnn(), output(0)->dev_tensor().as_megdnn(),
            intl::get_megdnn_workspace_from_var(output().back()));
}

MAKE_NODE_PROP_WITH_ZERO_SHAPE_1(Cumsum, 0)

void Cumsum::add_input_layout_constraint() {
    input(0)->add_layout_constraint_contiguous();
}

void Cumsum::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto infer_shape = [](TensorShape& dest, const InpVal& iv) {
        auto ishp = iv.val.at(0).shape();
        dest = ishp;
        return true;
    };
    owner_graph()->static_infer_manager().register_shape_infer(
            output(0), {SourceType::DEP, {{input(0), DepType::SHAPE}}, infer_shape});
    auto infer_workspace = [this](TensorShape& dest, const InpVal& iv) {
        auto dtype = input(0)->dtype();
        auto ishp = iv.val.at(0).shape();
        TensorLayout ily(ishp, dtype);
        Param real_param = param();
        if (real_param.axis < 0)
            real_param.axis += ishp.ndim;
        megdnn_opr()->param() = real_param;
        dest.ndim = 1;
        dest[0] = megdnn_opr()->get_workspace_in_bytes(ily, ily);
        return true;
    };
    owner_graph()->static_infer_manager().register_shape_infer(
            output(1),
            {SourceType::DEP, {{input(0), DepType::SHAPE}}, infer_workspace});
}

/* ================= NvOf =================  */

#if MGB_CUDA
MGB_DYN_TYPE_OBJ_FINAL_IMPL(NvOf);

NvOf::NvOf(VarNode* opr, const Param& param, const OperatorNodeConfig& config)
        : Super{opr->owner_graph(), config, "NvOf", {opr}}, m_param{param} {
    mgb_assert(opr->dtype() == dtype::Uint8());
    add_input({opr});
    //! NvOf hava only one output
    add_output(None);
    mgb_log_debug("init nvof engine with precision: %u", m_param.precision);
}

void NvOf::init_output_dtype() {
    output(0)->dtype(dtype::Int16());
}

SymbolVar NvOf::make(
        SymbolVar opr, const Param& param, const OperatorNodeConfig& config) {
    return opr.insert_single_output_opr<NvOf>(opr.node(), param, config);
}

void NvOf::scn_do_execute() {
    auto input_shape = this->input()[0]->shape();
    std::vector<size_t> t_shape;
    for (size_t i = 0; i < 5; i++) {
        t_shape.push_back(input_shape[i]);
    }
    auto c = this->comp_node();
    //! comp_node may init on CUDA or CPU, eg: lar with --cpu
    //! if ON CUDA, need sync, caused by we use different stream
    if (CompNode::DeviceType::CUDA == c.device_type()) {
        c.sync();
    } else {
        mgb_log_warn(
                "NvOf opr on non CUDA comp_node, which will triger H2D and "
                "D2H!!");
    }

    //! create NvOF engine at same device id of comp_node, can not get
    //! comp_node device id, when NvOf:NvOf, so init at scn_do_execute
    std::lock_guard<std::mutex> lock(m_lock);
    if (init_flag == false || vshape != t_shape) {
        vshape = t_shape;
        //! nvof sdk do not imp p2p copy, so init nvof engine on the same
        //! device with mgb comp_node
        nv_flow_extractor = std::make_shared<NVFlowExtractor>(
                c.locator().device, vshape, m_param.precision, true, true);
        init_flag = true;
    }

    nv_flow_extractor->extract_flow(
            static_cast<unsigned char*>(input(0)->dev_tensor().as_megdnn().raw_ptr()),
            vshape,
            reinterpret_cast<int16_t*>(output(0)->dev_tensor().as_megdnn().raw_ptr()));
}

void NvOf::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto infer_shape = [](TensorShape& dest, const InpVal& iv) {
        auto out_grid_size = NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;
        auto ishp = iv.val.at(0).shape();
        //! nvof input format: nthwc4
        mgb_assert(ishp.ndim == 5);
        //! now only support RGBA format channel data
        mgb_assert(ishp[4] == 4);
        SmallVector<size_t> tv;
        tv.push_back(ishp[0]);
        tv.push_back(ishp[1] - 1);
        tv.push_back((ishp[2] + out_grid_size - 1) / out_grid_size);
        tv.push_back((ishp[3] + out_grid_size - 1) / out_grid_size);
        tv.push_back(ishp[4] / 2);
        dest = tv;

        return true;
    };
    owner_graph()->static_infer_manager().register_shape_infer(
            output(0), {SourceType::DEP, {{input(0), DepType::SHAPE}}, infer_shape});
}
#endif

/* ================= CondTake =================  */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(CondTake);

CondTake::CondTake(
        VarNode* data, VarNode* mask, const Param& param,
        const OperatorNodeConfig& config)
        : Super(data->owner_graph(), config, "cond_take", {data, mask}) {
    init_megdnn_opr(*this, param);
    add_input({data, mask});
    auto dtypes = megdnn_opr()->infer_dtype(data->dtype(), mask->dtype());
    for (int i = 0; i < 2; ++i) {
        output(i)
                ->add_flag(VarNode::Flag::NO_SYS_MEM_ALLOC)
                .add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE)
                .dtype(dtypes[i]);
    }
}

CondTake::NodeProp* CondTake::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    ret->add_dep_type_existing_var(input(0), NodeProp::DepType::VALUE_ALLOW_EMPTY);
    ret->add_dep_type_existing_var(input(1), NodeProp::DepType::VALUE_ALLOW_EMPTY);
    return ret;
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(CondTake) {
    mgb_assert(out_grad.size() == 3 && !out_grad[2]);
    if (wrt_idx == 0 && out_grad[0]) {
        SymbolVar data_sym{opr.input(0)};
        auto inp_set = IndexingIncrMultiAxisVec::make(
                data_sym.flatten().fill_retain_dtype(0), out_grad[0],
                {indexing::AxisIndexer::make_index(0, opr.output(1))});
        return inp_set.reshape(data_sym.symshape()).node();
    }
    return nullptr;
}
#endif

std::array<SymbolVar, 2> CondTake::make(
        SymbolVar data, SymbolVar mask, const Param& param,
        const OperatorNodeConfig& config) {
    auto ov0 = data.insert_single_output_opr<CondTake>(
            data.node(), mask.node(), param, config);
    return {ov0, ov0.node()->owner_opr()->output(1)};
}

void CondTake::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto infer_workspace = [this](TensorShape& dest, const InpVal& iv) {
        auto dtype = input(0)->dtype();
        TensorLayout ily(iv.val[0].shape(), dtype);
        dest.ndim = 1;
        TensorLayout mly(iv.val[0].shape(), dtype::Int32());
        dest.shape[0] = megdnn_opr()->get_workspace_in_bytes(ily, mly);
        return true;
    };
    owner_graph()->static_infer_manager().register_shape_infer(
            output(2),
            {SourceType::DEP, {{input(0), DepType::SHAPE}}, infer_workspace});
}

void CondTake::add_input_layout_constraint() {
    mixin::megdnn_utils::add_input_layout_constraint_contig(*this);
}

void CondTake::scn_do_execute() {
    auto&& data = input(0)->dev_tensor();
    auto&& mask = input(1)->dev_tensor();
    intl::MegDNNDynOutMallocImpl dyn_malloc{this, comp_node()};
    if (data.layout().is_empty()) {
        mgb_assert(
                data.layout().eq_shape(mask.layout()),
                "CondTake shape differs: data=%s mask=%s",
                data.layout().TensorShape::to_string().c_str(),
                mask.layout().TensorShape::to_string().c_str());
        dyn_malloc.alloc_output(0, data.layout().dtype, {0}, nullptr);
        dyn_malloc.alloc_output(1, dtype::Int32(), {0}, nullptr);
    } else {
        megdnn_opr()->exec(
                data.as_megdnn(), mask.as_megdnn(),
                intl::get_megdnn_workspace_from_var(output().back()), &dyn_malloc);
    }
}

/* ================= TopK =================  */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(TopK);

TopK::TopK(
        VarNode* data, VarNode* k, const Param& param, const OperatorNodeConfig& config)
        : Super(data->owner_graph(), config, "top_k", {data, k}) {
    init_megdnn_opr(*this, param);
    add_input({data, k});
    if (param.mode == Param::Mode::KTH_ONLY) {
        output(1)
                ->add_flag(VarNode::Flag::VOLATILE_CONTENT)
                .add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
    }
}

std::array<SymbolVar, 2> TopK::make(
        SymbolVar data, SymbolVar k, const Param& param,
        const OperatorNodeConfig& config) {
    auto opr = data.node()->owner_graph()->insert_opr(
            std::make_unique<TopK>(data.node(), k.node(), param, config));
    auto o1 = opr->output(1);
    if (param.mode == Param::Mode::KTH_ONLY) {
        o1 = nullptr;
    }
    return {opr->output(0), o1};
}

void TopK::init_output_dtype() {
    mgb_assert(
            input(1)->dtype() == dtype::Int32{}, "k must be int32, got %s",
            input(1)->dtype().name());
    output(0)->dtype(input(0)->dtype());
    output(1)->dtype(dtype::Int32{});
}

void TopK::add_input_layout_constraint() {
    auto check = [](const TensorLayout& layout) {
        mgb_assert(
                layout.ndim == 2, "top-k input must be two-dim, got %s",
                layout.TensorShape::to_string().c_str());
        return layout.stride[1] == 1;
    };
    input(0)->add_layout_constraint(check);
}

void TopK::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto&& mgr = owner_graph()->static_infer_manager();

    auto infer_oshp0 = [this](TensorShape& dst, const InpVal& iv) {
        auto&& k_tensor = iv.val[1].value();
        mgb_assert(
                k_tensor.shape().is_scalar(), "k must be scalar, got %s",
                k_tensor.shape().to_string().c_str());
        TensorLayout o0, o1;
        megdnn_opr()->deduce_layout(
                k_tensor.ptr<int>()[0], {iv.val[0].shape(), input(0)->dtype()}, o0, o1);
        dst = o0;
        return true;
    };
    mgr.register_shape_infer(
            output(0), {SourceType::DEP,
                        {{input(0), DepType::SHAPE}, {input(1), DepType::VALUE}},
                        infer_oshp0});

    if (param().mode == Param::Mode::KTH_ONLY) {
        mgr.register_shape_infer(output(1), ShapeInferDesc::make_const({}));
    } else {
        mgr.register_shape_infer(output(1), ShapeInferDesc::make_identity(output(0)));
    }

    auto infer_workspace = [this](TensorShape& dst, const InpVal& iv) {
        // active comp_node for cuda launch kernel in get_workspace_in_bytes
        comp_node().activate();
        auto k = iv.val[3].value().ptr<int>()[0];
        auto size = megdnn_opr()->get_workspace_in_bytes(
                k, {iv.val[0].shape(), input(0)->dtype()},
                {iv.val[1].shape(), output(0)->dtype()},
                {iv.val[2].shape(), output(1)->dtype()});
        dst.ndim = 1;
        dst.shape[0] = size;
        return true;
    };
    mgr.register_shape_infer(
            output(2), {SourceType::DEP,
                        {{input(0), DepType::SHAPE},
                         {output(0), DepType::SHAPE},
                         {output(1), DepType::SHAPE},
                         {input(1), DepType::VALUE}},
                        infer_workspace});
}

void TopK::scn_do_execute() {
    auto&& mgr = owner_graph()->static_infer_manager();
    auto k = mgr.infer_value(input(1)).ptr<int>()[0];
    megdnn_opr()->exec(
            k, input(0)->dev_tensor().as_megdnn(), output(0)->dev_tensor().as_megdnn(),
            output(1)->dev_tensor().as_megdnn(),
            intl::get_megdnn_workspace_from_var(output(2)));
}

void TopK::record_execute_deps(ExecDependencyArray& deps) {
    record_megdnn_opr(deps);
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(TopK) {
    // TopK has no gradient on the input k
    if (wrt_idx)
        return nullptr;
    if (opr.param().mode == TopK::Param::Mode::KTH_ONLY) {
        mgb_assert(out_grad[0] && !out_grad[1] && !out_grad[2]);
        auto add_axis = [](SymbolVar x) {
            return opr::AxisAddRemove::make(
                    x, {opr::AxisAddRemove::AxisDesc::make_add(1)});
        };
        SymbolVar mask = opr::eq(add_axis(opr.output(0)), opr.input(0)),
                  og = add_axis(out_grad[0]) / opr::reduce_ax_sum(mask, 1);
        return (og * mask).node();
    }
    if (!out_grad[0])
        return nullptr;
    return ArgsortBackward::make(out_grad[0], opr.output(1), opr.input(0)).node();
}
#endif

/* ================= CheckNonFinite =================  */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(CheckNonFinite);
CheckNonFinite::CheckNonFinite(
        const VarNodeArrayView& inp, const Param& param,
        const OperatorNodeConfig& config)
        : Super(OperatorNodeBaseCtorParam{
                  inp[0]->owner_graph(), config, "check_non_finite", inp}),
          m_scale(param.scale) {
    mgb_assert(!inp.empty());

    for (auto&& i : inp) {
        add_input({i});
        add_output(None)
                ->dtype(dtype::Float32())
                .add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
    }
    add_output(None)->dtype(dtype::Int32()).add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
    cg::add_workspace_output(this);
}

SymbolVarArray CheckNonFinite::make(
        const VarNodeArrayView& inp, const Param& param,
        const OperatorNodeConfig& config) {
    mgb_assert(!inp.empty());
    intl::BatchedDTypePromotion dtp{inp};
    auto outputs =
            inp[0]->owner_graph()
                    ->insert_opr(std::make_unique<CheckNonFinite>(inp, param, config))
                    ->output();
    mgb_assert(outputs.size() == inp.size() + 2);
    SymbolVarArray ret(outputs.size() - 1);
    for (size_t i = 0; i < ret.size(); ++i)
        ret[i] = outputs[i];
    return ret;
}

void CheckNonFinite::scn_do_execute() {
    size_t size = input().size();
    megdnn::TensorNDArray oup_arr(size);
    // copy an outputs to the dnn for inplace
    for (size_t i = 0; i < size; ++i) {
        oup_arr[i] = output(i)
                             ->dev_tensor()
                             .copy_from_fixlayout(input(i)->dev_tensor())
                             .as_megdnn();
    }
    megdnn_opr()->param().scale = m_scale;
    megdnn_opr()->exec(
            oup_arr, output(size)->dev_tensor().as_megdnn(),
            intl::get_megdnn_workspace_from_var(output(size + 1)));
}

void CheckNonFinite::init_output_static_infer_desc() {
    using namespace cg::static_infer;

    auto&& mgr = owner_graph()->static_infer_manager();
    size_t size = input().size();
    for (size_t i = 0; i < size; ++i) {
        mgr.register_shape_infer(output(i), ShapeInferDesc::make_identity(input(i)));
    }
    auto infer_oshp = [](TensorShape& dest, const InpVal& iv) {
        TensorLayout dst;
        dst.shape[0] = 1;
        dst.ndim = 1;
        dst.dtype = dtype::Int32();
        dst.init_contiguous_stride();
        dest = dst;
        return true;
    };
    DepVal deps;
    for (auto i : input())
        deps.push_back({i, DepType::SHAPE});
    mgr.register_shape_infer(output(size), {SourceType::DEP, deps, infer_oshp});

    auto infer_wk = [this](TensorShape& dest, const InpVal& inp) {
        dest.ndim = 1;
        SmallVector<megdnn::TensorLayout> inp_arr(input().size());
        for (size_t i = 0; i < input().size(); ++i) {
            inp_arr[i] = {inp.val.at(i).shape(), input(0)->dtype()};
        }
        dest.shape[0] = megdnn_opr()->get_workspace_in_bytes(
                inp_arr, {output(input().size() + 1)->shape(),
                          output(input().size() + 1)->dtype()});
        return true;
    };
    mgr.register_shape_infer(output(size + 1), {SourceType::DEP, deps, infer_wk});
}

void CheckNonFinite::add_input_layout_constraint() {
    for (auto i : input()) {
        i->add_layout_constraint_contiguous();
    }
}
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
