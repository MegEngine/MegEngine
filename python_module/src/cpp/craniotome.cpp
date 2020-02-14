/**
 * \file python_module/src/cpp/craniotome.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#include "./craniotome.h"
#include "./python_helper.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/graph/grad_impl.h"
#include "megbrain/serialization/sereg.h"

using namespace mgb;
using namespace opr;

MGB_DYN_TYPE_OBJ_FINAL_IMPL(CraniotomeDesc);
MGB_DYN_TYPE_OBJ_FINAL_IMPL(Craniotome);


bool CraniotomeDesc::is_same_st(const mgb::Hashable &rhs) const {
    auto rp = static_cast<const CraniotomeDesc&>(rhs).py_self();
    size_t ref0 = rp->ob_refcnt;
    bool ret;
    {
        PYTHON_GIL;
        Py_INCREF(rp);
        ret = _is_same(rp);
    }
    size_t ref1 = rp->ob_refcnt;
    mgb_assert(ref0 == ref1,
            "reference count changed from %zu to %zu",
            ref0, ref1);
    return ret;
}

size_t CraniotomeDesc::hash() const {
    return _hash();
}


PyObject* CraniotomeDesc::py_self() const {
    if (!m_py_self) {
        PYTHON_GIL;
        PyObject* dst = PyList_New(0);
        mgb_assert(dst);
        PyObjRefKeeper dst_ref{dst};

        Py_INCREF(dst);
        _setup_self(dst);
        mgb_assert(dst->ob_refcnt == 1);

        mgb_assert(PyList_Size(dst) == 1);
        m_py_self = PyList_GetItem(dst, 0);
    }

    return m_py_self;
}

class Craniotome::FuncDelCallbackInvoker final
        : public UserDataContainer::UserData {
    MGB_TYPEINFO_OBJ_DECL;
    SmallVector<Craniotome*> m_oprs;

public:
    ~FuncDelCallbackInvoker() {
        Craniotome* cur_opr = nullptr;
        MGB_MARK_USED_VAR(cur_opr);
        MGB_TRY {
            std::vector<size_t> arr;
            for (auto i : m_oprs) {
                cur_opr = i;
                mgb_assert(i->m_on_graph_compile_called);
                i->m_desc->_on_graph_compile_or_func_del(arr);
                i->m_on_graph_compile_called = false;
            }
        }
        MGB_HANDLE_EXCEPTION_DTOR(
                ssprintf("craniotome opr %s", cur_opr->cname()).c_str());
    }
    void add(Craniotome* opr) { m_oprs.push_back(opr); }
};
MGB_TYPEINFO_OBJ_IMPL(Craniotome::FuncDelCallbackInvoker);

Craniotome::Craniotome(
        mgb::ComputingGraph *graph, std::unique_ptr<CraniotomeDesc> desc,
        const VarNodeArray &inputs, const OperatorNodeConfig &config):
    Super{graph, config, desc->_get_opr_type_name().c_str(), inputs},
    m_node_flag{desc->_node_flag()},
    m_desc{std::move(desc)}
{
    for (auto i: inputs)
        add_input({i});
    m_nr_dev_value_inp = input().size() - m_desc->_get_nr_dev_comp_order_deps();
    m_desc->_get_all_io_vars = [this]() {
        SymbolVarArray ret;
        ret.reserve(input().size() + output().size());
        for (auto i: input())
            ret.push_back(i);
        for (auto i: output())
            ret.push_back(i);
        return ret;
    };

    auto nr_out = m_desc->_get_nr_outputs();
    if (nr_out > 1) {
        for (size_t i = 0, it = nr_out; i < it; ++ i)
            add_output(ssprintf("o%zu", i));
    } else {
        mgb_assert(nr_out == 1,
                "could not create an operator with %zu outputs: %s",
                nr_out, cname());
        add_output(None);
    }
    if (output_no_sys_mem_alloc()) {
        for (auto i: output())
            i->add_flag(VarNode::Flag::NO_SYS_MEM_ALLOC);
    }
    if (m_node_flag & NodeFlag::ALLOW_EMPTY_OUTPUT) {
        for (auto i: output())
            i->add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
    }
    add_equivalence_component<HashableObjPtrWrapper>(m_desc.get());

    // init comp node early because desc may access it
    this->init_output_comp_node();
    m_desc->owner_opr = this;
}

Craniotome::~Craniotome() noexcept {
    if (m_on_graph_compile_called) {
        m_desc->_on_graph_compile_or_func_del({});
        m_on_graph_compile_called = false;
    }
}

Craniotome::NodeProp* Craniotome::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    if (m_node_flag & NodeFlag::DISALLOW_DUPLICATE) {
        ret->add_flag(NodeProp::Flag::NO_AUTOMATIC_DUP);
    }
    if (m_nr_dev_value_inp < input().size()) {
        using DT = NodeProp::DepType;
        SmallVector<DT> dep_types(input().size(), DT::DEV_VALUE);
        for (size_t i = m_nr_dev_value_inp; i < dep_types.size(); ++i) {
            dep_types[i] = DT::DEV_COMP_ORDER;
        }
        ret->reset_dep_type(input(), dep_types);
    }
    return ret;
}

void Craniotome::scn_do_execute() {
    m_prev_inferred_shape.invalidate();
    auto &&env = CompNodeEnv::from_comp_node(comp_node());
    env.activate();
    std::vector<CompGraphCallbackValueProxy> inpval(m_nr_dev_value_inp);
    std::vector<SharedND> outval(output().size());
    auto dest_cn = comp_node();
    for (size_t i = 0; i < inpval.size(); ++ i) {
        auto ivar = input(i);
        if (ivar->comp_node() == dest_cn) {
            inpval[i].setup(input(i)->dev_tensor(), false);
        } else {
            auto tensor = input(i)->dev_tensor();
            tensor.comp_node(dest_cn);
            inpval[i].setup(tensor, false);
        }
    }

    TensorShapeArray orig_shape;
    std::vector<void*> orig_ptr;
    if (output_no_sys_mem_alloc()) {
        for (size_t i = 0; i < outval.size(); ++ i)
            outval[i].assign(output(i));
    } else {
        for (size_t i = 0; i < outval.size(); ++ i) {
            outval[i].assign(output(i)->dev_tensor());
            orig_shape.push_back(output(i)->shape());
            orig_ptr.push_back(output(i)->dev_tensor().raw_ptr());
        }
    }
    m_desc->_execute(inpval, outval);
    mgb_assert(outval.size() == output().size());
    if (!output_no_sys_mem_alloc()) {
        for (size_t i = 0; i < outval.size(); ++ i) {
            mgb_assert(output(i)->shape().eq_shape(orig_shape[i]) &&
                    orig_ptr[i] == output(i)->dev_tensor().raw_ptr(),
                    "%s: shape or ptr of output %zu changed",
                    cname(), i);
        }
    }
}

void Craniotome::get_output_var_shape(const TensorShapeArray &inp_shape,
        TensorShapeArray &out_shape) const {
    TensorShapeVec cvt_ishp(inp_shape.size());
    for (size_t i = 0; i < cvt_ishp.size(); ++ i)
        cvt_ishp[i] = npy::shape2vec(inp_shape[i]);
    auto cvt_oshp = m_desc->_infer_shape(cvt_ishp);
    mgb_assert(cvt_oshp.size() == output().size());
    out_shape.resize(cvt_oshp.size());
    for (size_t i = 0; i < cvt_oshp.size(); ++ i)
        out_shape[i] = npy::vec2shape(cvt_oshp[i]);
}

MGB_IMPL_OPR_GRAD(Craniotome) {
    if (wrt_idx >= opr.nr_dev_value_inp()) {
        return nullptr;
    }
    SymbolVarArray isv(opr.nr_dev_value_inp()), osv(opr.output().size()),
            ogsv(out_grad.size());
    for (size_t i = 0; i < isv.size(); ++i)
        isv[i] = opr.input(i);
    for (size_t i = 0; i < osv.size(); ++i)
        osv[i] = opr.output(i);
    for (size_t i = 0; i < out_grad.size(); ++i)
        ogsv[i] = out_grad[i];

    auto ret = cg::to_var_node_array(const_cast<CraniotomeDesc&>(opr.desc())
                                             ._grad(wrt_idx, isv, osv, ogsv));

    auto update_shape = [&opr](size_t i, VarNode* var) {
        auto inp = opr.input(i);
        if (var && cg::is_static_var_shape(inp) &&
            !cg::is_static_var_shape(var)) {
            var = SymbolVar{var}.reshape(SymbolVar{inp}.symshape()).node();
        }
        return var;
    };
    if (ret.size() != 1) {
        mgb_assert(ret.size() == opr.input().size());
        for (size_t i = 0; i < ret.size(); ++i) {
            ret[i] = update_shape(i, ret[i]);
        }
        return ret;
    }
    return update_shape(wrt_idx, ret[0]);
}

void Craniotome::add_input_layout_constraint() {
    for (auto i : input())
        i->add_layout_constraint_contiguous();

    if (!m_on_graph_compile_called) {
        // check used outputs and call _on_graph_compile
        auto graph = owner_graph();
        auto&& out = output();
        std::vector<size_t> used_outputs;
        used_outputs.reserve(out.size());
        for (size_t i = 0; i < out.size(); ++i) {
            if (!graph->var_receiver_in_current_comp_seq(out[i]).empty()) {
                used_outputs.push_back(i);
            }
        }
        mgb_assert(!used_outputs.empty());
        m_desc->_on_graph_compile_or_func_del(used_outputs);
        auto seq = graph->current_comp_seq();
        if (seq) {
            seq->user_data()
                    .get_user_data_or_create<FuncDelCallbackInvoker>()
                    ->add(this);
        } else {
            mgb_assert(graph->options().eager_evaluation);
        }
        m_on_graph_compile_called = true;
    }
}

SymbolVarArray Craniotome::make(
        std::unique_ptr<CraniotomeDesc> desc,
        const SymbolVarArray &inputs,
        const OperatorNodeConfig &config) {
    VarNodeArray inp_vn(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++ i)
        inp_vn[i] = inputs[i].node();
    ComputingGraph *graph;
    if (!inputs.empty()) {
        graph = inp_vn[0]->owner_graph();
    } else {
        graph = &desc->_get_comp_graph().get();
        mgb_assert(graph);
    }
    auto opr = graph->insert_opr(
            std::make_unique<Craniotome>(
                graph, std::move(desc), inp_vn, config));
    SymbolVarArray rst;
    for (auto i: opr->output())
        rst.push_back(i);
    return rst;
}

void Craniotome::init_output_static_infer_desc() {
    if (!(m_node_flag & NodeFlag::DYNAMIC_OUTPUT_SHAPE)) {
        Super::init_output_static_infer_desc();
    } else if (input().empty()) {
        using namespace cg::static_infer;
        auto &&mgr = owner_graph()->static_infer_manager();
        for (size_t idx = 0; idx < output().size(); ++ idx) {
            auto infer = [this, idx](TensorShape &dest, const InpVal &) {
                if (!m_prev_inferred_shape.valid()) {
                    auto &&shp = m_prev_inferred_shape.emplace();
                    shp.resize(output().size());
                    get_output_var_shape({}, shp);
                }
                dest = m_prev_inferred_shape->at(idx);
                return true;
            };
            mgr.register_shape_infer(output(idx),
                    {SourceType::MUTABLE, {}, infer});
        }
    }
}

void Craniotome::init_output_dtype() {
    PYTHON_GIL;

    auto input_dtypes = PyList_New(input().size()),
         ret = PyList_New(output().size());
    mgb_assert(input_dtypes);
    PyObjRefKeeper input_dtypes_ref{input_dtypes};

    mgb_assert(ret);
    PyObjRefKeeper ret_ref{ret};

    for (size_t i = 0; i < input().size(); ++ i) {
        auto err = PyList_SetItem(input_dtypes, i,
                npy::dtype_mgb2np(input(i)->dtype()));
        mgb_assert(!err);
    }

    // it seems that we need to incref before passing it to swig director method
    Py_INCREF(input_dtypes);
    Py_INCREF(ret);
    if (!m_desc->_init_output_dtype(input_dtypes, ret)) {
        Super::init_output_dtype();
        return;
    }

    mgb_assert(PyList_Check(ret),
            "_init_output_dtype should return list");
    mgb_assert(PyList_Size(ret) == static_cast<Py_ssize_t>(output().size()),
                "_init_output_dtype list size not equal to number of outputs");
    for (size_t i = 0; i < output().size(); ++ i) {
        auto cur = PyList_GetItem(ret, i);
        mgb_assert(cur, "failed to get dtype for output %zu", i);
        output(i)->dtype(npy::dtype_np2mgb(cur));
    }

    mgb_assert(input_dtypes->ob_refcnt == 1);
    mgb_assert(ret->ob_refcnt == 1);
}

// serialization
namespace {

    void craniotome_dumper(
            serialization::OprDumpContext &ctx,
            const cg::OperatorNodeBase &opr) {

        auto &&desc = opr.cast_final_safe<Craniotome>().desc();
        auto result = PyList_New(0);
        mgb_assert(result);
        PyObjRefKeeper result_ref{result};

        Py_INCREF(result);
        desc._setup_serialize_params(result);
        mgb_assert(result->ob_refcnt == 1);

        auto sz = PyList_Size(result);
        mgb_assert(sz >= 1 && sz <= 2);

        auto name_obj = PyList_GetItem(result, 0);
        mgb_assert(name_obj && PyUnicode_Check(name_obj));
        Py_ssize_t name_size;
        const char *name_str = PyUnicode_AsUTF8AndSize(name_obj, &name_size);
        mgb_assert(name_str);

        char *param_str = nullptr;
        Py_ssize_t param_size = 0;
        if (sz == 2) {
            auto param_obj = PyList_GetItem(result, 1);
            mgb_assert(param_obj && PyBytes_Check(param_obj));
            auto err = PyBytes_AsStringAndSize(
                    param_obj, &param_str, &param_size);
            mgb_assert(!err);
        }


        ctx.dump_buf_with_len(name_str, name_size);
        if (param_str) {
            ctx.dump_buf_with_len(param_str, param_size);
        }
    }

    cg::OperatorNodeBase* craniotome_shallow_copy(
            const serialization::OprShallowCopyContext &ctx,
            const cg::OperatorNodeBase &opr, const VarNodeArray &inputs,
            const OperatorNodeConfig &config) {

        MGB_MARK_USED_VAR(ctx);
        auto &&orig_desc = opr.cast_final_safe<Craniotome>().desc();
        std::unique_ptr<CraniotomeDesc> desc;
        mgb_assert(!orig_desc._set_copy_result);
        orig_desc._set_copy_result = [&desc](CraniotomeDesc *r) {
            mgb_assert(!desc);
            desc.reset(r);
        };
        orig_desc._copy();
        mgb_assert(desc);
        orig_desc._set_copy_result = {};

        mgb_assert(&orig_desc != desc.get());
        return Craniotome::make(std::move(desc),
                {inputs.begin(), inputs.end()}, config).at(0).node(
                    )->owner_opr();
    }

    class _RegDumper {
        public:
            _RegDumper() {
                serialization::OprRegistry::add_using_dynamic_loader(
                        Craniotome::typeinfo(), "Craniotome",
                        craniotome_dumper);
                MGB_REG_OPR_SHALLOW_COPY_IMPL(
                        Craniotome, craniotome_shallow_copy);
            }
    };
    _RegDumper _reg_dumper;

} // anonymous namespace

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
