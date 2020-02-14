/**
 * \file src/opr-mm/impl/collective_comm.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/collective_comm.h"

#include "megbrain/comp_node_env.h"
#include "megbrain/graph/event.h"
#include "megbrain/graph/grad_impl.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/megray_helper.h"
#include "megbrain/opr/group_manager.h"
#include "megbrain/serialization/sereg.h"
#include "megbrain/version_symbol.h"

using namespace mgb;
using namespace opr;

MGB_DYN_TYPE_OBJ_FINAL_IMPL(CollectiveComm);

#define FOREACH_MODE(cb)                                                   \
    cb(ALL_REDUCE_SUM) cb(ALL_REDUCE_MAX) cb(ALL_REDUCE_MIN) cb(BROADCAST) \
            cb(REDUCE_SUM) cb(ALL_GATHER) cb(REDUCE_SCATTER_SUM)

namespace {

const char* get_param_name(CollectiveComm::Param param) {
    using Mode = CollectiveComm::Param::Mode;
    switch (param.mode) {
#define C(_m)      \
    case Mode::_m: \
        return #_m;
        FOREACH_MODE(C)
#undef C
        default:
            mgb_throw(MegBrainError, "bad CollectiveComm mode");
    }
}

MegRay::DType get_megray_dtype(megdnn::DType dtype) {
    switch(dtype.enumv()) {
        case DTypeEnum::Int8:
            return MegRay::DType::MEGRAY_INT8;
        case DTypeEnum::Int32:
            return MegRay::DType::MEGRAY_INT32;
        case DTypeEnum::Float32:
            return MegRay::DType::MEGRAY_FLOAT32;
#ifndef MEGDNN_DISABLE_FLOAT16
        case DTypeEnum::Float16:
            return MegRay::DType::MEGRAY_FLOAT16;
#endif
        default:
            mgb_throw(MegBrainError, "bad CollectiveComm dtype");
    }
}

MegRay::Backend get_megray_backend(const std::string& backend) {
    if (backend == "nccl") {
        return MegRay::MEGRAY_NCCL;
    } else if (backend == "ucx") {
        return MegRay::MEGRAY_UCX;
    } else {
        mgb_throw(MegBrainError, "back CollectiveComm backend");
    }
}

cudaStream_t get_stream(VarNode* var) {
    return CompNodeEnv::from_comp_node(var->comp_node()).cuda_env().stream;
}
}  // anonymous namespace

class CollectiveComm::ModeTrait {
    class BROADCAST;
    class REDUCE_SUM;
    class REDUCE_SCATTER_SUM;
    class ALL_GATHER;
    class ALL_REDUCE_SUM;
    class ALL_REDUCE_MAX;
    class ALL_REDUCE_MIN;

    class ReducedBasedTrait;
    class AllReduceBase;
    class ReduceBase;

protected:
    using Mode = Param::Mode;

    static void chk_shape_equal(const TensorShapeArray& shp) {
        for (size_t i = 1; i < shp.size(); ++i) {
            mgb_throw_if(!shp[0].eq_shape(shp[i]), GraphError,
                         "input shapes should be equal");
        }
    }

    static void add_output_var_all2all(CollectiveComm* opr) {
        mgb_assert(opr->nr_devices() >= 2);
        auto pname = get_param_name(opr->param());
        // sublinear would setup opr->config if inputs.size() is 1,
        // bypass this situation
        mgb_assert(
                !opr->config().has_comp_node_set() || opr->input().size() == 1,
                "comp node should not be set in %s mode", pname);
        for (auto i : opr->input()) {
            opr->add_output(ssprintf("%s:%s", pname, i->cname()))
                    ->comp_node(i->comp_node());
        }
    }

public:
    virtual ~ModeTrait() = default;

    //! add output var for the opr
    virtual void add_output_var(CollectiveComm* opr,
                                const CompNode::UnorderedSet& inp_cn) = 0;

    /*!
     * \brief the vars on whose comp node the computing should be performed
     * if None, output vars would be used
     */
    virtual Maybe<VarNodeArray> comp_vars(CollectiveComm* opr) {
        return None;
    }

    virtual void get_output_var_shape(const CollectiveComm* opr,
                                      const TensorShapeArray& ishp,
                                      TensorShapeArray& oshp) = 0;

    virtual void exec(CollectiveComm* opr) = 0;

    //! gradient mode
    virtual Mode grad_mode() = 0;

    static ModeTrait& from_mode(Mode mode);
};

class CollectiveComm::ModeTrait::ALL_GATHER : public ModeTrait {
    void add_output_var(CollectiveComm* opr,
                        const CompNode::UnorderedSet&) override {
        add_output_var_all2all(opr);
    }

    void get_output_var_shape(const CollectiveComm* opr,
                              const TensorShapeArray& ishp,
                              TensorShapeArray& oshp) override {
        chk_shape_equal(ishp);
        auto soshp = ishp[0];
        soshp[0] *= opr->nr_devices();
        for (auto& i : oshp)
            i = soshp;
    }

    void exec(CollectiveComm* opr) override {
        auto ivar = opr->input(0), ovar = opr->output(0);
        auto &&iv = ivar->dev_tensor(), &&ov = ovar->dev_tensor();
        mgb_assert(ivar->comp_node().mem_node() ==
                   ovar->comp_node().mem_node());
        auto status = opr->m_megray_comm->all_gather(
                (void*)iv.raw_ptr(), (void*)ov.raw_ptr(),
                iv.shape().total_nr_elems(),
                get_megray_dtype(iv.dtype()),
                opr->megray_ctx());
        mgb_assert(status == MegRay::MEGRAY_OK, "MegRay all_gather failed");
    }

    Mode grad_mode() override { return Mode::REDUCE_SCATTER_SUM; }
};

class CollectiveComm::ModeTrait::REDUCE_SCATTER_SUM : public ModeTrait {
    void add_output_var(CollectiveComm* opr,
                        const CompNode::UnorderedSet&) override {
        add_output_var_all2all(opr);
    }

    void get_output_var_shape(const CollectiveComm* opr,
                              const TensorShapeArray& ishp,
                              TensorShapeArray& oshp) override {
        chk_shape_equal(ishp);
        auto soshp = ishp[0];
        mgb_throw_if(soshp.shape[0] % opr->nr_devices(), GraphError,
                     "input size can not be divided equally: "
                     "size=%zu parts=%zu",
                     soshp[0], ishp.size());
        soshp[0] /= opr->nr_devices();
        for (auto& i : oshp)
            i = soshp;
    }

    void exec(CollectiveComm* opr) override {
        auto ivar = opr->input(0), ovar = opr->output(0);
        auto &&iv = ivar->dev_tensor(), &&ov = ovar->dev_tensor();
        mgb_assert(ivar->comp_node().mem_node() ==
                   ovar->comp_node().mem_node());

        size_t buff_len = ov.shape().total_nr_elems();// * opr->m_nr_devices;
        auto status = opr->m_megray_comm->reduce_scatter(
                (void*)iv.raw_ptr(), (void*)ov.raw_ptr(), buff_len,
                get_megray_dtype(ov.dtype()), MegRay::ReduceOp::MEGRAY_SUM,
                opr->megray_ctx());
        mgb_assert(status == MegRay::MEGRAY_OK, "MegRay reduce_scatter failed");
    }

    Mode grad_mode() override { return Mode::ALL_GATHER; }
};

/* ================= ModeTrait impls ================= */

class CollectiveComm::ModeTrait::ReducedBasedTrait {
protected:
    ~ReducedBasedTrait() = default;

    virtual MegRay::ReduceOp op() const = 0;
};

class CollectiveComm::ModeTrait::AllReduceBase : public ReducedBasedTrait,
                                                   public ModeTrait {
    void add_output_var(CollectiveComm* opr,
                        const CompNode::UnorderedSet&) override {
        add_output_var_all2all(opr);
    }

    void get_output_var_shape(const CollectiveComm*,
                              const TensorShapeArray& ishp,
                              TensorShapeArray& oshp) override {
        chk_shape_equal(ishp);
        oshp = ishp;
    }

    void exec(CollectiveComm* opr) override {
        auto ivar = opr->input(0), ovar = opr->output(0);
        auto &&iv = ivar->dev_tensor(), &&ov = ovar->dev_tensor();
        mgb_assert(ivar->comp_node().mem_node() ==
                   ovar->comp_node().mem_node());
        auto status = opr->m_megray_comm->all_reduce(
                (void*)iv.raw_ptr(), (void*)ov.raw_ptr(),
                iv.shape().total_nr_elems(),
                get_megray_dtype(iv.dtype()), op(),
                opr->megray_ctx());
        mgb_assert(status == MegRay::MEGRAY_OK, "MegRay all_reduce failed");
    }

    Mode grad_mode() override { return Mode::ALL_REDUCE_SUM; }
};

class CollectiveComm::ModeTrait::ALL_REDUCE_SUM final : public AllReduceBase {
    MegRay::ReduceOp op() const override { return MegRay::ReduceOp::MEGRAY_SUM; }
};

class CollectiveComm::ModeTrait::ALL_REDUCE_MAX final : public AllReduceBase {
    MegRay::ReduceOp op() const override { return MegRay::ReduceOp::MEGRAY_MAX; }
};

class CollectiveComm::ModeTrait::ALL_REDUCE_MIN final : public AllReduceBase {
    MegRay::ReduceOp op() const override { return MegRay::ReduceOp::MEGRAY_MIN; }
};

class CollectiveComm::ModeTrait::ReduceBase : public ReducedBasedTrait,
                                                public ModeTrait {
    void add_output_var(CollectiveComm* opr,
                        const CompNode::UnorderedSet& inp_cn) override {
        add_output_var_all2all(opr);
    }

    void get_output_var_shape(const CollectiveComm* opr,
                              const TensorShapeArray& ishp,
                              TensorShapeArray& oshp) override {
        MGB_MARK_USED_VAR(opr);
        chk_shape_equal(ishp);
        if (opr->is_root()) {
            oshp[0] = ishp[0];
        } else {
            oshp[0] = TensorShape{1};
        }
    }

    void exec(CollectiveComm* opr) override {
        auto ovar = opr->output(0);
        auto&& iv = opr->input(0)->dev_tensor();
        void* recvbuf = nullptr;
        if (opr->is_root()) {
            recvbuf = ovar->dev_tensor().raw_ptr();
        }
        auto status = opr->m_megray_comm->reduce(
                (void*)iv.raw_ptr(), recvbuf,
                iv.shape().total_nr_elems(),
                get_megray_dtype(iv.dtype()), op(),
                opr->m_root, opr->megray_ctx());
        mgb_assert(status == MegRay::MEGRAY_OK, "MegRay reduce failed");
    }
};

class CollectiveComm::ModeTrait::REDUCE_SUM final : public ReduceBase {
    MegRay::ReduceOp op() const override { return MegRay::ReduceOp::MEGRAY_SUM; }

    Mode grad_mode() override { return Mode::BROADCAST; }
};

class CollectiveComm::ModeTrait::BROADCAST : public ModeTrait {
    void add_output_var(CollectiveComm* opr,
                        const CompNode::UnorderedSet&) override {
        if (opr->input().size() > 0) {
            add_output_var_all2all(opr);
            return;
        }

        const auto& cns = opr->config().comp_node();
        mgb_assert(cns.size() == 1, "exactly one comp_node expected, got %zu", cns.size());
        auto pname = get_param_name(opr->param());
        opr->add_output(ssprintf("%s:%s", pname, opr->key().c_str()))->comp_node(cns[0]);
    }

    void get_output_var_shape(const CollectiveComm*,
                              const TensorShapeArray& ishp,
                              TensorShapeArray& oshp) override {
        mgb_assert(false, "BROADCAST should not use get_output_var_shape");
    }

    void exec(CollectiveComm* opr) override {
        auto ovar = opr->output(0);
        auto&& ov = ovar->dev_tensor();
        mgb_assert(opr->input().size() < 2,
                   "input size of BROADCAST must be either 0 or 1");
        void* buff;
        DType datatype;
        size_t length;
        if (opr->is_root()) {
            auto ivar = opr->input(0);
            auto&& iv = ivar->dev_tensor();
            datatype = iv.dtype();
            buff = (void*)iv.raw_ptr();
            length = iv.shape().total_nr_elems();
        } else {
            buff = NULL;
            datatype = ov.dtype();
            length = ov.shape().total_nr_elems();
        }
        auto status = opr->m_megray_comm->broadcast(
                buff, (void*)ov.raw_ptr(), length,
                get_megray_dtype(datatype), opr->m_root,
                opr->megray_ctx());
        mgb_assert(status == MegRay::MEGRAY_OK, "MegRay broadcast failed");
    }

    Mode grad_mode() override { return Mode::REDUCE_SUM; }
};

CollectiveComm::ModeTrait& CollectiveComm::ModeTrait::from_mode(Mode mode) {
    switch (mode) {
#define c(_m)          \
    case Mode::_m: {   \
        static _m ins; \
        return ins;    \
    }
        FOREACH_MODE(c)
        default:
            mgb_assert(0);
#undef c
    }
}

/* ================= CollectiveComm ================= */

CollectiveComm::CollectiveComm(
        VarNodeArray inputs, ComputingGraph* const graph,
        const std::string& key, const size_t nr_devices, const uint32_t rank,
        const uint32_t root, std::shared_ptr<GroupClient> group_client,
        const Param& param, const DType& dtype, const std::string& backend,
        const SmallVector<std::shared_ptr<DeviceTensorND>>& dev_buffer_arr,
        const OperatorNodeConfig& config,
        const std::shared_ptr<DTypeScalar>& disable)
        : Super{graph, config, get_param_name(param), inputs},
          m_param{param},
          m_dtype(dtype),
          m_backend(backend),
          m_group_client{std::move(group_client)},
          m_nr_devices(nr_devices),
          m_rank(rank),
          m_key(key),
          m_root(root),
          m_dev_buffers(dev_buffer_arr),
          m_disable{disable} {
    for (auto i : inputs) {
        mgb_assert(i->comp_node().device_type() == CompNode::DeviceType::CUDA,
                   "CollectiveComm currectly only supports CUDA");
    }
    for (auto i : config.comp_node()) {
        mgb_assert(i.device_type() == CompNode::DeviceType::CUDA,
                   "CollectiveComm currectly only supports CUDA");
    }

    CompNode::UnorderedSet inp_cn;
    ThinHashSet<int> inp_dev;

    for (auto i : inputs) {
        add_input({i});
        inp_cn.insert(i->comp_node());
        inp_dev.insert(
                CompNodeEnv::from_comp_node(i->comp_node()).cuda_env().device);
    }
    mgb_assert(
            inp_dev.size() == inputs.size(),
            "CollectiveComm inputs should not contain duplicated input device");

    ModeTrait::from_mode(param.mode).add_output_var(this, inp_cn);
    m_megray_ctx = MegRay::CudaContext::make(get_stream(output(0)));

    add_equivalence_component<PODHash<Param>>(&m_param);
    add_equivalence_component<PODHash<size_t>>(&m_nr_devices);
    m_hash = XXHash{}.update(key.data(), key.size() * sizeof(char)).digest();
    add_equivalence_component<PODHash<size_t>>(&m_hash);
}

SymbolVarArray CollectiveComm::make(
        const SymbolVarArray& inputs, ComputingGraph* const graph,
        const std::string& key, const size_t nr_devices, const uint32_t rank,
               const uint32_t root, std::shared_ptr<GroupClient> group_client,
               const Param& param, const DType& dtype, const std::string& backend,
               const OperatorNodeConfig& config,
        const std::shared_ptr<DTypeScalar>& disable) {
    SmallVector<std::shared_ptr<DeviceTensorND>> dev_buffer_arr(nr_devices,
                                                                nullptr);
    return make(inputs, graph, key, nr_devices, rank, root, group_client,
                dev_buffer_arr, param, dtype, backend, config);
}

SymbolVarArray CollectiveComm::make(
        const SymbolVarArray& inputs, ComputingGraph* const graph,
        const std::string& key, const size_t nr_devices, const uint32_t rank,
               const uint32_t root, std::shared_ptr<GroupClient> group_client,
        const SmallVector<std::shared_ptr<DeviceTensorND>>& dev_buffer_arr,
        const Param& param, const DType& dtype, const std::string& backend,
        const OperatorNodeConfig& config,
        const std::shared_ptr<DTypeScalar>& disable) {
    auto inpvars = cg::to_var_node_array(inputs);
    auto opr = graph->insert_opr(std::make_unique<CollectiveComm>(
            inpvars, graph, key, nr_devices, rank, root, std::move(group_client),
            param, dtype, backend, dev_buffer_arr, config, disable));
    mgb_assert(!opr->output().empty());
    return cg::to_symbol_var_array(opr->output());
}

void CollectiveComm::opr_register() {
    if (m_init)
        return;
    auto&& cuda_env = CompNodeEnv::from_comp_node(output(0)->comp_node())
                                          .cuda_env();

    auto hash = m_group_client->opr_register(m_key, m_nr_devices, m_rank,
            reinterpret_cast<uintptr_t>(cuda_env.stream));

    auto megray_comm_builder =
            owner_graph()
                    ->options()
                    .user_data
                    .get_user_data_or_create<MegRayCommunicatorBuilder>();

    m_megray_comm = megray_comm_builder->get_megray_comm(
            hash, m_key, m_nr_devices, m_rank,
            get_megray_backend(m_backend), m_group_client);

    m_init = true;
}

void CollectiveComm::add_input_layout_constraint() {
    // Enable shape infer *after* static infer phase. This is only used by
    // BROADCAST operation.
    m_enable_shape_infer = true;
    for (auto i : input()) {
        i->add_layout_constraint_contiguous();
    }
}

void CollectiveComm::get_output_var_shape(const TensorShapeArray& inp_shape,
                                            TensorShapeArray& out_shape) const {
    ModeTrait::from_mode(m_param.mode)
            .get_output_var_shape(const_cast<CollectiveComm*>(this),
                                  inp_shape, out_shape);
}

void CollectiveComm::init_output_comp_node() {
    mgb_assert(output().size() == 1, "exactly one output expected, got %zu", output().size());
    owner_graph()->seq_comp_node_optimizer().register_stream_var(output()[0],
        {CompNode::Stream::NCCL, cg::SeqCompNodeOptimizer::StreamPropType::WEAK});
}

void CollectiveComm::init_output_mem_plan(bool dynamic) {
    for (size_t i = 0; i < output().size(); i++) {
        if (m_dev_buffers[i]) {
            output(i)->init_mem_plan(m_dev_buffers[i].get());
        } else {
            if (is_static_var_storage(output(i)) == !dynamic &&
                !output(i)->contain_flag(VarNode::Flag::NO_SYS_MEM_ALLOC))
                output(i)->init_mem_plan();
        }
    }
}

void CollectiveComm::mem_plan_fwd_in2out_writable() {
    if (m_param.mode == Param::Mode::ALL_REDUCE_SUM) {
        for (size_t i = 0; i < output().size(); ++i) {
            output(i)->set_fwd_in2out_writable(input(i));
        }
    }
}

cg::OperatorNodeBase::NodeProp* CollectiveComm::do_make_node_prop() const {
    auto prop = OperatorNodeBase::do_make_node_prop();
    prop->add_flag(NodeProp::Flag::CROSS_COMP_NODE_MEMORY);
    prop->add_flag(NodeProp::Flag::NO_AUTOMATIC_DUP);
    return prop;
}

void CollectiveComm::do_execute(ExecEnv& env) {
    auto&& trait = ModeTrait::from_mode(m_param.mode);
    mgb_assert(owner_graph()->options().async_exec_level,
               "collective comm must be used with async dispatch");
    mgb_assert(output().size() == 1,
               "collective comm only support exactly one output");

    auto disable = m_disable->get_cast<int>();
    if (disable == 1)
        return;
    mgb_assert(disable == 0,
               "disable flag on CollectiveComm can only be 0 or 1,"
               " got %d actually.",
               disable);

    auto cn = output(0)->comp_node();
    auto runner = [this, cn, &trait] {
        opr_register();
        cn.activate();

        owner_graph()->event().signal_inplace<cg::event::BeforeKernel>(this, cn);
        trait.exec(this);
        owner_graph()->event().signal_inplace<cg::event::AfterKernel>(this, cn);

#if CUDART_VERSION < 9000
#pragma message "legacy CUDA; use sync to avoid blocking"
        // nccl hangs occasionally without this sync()
        cn.sync();
#endif
    };
    env.dispatch_on_comp_node(cn, runner);
}

void CollectiveComm::on_output_comp_node_stream_changed() {}

VarNodeArray CollectiveComm::grad(const VarNodeArray& out_grads) const {
    auto mode = ModeTrait::from_mode(m_param.mode).grad_mode();
    SymbolVarArray og_syms;
    if (m_param.mode == Param::Mode::REDUCE_SUM) {
        for (size_t i = 0; i < output().size(); i++) {
            if (out_grads[i])
                og_syms.push_back(out_grads[i]);
        }
        mgb_assert(og_syms.size() == 1);
    } else {
        for (size_t i = 0; i < output().size(); i++) {
            if (!out_grads[i]) {
                mgb_assert(m_param.mode != Param::Mode::REDUCE_SCATTER_SUM,
                           "null out grad in CollctiveCommMM currently "
                           "unsupported when the forward mode is "
                           "Reduce_Scatter_Sum.");
                DTypeScalar dval{output(i)->dtype()};
                dval.set_retain_dtype(0);
                auto zeros =
                        SymbolVar::make_scalar(dval, *output(i)->owner_graph(),
                                               output(i)->comp_node())
                                .broadcast(SymbolVar(output(i)).symshape());
                og_syms.push_back(zeros);
            } else {
                og_syms.push_back(out_grads[i]);
            }
        }
    }

    OperatorNodeConfig::CompNodeArray cn_arr;
    if (m_param.mode == Param::Mode::REDUCE_SUM) {
        for (auto i : input()) {
            cn_arr.push_back(i->comp_node());
        }
    } else if (m_param.mode == Param::Mode::BROADCAST) {
        if (!input().empty()) {
            cn_arr.push_back(input(0)->comp_node());
        }
    }

    auto gvar = CollectiveComm::make(
            og_syms, owner_graph(), m_key + ":grad", m_nr_devices, m_rank, m_root,
            m_group_client, mode, m_dtype, m_backend,
            OperatorNodeConfig{}.comp_node_arr(cn_arr));

    if (m_param.mode == Param::Mode::ALL_REDUCE_MAX) {
        for (size_t i = 0; i < input().size(); ++i) {
            gvar[i] = Elemwise::make({output(i), input(i), gvar[i]},
                                     Elemwise::Mode::COND_LEQ_MOV);
        }
    } else if (m_param.mode == Param::Mode::ALL_REDUCE_MIN) {
        for (size_t i = 0; i < input().size(); ++i) {
            gvar[i] = Elemwise::make({input(i), output(i), gvar[i]},
                                     Elemwise::Mode::COND_LEQ_MOV);
        }
    } else if (m_param.mode == Param::Mode::BROADCAST) {
        if (!input().empty()) {
            CompNode&& master_out_cn = input(0)->comp_node();
            SymbolVarArray rst;
            for (auto i : gvar) {
                if (i.node()->comp_node() == master_out_cn) {
                    mgb_assert(rst.empty());
                    rst.push_back(i);
                }
            }
            gvar = rst;
        }
    }
    return cg::to_var_node_array(gvar);
}

MGB_IMPL_OPR_GRAD(CollectiveComm) {
    return opr.grad(out_grad);
}

void CollectiveComm::init_output_dtype() {
    if (m_dtype.valid()) {
        for (size_t i = 0; i < input().size(); ++i) {
            mgb_assert(m_dtype == input(i)->dtype(),
                       "any given input's dtype should be identical to that "
                       "specified from opr's argument");
        }
        for (auto i : output()) {
            if (!i->dtype().valid())
                i->dtype(m_dtype);
        }
    } else {
        Super::init_output_dtype();
    }
}

void CollectiveComm::init_output_static_infer_desc() {
    if (m_param.mode == Param::Mode::REDUCE_SUM) {
        using namespace cg::static_infer;
        auto&& mgr = owner_graph()->static_infer_manager();

        auto infer_shape_from_input = [](TensorShape& dest, const InpVal& inp_val) {
            dest = inp_val.val[0].shape();
            return true;
        };

        auto infer_shape_constant = [](TensorShape& dest, const InpVal&) {
            dest = TensorShape{1};
            return true;
        };

        mgb_assert(input().size() == 1);
        mgb_assert(output().size() == 1);

        if (is_root()) {
            mgr.register_shape_infer(output(0),
                {SourceType::DEP, {{input(0), DepType::SHAPE}}, infer_shape_from_input});
        } else {
            mgr.register_shape_infer(output(0),
                {SourceType::CONSTANT, {}, infer_shape_constant});
        }

    } else if (m_param.mode == Param::Mode::BROADCAST) {
        using namespace cg::static_infer;
        auto&& mgr = owner_graph()->static_infer_manager();

        auto infer_shape_from_input = [this](TensorShape& dest, const InpVal& inp_val) {
            if (!m_broadcast_output_shape.valid()) {
                m_broadcast_output_shape = inp_val.val[0].shape();
                m_group_client->set_output_shape(m_key, m_broadcast_output_shape.val());
            }
            dest = inp_val.val[0].shape();
            return true;
        };

        auto get_shape_from_server = [this](TensorShape& dest, const InpVal&) {
            if (!m_enable_shape_infer) {
                return false;
            }

            if (!m_broadcast_output_shape.valid()) {
                m_broadcast_output_shape = m_group_client->get_output_shape(m_key);
            }
            dest = m_broadcast_output_shape.val();
            return true;
        };

        mgb_assert(output().size() == 1);

        if (is_root()) {
            mgb_assert(input().size() == 1);
            mgr.register_shape_infer(output(0),
                {SourceType::DEP, {{input(0), DepType::SHAPE}}, infer_shape_from_input});
        } else {
            mgr.register_shape_infer(output(0),
                {SourceType::MUTABLE, {}, get_shape_from_server});
        }

    } else {
        Super::init_output_static_infer_desc();
    }
}

/* ===================== shallow copy ===================== */

namespace mgb {
namespace opr {

cg::OperatorNodeBase* opr_shallow_copy_collective_mm(
        const serialization::OprShallowCopyContext& ctx,
        const cg::OperatorNodeBase& opr_, const VarNodeArray& inputs,
        const OperatorNodeConfig& config) {
    auto&& opr = opr_.cast_final_safe<opr::CollectiveComm>();
    return opr::CollectiveComm::make(to_symbol_var_array(inputs),
                                     ctx.owner_graph(opr_, inputs), opr.key(),
                                     opr.nr_devices(), opr.rank(), opr.root(),
                                     opr.group_client(), opr.dev_buffers(),
                                     opr.param(), opr.dtype(), opr.backend(), config)[0]
            .node()
            ->owner_opr();
}
MGB_REG_OPR_SHALLOW_COPY(CollectiveComm, opr_shallow_copy_collective_mm);

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
