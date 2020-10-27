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
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/megray_helper.h"
#include "megbrain/opr/group_manager.h"
#include "megbrain/serialization/sereg.h"
#include "megbrain/version_symbol.h"

using namespace mgb;
using namespace opr;

MGB_DYN_TYPE_OBJ_FINAL_IMPL(CollectiveComm);

#define FOREACH_MODE(cb)                                                    \
    cb(ALL_REDUCE_SUM) cb(ALL_REDUCE_MAX) cb(ALL_REDUCE_MIN) cb(BROADCAST)  \
            cb(REDUCE_SUM) cb(ALL_GATHER) cb(REDUCE_SCATTER_SUM) cb(GATHER) \
            cb(SCATTER) cb(ALL_TO_ALL)

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

cudaStream_t get_stream(VarNode* var) {
    return CompNodeEnv::from_comp_node(var->comp_node()).cuda_env().stream;
}
}  // anonymous namespace

/* ================= ModeTrait ================= */

class CollectiveComm::ModeTrait {
    class BROADCAST;
    class REDUCE_SUM;
    class REDUCE_SCATTER_SUM;
    class ALL_GATHER;
    class ALL_REDUCE_SUM;
    class ALL_REDUCE_MAX;
    class ALL_REDUCE_MIN;
    class GATHER;
    class SCATTER;
    class ALL_TO_ALL;

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

public:
    virtual ~ModeTrait() = default;

    /*!
     * \brief the vars on whose comp node the computing should be performed
     * if None, output vars would be used
     */
    virtual Maybe<VarNodeArray> comp_vars(CollectiveComm* opr) {
        return None;
    }

    VarNode* full_grad(VarNode* out_grad, const CollectiveComm* opr) const {
        auto mode = ModeTrait::from_mode(opr->param().mode).grad_mode();
        SymbolVarArray og_syms;

        if (out_grad != nullptr) {
            og_syms.push_back(out_grad);
        }

        auto&& cn = opr->output(0)->comp_node();

        auto gvar = CollectiveComm::make(
                og_syms, opr->owner_graph(), opr->key() + ":grad",
                opr->nr_devices(), opr->is_root(), opr->rank(), false,
                opr->group_client(), mode, opr->dtype(), opr->backend(), {cn});

        return gvar[0].node();
    }

    virtual VarNode* local_grad(VarNode* out_grad, const CollectiveComm* opr) const {
        mgb_throw(MegBrainError,
                  "only all_reduce all_to_all all_gather reduce_scatter "
                  "support local_grad");
    }

    virtual VarNode* grad(VarNode* out_grad, const CollectiveComm* opr) const {
        if (opr->local_grad()){
            return local_grad(out_grad, opr);
        } else {
            return full_grad(out_grad, opr);
        }
    }

    VarNode* zeros(mgb::cg::ComputingGraph &graph, CompNode node, const SymbolVar& shape,
                 DType dtype) const {
        auto zero = SymbolVar::make_scalar(0, graph, node);
        auto zero_tensor = opr::TypeCvt::make(zero, dtype).broadcast(shape);
        return zero_tensor.node();
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

    VarNode* local_grad(VarNode* out_grad, const CollectiveComm* opr) const override {
        auto nr_devices = opr->nr_devices();
        auto rank = opr->rank();
        opr::Subtensor::IndexDesc axis;
        auto shape0 = opr::GetVarShape::make(out_grad, 0);
        axis.push_back({0, shape0 * rank / (int)nr_devices,
                        shape0 * (rank + 1) / (int)nr_devices});
        auto grad = opr::Subtensor::make(out_grad, axis);
        return grad.node();
    }
};

class CollectiveComm::ModeTrait::REDUCE_SCATTER_SUM : public ModeTrait {
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

    VarNode* local_grad(VarNode* out_grad, const CollectiveComm* opr) const override {
        VarNodeArray grads;
        auto zeros_tensor =
                zeros(*out_grad->owner_graph(), out_grad->comp_node(),
                      opr::GetVarShape::make(out_grad), out_grad->dtype());
        for (size_t i = 0;i < opr->nr_devices();i++) {
            if (i == opr->rank()) {
                grads.push_back(out_grad);
            } else {
                grads.push_back(zeros_tensor);
            }
        }
        auto grad = opr::Concat::make(grads, 0);
        return grad.node();
    }
};

class CollectiveComm::ModeTrait::ReducedBasedTrait {
protected:
    ~ReducedBasedTrait() = default;

    virtual MegRay::ReduceOp op() const = 0;
};

class CollectiveComm::ModeTrait::AllReduceBase : public ReducedBasedTrait,
                                                   public ModeTrait {
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

public:
    VarNode* local_grad(VarNode* out_grad,
                        const CollectiveComm* opr) const override {
        return out_grad;
    }
};

class CollectiveComm::ModeTrait::ALL_REDUCE_SUM final : public AllReduceBase {
    MegRay::ReduceOp op() const override { return MegRay::ReduceOp::MEGRAY_SUM; }
};

class CollectiveComm::ModeTrait::ALL_REDUCE_MAX final : public AllReduceBase {
    MegRay::ReduceOp op() const override { return MegRay::ReduceOp::MEGRAY_MAX; }

    VarNode* grad(VarNode* out_grad, const CollectiveComm* opr) const override {
        VarNode* grad;
        if (opr->local_grad()) {
            grad = local_grad(out_grad, opr);
        } else {
            grad = full_grad(out_grad, opr);
        }

        grad = opr::Elemwise::make({opr->output(0), opr->input(0), grad},
                                   Elemwise::Mode::COND_LEQ_MOV)
                       .node();
        return grad;
    }
};

class CollectiveComm::ModeTrait::ALL_REDUCE_MIN final : public AllReduceBase {
    MegRay::ReduceOp op() const override { return MegRay::ReduceOp::MEGRAY_MIN; }

    VarNode* grad(VarNode* out_grad, const CollectiveComm* opr) const override {
        VarNode* grad;
        if (opr->local_grad()) {
            grad = local_grad(out_grad, opr);
        } else {
            grad = full_grad(out_grad, opr);
        }

        grad = opr::Elemwise::make({opr->input(0), opr->output(0), grad},
                                   Elemwise::Mode::COND_LEQ_MOV)
                       .node();
        return grad;
    }
};

class CollectiveComm::ModeTrait::ReduceBase : public ReducedBasedTrait,
                                                public ModeTrait {
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

    VarNode* grad(VarNode* out_grad, const CollectiveComm* opr) const override {
        VarNode* input = opr->is_root() ? out_grad : nullptr;
        return full_grad(input, opr);
    }

    Mode grad_mode() override { return Mode::BROADCAST; }
};

class CollectiveComm::ModeTrait::BROADCAST : public ModeTrait {
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

class CollectiveComm::ModeTrait::GATHER : public ModeTrait {
    void get_output_var_shape(const CollectiveComm* opr,
                              const TensorShapeArray& ishp,
                              TensorShapeArray& oshp) override {
        MGB_MARK_USED_VAR(opr);
        chk_shape_equal(ishp);
        if (opr->is_root()) {
            oshp[0] = ishp[0];
            oshp[0][0] *= opr->nr_devices();
        } else {
            oshp[0] = TensorShape{1};
        }
    }

    void exec(CollectiveComm* opr) override {
        auto&& iv = opr->input(0)->dev_tensor();
        void* recvbuf = nullptr;
        if (opr->is_root()) {
            recvbuf = opr->output(0)->dev_tensor().raw_ptr();
        }
        auto status = opr->m_megray_comm->gather(
                (void*)iv.raw_ptr(), recvbuf, iv.shape().total_nr_elems(),
                get_megray_dtype(iv.dtype()), opr->m_root, opr->megray_ctx());
        mgb_assert(status == MegRay::MEGRAY_OK, "MegRay gather failed");
    }

    VarNode* grad(VarNode* out_grad, const CollectiveComm* opr) const override {
        VarNode* input = opr->is_root() ? out_grad : nullptr;
        return full_grad(input, opr);
    }

    Mode grad_mode() override { return Mode::SCATTER; }
};

class CollectiveComm::ModeTrait::SCATTER : public ModeTrait {
    void get_output_var_shape(const CollectiveComm* opr,
                              const TensorShapeArray& ishp,
                              TensorShapeArray& oshp) override {
        mgb_throw(MegBrainError, "SCATTER should not use get_output_var_shape");
    }

    void exec(CollectiveComm* opr) override {
        auto&& ov = opr->output(0)->dev_tensor();
        void* sendbuf = nullptr;
        void* recvbuf = ov.raw_ptr();
        if (opr->is_root()) {
            sendbuf = opr->input(0)->dev_tensor().raw_ptr();
        }
        auto status = opr->m_megray_comm->scatter(
                sendbuf, recvbuf, ov.shape().total_nr_elems(),
                get_megray_dtype(ov.dtype()), opr->m_root, opr->megray_ctx());
        mgb_assert(status == MegRay::MEGRAY_OK, "MegRay scatter failed");
    }

    Mode grad_mode() override { return Mode::GATHER; }
};

class CollectiveComm::ModeTrait::ALL_TO_ALL : public ModeTrait {
    void get_output_var_shape(const CollectiveComm* opr,
                              const TensorShapeArray& ishp,
                              TensorShapeArray& oshp) override {
        chk_shape_equal(ishp);
        oshp = ishp;
    }

    void exec(CollectiveComm* opr) override {
        auto&& iv = opr->input(0)->dev_tensor();
        auto&& ov = opr->output(0)->dev_tensor();
        auto status = opr->m_megray_comm->all_to_all(
                (void*)iv.raw_ptr(), (void*)ov.raw_ptr(),
                iv.shape().total_nr_elems() / opr->nr_devices(),
                get_megray_dtype(iv.dtype()), opr->megray_ctx());
        mgb_assert(status == MegRay::MEGRAY_OK, "MegRay all_to_all failed");
    }

    Mode grad_mode() override { return Mode::ALL_TO_ALL; }

    VarNode* local_grad(VarNode* out_grad, const CollectiveComm* opr) const override {
        VarNodeArray grads;
        auto grad_shape = opr::GetVarShape::make(out_grad);
        auto zeros_tensor =
                zeros(*out_grad->owner_graph(), out_grad->comp_node(),
                      grad_shape, out_grad->dtype());

        auto nr_devices = opr->nr_devices();
        auto rank = opr->rank();
        opr::Subtensor::IndexDesc axis;
        auto shape0 = opr::GetVarShape::make(out_grad, 0);
        axis.push_back({0, shape0 * rank / (int)nr_devices,
                        shape0 * (rank + 1) / (int)nr_devices});
        auto sub_grad = opr::Subtensor::make(out_grad, axis);

        return opr::SetSubtensor::make(zeros_tensor, sub_grad, axis).node();
    }
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
        const std::string& key, const size_t nr_devices, const bool is_root,
        const int rank, const bool local_grad,
        std::shared_ptr<GroupClient> group_client, const Param& param,
        const DType& dtype, const std::string& backend,
        const SmallVector<std::shared_ptr<DeviceTensorND>>& dev_buffer_arr,
        const OperatorNodeConfig& config,
        const std::shared_ptr<DTypeScalar>& disable)
        : Super{graph, config, get_param_name(param), inputs},
          m_param{param},
          m_dtype(dtype),
          m_backend(backend),
          m_group_client{std::move(group_client)},
          m_nr_devices(nr_devices),
          m_is_root(is_root),
          m_rank(rank),
          m_local_grad(local_grad),
          m_key(key),
          m_dev_buffers(dev_buffer_arr),
          m_disable{disable} {
    // add input
    mgb_assert(inputs.size() <= 1, "one or zero input expected, got %zu", inputs.size());
    if (inputs.size() > 0) {
        mgb_assert(inputs[0]->comp_node().device_type() == CompNode::DeviceType::CUDA,
                   "CollectiveComm currectly only supports CUDA");
        add_input({inputs[0]});
    }

    // add output
    add_output(ssprintf("%s:%s", get_param_name(param), key.c_str()));

    // set comp node
    const auto& cns = config.comp_node();
    mgb_assert(cns.size() <= 1, "one or zero comp node expected, got %zu", cns.size());
    if (cns.size() > 0) {
        mgb_assert(cns[0].device_type() == CompNode::DeviceType::CUDA,
                   "CollectiveComm currectly only supports CUDA");
        output(0)->comp_node(cns[0]);
    } else {
        output(0)->comp_node(inputs[0]->comp_node());
    }

    // set debug flag
    const char* c_debug = MGB_GETENV("MGE_MM_OPR_DEBUG");
    if (c_debug != nullptr and strcmp(c_debug, "1") == 0) {
        m_debug_mode = true;
    }

    // deduplication
    add_equivalence_component<PODHash<Param>>(&m_param);
    add_equivalence_component<PODHash<size_t>>(&m_nr_devices);
    m_hash = XXHash{}.update(key.data(), key.size() * sizeof(char)).digest();
    add_equivalence_component<PODHash<size_t>>(&m_hash);
}

SymbolVarArray CollectiveComm::make(
        const SymbolVarArray& inputs, ComputingGraph* const graph,
        const std::string& key, const size_t nr_devices, const bool is_root,
        const int rank, const bool local_grad,
        std::shared_ptr<GroupClient> group_client, const Param& param,
        const DType& dtype, const std::string& backend,
        const OperatorNodeConfig& config,
        const std::shared_ptr<DTypeScalar>& disable) {
    SmallVector<std::shared_ptr<DeviceTensorND>> dev_buffer_arr(nr_devices,
                                                                nullptr);
    return make(inputs, graph, key, nr_devices, is_root, rank, local_grad,
                group_client, dev_buffer_arr, param, dtype, backend, config);
}

SymbolVarArray CollectiveComm::make(
        const SymbolVarArray& inputs, ComputingGraph* const graph,
        const std::string& key, const size_t nr_devices, const bool is_root,
        const int rank, const bool local_grad,
        std::shared_ptr<GroupClient> group_client,
        const SmallVector<std::shared_ptr<DeviceTensorND>>& dev_buffer_arr,
        const Param& param, const DType& dtype, const std::string& backend,
        const OperatorNodeConfig& config,
        const std::shared_ptr<DTypeScalar>& disable) {
    auto inpvars = cg::to_var_node_array(inputs);
    auto opr = graph->insert_opr(std::make_unique<CollectiveComm>(
            inpvars, graph, key, nr_devices, is_root, rank, local_grad,
            std::move(group_client), param, dtype, backend, dev_buffer_arr,
            config, disable));
    mgb_assert(!opr->output().empty());
    return cg::to_symbol_var_array(opr->output());
}

void CollectiveComm::opr_register() {
    if (m_init)
        return;

    auto&& comp_node = output(0)->comp_node();
    bool use_cache = output(0)->owner_graph()->options().imperative_proxy_graph;
    struct GroupManager::RegisterInfo reg_info;

    if (use_cache and RegInfoCache::has_info(m_key)) {
        reg_info = RegInfoCache::get_info(m_key);
    } else {
        reg_info = m_group_client->opr_register(
                m_key, m_nr_devices, m_is_root, m_rank,
                comp_node.get_uid());
        if (use_cache) {
            RegInfoCache::set_info(m_key, reg_info);
        }
    }

    m_rank = reg_info.rank;
    m_root = reg_info.root_rank;

    m_megray_comm = MegRayCommBuilder::get_megray_comm(
            reg_info.hash, m_key, m_nr_devices, m_rank,
            get_megray_backend(m_backend), m_group_client);

    m_megray_ctx = MegRay::CudaContext::make(get_stream(output(0)));

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

void CollectiveComm::init_output_comp_node() {}

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

        if (m_debug_mode) {
            mgb_log_debug("collective comm: executing %s, rank = %d, key = %s",
                    cname(), rank(), key().c_str());
        }

        owner_graph()->event().signal_inplace<cg::event::BeforeKernel>(this, cn);
        trait.exec(this);
        owner_graph()->event().signal_inplace<cg::event::AfterKernel>(this, cn);
    };
    env.dispatch_on_comp_node(cn, runner);
}

void CollectiveComm::on_output_comp_node_stream_changed() {}

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
    if (m_param.mode == Param::Mode::BROADCAST ||
        m_param.mode == Param::Mode::SCATTER) {
        using namespace cg::static_infer;
        auto&& mgr = owner_graph()->static_infer_manager();

        auto infer_shape_from_input = [this](TensorShape& dest, const InpVal& inp_val) {
            dest = inp_val.val[0].shape();
            if (m_param.mode == Param::Mode::SCATTER) {
                dest[0] /= nr_devices();
            }
            if (is_root() && !m_output_shape.valid()) {
                m_output_shape = dest;
                m_group_client->set_output_shape(m_key, dest);
            }
            return true;
        };

        auto get_shape_from_server = [this](TensorShape& dest, const InpVal&) {
            if (!m_enable_shape_infer && !owner_graph()->options().imperative_proxy_graph) {
                return false;
            }

            if (!m_output_shape.valid()) {
                m_output_shape = m_group_client->get_output_shape(m_key);
            }

            dest = m_output_shape.val();
            return true;
        };

        mgb_assert(output().size() == 1);

        if (is_root() || input().size() > 0) {
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

VarNode* CollectiveComm::grad(VarNode* out_grad) const {
    return ModeTrait::from_mode(m_param.mode).grad(out_grad, this);
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(CollectiveComm) {
    mgb_assert(out_grad.size() == 1, "CollectiveComm should only have one grad");
    return opr.grad(out_grad[0]);
}
#endif

/* ===================== shallow copy ===================== */

namespace mgb {
namespace opr {

cg::OperatorNodeBase* opr_shallow_copy_collective_mm(
        const serialization::OprShallowCopyContext& ctx,
        const cg::OperatorNodeBase& opr_, const VarNodeArray& inputs,
        const OperatorNodeConfig& config) {
    auto&& opr = opr_.cast_final_safe<opr::CollectiveComm>();
    auto new_opr =
            CollectiveComm::make(
                    to_symbol_var_array(inputs), ctx.owner_graph(opr_, inputs),
                    opr.key(), opr.nr_devices(), opr.is_root(), opr.rank(),
                    opr.local_grad(), opr.group_client(), opr.dev_buffers(),
                    opr.param(), opr.dtype(), opr.backend(), config)[0]
                    .node()
                    ->owner_opr();
    new_opr->cast_final_safe<opr::CollectiveComm>().set_pack_hash(opr.pack_hash());
    return new_opr;
}
MGB_REG_OPR_SHALLOW_COPY(CollectiveComm, opr_shallow_copy_collective_mm);

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
