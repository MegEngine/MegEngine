/**
 * \file src/opr/impl/io.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/io.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/graph/event.h"
#include "megbrain/graph/grad_impl.h"
#include "megbrain/graph/exc_extra_info.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/serialization/opr_load_dump.h"

using namespace mgb;
using namespace opr;

namespace {
//! helper for implementing oprs that hold a device tensor value
namespace dv_helper {
void add_output(cg::OperatorNodeBase& opr, DType dtype,
                const Maybe<std::string>& name = None);
void init_output_mem_plan(const DeviceTensorND& val, cg::OperatorNodeBase& opr,
                          bool dynamic, size_t ovar_idx = 0);
void check_in_exec(const DeviceTensorND& val, VarNode* var);
}  // namespace dv_helper
}  // anonymous namespace

/* ===================== dv_helper ===================== */

void dv_helper::add_output(cg::OperatorNodeBase& opr, DType dtype,
                           const Maybe<std::string>& name) {
    mgb_assert(dtype.valid());
    opr.add_output(name)
            ->add_flag(VarNode::Flag::NO_MEM_RECLAIM)
            .add_flag(VarNode::Flag::PERSISTENT_DEVICE_VALUE)
            .add_flag(VarNode::Flag::DISALLOW_RT_FORCE_DYNAMIC_MEM_ALLOC)
            .dtype(dtype);
}

void dv_helper::init_output_mem_plan(const DeviceTensorND& val,
                                     cg::OperatorNodeBase& opr, bool dynamic,
                                     size_t ovar_idx) {
    mgb_assert(!dynamic);
    auto ovar = opr.output(ovar_idx);
    mgb_assert(val.dtype() == ovar->dtype(),
               "dtype mismatch: get=%s expect=%s opr=%s{%s}",
               val.dtype().name(), ovar->dtype().name(), opr.cname(),
               opr.dyn_typeinfo()->name);
    ovar->init_mem_plan(&val);
}

void dv_helper::check_in_exec(const DeviceTensorND& val, VarNode* var) {
    auto&& oval = var->dev_tensor();
    if(!(val.comp_node().mem_node() == oval.comp_node().mem_node() &&
         val.raw_ptr() == oval.raw_ptr() && val.layout().eq_layout(oval.layout())
         && val.dtype() == var->dtype())) {
        var->owner_opr()->owner_graph()->record_async_error(
            cg::OperatorNodeExcExtraInfo::ExcMaker{var->owner_opr()}
            .make_unique<MegBrainError>(ssprintf(
                "value changed in DeviceTensorHolder: cn=(%s,%s), ptr=(%p,%p), "
                "layout=(%s,%s), dtype=(%s,%s)",
                val.comp_node().to_string().c_str(),
                oval.comp_node().to_string().c_str(), val.raw_ptr(),
                oval.raw_ptr(), val.layout().to_string().c_str(),
                oval.layout().to_string().c_str(),
                val.dtype().name(), var->dtype().name())));
    }
}

/* ===================== HostIONodeBase ===================== */

void intl::HostIONodeBase::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto &&mgr = owner_graph()->static_infer_manager();
    auto infer_shp = [this](TensorShape &dest, const InpVal &) -> bool {
        dest = get_output_shape();
        return dest.ndim;
    };

    auto shape_type = static_infer_src_type();
    auto opr_load_ctx = owner_graph()->options().user_data.get_user_data<
        serialization::OprLoadContext>();
    if (opr_load_ctx.second) {
        mgb_assert(opr_load_ctx.second == 1);
        if (opr_load_ctx.first[0]->config().const_var_shape) {
            shape_type = cg::static_infer::SourceType::CONSTANT;
        }
    }
    mgr.register_shape_infer(output(0), {shape_type, {}, infer_shp});

    if (fill_in_static_infer(nullptr)) {
        auto infer_val = [this](DeviceTensorND &dest, const InpVal &) -> bool {
            if (fill_in_static_infer(&dest) && !dest.empty()) {
                return true;
            }
            return false;
        };
        mgr.register_value_infer(output(0),
                {static_infer_src_type(), {}, infer_val});
    }
}

cg::static_infer::SourceType
intl::HostIONodeBase::static_infer_src_type() const {
    return cg::static_infer::SourceType::MUTABLE;
}

/* ===================== DeviceTensorHolder ===================== */

class intl::DeviceTensorHolder::DevValueExecDep final : public ExecDependency {
    DeviceTensorStorage m_val;

public:
    explicit DevValueExecDep(DeviceTensorStorage val) : m_val{std::move(val)} {}
};


void intl::DeviceTensorHolder::init_output_format() {
    auto format = get_dev_tensor().format();
    mgb_assert(format.is_default(), "non-default tensor format: %s",
               format.to_string().c_str());
    // no need to set output foramt since it is initialized as default
}

void intl::DeviceTensorHolder::init_output_mem_plan(bool dynamic) {
    dv_helper::init_output_mem_plan(get_dev_tensor(), *this, dynamic);
}

void intl::DeviceTensorHolder::scn_do_execute() {
    dv_helper::check_in_exec(get_dev_tensor(), output(0));
}

void intl::DeviceTensorHolder::add_output(DType dtype) {
    mgb_assert(output().empty());
    dv_helper::add_output(*this, dtype);
}

void intl::DeviceTensorHolder::record_execute_deps(ExecDependencyArray& deps) {
    if (!output(0)->contain_flag(VarNode::Flag::MEMORY_NO_NEED)) {
        deps.emplace_back(
                std::make_unique<DevValueExecDep>(get_dev_tensor().storage()));
    }
}

/* ===================== Host2DeviceCopy ===================== */

class Host2DeviceCopy::HostValueExecDep final : public ExecDependency {
    std::shared_ptr<HostTensorND> m_hv;
    void* m_ptr;
    TensorShape m_shape;

public:
    explicit HostValueExecDep(std::shared_ptr<HostTensorND> hv)
            : m_hv{hv}, m_ptr{hv->raw_ptr()}, m_shape{hv->shape()} {}

    bool has_runtime_check() const override { return true; }

    void do_runtime_check() override {
        mgb_assert(m_hv->raw_ptr() == m_ptr && m_hv->shape().eq_shape(m_shape),
                   "host tensor changed: %p(%s) vs %p(%s)", m_hv->raw_ptr(),
                   m_hv->shape().to_string().c_str(), m_ptr,
                   m_shape.to_string().c_str());
    }
};

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Host2DeviceCopy);
Host2DeviceCopy::Host2DeviceCopy(ComputingGraph &graph,
        const std::shared_ptr<HostTensorND> &host_data,
        const Param &param,
        const OperatorNodeConfig &config):
    Super{&graph, config, "h2d", {}},
    m_param{param},
    m_host_data{host_data}
{
    auto out_cn = m_host_data->comp_node();
    if (config.has_comp_node_set())
        out_cn = config.get_single_comp_node();
    mgb_assert(out_cn.valid(), "can not get output comp node");

    if (param.allow_cpu_mem_fwd &&
            out_cn.mem_node() == CompNode::default_cpu().mem_node() &&
            host_data->comp_node().mem_node() == out_cn.mem_node()) {
        m_fwd_host_mem = true;
        dv_helper::add_output(*this, host_data->dtype());
    } else {
        m_fwd_host_mem = false;
        add_output(None)->dtype(host_data->dtype());
    }
    add_equivalence_component<ScalarHash<void*>>(host_data.get());
    add_equivalence_component<PODHash<Param>>(&m_param);

    this->comp_node(out_cn);

    output(0)->add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
}

const TensorShape& Host2DeviceCopy::get_output_shape() {
    return m_host_data->shape();
}

bool Host2DeviceCopy::fill_in_static_infer(DeviceTensorND* dest) {
    if (!m_param.enable_value_infer) {
        return false;
    }
    if (!dest) {
        // query whether static infer is supported
        return true;
    }
    if (m_host_data->storage().has_no_real_storage()) {
        return false;
    }
    dest->copy_from(*m_host_data);
    return true;
}

void Host2DeviceCopy::scn_do_execute() {
    if (m_fwd_host_mem) {
        mgb_assert(m_host_data->comp_node().mem_node() ==
                comp_node().mem_node());
        if (m_host_data_dev_cont_need_sync)
            m_host_data_dev_cont.copy_from_fixlayout(*m_host_data);
        dv_helper::check_in_exec(get_dev_tensor_in_mem_fwd(), output(0));
    } else {
        auto&& od = output(0)->dev_tensor();
        od.copy_from_fixlayout(*m_host_data);
    }
}

void Host2DeviceCopy::init_output_mem_plan(bool dynamic) {
    if (m_fwd_host_mem) {
        dv_helper::init_output_mem_plan(get_dev_tensor_in_mem_fwd(), *this,
                                        dynamic);
    } else {
        Super::init_output_mem_plan(dynamic);
    }
}

void Host2DeviceCopy::init_output_comp_node() {
}

const DeviceTensorND& Host2DeviceCopy::get_dev_tensor_in_mem_fwd() const {
    mgb_assert(m_fwd_host_mem);
    if (!m_host_data->layout().is_contiguous()) {
        m_host_data_dev_cont_need_sync = true;
        m_host_data_dev_cont.comp_node(comp_node()).
            dtype(m_host_data->dtype()).
            resize(m_host_data->shape());
        return m_host_data_dev_cont;
    }
    m_host_data_dev_cont_need_sync = false;

    m_host_data_dev_proxy = DeviceTensorND::make_proxy(*m_host_data);
    return m_host_data_dev_proxy;
}

cg::OperatorNodeBase::NodeProp* Host2DeviceCopy::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    if (m_fwd_host_mem) {
        ret->add_flag(NodeProp::Flag::IMPURE_OUTPUT_MEM_PLAN);
    }
    return ret;
}

SymbolVar Host2DeviceCopy::make(ComputingGraph &graph,
        const std::shared_ptr<HostTensorND> &host_data,
        const Param &param,
        const OperatorNodeConfig &config) {
    return graph.insert_opr(std::make_unique<Host2DeviceCopy>(
                graph, host_data, param, config))->output(0);
}

void Host2DeviceCopy::record_execute_deps(ExecDependencyArray& deps) {
    deps.emplace_back(
            std::make_unique<HostValueExecDep>(std::move(m_host_data)));
}

/* ===================== SharedDeviceTensor related ===================== */

intl::SharedDeviceTensorBase::SharedDeviceTensorBase(
        ComputingGraph& graph, const std::shared_ptr<DeviceTensorND>& dev_data,
        bool const_value, const OperatorNodeConfig& config)
        : Super{&graph, config, "shared", {}},
          m_dev_data{dev_data},
          m_const_value(const_value) {
    if (config.has_comp_node_set()) {
        mgb_assert(config.get_single_comp_node() == dev_data->comp_node());
    }
    add_output(dev_data->dtype());
    add_equivalence_component<ScalarHash<void*>>(dev_data.get());
}

const TensorShape& intl::SharedDeviceTensorBase::get_output_shape() {
    return m_dev_data->shape();
}

void intl::SharedDeviceTensorBase::init_output_comp_node() {
    if (config().has_comp_node_set()) {
        mgb_throw_if(config().get_single_comp_node() != m_dev_data->comp_node(),
                GraphError,
                "SharedDeviceTensor: comp node in config differs from that in"
                " dev_data");
    }
    comp_node(m_dev_data->comp_node());
}

cg::static_infer::SourceType SharedDeviceTensor::static_infer_src_type() const {
    return cg::static_infer::SourceType::CONSTANT;
}

SymbolVar SharedDeviceTensor::make(ComputingGraph &graph,
        const std::shared_ptr<DeviceTensorND> &dev_data,
        bool const_value,
        const OperatorNodeConfig &config) {
    return graph.insert_opr(std::make_unique<SharedDeviceTensor>(
                graph, dev_data, const_value, config))->output(0);
}

SymbolVar SharedDeviceTensor::make(ComputingGraph &graph,
        const HostTensorND &value,
        bool const_value,
        const OperatorNodeConfig &config) {
    auto cn = value.comp_node();
    if (config.has_comp_node_set())
        cn = config.get_single_comp_node();
    auto dev_v = std::make_shared<DeviceTensorND>();
    dev_v->comp_node(cn).copy_from(value).sync();
    return make(graph, dev_v, const_value, config);
}

MGB_DYN_TYPE_OBJ_FINAL_IMPL(SharedDeviceTensor);

cg::OperatorNodeBase::NodeProp*
VolatileSharedDeviceTensor::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    ret->add_flag(NodeProp::Flag::IMPURE_OUTPUT_MEM_PLAN);
    return ret;
}

SymbolVar VolatileSharedDeviceTensor::make(ComputingGraph &graph,
        const std::shared_ptr<DeviceTensorND> &dev_data,
        const OperatorNodeConfig &config) {
    return graph.insert_opr(std::make_unique<VolatileSharedDeviceTensor>(
                graph, dev_data, false, config))->output(0);
}

MGB_DYN_TYPE_OBJ_FINAL_IMPL(VolatileSharedDeviceTensor);

/* ============== SharedDeviceTensorWithFormat =============== */
void SharedDeviceTensorWithFormat::init_output_format() {
    output(0)->format(get_dev_tensor().format());
}

SymbolVar SharedDeviceTensorWithFormat::make(
        ComputingGraph& graph, const std::shared_ptr<DeviceTensorND>& dev_data,
        bool const_value, const OperatorNodeConfig& config) {
    auto&& opr =
            graph.insert_opr(std::make_unique<SharedDeviceTensorWithFormat>(
                                     graph, dev_data, const_value, config))
                    ->cast_final_safe<SharedDeviceTensorWithFormat>();
    return opr.output(0);
}

cg::static_infer::SourceType
SharedDeviceTensorWithFormat::static_infer_src_type() const {
    return cg::static_infer::SourceType::CONSTANT;
}

MGB_DYN_TYPE_OBJ_FINAL_IMPL(SharedDeviceTensorWithFormat);

/* ===================== ImmutableTensor ===================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(ImmutableTensor);

class ImmutableTensor::Value {
    std::mutex m_mtx;
    DeviceTensorND m_dev, m_static_infer;
    std::string m_summary;

    public:
        void setup(CompNode cn, const HostTensorND &val);

        bool initialized() const {
            return m_dev.shape_valid();
        }

        //! value on comp node
        const DeviceTensorND& dev() const {
            return m_dev;
        }

        //! get value on static infer CPU node
        DeviceTensorND& static_infer();

        //! string summary of the value
        const std::string& summary() const {
            return m_summary;
        }
};

void ImmutableTensor::Value::setup(CompNode cn, const HostTensorND &val) {
    mgb_assert(m_dev.empty() && !m_dev.shape_valid());
    m_dev.comp_node(cn).copy_from(val).sync();
    mgb_assert(val.empty() == m_dev.empty());

    auto one_elem = [](const TensorShape& shape) {
        for (size_t i = 0; i < shape.ndim; ++i) {
            if (shape[i] != 1)
                return false;
        }
        return true;
    };

    if (one_elem(val.shape())) {
        float v;
        static_cast_dtype(&v, val.dtype(), val.raw_ptr());
        m_summary = ssprintf("%.3g", v);
        if (val.shape().ndim != 1) {
            m_summary += val.shape().to_string();
        }
    } else {
        m_summary = ssprintf("const%s", val.shape().to_string().c_str());
    }
}

DeviceTensorND& ImmutableTensor::Value::static_infer() {
    MGB_LOCK_GUARD(m_mtx);
    if (m_static_infer.empty()) {
        mgb_assert(!m_dev.empty());
        m_static_infer.comp_node(CompNode::default_cpu()).copy_from(m_dev);
    }
    return m_static_infer;
}

class ImmutableTensor::DevValueCache final: public UserDataContainer::UserData {
    MGB_TYPEINFO_OBJ_DECL;
    CompNode m_comp_node;

    class TensorKey {
        struct Trait {
            size_t hash = 0, size_bytes = 0;
            TensorLayout layout;
        };
        Trait m_trait;
        std::vector<dt_byte> m_val;
        HostTensorND m_val_ref;

        const dt_byte* val_ptr() const {
            mgb_assert(m_trait.size_bytes);
            return m_val.empty() ? m_val_ref.raw_ptr() : m_val.data();
        }

        public:
            TensorKey() = default;
            TensorKey(const HostTensorND &v):
                m_val_ref{v}
            {
                mgb_assert(v.layout().is_contiguous() || v.layout().is_empty());
                m_trait.size_bytes = v.layout().span().high_byte;

                auto &&layout = m_trait.layout;
                // zero to enable byte-comparison
                memset(&layout, 0, sizeof(layout));
                layout.ndim = v.layout().ndim;
                layout.dtype = v.layout().dtype;
                for (size_t i = 0; i < layout.ndim; ++ i) {
                    layout.shape[i] = v.layout().shape[i];
                    layout.stride[i] = v.layout().stride[i];
                }
                XXHash hasher;
                if (!v.empty()) {
                    hasher.update(v.raw_ptr(), m_trait.size_bytes);
                }
                hasher.update(&m_trait.layout, sizeof(m_trait.layout));
                m_trait.hash = hasher.digest();
            }

            bool operator == (const TensorKey &rhs) const {
                return !memcmp(&m_trait, &rhs.m_trait, sizeof(Trait)) &&
                       ((m_trait.size_bytes == 0 &&
                         rhs.m_trait.size_bytes == 0) ||
                        !memcmp(val_ptr(), rhs.val_ptr(), m_trait.size_bytes));
            }

            size_t hash() const {
                return m_trait.hash;
            }

            //! copy from m_val_ref to m_val, to avoid refed value being
            //! modified
            void copy_val_permanent() {
                if (m_trait.size_bytes == 0) return;
                mgb_assert(m_val.empty());
                m_val.resize(m_trait.size_bytes);
                memcpy(m_val.data(), m_val_ref.raw_ptr(), m_trait.size_bytes);
                m_val_ref = {};
            }
    };
    struct ScalarKey {
        size_t hash = 0;
        DTypeScalar val;

        ScalarKey() = default;
        ScalarKey(const DTypeScalar &v):
            val{v}
        {
            hash = PODHash<DTypeScalar>::perform(&val, 1);
        }

        bool operator == (const ScalarKey &rhs) const {
            return val == rhs.val;
        }
    };
    struct Hash {
        size_t operator() (const TensorKey &key) const {
            return key.hash();
        }
        size_t operator() (const ScalarKey &key) const {
            return key.hash;
        }
    };

    std::unordered_map<TensorKey, Value, Hash> m_tensor2val;
    std::unordered_map<ScalarKey, Value, Hash> m_scalar2val;

    std::mutex m_mtx;

    void setup_value(Value &dest, const HostTensorND &val) {
        dest.setup(m_comp_node, val);
    }

    public:
        //! max number of elements for a tensor to be stored in this cache
        static constexpr size_t MAX_SIZE = TensorLayout::MAX_NDIM * 4;

        struct VarNodeCache;

        DevValueCache(const CompNodeEnv &env):
            m_comp_node{env.comp_node()}
        {
        }

        static DevValueCache& inst(CompNode cn) {
            auto &&env = CompNodeEnv::from_comp_node(cn);
            auto maker = [&]() {
                return std::make_shared<DevValueCache>(env);
            };
            return env.get_user_data<DevValueCache>(maker);
        }

        const Value& get(const HostTensorND &tensor) {
            if (tensor.shape().is_scalar()) {
                return get(DTypeScalar::make_from_raw(
                            tensor.dtype(), tensor.raw_ptr()));
            }

            MGB_LOCK_GUARD(m_mtx);
            TensorKey key{tensor};
            Value &item = m_tensor2val[key];
            if (!item.initialized()) {
                setup_value(item, tensor);
                const_cast<TensorKey&>(m_tensor2val.find(key)->first).
                    copy_val_permanent();
            }
            return item;
        }

        const Value& get(const DTypeScalar &scalar) {
            MGB_LOCK_GUARD(m_mtx);

            ScalarKey key{scalar};
            Value &item = m_scalar2val[key];
            if (!item.initialized()) {
                HostTensorND hv{m_comp_node, scalar.dtype()};
                hv.resize({1});
                memcpy(hv.raw_ptr(), scalar.storage(), scalar.dtype().size(1));
                setup_value(item, hv);
            }
            return item;
        }
};
MGB_TYPEINFO_OBJ_IMPL(ImmutableTensor::DevValueCache);
using ImmutableTensorDevValueCache = ImmutableTensor::DevValueCache;

struct ImmutableTensor::DevValueCache::VarNodeCache final:
        public UserDataContainer::UserData {
    ThinHashMap<const Value*, SymbolVar> val2var;

    MGB_TYPEINFO_OBJ_DECL;
};
MGB_TYPEINFO_OBJ_IMPL(ImmutableTensor::DevValueCache::VarNodeCache);

ImmutableTensor::ImmutableTensor(ComputingGraph &graph,
        const Value &value, const OperatorNodeConfig &config):
    Super{&graph, config, value.summary(), {}},
    m_value{value}
{
    mgb_assert(value.initialized());

    add_output(value.dev().dtype());
    add_equivalence_component<ScalarHash<const void*>>(&value);
    output(0)->add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
}

ImmutableTensor::~ImmutableTensor() noexcept = default;

SymbolVar ImmutableTensor::make(ComputingGraph &graph, const HostTensorND &val,
        const OperatorNodeConfig &config) {

    auto cn = val.comp_node();
    if (config.has_comp_node_set())
        cn = config.get_single_comp_node();

    if (val.shape().total_nr_elems() > DevValueCache::MAX_SIZE) {
        // tensor too large, do not dedup
        auto value = std::make_shared<Value>();
        value->setup(cn, val);
        return make_from_value(graph, *value, value, config);
    }

    auto &&cache = DevValueCache::inst(cn);
    return make_from_value(graph, cache.get(val), {}, config);
}

SymbolVar ImmutableTensor::make(ComputingGraph &graph, const DTypeScalar &val,
        const OperatorNodeConfig &config) {
    mgb_assert(config.has_comp_node_set(),
            "comp node must be set for constructing ImmutableTensor from "
            "DTypeScalar");

    auto cn = config.get_single_comp_node();
    auto &&cache = DevValueCache::inst(cn);
    return make_from_value(graph, cache.get(val), {}, config);
}

const DeviceTensorND& ImmutableTensor::value() const {
    return m_value.dev();
}
const DeviceTensorND& ImmutableTensor::host_value()  {
    return const_cast<Value*>(&m_value)->static_infer();
}

SymbolVar ImmutableTensor::make_from_value(
        ComputingGraph &graph,
        const Value &val, const std::shared_ptr<Value> &val_refkeep,
        const OperatorNodeConfig &config) {

    auto ud = graph.options().user_data.get_user_data_or_create
        <DevValueCache::VarNodeCache>(
                std::make_shared<DevValueCache::VarNodeCache>);
    SymbolVar &var = ud->val2var[&val];

    if (!var.node()) {
        var = graph.insert_opr(std::make_unique<ImmutableTensor>(
                graph, val, config))->output(0);
        if (val_refkeep) {
            auto &&opr = var.node()->owner_opr()->cast_final<ImmutableTensor>();
            mgb_assert(&opr.m_value == val_refkeep.get() &&
                    !opr.m_value_refkeep);
            opr.m_value_refkeep = val_refkeep;
        }
    }
#if !MGB_BUILD_SLIM_SERVING
    // FIXME: make() of immutable tensor would return immediately instead of
    // calling insert_opr() when hitting cache, so we need call it munually.
    // see MGE-81
    else {
        if (graph.options().eager_evaluation) {
            auto &&opr = var.node()->owner_opr();
            graph.insert_opr(std::unique_ptr<OperatorNodeBase>(opr));
        }
    }
#endif
    return var;
}

void ImmutableTensor::init_output_comp_node() {
    comp_node(m_value.dev().comp_node());
}

const TensorShape& ImmutableTensor::get_output_shape() {
    return m_value.dev().shape();
}

bool ImmutableTensor::fill_in_static_infer(DeviceTensorND *dest) {
    if (dest)
        *dest = const_cast<Value&>(m_value).static_infer();
    return true;
}

const DeviceTensorND& ImmutableTensor::get_dev_tensor() const {
    return m_value.dev();
}

cg::static_infer::SourceType ImmutableTensor::static_infer_src_type() const {
    return cg::static_infer::SourceType::CONSTANT;
}

/* ===================== Copy ===================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Copy);

Copy::Copy(VarNode *inp, const OperatorNodeConfig &config):
    Super{inp->owner_graph(), config, "copy", {inp}}
{
    add_input({inp});
    add_output(None);
}

SymbolVar Copy::make(SymbolVar inp, const OperatorNodeConfig &config) {
    return inp.insert_single_output_opr<Copy>(inp.node(), config);
}

void Copy::mem_plan_fwd_in2out_readonly() {
    if (owner_graph()->options().force_dynamic_alloc) {
        // copy on same CN in force_dynamic_alloc graphs usually used for
        // resolving dependency
        // TODO: add an option disable_auto_memfwd for Copy
        m_mem_fwd_success = false;
        return;
    }

    if (output(0)->comp_node().mem_node() == input(0)->comp_node().mem_node()) {
        m_mem_fwd_success = output(0)->set_fwd_in2out_readonly(
                input(0), SubTensorSpec::make_from_layout(input(0)->layout()));
    } else
        m_mem_fwd_success = false;
}

void Copy::init_output_comp_node() {
    Super::init_output_comp_node();
    if (output(0)->comp_node().mem_node() != input(0)->comp_node().mem_node()) {
        owner_graph()->seq_comp_node_optimizer().register_stream_var(
                output(0), {CompNode::Stream::COPY,
                            cg::SeqCompNodeOptimizer::StreamPropType::WEAK});
    }
}

void Copy::init_rt_force_dynamic_mem_alloc_imply_chain() {
    auto ivar = input(0), ovar = output(0);
    auto cn0 = ivar->comp_node(), cn1 = ovar->comp_node();
    if (cn0 != cn1 && cn0.mem_node() == cn1.mem_node()) {
        // make it possible to forward memory between comp nodes on the same mem
        // node
        ivar->add_rt_force_dynamic_mem_alloc_imply_chain(ovar);
        ovar->add_rt_force_dynamic_mem_alloc_imply_chain(ivar);
    }
}

void Copy::scn_do_execute() {
    auto &&od = output(0)->dev_tensor(),
         &&id = input(0)->dev_tensor();
    if (m_mem_fwd_success) {
        mgb_assert(od.raw_ptr() == id.raw_ptr() &&
                od.layout().eq_layout(id.layout()));
    } else {
        od.copy_from_fixlayout(id);
    }
}

Copy::NodeProp* Copy::do_make_node_prop() const {
    auto rst = Super::do_make_node_prop();
    using F = NodeProp::Flag;
    rst->add_flag(F::CROSS_COMP_NODE_MEMORY);
    rst->add_flag(F::NO_AUTOMATIC_DUP);
    return rst;
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(Copy) {
    mgb_assert(wrt_idx == 0);
    return Copy::make(out_grad[0],
            OperatorNodeConfig{}.follow_comp_node(opr.input(0))).node();
}
#endif

void Copy::add_input_layout_constraint() {
    if (input(0)->comp_node() != output(0)->comp_node()) {
        auto check = [this](const TensorLayout& layout) {
            auto handle = intl::get_megdnn_handle(this->comp_node());
            return handle->check_cross_dev_copy_constraint(layout);
        };
        input(0)->add_layout_constraint(check);
    }
}

void Copy::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    Super::init_output_static_infer_desc();
    owner_graph()->static_infer_manager().register_value_infer(
            output(0), ValueInferDesc::make_identity(input(0)));
}

/* ===================== MultipleDeviceTensorHolderBase ===================== */

class intl::MultipleDeviceTensorHolderBase::DevValuesExecDep final
        : public ExecDependency {
    SmallVector<DeviceTensorStorage> m_vals;

public:
    explicit DevValuesExecDep(const ValueArray& vals,
                              MultipleDeviceTensorHolderBase* opr) {
        mgb_assert(vals.size() == opr->output().size(),
                   "the output value size is diff from output var size");
        for (size_t index = 0; index < vals.size(); index++) {
            if (!opr->output(index)->contain_flag(
                        VarNode::Flag::MEMORY_NO_NEED)) {
                m_vals.emplace_back(std::move(vals[index]->storage()));
            }
        }
    }
};

intl::MultipleDeviceTensorHolderBase::MultipleDeviceTensorHolderBase(
        ComputingGraph& graph, ValueArray values,
        const OperatorNodeConfig& config)
        : Super(&graph, config, "multi_dv", {}), m_values{std::move(values)} {
    mgb_assert(
            !config.has_comp_node_set(),
            "comp node should not be set for MultipleDeviceTensorHolderBase");
    for (size_t i = 0; i < m_values.size(); ++i) {
        dv_helper::add_output(*this, m_values[i]->dtype(), ssprintf("o%zu", i));
        add_equivalence_component<ScalarHash<void*>>(m_values[i].get());
    }
}

void intl::MultipleDeviceTensorHolderBase::do_execute(ExecEnv& env) {
    // only dispatch to first comp node since all device values should be ready
    // due to PERSISTENT_DEVICE_VALUE
    auto work = [this]() {
        auto&& out = output();
        for (size_t i = 0; i < m_values.size(); ++i) {
            dv_helper::check_in_exec(*m_values[i], out[i]);
        }
    };
    env.dispatch_on_comp_node(output(0)->comp_node(), work);

    // Send BeforeKernel/AfterKernel event on every different comp_node
    ThinHashSet<mgb::CompNode> st = cg::get_opr_comp_node_set(this);
    for (auto cn : st) {
        auto send_event = [this, cn]() {
            this->owner_graph()
                    ->event()
                    .signal_inplace<cg::event::BeforeKernel>(this, cn);
            this->owner_graph()->event().signal_inplace<cg::event::AfterKernel>(
                    this, cn);
        };
        env.dispatch_on_comp_node(cn, send_event);
    }
}

void intl::MultipleDeviceTensorHolderBase::init_output_mem_plan(bool dynamic) {
    for (size_t i = 0; i < m_values.size(); ++i) {
        dv_helper::init_output_mem_plan(*m_values[i], *this, dynamic, i);
    }
}

void intl::MultipleDeviceTensorHolderBase::on_output_comp_node_stream_changed() {
    mgb_throw(SystemError, "comp node of device tensor should not change");
}

void intl::MultipleDeviceTensorHolderBase::init_output_comp_node() {
    for (size_t i = 0; i < m_values.size(); ++i) {
        output(i)->comp_node(m_values[i]->comp_node());
    }
}

void intl::MultipleDeviceTensorHolderBase::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto&& mgr = owner_graph()->static_infer_manager();
    for (size_t i = 0; i < m_values.size(); ++i) {
        auto infer_shp = [p = m_values[i].get()](TensorShape & dest,
                                                 const InpVal&)
                                 ->bool {
            dest = p->shape();
            return dest.ndim;
        };
        mgr.register_shape_infer(output(i),
                                 {SourceType::CONSTANT, {}, infer_shp});
    }
}

intl::MultipleDeviceTensorHolderBase::NodeProp*
intl::MultipleDeviceTensorHolderBase::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    ret->add_flag(NodeProp::Flag::CROSS_COMP_NODE_MEMORY);
    return ret;
}

void intl::MultipleDeviceTensorHolderBase::record_execute_deps(
        ExecDependencyArray& deps) {
    deps.emplace_back(std::make_unique<DevValuesExecDep>(values(), this));
}

/* ===================== MultipleDeviceTensorHolder ===================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(MultipleDeviceTensorHolder);

SymbolVarArray MultipleDeviceTensorHolder::make(
        ComputingGraph& graph, ValueArray values,
        const OperatorNodeConfig& config) {
    return cg::to_symbol_var_array(
            graph.insert_opr(
                         std::make_unique<MultipleDeviceTensorHolder>(
                                 graph, std::move(values), config))
                    ->output());
}

/* ================== MultipleDeviceTensorWithFormatHolder ================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(MultipleDeviceTensorWithFormatHolder);

SymbolVarArray MultipleDeviceTensorWithFormatHolder::make(
        ComputingGraph& graph, ValueArray values,
        const OperatorNodeConfig& config) {
    return cg::to_symbol_var_array(
            graph.insert_opr(
                         std::make_unique<MultipleDeviceTensorWithFormatHolder>(
                                 graph, std::move(values), config))
                    ->output());
}

void MultipleDeviceTensorWithFormatHolder::init_output_format() {
    for (size_t i = 0; i < m_values.size(); ++i) {
        output(i)->format(m_values[i]->format());
    }
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
