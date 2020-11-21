/**
 * \file src/serialization/impl/extern_c_opr.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/serialization/extern_c_opr.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/graph/extern_copr_api.h"
#include "megbrain/serialization/extern_c_opr_io.h"
#include "megbrain/serialization/opr_load_dump.h"

#include <cstdlib>

using namespace mgb;
using namespace serialization;
using namespace opr;

namespace {

const char PLACEHOLDER_TYPE_NAME[] = "placeholder";

typedef MGBOprDesc* (*opr_desc_transformer_t)(void* input);

using LoaderMap =
        std::unordered_map<std::string,
                           std::pair<MGBOprLoader, opr_desc_transformer_t>>;

//! singleton LoaderMap
LoaderMap& loader_map() {
    static LoaderMap ret;
    return ret;
}

class MGBOprDescHash final : public HashableVD {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

    MGBOprDesc* const m_desc;

    bool is_same_st(const Hashable& rhs) const override {
        return m_desc->is_same(m_desc,
                               static_cast<const MGBOprDescHash&>(rhs).m_desc);
    }

public:
    MGBOprDescHash(MGBOprDesc* desc) : m_desc{desc} {}

    size_t hash() const override { return m_desc->hash(m_desc); }
};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(MGBOprDescHash);

MGBDType dtype_cpp2c(DType dtype) {
    switch (dtype.enumv()) {
        case DTypeEnum::Float32:
            return MGB_DTYPE_FLOAT32;
        case DTypeEnum::Int32:
            return MGB_DTYPE_INT32;
        case DTypeEnum::Int16:
            return MGB_DTYPE_INT16;
        case DTypeEnum::Uint8:
            return MGB_DTYPE_UINT8;
#if !MEGDNN_DISABLE_FLOAT16
        case DTypeEnum::Float16:
            return MGB_DTYPE_FLOAT16;
#endif
        default:
            mgb_throw(InternalError, "unsupported dtype for extern C API: %s",
                      dtype.name());
    }
}

DType dtype_c2cpp(MGBDType dtype) {
    switch (dtype) {
        case MGB_DTYPE_UINT8:
            return dtype::Uint8{};
        case MGB_DTYPE_INT16:
            return dtype::Int16{};
        case MGB_DTYPE_INT32:
            return dtype::Int32{};
        case MGB_DTYPE_FLOAT32:
            return dtype::Float32{};
#if !MEGDNN_DISABLE_FLOAT16
        case MGB_DTYPE_FLOAT16:
            return dtype::Float16{};
#endif
        default:
            mgb_throw(SerializationError, "bad dtype value: %d",
                      static_cast<int>(dtype));
    }
}

template <typename S>
MGBTensor tensor_to_c(const TensorND<S>& src) {
    MGBTensor ret;
    ret.data = const_cast<void*>(static_cast<const void*>(src.raw_ptr()));
    ret.layout.dtype = dtype_cpp2c(src.dtype());
    ret.layout.shape = ExternCOprRunner::tensor_shape_to_c(src.shape());
    return ret;
}

struct MGBOprDescV23 {
    size_t nr_input, nr_output;

    //! operator type name
    const char* type_name;

    //! release this descriptor
    void (*release)(MGBOprDescV23* self);

    //! compute hash
    size_t (*hash)(const MGBOprDescV23* self);

    //! equality check
    int (*is_same)(const MGBOprDescV23* self, const MGBOprDescV23* rhs);

    //! perform the computation
    void (*execute)(const MGBOprDescV23* self, const MGBTensor* input,
                    const MGBTensor* output);

    //! infer output shapes from input shapes
    void (*infer_shape)(const MGBOprDescV23* self, const MGBTensorShape* input,
                        MGBTensorShape* output);

    //! custom user data to be associated with this descriptor
    void* user_data;

    static MGBOprDesc* as_opr_desc(void* v23_raw) {
        auto release = [](MGBOprDesc* self) {
            auto p = static_cast<MGBOprDescV23*>(self->user_data);
            p->release(p);
            delete self;
        };
        auto hash = [](const MGBOprDesc* self) {
            auto p = static_cast<MGBOprDescV23*>(self->user_data);
            return p->hash(p);
        };
        auto is_same = [](const MGBOprDesc* self, const MGBOprDesc* rhs) {
            auto p0 = static_cast<MGBOprDescV23*>(self->user_data);
            auto p1 = static_cast<MGBOprDescV23*>(rhs->user_data);
            return p0->is_same(p0, p1);
        };

        auto execute = [](const MGBOprDesc* self, const MGBTensor* input,
                          const MGBTensor* output) {
            auto p = static_cast<MGBOprDescV23*>(self->user_data);
            p->execute(p, input, output);
        };

        auto infer_shape = [](const MGBOprDesc* self,
                              const MGBTensorShape* input,
                              MGBTensorShape* output) {
            auto p = static_cast<MGBOprDescV23*>(self->user_data);
            p->infer_shape(p, input, output);
        };

        auto v23 = static_cast<MGBOprDescV23*>(v23_raw);
        auto ret = std::make_unique<MGBOprDesc>();
        mgb_init_opr_desc(ret.get(), v23->nr_output, v23->type_name);
        ret->user_data = v23;
#define ASSIGN(name) ret->name = name;
        MGB_OPR_DESC_FOREACH_MEM_FN(ASSIGN);
#undef ASSIGN
        return ret.release();
    }
};

//! impl MGBOprDesc for ExternCOprRunner::make_placeholder
class PlaceholderMGBOprDesc {
    struct UserData {
        std::string name;
        TensorShapeArray output_shapes;
        SmallVector<DType> output_dtypes;
        std::unique_ptr<uint8_t[]> data;
        size_t data_len;
    };

    static UserData* user_data(const MGBOprDesc* self) {
        return static_cast<UserData*>(self->user_data);
    }

    static void release(MGBOprDesc* self) {
        user_data(self)->~UserData();
        ::free(self);
    }

    static size_t hash(const MGBOprDesc* self) {
        return reinterpret_cast<size_t>(self);  // hash disabled
    }

    static int is_same(const MGBOprDesc* self, const MGBOprDesc* rhs) {
        return self == rhs;
    }

    //! perform the computation
    static void execute(const MGBOprDesc*, const MGBTensor*, const MGBTensor*) {
        mgb_throw(MegBrainError,
                  "placeholder ExternCOprRunner can not be executed");
    }

    static void infer_shape(const MGBOprDesc* self, const MGBTensorShape* input,
                            MGBTensorShape* output);

    static void infer_dtype(const struct MGBOprDesc* self,
                            const MGBDType* input, MGBDType* output);

public:
    static MGBOprDesc* make(size_t nr_input, const char* name,
                            const TensorShapeArray& output_shapes,
                            const SmallVector<DType>& output_dtypes,
                            const void* data, size_t data_len);

    static void dump(OprDumpContext& ctx, MGBOprDesc* desc);
};

}  // anonymous namespace

/* ===================== PlaceholderMGBOprDesc ===================== */
void PlaceholderMGBOprDesc::infer_shape(const MGBOprDesc* self,
                                        const MGBTensorShape* input,
                                        MGBTensorShape* output) {
    auto ud = user_data(self);
    for (size_t i = 0; i < ud->output_shapes.size(); ++i) {
        output[i] = ExternCOprRunner::tensor_shape_to_c(ud->output_shapes[i]);
    }
}

void PlaceholderMGBOprDesc::infer_dtype(const struct MGBOprDesc* self,
                                        const MGBDType* input,
                                        MGBDType* output) {
    auto ud = user_data(self);
    for (size_t i = 0; i < ud->output_dtypes.size(); ++i) {
        output[i] = dtype_cpp2c(ud->output_dtypes[i]);
    }
}

MGBOprDesc* PlaceholderMGBOprDesc::make(size_t nr_input, const char* name,
                                        const TensorShapeArray& output_shapes,
                                        const SmallVector<DType>& output_dtypes,
                                        const void* data, size_t data_len) {
    constexpr size_t align = std::max(alignof(MGBOprDesc), alignof(UserData)),
                     desc_size = ((sizeof(MGBOprDesc) - 1) / align + 1) * align;
    std::unique_ptr<uint8_t, void (*)(void*)> ptr(
            static_cast<uint8_t*>(malloc(desc_size + sizeof(UserData))),
            ::free);
    mgb_assert(ptr);
    auto del_ud = [](UserData* p) { p->~UserData(); };
    std::unique_ptr<UserData, decltype(del_ud)> ud(
            new (ptr.get() + desc_size) UserData, del_ud);
    ud->name = name;
    ud->output_shapes = output_shapes;
    ud->output_dtypes = output_dtypes;
    ud->data.reset(new uint8_t[data_len]);
    ud->data_len = data_len;
    memcpy(ud->data.get(), data, data_len);

    auto desc = new (ptr.get()) MGBOprDesc;
    mgb_init_opr_desc(desc, output_shapes.size(), PLACEHOLDER_TYPE_NAME);
    desc->user_data = ud.release();
#define s(n) desc->n = &PlaceholderMGBOprDesc::n;
    MGB_OPR_DESC_FOREACH_MEM_FN(s);
    if (!output_dtypes.empty()) {
        desc->infer_dtype = &PlaceholderMGBOprDesc::infer_dtype;
    }
#undef s
    return reinterpret_cast<MGBOprDesc*>(ptr.release());
}

void PlaceholderMGBOprDesc::dump(OprDumpContext& ctx, MGBOprDesc* desc) {
    mgb_assert(desc->type_name == PLACEHOLDER_TYPE_NAME,
               "only placeholder ExternCOprRunner can be dumped; got type %s",
               desc->type_name);
    auto ud = user_data(desc);
    ctx.dump_buf_with_len(ud->name.c_str(), ud->name.size());
    ctx.dump_buf_with_len(ud->data.get(), ud->data_len);
}

/* ===================== ExternCOprRunner ===================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(ExternCOprRunner);
ExternCOprRunner::ExternCOprRunner(std::string& name,
                                   const VarNodeArray& inputs,
                                   std::shared_ptr<MGBOprDesc> desc,
                                   const OperatorNodeConfig& config)
        : Super{inputs[0]->owner_graph(), config, desc->type_name, inputs},
          m_desc{std::move(desc)},
          m_dump_name{name},
          m_param{nullptr} {
    mgb_assert(m_desc->size == sizeof(MGBOprDesc),
               "invalid MGBOprDesc size: expect=%zu got=%u", sizeof(MGBOprDesc),
               m_desc->size);
    for (auto i : inputs) {
        add_input({i});
    }
    auto nr_out = m_desc->nr_output;
    if (nr_out > 1) {
        for (size_t i = 0, it = nr_out; i < it; ++i)
            add_output(ssprintf("o%zu", i));
    } else {
        mgb_assert(nr_out == 1,
                   "could not create an operator with %u outputs: %s", nr_out,
                   cname());
        add_output(None);
    }
    add_equivalence_component<MGBOprDescHash>(m_desc.get());
}

void ExternCOprRunner::get_output_var_shape(const TensorShapeArray& inp_shape,
                                            TensorShapeArray& out_shape) const {
    SmallVector<MGBTensorShape> c_inp(inp_shape.size()),
            c_out(out_shape.size());
    for (size_t i = 0; i < inp_shape.size(); ++i) {
        c_inp[i] = tensor_shape_to_c(inp_shape[i]);
    }
    m_desc->infer_shape(m_desc.get(), c_inp.data(), c_out.data());
    for (size_t i = 0; i < out_shape.size(); ++i) {
        out_shape[i] = tensor_shape_from_c(c_out[i]);
    }
}

void ExternCOprRunner::init_output_dtype() {
    if (!m_desc->infer_dtype) {
        Super::init_output_dtype();
        return;
    }
    SmallVector<MGBDType> inp_dtypes, out_dtypes(output().size());
    inp_dtypes.reserve(input().size());
    for (auto i : input()) {
        inp_dtypes.push_back(dtype_cpp2c(i->dtype()));
    }
    m_desc->infer_dtype(m_desc.get(), inp_dtypes.data(), out_dtypes.data());
    for (size_t i = 0; i < out_dtypes.size(); ++i) {
        output(i)->dtype(dtype_c2cpp(out_dtypes[i]));
    }
}
void ExternCOprRunner::check_param() {
    //! check extern dynamic param validity
    //! nr_input=0 or nr_output=0 means do not provide input/output
    //! ExternDeviceTensor for some case, ExternCOprParam may only config
    //! device_id, extra_info, etc. so we need consider nr_input=0 or
    //! nr_output=0
    auto check = [](size_t nr_config_tensor, size_t var_node_size,
                    ExternDeviceTensor* e_tensor,
                    const VarNodeArray& var_node_array, const char* msg) {
        mgb_assert(e_tensor, "%s ExternDeviceTensor should not be null!!", msg);
        mgb_assert(
                nr_config_tensor == var_node_size,
                "param %s size provided by `config_extern_c_opr_dynamic_param` "
                "mismatch with the number of %s, got %zu, expected %zu",
                msg, msg, nr_config_tensor, var_node_size);
        for (size_t i = 0; i < nr_config_tensor; i++) {
            mgb_assert(e_tensor[i].device_ptr,
                       "%s ExternDeviceTensor(index: %zu) device_ptr should "
                       "not be null!!",
                       msg, i);
            auto param_shape = e_tensor[i].layout.shape;
            auto shape = var_node_array.at(i)->shape();
            auto param_dtype = e_tensor[i].layout.dtype;
            auto dtype = dtype_cpp2c(var_node_array.at(i)->dtype());
            mgb_assert(param_dtype == dtype,
                       "%s dtype provided mismatch, expected: %u, got: %d", msg,
                       param_dtype, dtype);
            mgb_assert(shape.ndim == param_shape.ndim,
                       "%s ndim provided mismatch got: %u, expect: %zu of "
                       "index: %zu",
                       msg, param_shape.ndim, shape.ndim, i);
            for (size_t j = 0; j < shape.ndim; j++) {
                mgb_assert(param_shape.shape[j] == shape.shape[j],
                           "config %s shape should same with c opr %s shape: "
                           "(got: %u expect: %zu) of index: %zu",
                           msg, msg, param_shape.shape[j], shape.shape[j], j);
            }
        }
    };

    if (m_param && m_param->nr_input > 0) {
        check(m_param->nr_input, input().size(), m_param->input, input(),
              "input");
    }

    if (m_param && m_param->nr_output > 0) {
        check(m_param->nr_output, output().size(), m_param->output, output(),
              "output");
    }
}

void ExternCOprRunner::scn_do_execute() {
    SmallVector<MGBTensor> c_inp(input().size()), c_out(output().size());
    SmallVector<HostTensorND> cpu_inp, cpu_out;
    check_param();

    bool need_copy = false;
    if (comp_node().device_type() == CompNode::DeviceType::CPU) {
        for (size_t i = 0; i < input().size(); ++i) {
            c_inp[i] = tensor_to_c(input(i)->dev_tensor());
        }
        for (size_t i = 0; i < output().size(); ++i) {
            c_out[i] = tensor_to_c(output(i)->dev_tensor());
        }
    } else {
        need_copy = true;
        mgb_log_debug(
                "copy is needed to execute extern C "
                "opr `%s' on comp node `%s'",
                cname(), comp_node().to_string().c_str());
        cpu_inp.resize(input().size());
        cpu_out.resize(output().size());
        for (size_t i = 0; i < input().size(); ++i) {
            cpu_inp[i].copy_from(input(i)->dev_tensor());
            c_inp[i] = tensor_to_c(cpu_inp[i]);
        }
        for (size_t i = 0; i < output().size(); ++i) {
            cpu_out[i]
                    .comp_node(comp_node())
                    .dtype(output(i)->dtype())
                    .resize(output(i)->shape());
            c_out[i] = tensor_to_c(cpu_out[i]);
        }
    }

    if (need_copy) {
        comp_node().sync();
        m_desc->execute(m_desc.get(), c_inp.data(), c_out.data());

        for (size_t i = 0; i < output().size(); ++i)
            output(i)->dev_tensor().copy_from_fixlayout(cpu_out[i]);
    } else {
        CompNodeEnv::from_comp_node(comp_node())
                .cpu_env()
                .dispatch([this, c_inp, c_out]() mutable {
                    m_desc->execute(m_desc.get(), c_inp.data(), c_out.data());
                });
    }
}

void ExternCOprRunner::add_input_layout_constraint() {
    for (auto i : input())
        i->add_layout_constraint_contiguous();
}

cg::OperatorNodeBase* ExternCOprRunner::make_placeholder(
        const SymbolVarArray& inputs, const TensorShapeArray& output_shapes,
        const char* name, const void* data, size_t data_len,
        const OperatorNodeConfig& config,
        const SmallVector<DType>& output_dtypes) {
    auto desc = PlaceholderMGBOprDesc::make(inputs.size(), name, output_shapes,
                                            output_dtypes, data, data_len);

    VarNodeArray var_inp(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        var_inp[i] = inputs[i].node();
    }

    auto dump_name = std::string{name};
    return make_from_desc(dump_name, var_inp, desc, config);
}

cg::OperatorNodeBase* ExternCOprRunner::make_from_desc(
        std::string& name, const VarNodeArray& inputs, MGBOprDesc* desc,
        const OperatorNodeConfig& config) {
    auto desc_del = [](MGBOprDesc* ptr) { ptr->release(ptr); };
    return make_from_desc_shared(name, inputs, {desc, desc_del}, config);
}

cg::OperatorNodeBase* ExternCOprRunner::make_from_desc_shared(
        std::string& name, const VarNodeArray& inputs,
        std::shared_ptr<MGBOprDesc> desc, const OperatorNodeConfig& config) {
    mgb_assert(!inputs.empty() && desc->nr_output);

#define CHECK(name) mgb_assert(desc->name, #name " is not given");
    MGB_OPR_DESC_FOREACH_MEM_FN(CHECK);
#undef CHECK

    if (!config.name().valid())
        const_cast<OperatorNodeConfig&>(config).name(name);

    auto opr = inputs[0]->owner_graph()->insert_opr(
            std::make_unique<ExternCOprRunner>(name, inputs, std::move(desc),
                                               config));
    return &opr->cast_final_safe<ExternCOprRunner>();
}

bool ExternCOprRunner::unregister_loader(const char* name) {
    return loader_map().erase(name);
}

void ExternCOprRunner::dump(OprDumpContext& ctx,
                            const cg::OperatorNodeBase& opr_) {
    auto&& opr = opr_.cast_final<ExternCOprRunner>();
    PlaceholderMGBOprDesc::dump(ctx, opr.m_desc.get());
}

cg::OperatorNodeBase* ExternCOprRunner::load(OprLoadContext& ctx,
                                             const cg::VarNodeArray& inputs,
                                             const OperatorNodeConfig& config) {
    auto dump_name = ctx.load_buf_with_len();
    auto name = dump_name;
    //! use to compat dump ExternCOprRunner with more info
    if (auto index = name.find(":"))
        name = name.substr(0, index);
    auto&& map = loader_map();
    auto iter = map.find(name);
    mgb_assert(iter != map.end(),
               "can not find loader for ExternCOprRunner `%s'", name.c_str());
    auto data = ctx.load_shared_buf_with_len();
    auto desc = iter->second.first.create_desc(inputs.size(), data.data(),
                                               data.size());
    if (auto trans = iter->second.second) {
        desc = trans(desc);
    }
    return make_from_desc(dump_name, inputs, desc, config);
}

cg::OperatorNodeBase* ExternCOprRunner::shallow_copy(
        const serialization::OprShallowCopyContext& ctx,
        const cg::OperatorNodeBase& opr_, const VarNodeArray& inputs,
        const OperatorNodeConfig& config) {
    auto&& opr = opr_.cast_final_safe<ExternCOprRunner>();
    auto dump_name = opr.m_dump_name;
    return make_from_desc_shared(dump_name, inputs, opr.m_desc, config);
}

MGBTensorShape ExternCOprRunner::tensor_shape_to_c(const TensorShape& shape) {
    mgb_assert(shape.ndim <= MGB_TENSOR_MAX_NDIM, "shape ndim too large: %zu",
               shape.ndim);
    MGBTensorShape ret;
    ret.ndim = shape.ndim;
    for (size_t i = 0; i < shape.ndim; ++i) {
        ret.shape[i] = shape[i];
    }
    return ret;
}

TensorShape ExternCOprRunner::tensor_shape_from_c(const MGBTensorShape& shape) {
    mgb_assert(shape.ndim <= TensorShape::MAX_NDIM, "shape ndim too large: %u",
               shape.ndim);
    TensorShape ret;
    ret.ndim = shape.ndim;
    for (size_t i = 0; i < shape.ndim; ++i) {
        ret.shape[i] = shape.shape[i];
    }
    return ret;
}

void mgb::config_extern_c_opr_dynamic_param(
        std::unique_ptr<cg::AsyncExecutable>& func,
        std::shared_ptr<ExternCOprParam> param) {
    mgb_throw_if(!param, MegBrainError, "invalid ExternCOprParam param!!");

    auto find_config_opr = false;

    auto cb = [&](cg::OperatorNodeBase* opr) {
        if (auto c_opr = opr->try_cast_final<opr::ExternCOprRunner>()) {
            auto dump_name = c_opr->get_dump_name().c_str();
            if (!param->extern_c_opr_dump_name ||
                !strncmp(param->extern_c_opr_dump_name, dump_name,
                         strlen(dump_name))) {
                c_opr->set_param(param);
                find_config_opr = true;
                mgb_log_debug("config dynamic param for extern c opr: %s",
                              dump_name);
            }
        }

        return !find_config_opr;
    };

    func->iter_opr_seq(cb);

    mgb_throw_if(!find_config_opr, MegBrainError,
                 "graph do not include a ExternCOprRunner opr or error config "
                 "extern_c_opr_dump_name!!");
}

/* ===================== public APIs ===================== */
const MGBExternCOprApi* mgb_get_extern_c_opr_api_versioned(int version) {
    auto unreg = [](const char* name) -> int {
        return ExternCOprRunner::unregister_loader(name);
    };
    if (version == 0x23) {
        auto reg23 = [](const MGBOprLoader* loader) -> int {
            return loader_map()
                    .insert({loader->name,
                             {*loader, MGBOprDescV23::as_opr_desc}})
                    .second;
        };
        static const MGBExternCOprApi ret = {reg23, unreg};
        return &ret;
    }
    if (version != MGB_EXTERN_C_OPR_VERSION)
        return nullptr;

    auto reg = [](const MGBOprLoader* loader) -> int {
        return loader_map().insert({loader->name, {*loader, nullptr}}).second;
    };
    static const MGBExternCOprApi ret = {reg, unreg};
    return &ret;
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
