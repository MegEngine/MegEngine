/**
 * \file src/jit/impl/halide/halide_executable.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./halide_executable.h"

#if MGB_JIT_HALIDE

#include "megbrain/jit/utils.h"

using namespace mgb;
using namespace jit;
using namespace Halide;

HalideExecutable::FunctionHandle::~FunctionHandle() {
    if (device_release && uctx_map) {
        for (auto&& i : uctx_map->cn2uctx) {
            device_release(i.second);
        }
    }
    delete uctx_map;
    if (dl_handle) {
        ExecutableHelper::get().unload_lib(dl_handle);
    }
}

HalideExecutable::TargetTraitUserData* HalideExecutable::TargetTrait::user_data(
        const HalideExecutable& hl_exec,
        thin_function<std::unique_ptr<TargetTraitUserData>()> maker) {
    MGB_LOCK_GUARD(hl_exec.m_target_trait_user_data_mtx);
    if (!hl_exec.m_target_trait_user_data) {
        hl_exec.m_target_trait_user_data = maker();
    }
    return hl_exec.m_target_trait_user_data.get();
}

/* =================== HalideExecutable ==================== */

HalideExecutable::~HalideExecutable() = default;

HalideExecutable::HalideExecutable(std::shared_ptr<TargetTrait> target_trait,
                                   const InternalGraph& graph,
                                   const JITExecutor::Args& args)
        : m_target_trait{std::move(target_trait)} {
    ThinHashMap<VarNode*, const JITExecutor::Args::Data*> placeholders_to_inps;
    for (auto&& inp : args.inputs) {
        VarNode* placeholder = graph.placeholders().at(inp.idx)->output(0);
        placeholders_to_inps[placeholder] = &inp;
    }

    using AstNodePtr = ast_hl::AstNodePtr;
    ThinHashMap<VarNode*, AstNodePtr> mgb2halide;

    auto on_opr = [&](cg::OperatorNodeBase* opr) {
        auto var = opr->output(0);
        AstNodePtr ptr;
        if (opr->same_type<JITPlaceholder>()) {
            auto data = placeholders_to_inps.at(var);
            auto&& ph = opr->cast_final_safe<JITPlaceholder>();
            if (ph.is_host_value_shape_input()) {
                ptr = std::make_shared<ast_hl::InputHostValueShapeOp>();
                ptr->m_layout = data->layout;
            } else {
                ptr = mgb_var_to_halide_buffer(data->from);
                m_value_inputs.emplace_back(static_cast<size_t>(data->idx),
                                            ptr);
            }
        } else {
            ptr = ast_hl::make_from_opr(opr);
            for (auto inp : opr->input()) {
                ptr->m_inputs.push_back(mgb2halide.at(inp));
            }
            ptr->init(opr);
        }
        mgb2halide[var] = std::move(ptr);
    };

    cg::DepOprIter{on_opr}.add(graph.output());

    std::sort(m_value_inputs.begin(), m_value_inputs.end());
    m_halide_output = mgb2halide.at(graph.output());
}

void HalideExecutable::execute(JITExecutor* fusion_opr) {
    // load func_ptr for current comp node
    auto comp_node = fusion_opr->comp_node();
    std::atomic<FunctionHandle*>* func_ptr_ref;
    {
        MGB_LOCK_GUARD(m_mtx);
        func_ptr_ref = &m_cn2func[comp_node];
    }
    auto func_ptr = func_ptr_ref->load();
    if (!func_ptr) {
        std::pair<std::mutex, FunctionHandle>* func_maker;
        {
            MGB_LOCK_GUARD(m_mtx);
            func_maker =
                    &m_feature_set2func[m_target_trait->features(comp_node)];
        }

        // compile the function
        MGB_LOCK_GUARD(func_maker->first);
        if (!(func_ptr = func_ptr_ref->load())) {
            if (!func_maker->second.execute) {
                func_maker->second = compile_and_load(comp_node);
                mgb_assert(func_maker->second.execute);
            }
            func_ptr = &func_maker->second;
            func_ptr_ref->store(func_ptr);
        }
    }

    void* user_context = nullptr;
    if (func_ptr->uctx_map) {
        MGB_LOCK_GUARD(func_ptr->uctx_map->mtx);
        auto&& ptr = func_ptr->uctx_map->cn2uctx[comp_node];
        if (!ptr) {
            ptr = m_target_trait->get_user_context(comp_node);
        }
        user_context = ptr;
    }

    invoke(user_context, *func_ptr, fusion_opr->input(), fusion_opr->output(0));
}

std::vector<Halide::Argument> HalideExecutable::halide_inputs() const {
    std::vector<Argument> args;
    for (auto&& i : m_value_inputs) {
        auto&& input_buffer =
                i.second->cast_final_safe<ast_hl::InputDevValueOp>();
        args.emplace_back(input_buffer.m_buffer);
    }
    return args;
}

HalideExecutable::FunctionHandle HalideExecutable::compile_and_load(
        CompNode comp_node) const {
    Target target = get_host_target();
    auto req_features = m_target_trait->features(comp_node);
    target.set_feature(Target::UserContext);
    if (MGB_GETENV("MGB_HALIDE_DEBUG")) {
        target.set_feature(Target::Debug);
    }
    for (size_t i = 0; i < req_features.size(); ++i) {
        if (req_features.test(i)) {
            target.set_feature(static_cast<Target::Feature>(i));
        }
    }

    return m_target_trait->compile_and_load(comp_node, target, *this);
}

void HalideExecutable::invoke(void* user_context, const FunctionHandle& handle,
                              const VarNodeArray& inputs, VarNode* output) {
    mgb_assert(handle.execute && handle.get_device_interface);
    halide_device_interface_t* device_interface = handle.get_device_interface();

    size_t nr_inputs = m_value_inputs.size(), argv_idx = 0;
    void* argv[nr_inputs + 2];

    halide_buffer_t image_args[nr_inputs + 1];
    size_t nr_dims = (nr_inputs + 1) * TensorLayout::MAX_NDIM;
    halide_dimension_t image_dims_buf[nr_dims];
    memset(image_dims_buf, 0, sizeof(halide_dimension_t) * nr_dims);
    size_t image_arg_idx = 0;
    halide_dimension_t* image_dims_ptr = image_dims_buf;

    auto add_tensor_arg = [&](const DeviceTensorND& tensor) {
        int ndim = tensor.layout().ndim;
        for (int i = ndim - 1; i >= 0; i--) {
            image_dims_ptr->extent = tensor.layout()[i];
            image_dims_ptr->stride = tensor.layout().stride[i];
            image_dims_ptr++;
        }
        auto dtype = tensor.dtype();
        halide_type_t type = dtype_mgb2halide(dtype);
        image_args[image_arg_idx] = {
                reinterpret_cast<uint64_t>(tensor.raw_ptr()),
                device_interface,
                nullptr,
                0,
                type,
                ndim,
                image_dims_ptr - ndim,
                nullptr};
        argv[argv_idx++] = &image_args[image_arg_idx++];
    };

    argv[argv_idx++] = &user_context;
    for (auto&& i : m_value_inputs) {
        add_tensor_arg(inputs.at(i.first)->dev_tensor());
    }
    add_tensor_arg(output->dev_tensor());
    mgb_assert(argv_idx == nr_inputs + 2);
    mgb_assert(image_dims_ptr <= image_dims_buf + nr_dims);
    auto err = handle.execute(argv);
    mgb_throw_if(err, SystemError, "failed to execute halide function: err=%d",
                 err);
}

halide_type_t HalideExecutable::dtype_mgb2halide(DType dtype) {
    if (dtype == dtype::Float32()) {
        return halide_type_of<float>();
    } else if (dtype == dtype::Float16()) {
        return halide_type_of<float16_t>();
    } else if (dtype == dtype::Int32()) {
        return halide_type_of<int>();
    } else {
        mgb_throw(InternalError,
                  "dtype(%s) is not any of [Float16, Float32, Int32]",
                  dtype.name());
    }
}

ast_hl::AstNodePtr HalideExecutable::mgb_var_to_halide_buffer(VarNode* var) {
    auto res = std::make_shared<ast_hl::InputDevValueOp>();
    res->m_layout = var->layout();
    int ndim = var->layout().ndim;
    halide_dimension_t halide_dim[ndim];
    memset(halide_dim, 0, sizeof(halide_dimension_t) * ndim);
    for (int i = ndim - 1; i >= 0; i--) {
        halide_dim[ndim - 1 - i].extent = res->m_layout[i];
        halide_dim[ndim - 1 - i].stride = res->m_layout.stride[i];
    }

    halide_buffer_t buf{
            0,    nullptr,    nullptr, 0, dtype_mgb2halide(var->dtype()),
            ndim, halide_dim, nullptr};

    res->m_buffer = Buffer<>{buf};
    res->init(nullptr);
    return res;
}

#endif  // MGB_JIT_HALIDE

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
