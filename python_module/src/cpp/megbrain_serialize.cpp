/**
 * \file python_module/src/cpp/megbrain_serialize.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#include "./megbrain_serialize.h"
#include "./python_helper.h"

#include "megbrain/opr/basic_arith.h"

using namespace mgb;
using namespace serialization;

TensorValueDumperContext::~TensorValueDumperContext() noexcept = default;
TensorValueLoaderContext::~TensorValueLoaderContext() noexcept = default;

PyObject* TensorValueDumperContext::_value() {
    return npy::ndarray_from_tensor(m_value, npy::ShareType::TRY_SHARE);
}

void TensorValueDumperContext::_write(PyObject *bytes) {
    mgb_assert(PyBytes_Check(bytes));
    auto arr_len = PyBytes_Size(bytes);
    auto arr_buf = PyBytes_AsString(bytes);
    m_fout.write(arr_buf, arr_len);
}

std::vector<size_t> TensorValueLoaderContext::_get_shape() const {
    mgb_assert(m_layout.is_contiguous());
    return npy::shape2vec(m_layout);
}

PyObject* TensorValueLoaderContext::_get_dtype() const {
    return npy::dtype_mgb2np(m_layout.dtype);
}

PyObject* TensorValueLoaderContext::_read(size_t n) {
    // Creates a PyBytes with uninitialized content
    PyObject* bytes = PyBytes_FromStringAndSize(nullptr, n);
    m_fin.read(PyBytes_AsString(bytes), n);
    return bytes;
}

std::string _get_info_for_strip(const SymbolVarArray &dest_vars) {
    std::unordered_set<const char*> opr_types, dtype_names, elemwise_modes;

    auto on_opr = [&](cg::OperatorNodeBase *opr) {
        if (GraphDumper::should_remove_in_dump(opr))
            return;
        opr_types.insert(opr->dyn_typeinfo()->name);
        for (auto i: opr->output())
            dtype_names.insert(i->dtype().name());
        if (opr->same_type<opr::Elemwise>()) {
            auto mode = opr->cast_final<opr::Elemwise>().param().mode;
            elemwise_modes.insert(
                    megdnn::Elemwise::ModeTrait::from_mode(mode).name);
        }
    };
    cg::DepOprIter opr_iter{on_opr};
    for (auto i: dest_vars)
        opr_iter.add(i.node()->owner_opr());

    auto to_json = [](const std::unordered_set<const char*> &v) {
        std::vector<std::string> vs(v.begin(), v.end());
        std::sort(vs.begin(), vs.end());
        auto ret = json::Array::make();
        for (auto &&i: vs)
            ret->add(json::String::make(i));
        return ret;
    };

    return json::Object::make({
            {"opr_types", to_json(opr_types)},
            {"dtypes", to_json(dtype_names)},
            {"elemwise_modes", to_json(elemwise_modes)},
            })->to_string();
}

void _serialize_comp_graph_to_file(
        const char *fpath, bool append, GraphDumpFormat format,
        const SymbolVarArray &output_vars,
        int keep_var_name, bool keep_param_name, bool keep_opr_priority,
        _TensorValueDumperCallback *tensor_value_dumper,
        std::vector<size_t> &stat,
        std::vector<std::string> &inputs,
        std::vector<std::string> &outputs,
        std::vector<std::string> &params) {

    auto dumper = GraphDumper::make(
            OutputFile::make_fs(fpath, append ? 'a' : 'w'), format);
    GraphDumper::DumpConfig config{keep_var_name, keep_param_name,
                                   keep_opr_priority};

    if (tensor_value_dumper) {
        config.tensor_value_dumper = [f=tensor_value_dumper](
                OutputFile &fout, const cg::OperatorNodeBase &opr,
                const HostTensorND &value) {
            mgb_assert(value.layout().is_contiguous());
            TensorValueDumperContext ctx{fout, opr, value};
            f->call(ctx);
        };
    }

    auto rst = dumper->dump(output_vars, config);
    inputs = std::move(rst.inputs);
    outputs = std::move(rst.outputs);
    params = std::move(rst.params);
    stat = {rst.nr_opr, rst.tot_bytes, rst.tensor_value_bytes,
            rst.content_hash};
}

CompGraph _load_comp_graph_from_file(
        const char* fpath, _CompNodeMapperCallback* cn_mapper,
        _TensorValueLoaderCallback* tensor_value_loader,
        std::vector<std::pair<std::string, SymbolVar>>& output_var_map,
        SymbolVarArray& output_var_list) {
    auto file = InputFile::make_fs(fpath);
    auto format = GraphLoader::identify_graph_dump_format(*file);
    mgb_throw_if(!format.valid(), SerializationError,
                 "unknown model format (input is likely not a MegBrain model)");
    auto loader = GraphLoader::make(std::move(file), format.val());
    GraphLoader::LoadConfig config;
    if (cn_mapper) {
        config.comp_node_mapper = [f = cn_mapper](CompNode::Locator& locator) {
            locator = CompNode::Locator::parse(f->call(locator.to_string()));
        };
    }
    if (tensor_value_loader) {
        config.tensor_value_loader = [f = tensor_value_loader](
                                             void* ptr,
                                             const TensorLayout& layout,
                                             InputFile& fin) {
            TensorValueLoaderContext ctx{layout, fin};
            PyObjRefKeeper value = f->call(ctx);
            mgb_assert(value.get()->ob_refcnt > 0);
            if (ptr) {
                HostTensorStorage storage;
                // Unmanaged shared_ptr
                storage.reset(CompNode::default_cpu(),
                              layout.span().dist_byte(),
                              {std::shared_ptr<dt_byte>(),
                               reinterpret_cast<dt_byte*>(ptr)});
                HostTensorND tensor;
                tensor.reset(storage, layout);
                npy::np2tensor(value.get(), npy::Meth::copy_into(&tensor),
                               layout.dtype);
            }
        };
    }
    auto rst = loader->load(config);
    output_var_map = {rst.output_var_map.begin(), rst.output_var_map.end()};
    output_var_list = std::move(rst.output_var_list);

    std::unordered_map<HostTensorND*, const std::string*> tensor2name;
    for (const auto& pair : rst.tensor_map) {
        tensor2name[pair.second.get()] = &pair.first;
    }
    auto cb = [&tensor2name, graph=rst.graph](cg::OperatorNodeBase* opr) {
        if (!opr->same_type<opr::Host2DeviceCopy>())
            return;

        auto& h2d = opr->cast_final_safe<opr::Host2DeviceCopy>();
        auto it = tensor2name.find(h2d.host_data().get());
        mgb_throw_if(it == tensor2name.end(), GraphError,
                     "unbound Host2DeviceCopy in loaded graph");
        h2d.output(0)->name(*it->second);
        mark_as_input(graph.get(), h2d.output(0));
    };
    cg::DepOprIter iter{cb};
    for (const auto& var : output_var_list) {
        iter.add(var.node()->owner_opr());
    }
    return CompGraph::make_from_shared_ptr(rst.graph);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
