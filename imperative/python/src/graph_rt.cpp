/**
 * \file imperative/python/src/graph_rt.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./graph_rt.h"

#include "megbrain/graph/cg.h"
#include "megbrain/serialization/serializer.h"
#include "megbrain/imperative/opr_utility.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/imperative.h"
#include "./helper.h"
#include "megbrain/plugin/profiler.h"
#include "./common.h"
#include "megbrain/gopt/inference.h"


namespace py = pybind11;

using namespace mgb;
using namespace imperative;
namespace ser = mgb::serialization;

using _OptimizeForInferenceOptions = mgb::gopt::OptimizeForInferenceOptions;
using _LayoutTransform = _OptimizeForInferenceOptions::LayoutTransform;

namespace {
class _CompGraphProfilerImpl {
    std::shared_ptr<ComputingGraph> m_comp_graph;
    GraphProfiler m_profiler;
    public:
        _CompGraphProfilerImpl(std::shared_ptr<ComputingGraph> cg):
            m_comp_graph{cg},
            m_profiler{m_comp_graph.get()}
        {
        }

        std::string _get_result() {
            auto json = m_profiler.to_json_full(
                    m_comp_graph->current_comp_seq());
            return json->to_string();
        }
};

struct WeakRendezvousArray:
    public std::vector<std::weak_ptr<RendezvousBase>>,
    public UserDataContainer::UserData {
    MGB_TYPEINFO_OBJ_DECL;
};
MGB_TYPEINFO_OBJ_IMPL(WeakRendezvousArray);
}
#define DEF_READWRITE(name) .def_readwrite(#name, &CURRENT_CLASS::name)

template<typename T>
auto def_rendezvous(py::object m, const char* name) {
    return py::class_<Rendezvous<T>, std::shared_ptr<Rendezvous<T>>>(m, name)
        .def(py::init([](){return Rendezvous<T>::make();}))
        .def("set", [](Rendezvous<T>& r, T v) {r.set(std::move(v));})
        .def("get", [](Rendezvous<T>& r) {return r.get();}, py::call_guard<py::gil_scoped_release>())
        .def("drop", &Rendezvous<T>::drop)
        .def("reset", &Rendezvous<T>::reset)
        .def("set_exception", [](Rendezvous<T>& r, std::string&& message) {
            r.set_exception(std::make_exception_ptr(
                    std::runtime_error(std::move(message))));
        });
}

using TensorAttr = LogicalTensorDesc;
using HostNDWithEvent = std::pair<HostTensorND, std::shared_ptr<CompNode::Event>>;

std::vector<mgb::cg::VarNode*>  _replace_vars(const std::vector<mgb::cg::VarNode*>& repl_src,
                                 const std::vector<mgb::cg::VarNode*>& repl_dst,
                                 const std::vector<mgb::cg::VarNode*>& vars) {
        mgb::ThinHashMap<SymbolVar, SymbolVar> varmap;
        for (size_t i = 0; i < repl_src.size(); ++i) {
            varmap[SymbolVar(repl_src[i])] = SymbolVar(repl_dst[i]);
        }
        SymbolVarArray symvars(vars.begin(), vars.end());
        auto sym_result = mgb::cg::replace_vars(symvars, varmap);
        std::vector<mgb::cg::VarNode*> result;
        for (auto symvar : sym_result){
            result.push_back(symvar.node());
        }
        return result;
    }

typedef std::vector<mgb::cg::OperatorNodeBase*> OperatorArray;
std::vector<mgb::cg::VarNode*> _replace_oprs(const OperatorArray& repl_src,
                                 const OperatorArray& repl_dst,
                                 const std::vector<mgb::cg::VarNode*>& vars) {
        mgb::ThinHashMap<mgb::cg::OperatorNodeBase*, mgb::cg::OperatorNodeBase*>
                oprmap;
        for (size_t i = 0; i < repl_src.size(); ++i) {
            oprmap[repl_src[i]] = repl_dst[i];
        }
        const SymbolVarArray symvars(vars.begin(), vars.end());
        auto sym_result = mgb::cg::replace_oprs(symvars, oprmap);
        std::vector<mgb::cg::VarNode*> result;
        for (auto symvar : sym_result){
            result.push_back(symvar.node());
        }
        return result;
    }



void _set_priority_to_id(const std::vector<mgb::cg::VarNode*>& dest_vars) {
        auto on_opr = [](mgb::cg::OperatorNodeBase* opr) {
            if (opr->node_prop().attribute().priority == 0) {
                opr->node_prop().attribute().priority = opr->id();
            }
        };
        mgb::cg::DepOprIter dep_iter{on_opr};
        for (const auto& var : dest_vars) {
            dep_iter.add(SymbolVar(var));
        }
}



void init_graph_rt(py::module m) {

   static const std::unique_ptr<mgb::OprFootprint> _imperative_sm_opr_footprint_ptr{std::make_unique<mgb::OprFootprint>()};

    def_rendezvous<DeviceTensorND>(m, "DeviceTensorNDRendezvous");

    def_rendezvous<HostNDWithEvent>(m, "HostTensorNDRendezvous");

    def_rendezvous<TensorAttr>(m, "TensorAttrRendezvous");

    py::class_<cg::VarNode, GraphNodePtr<cg::VarNode>>(m, "VarNode")
        .def_property_readonly("owner", [](cg::VarNode* v) {return v->owner_opr();})
        .def_property_readonly("graph", [](cg::VarNode* v) {return v->owner_graph();})
        .def_property("name", py::overload_cast<>(&VarNode::name, py::const_),
                      py::overload_cast<std::string>(&VarNode::name))
        .def_property_readonly("dtype", [](cg::VarNode* v) {return v->dtype();})
        .def_property_readonly("comp_node", [](cg::VarNode* v) {return v->comp_node();})
        .def_property_readonly("shape", [](cg::VarNode* v) -> const TensorShape* {
                auto&& mgr = v->owner_graph()->static_infer_manager();
                return mgr.infer_shape_fallible(v);
            })
        .def_property_readonly("value", [](cg::VarNode* v) -> py::object {
                auto&& mgr = v->owner_graph()->static_infer_manager();
                auto&& type = mgr.get_infer_type(v);
                using InferType = cg::static_infer::InferType;
                if (!(type.value & (InferType::CONST | InferType::RT_STATIC))) {
                    return py::none();
                }
                auto* val = mgr.infer_value_fallible(v);
                if (!val) {
                    return py::none();
                }
                return py::cast(*val).attr("numpy")();
            })
        .def_property_readonly("id",[](cg::VarNode* v){
            return (v->id());
        })
        .def("__repr__", [](cg::VarNode* v) {
            return "Var:" + v->name();
        });

    py::class_<cg::OperatorNodeBase, GraphNodePtr<cg::OperatorNodeBase>>(m, "OperatorNode")
        .def_property_readonly("graph", [](cg::OperatorNodeBase* opr) {return opr->owner_graph();})
        .def_property("name", py::overload_cast<>(&cg::OperatorNodeBase::name, py::const_),
                      py::overload_cast<std::string>(&cg::OperatorNodeBase::name))
        .def_property_readonly("inputs", [](cg::OperatorNodeBase* opr) {
                return to_tuple(opr->input());
            })
        .def_property_readonly("outputs", [](cg::OperatorNodeBase* opr) {
                return to_tuple(opr->usable_output());
            })
        .def_property_readonly("id",[](cg::OperatorNodeBase* opr){
            return opr->id();
        })
        .def_property_readonly("params",[](cg::OperatorNodeBase* opr){
            return _imperative_sm_opr_footprint_ptr->calc_footprint(opr).param->to_string();
        })
        .def_property_readonly("type",[](cg::OperatorNodeBase* opr){
            return opr->dyn_typeinfo()->name;
        })
        .def("__repr__", [](cg::OperatorNodeBase* opr){
            return "Opr:" + opr->name();
        });

    py::class_<cg::AsyncExecutable>(m, "AsyncExecutable")
        .def("execute", &cg::AsyncExecutable::execute, py::call_guard<py::gil_scoped_release>())
        .def("wait", &cg::AsyncExecutable::wait, py::call_guard<py::gil_scoped_release>())
        // only used for exception handle
        .def_property_readonly("_all_rendezvous", [](cg::AsyncExecutable* exec) {
            auto ud = exec->owner_graph()->options().user_data
                        .get_user_data<WeakRendezvousArray>();
            std::vector<std::shared_ptr<RendezvousBase>> ret;
            if (ud.second) {
                for (auto&& r: *ud.first[0]) {
                    if (auto p = r.lock()) {
                        ret.emplace_back(std::move(p));
                    }
                }
            }
            return ret;
        });

    auto PyComputingGraph = py::class_<cg::ComputingGraph, std::shared_ptr<cg::ComputingGraph>>(m, "ComputingGraph")
        .def(py::init(py::overload_cast<>(&cg::ComputingGraph::make)))
        .def("compile", [](cg::ComputingGraph& graph, const std::vector<cg::VarNode*>& dest_vars) {
                mgb_assert(!dest_vars.empty());
                cg::ComputingGraph::OutputSpec spec;
                for (auto v : dest_vars) {
                    spec.emplace_back(v, nullptr);
                }
                return graph.compile(spec);
            })
        .def_property_readonly("options", py::overload_cast<>(&cg::ComputingGraph::options));

    py::class_<_CompGraphProfilerImpl, std::shared_ptr<_CompGraphProfilerImpl>>(m, "GraphProfiler")
        .def(py::init([](std::shared_ptr<ComputingGraph> graph) {
                return std::make_shared<_CompGraphProfilerImpl>(graph);
                }))
        .def("get", [](_CompGraphProfilerImpl& profiler) { return profiler._get_result(); });

    auto GraphOptimizeOptions = py::class_<_OptimizeForInferenceOptions>(m, "GraphOptimizeOptions")
        .def(py::init())
        .def_readwrite("f16_io_f32_comp", &_OptimizeForInferenceOptions::f16_io_f32_comp)
        .def_readwrite("f16_io_comp", &_OptimizeForInferenceOptions::f16_io_comp)
        .def_readwrite("fuse_conv_bias_nonlinearity", &_OptimizeForInferenceOptions::fuse_conv_bias_nonlinearity)
        .def_readwrite("fuse_conv_bias_with_z", &_OptimizeForInferenceOptions::fuse_conv_bias_with_z)
        .def_readwrite("layout_transform", &_OptimizeForInferenceOptions::layout_transform)
        ;

    py::enum_<_LayoutTransform>(GraphOptimizeOptions, "LayoutTransform")
        .value("DEFAULT", _LayoutTransform::DEFAULT)
        .value("NCHW4", _LayoutTransform::NCHW4)
        .value("NHWCD4", _LayoutTransform::NHWCD4)
        .value("NCHW88", _LayoutTransform::NCHW88)
        .value("NCHW44", _LayoutTransform::NCHW44)
        .value("NCHW44_DOT", _LayoutTransform::NCHW44_DOT)
        .value("NCHW32", _LayoutTransform::NCHW32)
        .value("CHWN4", _LayoutTransform::CHWN4)
        .export_values()
        ;

    m.def("optimize_for_inference", [](const VarNodeArray& dest_vars, const _OptimizeForInferenceOptions& opt) {
        SymbolVarArray symvars(dest_vars.begin(), dest_vars.end());
        auto res_symvars = mgb::gopt::optimize_for_inference(symvars, opt);
        VarNodeArray vars;
        for (auto& si: res_symvars)
            vars.push_back(si.node());
        return vars;
    });

    m.def("get_info_for_strip", [](const std::vector<VarNode*>& dest_vars) {
        std::unordered_set<const char*> opr_types, dtype_names, elemwise_modes;
        auto on_opr = [&](cg::OperatorNodeBase *opr) {
            if (ser::GraphDumper::should_remove_in_dump(opr))
                return;
            opr_types.insert(opr->dyn_typeinfo()->name);
            for (auto i : opr->output())
                dtype_names.insert(i->dtype().name());
            if (opr->same_type<opr::Elemwise>()) {
                auto mode = opr->cast_final<opr::Elemwise>().param().mode;
                elemwise_modes.insert(
                        megdnn::Elemwise::ModeTrait::from_mode(mode).name);
            }
        };
        cg::DepOprIter opr_iter{on_opr};
        for (auto i : dest_vars)
            opr_iter.add(i->owner_opr());

        auto to_json = [](const std::unordered_set<const char*> &v) {
            std::vector<std::string> vs(v.begin(), v.end());
            std::sort(vs.begin(), vs.end());
            auto ret = json::Array::make();
            for (auto &&i : vs)
                ret->add(json::String::make(i));
            return ret;
        };

        return json::Object::make({
            {"opr_types", to_json(opr_types)},
            {"dtypes", to_json(dtype_names)},
            {"elemwise_modes", to_json(elemwise_modes)},
        })->to_string();
    });

    m.def("dump_graph", [](
        const std::vector<VarNode*>& dest_vars,
        int keep_var_name,
        bool keep_param_name,
        bool keep_opr_priority,
        py::list& stat,
        py::list& inputs,
        py::list& outputs,
        py::list& params
    ) {
        std::vector<uint8_t> buf;
        auto dumper = ser::GraphDumper::make(ser::OutputFile::make_vector_proxy(&buf));
        SymbolVarArray symvars(dest_vars.begin(), dest_vars.end());

        ser::GraphDumper::DumpConfig config{keep_var_name, keep_param_name,
                                       keep_opr_priority};

        auto rst = dumper->dump(symvars, config);
        for (auto i : rst.inputs) {
            inputs.append(py::cast(i));
        }
        for (auto i : rst.outputs) {
            outputs.append(py::cast(i));
        }
        for (auto i : rst.params) {
            params.append(py::cast(i));
        }
        auto rst_stat =
                std::vector{rst.nr_opr, rst.tot_bytes, rst.tensor_value_bytes,
                            static_cast<size_t>(rst.content_hash)};
        for (auto i : rst_stat) {
            stat.append(py::cast(i));
        }
        return py::bytes(reinterpret_cast<const char*>(&buf[0]), buf.size());
    });

    m.def("load_graph", [](
        std::string& buf,
        py::list& output_var_map,
        py::list& output_var_list
    ) {
        auto file = ser::InputFile::make_mem_proxy(buf.c_str(), buf.length());
        auto format = ser::GraphLoader::identify_graph_dump_format(*file);
        auto loader = ser::GraphLoader::make(std::move(file), format.val());
        ser::GraphLoader::LoadConfig config;
        auto rst = loader->load(config);
        for (auto i : rst.output_var_map) {
            output_var_map.append(py::make_tuple(i.first, i.second.node()));
        }
        for (auto i : rst.output_var_list) {
            output_var_list.append(i.node());
        }
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
        };
        cg::DepOprIter iter{cb};
        for (const auto& var : rst.output_var_list) {
            iter.add(var);
        }
        return rst.graph;

    });

#define CURRENT_CLASS cg::ComputingGraph::Options

    auto PyComputingGraphOptions = py::class_<cg::ComputingGraph::Options>(PyComputingGraph, "Options")
        // DEF_READWRITE(opr_attribute)
        DEF_READWRITE(seq_opt)
        DEF_READWRITE(graph_opt)
        DEF_READWRITE(graph_opt_level)
        DEF_READWRITE(log_level)
        DEF_READWRITE(async_exec_level)
        DEF_READWRITE(force_dynamic_alloc)
        DEF_READWRITE(var_sanity_check_first_run)
        DEF_READWRITE(allocate_static_mem_after_graph_compile)
        DEF_READWRITE(fake_next_exec)
        DEF_READWRITE(enable_sublinear_memory_opt)
        DEF_READWRITE(no_profiling_on_shape_change)
        DEF_READWRITE(enable_var_mem_defragment)
        DEF_READWRITE(enable_grad_var_static_reshape)
        DEF_READWRITE(enable_memory_swap)
        DEF_READWRITE(comp_node_seq_record_level)
        DEF_READWRITE(no_force_inplace)
        DEF_READWRITE(sublinear_mem_config)
        // DEF_READWRITE(eager_evaluation)
        // DEF_READWRITE(imperative_proxy_graph)
        // DEF_READWRITE(extra_vardeps)
        // DEF_READWRITE(user_data)
        ;

#undef CURRENT_CLASS
#define CURRENT_CLASS cg::ComputingGraph::Options::SeqOpt

    py::class_<cg::ComputingGraph::Options::SeqOpt>(PyComputingGraphOptions, "SeqOpt")
        DEF_READWRITE(enable_mem_plan_opt)
        DEF_READWRITE(enable_mem_reuse_alloc)
        DEF_READWRITE(enable_seq_comp_node_opt);

#undef CURRENT_CLASS
#define CURRENT_CLASS cg::ComputingGraph::Options::GraphOpt

    py::class_<cg::ComputingGraph::Options::GraphOpt>(PyComputingGraphOptions, "GraphOpt")
        DEF_READWRITE(jit)
        DEF_READWRITE(tensorrt);

#undef CURRENT_CLASS

#define CURRENT_CLASS cg::ComputingGraph::Options::SublinearMemConfig

    py::class_<cg::ComputingGraph::Options::SublinearMemConfig>(PyComputingGraphOptions, "SublinearMemConfig")
        DEF_READWRITE(thresh_nr_try)
        DEF_READWRITE(genetic_nr_iter)
        DEF_READWRITE(genetic_pool_size)
        DEF_READWRITE(lb_memory)
        DEF_READWRITE(num_worker);

#undef CURRENT_CLASS
    auto common = rel_import("common", m, 1);

    common.def("invoke_op", [](const OpDef& def, const std::vector<cg::VarNode*> inputs, cg::ComputingGraph* graph) {
            cg::VarNodeArray vinputs(inputs.begin(), inputs.end());
            return to_tuple(OpDef::apply_on_var_node(def, vinputs));
        },
        py::arg(), py::arg(), py::arg("graph") = py::none());

    auto input_callback = [](auto callback,
                             const CompNode& comp_node,
                             const DType& dtype,
                             const TensorShape& shape,
                             const std::vector<cg::VarNode*>& inputs,
                             cg::ComputingGraph* graph,
                             bool use_static_shape) {
        if (!graph) {
            graph = inputs[0]->owner_graph();
        }
        SymbolVarArray sinputs;
        for (auto i : inputs) {
            sinputs.emplace_back(i);
        }
        static_assert(!std::is_reference<decltype(callback)>::value);
        auto soutputs = opr::InputCallback::make(*graph, std::move(callback),
                                                 comp_node, dtype, shape,
                                                 sinputs, use_static_shape);
        std::vector<VarNode*> outputs;
        outputs.reserve(soutputs.size());
        for (auto i : soutputs) {
            outputs.push_back(i.node());
        }
        return outputs;
    };

    m.def("make_shared", [](cg::ComputingGraph* graph, const DeviceTensorND& data) {
            return opr::SharedDeviceTensor::make(*graph, std::make_shared<DeviceTensorND>(data)).node();
        });

    m.def("make_const", [](cg::ComputingGraph* graph, py::array data, CompNode cn, DType dtype) {
            if (!cn.valid()) {
                cn = CompNode::load(get_default_device());
            }
            auto hv = npy::np2tensor(data.ptr(), npy::Meth::borrow(cn), dtype);
            return opr::ImmutableTensor::make(*graph, hv, OperatorNodeConfig(cn)).node();
        });

    m.def("make_h2d", [](cg::ComputingGraph& graph, CompNode cn, DType dtype, TensorShape shape, std::optional<std::string> name) {
            if (!cn.valid()) {
                throw py::type_error("device must be valid");
            }
            if (!dtype.valid()) {
                throw py::type_error("dtype must be valid");
            }
            OperatorNodeConfig config;
            if (name) {
                config.name(*name);
            }
            return opr::Host2DeviceCopy::make(graph, std::make_shared<HostTensorND>(cn, shape, dtype), config).node();
        }, py::arg(), py::arg(), py::arg(), py::arg() = py::none(), py::arg() = py::none());

    m.def("_replace_vars", &_replace_vars,py::arg(),py::arg(),py::arg());
    m.def("_replace_oprs", &_replace_oprs,py::arg(),py::arg(),py::arg());
    m.def("_set_priority_to_id",&_set_priority_to_id,py::arg());

    m.def("input_callback", [input_callback](std::function<DeviceTensorND(void)> callback,
                                             const CompNode& comp_node,
                                             const DType& dtype,
                                             const TensorShape& shape,
                                             const std::vector<cg::VarNode*>& inputs,
                                             cg::ComputingGraph* graph,
                                             bool use_static_shape) {
            return input_callback(
                [f=std::move(callback)](){py::gil_scoped_acquire _; return f();},
                comp_node, dtype, shape, inputs, graph, use_static_shape);
        },
        py::arg(), py::arg(), py::arg(), py::arg() = py::none(), py::arg() = py::tuple(),
        py::arg("graph") = py::none(), py::arg("use_static_shape") = false);

    m.def("input_callback", [input_callback](std::shared_ptr<Rendezvous<DeviceTensorND>> p,
                                             const CompNode& comp_node,
                                             const DType& dtype,
                                             const TensorShape& shape,
                                             const std::vector<cg::VarNode*>& inputs,
                                             cg::ComputingGraph* graph,
                                             bool use_static_shape) {
            auto f = [p]() -> DeviceTensorND {
                return p->get();
            };
            return input_callback(std::move(f), comp_node, dtype, shape, inputs, graph, use_static_shape);
        },
        py::arg(), py::arg(), py::arg(), py::arg() = py::none(), py::arg() = py::tuple(), 
        py::arg("graph") = py::none(), py::arg("use_static_shape") = false);

    auto output_callback = [](auto callback, const std::vector<cg::VarNode*>& inputs,
            std::shared_ptr<RendezvousBase> r = {}, bool borrow = false, bool prefer_host_value = false) {
        if (r) {
            mgb_assert(inputs.size());
            auto cg = inputs[0]->owner_graph();
            cg->options().user_data.get_user_data_or_create<WeakRendezvousArray>()
                    ->emplace_back(r);
        }
        SymbolVarArray sinputs;
        for (auto i : inputs) {
            sinputs.emplace_back(i);
        }
        static_assert(!std::is_reference<decltype(callback)>::value);
        opr::OutputCallback::Param param{std::move(callback), borrow, prefer_host_value};
        auto output = opr::OutputCallback::make(std::move(param), sinputs);
        return output.node();
    };

    m.def("output_callback", [output_callback](std::function<void(DeviceTensorND)> callback, std::vector<cg::VarNode*> inputs) {
        auto f = [f=std::move(callback)](DeviceTensorND dv) {
            auto task = [f=std::move(f), dv=std::move(dv)]() {
                f(dv);
            };
            py_task_q.add_task(std::move(task));
        };
        return output_callback(std::move(f), std::move(inputs));
    });

    m.def("output_callback", [output_callback](std::shared_ptr<Rendezvous<DeviceTensorND>> p, std::vector<cg::VarNode*> inputs) {
        auto f = [p](DeviceTensorND dv) {
            p->set(std::move(dv));
        };
        return output_callback(std::move(f), std::move(inputs), p);
    });

    m.def("value_output_callback", [output_callback](std::shared_ptr<Rendezvous<HostNDWithEvent>> p, std::vector<cg::VarNode*> inputs) {
        auto f = [p](DeviceTensorND dv) {
            HostNDWithEvent hv_with_event;
            hv_with_event.first.copy_from(dv);
            hv_with_event.second = dv.comp_node().create_event();
            hv_with_event.second->record();
            p->set(std::move(hv_with_event));
        };
        return output_callback(std::move(f), std::move(inputs), p, true, true);
    });

    m.def("attr_output_callback", [output_callback](std::shared_ptr<Rendezvous<TensorAttr>> p, std::vector<cg::VarNode*> inputs) {
        auto f = [p](DeviceTensorND dv) {
            p->set(TensorAttr{TensorLayout{dv.shape(), dv.dtype()}, dv.comp_node()});
        };
        return output_callback(std::move(f), std::move(inputs), p, true);
    });
}
