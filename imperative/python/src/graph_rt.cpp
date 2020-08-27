#include "./graph_rt.h"

#include "megbrain/imperative/opr_utility.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/imperative.h"
#include "./helper.h"

namespace py = pybind11;

using namespace mgb;
using namespace imperative;

#define DEF_READWRITE(name) .def_readwrite(#name, &CURRENT_CLASS::name)

template<typename T>
auto def_rendezvous(py::object m, const char* name) {
    return py::class_<Rendezvous<T>, std::shared_ptr<Rendezvous<T>>>(m, name)
        .def(py::init([](){return std::make_shared<Rendezvous<T>>();}))
        .def("set", [](Rendezvous<T>& r, T v) {r.set(std::move(v));})
        .def("get", [](Rendezvous<T>& r) {return r.get();}, py::call_guard<py::gil_scoped_release>())
        .def("reset", &Rendezvous<T>::reset);
}

using TensorAttr = LogicalTensorDesc;

void init_graph_rt(py::module m) {
    def_rendezvous<DeviceTensorND>(m, "DeviceTensorNDRendezvous");

    def_rendezvous<TensorAttr>(m, "TensorAttrRendezvous");

    py::class_<cg::VarNode, GraphNodePtr<cg::VarNode>>(m, "VarNode")
        .def_property_readonly("owner", [](cg::VarNode* v) {return v->owner_opr();})
        .def_property_readonly("graph", [](cg::VarNode* v) {return v->owner_graph();})
        .def_property_readonly("dtype", [](cg::VarNode* v) {return v->dtype();})
        .def_property_readonly("comp_node", [](cg::VarNode* v) {return v->comp_node();});

    py::class_<cg::OperatorNodeBase, GraphNodePtr<cg::OperatorNodeBase>>(m, "OperatorNode")
        .def_property_readonly("graph", [](cg::OperatorNodeBase* opr) {return opr->owner_graph();})
        .def_property_readonly("inputs", [](cg::OperatorNodeBase* opr) {
                return to_tuple(opr->input());
            })
        .def_property_readonly("outputs", [](cg::OperatorNodeBase* opr) {
                return to_tuple(opr->output());
            });

    py::class_<cg::AsyncExecutable>(m, "AsyncExecutable")
        .def("execute", &cg::AsyncExecutable::execute, py::call_guard<py::gil_scoped_release>())
        .def("wait", &cg::AsyncExecutable::wait, py::call_guard<py::gil_scoped_release>());

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

    auto common = rel_import("common", m, 1);

    common.def("invoke_op", [](const OpDef& def, const std::vector<cg::VarNode*> inputs, cg::ComputingGraph* graph) {
            cg::VarNodeArray vinputs(inputs.begin(), inputs.end());
            auto opr = OpDef::apply_on_var_node(def, vinputs);
            auto outputs = opr->output();
            return to_tuple(outputs);
        },
        py::arg(), py::arg(), py::arg("graph") = py::none());

    auto input_callback = [](auto callback,
                             const CompNode& comp_node,
                             const DType& dtype,
                             const std::vector<cg::VarNode*>& inputs,
                             cg::ComputingGraph* graph) {
        if (!graph) {
            graph = inputs[0]->owner_graph();
        }
        SymbolVarArray sinputs;
        for (auto i : inputs) {
            sinputs.emplace_back(i);
        }
        static_assert(!std::is_reference<decltype(callback)>::value);
        auto soutputs = opr::InputCallback::make(*graph, std::move(callback), comp_node, dtype, sinputs);
        std::vector<VarNode*> outputs;
        outputs.reserve(soutputs.size());
        for (auto i : soutputs) {
            outputs.push_back(i.node());
        }
        return outputs;
    };

    m.def("input_callback", [input_callback](std::function<DeviceTensorND(void)> callback,
                                             const CompNode& comp_node,
                                             const DType& dtype,
                                             const std::vector<cg::VarNode*>& inputs,
                                             cg::ComputingGraph* graph) {
            return input_callback([f=std::move(callback)](){py::gil_scoped_acquire _; return f();}, comp_node, dtype, inputs, graph);
        },
        py::arg(), py::arg(), py::arg(), py::arg() = py::tuple(), py::arg("graph") = py::none());

    m.def("input_callback", [input_callback](std::shared_ptr<Rendezvous<DeviceTensorND>> p,
                                             const CompNode& comp_node,
                                             const DType& dtype,
                                             const std::vector<cg::VarNode*>& inputs,
                                             cg::ComputingGraph* graph) {
            auto f = [p]() -> DeviceTensorND {
                return p->get();
            };
            return input_callback(std::move(f), comp_node, dtype, inputs, graph);
        },
        py::arg(), py::arg(), py::arg(), py::arg() = py::tuple(), py::arg("graph") = py::none());

    auto output_callback = [](auto callback, const std::vector<cg::VarNode*>& inputs, bool borrow = false) {
        SymbolVarArray sinputs;
        for (auto i : inputs) {
            sinputs.emplace_back(i);
        }
        static_assert(!std::is_reference<decltype(callback)>::value);
        opr::OutputCallback::Param param{std::move(callback), borrow};
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
        return output_callback(std::move(f), std::move(inputs));
    });

    m.def("attr_output_callback", [output_callback](std::shared_ptr<Rendezvous<TensorAttr>> p, std::vector<cg::VarNode*> inputs) {
        auto f = [p](DeviceTensorND dv) {
            p->set(TensorAttr{TensorLayout{dv.shape(), dv.dtype()}, dv.comp_node()});
        };
        return output_callback(std::move(f), std::move(inputs), true);
    });
}
