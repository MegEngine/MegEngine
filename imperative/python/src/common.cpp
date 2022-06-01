#include "./common.h"

#include <pybind11/operators.h>
#include <pybind11/pytypes.h>

#include "./helper.h"
#include "./numpy_dtypes.h"
#include "megbrain/comp_node.h"
#include "megbrain/graph.h"
#include "megbrain/imperative/physical_tensor.h"
#if MGB_ENABLE_OPR_MM
#include "megbrain/opr/mm_handler.h"
#endif

#if MEGDNN_WITH_CUDA
#include "cuda_sm_gen.h"
#endif

namespace py = pybind11;
using namespace mgb;
using namespace imperative;

namespace {

template <typename XTensorND>
auto def_TensorND(py::object parent, const char* name) {
    return py::class_<XTensorND>(parent, name)
            .def_property_readonly(
                    "shape", py::overload_cast<>(&XTensorND::shape, py::const_))
            .def_property_readonly(
                    "dtype", py::overload_cast<>(&XTensorND::dtype, py::const_))
            .def_property_readonly(
                    "comp_node", py::overload_cast<>(&XTensorND::comp_node, py::const_))
            .def("copy_from", &XTensorND::template copy_from<DeviceTensorStorage>)
            .def("copy_from", &XTensorND::template copy_from<HostTensorStorage>)
            .def("copy_from_fixlayout",
                 py::overload_cast<const DeviceTensorND&>(
                         &XTensorND::template copy_from_fixlayout<DeviceTensorStorage>))
            .def("copy_from_fixlayout",
                 py::overload_cast<const HostTensorND&>(
                         &XTensorND::template copy_from_fixlayout<HostTensorStorage>));
}

std::string default_device = "xpux";

}  // namespace

void set_default_device(const std::string& device) {
    default_device = device;
}

void init_nccl_env(const std::string& ip, int port, int nranks, int rank, int root) {
#if MGB_ENABLE_OPR_MM
    auto&& help = mgb::opr::BatchSendRecvHelper::getInstance();
    bool res = help->init(nranks, rank, ip, port, root);
    auto p = help->get(std::string("init_all_cards"));
#else
    mgb_throw(
            MegBrainError,
            "MegEngine compiled without MM opr, doesn't support init_nccl_env");
#endif
}

std::string get_default_device() {
    return default_device;
}

py::handle py_comp_node_type;

void init_common(py::module m) {
    auto PyCompNode =
            py::class_<CompNode>(m, "CompNode")
                    .def(py::init())
                    .def(py::init(
                            py::overload_cast<const std::string&>(&CompNode::load)))
                    .def_property_readonly(
                            "logical_name",
                            [](const CompNode& cn) { return cn.to_string_logical(); })
                    .def_property_readonly(
                            "physical_name",
                            [](const CompNode& cn) { return cn.to_string_physical(); })
                    .def_property_readonly(
                            "get_mem_status_bytes",
                            [](const CompNode& cn) {
                                return cn.get_mem_status_bytes();
                            })
                    .def_property_readonly(
                            "get_used_memory",
                            [](const CompNode& cn) { return cn.get_used_memory(); })
                    .def_property_readonly(
                            "get_max_used_memory",
                            [](const CompNode& cn) { return cn.get_max_used_memory(); })
                    .def_property_readonly(
                            "get_reserved_memory",
                            [](const CompNode& cn) { return cn.get_reserved_memory(); })
                    .def_property_readonly(
                            "get_max_reserved_memory",
                            [](const CompNode& cn) {
                                return cn.get_max_reserved_memory();
                            })
                    .def_static(
                            "reset_max_memory_stats",
                            [](const CompNode& cn) {
                                cn.reset_max_used_memory();
                                cn.reset_max_reserved_memory();
                            })
                    .def("create_event", &CompNode::create_event,
                         py::arg("flags") = 0ul)
                    .def_static("_set_default_device", &set_default_device)
                    .def_static("_get_default_device", &get_default_device)
                    .def("__str__", &CompNode::to_string_logical)
                    .def("__repr__",
                         [](const CompNode& cn) {
                             return mgb::ssprintf(
                                     "CompNode(\"%s\" from \"%s\")",
                                     cn.to_string_physical().c_str(),
                                     cn.to_string_logical().c_str());
                         })
                    .def("__hash__", [](CompNode cn) { return mgb::hash(cn); })
                    .def_static("_sync_all", &CompNode::sync_all)
                    .def(py::self == py::self)
                    .def_static(
                            "_get_device_count", &CompNode::get_device_count,
                            "Get total number of specific devices on this system")
                    .def(py::pickle(
                            [](const CompNode& cn) {
                                return py::str(cn.to_string_logical());
                            },
                            [](py::str cn) { return CompNode::load(cn); }));

    py_comp_node_type = PyCompNode.inc_ref();

    py::class_<CompNode::Event, std::shared_ptr<CompNode::Event>>(PyCompNode, "Event")
            .def("record", &CompNode::Event::record)
            .def("wait", &CompNode::Event::host_wait);

    py::implicitly_convertible<std::string, CompNode>();

    py::class_<CompNode::DeviceProperties>(m, "DeviceProperties")
            .def(py::init())
            .def_property_readonly(
                    "name",
                    [](const CompNode::DeviceProperties prop) { return prop.name; })
            .def_property_readonly(
                    "total_memory",
                    [](const CompNode::DeviceProperties prop) {
                        return prop.total_memory;
                    })
            .def_property_readonly(
                    "major",
                    [](const CompNode::DeviceProperties prop) { return prop.major; })
            .def_property_readonly("minor", [](const CompNode::DeviceProperties prop) {
                return prop.minor;
            });

    def_TensorND<DeviceTensorND>(m, "DeviceTensorND")
            .def("numpy", [](const DeviceTensorND& self) {
                HostTensorND hv;
                hv.copy_from(self).sync();
                return py::reinterpret_steal<py::object>(
                        npy::ndarray_from_tensor(hv, npy::ShareType::TRY_SHARE));
            });

    def_TensorND<HostTensorND>(m, "HostTensorND")
            .def(py::init([](py::array data, CompNode cn, DType dtype) {
                if (!cn.valid()) {
                    throw py::type_error("device must not be None");
                }
                return npy::np2tensor(data.ptr(), npy::Meth::borrow(cn), dtype);
            }))
            .def("numpy", [](const HostTensorND& self) {
                return py::reinterpret_steal<py::object>(
                        npy::ndarray_from_tensor(self, npy::ShareType::TRY_SHARE));
            });

    py::class_<cg::OperatorNodeConfig>(m, "OperatorNodeConfig")
            .def(py::init())
            .def_property(
                    "name",
                    [](const OperatorNodeConfig& config) -> py::object {
                        auto name = config.name();
                        if (name.valid()) {
                            return py::str(name.val());
                        } else {
                            return py::none();
                        }
                    },
                    [](OperatorNodeConfig& config, std::string name) {
                        config.name(std::move(name));
                    })
            .def_property(
                    "dtype",
                    [](const OperatorNodeConfig& config) {
                        return config.output_dtype();
                    },
                    [](OperatorNodeConfig& config, DType dtype) {
                        config.output_dtype(dtype);
                    })
            .def_property(
                    "comp_node_arr",
                    [](const OperatorNodeConfig& config) -> py::tuple {
                        auto arr = config.comp_node();
                        std::vector<CompNode> tmp(arr.begin(), arr.end());
                        return py::cast(tmp);
                    },
                    [](OperatorNodeConfig& config, std::vector<CompNode> cns) {
                        config.comp_node_arr({cns.begin(), cns.end()});
                    })
            .def_property(
                    "comp_node",
                    [](const OperatorNodeConfig& config) {
                        auto arr = config.comp_node();
                        if (arr.size() != 1) {
                            throw py::value_error("invalid number of comp_node");
                        }
                        return arr[0];
                    },
                    [](OperatorNodeConfig& config, CompNode cn) {
                        OperatorNodeConfig::CompNodeArray arr{cn};
                        config.comp_node_arr(arr);
                    });

    py::class_<LogicalTensorDesc>(m, "TensorAttr")
            .def(py::init())
            .def(py::init([](const TensorShape& shape, const DType& dtype,
                             const CompNode& comp_node) {
                return LogicalTensorDesc{TensorLayout{shape, dtype}, comp_node};
            }))
            .def_property(
                    "shape",
                    [](const LogicalTensorDesc& desc) {
                        return static_cast<TensorShape>(desc.layout);
                    },
                    [](LogicalTensorDesc& desc, TensorShape shape) {})
            .def_property(
                    "dtype",
                    [](const LogicalTensorDesc& desc) { return desc.layout.dtype; },
                    [](LogicalTensorDesc& desc, DType dtype) {
                        desc.layout.dtype = dtype;
                    })
            .def_readwrite("comp_node", &LogicalTensorDesc::comp_node);

    py::enum_<CompNode::DeviceType>(m, "DeviceType")
            .value("UNSPEC", CompNode::DeviceType::UNSPEC)
            .value("CUDA", CompNode::DeviceType::CUDA)
            .value("ROCM", CompNode::DeviceType::ROCM)
            .value("CPU", CompNode::DeviceType::CPU)
            .value("CAMBRICON", CompNode::DeviceType::CAMBRICON)
            .value("ATLAS", CompNode::DeviceType::ATLAS)
            .value("MULTITHREAD", CompNode::DeviceType::MULTITHREAD)
            .value("MAX_DEVICE_ID", CompNode::DeviceType::MAX_DEVICE_ID);

    m.def("set_prealloc_config", &CompNode::set_prealloc_config,
          "specifies how to pre-allocate from raw dev allocator");

    m.def("get_device_prop", &CompNode::get_device_prop);

    m.def("get_supported_sm_versions", []() {
#if MEGDNN_WITH_CUDA
        static const char* mge_gen_code = MGE_CUDA_GENCODE;
#else
        static const char* mge_gen_code = "-1";
#endif
        return mge_gen_code;
    });

    m.def("what_is_xpu",
          [] { return CompNode::Locator::parse("xpux").to_physical().type; });

    m.def("init_nccl_env", &init_nccl_env);

    init_npy_num_bfloat16(m);
    init_npy_num_intbx(m);
    init_dtypes(m);
}
