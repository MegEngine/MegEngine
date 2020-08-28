#include "./common.h"

#include <pybind11/operators.h>

#include "megbrain/comp_node.h"
#include "megbrain/graph.h"
#include "megbrain/imperative/physical_tensor.h"
#include "./numpy_dtypes.h"
#include "./helper.h"

namespace py = pybind11;
using namespace mgb;
using namespace imperative;

void init_common(py::module m) {
    py::class_<CompNode>(m, "CompNode")
        .def(py::init())
        .def(py::init(py::overload_cast<const std::string&>(&CompNode::load)))
        .def("__str__", &CompNode::to_string_logical)
        .def_static("_sync_all", &CompNode::sync_all)
        .def(py::self == py::self)
        .def_static("_get_device_count", &CompNode::get_device_count,
                    "Get total number of specific devices on this system")
        .def(py::pickle(
                [](const CompNode& cn) {
                    return py::str(cn.to_string_logical());
                },
                [](py::str cn) {
                    return CompNode::load(cn);
                }));

    py::implicitly_convertible<std::string, CompNode>();

    py::class_<DeviceTensorND>(m, "DeviceTensorND")
        .def(py::init())
        .def_property_readonly("shape", py::overload_cast<>(&DeviceTensorND::shape, py::const_))
        .def_property_readonly("dtype", py::overload_cast<>(&DeviceTensorND::dtype, py::const_))
        .def_property_readonly("comp_node", py::overload_cast<>(&DeviceTensorND::comp_node, py::const_))
        .def("numpy", [](const DeviceTensorND& self) {
                HostTensorND hv;
                hv.copy_from(self).sync();
                return py::handle(npy::ndarray_from_tensor(hv, npy::ShareType::TRY_SHARE));
            });

    py::class_<cg::OperatorNodeConfig>(m, "OperatorNodeConfig")
        .def(py::init())
        .def_property("name",
            [](const OperatorNodeConfig& config) -> py::object {
                auto name = config.name();
                if (name.valid()) {
                    return py::str(name.val());
                } else {
                    return py::none();
                }
            },
            [](OperatorNodeConfig& config, std::string name){
                config.name(std::move(name));
            })
        .def_property("dtype",
            [](const OperatorNodeConfig& config) {
                return config.output_dtype();
            },
            [](OperatorNodeConfig& config, DType dtype) {
                config.output_dtype(dtype);
            })
        .def_property("comp_node_arr",
            [](const OperatorNodeConfig& config) -> py::tuple {
                auto arr = config.comp_node();
                std::vector<CompNode> tmp(arr.begin(), arr.end());
                return py::cast(tmp);
            },
            [](OperatorNodeConfig& config, std::vector<CompNode> cns) {
                config.comp_node_arr({cns.begin(), cns.end()});
            })
        .def_property("comp_node",
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
        .def(py::init([](const TensorShape& shape, const DType& dtype, const CompNode& comp_node){
                return LogicalTensorDesc{TensorLayout{shape, dtype}, comp_node};
            }))
        .def_property("shape",
            [](const LogicalTensorDesc& desc) {
                return static_cast<TensorShape>(desc.layout);
            },
            [](LogicalTensorDesc& desc, TensorShape shape) {
            })
        .def_property("dtype",
            [](const LogicalTensorDesc& desc) {
                return desc.layout.dtype;
            },
            [](LogicalTensorDesc& desc, DType dtype) {
                desc.layout.dtype = dtype;
            })
        .def_readwrite("comp_node", &LogicalTensorDesc::comp_node);

    py::enum_<CompNode::DeviceType>(m, "DeviceType")
            .value("UNSPEC", CompNode::DeviceType::UNSPEC)
            .value("CUDA", CompNode::DeviceType::CUDA)
            .value("CPU", CompNode::DeviceType::CPU)
            .value("MULTITHREAD", CompNode::DeviceType::MULTITHREAD)
            .value("MAX_DEVICE_ID", CompNode::DeviceType::MAX_DEVICE_ID);

    init_npy_num_bfloat16(m);
    init_npy_num_intbx(m);
}
