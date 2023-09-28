// clang-format off
py::class_<AdaptivePooling, std::shared_ptr<AdaptivePooling>, OpDef> AdaptivePoolingInst(m, "AdaptivePooling");

py::enum_<AdaptivePooling::Mode>(AdaptivePoolingInst, "Mode")
    .value("MAX", AdaptivePooling::Mode::MAX)
    .value("AVERAGE", AdaptivePooling::Mode::AVERAGE)
    .value("AVERAGE_COUNT_EXCLUDE_PADDING", AdaptivePooling::Mode::AVERAGE_COUNT_EXCLUDE_PADDING)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "MAX") return AdaptivePooling::Mode::MAX;
        if (str == "AVERAGE") return AdaptivePooling::Mode::AVERAGE;
        if (str == "AVERAGE_COUNT_EXCLUDE_PADDING") return AdaptivePooling::Mode::AVERAGE_COUNT_EXCLUDE_PADDING;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, AdaptivePooling::Mode>();

py::enum_<AdaptivePooling::Format>(AdaptivePoolingInst, "Format")
    .value("NCHW", AdaptivePooling::Format::NCHW)
    .value("NHWC", AdaptivePooling::Format::NHWC)
    .value("NHWCD4", AdaptivePooling::Format::NHWCD4)
    .value("NCHW4", AdaptivePooling::Format::NCHW4)
    .value("NCHW8", AdaptivePooling::Format::NCHW8)
    .value("NCHW32", AdaptivePooling::Format::NCHW32)
    .value("NCHW88", AdaptivePooling::Format::NCHW88)
    .value("NCHW44", AdaptivePooling::Format::NCHW44)
    .value("NCHW44_DOT", AdaptivePooling::Format::NCHW44_DOT)
    .value("NCHW4_NCHW32", AdaptivePooling::Format::NCHW4_NCHW32)
    .value("NCHW32_NCHW4", AdaptivePooling::Format::NCHW32_NCHW4)
    .value("NCHW4_NCHW", AdaptivePooling::Format::NCHW4_NCHW)
    .value("NHWC_NCHW", AdaptivePooling::Format::NHWC_NCHW)
    .value("NHWC_NCHW4_IC_SMALL", AdaptivePooling::Format::NHWC_NCHW4_IC_SMALL)
    .value("NCHW_NCHW4_IC_SMALL", AdaptivePooling::Format::NCHW_NCHW4_IC_SMALL)
    .value("CHWN4", AdaptivePooling::Format::CHWN4)
    .value("NCHW64", AdaptivePooling::Format::NCHW64)
    .value("NCHW4_NHWC", AdaptivePooling::Format::NCHW4_NHWC)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "NCHW") return AdaptivePooling::Format::NCHW;
        if (str == "NHWC") return AdaptivePooling::Format::NHWC;
        if (str == "NHWCD4") return AdaptivePooling::Format::NHWCD4;
        if (str == "NCHW4") return AdaptivePooling::Format::NCHW4;
        if (str == "NCHW8") return AdaptivePooling::Format::NCHW8;
        if (str == "NCHW32") return AdaptivePooling::Format::NCHW32;
        if (str == "NCHW88") return AdaptivePooling::Format::NCHW88;
        if (str == "NCHW44") return AdaptivePooling::Format::NCHW44;
        if (str == "NCHW44_DOT") return AdaptivePooling::Format::NCHW44_DOT;
        if (str == "NCHW4_NCHW32") return AdaptivePooling::Format::NCHW4_NCHW32;
        if (str == "NCHW32_NCHW4") return AdaptivePooling::Format::NCHW32_NCHW4;
        if (str == "NCHW4_NCHW") return AdaptivePooling::Format::NCHW4_NCHW;
        if (str == "NHWC_NCHW") return AdaptivePooling::Format::NHWC_NCHW;
        if (str == "NHWC_NCHW4_IC_SMALL") return AdaptivePooling::Format::NHWC_NCHW4_IC_SMALL;
        if (str == "NCHW_NCHW4_IC_SMALL") return AdaptivePooling::Format::NCHW_NCHW4_IC_SMALL;
        if (str == "CHWN4") return AdaptivePooling::Format::CHWN4;
        if (str == "NCHW64") return AdaptivePooling::Format::NCHW64;
        if (str == "NCHW4_NHWC") return AdaptivePooling::Format::NCHW4_NHWC;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, AdaptivePooling::Format>();

AdaptivePoolingInst
    .def(py::init<::megdnn::param::AdaptivePooling::Mode, ::megdnn::param::AdaptivePooling::Format, std::vector<int32_t>, std::string>(), py::arg("mode") = ::megdnn::param::AdaptivePooling::Mode::MAX, py::arg("format") = ::megdnn::param::AdaptivePooling::Format::NCHW, py::arg("shape"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("mode", &AdaptivePooling::mode)
    .def_readwrite("format", &AdaptivePooling::format)
    .def_readwrite("shape", &AdaptivePooling::shape);

py::class_<AddAxis, std::shared_ptr<AddAxis>, OpDef> AddAxisInst(m, "AddAxis");

AddAxisInst
    .def(py::init<std::vector<int32_t>, std::string>(), py::arg("axis"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("axis", &AddAxis::axis);

py::class_<Argmax, std::shared_ptr<Argmax>, OpDef> ArgmaxInst(m, "Argmax");

ArgmaxInst
    .def(py::init<int32_t, std::string>(), py::arg("axis") = 0, py::arg("scope") = {})
    .def_readwrite("axis", &Argmax::axis);

py::class_<Argmin, std::shared_ptr<Argmin>, OpDef> ArgminInst(m, "Argmin");

ArgminInst
    .def(py::init<int32_t, std::string>(), py::arg("axis") = 0, py::arg("scope") = {})
    .def_readwrite("axis", &Argmin::axis);

py::class_<Argsort, std::shared_ptr<Argsort>, OpDef> ArgsortInst(m, "Argsort");

py::enum_<Argsort::Order>(ArgsortInst, "Order")
    .value("ASCENDING", Argsort::Order::ASCENDING)
    .value("DESCENDING", Argsort::Order::DESCENDING)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "ASCENDING") return Argsort::Order::ASCENDING;
        if (str == "DESCENDING") return Argsort::Order::DESCENDING;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, Argsort::Order>();

ArgsortInst
    .def(py::init<::megdnn::param::Argsort::Order, std::string>(), py::arg("order") = ::megdnn::param::Argsort::Order::ASCENDING, py::arg("scope") = {})
    .def_readwrite("order", &Argsort::order);

py::class_<AssertEqual, std::shared_ptr<AssertEqual>, OpDef> AssertEqualInst(m, "AssertEqual");

AssertEqualInst
    .def(py::init<float, bool, std::string>(), py::arg("maxerr") = 0.0001, py::arg("verbose") = false, py::arg("scope") = {})
    .def_readwrite("maxerr", &AssertEqual::maxerr)
    .def_readwrite("verbose", &AssertEqual::verbose);

py::class_<AtlasRuntime, std::shared_ptr<AtlasRuntime>, OpDef> AtlasRuntimeInst(m, "AtlasRuntime");

AtlasRuntimeInst
    .def(py::init<std::string, size_t, std::string>(), py::arg("buf"), py::arg("buf_size"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("buf", &AtlasRuntime::buf)
    .def_readwrite("buf_size", &AtlasRuntime::buf_size);

py::class_<Barrier, std::shared_ptr<Barrier>, OpDef> BarrierInst(m, "Barrier");

BarrierInst
    .def(py::init<::mgb::CompNode, uint32_t, std::string>(), py::arg("comp_node"), py::arg("nr_outputs"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("comp_node", &Barrier::comp_node)
    .def_readwrite("nr_outputs", &Barrier::nr_outputs);

py::class_<BatchConvBias, std::shared_ptr<BatchConvBias>, OpDef> BatchConvBiasInst(m, "BatchConvBias");

py::enum_<BatchConvBias::NonlineMode>(BatchConvBiasInst, "NonlineMode")
    .value("IDENTITY", BatchConvBias::NonlineMode::IDENTITY)
    .value("RELU", BatchConvBias::NonlineMode::RELU)
    .value("SIGMOID", BatchConvBias::NonlineMode::SIGMOID)
    .value("H_SWISH", BatchConvBias::NonlineMode::H_SWISH)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "IDENTITY") return BatchConvBias::NonlineMode::IDENTITY;
        if (str == "RELU") return BatchConvBias::NonlineMode::RELU;
        if (str == "SIGMOID") return BatchConvBias::NonlineMode::SIGMOID;
        if (str == "H_SWISH") return BatchConvBias::NonlineMode::H_SWISH;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, BatchConvBias::NonlineMode>();

py::enum_<BatchConvBias::Mode>(BatchConvBiasInst, "Mode")
    .value("CROSS_CORRELATION", BatchConvBias::Mode::CROSS_CORRELATION)
    .value("CONVOLUTION", BatchConvBias::Mode::CONVOLUTION)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "CROSS_CORRELATION") return BatchConvBias::Mode::CROSS_CORRELATION;
        if (str == "CONVOLUTION") return BatchConvBias::Mode::CONVOLUTION;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, BatchConvBias::Mode>();

py::enum_<BatchConvBias::Sparse>(BatchConvBiasInst, "Sparse")
    .value("DENSE", BatchConvBias::Sparse::DENSE)
    .value("GROUP", BatchConvBias::Sparse::GROUP)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "DENSE") return BatchConvBias::Sparse::DENSE;
        if (str == "GROUP") return BatchConvBias::Sparse::GROUP;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, BatchConvBias::Sparse>();

BatchConvBiasInst.attr("Format") = AdaptivePoolingInst.attr("Format");

py::enum_<BatchConvBias::ComputeMode>(BatchConvBiasInst, "ComputeMode")
    .value("DEFAULT", BatchConvBias::ComputeMode::DEFAULT)
    .value("FLOAT32", BatchConvBias::ComputeMode::FLOAT32)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "DEFAULT") return BatchConvBias::ComputeMode::DEFAULT;
        if (str == "FLOAT32") return BatchConvBias::ComputeMode::FLOAT32;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, BatchConvBias::ComputeMode>();

py::enum_<BatchConvBias::Strategy>(BatchConvBiasInst, "Strategy")
    .value("HEURISTIC", BatchConvBias::Strategy::HEURISTIC)
    .value("PROFILE", BatchConvBias::Strategy::PROFILE)
    .value("REPRODUCIBLE", BatchConvBias::Strategy::REPRODUCIBLE)
    .value("OPTIMIZED", BatchConvBias::Strategy::OPTIMIZED)
    .def("__or__", [](BatchConvBias::Strategy s0, BatchConvBias::Strategy s1) { 
         return static_cast<BatchConvBias::Strategy>(uint32_t(s0) | uint32_t(s1));
      })
    .def("__and__", [](BatchConvBias::Strategy s0, BatchConvBias::Strategy s1) {
         return static_cast<BatchConvBias::Strategy>(uint32_t(s0) & uint32_t(s1));
    })
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "HEURISTIC") return BatchConvBias::Strategy::HEURISTIC;
        if (str == "PROFILE") return BatchConvBias::Strategy::PROFILE;
        if (str == "REPRODUCIBLE") return BatchConvBias::Strategy::REPRODUCIBLE;
        if (str == "OPTIMIZED") return BatchConvBias::Strategy::OPTIMIZED;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, BatchConvBias::Strategy>();

BatchConvBiasInst
    .def(py::init<::megdnn::param::BatchConvBias::NonlineMode, ::megdnn::param::BatchConvBias::Mode, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, ::megdnn::param::BatchConvBias::Sparse, ::megdnn::param::BatchConvBias::Format, ::megdnn::param::BatchConvBias::ComputeMode, ::megdnn::param::ExecutionPolicy::Strategy, uint64_t, ::megdnn::DType, std::string>(), py::arg("nonlineMode") = ::megdnn::param::BatchConvBias::NonlineMode::IDENTITY, py::arg("mode") = ::megdnn::param::BatchConvBias::Mode::CROSS_CORRELATION, py::arg("pad_h") = 0, py::arg("pad_w") = 0, py::arg("stride_h") = 1, py::arg("stride_w") = 1, py::arg("dilate_h") = 1, py::arg("dilate_w") = 1, py::arg("sparse") = ::megdnn::param::BatchConvBias::Sparse::DENSE, py::arg("format") = ::megdnn::param::BatchConvBias::Format::NCHW, py::arg("compute_mode") = ::megdnn::param::BatchConvBias::ComputeMode::DEFAULT, py::arg("strategy") = static_cast<::megdnn::param::ExecutionPolicy::Strategy>(1), py::arg("workspace_limit") = 18446744073709551615ull, py::arg("dtype"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("nonlineMode", &BatchConvBias::nonlineMode)
    .def_readwrite("mode", &BatchConvBias::mode)
    .def_readwrite("pad_h", &BatchConvBias::pad_h)
    .def_readwrite("pad_w", &BatchConvBias::pad_w)
    .def_readwrite("stride_h", &BatchConvBias::stride_h)
    .def_readwrite("stride_w", &BatchConvBias::stride_w)
    .def_readwrite("dilate_h", &BatchConvBias::dilate_h)
    .def_readwrite("dilate_w", &BatchConvBias::dilate_w)
    .def_readwrite("sparse", &BatchConvBias::sparse)
    .def_readwrite("format", &BatchConvBias::format)
    .def_readwrite("compute_mode", &BatchConvBias::compute_mode)
    .def_readwrite("strategy", &BatchConvBias::strategy)
    .def_readwrite("workspace_limit", &BatchConvBias::workspace_limit)
    .def_readwrite("dtype", &BatchConvBias::dtype);

py::class_<BatchNorm, std::shared_ptr<BatchNorm>, OpDef> BatchNormInst(m, "BatchNorm");

py::enum_<BatchNorm::ParamDim>(BatchNormInst, "ParamDim")
    .value("DIM_11HW", BatchNorm::ParamDim::DIM_11HW)
    .value("DIM_1CHW", BatchNorm::ParamDim::DIM_1CHW)
    .value("DIM_1C11", BatchNorm::ParamDim::DIM_1C11)
    .value("DIM_111C", BatchNorm::ParamDim::DIM_111C)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "DIM_11HW") return BatchNorm::ParamDim::DIM_11HW;
        if (str == "DIM_1CHW") return BatchNorm::ParamDim::DIM_1CHW;
        if (str == "DIM_1C11") return BatchNorm::ParamDim::DIM_1C11;
        if (str == "DIM_111C") return BatchNorm::ParamDim::DIM_111C;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, BatchNorm::ParamDim>();

py::enum_<BatchNorm::FwdMode>(BatchNormInst, "FwdMode")
    .value("TRAINING", BatchNorm::FwdMode::TRAINING)
    .value("INFERENCE", BatchNorm::FwdMode::INFERENCE)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "TRAINING") return BatchNorm::FwdMode::TRAINING;
        if (str == "INFERENCE") return BatchNorm::FwdMode::INFERENCE;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, BatchNorm::FwdMode>();

BatchNormInst
    .def(py::init<::megdnn::param::BN::ParamDim, ::megdnn::param::BN::FwdMode, double, double, float, float, std::string>(), py::arg("param_dim") = ::megdnn::param::BN::ParamDim::DIM_11HW, py::arg("fwd_mode") = ::megdnn::param::BN::FwdMode::TRAINING, py::arg("epsilon") = 1e-4f, py::arg("avg_factor") = 1.f, py::arg("scale") = 1.f, py::arg("bias") = 0.f, py::arg("scope") = {})
    .def_readwrite("param_dim", &BatchNorm::param_dim)
    .def_readwrite("fwd_mode", &BatchNorm::fwd_mode)
    .def_readwrite("epsilon", &BatchNorm::epsilon)
    .def_readwrite("avg_factor", &BatchNorm::avg_factor)
    .def_readwrite("scale", &BatchNorm::scale)
    .def_readwrite("bias", &BatchNorm::bias);

py::class_<BatchNormBackward, std::shared_ptr<BatchNormBackward>, OpDef> BatchNormBackwardInst(m, "BatchNormBackward");

BatchNormBackwardInst.attr("ParamDim") = BatchNormInst.attr("ParamDim");

BatchNormBackwardInst.attr("FwdMode") = BatchNormInst.attr("FwdMode");

BatchNormBackwardInst
    .def(py::init<::megdnn::param::BN::ParamDim, ::megdnn::param::BN::FwdMode, double, double, float, float, std::string>(), py::arg("param_dim") = ::megdnn::param::BN::ParamDim::DIM_11HW, py::arg("fwd_mode") = ::megdnn::param::BN::FwdMode::TRAINING, py::arg("epsilon") = 1e-4f, py::arg("avg_factor") = 1.f, py::arg("scale") = 1.f, py::arg("bias") = 0.f, py::arg("scope") = {})
    .def_readwrite("param_dim", &BatchNormBackward::param_dim)
    .def_readwrite("fwd_mode", &BatchNormBackward::fwd_mode)
    .def_readwrite("epsilon", &BatchNormBackward::epsilon)
    .def_readwrite("avg_factor", &BatchNormBackward::avg_factor)
    .def_readwrite("scale", &BatchNormBackward::scale)
    .def_readwrite("bias", &BatchNormBackward::bias);

py::class_<BatchedIncrMeshIndexing, std::shared_ptr<BatchedIncrMeshIndexing>, OpDef> BatchedIncrMeshIndexingInst(m, "BatchedIncrMeshIndexing");

BatchedIncrMeshIndexingInst
    .def(py::init<std::vector<std::tuple<int8_t, bool, bool, bool, bool>>, std::string>(), py::arg("items"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("items", &BatchedIncrMeshIndexing::items);

py::class_<BatchedMatrixMul, std::shared_ptr<BatchedMatrixMul>, OpDef> BatchedMatrixMulInst(m, "BatchedMatrixMul");

py::enum_<BatchedMatrixMul::ComputeMode>(BatchedMatrixMulInst, "ComputeMode")
    .value("DEFAULT", BatchedMatrixMul::ComputeMode::DEFAULT)
    .value("FLOAT32", BatchedMatrixMul::ComputeMode::FLOAT32)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "DEFAULT") return BatchedMatrixMul::ComputeMode::DEFAULT;
        if (str == "FLOAT32") return BatchedMatrixMul::ComputeMode::FLOAT32;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, BatchedMatrixMul::ComputeMode>();

py::enum_<BatchedMatrixMul::Format>(BatchedMatrixMulInst, "Format")
    .value("DEFAULT", BatchedMatrixMul::Format::DEFAULT)
    .value("MK4", BatchedMatrixMul::Format::MK4)
    .value("MK8", BatchedMatrixMul::Format::MK8)
    .value("MK4_DOT", BatchedMatrixMul::Format::MK4_DOT)
    .value("N32K4_DOT", BatchedMatrixMul::Format::N32K4_DOT)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "DEFAULT") return BatchedMatrixMul::Format::DEFAULT;
        if (str == "MK4") return BatchedMatrixMul::Format::MK4;
        if (str == "MK8") return BatchedMatrixMul::Format::MK8;
        if (str == "MK4_DOT") return BatchedMatrixMul::Format::MK4_DOT;
        if (str == "N32K4_DOT") return BatchedMatrixMul::Format::N32K4_DOT;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, BatchedMatrixMul::Format>();

BatchedMatrixMulInst.attr("Strategy") = BatchConvBiasInst.attr("Strategy");

BatchedMatrixMulInst
    .def(py::init<bool, bool, ::megdnn::param::MatrixMul::ComputeMode, ::megdnn::param::MatrixMul::Format, ::megdnn::param::ExecutionPolicy::Strategy, uint64_t, uint32_t, uint32_t, std::string>(), py::arg("transposeA") = false, py::arg("transposeB") = false, py::arg("compute_mode") = ::megdnn::param::MatrixMul::ComputeMode::DEFAULT, py::arg("format") = ::megdnn::param::MatrixMul::Format::DEFAULT, py::arg("strategy") = static_cast<::megdnn::param::ExecutionPolicy::Strategy>(1), py::arg("workspace_limit") = 18446744073709551615ull, py::arg("dimA"), py::arg("dimB"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("transposeA", &BatchedMatrixMul::transposeA)
    .def_readwrite("transposeB", &BatchedMatrixMul::transposeB)
    .def_readwrite("compute_mode", &BatchedMatrixMul::compute_mode)
    .def_readwrite("format", &BatchedMatrixMul::format)
    .def_readwrite("strategy", &BatchedMatrixMul::strategy)
    .def_readwrite("workspace_limit", &BatchedMatrixMul::workspace_limit)
    .def_readwrite("dimA", &BatchedMatrixMul::dimA)
    .def_readwrite("dimB", &BatchedMatrixMul::dimB);

py::class_<BatchedMeshIndexing, std::shared_ptr<BatchedMeshIndexing>, OpDef> BatchedMeshIndexingInst(m, "BatchedMeshIndexing");

BatchedMeshIndexingInst
    .def(py::init<std::vector<std::tuple<int8_t, bool, bool, bool, bool>>, std::string>(), py::arg("items"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("items", &BatchedMeshIndexing::items);

py::class_<BatchedSetMeshIndexing, std::shared_ptr<BatchedSetMeshIndexing>, OpDef> BatchedSetMeshIndexingInst(m, "BatchedSetMeshIndexing");

BatchedSetMeshIndexingInst
    .def(py::init<std::vector<std::tuple<int8_t, bool, bool, bool, bool>>, std::string>(), py::arg("items"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("items", &BatchedSetMeshIndexing::items);

py::class_<BetaRNG, std::shared_ptr<BetaRNG>, OpDef> BetaRNGInst(m, "BetaRNG");

BetaRNGInst
    .def(py::init<uint64_t, size_t, std::string>(), py::arg("seed") = 0, py::arg("handle"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("seed", &BetaRNG::seed)
    .def_readwrite("handle", &BetaRNG::handle);

py::class_<Borrow, std::shared_ptr<Borrow>, OpDef> BorrowInst(m, "Borrow");

BorrowInst
    .def(py::init<::mgb::CompNode, std::string>(), py::arg("comp_node"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("comp_node", &Borrow::comp_node);

py::class_<Broadcast, std::shared_ptr<Broadcast>, OpDef> BroadcastInst(m, "Broadcast");

BroadcastInst
    .def(py::init<std::vector<int32_t>, std::string>(), py::arg("shape"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("shape", &Broadcast::shape);

py::class_<CambriconRuntime, std::shared_ptr<CambriconRuntime>, OpDef> CambriconRuntimeInst(m, "CambriconRuntime");

CambriconRuntimeInst
    .def(py::init<std::string, size_t, std::string, bool, std::string>(), py::arg("buf"), py::arg("buf_size"), py::arg("symbol"), py::arg("tensor_dim_mutable"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("buf", &CambriconRuntime::buf)
    .def_readwrite("buf_size", &CambriconRuntime::buf_size)
    .def_readwrite("symbol", &CambriconRuntime::symbol)
    .def_readwrite("tensor_dim_mutable", &CambriconRuntime::tensor_dim_mutable);

py::class_<CheckNonFinite, std::shared_ptr<CheckNonFinite>, OpDef> CheckNonFiniteInst(m, "CheckNonFinite");

CheckNonFiniteInst
    .def(py::init<float, std::string>(), py::arg("scale") = 1.0, py::arg("scope") = {})
    .def_readwrite("scale", &CheckNonFinite::scale);

py::class_<CollectiveComm, std::shared_ptr<CollectiveComm>, OpDef> CollectiveCommInst(m, "CollectiveComm");

py::enum_<CollectiveComm::Mode>(CollectiveCommInst, "Mode")
    .value("REDUCE_SUM", CollectiveComm::Mode::REDUCE_SUM)
    .value("BROADCAST", CollectiveComm::Mode::BROADCAST)
    .value("ALL_GATHER", CollectiveComm::Mode::ALL_GATHER)
    .value("REDUCE_SCATTER_SUM", CollectiveComm::Mode::REDUCE_SCATTER_SUM)
    .value("ALL_REDUCE_SUM", CollectiveComm::Mode::ALL_REDUCE_SUM)
    .value("ALL_REDUCE_MAX", CollectiveComm::Mode::ALL_REDUCE_MAX)
    .value("ALL_REDUCE_MIN", CollectiveComm::Mode::ALL_REDUCE_MIN)
    .value("ALL_REDUCE_PROD", CollectiveComm::Mode::ALL_REDUCE_PROD)
    .value("GATHER", CollectiveComm::Mode::GATHER)
    .value("SCATTER", CollectiveComm::Mode::SCATTER)
    .value("ALL_TO_ALL", CollectiveComm::Mode::ALL_TO_ALL)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "REDUCE_SUM") return CollectiveComm::Mode::REDUCE_SUM;
        if (str == "BROADCAST") return CollectiveComm::Mode::BROADCAST;
        if (str == "ALL_GATHER") return CollectiveComm::Mode::ALL_GATHER;
        if (str == "REDUCE_SCATTER_SUM") return CollectiveComm::Mode::REDUCE_SCATTER_SUM;
        if (str == "ALL_REDUCE_SUM") return CollectiveComm::Mode::ALL_REDUCE_SUM;
        if (str == "ALL_REDUCE_MAX") return CollectiveComm::Mode::ALL_REDUCE_MAX;
        if (str == "ALL_REDUCE_MIN") return CollectiveComm::Mode::ALL_REDUCE_MIN;
        if (str == "ALL_REDUCE_PROD") return CollectiveComm::Mode::ALL_REDUCE_PROD;
        if (str == "GATHER") return CollectiveComm::Mode::GATHER;
        if (str == "SCATTER") return CollectiveComm::Mode::SCATTER;
        if (str == "ALL_TO_ALL") return CollectiveComm::Mode::ALL_TO_ALL;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, CollectiveComm::Mode>();

CollectiveCommInst
    .def(py::init<::megdnn::param::CollectiveComm::Mode, std::string, uint32_t, uint32_t, bool, bool, std::string, uint32_t, ::megdnn::DType, std::string, std::string, std::string>(), py::arg("mode") = ::megdnn::param::CollectiveComm::Mode::REDUCE_SUM, py::arg("key"), py::arg("nr_devices"), py::arg("rank"), py::arg("is_root"), py::arg("local_grad"), py::arg("addr"), py::arg("port"), py::arg("dtype"), py::arg("backend"), py::arg("comp_node"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("mode", &CollectiveComm::mode)
    .def_readwrite("key", &CollectiveComm::key)
    .def_readwrite("nr_devices", &CollectiveComm::nr_devices)
    .def_readwrite("rank", &CollectiveComm::rank)
    .def_readwrite("is_root", &CollectiveComm::is_root)
    .def_readwrite("local_grad", &CollectiveComm::local_grad)
    .def_readwrite("addr", &CollectiveComm::addr)
    .def_readwrite("port", &CollectiveComm::port)
    .def_readwrite("dtype", &CollectiveComm::dtype)
    .def_readwrite("backend", &CollectiveComm::backend)
    .def_readwrite("comp_node", &CollectiveComm::comp_node);

py::class_<Concat, std::shared_ptr<Concat>, OpDef> ConcatInst(m, "Concat");

ConcatInst
    .def(py::init<int32_t, ::mgb::CompNode, std::string>(), py::arg("axis") = 0, py::arg("comp_node"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("axis", &Concat::axis)
    .def_readwrite("comp_node", &Concat::comp_node);

py::class_<CondTake, std::shared_ptr<CondTake>, OpDef> CondTakeInst(m, "CondTake");

CondTakeInst
    .def(py::init<>());

py::class_<ConvBias, std::shared_ptr<ConvBias>, OpDef> ConvBiasInst(m, "ConvBias");

ConvBiasInst.attr("NonlineMode") = BatchConvBiasInst.attr("NonlineMode");

ConvBiasInst.attr("Mode") = BatchConvBiasInst.attr("Mode");

ConvBiasInst.attr("Sparse") = BatchConvBiasInst.attr("Sparse");

ConvBiasInst.attr("Format") = AdaptivePoolingInst.attr("Format");

ConvBiasInst.attr("ComputeMode") = BatchConvBiasInst.attr("ComputeMode");

ConvBiasInst.attr("Strategy") = BatchConvBiasInst.attr("Strategy");

ConvBiasInst
    .def(py::init<::megdnn::param::ConvBias::NonlineMode, ::megdnn::param::ConvBias::Mode, ::megdnn::param::ConvBias::Sparse, ::megdnn::param::ConvBias::Format, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, ::megdnn::param::ConvBias::ComputeMode, ::megdnn::param::ExecutionPolicy::Strategy, uint64_t, ::megdnn::DType, std::string>(), py::arg("nonlineMode") = ::megdnn::param::ConvBias::NonlineMode::IDENTITY, py::arg("mode") = ::megdnn::param::ConvBias::Mode::CROSS_CORRELATION, py::arg("sparse") = ::megdnn::param::ConvBias::Sparse::DENSE, py::arg("format") = ::megdnn::param::ConvBias::Format::NCHW, py::arg("pad_h") = 0, py::arg("pad_w") = 0, py::arg("stride_h") = 1, py::arg("stride_w") = 1, py::arg("dilate_h") = 1, py::arg("dilate_w") = 1, py::arg("compute_mode") = ::megdnn::param::ConvBias::ComputeMode::DEFAULT, py::arg("strategy") = static_cast<::megdnn::param::ExecutionPolicy::Strategy>(1), py::arg("workspace_limit") = 18446744073709551615ull, py::arg("dtype"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("nonlineMode", &ConvBias::nonlineMode)
    .def_readwrite("mode", &ConvBias::mode)
    .def_readwrite("sparse", &ConvBias::sparse)
    .def_readwrite("format", &ConvBias::format)
    .def_readwrite("pad_h", &ConvBias::pad_h)
    .def_readwrite("pad_w", &ConvBias::pad_w)
    .def_readwrite("stride_h", &ConvBias::stride_h)
    .def_readwrite("stride_w", &ConvBias::stride_w)
    .def_readwrite("dilate_h", &ConvBias::dilate_h)
    .def_readwrite("dilate_w", &ConvBias::dilate_w)
    .def_readwrite("compute_mode", &ConvBias::compute_mode)
    .def_readwrite("strategy", &ConvBias::strategy)
    .def_readwrite("workspace_limit", &ConvBias::workspace_limit)
    .def_readwrite("dtype", &ConvBias::dtype);

py::class_<Convolution, std::shared_ptr<Convolution>, OpDef> ConvolutionInst(m, "Convolution");

ConvolutionInst.attr("Mode") = BatchConvBiasInst.attr("Mode");

ConvolutionInst.attr("Sparse") = BatchConvBiasInst.attr("Sparse");

ConvolutionInst.attr("Format") = AdaptivePoolingInst.attr("Format");

ConvolutionInst.attr("ComputeMode") = BatchConvBiasInst.attr("ComputeMode");

ConvolutionInst.attr("Strategy") = BatchConvBiasInst.attr("Strategy");

ConvolutionInst
    .def(py::init<::megdnn::param::Convolution::Mode, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, ::megdnn::param::Convolution::Sparse, ::megdnn::param::Convolution::Format, ::megdnn::param::Convolution::ComputeMode, ::megdnn::param::ExecutionPolicy::Strategy, uint64_t, std::string>(), py::arg("mode") = ::megdnn::param::Convolution::Mode::CROSS_CORRELATION, py::arg("pad_h") = 0, py::arg("pad_w") = 0, py::arg("stride_h") = 1, py::arg("stride_w") = 1, py::arg("dilate_h") = 1, py::arg("dilate_w") = 1, py::arg("sparse") = ::megdnn::param::Convolution::Sparse::DENSE, py::arg("format") = ::megdnn::param::Convolution::Format::NCHW, py::arg("compute_mode") = ::megdnn::param::Convolution::ComputeMode::DEFAULT, py::arg("strategy") = static_cast<::megdnn::param::ExecutionPolicy::Strategy>(1), py::arg("workspace_limit") = 18446744073709551615ull, py::arg("scope") = {})
    .def_readwrite("mode", &Convolution::mode)
    .def_readwrite("pad_h", &Convolution::pad_h)
    .def_readwrite("pad_w", &Convolution::pad_w)
    .def_readwrite("stride_h", &Convolution::stride_h)
    .def_readwrite("stride_w", &Convolution::stride_w)
    .def_readwrite("dilate_h", &Convolution::dilate_h)
    .def_readwrite("dilate_w", &Convolution::dilate_w)
    .def_readwrite("sparse", &Convolution::sparse)
    .def_readwrite("format", &Convolution::format)
    .def_readwrite("compute_mode", &Convolution::compute_mode)
    .def_readwrite("strategy", &Convolution::strategy)
    .def_readwrite("workspace_limit", &Convolution::workspace_limit);

py::class_<Convolution3D, std::shared_ptr<Convolution3D>, OpDef> Convolution3DInst(m, "Convolution3D");

py::enum_<Convolution3D::Mode>(Convolution3DInst, "Mode")
    .value("CROSS_CORRELATION", Convolution3D::Mode::CROSS_CORRELATION)
    .value("CONVOLUTION", Convolution3D::Mode::CONVOLUTION)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "CROSS_CORRELATION") return Convolution3D::Mode::CROSS_CORRELATION;
        if (str == "CONVOLUTION") return Convolution3D::Mode::CONVOLUTION;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, Convolution3D::Mode>();

py::enum_<Convolution3D::Sparse>(Convolution3DInst, "Sparse")
    .value("DENSE", Convolution3D::Sparse::DENSE)
    .value("GROUP", Convolution3D::Sparse::GROUP)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "DENSE") return Convolution3D::Sparse::DENSE;
        if (str == "GROUP") return Convolution3D::Sparse::GROUP;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, Convolution3D::Sparse>();

py::enum_<Convolution3D::DataType>(Convolution3DInst, "DataType")
    .value("FLOAT", Convolution3D::DataType::FLOAT)
    .value("FLOAT_IO16xC32", Convolution3D::DataType::FLOAT_IO16xC32)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "FLOAT") return Convolution3D::DataType::FLOAT;
        if (str == "FLOAT_IO16xC32") return Convolution3D::DataType::FLOAT_IO16xC32;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, Convolution3D::DataType>();

py::enum_<Convolution3D::Format>(Convolution3DInst, "Format")
    .value("NCDHW", Convolution3D::Format::NCDHW)
    .value("NDHWC", Convolution3D::Format::NDHWC)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "NCDHW") return Convolution3D::Format::NCDHW;
        if (str == "NDHWC") return Convolution3D::Format::NDHWC;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, Convolution3D::Format>();

Convolution3DInst.attr("Strategy") = BatchConvBiasInst.attr("Strategy");

Convolution3DInst
    .def(py::init<::megdnn::param::Convolution3D::Mode, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, ::megdnn::param::Convolution3D::Sparse, ::megdnn::param::Convolution3D::DataType, ::megdnn::param::Convolution3D::Format, ::megdnn::param::ExecutionPolicy::Strategy, uint64_t, std::string>(), py::arg("mode") = ::megdnn::param::Convolution3D::Mode::CROSS_CORRELATION, py::arg("pad_d") = 0, py::arg("pad_h") = 0, py::arg("pad_w") = 0, py::arg("stride_d") = 1, py::arg("stride_h") = 1, py::arg("stride_w") = 1, py::arg("dilate_d") = 1, py::arg("dilate_h") = 1, py::arg("dilate_w") = 1, py::arg("sparse") = ::megdnn::param::Convolution3D::Sparse::DENSE, py::arg("data_type") = ::megdnn::param::Convolution3D::DataType::FLOAT, py::arg("format") = ::megdnn::param::Convolution3D::Format::NCDHW, py::arg("strategy") = static_cast<::megdnn::param::ExecutionPolicy::Strategy>(1), py::arg("workspace_limit") = 18446744073709551615ull, py::arg("scope") = {})
    .def_readwrite("mode", &Convolution3D::mode)
    .def_readwrite("pad_d", &Convolution3D::pad_d)
    .def_readwrite("pad_h", &Convolution3D::pad_h)
    .def_readwrite("pad_w", &Convolution3D::pad_w)
    .def_readwrite("stride_d", &Convolution3D::stride_d)
    .def_readwrite("stride_h", &Convolution3D::stride_h)
    .def_readwrite("stride_w", &Convolution3D::stride_w)
    .def_readwrite("dilate_d", &Convolution3D::dilate_d)
    .def_readwrite("dilate_h", &Convolution3D::dilate_h)
    .def_readwrite("dilate_w", &Convolution3D::dilate_w)
    .def_readwrite("sparse", &Convolution3D::sparse)
    .def_readwrite("data_type", &Convolution3D::data_type)
    .def_readwrite("format", &Convolution3D::format)
    .def_readwrite("strategy", &Convolution3D::strategy)
    .def_readwrite("workspace_limit", &Convolution3D::workspace_limit);

py::class_<Convolution3DBackwardData, std::shared_ptr<Convolution3DBackwardData>, OpDef> Convolution3DBackwardDataInst(m, "Convolution3DBackwardData");

Convolution3DBackwardDataInst.attr("Mode") = Convolution3DInst.attr("Mode");

Convolution3DBackwardDataInst.attr("Sparse") = Convolution3DInst.attr("Sparse");

Convolution3DBackwardDataInst.attr("DataType") = Convolution3DInst.attr("DataType");

Convolution3DBackwardDataInst.attr("Format") = Convolution3DInst.attr("Format");

Convolution3DBackwardDataInst.attr("Strategy") = BatchConvBiasInst.attr("Strategy");

Convolution3DBackwardDataInst
    .def(py::init<::megdnn::param::Convolution3D::Mode, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, ::megdnn::param::Convolution3D::Sparse, ::megdnn::param::Convolution3D::DataType, ::megdnn::param::Convolution3D::Format, ::megdnn::param::ExecutionPolicy::Strategy, uint64_t, std::string>(), py::arg("mode") = ::megdnn::param::Convolution3D::Mode::CROSS_CORRELATION, py::arg("pad_d") = 0, py::arg("pad_h") = 0, py::arg("pad_w") = 0, py::arg("stride_d") = 1, py::arg("stride_h") = 1, py::arg("stride_w") = 1, py::arg("dilate_d") = 1, py::arg("dilate_h") = 1, py::arg("dilate_w") = 1, py::arg("sparse") = ::megdnn::param::Convolution3D::Sparse::DENSE, py::arg("data_type") = ::megdnn::param::Convolution3D::DataType::FLOAT, py::arg("format") = ::megdnn::param::Convolution3D::Format::NCDHW, py::arg("strategy") = static_cast<::megdnn::param::ExecutionPolicy::Strategy>(1), py::arg("workspace_limit") = 18446744073709551615ull, py::arg("scope") = {})
    .def_readwrite("mode", &Convolution3DBackwardData::mode)
    .def_readwrite("pad_d", &Convolution3DBackwardData::pad_d)
    .def_readwrite("pad_h", &Convolution3DBackwardData::pad_h)
    .def_readwrite("pad_w", &Convolution3DBackwardData::pad_w)
    .def_readwrite("stride_d", &Convolution3DBackwardData::stride_d)
    .def_readwrite("stride_h", &Convolution3DBackwardData::stride_h)
    .def_readwrite("stride_w", &Convolution3DBackwardData::stride_w)
    .def_readwrite("dilate_d", &Convolution3DBackwardData::dilate_d)
    .def_readwrite("dilate_h", &Convolution3DBackwardData::dilate_h)
    .def_readwrite("dilate_w", &Convolution3DBackwardData::dilate_w)
    .def_readwrite("sparse", &Convolution3DBackwardData::sparse)
    .def_readwrite("data_type", &Convolution3DBackwardData::data_type)
    .def_readwrite("format", &Convolution3DBackwardData::format)
    .def_readwrite("strategy", &Convolution3DBackwardData::strategy)
    .def_readwrite("workspace_limit", &Convolution3DBackwardData::workspace_limit);

py::class_<ConvolutionBackwardData, std::shared_ptr<ConvolutionBackwardData>, OpDef> ConvolutionBackwardDataInst(m, "ConvolutionBackwardData");

ConvolutionBackwardDataInst.attr("Mode") = BatchConvBiasInst.attr("Mode");

ConvolutionBackwardDataInst.attr("Sparse") = BatchConvBiasInst.attr("Sparse");

ConvolutionBackwardDataInst.attr("Format") = AdaptivePoolingInst.attr("Format");

ConvolutionBackwardDataInst.attr("ComputeMode") = BatchConvBiasInst.attr("ComputeMode");

ConvolutionBackwardDataInst.attr("Strategy") = BatchConvBiasInst.attr("Strategy");

ConvolutionBackwardDataInst
    .def(py::init<::megdnn::param::Convolution::Mode, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, ::megdnn::param::Convolution::Sparse, ::megdnn::param::Convolution::Format, ::megdnn::param::Convolution::ComputeMode, ::megdnn::param::ExecutionPolicy::Strategy, uint64_t, ::megdnn::DType, std::string>(), py::arg("mode") = ::megdnn::param::Convolution::Mode::CROSS_CORRELATION, py::arg("pad_h") = 0, py::arg("pad_w") = 0, py::arg("stride_h") = 1, py::arg("stride_w") = 1, py::arg("dilate_h") = 1, py::arg("dilate_w") = 1, py::arg("sparse") = ::megdnn::param::Convolution::Sparse::DENSE, py::arg("format") = ::megdnn::param::Convolution::Format::NCHW, py::arg("compute_mode") = ::megdnn::param::Convolution::ComputeMode::DEFAULT, py::arg("strategy") = static_cast<::megdnn::param::ExecutionPolicy::Strategy>(1), py::arg("workspace_limit") = 18446744073709551615ull, py::arg("dtype"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("mode", &ConvolutionBackwardData::mode)
    .def_readwrite("pad_h", &ConvolutionBackwardData::pad_h)
    .def_readwrite("pad_w", &ConvolutionBackwardData::pad_w)
    .def_readwrite("stride_h", &ConvolutionBackwardData::stride_h)
    .def_readwrite("stride_w", &ConvolutionBackwardData::stride_w)
    .def_readwrite("dilate_h", &ConvolutionBackwardData::dilate_h)
    .def_readwrite("dilate_w", &ConvolutionBackwardData::dilate_w)
    .def_readwrite("sparse", &ConvolutionBackwardData::sparse)
    .def_readwrite("format", &ConvolutionBackwardData::format)
    .def_readwrite("compute_mode", &ConvolutionBackwardData::compute_mode)
    .def_readwrite("strategy", &ConvolutionBackwardData::strategy)
    .def_readwrite("workspace_limit", &ConvolutionBackwardData::workspace_limit)
    .def_readwrite("dtype", &ConvolutionBackwardData::dtype);

py::class_<Copy, std::shared_ptr<Copy>, OpDef> CopyInst(m, "Copy");

CopyInst
    .def(py::init<::mgb::CompNode, std::string>(), py::arg("comp_node"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("comp_node", &Copy::comp_node);

py::class_<Correlation, std::shared_ptr<Correlation>, OpDef> CorrelationInst(m, "Correlation");

py::enum_<Correlation::Format>(CorrelationInst, "Format")
    .value("NCHW", Correlation::Format::NCHW)
    .value("NHWC", Correlation::Format::NHWC)
    .value("NHWCD4", Correlation::Format::NHWCD4)
    .value("NCHW4", Correlation::Format::NCHW4)
    .value("NCHW8", Correlation::Format::NCHW8)
    .value("NCHW32", Correlation::Format::NCHW32)
    .value("NCHW88", Correlation::Format::NCHW88)
    .value("NCHW44", Correlation::Format::NCHW44)
    .value("NCHW44_DOT", Correlation::Format::NCHW44_DOT)
    .value("NCHW_WINOGRAD", Correlation::Format::NCHW_WINOGRAD)
    .value("NCHW88_WINOGRAD", Correlation::Format::NCHW88_WINOGRAD)
    .value("NCHW44_WINOGRAD", Correlation::Format::NCHW44_WINOGRAD)
    .value("NCHW4_NCHW32", Correlation::Format::NCHW4_NCHW32)
    .value("NCHW32_NCHW4", Correlation::Format::NCHW32_NCHW4)
    .value("NCHW4_NCHW", Correlation::Format::NCHW4_NCHW)
    .value("NHWC_NCHW", Correlation::Format::NHWC_NCHW)
    .value("NHWC_NCHW4_IC_SMALL", Correlation::Format::NHWC_NCHW4_IC_SMALL)
    .value("NCHW_NCHW4_IC_SMALL", Correlation::Format::NCHW_NCHW4_IC_SMALL)
    .value("CHWN4", Correlation::Format::CHWN4)
    .value("NCHW4_NHWC", Correlation::Format::NCHW4_NHWC)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "NCHW") return Correlation::Format::NCHW;
        if (str == "NHWC") return Correlation::Format::NHWC;
        if (str == "NHWCD4") return Correlation::Format::NHWCD4;
        if (str == "NCHW4") return Correlation::Format::NCHW4;
        if (str == "NCHW8") return Correlation::Format::NCHW8;
        if (str == "NCHW32") return Correlation::Format::NCHW32;
        if (str == "NCHW88") return Correlation::Format::NCHW88;
        if (str == "NCHW44") return Correlation::Format::NCHW44;
        if (str == "NCHW44_DOT") return Correlation::Format::NCHW44_DOT;
        if (str == "NCHW_WINOGRAD") return Correlation::Format::NCHW_WINOGRAD;
        if (str == "NCHW88_WINOGRAD") return Correlation::Format::NCHW88_WINOGRAD;
        if (str == "NCHW44_WINOGRAD") return Correlation::Format::NCHW44_WINOGRAD;
        if (str == "NCHW4_NCHW32") return Correlation::Format::NCHW4_NCHW32;
        if (str == "NCHW32_NCHW4") return Correlation::Format::NCHW32_NCHW4;
        if (str == "NCHW4_NCHW") return Correlation::Format::NCHW4_NCHW;
        if (str == "NHWC_NCHW") return Correlation::Format::NHWC_NCHW;
        if (str == "NHWC_NCHW4_IC_SMALL") return Correlation::Format::NHWC_NCHW4_IC_SMALL;
        if (str == "NCHW_NCHW4_IC_SMALL") return Correlation::Format::NCHW_NCHW4_IC_SMALL;
        if (str == "CHWN4") return Correlation::Format::CHWN4;
        if (str == "NCHW4_NHWC") return Correlation::Format::NCHW4_NHWC;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, Correlation::Format>();

CorrelationInst
    .def(py::init<::megdnn::param::Correlation::Format, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, bool, std::string>(), py::arg("format") = ::megdnn::param::Correlation::Format::NCHW, py::arg("kernel_size") = 1, py::arg("max_displacement") = 1, py::arg("stride1") = 1, py::arg("stride2") = 1, py::arg("pad_size") = 0, py::arg("is_multiply") = true, py::arg("scope") = {})
    .def_readwrite("format", &Correlation::format)
    .def_readwrite("kernel_size", &Correlation::kernel_size)
    .def_readwrite("max_displacement", &Correlation::max_displacement)
    .def_readwrite("stride1", &Correlation::stride1)
    .def_readwrite("stride2", &Correlation::stride2)
    .def_readwrite("pad_size", &Correlation::pad_size)
    .def_readwrite("is_multiply", &Correlation::is_multiply);

py::class_<Cross, std::shared_ptr<Cross>, OpDef> CrossInst(m, "Cross");

CrossInst
    .def(py::init<int32_t, int32_t, int32_t, std::string>(), py::arg("axisa") = -1, py::arg("axisb") = -1, py::arg("axisc") = -1, py::arg("scope") = {})
    .def_readwrite("axisa", &Cross::axisa)
    .def_readwrite("axisb", &Cross::axisb)
    .def_readwrite("axisc", &Cross::axisc);

py::class_<Cumsum, std::shared_ptr<Cumsum>, OpDef> CumsumInst(m, "Cumsum");

CumsumInst
    .def(py::init<int32_t, bool, bool, std::string>(), py::arg("axis") = 2147483647, py::arg("exclusive") = true, py::arg("reverse") = false, py::arg("scope") = {})
    .def_readwrite("axis", &Cumsum::axis)
    .def_readwrite("exclusive", &Cumsum::exclusive)
    .def_readwrite("reverse", &Cumsum::reverse);

py::class_<CvtColor, std::shared_ptr<CvtColor>, OpDef> CvtColorInst(m, "CvtColor");

py::enum_<CvtColor::Mode>(CvtColorInst, "Mode")
    .value("RGB2GRAY", CvtColor::Mode::RGB2GRAY)
    .value("RGB2YUV", CvtColor::Mode::RGB2YUV)
    .value("YUV2RGB", CvtColor::Mode::YUV2RGB)
    .value("GRAY2RGB", CvtColor::Mode::GRAY2RGB)
    .value("RGBA2RGB", CvtColor::Mode::RGBA2RGB)
    .value("RGBA2BGR", CvtColor::Mode::RGBA2BGR)
    .value("RGBA2GRAY", CvtColor::Mode::RGBA2GRAY)
    .value("RGB2BGR", CvtColor::Mode::RGB2BGR)
    .value("BGR2GRAY", CvtColor::Mode::BGR2GRAY)
    .value("BGR2RGB", CvtColor::Mode::BGR2RGB)
    .value("YUV2GRAY_NV21", CvtColor::Mode::YUV2GRAY_NV21)
    .value("YUV2RGB_NV21", CvtColor::Mode::YUV2RGB_NV21)
    .value("YUV2BGR_NV21", CvtColor::Mode::YUV2BGR_NV21)
    .value("YUV2GRAY_NV12", CvtColor::Mode::YUV2GRAY_NV12)
    .value("YUV2RGB_NV12", CvtColor::Mode::YUV2RGB_NV12)
    .value("YUV2BGR_NV12", CvtColor::Mode::YUV2BGR_NV12)
    .value("YUV2GRAY_YV12", CvtColor::Mode::YUV2GRAY_YV12)
    .value("YUV2RGB_YV12", CvtColor::Mode::YUV2RGB_YV12)
    .value("YUV2BGR_YV12", CvtColor::Mode::YUV2BGR_YV12)
    .value("YUV2GRAY_YU12", CvtColor::Mode::YUV2GRAY_YU12)
    .value("YUV2RGB_YU12", CvtColor::Mode::YUV2RGB_YU12)
    .value("YUV2BGR_YU12", CvtColor::Mode::YUV2BGR_YU12)
    .value("YCrCb2RGB", CvtColor::Mode::YCrCb2RGB)
    .value("YCrCb2BGR", CvtColor::Mode::YCrCb2BGR)
    .value("BT601_YUV2RGB_NV21", CvtColor::Mode::BT601_YUV2RGB_NV21)
    .value("BT601_YUV2BGR_NV21", CvtColor::Mode::BT601_YUV2BGR_NV21)
    .value("BT601_YUV2RGB_NV12", CvtColor::Mode::BT601_YUV2RGB_NV12)
    .value("BT601_YUV2BGR_NV12", CvtColor::Mode::BT601_YUV2BGR_NV12)
    .value("BT601_YUV2RGB_YV12", CvtColor::Mode::BT601_YUV2RGB_YV12)
    .value("BT601_YUV2BGR_YV12", CvtColor::Mode::BT601_YUV2BGR_YV12)
    .value("BT601_YUV2RGB_YU12", CvtColor::Mode::BT601_YUV2RGB_YU12)
    .value("BT601_YUV2BGR_YU12", CvtColor::Mode::BT601_YUV2BGR_YU12)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "RGB2GRAY") return CvtColor::Mode::RGB2GRAY;
        if (str == "RGB2YUV") return CvtColor::Mode::RGB2YUV;
        if (str == "YUV2RGB") return CvtColor::Mode::YUV2RGB;
        if (str == "GRAY2RGB") return CvtColor::Mode::GRAY2RGB;
        if (str == "RGBA2RGB") return CvtColor::Mode::RGBA2RGB;
        if (str == "RGBA2BGR") return CvtColor::Mode::RGBA2BGR;
        if (str == "RGBA2GRAY") return CvtColor::Mode::RGBA2GRAY;
        if (str == "RGB2BGR") return CvtColor::Mode::RGB2BGR;
        if (str == "BGR2GRAY") return CvtColor::Mode::BGR2GRAY;
        if (str == "BGR2RGB") return CvtColor::Mode::BGR2RGB;
        if (str == "YUV2GRAY_NV21") return CvtColor::Mode::YUV2GRAY_NV21;
        if (str == "YUV2RGB_NV21") return CvtColor::Mode::YUV2RGB_NV21;
        if (str == "YUV2BGR_NV21") return CvtColor::Mode::YUV2BGR_NV21;
        if (str == "YUV2GRAY_NV12") return CvtColor::Mode::YUV2GRAY_NV12;
        if (str == "YUV2RGB_NV12") return CvtColor::Mode::YUV2RGB_NV12;
        if (str == "YUV2BGR_NV12") return CvtColor::Mode::YUV2BGR_NV12;
        if (str == "YUV2GRAY_YV12") return CvtColor::Mode::YUV2GRAY_YV12;
        if (str == "YUV2RGB_YV12") return CvtColor::Mode::YUV2RGB_YV12;
        if (str == "YUV2BGR_YV12") return CvtColor::Mode::YUV2BGR_YV12;
        if (str == "YUV2GRAY_YU12") return CvtColor::Mode::YUV2GRAY_YU12;
        if (str == "YUV2RGB_YU12") return CvtColor::Mode::YUV2RGB_YU12;
        if (str == "YUV2BGR_YU12") return CvtColor::Mode::YUV2BGR_YU12;
        if (str == "YCrCb2RGB") return CvtColor::Mode::YCrCb2RGB;
        if (str == "YCrCb2BGR") return CvtColor::Mode::YCrCb2BGR;
        if (str == "BT601_YUV2RGB_NV21") return CvtColor::Mode::BT601_YUV2RGB_NV21;
        if (str == "BT601_YUV2BGR_NV21") return CvtColor::Mode::BT601_YUV2BGR_NV21;
        if (str == "BT601_YUV2RGB_NV12") return CvtColor::Mode::BT601_YUV2RGB_NV12;
        if (str == "BT601_YUV2BGR_NV12") return CvtColor::Mode::BT601_YUV2BGR_NV12;
        if (str == "BT601_YUV2RGB_YV12") return CvtColor::Mode::BT601_YUV2RGB_YV12;
        if (str == "BT601_YUV2BGR_YV12") return CvtColor::Mode::BT601_YUV2BGR_YV12;
        if (str == "BT601_YUV2RGB_YU12") return CvtColor::Mode::BT601_YUV2RGB_YU12;
        if (str == "BT601_YUV2BGR_YU12") return CvtColor::Mode::BT601_YUV2BGR_YU12;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, CvtColor::Mode>();

CvtColorInst
    .def(py::init<::megdnn::param::CvtColor::Mode, std::string>(), py::arg("mode") = ::megdnn::param::CvtColor::Mode::RGB2GRAY, py::arg("scope") = {})
    .def_readwrite("mode", &CvtColor::mode);

py::class_<DeformableConv, std::shared_ptr<DeformableConv>, OpDef> DeformableConvInst(m, "DeformableConv");

DeformableConvInst.attr("Mode") = BatchConvBiasInst.attr("Mode");

DeformableConvInst.attr("Sparse") = BatchConvBiasInst.attr("Sparse");

DeformableConvInst.attr("Format") = AdaptivePoolingInst.attr("Format");

DeformableConvInst.attr("ComputeMode") = BatchConvBiasInst.attr("ComputeMode");

DeformableConvInst.attr("Strategy") = BatchConvBiasInst.attr("Strategy");

DeformableConvInst
    .def(py::init<::megdnn::param::Convolution::Mode, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, ::megdnn::param::Convolution::Sparse, ::megdnn::param::Convolution::Format, ::megdnn::param::Convolution::ComputeMode, ::megdnn::param::ExecutionPolicy::Strategy, uint64_t, std::string>(), py::arg("mode") = ::megdnn::param::Convolution::Mode::CROSS_CORRELATION, py::arg("pad_h") = 0, py::arg("pad_w") = 0, py::arg("stride_h") = 1, py::arg("stride_w") = 1, py::arg("dilate_h") = 1, py::arg("dilate_w") = 1, py::arg("sparse") = ::megdnn::param::Convolution::Sparse::DENSE, py::arg("format") = ::megdnn::param::Convolution::Format::NCHW, py::arg("compute_mode") = ::megdnn::param::Convolution::ComputeMode::DEFAULT, py::arg("strategy") = static_cast<::megdnn::param::ExecutionPolicy::Strategy>(1), py::arg("workspace_limit") = 18446744073709551615ull, py::arg("scope") = {})
    .def_readwrite("mode", &DeformableConv::mode)
    .def_readwrite("pad_h", &DeformableConv::pad_h)
    .def_readwrite("pad_w", &DeformableConv::pad_w)
    .def_readwrite("stride_h", &DeformableConv::stride_h)
    .def_readwrite("stride_w", &DeformableConv::stride_w)
    .def_readwrite("dilate_h", &DeformableConv::dilate_h)
    .def_readwrite("dilate_w", &DeformableConv::dilate_w)
    .def_readwrite("sparse", &DeformableConv::sparse)
    .def_readwrite("format", &DeformableConv::format)
    .def_readwrite("compute_mode", &DeformableConv::compute_mode)
    .def_readwrite("strategy", &DeformableConv::strategy)
    .def_readwrite("workspace_limit", &DeformableConv::workspace_limit);

py::class_<DeformablePSROIPooling, std::shared_ptr<DeformablePSROIPooling>, OpDef> DeformablePSROIPoolingInst(m, "DeformablePSROIPooling");

DeformablePSROIPoolingInst
    .def(py::init<bool, float, float, uint32_t, uint32_t, uint32_t, uint32_t, std::string>(), py::arg("no_trans") = true, py::arg("spatial_scale") = 1, py::arg("trans_std") = 1, py::arg("pooled_h") = 1, py::arg("pooled_w") = 1, py::arg("part_size") = 1, py::arg("sample_per_part") = 1, py::arg("scope") = {})
    .def_readwrite("no_trans", &DeformablePSROIPooling::no_trans)
    .def_readwrite("spatial_scale", &DeformablePSROIPooling::spatial_scale)
    .def_readwrite("trans_std", &DeformablePSROIPooling::trans_std)
    .def_readwrite("pooled_h", &DeformablePSROIPooling::pooled_h)
    .def_readwrite("pooled_w", &DeformablePSROIPooling::pooled_w)
    .def_readwrite("part_size", &DeformablePSROIPooling::part_size)
    .def_readwrite("sample_per_part", &DeformablePSROIPooling::sample_per_part);

py::class_<Diag, std::shared_ptr<Diag>, OpDef> DiagInst(m, "Diag");

DiagInst
    .def(py::init<int32_t, std::string>(), py::arg("k") = 0, py::arg("scope") = {})
    .def_readwrite("k", &Diag::k);

py::class_<Dimshuffle, std::shared_ptr<Dimshuffle>, OpDef> DimshuffleInst(m, "Dimshuffle");

DimshuffleInst
    .def(py::init<std::vector<int32_t>, std::string>(), py::arg("pattern"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("pattern", &Dimshuffle::pattern);

py::class_<Dot, std::shared_ptr<Dot>, OpDef> DotInst(m, "Dot");

DotInst
    .def(py::init<>());

py::class_<Dropout, std::shared_ptr<Dropout>, OpDef> DropoutInst(m, "Dropout");

DropoutInst
    .def(py::init<float, uint64_t, size_t, std::string>(), py::arg("drop_prob") = 0, py::arg("seed") = 0, py::arg("handle"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("drop_prob", &Dropout::drop_prob)
    .def_readwrite("seed", &Dropout::seed)
    .def_readwrite("handle", &Dropout::handle);

py::class_<Elemwise, std::shared_ptr<Elemwise>, OpDef> ElemwiseInst(m, "Elemwise");

py::enum_<Elemwise::Mode>(ElemwiseInst, "Mode")
    .value("RELU", Elemwise::Mode::RELU)
    .value("ABS", Elemwise::Mode::ABS)
    .value("ACOS", Elemwise::Mode::ACOS)
    .value("ASIN", Elemwise::Mode::ASIN)
    .value("CEIL", Elemwise::Mode::CEIL)
    .value("COS", Elemwise::Mode::COS)
    .value("EXP", Elemwise::Mode::EXP)
    .value("EXPM1", Elemwise::Mode::EXPM1)
    .value("FLOOR", Elemwise::Mode::FLOOR)
    .value("LOG", Elemwise::Mode::LOG)
    .value("LOG1P", Elemwise::Mode::LOG1P)
    .value("NEGATE", Elemwise::Mode::NEGATE)
    .value("SIGMOID", Elemwise::Mode::SIGMOID)
    .value("SIN", Elemwise::Mode::SIN)
    .value("TANH", Elemwise::Mode::TANH)
    .value("ABS_GRAD", Elemwise::Mode::ABS_GRAD)
    .value("ADD", Elemwise::Mode::ADD)
    .value("FLOOR_DIV", Elemwise::Mode::FLOOR_DIV)
    .value("MAX", Elemwise::Mode::MAX)
    .value("MIN", Elemwise::Mode::MIN)
    .value("MOD", Elemwise::Mode::MOD)
    .value("MUL", Elemwise::Mode::MUL)
    .value("POW", Elemwise::Mode::POW)
    .value("SIGMOID_GRAD", Elemwise::Mode::SIGMOID_GRAD)
    .value("SUB", Elemwise::Mode::SUB)
    .value("SWITCH_GT0", Elemwise::Mode::SWITCH_GT0)
    .value("TANH_GRAD", Elemwise::Mode::TANH_GRAD)
    .value("TRUE_DIV", Elemwise::Mode::TRUE_DIV)
    .value("LOG_SUM_EXP", Elemwise::Mode::LOG_SUM_EXP)
    .value("LT", Elemwise::Mode::LT)
    .value("LEQ", Elemwise::Mode::LEQ)
    .value("EQ", Elemwise::Mode::EQ)
    .value("SHL", Elemwise::Mode::SHL)
    .value("SHR", Elemwise::Mode::SHR)
    .value("COND_LEQ_MOV", Elemwise::Mode::COND_LEQ_MOV)
    .value("FUSE_MUL_ADD3", Elemwise::Mode::FUSE_MUL_ADD3)
    .value("FUSE_MUL_ADD4", Elemwise::Mode::FUSE_MUL_ADD4)
    .value("FUSE_ADD_RELU", Elemwise::Mode::FUSE_ADD_RELU)
    .value("FUSE_ADD_SIGMOID", Elemwise::Mode::FUSE_ADD_SIGMOID)
    .value("FUSE_ADD_TANH", Elemwise::Mode::FUSE_ADD_TANH)
    .value("FAST_TANH", Elemwise::Mode::FAST_TANH)
    .value("FAST_TANH_GRAD", Elemwise::Mode::FAST_TANH_GRAD)
    .value("ROUND", Elemwise::Mode::ROUND)
    .value("RMULH", Elemwise::Mode::RMULH)
    .value("ATAN2", Elemwise::Mode::ATAN2)
    .value("ERF", Elemwise::Mode::ERF)
    .value("ERFINV", Elemwise::Mode::ERFINV)
    .value("ERFC", Elemwise::Mode::ERFC)
    .value("ERFCINV", Elemwise::Mode::ERFCINV)
    .value("H_SWISH", Elemwise::Mode::H_SWISH)
    .value("H_SWISH_GRAD", Elemwise::Mode::H_SWISH_GRAD)
    .value("FUSE_ADD_H_SWISH", Elemwise::Mode::FUSE_ADD_H_SWISH)
    .value("NOT", Elemwise::Mode::NOT)
    .value("AND", Elemwise::Mode::AND)
    .value("OR", Elemwise::Mode::OR)
    .value("XOR", Elemwise::Mode::XOR)
    .value("SILU", Elemwise::Mode::SILU)
    .value("SILU_GRAD", Elemwise::Mode::SILU_GRAD)
    .value("GELU", Elemwise::Mode::GELU)
    .value("GELU_GRAD", Elemwise::Mode::GELU_GRAD)
    .value("COND_LT_MOV", Elemwise::Mode::COND_LT_MOV)
    .value("NEQ", Elemwise::Mode::NEQ)
    .value("ISNAN", Elemwise::Mode::ISNAN)
    .value("ISINF", Elemwise::Mode::ISINF)
    .value("SINH", Elemwise::Mode::SINH)
    .value("COSH", Elemwise::Mode::COSH)
    .value("ASINH", Elemwise::Mode::ASINH)
    .value("ACOSH", Elemwise::Mode::ACOSH)
    .value("ATANH", Elemwise::Mode::ATANH)
    .value("TAN", Elemwise::Mode::TAN)
    .value("ASINH_GRAD", Elemwise::Mode::ASINH_GRAD)
    .value("ACOSH_GRAD", Elemwise::Mode::ACOSH_GRAD)
    .value("ATANH_GRAD", Elemwise::Mode::ATANH_GRAD)
    .value("PRELU", Elemwise::Mode::PRELU)
    .value("CLIP", Elemwise::Mode::CLIP)
    .value("PRELU_GRAD", Elemwise::Mode::PRELU_GRAD)
    .value("SOFTPLUS", Elemwise::Mode::SOFTPLUS)
    .value("SOFTPLUS_GRAD", Elemwise::Mode::SOFTPLUS_GRAD)
    .value("RELU6", Elemwise::Mode::RELU6)
    .value("RELU6_GRAD", Elemwise::Mode::RELU6_GRAD)
    .value("HSIGMOID", Elemwise::Mode::HSIGMOID)
    .value("HSIGMOID_GRAD", Elemwise::Mode::HSIGMOID_GRAD)
    .value("LOGSIGMOID", Elemwise::Mode::LOGSIGMOID)
    .value("SQRT", Elemwise::Mode::SQRT)
    .value("SQUARE", Elemwise::Mode::SQUARE)
    .value("SIGN", Elemwise::Mode::SIGN)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "RELU") return Elemwise::Mode::RELU;
        if (str == "ABS") return Elemwise::Mode::ABS;
        if (str == "ACOS") return Elemwise::Mode::ACOS;
        if (str == "ASIN") return Elemwise::Mode::ASIN;
        if (str == "CEIL") return Elemwise::Mode::CEIL;
        if (str == "COS") return Elemwise::Mode::COS;
        if (str == "EXP") return Elemwise::Mode::EXP;
        if (str == "EXPM1") return Elemwise::Mode::EXPM1;
        if (str == "FLOOR") return Elemwise::Mode::FLOOR;
        if (str == "LOG") return Elemwise::Mode::LOG;
        if (str == "LOG1P") return Elemwise::Mode::LOG1P;
        if (str == "NEGATE") return Elemwise::Mode::NEGATE;
        if (str == "SIGMOID") return Elemwise::Mode::SIGMOID;
        if (str == "SIN") return Elemwise::Mode::SIN;
        if (str == "TANH") return Elemwise::Mode::TANH;
        if (str == "ABS_GRAD") return Elemwise::Mode::ABS_GRAD;
        if (str == "ADD") return Elemwise::Mode::ADD;
        if (str == "FLOOR_DIV") return Elemwise::Mode::FLOOR_DIV;
        if (str == "MAX") return Elemwise::Mode::MAX;
        if (str == "MIN") return Elemwise::Mode::MIN;
        if (str == "MOD") return Elemwise::Mode::MOD;
        if (str == "MUL") return Elemwise::Mode::MUL;
        if (str == "POW") return Elemwise::Mode::POW;
        if (str == "SIGMOID_GRAD") return Elemwise::Mode::SIGMOID_GRAD;
        if (str == "SUB") return Elemwise::Mode::SUB;
        if (str == "SWITCH_GT0") return Elemwise::Mode::SWITCH_GT0;
        if (str == "TANH_GRAD") return Elemwise::Mode::TANH_GRAD;
        if (str == "TRUE_DIV") return Elemwise::Mode::TRUE_DIV;
        if (str == "LOG_SUM_EXP") return Elemwise::Mode::LOG_SUM_EXP;
        if (str == "LT") return Elemwise::Mode::LT;
        if (str == "LEQ") return Elemwise::Mode::LEQ;
        if (str == "EQ") return Elemwise::Mode::EQ;
        if (str == "SHL") return Elemwise::Mode::SHL;
        if (str == "SHR") return Elemwise::Mode::SHR;
        if (str == "COND_LEQ_MOV") return Elemwise::Mode::COND_LEQ_MOV;
        if (str == "FUSE_MUL_ADD3") return Elemwise::Mode::FUSE_MUL_ADD3;
        if (str == "FUSE_MUL_ADD4") return Elemwise::Mode::FUSE_MUL_ADD4;
        if (str == "FUSE_ADD_RELU") return Elemwise::Mode::FUSE_ADD_RELU;
        if (str == "FUSE_ADD_SIGMOID") return Elemwise::Mode::FUSE_ADD_SIGMOID;
        if (str == "FUSE_ADD_TANH") return Elemwise::Mode::FUSE_ADD_TANH;
        if (str == "FAST_TANH") return Elemwise::Mode::FAST_TANH;
        if (str == "FAST_TANH_GRAD") return Elemwise::Mode::FAST_TANH_GRAD;
        if (str == "ROUND") return Elemwise::Mode::ROUND;
        if (str == "RMULH") return Elemwise::Mode::RMULH;
        if (str == "ATAN2") return Elemwise::Mode::ATAN2;
        if (str == "ERF") return Elemwise::Mode::ERF;
        if (str == "ERFINV") return Elemwise::Mode::ERFINV;
        if (str == "ERFC") return Elemwise::Mode::ERFC;
        if (str == "ERFCINV") return Elemwise::Mode::ERFCINV;
        if (str == "H_SWISH") return Elemwise::Mode::H_SWISH;
        if (str == "H_SWISH_GRAD") return Elemwise::Mode::H_SWISH_GRAD;
        if (str == "FUSE_ADD_H_SWISH") return Elemwise::Mode::FUSE_ADD_H_SWISH;
        if (str == "NOT") return Elemwise::Mode::NOT;
        if (str == "AND") return Elemwise::Mode::AND;
        if (str == "OR") return Elemwise::Mode::OR;
        if (str == "XOR") return Elemwise::Mode::XOR;
        if (str == "SILU") return Elemwise::Mode::SILU;
        if (str == "SILU_GRAD") return Elemwise::Mode::SILU_GRAD;
        if (str == "GELU") return Elemwise::Mode::GELU;
        if (str == "GELU_GRAD") return Elemwise::Mode::GELU_GRAD;
        if (str == "COND_LT_MOV") return Elemwise::Mode::COND_LT_MOV;
        if (str == "NEQ") return Elemwise::Mode::NEQ;
        if (str == "ISNAN") return Elemwise::Mode::ISNAN;
        if (str == "ISINF") return Elemwise::Mode::ISINF;
        if (str == "SINH") return Elemwise::Mode::SINH;
        if (str == "COSH") return Elemwise::Mode::COSH;
        if (str == "ASINH") return Elemwise::Mode::ASINH;
        if (str == "ACOSH") return Elemwise::Mode::ACOSH;
        if (str == "ATANH") return Elemwise::Mode::ATANH;
        if (str == "TAN") return Elemwise::Mode::TAN;
        if (str == "ASINH_GRAD") return Elemwise::Mode::ASINH_GRAD;
        if (str == "ACOSH_GRAD") return Elemwise::Mode::ACOSH_GRAD;
        if (str == "ATANH_GRAD") return Elemwise::Mode::ATANH_GRAD;
        if (str == "PRELU") return Elemwise::Mode::PRELU;
        if (str == "CLIP") return Elemwise::Mode::CLIP;
        if (str == "PRELU_GRAD") return Elemwise::Mode::PRELU_GRAD;
        if (str == "SOFTPLUS") return Elemwise::Mode::SOFTPLUS;
        if (str == "SOFTPLUS_GRAD") return Elemwise::Mode::SOFTPLUS_GRAD;
        if (str == "RELU6") return Elemwise::Mode::RELU6;
        if (str == "RELU6_GRAD") return Elemwise::Mode::RELU6_GRAD;
        if (str == "HSIGMOID") return Elemwise::Mode::HSIGMOID;
        if (str == "HSIGMOID_GRAD") return Elemwise::Mode::HSIGMOID_GRAD;
        if (str == "LOGSIGMOID") return Elemwise::Mode::LOGSIGMOID;
        if (str == "SQRT") return Elemwise::Mode::SQRT;
        if (str == "SQUARE") return Elemwise::Mode::SQUARE;
        if (str == "SIGN") return Elemwise::Mode::SIGN;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, Elemwise::Mode>();

ElemwiseInst
    .def(py::init<::megdnn::param::Elemwise::Mode, std::string>(), py::arg("mode") = ::megdnn::param::Elemwise::Mode::RELU, py::arg("scope") = {})
    .def_readwrite("mode", &Elemwise::mode);

py::class_<ElemwiseMultiType, std::shared_ptr<ElemwiseMultiType>, OpDef> ElemwiseMultiTypeInst(m, "ElemwiseMultiType");

py::enum_<ElemwiseMultiType::Mode>(ElemwiseMultiTypeInst, "Mode")
    .value("FUSE_MUL_ADD3_INT16x32x32x32", ElemwiseMultiType::Mode::FUSE_MUL_ADD3_INT16x32x32x32)
    .value("FUSE_MUL_ADD3_IXxF32xF32xI8", ElemwiseMultiType::Mode::FUSE_MUL_ADD3_IXxF32xF32xI8)
    .value("ROUND_SHR_SATURATE_IXxI8xI8", ElemwiseMultiType::Mode::ROUND_SHR_SATURATE_IXxI8xI8)
    .value("FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT16x16x16x8", ElemwiseMultiType::Mode::FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT16x16x16x8)
    .value("FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT32x32x32x8", ElemwiseMultiType::Mode::FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT32x32x32x8)
    .value("ROUND_SHR_SATURATE_IXxI8xI16", ElemwiseMultiType::Mode::ROUND_SHR_SATURATE_IXxI8xI16)
    .value("QADD", ElemwiseMultiType::Mode::QADD)
    .value("QFUSE_ADD_RELU", ElemwiseMultiType::Mode::QFUSE_ADD_RELU)
    .value("QMUL", ElemwiseMultiType::Mode::QMUL)
    .value("QMIN", ElemwiseMultiType::Mode::QMIN)
    .value("QMAX", ElemwiseMultiType::Mode::QMAX)
    .value("QSUB", ElemwiseMultiType::Mode::QSUB)
    .value("QTRUE_DIV", ElemwiseMultiType::Mode::QTRUE_DIV)
    .value("QFUSE_ADD_SIGMOID", ElemwiseMultiType::Mode::QFUSE_ADD_SIGMOID)
    .value("QFUSE_ADD_TANH", ElemwiseMultiType::Mode::QFUSE_ADD_TANH)
    .value("QRELU", ElemwiseMultiType::Mode::QRELU)
    .value("QABS", ElemwiseMultiType::Mode::QABS)
    .value("QSIGMOID", ElemwiseMultiType::Mode::QSIGMOID)
    .value("QEXP", ElemwiseMultiType::Mode::QEXP)
    .value("QTANH", ElemwiseMultiType::Mode::QTANH)
    .value("QFUSE_MUL_ADD3", ElemwiseMultiType::Mode::QFUSE_MUL_ADD3)
    .value("QFAST_TANH", ElemwiseMultiType::Mode::QFAST_TANH)
    .value("QNEGATE", ElemwiseMultiType::Mode::QNEGATE)
    .value("QACOS", ElemwiseMultiType::Mode::QACOS)
    .value("QASIN", ElemwiseMultiType::Mode::QASIN)
    .value("QCEIL", ElemwiseMultiType::Mode::QCEIL)
    .value("QCOS", ElemwiseMultiType::Mode::QCOS)
    .value("QEXPM1", ElemwiseMultiType::Mode::QEXPM1)
    .value("QFLOOR", ElemwiseMultiType::Mode::QFLOOR)
    .value("QLOG", ElemwiseMultiType::Mode::QLOG)
    .value("QLOG1P", ElemwiseMultiType::Mode::QLOG1P)
    .value("QSIN", ElemwiseMultiType::Mode::QSIN)
    .value("QROUND", ElemwiseMultiType::Mode::QROUND)
    .value("QERF", ElemwiseMultiType::Mode::QERF)
    .value("QERFINV", ElemwiseMultiType::Mode::QERFINV)
    .value("QERFC", ElemwiseMultiType::Mode::QERFC)
    .value("QERFCINV", ElemwiseMultiType::Mode::QERFCINV)
    .value("QABS_GRAD", ElemwiseMultiType::Mode::QABS_GRAD)
    .value("QFLOOR_DIV", ElemwiseMultiType::Mode::QFLOOR_DIV)
    .value("QMOD", ElemwiseMultiType::Mode::QMOD)
    .value("QSIGMOID_GRAD", ElemwiseMultiType::Mode::QSIGMOID_GRAD)
    .value("QSWITCH_GT0", ElemwiseMultiType::Mode::QSWITCH_GT0)
    .value("QTANH_GRAD", ElemwiseMultiType::Mode::QTANH_GRAD)
    .value("QLT", ElemwiseMultiType::Mode::QLT)
    .value("QLEQ", ElemwiseMultiType::Mode::QLEQ)
    .value("QEQ", ElemwiseMultiType::Mode::QEQ)
    .value("QPOW", ElemwiseMultiType::Mode::QPOW)
    .value("QLOG_SUM_EXP", ElemwiseMultiType::Mode::QLOG_SUM_EXP)
    .value("QFAST_TANH_GRAD", ElemwiseMultiType::Mode::QFAST_TANH_GRAD)
    .value("QATAN2", ElemwiseMultiType::Mode::QATAN2)
    .value("QCOND_LEQ_MOV", ElemwiseMultiType::Mode::QCOND_LEQ_MOV)
    .value("QH_SWISH", ElemwiseMultiType::Mode::QH_SWISH)
    .value("QFUSE_ADD_H_SWISH", ElemwiseMultiType::Mode::QFUSE_ADD_H_SWISH)
    .value("QH_SWISH_GRAD", ElemwiseMultiType::Mode::QH_SWISH_GRAD)
    .value("FUSE_MUL_ADD3_INT16xF32xF32xF32", ElemwiseMultiType::Mode::FUSE_MUL_ADD3_INT16xF32xF32xF32)
    .value("MUL_INT16xF32xF32", ElemwiseMultiType::Mode::MUL_INT16xF32xF32)
    .value("FUSE_MUL_ADD3_UINT8xF32xF32xF32", ElemwiseMultiType::Mode::FUSE_MUL_ADD3_UINT8xF32xF32xF32)
    .value("QCOND_LT_MOV", ElemwiseMultiType::Mode::QCOND_LT_MOV)
    .value("EQ", ElemwiseMultiType::Mode::EQ)
    .value("NEQ", ElemwiseMultiType::Mode::NEQ)
    .value("LT", ElemwiseMultiType::Mode::LT)
    .value("LEQ", ElemwiseMultiType::Mode::LEQ)
    .value("ISNAN", ElemwiseMultiType::Mode::ISNAN)
    .value("ISINF", ElemwiseMultiType::Mode::ISINF)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "FUSE_MUL_ADD3_INT16x32x32x32") return ElemwiseMultiType::Mode::FUSE_MUL_ADD3_INT16x32x32x32;
        if (str == "FUSE_MUL_ADD3_IXxF32xF32xI8") return ElemwiseMultiType::Mode::FUSE_MUL_ADD3_IXxF32xF32xI8;
        if (str == "ROUND_SHR_SATURATE_IXxI8xI8") return ElemwiseMultiType::Mode::ROUND_SHR_SATURATE_IXxI8xI8;
        if (str == "FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT16x16x16x8") return ElemwiseMultiType::Mode::FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT16x16x16x8;
        if (str == "FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT32x32x32x8") return ElemwiseMultiType::Mode::FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT32x32x32x8;
        if (str == "ROUND_SHR_SATURATE_IXxI8xI16") return ElemwiseMultiType::Mode::ROUND_SHR_SATURATE_IXxI8xI16;
        if (str == "QADD") return ElemwiseMultiType::Mode::QADD;
        if (str == "QFUSE_ADD_RELU") return ElemwiseMultiType::Mode::QFUSE_ADD_RELU;
        if (str == "QMUL") return ElemwiseMultiType::Mode::QMUL;
        if (str == "QMIN") return ElemwiseMultiType::Mode::QMIN;
        if (str == "QMAX") return ElemwiseMultiType::Mode::QMAX;
        if (str == "QSUB") return ElemwiseMultiType::Mode::QSUB;
        if (str == "QTRUE_DIV") return ElemwiseMultiType::Mode::QTRUE_DIV;
        if (str == "QFUSE_ADD_SIGMOID") return ElemwiseMultiType::Mode::QFUSE_ADD_SIGMOID;
        if (str == "QFUSE_ADD_TANH") return ElemwiseMultiType::Mode::QFUSE_ADD_TANH;
        if (str == "QRELU") return ElemwiseMultiType::Mode::QRELU;
        if (str == "QABS") return ElemwiseMultiType::Mode::QABS;
        if (str == "QSIGMOID") return ElemwiseMultiType::Mode::QSIGMOID;
        if (str == "QEXP") return ElemwiseMultiType::Mode::QEXP;
        if (str == "QTANH") return ElemwiseMultiType::Mode::QTANH;
        if (str == "QFUSE_MUL_ADD3") return ElemwiseMultiType::Mode::QFUSE_MUL_ADD3;
        if (str == "QFAST_TANH") return ElemwiseMultiType::Mode::QFAST_TANH;
        if (str == "QNEGATE") return ElemwiseMultiType::Mode::QNEGATE;
        if (str == "QACOS") return ElemwiseMultiType::Mode::QACOS;
        if (str == "QASIN") return ElemwiseMultiType::Mode::QASIN;
        if (str == "QCEIL") return ElemwiseMultiType::Mode::QCEIL;
        if (str == "QCOS") return ElemwiseMultiType::Mode::QCOS;
        if (str == "QEXPM1") return ElemwiseMultiType::Mode::QEXPM1;
        if (str == "QFLOOR") return ElemwiseMultiType::Mode::QFLOOR;
        if (str == "QLOG") return ElemwiseMultiType::Mode::QLOG;
        if (str == "QLOG1P") return ElemwiseMultiType::Mode::QLOG1P;
        if (str == "QSIN") return ElemwiseMultiType::Mode::QSIN;
        if (str == "QROUND") return ElemwiseMultiType::Mode::QROUND;
        if (str == "QERF") return ElemwiseMultiType::Mode::QERF;
        if (str == "QERFINV") return ElemwiseMultiType::Mode::QERFINV;
        if (str == "QERFC") return ElemwiseMultiType::Mode::QERFC;
        if (str == "QERFCINV") return ElemwiseMultiType::Mode::QERFCINV;
        if (str == "QABS_GRAD") return ElemwiseMultiType::Mode::QABS_GRAD;
        if (str == "QFLOOR_DIV") return ElemwiseMultiType::Mode::QFLOOR_DIV;
        if (str == "QMOD") return ElemwiseMultiType::Mode::QMOD;
        if (str == "QSIGMOID_GRAD") return ElemwiseMultiType::Mode::QSIGMOID_GRAD;
        if (str == "QSWITCH_GT0") return ElemwiseMultiType::Mode::QSWITCH_GT0;
        if (str == "QTANH_GRAD") return ElemwiseMultiType::Mode::QTANH_GRAD;
        if (str == "QLT") return ElemwiseMultiType::Mode::QLT;
        if (str == "QLEQ") return ElemwiseMultiType::Mode::QLEQ;
        if (str == "QEQ") return ElemwiseMultiType::Mode::QEQ;
        if (str == "QPOW") return ElemwiseMultiType::Mode::QPOW;
        if (str == "QLOG_SUM_EXP") return ElemwiseMultiType::Mode::QLOG_SUM_EXP;
        if (str == "QFAST_TANH_GRAD") return ElemwiseMultiType::Mode::QFAST_TANH_GRAD;
        if (str == "QATAN2") return ElemwiseMultiType::Mode::QATAN2;
        if (str == "QCOND_LEQ_MOV") return ElemwiseMultiType::Mode::QCOND_LEQ_MOV;
        if (str == "QH_SWISH") return ElemwiseMultiType::Mode::QH_SWISH;
        if (str == "QFUSE_ADD_H_SWISH") return ElemwiseMultiType::Mode::QFUSE_ADD_H_SWISH;
        if (str == "QH_SWISH_GRAD") return ElemwiseMultiType::Mode::QH_SWISH_GRAD;
        if (str == "FUSE_MUL_ADD3_INT16xF32xF32xF32") return ElemwiseMultiType::Mode::FUSE_MUL_ADD3_INT16xF32xF32xF32;
        if (str == "MUL_INT16xF32xF32") return ElemwiseMultiType::Mode::MUL_INT16xF32xF32;
        if (str == "FUSE_MUL_ADD3_UINT8xF32xF32xF32") return ElemwiseMultiType::Mode::FUSE_MUL_ADD3_UINT8xF32xF32xF32;
        if (str == "QCOND_LT_MOV") return ElemwiseMultiType::Mode::QCOND_LT_MOV;
        if (str == "EQ") return ElemwiseMultiType::Mode::EQ;
        if (str == "NEQ") return ElemwiseMultiType::Mode::NEQ;
        if (str == "LT") return ElemwiseMultiType::Mode::LT;
        if (str == "LEQ") return ElemwiseMultiType::Mode::LEQ;
        if (str == "ISNAN") return ElemwiseMultiType::Mode::ISNAN;
        if (str == "ISINF") return ElemwiseMultiType::Mode::ISINF;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, ElemwiseMultiType::Mode>();

ElemwiseMultiTypeInst
    .def(py::init<::megdnn::param::ElemwiseMultiType::Mode, ::megdnn::DType, std::string>(), py::arg("mode") = ::megdnn::param::ElemwiseMultiType::Mode::FUSE_MUL_ADD3_INT16x32x32x32, py::arg("dtype"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("mode", &ElemwiseMultiType::mode)
    .def_readwrite("dtype", &ElemwiseMultiType::dtype);

py::class_<ExponentialRNG, std::shared_ptr<ExponentialRNG>, OpDef> ExponentialRNGInst(m, "ExponentialRNG");

ExponentialRNGInst
    .def(py::init<uint64_t, size_t, std::string>(), py::arg("seed") = 0, py::arg("handle"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("seed", &ExponentialRNG::seed)
    .def_readwrite("handle", &ExponentialRNG::handle);

py::class_<ExternOpr, std::shared_ptr<ExternOpr>, OpDef> ExternOprInst(m, "ExternOpr");

ExternOprInst
    .def(py::init<std::vector<std::vector<size_t>>, std::string, std::string, size_t, std::vector<::megdnn::DType>, std::string>(), py::arg("output_shapes"), py::arg("name"), py::arg("data"), py::arg("data_len"), py::arg("output_dtypes"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("output_shapes", &ExternOpr::output_shapes)
    .def_readwrite("name", &ExternOpr::name)
    .def_readwrite("data", &ExternOpr::data)
    .def_readwrite("data_len", &ExternOpr::data_len)
    .def_readwrite("output_dtypes", &ExternOpr::output_dtypes);

py::class_<Eye, std::shared_ptr<Eye>, OpDef> EyeInst(m, "Eye");

EyeInst
    .def(py::init<int32_t, ::megdnn::DType, ::mgb::CompNode, std::string>(), py::arg("k") = 0, py::arg("dtype") = megdnn::DType::from_enum(megdnn::DTypeEnum::Float32), py::arg("comp_node"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("k", &Eye::k)
    .def_readwrite("dtype", &Eye::dtype)
    .def_readwrite("comp_node", &Eye::comp_node);

py::class_<FakeQuant, std::shared_ptr<FakeQuant>, OpDef> FakeQuantInst(m, "FakeQuant");

FakeQuantInst
    .def(py::init<int32_t, int32_t, std::string>(), py::arg("qmin") = -2147483648, py::arg("qmax") = 2147483647, py::arg("scope") = {})
    .def_readwrite("qmin", &FakeQuant::qmin)
    .def_readwrite("qmax", &FakeQuant::qmax);

py::class_<FastpathCopy, std::shared_ptr<FastpathCopy>, OpDef> FastpathCopyInst(m, "FastpathCopy");

FastpathCopyInst
    .def(py::init<>());

py::class_<Fill, std::shared_ptr<Fill>, OpDef> FillInst(m, "Fill");

FillInst
    .def(py::init<float, ::megdnn::DType, ::mgb::CompNode, std::string>(), py::arg("value") = 0, py::arg("dtype"), py::arg("comp_node"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("value", &Fill::value)
    .def_readwrite("dtype", &Fill::dtype)
    .def_readwrite("comp_node", &Fill::comp_node);

py::class_<FillLike, std::shared_ptr<FillLike>, OpDef> FillLikeInst(m, "FillLike");

FillLikeInst
    .def(py::init<float, ::mgb::CompNode, std::string>(), py::arg("value") = 0, py::arg("comp_node"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("value", &FillLike::value)
    .def_readwrite("comp_node", &FillLike::comp_node);

py::class_<Flip, std::shared_ptr<Flip>, OpDef> FlipInst(m, "Flip");

FlipInst
    .def(py::init<bool, bool, std::string>(), py::arg("vertical") = false, py::arg("horizontal") = false, py::arg("scope") = {})
    .def_readwrite("vertical", &Flip::vertical)
    .def_readwrite("horizontal", &Flip::horizontal);

py::class_<GammaRNG, std::shared_ptr<GammaRNG>, OpDef> GammaRNGInst(m, "GammaRNG");

GammaRNGInst
    .def(py::init<uint64_t, size_t, std::string>(), py::arg("seed") = 0, py::arg("handle"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("seed", &GammaRNG::seed)
    .def_readwrite("handle", &GammaRNG::handle);

py::class_<GaussianRNG, std::shared_ptr<GaussianRNG>, OpDef> GaussianRNGInst(m, "GaussianRNG");

GaussianRNGInst
    .def(py::init<uint64_t, float, float, ::megdnn::DType, size_t, std::string>(), py::arg("seed") = 0, py::arg("mean") = 0, py::arg("std") = 1, py::arg("dtype") = megdnn::DType::from_enum(megdnn::DTypeEnum::Float32), py::arg("handle"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("seed", &GaussianRNG::seed)
    .def_readwrite("mean", &GaussianRNG::mean)
    .def_readwrite("std", &GaussianRNG::std)
    .def_readwrite("dtype", &GaussianRNG::dtype)
    .def_readwrite("handle", &GaussianRNG::handle);

py::class_<GeneralNorm, std::shared_ptr<GeneralNorm>, OpDef> GeneralNormInst(m, "GeneralNorm");

GeneralNormInst
    .def(py::init<bool, float, uint64_t, uint64_t, std::string>(), py::arg("affine") = true, py::arg("eps") = 1e-5f, py::arg("axis_start") = 0, py::arg("axis_end") = 0, py::arg("scope") = {})
    .def_readwrite("affine", &GeneralNorm::affine)
    .def_readwrite("eps", &GeneralNorm::eps)
    .def_readwrite("axis_start", &GeneralNorm::axis_start)
    .def_readwrite("axis_end", &GeneralNorm::axis_end);

py::class_<GetVarShape, std::shared_ptr<GetVarShape>, OpDef> GetVarShapeInst(m, "GetVarShape");

GetVarShapeInst
    .def(py::init<int32_t, std::string>(), py::arg("axis") = ::megdnn::param::OptionalAxisV1::INVALID_AXIS, py::arg("scope") = {})
    .def_readwrite("axis", &GetVarShape::axis);

py::class_<GroupLocal, std::shared_ptr<GroupLocal>, OpDef> GroupLocalInst(m, "GroupLocal");

GroupLocalInst.attr("Mode") = BatchConvBiasInst.attr("Mode");

GroupLocalInst.attr("Sparse") = BatchConvBiasInst.attr("Sparse");

GroupLocalInst.attr("Format") = AdaptivePoolingInst.attr("Format");

GroupLocalInst.attr("ComputeMode") = BatchConvBiasInst.attr("ComputeMode");

GroupLocalInst
    .def(py::init<::megdnn::param::Convolution::Mode, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, ::megdnn::param::Convolution::Sparse, ::megdnn::param::Convolution::Format, ::megdnn::param::Convolution::ComputeMode, std::string>(), py::arg("mode") = ::megdnn::param::Convolution::Mode::CROSS_CORRELATION, py::arg("pad_h") = 0, py::arg("pad_w") = 0, py::arg("stride_h") = 1, py::arg("stride_w") = 1, py::arg("dilate_h") = 1, py::arg("dilate_w") = 1, py::arg("sparse") = ::megdnn::param::Convolution::Sparse::DENSE, py::arg("format") = ::megdnn::param::Convolution::Format::NCHW, py::arg("compute_mode") = ::megdnn::param::Convolution::ComputeMode::DEFAULT, py::arg("scope") = {})
    .def_readwrite("mode", &GroupLocal::mode)
    .def_readwrite("pad_h", &GroupLocal::pad_h)
    .def_readwrite("pad_w", &GroupLocal::pad_w)
    .def_readwrite("stride_h", &GroupLocal::stride_h)
    .def_readwrite("stride_w", &GroupLocal::stride_w)
    .def_readwrite("dilate_h", &GroupLocal::dilate_h)
    .def_readwrite("dilate_w", &GroupLocal::dilate_w)
    .def_readwrite("sparse", &GroupLocal::sparse)
    .def_readwrite("format", &GroupLocal::format)
    .def_readwrite("compute_mode", &GroupLocal::compute_mode);

py::class_<GroupNorm, std::shared_ptr<GroupNorm>, OpDef> GroupNormInst(m, "GroupNorm");

GroupNormInst.attr("Format") = AdaptivePoolingInst.attr("Format");

GroupNormInst
    .def(py::init<bool, float, uint32_t, ::megdnn::param::GroupNorm::Format, std::string>(), py::arg("affine") = true, py::arg("eps") = 1e-5f, py::arg("group") = 1, py::arg("format") = ::megdnn::param::GroupNorm::Format::NCHW, py::arg("scope") = {})
    .def_readwrite("affine", &GroupNorm::affine)
    .def_readwrite("eps", &GroupNorm::eps)
    .def_readwrite("group", &GroupNorm::group)
    .def_readwrite("format", &GroupNorm::format);

py::class_<Identity, std::shared_ptr<Identity>, OpDef> IdentityInst(m, "Identity");

IdentityInst
    .def(py::init<>());

py::class_<Images2Neibs, std::shared_ptr<Images2Neibs>, OpDef> Images2NeibsInst(m, "Images2Neibs");

Images2NeibsInst
    .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, std::string>(), py::arg("pad_h") = 0, py::arg("pad_w") = 0, py::arg("stride_h") = 1, py::arg("stride_w") = 1, py::arg("dilate_h") = 1, py::arg("dilate_w") = 1, py::arg("window_h") = 3, py::arg("window_w") = 3, py::arg("scope") = {})
    .def_readwrite("pad_h", &Images2Neibs::pad_h)
    .def_readwrite("pad_w", &Images2Neibs::pad_w)
    .def_readwrite("stride_h", &Images2Neibs::stride_h)
    .def_readwrite("stride_w", &Images2Neibs::stride_w)
    .def_readwrite("dilate_h", &Images2Neibs::dilate_h)
    .def_readwrite("dilate_w", &Images2Neibs::dilate_w)
    .def_readwrite("window_h", &Images2Neibs::window_h)
    .def_readwrite("window_w", &Images2Neibs::window_w);

py::class_<IncrMeshIndexing, std::shared_ptr<IncrMeshIndexing>, OpDef> IncrMeshIndexingInst(m, "IncrMeshIndexing");

IncrMeshIndexingInst
    .def(py::init<std::vector<std::tuple<int8_t, bool, bool, bool, bool>>, std::string>(), py::arg("items"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("items", &IncrMeshIndexing::items);

py::class_<IncrSubtensor, std::shared_ptr<IncrSubtensor>, OpDef> IncrSubtensorInst(m, "IncrSubtensor");

IncrSubtensorInst
    .def(py::init<std::vector<std::tuple<int8_t, bool, bool, bool, bool>>, std::string>(), py::arg("items"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("items", &IncrSubtensor::items);

py::class_<IndexingIncrMultiAxisVec, std::shared_ptr<IndexingIncrMultiAxisVec>, OpDef> IndexingIncrMultiAxisVecInst(m, "IndexingIncrMultiAxisVec");

IndexingIncrMultiAxisVecInst
    .def(py::init<std::vector<std::tuple<int8_t, bool, bool, bool, bool>>, std::string>(), py::arg("items"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("items", &IndexingIncrMultiAxisVec::items);

py::class_<IndexingMultiAxisVec, std::shared_ptr<IndexingMultiAxisVec>, OpDef> IndexingMultiAxisVecInst(m, "IndexingMultiAxisVec");

IndexingMultiAxisVecInst
    .def(py::init<std::vector<std::tuple<int8_t, bool, bool, bool, bool>>, std::string>(), py::arg("items"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("items", &IndexingMultiAxisVec::items);

py::class_<IndexingOneHot, std::shared_ptr<IndexingOneHot>, OpDef> IndexingOneHotInst(m, "IndexingOneHot");

IndexingOneHotInst
    .def(py::init<int32_t, int32_t, std::string>(), py::arg("axis") = 0, py::arg("ndim"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("axis", &IndexingOneHot::axis)
    .def_readwrite("ndim", &IndexingOneHot::ndim);

py::class_<IndexingSetMultiAxisVec, std::shared_ptr<IndexingSetMultiAxisVec>, OpDef> IndexingSetMultiAxisVecInst(m, "IndexingSetMultiAxisVec");

IndexingSetMultiAxisVecInst
    .def(py::init<std::vector<std::tuple<int8_t, bool, bool, bool, bool>>, std::string>(), py::arg("items"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("items", &IndexingSetMultiAxisVec::items);

py::class_<IndexingSetOneHot, std::shared_ptr<IndexingSetOneHot>, OpDef> IndexingSetOneHotInst(m, "IndexingSetOneHot");

IndexingSetOneHotInst
    .def(py::init<int32_t, int32_t, std::string>(), py::arg("axis") = 0, py::arg("ndim"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("axis", &IndexingSetOneHot::axis)
    .def_readwrite("ndim", &IndexingSetOneHot::ndim);

py::class_<InplaceAdd, std::shared_ptr<InplaceAdd>, OpDef> InplaceAddInst(m, "InplaceAdd");

InplaceAddInst
    .def(py::init<>());

py::class_<InstanceNorm, std::shared_ptr<InstanceNorm>, OpDef> InstanceNormInst(m, "InstanceNorm");

InstanceNormInst.attr("Format") = AdaptivePoolingInst.attr("Format");

InstanceNormInst
    .def(py::init<bool, float, uint32_t, ::megdnn::param::GroupNorm::Format, std::string>(), py::arg("affine") = true, py::arg("eps") = 1e-5f, py::arg("group") = 1, py::arg("format") = ::megdnn::param::GroupNorm::Format::NCHW, py::arg("scope") = {})
    .def_readwrite("affine", &InstanceNorm::affine)
    .def_readwrite("eps", &InstanceNorm::eps)
    .def_readwrite("group", &InstanceNorm::group)
    .def_readwrite("format", &InstanceNorm::format);

py::class_<LAMBUpdate, std::shared_ptr<LAMBUpdate>, OpDef> LAMBUpdateInst(m, "LAMBUpdate");

LAMBUpdateInst
    .def(py::init<float, float, float, float, float, float, bool, bool, std::string>(), py::arg("beta_1") = 1.f, py::arg("beta_2") = 1.f, py::arg("step") = 1.f, py::arg("lr") = 1.f, py::arg("weight_decay") = 1.f, py::arg("eps") = 1.f, py::arg("bias_correction") = true, py::arg("always_adapt") = false, py::arg("scope") = {})
    .def_readwrite("beta_1", &LAMBUpdate::beta_1)
    .def_readwrite("beta_2", &LAMBUpdate::beta_2)
    .def_readwrite("step", &LAMBUpdate::step)
    .def_readwrite("lr", &LAMBUpdate::lr)
    .def_readwrite("weight_decay", &LAMBUpdate::weight_decay)
    .def_readwrite("eps", &LAMBUpdate::eps)
    .def_readwrite("bias_correction", &LAMBUpdate::bias_correction)
    .def_readwrite("always_adapt", &LAMBUpdate::always_adapt);

py::class_<LRN, std::shared_ptr<LRN>, OpDef> LRNInst(m, "LRN");

LRNInst
    .def(py::init<uint32_t, float, float, float, std::string>(), py::arg("n") = 5, py::arg("k") = 2.f, py::arg("alpha") = 1e-4f, py::arg("beta") = 0.75f, py::arg("scope") = {})
    .def_readwrite("n", &LRN::n)
    .def_readwrite("k", &LRN::k)
    .def_readwrite("alpha", &LRN::alpha)
    .def_readwrite("beta", &LRN::beta);

py::class_<LSQ, std::shared_ptr<LSQ>, OpDef> LSQInst(m, "LSQ");

LSQInst
    .def(py::init<int32_t, int32_t, std::string>(), py::arg("qmin") = -2147483648, py::arg("qmax") = 2147483647, py::arg("scope") = {})
    .def_readwrite("qmin", &LSQ::qmin)
    .def_readwrite("qmax", &LSQ::qmax);

py::class_<LSTM, std::shared_ptr<LSTM>, OpDef> LSTMInst(m, "LSTM");

LSTMInst.attr("FwdMode") = BatchNormInst.attr("FwdMode");

LSTMInst
    .def(py::init<uint32_t, bool, bool, uint32_t, uint32_t, float, ::megdnn::param::LSTM::FwdMode, std::string>(), py::arg("num_layers") = 1, py::arg("bidirectional") = false, py::arg("bias") = true, py::arg("hidden_size") = 128, py::arg("proj_size") = 0, py::arg("dropout") = 0.f, py::arg("fwd_mode") = ::megdnn::param::LSTM::FwdMode::TRAINING, py::arg("scope") = {})
    .def_readwrite("num_layers", &LSTM::num_layers)
    .def_readwrite("bidirectional", &LSTM::bidirectional)
    .def_readwrite("bias", &LSTM::bias)
    .def_readwrite("hidden_size", &LSTM::hidden_size)
    .def_readwrite("proj_size", &LSTM::proj_size)
    .def_readwrite("dropout", &LSTM::dropout)
    .def_readwrite("fwd_mode", &LSTM::fwd_mode);

py::class_<LSTMCell, std::shared_ptr<LSTMCell>, OpDef> LSTMCellInst(m, "LSTMCell");

LSTMCellInst
    .def(py::init<>());

py::class_<LayerNorm, std::shared_ptr<LayerNorm>, OpDef> LayerNormInst(m, "LayerNorm");

LayerNormInst
    .def(py::init<bool, float, uint64_t, uint64_t, std::string>(), py::arg("affine") = true, py::arg("eps") = 1e-5f, py::arg("normalized_dim") = 1, py::arg("normalized_size") = 1, py::arg("scope") = {})
    .def_readwrite("affine", &LayerNorm::affine)
    .def_readwrite("eps", &LayerNorm::eps)
    .def_readwrite("normalized_dim", &LayerNorm::normalized_dim)
    .def_readwrite("normalized_size", &LayerNorm::normalized_size);

py::class_<Linspace, std::shared_ptr<Linspace>, OpDef> LinspaceInst(m, "Linspace");

LinspaceInst
    .def(py::init<bool, ::mgb::CompNode, std::string>(), py::arg("endpoint") = true, py::arg("comp_node"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("endpoint", &Linspace::endpoint)
    .def_readwrite("comp_node", &Linspace::comp_node);

py::class_<MagicMindRuntime, std::shared_ptr<MagicMindRuntime>, OpDef> MagicMindRuntimeInst(m, "MagicMindRuntime");

MagicMindRuntimeInst
    .def(py::init<std::string, size_t, std::string>(), py::arg("buf"), py::arg("buf_size"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("buf", &MagicMindRuntime::buf)
    .def_readwrite("buf_size", &MagicMindRuntime::buf_size);

py::class_<MaskedFill, std::shared_ptr<MaskedFill>, OpDef> MaskedFillInst(m, "MaskedFill");

MaskedFillInst
    .def(py::init<float, std::string>(), py::arg("value") = 0, py::arg("scope") = {})
    .def_readwrite("value", &MaskedFill::value);

py::class_<MatrixInverse, std::shared_ptr<MatrixInverse>, OpDef> MatrixInverseInst(m, "MatrixInverse");

MatrixInverseInst
    .def(py::init<>());

py::class_<MatrixMul, std::shared_ptr<MatrixMul>, OpDef> MatrixMulInst(m, "MatrixMul");

MatrixMulInst.attr("ComputeMode") = BatchedMatrixMulInst.attr("ComputeMode");

MatrixMulInst.attr("Format") = BatchedMatrixMulInst.attr("Format");

MatrixMulInst.attr("Strategy") = BatchConvBiasInst.attr("Strategy");

MatrixMulInst
    .def(py::init<bool, bool, ::megdnn::param::MatrixMul::ComputeMode, ::megdnn::param::MatrixMul::Format, ::megdnn::param::ExecutionPolicy::Strategy, uint64_t, uint32_t, uint32_t, std::string>(), py::arg("transposeA") = false, py::arg("transposeB") = false, py::arg("compute_mode") = ::megdnn::param::MatrixMul::ComputeMode::DEFAULT, py::arg("format") = ::megdnn::param::MatrixMul::Format::DEFAULT, py::arg("strategy") = static_cast<::megdnn::param::ExecutionPolicy::Strategy>(1), py::arg("workspace_limit") = 18446744073709551615ull, py::arg("dimA"), py::arg("dimB"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("transposeA", &MatrixMul::transposeA)
    .def_readwrite("transposeB", &MatrixMul::transposeB)
    .def_readwrite("compute_mode", &MatrixMul::compute_mode)
    .def_readwrite("format", &MatrixMul::format)
    .def_readwrite("strategy", &MatrixMul::strategy)
    .def_readwrite("workspace_limit", &MatrixMul::workspace_limit)
    .def_readwrite("dimA", &MatrixMul::dimA)
    .def_readwrite("dimB", &MatrixMul::dimB);

py::class_<MeshGrid, std::shared_ptr<MeshGrid>, OpDef> MeshGridInst(m, "MeshGrid");

MeshGridInst
    .def(py::init<std::string, std::string>(), py::arg("indexing"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("indexing", &MeshGrid::indexing);

py::class_<MeshIndexing, std::shared_ptr<MeshIndexing>, OpDef> MeshIndexingInst(m, "MeshIndexing");

MeshIndexingInst
    .def(py::init<std::vector<std::tuple<int8_t, bool, bool, bool, bool>>, std::string>(), py::arg("items"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("items", &MeshIndexing::items);

py::class_<MultiHeadAttn, std::shared_ptr<MultiHeadAttn>, OpDef> MultiHeadAttnInst(m, "MultiHeadAttn");

py::enum_<MultiHeadAttn::AttnMaskType>(MultiHeadAttnInst, "AttnMaskType")
    .value("NO_MASK", MultiHeadAttn::AttnMaskType::NO_MASK)
    .value("DEFAULT_MASK", MultiHeadAttn::AttnMaskType::DEFAULT_MASK)
    .value("CUDNN_STYLE_MASK", MultiHeadAttn::AttnMaskType::CUDNN_STYLE_MASK)
    .value("USER_DEFINED_MASK", MultiHeadAttn::AttnMaskType::USER_DEFINED_MASK)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "NO_MASK") return MultiHeadAttn::AttnMaskType::NO_MASK;
        if (str == "DEFAULT_MASK") return MultiHeadAttn::AttnMaskType::DEFAULT_MASK;
        if (str == "CUDNN_STYLE_MASK") return MultiHeadAttn::AttnMaskType::CUDNN_STYLE_MASK;
        if (str == "USER_DEFINED_MASK") return MultiHeadAttn::AttnMaskType::USER_DEFINED_MASK;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, MultiHeadAttn::AttnMaskType>();

py::enum_<MultiHeadAttn::TensorCombinationType>(MultiHeadAttnInst, "TensorCombinationType")
    .value("NONE", MultiHeadAttn::TensorCombinationType::NONE)
    .value("ONLY_MASK", MultiHeadAttn::TensorCombinationType::ONLY_MASK)
    .value("ONLY_BIASKV", MultiHeadAttn::TensorCombinationType::ONLY_BIASKV)
    .value("ALL", MultiHeadAttn::TensorCombinationType::ALL)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "NONE") return MultiHeadAttn::TensorCombinationType::NONE;
        if (str == "ONLY_MASK") return MultiHeadAttn::TensorCombinationType::ONLY_MASK;
        if (str == "ONLY_BIASKV") return MultiHeadAttn::TensorCombinationType::ONLY_BIASKV;
        if (str == "ALL") return MultiHeadAttn::TensorCombinationType::ALL;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, MultiHeadAttn::TensorCombinationType>();

MultiHeadAttnInst
    .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, bool, bool, bool, bool, float, uint32_t, ::megdnn::param::MultiHeadAttn::AttnMaskType, ::megdnn::param::MultiHeadAttn::TensorCombinationType, bool, bool, bool, bool, bool, uint64_t, float, float, size_t, std::string>(), py::arg("num_heads") = 1, py::arg("embeding_size") = 0, py::arg("k_size") = 0, py::arg("v_size") = 0, py::arg("qproj_size") = 0, py::arg("kproj_size") = 0, py::arg("vproj_size") = 0, py::arg("oproj_size") = 0, py::arg("qbias") = false, py::arg("kbias") = false, py::arg("vbias") = false, py::arg("obias") = false, py::arg("sm_scaler") = 1.f, py::arg("input_order") = 0, py::arg("attn_mask_type") = ::megdnn::param::MultiHeadAttn::AttnMaskType::NO_MASK, py::arg("tensor_combination_type") = ::megdnn::param::MultiHeadAttn::TensorCombinationType::NONE, py::arg("add_bias_kv") = false, py::arg("add_zero_attn") = false, py::arg("need_weights") = false, py::arg("reslink") = false, py::arg("training") = true, py::arg("seed") = 0, py::arg("attn_prob") = 0.f, py::arg("out_prob") = 0.f, py::arg("handle"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("num_heads", &MultiHeadAttn::num_heads)
    .def_readwrite("embeding_size", &MultiHeadAttn::embeding_size)
    .def_readwrite("k_size", &MultiHeadAttn::k_size)
    .def_readwrite("v_size", &MultiHeadAttn::v_size)
    .def_readwrite("qproj_size", &MultiHeadAttn::qproj_size)
    .def_readwrite("kproj_size", &MultiHeadAttn::kproj_size)
    .def_readwrite("vproj_size", &MultiHeadAttn::vproj_size)
    .def_readwrite("oproj_size", &MultiHeadAttn::oproj_size)
    .def_readwrite("qbias", &MultiHeadAttn::qbias)
    .def_readwrite("kbias", &MultiHeadAttn::kbias)
    .def_readwrite("vbias", &MultiHeadAttn::vbias)
    .def_readwrite("obias", &MultiHeadAttn::obias)
    .def_readwrite("sm_scaler", &MultiHeadAttn::sm_scaler)
    .def_readwrite("input_order", &MultiHeadAttn::input_order)
    .def_readwrite("attn_mask_type", &MultiHeadAttn::attn_mask_type)
    .def_readwrite("tensor_combination_type", &MultiHeadAttn::tensor_combination_type)
    .def_readwrite("add_bias_kv", &MultiHeadAttn::add_bias_kv)
    .def_readwrite("add_zero_attn", &MultiHeadAttn::add_zero_attn)
    .def_readwrite("need_weights", &MultiHeadAttn::need_weights)
    .def_readwrite("reslink", &MultiHeadAttn::reslink)
    .def_readwrite("training", &MultiHeadAttn::training)
    .def_readwrite("seed", &MultiHeadAttn::seed)
    .def_readwrite("attn_prob", &MultiHeadAttn::attn_prob)
    .def_readwrite("out_prob", &MultiHeadAttn::out_prob)
    .def_readwrite("handle", &MultiHeadAttn::handle);

py::class_<NMSKeep, std::shared_ptr<NMSKeep>, OpDef> NMSKeepInst(m, "NMSKeep");

NMSKeepInst
    .def(py::init<float, uint32_t, std::string>(), py::arg("iou_thresh"), py::arg("max_output"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("iou_thresh", &NMSKeep::iou_thresh)
    .def_readwrite("max_output", &NMSKeep::max_output);

py::class_<NonZero, std::shared_ptr<NonZero>, OpDef> NonZeroInst(m, "NonZero");

NonZeroInst
    .def(py::init<>());

py::class_<NvOf, std::shared_ptr<NvOf>, OpDef> NvOfInst(m, "NvOf");

NvOfInst
    .def(py::init<uint32_t, std::string>(), py::arg("precision") = 1, py::arg("scope") = {})
    .def_readwrite("precision", &NvOf::precision);

py::class_<Padding, std::shared_ptr<Padding>, OpDef> PaddingInst(m, "Padding");

py::enum_<Padding::PaddingMode>(PaddingInst, "PaddingMode")
    .value("REPLICATE", Padding::PaddingMode::REPLICATE)
    .value("REFLECT", Padding::PaddingMode::REFLECT)
    .value("CONSTANT", Padding::PaddingMode::CONSTANT)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "REPLICATE") return Padding::PaddingMode::REPLICATE;
        if (str == "REFLECT") return Padding::PaddingMode::REFLECT;
        if (str == "CONSTANT") return Padding::PaddingMode::CONSTANT;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, Padding::PaddingMode>();

PaddingInst
    .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, ::megdnn::param::Padding::PaddingMode, std::string>(), py::arg("front_offset_dim0") = 0, py::arg("front_offset_dim1") = 0, py::arg("front_offset_dim2") = 0, py::arg("front_offset_dim3") = 0, py::arg("front_offset_dim4") = 0, py::arg("front_offset_dim5") = 0, py::arg("front_offset_dim6") = 0, py::arg("back_offset_dim0") = 0, py::arg("back_offset_dim1") = 0, py::arg("back_offset_dim2") = 0, py::arg("back_offset_dim3") = 0, py::arg("back_offset_dim4") = 0, py::arg("back_offset_dim5") = 0, py::arg("back_offset_dim6") = 0, py::arg("padding_val") = 0, py::arg("padding_mode") = ::megdnn::param::Padding::PaddingMode::CONSTANT, py::arg("scope") = {})
    .def_readwrite("front_offset_dim0", &Padding::front_offset_dim0)
    .def_readwrite("front_offset_dim1", &Padding::front_offset_dim1)
    .def_readwrite("front_offset_dim2", &Padding::front_offset_dim2)
    .def_readwrite("front_offset_dim3", &Padding::front_offset_dim3)
    .def_readwrite("front_offset_dim4", &Padding::front_offset_dim4)
    .def_readwrite("front_offset_dim5", &Padding::front_offset_dim5)
    .def_readwrite("front_offset_dim6", &Padding::front_offset_dim6)
    .def_readwrite("back_offset_dim0", &Padding::back_offset_dim0)
    .def_readwrite("back_offset_dim1", &Padding::back_offset_dim1)
    .def_readwrite("back_offset_dim2", &Padding::back_offset_dim2)
    .def_readwrite("back_offset_dim3", &Padding::back_offset_dim3)
    .def_readwrite("back_offset_dim4", &Padding::back_offset_dim4)
    .def_readwrite("back_offset_dim5", &Padding::back_offset_dim5)
    .def_readwrite("back_offset_dim6", &Padding::back_offset_dim6)
    .def_readwrite("padding_val", &Padding::padding_val)
    .def_readwrite("padding_mode", &Padding::padding_mode);

py::class_<ParamPackConcat, std::shared_ptr<ParamPackConcat>, OpDef> ParamPackConcatInst(m, "ParamPackConcat");

ParamPackConcatInst
    .def(py::init<std::vector<int32_t>, std::string>(), py::arg("offsets"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("offsets", &ParamPackConcat::offsets);

py::class_<ParamPackSplit, std::shared_ptr<ParamPackSplit>, OpDef> ParamPackSplitInst(m, "ParamPackSplit");

ParamPackSplitInst
    .def(py::init<std::vector<int32_t>, std::vector<std::vector<size_t>>, std::string>(), py::arg("offsets"), py::arg("shapes"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("offsets", &ParamPackSplit::offsets)
    .def_readwrite("shapes", &ParamPackSplit::shapes);

py::class_<PermutationRNG, std::shared_ptr<PermutationRNG>, OpDef> PermutationRNGInst(m, "PermutationRNG");

PermutationRNGInst
    .def(py::init<uint64_t, ::megdnn::DType, size_t, std::string>(), py::arg("seed") = 0, py::arg("dtype") = megdnn::DType::from_enum(megdnn::DTypeEnum::Int32), py::arg("handle"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("seed", &PermutationRNG::seed)
    .def_readwrite("dtype", &PermutationRNG::dtype)
    .def_readwrite("handle", &PermutationRNG::handle);

py::class_<PixelShuffle, std::shared_ptr<PixelShuffle>, OpDef> PixelShuffleInst(m, "PixelShuffle");

PixelShuffleInst
    .def(py::init<int32_t, std::string>(), py::arg("factor"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("factor", &PixelShuffle::factor);

py::class_<PixelShuffleBackward, std::shared_ptr<PixelShuffleBackward>, OpDef> PixelShuffleBackwardInst(m, "PixelShuffleBackward");

PixelShuffleBackwardInst
    .def(py::init<int32_t, std::string>(), py::arg("factor"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("factor", &PixelShuffleBackward::factor);

py::class_<PoissonRNG, std::shared_ptr<PoissonRNG>, OpDef> PoissonRNGInst(m, "PoissonRNG");

PoissonRNGInst
    .def(py::init<uint64_t, size_t, std::string>(), py::arg("seed") = 0, py::arg("handle"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("seed", &PoissonRNG::seed)
    .def_readwrite("handle", &PoissonRNG::handle);

py::class_<Pooling, std::shared_ptr<Pooling>, OpDef> PoolingInst(m, "Pooling");

PoolingInst.attr("Mode") = AdaptivePoolingInst.attr("Mode");

PoolingInst.attr("Format") = AdaptivePoolingInst.attr("Format");

PoolingInst.attr("Strategy") = BatchConvBiasInst.attr("Strategy");

PoolingInst
    .def(py::init<::megdnn::param::Pooling::Mode, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, ::megdnn::param::Pooling::Format, ::megdnn::param::ExecutionPolicy::Strategy, uint64_t, std::string>(), py::arg("mode") = ::megdnn::param::Pooling::Mode::MAX, py::arg("pad_h") = 0, py::arg("pad_w") = 0, py::arg("stride_h") = 2, py::arg("stride_w") = 2, py::arg("window_h") = 2, py::arg("window_w") = 2, py::arg("format") = ::megdnn::param::Pooling::Format::NCHW, py::arg("strategy") = static_cast<::megdnn::param::ExecutionPolicy::Strategy>(1), py::arg("workspace_limit") = 18446744073709551615ull, py::arg("scope") = {})
    .def_readwrite("mode", &Pooling::mode)
    .def_readwrite("pad_h", &Pooling::pad_h)
    .def_readwrite("pad_w", &Pooling::pad_w)
    .def_readwrite("stride_h", &Pooling::stride_h)
    .def_readwrite("stride_w", &Pooling::stride_w)
    .def_readwrite("window_h", &Pooling::window_h)
    .def_readwrite("window_w", &Pooling::window_w)
    .def_readwrite("format", &Pooling::format)
    .def_readwrite("strategy", &Pooling::strategy)
    .def_readwrite("workspace_limit", &Pooling::workspace_limit);

py::class_<RNN, std::shared_ptr<RNN>, OpDef> RNNInst(m, "RNN");

py::enum_<RNN::NonlineMode>(RNNInst, "NonlineMode")
    .value("IDENTITY", RNN::NonlineMode::IDENTITY)
    .value("RELU", RNN::NonlineMode::RELU)
    .value("TANH", RNN::NonlineMode::TANH)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "IDENTITY") return RNN::NonlineMode::IDENTITY;
        if (str == "RELU") return RNN::NonlineMode::RELU;
        if (str == "TANH") return RNN::NonlineMode::TANH;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, RNN::NonlineMode>();

RNNInst.attr("FwdMode") = BatchNormInst.attr("FwdMode");

RNNInst
    .def(py::init<uint32_t, bool, bool, uint32_t, float, ::megdnn::param::RNN::NonlineMode, ::megdnn::param::RNN::FwdMode, std::string>(), py::arg("num_layers") = 1, py::arg("bidirectional") = false, py::arg("bias") = true, py::arg("hidden_size") = 128, py::arg("dropout") = 0.f, py::arg("nonlineMode") = ::megdnn::param::RNN::NonlineMode::IDENTITY, py::arg("fwd_mode") = ::megdnn::param::RNN::FwdMode::TRAINING, py::arg("scope") = {})
    .def_readwrite("num_layers", &RNN::num_layers)
    .def_readwrite("bidirectional", &RNN::bidirectional)
    .def_readwrite("bias", &RNN::bias)
    .def_readwrite("hidden_size", &RNN::hidden_size)
    .def_readwrite("dropout", &RNN::dropout)
    .def_readwrite("nonlineMode", &RNN::nonlineMode)
    .def_readwrite("fwd_mode", &RNN::fwd_mode);

py::class_<RNNCell, std::shared_ptr<RNNCell>, OpDef> RNNCellInst(m, "RNNCell");

RNNCellInst.attr("NonlineMode") = RNNInst.attr("NonlineMode");

RNNCellInst
    .def(py::init<::megdnn::param::RNNCell::NonlineMode, std::string>(), py::arg("nonlineMode") = ::megdnn::param::RNNCell::NonlineMode::IDENTITY, py::arg("scope") = {})
    .def_readwrite("nonlineMode", &RNNCell::nonlineMode);

py::class_<ROIAlign, std::shared_ptr<ROIAlign>, OpDef> ROIAlignInst(m, "ROIAlign");

py::enum_<ROIAlign::Mode>(ROIAlignInst, "Mode")
    .value("MAX", ROIAlign::Mode::MAX)
    .value("AVERAGE", ROIAlign::Mode::AVERAGE)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "MAX") return ROIAlign::Mode::MAX;
        if (str == "AVERAGE") return ROIAlign::Mode::AVERAGE;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, ROIAlign::Mode>();

ROIAlignInst.attr("Format") = AdaptivePoolingInst.attr("Format");

ROIAlignInst
    .def(py::init<::megdnn::param::ROIAlign::Mode, ::megdnn::param::ROIAlign::Format, float, float, uint32_t, uint32_t, uint32_t, uint32_t, std::string>(), py::arg("mode") = ::megdnn::param::ROIAlign::Mode::MAX, py::arg("format") = ::megdnn::param::ROIAlign::Format::NCHW, py::arg("spatial_scale") = 1.0, py::arg("offset") = 0.0, py::arg("pooled_height") = 1, py::arg("pooled_width") = 1, py::arg("sample_height") = 2, py::arg("sample_width") = 2, py::arg("scope") = {})
    .def_readwrite("mode", &ROIAlign::mode)
    .def_readwrite("format", &ROIAlign::format)
    .def_readwrite("spatial_scale", &ROIAlign::spatial_scale)
    .def_readwrite("offset", &ROIAlign::offset)
    .def_readwrite("pooled_height", &ROIAlign::pooled_height)
    .def_readwrite("pooled_width", &ROIAlign::pooled_width)
    .def_readwrite("sample_height", &ROIAlign::sample_height)
    .def_readwrite("sample_width", &ROIAlign::sample_width);

py::class_<ROIPooling, std::shared_ptr<ROIPooling>, OpDef> ROIPoolingInst(m, "ROIPooling");

py::enum_<ROIPooling::Mode>(ROIPoolingInst, "Mode")
    .value("MAX", ROIPooling::Mode::MAX)
    .value("AVERAGE", ROIPooling::Mode::AVERAGE)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "MAX") return ROIPooling::Mode::MAX;
        if (str == "AVERAGE") return ROIPooling::Mode::AVERAGE;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, ROIPooling::Mode>();

ROIPoolingInst
    .def(py::init<::megdnn::param::ROIPooling::Mode, float, std::string>(), py::arg("mode") = ::megdnn::param::ROIPooling::Mode::MAX, py::arg("scale") = 1.f, py::arg("scope") = {})
    .def_readwrite("mode", &ROIPooling::mode)
    .def_readwrite("scale", &ROIPooling::scale);

py::class_<Reduce, std::shared_ptr<Reduce>, OpDef> ReduceInst(m, "Reduce");

py::enum_<Reduce::Mode>(ReduceInst, "Mode")
    .value("SUM", Reduce::Mode::SUM)
    .value("SUM_SQR", Reduce::Mode::SUM_SQR)
    .value("PRODUCT", Reduce::Mode::PRODUCT)
    .value("MIN", Reduce::Mode::MIN)
    .value("MAX", Reduce::Mode::MAX)
    .value("MEAN", Reduce::Mode::MEAN)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "SUM") return Reduce::Mode::SUM;
        if (str == "SUM_SQR") return Reduce::Mode::SUM_SQR;
        if (str == "PRODUCT") return Reduce::Mode::PRODUCT;
        if (str == "MIN") return Reduce::Mode::MIN;
        if (str == "MAX") return Reduce::Mode::MAX;
        if (str == "MEAN") return Reduce::Mode::MEAN;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, Reduce::Mode>();

py::enum_<Reduce::DataType>(ReduceInst, "DataType")
    .value("DEFAULT", Reduce::DataType::DEFAULT)
    .value("FLOAT_IO16xC32", Reduce::DataType::FLOAT_IO16xC32)
    .value("FLOAT_O32xC32", Reduce::DataType::FLOAT_O32xC32)
    .value("FLOAT_O16xC32", Reduce::DataType::FLOAT_O16xC32)
    .value("QUINT_I8xO32", Reduce::DataType::QUINT_I8xO32)
    .value("QINT_I8xO32", Reduce::DataType::QINT_I8xO32)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "DEFAULT") return Reduce::DataType::DEFAULT;
        if (str == "FLOAT_IO16xC32") return Reduce::DataType::FLOAT_IO16xC32;
        if (str == "FLOAT_O32xC32") return Reduce::DataType::FLOAT_O32xC32;
        if (str == "FLOAT_O16xC32") return Reduce::DataType::FLOAT_O16xC32;
        if (str == "QUINT_I8xO32") return Reduce::DataType::QUINT_I8xO32;
        if (str == "QINT_I8xO32") return Reduce::DataType::QINT_I8xO32;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, Reduce::DataType>();

ReduceInst
    .def(py::init<::megdnn::param::Reduce::Mode, int32_t, ::megdnn::param::Reduce::DataType, bool, std::string>(), py::arg("mode") = ::megdnn::param::Reduce::Mode::SUM, py::arg("axis") = 2147483647, py::arg("data_type") = ::megdnn::param::Reduce::DataType::DEFAULT, py::arg("keepdim") = true, py::arg("scope") = {})
    .def_readwrite("mode", &Reduce::mode)
    .def_readwrite("axis", &Reduce::axis)
    .def_readwrite("data_type", &Reduce::data_type)
    .def_readwrite("keepdim", &Reduce::keepdim);

py::class_<RegionRestrictedConvolution, std::shared_ptr<RegionRestrictedConvolution>, OpDef> RegionRestrictedConvolutionInst(m, "RegionRestrictedConvolution");

RegionRestrictedConvolutionInst.attr("Mode") = BatchConvBiasInst.attr("Mode");

RegionRestrictedConvolutionInst.attr("Sparse") = BatchConvBiasInst.attr("Sparse");

RegionRestrictedConvolutionInst.attr("Format") = AdaptivePoolingInst.attr("Format");

RegionRestrictedConvolutionInst.attr("ComputeMode") = BatchConvBiasInst.attr("ComputeMode");

RegionRestrictedConvolutionInst
    .def(py::init<::megdnn::param::Convolution::Mode, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, ::megdnn::param::Convolution::Sparse, ::megdnn::param::Convolution::Format, ::megdnn::param::Convolution::ComputeMode, std::string>(), py::arg("mode") = ::megdnn::param::Convolution::Mode::CROSS_CORRELATION, py::arg("pad_h") = 0, py::arg("pad_w") = 0, py::arg("stride_h") = 1, py::arg("stride_w") = 1, py::arg("dilate_h") = 1, py::arg("dilate_w") = 1, py::arg("sparse") = ::megdnn::param::Convolution::Sparse::DENSE, py::arg("format") = ::megdnn::param::Convolution::Format::NCHW, py::arg("compute_mode") = ::megdnn::param::Convolution::ComputeMode::DEFAULT, py::arg("scope") = {})
    .def_readwrite("mode", &RegionRestrictedConvolution::mode)
    .def_readwrite("pad_h", &RegionRestrictedConvolution::pad_h)
    .def_readwrite("pad_w", &RegionRestrictedConvolution::pad_w)
    .def_readwrite("stride_h", &RegionRestrictedConvolution::stride_h)
    .def_readwrite("stride_w", &RegionRestrictedConvolution::stride_w)
    .def_readwrite("dilate_h", &RegionRestrictedConvolution::dilate_h)
    .def_readwrite("dilate_w", &RegionRestrictedConvolution::dilate_w)
    .def_readwrite("sparse", &RegionRestrictedConvolution::sparse)
    .def_readwrite("format", &RegionRestrictedConvolution::format)
    .def_readwrite("compute_mode", &RegionRestrictedConvolution::compute_mode);

py::class_<RegionRestrictedConvolutionBackwardData, std::shared_ptr<RegionRestrictedConvolutionBackwardData>, OpDef> RegionRestrictedConvolutionBackwardDataInst(m, "RegionRestrictedConvolutionBackwardData");

RegionRestrictedConvolutionBackwardDataInst.attr("Mode") = BatchConvBiasInst.attr("Mode");

RegionRestrictedConvolutionBackwardDataInst.attr("Sparse") = BatchConvBiasInst.attr("Sparse");

RegionRestrictedConvolutionBackwardDataInst.attr("Format") = AdaptivePoolingInst.attr("Format");

RegionRestrictedConvolutionBackwardDataInst.attr("ComputeMode") = BatchConvBiasInst.attr("ComputeMode");

RegionRestrictedConvolutionBackwardDataInst
    .def(py::init<::megdnn::param::Convolution::Mode, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, ::megdnn::param::Convolution::Sparse, ::megdnn::param::Convolution::Format, ::megdnn::param::Convolution::ComputeMode, std::string>(), py::arg("mode") = ::megdnn::param::Convolution::Mode::CROSS_CORRELATION, py::arg("pad_h") = 0, py::arg("pad_w") = 0, py::arg("stride_h") = 1, py::arg("stride_w") = 1, py::arg("dilate_h") = 1, py::arg("dilate_w") = 1, py::arg("sparse") = ::megdnn::param::Convolution::Sparse::DENSE, py::arg("format") = ::megdnn::param::Convolution::Format::NCHW, py::arg("compute_mode") = ::megdnn::param::Convolution::ComputeMode::DEFAULT, py::arg("scope") = {})
    .def_readwrite("mode", &RegionRestrictedConvolutionBackwardData::mode)
    .def_readwrite("pad_h", &RegionRestrictedConvolutionBackwardData::pad_h)
    .def_readwrite("pad_w", &RegionRestrictedConvolutionBackwardData::pad_w)
    .def_readwrite("stride_h", &RegionRestrictedConvolutionBackwardData::stride_h)
    .def_readwrite("stride_w", &RegionRestrictedConvolutionBackwardData::stride_w)
    .def_readwrite("dilate_h", &RegionRestrictedConvolutionBackwardData::dilate_h)
    .def_readwrite("dilate_w", &RegionRestrictedConvolutionBackwardData::dilate_w)
    .def_readwrite("sparse", &RegionRestrictedConvolutionBackwardData::sparse)
    .def_readwrite("format", &RegionRestrictedConvolutionBackwardData::format)
    .def_readwrite("compute_mode", &RegionRestrictedConvolutionBackwardData::compute_mode);

py::class_<Remap, std::shared_ptr<Remap>, OpDef> RemapInst(m, "Remap");

py::enum_<Remap::InterpolationMode>(RemapInst, "InterpolationMode")
    .value("NEAREST", Remap::InterpolationMode::NEAREST)
    .value("LINEAR", Remap::InterpolationMode::LINEAR)
    .value("AREA", Remap::InterpolationMode::AREA)
    .value("CUBIC", Remap::InterpolationMode::CUBIC)
    .value("LANCZOS4", Remap::InterpolationMode::LANCZOS4)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "NEAREST") return Remap::InterpolationMode::NEAREST;
        if (str == "LINEAR") return Remap::InterpolationMode::LINEAR;
        if (str == "AREA") return Remap::InterpolationMode::AREA;
        if (str == "CUBIC") return Remap::InterpolationMode::CUBIC;
        if (str == "LANCZOS4") return Remap::InterpolationMode::LANCZOS4;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, Remap::InterpolationMode>();

py::enum_<Remap::BorderMode>(RemapInst, "BorderMode")
    .value("REPLICATE", Remap::BorderMode::REPLICATE)
    .value("REFLECT", Remap::BorderMode::REFLECT)
    .value("REFLECT_101", Remap::BorderMode::REFLECT_101)
    .value("WRAP", Remap::BorderMode::WRAP)
    .value("CONSTANT", Remap::BorderMode::CONSTANT)
    .value("TRANSPARENT", Remap::BorderMode::TRANSPARENT)
    .value("ISOLATED", Remap::BorderMode::ISOLATED)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "REPLICATE") return Remap::BorderMode::REPLICATE;
        if (str == "REFLECT") return Remap::BorderMode::REFLECT;
        if (str == "REFLECT_101") return Remap::BorderMode::REFLECT_101;
        if (str == "WRAP") return Remap::BorderMode::WRAP;
        if (str == "CONSTANT") return Remap::BorderMode::CONSTANT;
        if (str == "TRANSPARENT") return Remap::BorderMode::TRANSPARENT;
        if (str == "ISOLATED") return Remap::BorderMode::ISOLATED;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, Remap::BorderMode>();

RemapInst.attr("Format") = AdaptivePoolingInst.attr("Format");

RemapInst
    .def(py::init<::megdnn::param::Remap::InterpolationMode, ::megdnn::param::Remap::BorderMode, ::megdnn::param::Remap::Format, float, std::string>(), py::arg("imode") = ::megdnn::param::Remap::InterpolationMode::LINEAR, py::arg("border_type") = ::megdnn::param::Remap::BorderMode::REPLICATE, py::arg("format") = ::megdnn::param::Remap::Format::NHWC, py::arg("scalar") = 0.f, py::arg("scope") = {})
    .def_readwrite("imode", &Remap::imode)
    .def_readwrite("border_type", &Remap::border_type)
    .def_readwrite("format", &Remap::format)
    .def_readwrite("scalar", &Remap::scalar);

py::class_<RemoteRecv, std::shared_ptr<RemoteRecv>, OpDef> RemoteRecvInst(m, "RemoteRecv");

RemoteRecvInst
    .def(py::init<std::string, std::string, uint32_t, uint32_t, ::mgb::CompNode, std::vector<int32_t>, ::megdnn::DType, std::string, std::string>(), py::arg("key"), py::arg("addr"), py::arg("port"), py::arg("rank_from"), py::arg("cn"), py::arg("shape"), py::arg("dtype"), py::arg("backend"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("key", &RemoteRecv::key)
    .def_readwrite("addr", &RemoteRecv::addr)
    .def_readwrite("port", &RemoteRecv::port)
    .def_readwrite("rank_from", &RemoteRecv::rank_from)
    .def_readwrite("cn", &RemoteRecv::cn)
    .def_readwrite("shape", &RemoteRecv::shape)
    .def_readwrite("dtype", &RemoteRecv::dtype)
    .def_readwrite("backend", &RemoteRecv::backend);

py::class_<RemoteSend, std::shared_ptr<RemoteSend>, OpDef> RemoteSendInst(m, "RemoteSend");

RemoteSendInst
    .def(py::init<std::string, std::string, uint32_t, uint32_t, std::string, std::string>(), py::arg("key"), py::arg("addr"), py::arg("port"), py::arg("rank_to"), py::arg("backend"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("key", &RemoteSend::key)
    .def_readwrite("addr", &RemoteSend::addr)
    .def_readwrite("port", &RemoteSend::port)
    .def_readwrite("rank_to", &RemoteSend::rank_to)
    .def_readwrite("backend", &RemoteSend::backend);

py::class_<RemoveAxis, std::shared_ptr<RemoveAxis>, OpDef> RemoveAxisInst(m, "RemoveAxis");

RemoveAxisInst
    .def(py::init<std::vector<int32_t>, std::string>(), py::arg("axis"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("axis", &RemoveAxis::axis);

py::class_<Reshape, std::shared_ptr<Reshape>, OpDef> ReshapeInst(m, "Reshape");

ReshapeInst
    .def(py::init<int32_t, std::vector<int32_t>, std::string>(), py::arg("axis") = ::megdnn::param::OptionalAxisV1::INVALID_AXIS, py::arg("shape"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("axis", &Reshape::axis)
    .def_readwrite("shape", &Reshape::shape);

py::class_<Resize, std::shared_ptr<Resize>, OpDef> ResizeInst(m, "Resize");

ResizeInst.attr("InterpolationMode") = RemapInst.attr("InterpolationMode");

ResizeInst.attr("Format") = AdaptivePoolingInst.attr("Format");

ResizeInst
    .def(py::init<::megdnn::param::Resize::InterpolationMode, ::megdnn::param::Resize::Format, std::string>(), py::arg("imode") = ::megdnn::param::Resize::InterpolationMode::LINEAR, py::arg("format") = ::megdnn::param::Resize::Format::NHWC, py::arg("scope") = {})
    .def_readwrite("imode", &Resize::imode)
    .def_readwrite("format", &Resize::format);

py::class_<Resize3D, std::shared_ptr<Resize3D>, OpDef> Resize3DInst(m, "Resize3D");

Resize3DInst.attr("InterpolationMode") = RemapInst.attr("InterpolationMode");

Resize3DInst.attr("Format") = Convolution3DInst.attr("Format");

Resize3DInst
    .def(py::init<::megdnn::param::Resize3D::InterpolationMode, ::megdnn::param::Resize3D::Format, bool, std::string>(), py::arg("imode") = ::megdnn::param::Resize3D::InterpolationMode::LINEAR, py::arg("format") = ::megdnn::param::Resize3D::Format::NDHWC, py::arg("align_corners") = false, py::arg("scope") = {})
    .def_readwrite("imode", &Resize3D::imode)
    .def_readwrite("format", &Resize3D::format)
    .def_readwrite("align_corners", &Resize3D::align_corners);

py::class_<Rotate, std::shared_ptr<Rotate>, OpDef> RotateInst(m, "Rotate");

RotateInst
    .def(py::init<bool, std::string>(), py::arg("clockwise") = true, py::arg("scope") = {})
    .def_readwrite("clockwise", &Rotate::clockwise);

py::class_<SVD, std::shared_ptr<SVD>, OpDef> SVDInst(m, "SVD");

SVDInst
    .def(py::init<bool, bool, std::string>(), py::arg("full_matrices") = false, py::arg("compute_uv") = true, py::arg("scope") = {})
    .def_readwrite("full_matrices", &SVD::full_matrices)
    .def_readwrite("compute_uv", &SVD::compute_uv);

py::class_<SetMeshIndexing, std::shared_ptr<SetMeshIndexing>, OpDef> SetMeshIndexingInst(m, "SetMeshIndexing");

SetMeshIndexingInst
    .def(py::init<std::vector<std::tuple<int8_t, bool, bool, bool, bool>>, std::string>(), py::arg("items"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("items", &SetMeshIndexing::items);

py::class_<SetSubtensor, std::shared_ptr<SetSubtensor>, OpDef> SetSubtensorInst(m, "SetSubtensor");

SetSubtensorInst
    .def(py::init<std::vector<std::tuple<int8_t, bool, bool, bool, bool>>, std::string>(), py::arg("items"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("items", &SetSubtensor::items);

py::class_<ShuffleRNG, std::shared_ptr<ShuffleRNG>, OpDef> ShuffleRNGInst(m, "ShuffleRNG");

ShuffleRNGInst
    .def(py::init<uint64_t, size_t, std::string>(), py::arg("seed") = 0, py::arg("handle"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("seed", &ShuffleRNG::seed)
    .def_readwrite("handle", &ShuffleRNG::handle);

py::class_<SlidingWindowTranspose, std::shared_ptr<SlidingWindowTranspose>, OpDef> SlidingWindowTransposeInst(m, "SlidingWindowTranspose");

SlidingWindowTransposeInst
    .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, std::string>(), py::arg("out_h") = 0, py::arg("out_w") = 0, py::arg("pad_h") = 0, py::arg("pad_w") = 0, py::arg("stride_h") = 1, py::arg("stride_w") = 1, py::arg("dilate_h") = 1, py::arg("dilate_w") = 1, py::arg("window_h") = 3, py::arg("window_w") = 3, py::arg("scope") = {})
    .def_readwrite("out_h", &SlidingWindowTranspose::out_h)
    .def_readwrite("out_w", &SlidingWindowTranspose::out_w)
    .def_readwrite("pad_h", &SlidingWindowTranspose::pad_h)
    .def_readwrite("pad_w", &SlidingWindowTranspose::pad_w)
    .def_readwrite("stride_h", &SlidingWindowTranspose::stride_h)
    .def_readwrite("stride_w", &SlidingWindowTranspose::stride_w)
    .def_readwrite("dilate_h", &SlidingWindowTranspose::dilate_h)
    .def_readwrite("dilate_w", &SlidingWindowTranspose::dilate_w)
    .def_readwrite("window_h", &SlidingWindowTranspose::window_h)
    .def_readwrite("window_w", &SlidingWindowTranspose::window_w);

py::class_<Softmax, std::shared_ptr<Softmax>, OpDef> SoftmaxInst(m, "Softmax");

SoftmaxInst
    .def(py::init<int32_t, std::string>(), py::arg("axis") = -1, py::arg("scope") = {})
    .def_readwrite("axis", &Softmax::axis);

py::class_<Split, std::shared_ptr<Split>, OpDef> SplitInst(m, "Split");

SplitInst
    .def(py::init<int32_t, int32_t, std::string>(), py::arg("axis"), py::arg("nsections"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("axis", &Split::axis)
    .def_readwrite("nsections", &Split::nsections);

py::class_<Stack, std::shared_ptr<Stack>, OpDef> StackInst(m, "Stack");

StackInst
    .def(py::init<int32_t, ::mgb::CompNode, std::string>(), py::arg("axis") = 0, py::arg("comp_node"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("axis", &Stack::axis)
    .def_readwrite("comp_node", &Stack::comp_node);

py::class_<Subtensor, std::shared_ptr<Subtensor>, OpDef> SubtensorInst(m, "Subtensor");

SubtensorInst
    .def(py::init<std::vector<std::tuple<int8_t, bool, bool, bool, bool>>, std::vector<std::tuple<int32_t, int32_t, int32_t, int32_t>>, std::string>(), py::arg("items"), py::arg("slice_items"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("items", &Subtensor::items)
    .def_readwrite("slice_items", &Subtensor::slice_items);

py::class_<TQT, std::shared_ptr<TQT>, OpDef> TQTInst(m, "TQT");

TQTInst
    .def(py::init<int32_t, int32_t, std::string>(), py::arg("qmin") = -2147483648, py::arg("qmax") = 2147483647, py::arg("scope") = {})
    .def_readwrite("qmin", &TQT::qmin)
    .def_readwrite("qmax", &TQT::qmax);

py::class_<TensorRTRuntime, std::shared_ptr<TensorRTRuntime>, OpDef> TensorRTRuntimeInst(m, "TensorRTRuntime");

TensorRTRuntimeInst
    .def(py::init<std::string, size_t, std::string>(), py::arg("buf"), py::arg("buf_size"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("buf", &TensorRTRuntime::buf)
    .def_readwrite("buf_size", &TensorRTRuntime::buf_size);

py::class_<TopK, std::shared_ptr<TopK>, OpDef> TopKInst(m, "TopK");

py::enum_<TopK::Mode>(TopKInst, "Mode")
    .value("KTH_ONLY", TopK::Mode::KTH_ONLY)
    .value("VALUE_IDX_NOSORT", TopK::Mode::VALUE_IDX_NOSORT)
    .value("VALUE_IDX_SORTED", TopK::Mode::VALUE_IDX_SORTED)
    .def(py::init([](const std::string& in) {
        auto&& str = normalize_enum(in);
        if (str == "KTH_ONLY") return TopK::Mode::KTH_ONLY;
        if (str == "VALUE_IDX_NOSORT") return TopK::Mode::VALUE_IDX_NOSORT;
        if (str == "VALUE_IDX_SORTED") return TopK::Mode::VALUE_IDX_SORTED;
        throw py::cast_error("invalid enum value " + in);
    }));
py::implicitly_convertible<std::string, TopK::Mode>();

TopKInst
    .def(py::init<::megdnn::param::TopK::Mode, std::string>(), py::arg("mode") = ::megdnn::param::TopK::Mode::KTH_ONLY, py::arg("scope") = {})
    .def_readwrite("mode", &TopK::mode);

py::class_<TypeCvt, std::shared_ptr<TypeCvt>, OpDef> TypeCvtInst(m, "TypeCvt");

TypeCvtInst
    .def(py::init<::megdnn::DType, std::string>(), py::arg("dtype"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("dtype", &TypeCvt::dtype);

py::class_<UniformRNG, std::shared_ptr<UniformRNG>, OpDef> UniformRNGInst(m, "UniformRNG");

UniformRNGInst
    .def(py::init<uint64_t, ::megdnn::DType, size_t, std::string>(), py::arg("seed") = 0, py::arg("dtype") = megdnn::DType::from_enum(megdnn::DTypeEnum::Float32), py::arg("handle"), py::arg("scope") = {})
    .def(py::init<>())
    .def_readwrite("seed", &UniformRNG::seed)
    .def_readwrite("dtype", &UniformRNG::dtype)
    .def_readwrite("handle", &UniformRNG::handle);

py::class_<WarpAffine, std::shared_ptr<WarpAffine>, OpDef> WarpAffineInst(m, "WarpAffine");

WarpAffineInst.attr("InterpolationMode") = RemapInst.attr("InterpolationMode");

WarpAffineInst.attr("BorderMode") = RemapInst.attr("BorderMode");

WarpAffineInst.attr("Format") = AdaptivePoolingInst.attr("Format");

WarpAffineInst
    .def(py::init<::megdnn::param::WarpAffine::InterpolationMode, ::megdnn::param::WarpAffine::BorderMode, float, ::megdnn::param::WarpAffine::Format, std::string>(), py::arg("imode") = ::megdnn::param::WarpAffine::InterpolationMode::LINEAR, py::arg("border_mode") = ::megdnn::param::WarpAffine::BorderMode::REPLICATE, py::arg("border_val") = .0f, py::arg("format") = ::megdnn::param::WarpAffine::Format::NHWC, py::arg("scope") = {})
    .def_readwrite("imode", &WarpAffine::imode)
    .def_readwrite("border_mode", &WarpAffine::border_mode)
    .def_readwrite("border_val", &WarpAffine::border_val)
    .def_readwrite("format", &WarpAffine::format);

py::class_<WarpPerspective, std::shared_ptr<WarpPerspective>, OpDef> WarpPerspectiveInst(m, "WarpPerspective");

WarpPerspectiveInst.attr("InterpolationMode") = RemapInst.attr("InterpolationMode");

WarpPerspectiveInst.attr("BorderMode") = RemapInst.attr("BorderMode");

WarpPerspectiveInst.attr("Format") = AdaptivePoolingInst.attr("Format");

WarpPerspectiveInst
    .def(py::init<::megdnn::param::WarpPerspective::InterpolationMode, ::megdnn::param::WarpPerspective::BorderMode, ::megdnn::param::WarpPerspective::Format, float, std::string>(), py::arg("imode") = ::megdnn::param::WarpPerspective::InterpolationMode::LINEAR, py::arg("bmode") = ::megdnn::param::WarpPerspective::BorderMode::REPLICATE, py::arg("format") = ::megdnn::param::WarpPerspective::Format::NCHW, py::arg("border_val") = .0f, py::arg("scope") = {})
    .def_readwrite("imode", &WarpPerspective::imode)
    .def_readwrite("bmode", &WarpPerspective::bmode)
    .def_readwrite("format", &WarpPerspective::format)
    .def_readwrite("border_val", &WarpPerspective::border_val);

py::class_<WarpPerspectiveBackwardData, std::shared_ptr<WarpPerspectiveBackwardData>, OpDef> WarpPerspectiveBackwardDataInst(m, "WarpPerspectiveBackwardData");

WarpPerspectiveBackwardDataInst.attr("InterpolationMode") = RemapInst.attr("InterpolationMode");

WarpPerspectiveBackwardDataInst.attr("BorderMode") = RemapInst.attr("BorderMode");

WarpPerspectiveBackwardDataInst.attr("Format") = AdaptivePoolingInst.attr("Format");

WarpPerspectiveBackwardDataInst
    .def(py::init<::megdnn::param::WarpPerspective::InterpolationMode, ::megdnn::param::WarpPerspective::BorderMode, ::megdnn::param::WarpPerspective::Format, float, std::string>(), py::arg("imode") = ::megdnn::param::WarpPerspective::InterpolationMode::LINEAR, py::arg("bmode") = ::megdnn::param::WarpPerspective::BorderMode::REPLICATE, py::arg("format") = ::megdnn::param::WarpPerspective::Format::NCHW, py::arg("border_val") = .0f, py::arg("scope") = {})
    .def_readwrite("imode", &WarpPerspectiveBackwardData::imode)
    .def_readwrite("bmode", &WarpPerspectiveBackwardData::bmode)
    .def_readwrite("format", &WarpPerspectiveBackwardData::format)
    .def_readwrite("border_val", &WarpPerspectiveBackwardData::border_val);

py::class_<WarpPerspectiveBackwardMat, std::shared_ptr<WarpPerspectiveBackwardMat>, OpDef> WarpPerspectiveBackwardMatInst(m, "WarpPerspectiveBackwardMat");

WarpPerspectiveBackwardMatInst.attr("InterpolationMode") = RemapInst.attr("InterpolationMode");

WarpPerspectiveBackwardMatInst.attr("BorderMode") = RemapInst.attr("BorderMode");

WarpPerspectiveBackwardMatInst.attr("Format") = AdaptivePoolingInst.attr("Format");

WarpPerspectiveBackwardMatInst
    .def(py::init<::megdnn::param::WarpPerspective::InterpolationMode, ::megdnn::param::WarpPerspective::BorderMode, ::megdnn::param::WarpPerspective::Format, float, std::string>(), py::arg("imode") = ::megdnn::param::WarpPerspective::InterpolationMode::LINEAR, py::arg("bmode") = ::megdnn::param::WarpPerspective::BorderMode::REPLICATE, py::arg("format") = ::megdnn::param::WarpPerspective::Format::NCHW, py::arg("border_val") = .0f, py::arg("scope") = {})
    .def_readwrite("imode", &WarpPerspectiveBackwardMat::imode)
    .def_readwrite("bmode", &WarpPerspectiveBackwardMat::bmode)
    .def_readwrite("format", &WarpPerspectiveBackwardMat::format)
    .def_readwrite("border_val", &WarpPerspectiveBackwardMat::border_val);

py::class_<Where, std::shared_ptr<Where>, OpDef> WhereInst(m, "Where");

WhereInst
    .def(py::init<>());

py::class_<WhereBackward, std::shared_ptr<WhereBackward>, OpDef> WhereBackwardInst(m, "WhereBackward");

WhereBackwardInst
    .def(py::init<>());

// clang-format on
