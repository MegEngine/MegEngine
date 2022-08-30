// clang-format off
class AdaptivePooling : public OpDefImplBase<AdaptivePooling> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using Mode = ::megdnn::param::AdaptivePooling::Mode;
    using Format = ::megdnn::param::AdaptivePooling::Format;
    Mode mode = ::megdnn::param::AdaptivePooling::Mode::MAX;
    Format format = ::megdnn::param::AdaptivePooling::Format::NCHW;
    std::vector<int32_t> shape;
    AdaptivePooling() = default;
    AdaptivePooling(Mode mode_, Format format_, std::vector<int32_t> shape_, std::string scope_ = {}): mode(mode_), format(format_), shape(shape_) { set_scope(scope_); }
    AdaptivePooling(::megdnn::param::AdaptivePooling packed_param_0, std::vector<int32_t> shape_): mode(packed_param_0.mode), format(packed_param_0.format), shape(shape_) {}
    ::megdnn::param::AdaptivePooling param() const {
        return {mode, format};
    }
};

class AddAxis : public OpDefImplBase<AddAxis> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    std::vector<int32_t> axis;
    AddAxis() = default;
    AddAxis(std::vector<int32_t> axis_, std::string scope_ = {}): axis(axis_) { set_scope(scope_); }
};

class Argmax : public OpDefImplBase<Argmax> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    int32_t axis = 0;
    Argmax() = default;
    Argmax(int32_t axis_, std::string scope_ = {}): axis(axis_) { set_scope(scope_); }
    Argmax(::megdnn::param::Axis packed_param_0): axis(packed_param_0.axis) {}
    ::megdnn::param::Axis param() const {
        return {axis};
    }
};

class Argmin : public OpDefImplBase<Argmin> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    int32_t axis = 0;
    Argmin() = default;
    Argmin(int32_t axis_, std::string scope_ = {}): axis(axis_) { set_scope(scope_); }
    Argmin(::megdnn::param::Axis packed_param_0): axis(packed_param_0.axis) {}
    ::megdnn::param::Axis param() const {
        return {axis};
    }
};

class Argsort : public OpDefImplBase<Argsort> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using Order = ::megdnn::param::Argsort::Order;
    Order order = ::megdnn::param::Argsort::Order::ASCENDING;
    Argsort() = default;
    Argsort(Order order_, std::string scope_ = {}): order(order_) { set_scope(scope_); }
    Argsort(::megdnn::param::Argsort packed_param_0): order(packed_param_0.order) {}
    ::megdnn::param::Argsort param() const {
        return {order};
    }
};

class AssertEqual : public OpDefImplBase<AssertEqual> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    float maxerr = 0.0001;
    bool verbose = false;
    AssertEqual() = default;
    AssertEqual(float maxerr_, bool verbose_, std::string scope_ = {}): maxerr(maxerr_), verbose(verbose_) { set_scope(scope_); }
    AssertEqual(::megdnn::param::AssertEqual packed_param_0): maxerr(packed_param_0.maxerr), verbose(packed_param_0.verbose) {}
    ::megdnn::param::AssertEqual param() const {
        return {maxerr, verbose};
    }
};

class AtlasRuntime : public OpDefImplBase<AtlasRuntime> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    std::string buf;
    size_t buf_size;
    AtlasRuntime() = default;
    AtlasRuntime(std::string buf_, size_t buf_size_, std::string scope_ = {}): buf(buf_), buf_size(buf_size_) { set_scope(scope_); }
};

class Barrier : public OpDefImplBase<Barrier> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    ::mgb::CompNode comp_node;
    uint32_t nr_outputs;
    Barrier() = default;
    Barrier(::mgb::CompNode comp_node_, uint32_t nr_outputs_, std::string scope_ = {}): comp_node(comp_node_), nr_outputs(nr_outputs_) { set_scope(scope_); }
};

class BatchConvBias : public OpDefImplBase<BatchConvBias> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using NonlineMode = ::megdnn::param::BatchConvBias::NonlineMode;
    using Mode = ::megdnn::param::BatchConvBias::Mode;
    using Sparse = ::megdnn::param::BatchConvBias::Sparse;
    using Format = ::megdnn::param::BatchConvBias::Format;
    using ComputeMode = ::megdnn::param::BatchConvBias::ComputeMode;
    using Strategy = ::megdnn::param::ExecutionPolicy::Strategy;
    NonlineMode nonlineMode = ::megdnn::param::BatchConvBias::NonlineMode::IDENTITY;
    Mode mode = ::megdnn::param::BatchConvBias::Mode::CROSS_CORRELATION;
    uint32_t pad_h = 0;
    uint32_t pad_w = 0;
    uint32_t stride_h = 1;
    uint32_t stride_w = 1;
    uint32_t dilate_h = 1;
    uint32_t dilate_w = 1;
    Sparse sparse = ::megdnn::param::BatchConvBias::Sparse::DENSE;
    Format format = ::megdnn::param::BatchConvBias::Format::NCHW;
    ComputeMode compute_mode = ::megdnn::param::BatchConvBias::ComputeMode::DEFAULT;
    Strategy strategy = static_cast<::megdnn::param::ExecutionPolicy::Strategy>(1);
    uint64_t workspace_limit = 18446744073709551615ull;
    ::megdnn::DType dtype;
    BatchConvBias() = default;
    BatchConvBias(NonlineMode nonlineMode_, Mode mode_, uint32_t pad_h_, uint32_t pad_w_, uint32_t stride_h_, uint32_t stride_w_, uint32_t dilate_h_, uint32_t dilate_w_, Sparse sparse_, Format format_, ComputeMode compute_mode_, Strategy strategy_, uint64_t workspace_limit_, ::megdnn::DType dtype_, std::string scope_ = {}): nonlineMode(nonlineMode_), mode(mode_), pad_h(pad_h_), pad_w(pad_w_), stride_h(stride_h_), stride_w(stride_w_), dilate_h(dilate_h_), dilate_w(dilate_w_), sparse(sparse_), format(format_), compute_mode(compute_mode_), strategy(strategy_), workspace_limit(workspace_limit_), dtype(dtype_) {
        set_scope(scope_);
        mgb_assert(static_cast<uint32_t>(strategy) <= uint32_t(8));
    }
    BatchConvBias(::megdnn::param::BatchConvBias packed_param_0, ::megdnn::param::ExecutionPolicy packed_param_1, ::megdnn::DType dtype_): nonlineMode(packed_param_0.nonlineMode), mode(packed_param_0.mode), pad_h(packed_param_0.pad_h), pad_w(packed_param_0.pad_w), stride_h(packed_param_0.stride_h), stride_w(packed_param_0.stride_w), dilate_h(packed_param_0.dilate_h), dilate_w(packed_param_0.dilate_w), sparse(packed_param_0.sparse), format(packed_param_0.format), compute_mode(packed_param_0.compute_mode), strategy(packed_param_1.strategy), workspace_limit(packed_param_1.workspace_limit), dtype(dtype_) {
        mgb_assert(static_cast<uint32_t>(strategy) <= uint32_t(8));
    }
    ::megdnn::param::BatchConvBias param() const {
        return {nonlineMode, mode, pad_h, pad_w, stride_h, stride_w, dilate_h, dilate_w, sparse, format, compute_mode};
    }
    ::megdnn::param::ExecutionPolicy policy() const {
        return {strategy, workspace_limit};
    }
};

class BatchNorm : public OpDefImplBase<BatchNorm> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using ParamDim = ::megdnn::param::BN::ParamDim;
    using FwdMode = ::megdnn::param::BN::FwdMode;
    ParamDim param_dim = ::megdnn::param::BN::ParamDim::DIM_11HW;
    FwdMode fwd_mode = ::megdnn::param::BN::FwdMode::TRAINING;
    double epsilon = 1e-4f;
    double avg_factor = 1.f;
    float scale = 1.f;
    float bias = 0.f;
    BatchNorm() = default;
    BatchNorm(ParamDim param_dim_, FwdMode fwd_mode_, double epsilon_, double avg_factor_, float scale_, float bias_, std::string scope_ = {}): param_dim(param_dim_), fwd_mode(fwd_mode_), epsilon(epsilon_), avg_factor(avg_factor_), scale(scale_), bias(bias_) { set_scope(scope_); }
    BatchNorm(::megdnn::param::BN packed_param_0): param_dim(packed_param_0.param_dim), fwd_mode(packed_param_0.fwd_mode), epsilon(packed_param_0.epsilon), avg_factor(packed_param_0.avg_factor), scale(packed_param_0.scale), bias(packed_param_0.bias) {}
    ::megdnn::param::BN param() const {
        return {param_dim, fwd_mode, epsilon, avg_factor, scale, bias};
    }
};

class BatchNormBackward : public OpDefImplBase<BatchNormBackward> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using ParamDim = ::megdnn::param::BN::ParamDim;
    using FwdMode = ::megdnn::param::BN::FwdMode;
    ParamDim param_dim = ::megdnn::param::BN::ParamDim::DIM_11HW;
    FwdMode fwd_mode = ::megdnn::param::BN::FwdMode::TRAINING;
    double epsilon = 1e-4f;
    double avg_factor = 1.f;
    float scale = 1.f;
    float bias = 0.f;
    BatchNormBackward() = default;
    BatchNormBackward(ParamDim param_dim_, FwdMode fwd_mode_, double epsilon_, double avg_factor_, float scale_, float bias_, std::string scope_ = {}): param_dim(param_dim_), fwd_mode(fwd_mode_), epsilon(epsilon_), avg_factor(avg_factor_), scale(scale_), bias(bias_) { set_scope(scope_); }
    BatchNormBackward(::megdnn::param::BN packed_param_0): param_dim(packed_param_0.param_dim), fwd_mode(packed_param_0.fwd_mode), epsilon(packed_param_0.epsilon), avg_factor(packed_param_0.avg_factor), scale(packed_param_0.scale), bias(packed_param_0.bias) {}
    ::megdnn::param::BN param() const {
        return {param_dim, fwd_mode, epsilon, avg_factor, scale, bias};
    }
};

class BatchedIncrMeshIndexing : public OpDefImplBase<BatchedIncrMeshIndexing> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    std::vector<std::tuple<int8_t, bool, bool, bool, bool>> items;
    BatchedIncrMeshIndexing() = default;
    BatchedIncrMeshIndexing(std::vector<std::tuple<int8_t, bool, bool, bool, bool>> items_, std::string scope_ = {}): items(items_) { set_scope(scope_); }
};

class BatchedMatrixMul : public OpDefImplBase<BatchedMatrixMul> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using ComputeMode = ::megdnn::param::MatrixMul::ComputeMode;
    using Format = ::megdnn::param::MatrixMul::Format;
    using Strategy = ::megdnn::param::ExecutionPolicy::Strategy;
    bool transposeA = false;
    bool transposeB = false;
    ComputeMode compute_mode = ::megdnn::param::MatrixMul::ComputeMode::DEFAULT;
    Format format = ::megdnn::param::MatrixMul::Format::DEFAULT;
    Strategy strategy = static_cast<::megdnn::param::ExecutionPolicy::Strategy>(1);
    uint64_t workspace_limit = 18446744073709551615ull;
    uint32_t dimA;
    uint32_t dimB;
    BatchedMatrixMul() = default;
    BatchedMatrixMul(bool transposeA_, bool transposeB_, ComputeMode compute_mode_, Format format_, Strategy strategy_, uint64_t workspace_limit_, uint32_t dimA_, uint32_t dimB_, std::string scope_ = {}): transposeA(transposeA_), transposeB(transposeB_), compute_mode(compute_mode_), format(format_), strategy(strategy_), workspace_limit(workspace_limit_), dimA(dimA_), dimB(dimB_) {
        set_scope(scope_);
        mgb_assert(static_cast<uint32_t>(strategy) <= uint32_t(8));
    }
    BatchedMatrixMul(::megdnn::param::MatrixMul packed_param_0, ::megdnn::param::ExecutionPolicy packed_param_1, uint32_t dimA_, uint32_t dimB_): transposeA(packed_param_0.transposeA), transposeB(packed_param_0.transposeB), compute_mode(packed_param_0.compute_mode), format(packed_param_0.format), strategy(packed_param_1.strategy), workspace_limit(packed_param_1.workspace_limit), dimA(dimA_), dimB(dimB_) {
        mgb_assert(static_cast<uint32_t>(strategy) <= uint32_t(8));
    }
    ::megdnn::param::MatrixMul param() const {
        return {transposeA, transposeB, compute_mode, format};
    }
    ::megdnn::param::ExecutionPolicy policy() const {
        return {strategy, workspace_limit};
    }
};

class BatchedMeshIndexing : public OpDefImplBase<BatchedMeshIndexing> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    std::vector<std::tuple<int8_t, bool, bool, bool, bool>> items;
    BatchedMeshIndexing() = default;
    BatchedMeshIndexing(std::vector<std::tuple<int8_t, bool, bool, bool, bool>> items_, std::string scope_ = {}): items(items_) { set_scope(scope_); }
};

class BatchedSetMeshIndexing : public OpDefImplBase<BatchedSetMeshIndexing> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    std::vector<std::tuple<int8_t, bool, bool, bool, bool>> items;
    BatchedSetMeshIndexing() = default;
    BatchedSetMeshIndexing(std::vector<std::tuple<int8_t, bool, bool, bool, bool>> items_, std::string scope_ = {}): items(items_) { set_scope(scope_); }
};

class BetaRNG : public OpDefImplBase<BetaRNG> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    uint64_t seed = 0;
    size_t handle;
    BetaRNG() = default;
    BetaRNG(uint64_t seed_, size_t handle_, std::string scope_ = {}): seed(seed_), handle(handle_) { set_scope(scope_); }
    BetaRNG(::megdnn::param::BetaRNG packed_param_0, size_t handle_): seed(packed_param_0.seed), handle(handle_) {}
    ::megdnn::param::BetaRNG param() const {
        return {seed};
    }
};

class Borrow : public OpDefImplBase<Borrow> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    ::mgb::CompNode comp_node;
    Borrow() = default;
    Borrow(::mgb::CompNode comp_node_, std::string scope_ = {}): comp_node(comp_node_) { set_scope(scope_); }
};

class Broadcast : public OpDefImplBase<Broadcast> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    std::vector<int32_t> shape;
    Broadcast() = default;
    Broadcast(std::vector<int32_t> shape_, std::string scope_ = {}): shape(shape_) { set_scope(scope_); }
    Broadcast(::megdnn::param::Empty, std::vector<int32_t> shape_): shape(shape_) {}
    ::megdnn::param::Empty param() const {
        return {};
    }
};

class CambriconRuntime : public OpDefImplBase<CambriconRuntime> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    std::string buf;
    size_t buf_size;
    std::string symbol;
    bool tensor_dim_mutable;
    CambriconRuntime() = default;
    CambriconRuntime(std::string buf_, size_t buf_size_, std::string symbol_, bool tensor_dim_mutable_, std::string scope_ = {}): buf(buf_), buf_size(buf_size_), symbol(symbol_), tensor_dim_mutable(tensor_dim_mutable_) { set_scope(scope_); }
};

class CheckNonFinite : public OpDefImplBase<CheckNonFinite> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    float scale = 1.0;
    CheckNonFinite() = default;
    CheckNonFinite(float scale_, std::string scope_ = {}): scale(scale_) { set_scope(scope_); }
    CheckNonFinite(::megdnn::param::CheckNonFinite packed_param_0): scale(packed_param_0.scale) {}
    ::megdnn::param::CheckNonFinite param() const {
        return {scale};
    }
};

class CollectiveComm : public OpDefImplBase<CollectiveComm> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using Mode = ::megdnn::param::CollectiveComm::Mode;
    Mode mode = ::megdnn::param::CollectiveComm::Mode::REDUCE_SUM;
    std::string key;
    uint32_t nr_devices;
    uint32_t rank;
    bool is_root;
    bool local_grad;
    std::string addr;
    uint32_t port;
    ::megdnn::DType dtype;
    std::string backend;
    std::string comp_node;
    CollectiveComm() = default;
    CollectiveComm(Mode mode_, std::string key_, uint32_t nr_devices_, uint32_t rank_, bool is_root_, bool local_grad_, std::string addr_, uint32_t port_, ::megdnn::DType dtype_, std::string backend_, std::string comp_node_, std::string scope_ = {}): mode(mode_), key(key_), nr_devices(nr_devices_), rank(rank_), is_root(is_root_), local_grad(local_grad_), addr(addr_), port(port_), dtype(dtype_), backend(backend_), comp_node(comp_node_) { set_scope(scope_); }
    CollectiveComm(::megdnn::param::CollectiveComm packed_param_0, std::string key_, uint32_t nr_devices_, uint32_t rank_, bool is_root_, bool local_grad_, std::string addr_, uint32_t port_, ::megdnn::DType dtype_, std::string backend_, std::string comp_node_): mode(packed_param_0.mode), key(key_), nr_devices(nr_devices_), rank(rank_), is_root(is_root_), local_grad(local_grad_), addr(addr_), port(port_), dtype(dtype_), backend(backend_), comp_node(comp_node_) {}
    ::megdnn::param::CollectiveComm param() const {
        return {mode};
    }
};

class Concat : public OpDefImplBase<Concat> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    int32_t axis = 0;
    ::mgb::CompNode comp_node;
    Concat() = default;
    Concat(int32_t axis_, ::mgb::CompNode comp_node_, std::string scope_ = {}): axis(axis_), comp_node(comp_node_) { set_scope(scope_); }
    Concat(::megdnn::param::Axis packed_param_0, ::mgb::CompNode comp_node_): axis(packed_param_0.axis), comp_node(comp_node_) {}
    ::megdnn::param::Axis param() const {
        return {axis};
    }
};

class CondTake : public OpDefImplBase<CondTake> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    CondTake() = default;
};

class ConvBias : public OpDefImplBase<ConvBias> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using NonlineMode = ::megdnn::param::ConvBias::NonlineMode;
    using Mode = ::megdnn::param::ConvBias::Mode;
    using Sparse = ::megdnn::param::ConvBias::Sparse;
    using Format = ::megdnn::param::ConvBias::Format;
    using ComputeMode = ::megdnn::param::ConvBias::ComputeMode;
    using Strategy = ::megdnn::param::ExecutionPolicy::Strategy;
    NonlineMode nonlineMode = ::megdnn::param::ConvBias::NonlineMode::IDENTITY;
    Mode mode = ::megdnn::param::ConvBias::Mode::CROSS_CORRELATION;
    Sparse sparse = ::megdnn::param::ConvBias::Sparse::DENSE;
    Format format = ::megdnn::param::ConvBias::Format::NCHW;
    uint32_t pad_h = 0;
    uint32_t pad_w = 0;
    uint32_t stride_h = 1;
    uint32_t stride_w = 1;
    uint32_t dilate_h = 1;
    uint32_t dilate_w = 1;
    ComputeMode compute_mode = ::megdnn::param::ConvBias::ComputeMode::DEFAULT;
    Strategy strategy = static_cast<::megdnn::param::ExecutionPolicy::Strategy>(1);
    uint64_t workspace_limit = 18446744073709551615ull;
    ::megdnn::DType dtype;
    ConvBias() = default;
    ConvBias(NonlineMode nonlineMode_, Mode mode_, Sparse sparse_, Format format_, uint32_t pad_h_, uint32_t pad_w_, uint32_t stride_h_, uint32_t stride_w_, uint32_t dilate_h_, uint32_t dilate_w_, ComputeMode compute_mode_, Strategy strategy_, uint64_t workspace_limit_, ::megdnn::DType dtype_, std::string scope_ = {}): nonlineMode(nonlineMode_), mode(mode_), sparse(sparse_), format(format_), pad_h(pad_h_), pad_w(pad_w_), stride_h(stride_h_), stride_w(stride_w_), dilate_h(dilate_h_), dilate_w(dilate_w_), compute_mode(compute_mode_), strategy(strategy_), workspace_limit(workspace_limit_), dtype(dtype_) {
        set_scope(scope_);
        mgb_assert(static_cast<uint32_t>(strategy) <= uint32_t(8));
    }
    ConvBias(::megdnn::param::ConvBias packed_param_0, ::megdnn::param::ExecutionPolicy packed_param_1, ::megdnn::DType dtype_): nonlineMode(packed_param_0.nonlineMode), mode(packed_param_0.mode), sparse(packed_param_0.sparse), format(packed_param_0.format), pad_h(packed_param_0.pad_h), pad_w(packed_param_0.pad_w), stride_h(packed_param_0.stride_h), stride_w(packed_param_0.stride_w), dilate_h(packed_param_0.dilate_h), dilate_w(packed_param_0.dilate_w), compute_mode(packed_param_0.compute_mode), strategy(packed_param_1.strategy), workspace_limit(packed_param_1.workspace_limit), dtype(dtype_) {
        mgb_assert(static_cast<uint32_t>(strategy) <= uint32_t(8));
    }
    ::megdnn::param::ConvBias param() const {
        return {nonlineMode, mode, sparse, format, pad_h, pad_w, stride_h, stride_w, dilate_h, dilate_w, compute_mode};
    }
    ::megdnn::param::ExecutionPolicy policy() const {
        return {strategy, workspace_limit};
    }
};

class Convolution : public OpDefImplBase<Convolution> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using Mode = ::megdnn::param::Convolution::Mode;
    using Sparse = ::megdnn::param::Convolution::Sparse;
    using Format = ::megdnn::param::Convolution::Format;
    using ComputeMode = ::megdnn::param::Convolution::ComputeMode;
    using Strategy = ::megdnn::param::ExecutionPolicy::Strategy;
    Mode mode = ::megdnn::param::Convolution::Mode::CROSS_CORRELATION;
    uint32_t pad_h = 0;
    uint32_t pad_w = 0;
    uint32_t stride_h = 1;
    uint32_t stride_w = 1;
    uint32_t dilate_h = 1;
    uint32_t dilate_w = 1;
    Sparse sparse = ::megdnn::param::Convolution::Sparse::DENSE;
    Format format = ::megdnn::param::Convolution::Format::NCHW;
    ComputeMode compute_mode = ::megdnn::param::Convolution::ComputeMode::DEFAULT;
    Strategy strategy = static_cast<::megdnn::param::ExecutionPolicy::Strategy>(1);
    uint64_t workspace_limit = 18446744073709551615ull;
    Convolution() = default;
    Convolution(Mode mode_, uint32_t pad_h_, uint32_t pad_w_, uint32_t stride_h_, uint32_t stride_w_, uint32_t dilate_h_, uint32_t dilate_w_, Sparse sparse_, Format format_, ComputeMode compute_mode_, Strategy strategy_, uint64_t workspace_limit_, std::string scope_ = {}): mode(mode_), pad_h(pad_h_), pad_w(pad_w_), stride_h(stride_h_), stride_w(stride_w_), dilate_h(dilate_h_), dilate_w(dilate_w_), sparse(sparse_), format(format_), compute_mode(compute_mode_), strategy(strategy_), workspace_limit(workspace_limit_) {
        set_scope(scope_);
        mgb_assert(static_cast<uint32_t>(strategy) <= uint32_t(8));
    }
    Convolution(::megdnn::param::Convolution packed_param_0, ::megdnn::param::ExecutionPolicy packed_param_1): mode(packed_param_0.mode), pad_h(packed_param_0.pad_h), pad_w(packed_param_0.pad_w), stride_h(packed_param_0.stride_h), stride_w(packed_param_0.stride_w), dilate_h(packed_param_0.dilate_h), dilate_w(packed_param_0.dilate_w), sparse(packed_param_0.sparse), format(packed_param_0.format), compute_mode(packed_param_0.compute_mode), strategy(packed_param_1.strategy), workspace_limit(packed_param_1.workspace_limit) {
        mgb_assert(static_cast<uint32_t>(strategy) <= uint32_t(8));
    }
    ::megdnn::param::Convolution param() const {
        return {mode, pad_h, pad_w, stride_h, stride_w, dilate_h, dilate_w, sparse, format, compute_mode};
    }
    ::megdnn::param::ExecutionPolicy policy() const {
        return {strategy, workspace_limit};
    }
};

class Convolution3D : public OpDefImplBase<Convolution3D> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using Mode = ::megdnn::param::Convolution3D::Mode;
    using Sparse = ::megdnn::param::Convolution3D::Sparse;
    using DataType = ::megdnn::param::Convolution3D::DataType;
    using Format = ::megdnn::param::Convolution3D::Format;
    using Strategy = ::megdnn::param::ExecutionPolicy::Strategy;
    Mode mode = ::megdnn::param::Convolution3D::Mode::CROSS_CORRELATION;
    uint32_t pad_d = 0;
    uint32_t pad_h = 0;
    uint32_t pad_w = 0;
    uint32_t stride_d = 1;
    uint32_t stride_h = 1;
    uint32_t stride_w = 1;
    uint32_t dilate_d = 1;
    uint32_t dilate_h = 1;
    uint32_t dilate_w = 1;
    Sparse sparse = ::megdnn::param::Convolution3D::Sparse::DENSE;
    DataType data_type = ::megdnn::param::Convolution3D::DataType::FLOAT;
    Format format = ::megdnn::param::Convolution3D::Format::NCDHW;
    Strategy strategy = static_cast<::megdnn::param::ExecutionPolicy::Strategy>(1);
    uint64_t workspace_limit = 18446744073709551615ull;
    Convolution3D() = default;
    Convolution3D(Mode mode_, uint32_t pad_d_, uint32_t pad_h_, uint32_t pad_w_, uint32_t stride_d_, uint32_t stride_h_, uint32_t stride_w_, uint32_t dilate_d_, uint32_t dilate_h_, uint32_t dilate_w_, Sparse sparse_, DataType data_type_, Format format_, Strategy strategy_, uint64_t workspace_limit_, std::string scope_ = {}): mode(mode_), pad_d(pad_d_), pad_h(pad_h_), pad_w(pad_w_), stride_d(stride_d_), stride_h(stride_h_), stride_w(stride_w_), dilate_d(dilate_d_), dilate_h(dilate_h_), dilate_w(dilate_w_), sparse(sparse_), data_type(data_type_), format(format_), strategy(strategy_), workspace_limit(workspace_limit_) {
        set_scope(scope_);
        mgb_assert(static_cast<uint32_t>(strategy) <= uint32_t(8));
    }
    Convolution3D(::megdnn::param::Convolution3D packed_param_0, ::megdnn::param::ExecutionPolicy packed_param_1): mode(packed_param_0.mode), pad_d(packed_param_0.pad_d), pad_h(packed_param_0.pad_h), pad_w(packed_param_0.pad_w), stride_d(packed_param_0.stride_d), stride_h(packed_param_0.stride_h), stride_w(packed_param_0.stride_w), dilate_d(packed_param_0.dilate_d), dilate_h(packed_param_0.dilate_h), dilate_w(packed_param_0.dilate_w), sparse(packed_param_0.sparse), data_type(packed_param_0.data_type), format(packed_param_0.format), strategy(packed_param_1.strategy), workspace_limit(packed_param_1.workspace_limit) {
        mgb_assert(static_cast<uint32_t>(strategy) <= uint32_t(8));
    }
    ::megdnn::param::Convolution3D param() const {
        return {mode, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, dilate_d, dilate_h, dilate_w, sparse, data_type, format};
    }
    ::megdnn::param::ExecutionPolicy policy() const {
        return {strategy, workspace_limit};
    }
};

class Convolution3DBackwardData : public OpDefImplBase<Convolution3DBackwardData> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using Mode = ::megdnn::param::Convolution3D::Mode;
    using Sparse = ::megdnn::param::Convolution3D::Sparse;
    using DataType = ::megdnn::param::Convolution3D::DataType;
    using Format = ::megdnn::param::Convolution3D::Format;
    using Strategy = ::megdnn::param::ExecutionPolicy::Strategy;
    Mode mode = ::megdnn::param::Convolution3D::Mode::CROSS_CORRELATION;
    uint32_t pad_d = 0;
    uint32_t pad_h = 0;
    uint32_t pad_w = 0;
    uint32_t stride_d = 1;
    uint32_t stride_h = 1;
    uint32_t stride_w = 1;
    uint32_t dilate_d = 1;
    uint32_t dilate_h = 1;
    uint32_t dilate_w = 1;
    Sparse sparse = ::megdnn::param::Convolution3D::Sparse::DENSE;
    DataType data_type = ::megdnn::param::Convolution3D::DataType::FLOAT;
    Format format = ::megdnn::param::Convolution3D::Format::NCDHW;
    Strategy strategy = static_cast<::megdnn::param::ExecutionPolicy::Strategy>(1);
    uint64_t workspace_limit = 18446744073709551615ull;
    Convolution3DBackwardData() = default;
    Convolution3DBackwardData(Mode mode_, uint32_t pad_d_, uint32_t pad_h_, uint32_t pad_w_, uint32_t stride_d_, uint32_t stride_h_, uint32_t stride_w_, uint32_t dilate_d_, uint32_t dilate_h_, uint32_t dilate_w_, Sparse sparse_, DataType data_type_, Format format_, Strategy strategy_, uint64_t workspace_limit_, std::string scope_ = {}): mode(mode_), pad_d(pad_d_), pad_h(pad_h_), pad_w(pad_w_), stride_d(stride_d_), stride_h(stride_h_), stride_w(stride_w_), dilate_d(dilate_d_), dilate_h(dilate_h_), dilate_w(dilate_w_), sparse(sparse_), data_type(data_type_), format(format_), strategy(strategy_), workspace_limit(workspace_limit_) {
        set_scope(scope_);
        mgb_assert(static_cast<uint32_t>(strategy) <= uint32_t(8));
    }
    Convolution3DBackwardData(::megdnn::param::Convolution3D packed_param_0, ::megdnn::param::ExecutionPolicy packed_param_1): mode(packed_param_0.mode), pad_d(packed_param_0.pad_d), pad_h(packed_param_0.pad_h), pad_w(packed_param_0.pad_w), stride_d(packed_param_0.stride_d), stride_h(packed_param_0.stride_h), stride_w(packed_param_0.stride_w), dilate_d(packed_param_0.dilate_d), dilate_h(packed_param_0.dilate_h), dilate_w(packed_param_0.dilate_w), sparse(packed_param_0.sparse), data_type(packed_param_0.data_type), format(packed_param_0.format), strategy(packed_param_1.strategy), workspace_limit(packed_param_1.workspace_limit) {
        mgb_assert(static_cast<uint32_t>(strategy) <= uint32_t(8));
    }
    ::megdnn::param::Convolution3D param() const {
        return {mode, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, dilate_d, dilate_h, dilate_w, sparse, data_type, format};
    }
    ::megdnn::param::ExecutionPolicy policy() const {
        return {strategy, workspace_limit};
    }
};

class ConvolutionBackwardData : public OpDefImplBase<ConvolutionBackwardData> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using Mode = ::megdnn::param::Convolution::Mode;
    using Sparse = ::megdnn::param::Convolution::Sparse;
    using Format = ::megdnn::param::Convolution::Format;
    using ComputeMode = ::megdnn::param::Convolution::ComputeMode;
    using Strategy = ::megdnn::param::ExecutionPolicy::Strategy;
    Mode mode = ::megdnn::param::Convolution::Mode::CROSS_CORRELATION;
    uint32_t pad_h = 0;
    uint32_t pad_w = 0;
    uint32_t stride_h = 1;
    uint32_t stride_w = 1;
    uint32_t dilate_h = 1;
    uint32_t dilate_w = 1;
    Sparse sparse = ::megdnn::param::Convolution::Sparse::DENSE;
    Format format = ::megdnn::param::Convolution::Format::NCHW;
    ComputeMode compute_mode = ::megdnn::param::Convolution::ComputeMode::DEFAULT;
    Strategy strategy = static_cast<::megdnn::param::ExecutionPolicy::Strategy>(1);
    uint64_t workspace_limit = 18446744073709551615ull;
    ::megdnn::DType dtype;
    ConvolutionBackwardData() = default;
    ConvolutionBackwardData(Mode mode_, uint32_t pad_h_, uint32_t pad_w_, uint32_t stride_h_, uint32_t stride_w_, uint32_t dilate_h_, uint32_t dilate_w_, Sparse sparse_, Format format_, ComputeMode compute_mode_, Strategy strategy_, uint64_t workspace_limit_, ::megdnn::DType dtype_, std::string scope_ = {}): mode(mode_), pad_h(pad_h_), pad_w(pad_w_), stride_h(stride_h_), stride_w(stride_w_), dilate_h(dilate_h_), dilate_w(dilate_w_), sparse(sparse_), format(format_), compute_mode(compute_mode_), strategy(strategy_), workspace_limit(workspace_limit_), dtype(dtype_) {
        set_scope(scope_);
        mgb_assert(static_cast<uint32_t>(strategy) <= uint32_t(8));
    }
    ConvolutionBackwardData(::megdnn::param::Convolution packed_param_0, ::megdnn::param::ExecutionPolicy packed_param_1, ::megdnn::DType dtype_): mode(packed_param_0.mode), pad_h(packed_param_0.pad_h), pad_w(packed_param_0.pad_w), stride_h(packed_param_0.stride_h), stride_w(packed_param_0.stride_w), dilate_h(packed_param_0.dilate_h), dilate_w(packed_param_0.dilate_w), sparse(packed_param_0.sparse), format(packed_param_0.format), compute_mode(packed_param_0.compute_mode), strategy(packed_param_1.strategy), workspace_limit(packed_param_1.workspace_limit), dtype(dtype_) {
        mgb_assert(static_cast<uint32_t>(strategy) <= uint32_t(8));
    }
    ::megdnn::param::Convolution param() const {
        return {mode, pad_h, pad_w, stride_h, stride_w, dilate_h, dilate_w, sparse, format, compute_mode};
    }
    ::megdnn::param::ExecutionPolicy policy() const {
        return {strategy, workspace_limit};
    }
};

class Copy : public OpDefImplBase<Copy> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    ::mgb::CompNode comp_node;
    Copy() = default;
    Copy(::mgb::CompNode comp_node_, std::string scope_ = {}): comp_node(comp_node_) { set_scope(scope_); }
};

class Correlation : public OpDefImplBase<Correlation> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using Format = ::megdnn::param::Correlation::Format;
    Format format = ::megdnn::param::Correlation::Format::NCHW;
    uint32_t kernel_size = 1;
    uint32_t max_displacement = 1;
    uint32_t stride1 = 1;
    uint32_t stride2 = 1;
    uint32_t pad_size = 0;
    bool is_multiply = true;
    Correlation() = default;
    Correlation(Format format_, uint32_t kernel_size_, uint32_t max_displacement_, uint32_t stride1_, uint32_t stride2_, uint32_t pad_size_, bool is_multiply_, std::string scope_ = {}): format(format_), kernel_size(kernel_size_), max_displacement(max_displacement_), stride1(stride1_), stride2(stride2_), pad_size(pad_size_), is_multiply(is_multiply_) { set_scope(scope_); }
    Correlation(::megdnn::param::Correlation packed_param_0): format(packed_param_0.format), kernel_size(packed_param_0.kernel_size), max_displacement(packed_param_0.max_displacement), stride1(packed_param_0.stride1), stride2(packed_param_0.stride2), pad_size(packed_param_0.pad_size), is_multiply(packed_param_0.is_multiply) {}
    ::megdnn::param::Correlation param() const {
        return {format, kernel_size, max_displacement, stride1, stride2, pad_size, is_multiply};
    }
};

class Cumsum : public OpDefImplBase<Cumsum> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    int32_t axis = 2147483647;
    bool exclusive = true;
    bool reverse = false;
    Cumsum() = default;
    Cumsum(int32_t axis_, bool exclusive_, bool reverse_, std::string scope_ = {}): axis(axis_), exclusive(exclusive_), reverse(reverse_) { set_scope(scope_); }
    Cumsum(::megdnn::param::Cumsum packed_param_0): axis(packed_param_0.axis), exclusive(packed_param_0.exclusive), reverse(packed_param_0.reverse) {}
    ::megdnn::param::Cumsum param() const {
        return {axis, exclusive, reverse};
    }
};

class CvtColor : public OpDefImplBase<CvtColor> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using Mode = ::megdnn::param::CvtColor::Mode;
    Mode mode = ::megdnn::param::CvtColor::Mode::RGB2GRAY;
    CvtColor() = default;
    CvtColor(Mode mode_, std::string scope_ = {}): mode(mode_) { set_scope(scope_); }
    CvtColor(::megdnn::param::CvtColor packed_param_0): mode(packed_param_0.mode) {}
    ::megdnn::param::CvtColor param() const {
        return {mode};
    }
};

class DeformableConv : public OpDefImplBase<DeformableConv> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using Mode = ::megdnn::param::Convolution::Mode;
    using Sparse = ::megdnn::param::Convolution::Sparse;
    using Format = ::megdnn::param::Convolution::Format;
    using ComputeMode = ::megdnn::param::Convolution::ComputeMode;
    using Strategy = ::megdnn::param::ExecutionPolicy::Strategy;
    Mode mode = ::megdnn::param::Convolution::Mode::CROSS_CORRELATION;
    uint32_t pad_h = 0;
    uint32_t pad_w = 0;
    uint32_t stride_h = 1;
    uint32_t stride_w = 1;
    uint32_t dilate_h = 1;
    uint32_t dilate_w = 1;
    Sparse sparse = ::megdnn::param::Convolution::Sparse::DENSE;
    Format format = ::megdnn::param::Convolution::Format::NCHW;
    ComputeMode compute_mode = ::megdnn::param::Convolution::ComputeMode::DEFAULT;
    Strategy strategy = static_cast<::megdnn::param::ExecutionPolicy::Strategy>(1);
    uint64_t workspace_limit = 18446744073709551615ull;
    DeformableConv() = default;
    DeformableConv(Mode mode_, uint32_t pad_h_, uint32_t pad_w_, uint32_t stride_h_, uint32_t stride_w_, uint32_t dilate_h_, uint32_t dilate_w_, Sparse sparse_, Format format_, ComputeMode compute_mode_, Strategy strategy_, uint64_t workspace_limit_, std::string scope_ = {}): mode(mode_), pad_h(pad_h_), pad_w(pad_w_), stride_h(stride_h_), stride_w(stride_w_), dilate_h(dilate_h_), dilate_w(dilate_w_), sparse(sparse_), format(format_), compute_mode(compute_mode_), strategy(strategy_), workspace_limit(workspace_limit_) {
        set_scope(scope_);
        mgb_assert(static_cast<uint32_t>(strategy) <= uint32_t(8));
    }
    DeformableConv(::megdnn::param::Convolution packed_param_0, ::megdnn::param::ExecutionPolicy packed_param_1): mode(packed_param_0.mode), pad_h(packed_param_0.pad_h), pad_w(packed_param_0.pad_w), stride_h(packed_param_0.stride_h), stride_w(packed_param_0.stride_w), dilate_h(packed_param_0.dilate_h), dilate_w(packed_param_0.dilate_w), sparse(packed_param_0.sparse), format(packed_param_0.format), compute_mode(packed_param_0.compute_mode), strategy(packed_param_1.strategy), workspace_limit(packed_param_1.workspace_limit) {
        mgb_assert(static_cast<uint32_t>(strategy) <= uint32_t(8));
    }
    ::megdnn::param::Convolution param() const {
        return {mode, pad_h, pad_w, stride_h, stride_w, dilate_h, dilate_w, sparse, format, compute_mode};
    }
    ::megdnn::param::ExecutionPolicy policy() const {
        return {strategy, workspace_limit};
    }
};

class DeformablePSROIPooling : public OpDefImplBase<DeformablePSROIPooling> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    bool no_trans = true;
    float spatial_scale = 1;
    float trans_std = 1;
    uint32_t pooled_h = 1;
    uint32_t pooled_w = 1;
    uint32_t part_size = 1;
    uint32_t sample_per_part = 1;
    DeformablePSROIPooling() = default;
    DeformablePSROIPooling(bool no_trans_, float spatial_scale_, float trans_std_, uint32_t pooled_h_, uint32_t pooled_w_, uint32_t part_size_, uint32_t sample_per_part_, std::string scope_ = {}): no_trans(no_trans_), spatial_scale(spatial_scale_), trans_std(trans_std_), pooled_h(pooled_h_), pooled_w(pooled_w_), part_size(part_size_), sample_per_part(sample_per_part_) { set_scope(scope_); }
    DeformablePSROIPooling(::megdnn::param::DeformablePSROIPooling packed_param_0): no_trans(packed_param_0.no_trans), spatial_scale(packed_param_0.spatial_scale), trans_std(packed_param_0.trans_std), pooled_h(packed_param_0.pooled_h), pooled_w(packed_param_0.pooled_w), part_size(packed_param_0.part_size), sample_per_part(packed_param_0.sample_per_part) {}
    ::megdnn::param::DeformablePSROIPooling param() const {
        return {no_trans, spatial_scale, trans_std, pooled_h, pooled_w, part_size, sample_per_part};
    }
};

class Diag : public OpDefImplBase<Diag> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    int32_t k = 0;
    Diag() = default;
    Diag(int32_t k_, std::string scope_ = {}): k(k_) { set_scope(scope_); }
    Diag(::megdnn::param::Diag packed_param_0): k(packed_param_0.k) {}
    ::megdnn::param::Diag param() const {
        return {k};
    }
};

class Dimshuffle : public OpDefImplBase<Dimshuffle> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    std::vector<int32_t> pattern;
    Dimshuffle() = default;
    Dimshuffle(std::vector<int32_t> pattern_, std::string scope_ = {}): pattern(pattern_) { set_scope(scope_); }
};

class Dot : public OpDefImplBase<Dot> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    Dot() = default;
    Dot(::megdnn::param::Empty) {}
    ::megdnn::param::Empty param() const {
        return {};
    }
};

class Dropout : public OpDefImplBase<Dropout> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    float drop_prob = 0;
    uint64_t seed = 0;
    size_t handle;
    Dropout() = default;
    Dropout(float drop_prob_, uint64_t seed_, size_t handle_, std::string scope_ = {}): drop_prob(drop_prob_), seed(seed_), handle(handle_) { set_scope(scope_); }
    Dropout(::megdnn::param::Dropout packed_param_0, size_t handle_): drop_prob(packed_param_0.drop_prob), seed(packed_param_0.seed), handle(handle_) {}
    ::megdnn::param::Dropout param() const {
        return {drop_prob, seed};
    }
};

class Elemwise : public OpDefImplBase<Elemwise> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using Mode = ::megdnn::param::Elemwise::Mode;
    Mode mode = ::megdnn::param::Elemwise::Mode::RELU;
    Elemwise() = default;
    Elemwise(Mode mode_, std::string scope_ = {}): mode(mode_) { set_scope(scope_); }
    Elemwise(::megdnn::param::Elemwise packed_param_0): mode(packed_param_0.mode) {}
    ::megdnn::param::Elemwise param() const {
        return {mode};
    }
};


template <>
struct ToStringTrait<Elemwise::Mode> {
    std::string operator()(Elemwise::Mode e) const {
        switch (e) {
            case Elemwise::Mode::RELU: return "RELU";
case Elemwise::Mode::ABS: return "ABS";
case Elemwise::Mode::ACOS: return "ACOS";
case Elemwise::Mode::ASIN: return "ASIN";
case Elemwise::Mode::CEIL: return "CEIL";
case Elemwise::Mode::COS: return "COS";
case Elemwise::Mode::EXP: return "EXP";
case Elemwise::Mode::EXPM1: return "EXPM1";
case Elemwise::Mode::FLOOR: return "FLOOR";
case Elemwise::Mode::LOG: return "LOG";
case Elemwise::Mode::LOG1P: return "LOG1P";
case Elemwise::Mode::NEGATE: return "NEGATE";
case Elemwise::Mode::SIGMOID: return "SIGMOID";
case Elemwise::Mode::SIN: return "SIN";
case Elemwise::Mode::TANH: return "TANH";
case Elemwise::Mode::ABS_GRAD: return "ABS_GRAD";
case Elemwise::Mode::ADD: return "ADD";
case Elemwise::Mode::FLOOR_DIV: return "FLOOR_DIV";
case Elemwise::Mode::MAX: return "MAX";
case Elemwise::Mode::MIN: return "MIN";
case Elemwise::Mode::MOD: return "MOD";
case Elemwise::Mode::MUL: return "MUL";
case Elemwise::Mode::POW: return "POW";
case Elemwise::Mode::SIGMOID_GRAD: return "SIGMOID_GRAD";
case Elemwise::Mode::SUB: return "SUB";
case Elemwise::Mode::SWITCH_GT0: return "SWITCH_GT0";
case Elemwise::Mode::TANH_GRAD: return "TANH_GRAD";
case Elemwise::Mode::TRUE_DIV: return "TRUE_DIV";
case Elemwise::Mode::LOG_SUM_EXP: return "LOG_SUM_EXP";
case Elemwise::Mode::LT: return "LT";
case Elemwise::Mode::LEQ: return "LEQ";
case Elemwise::Mode::EQ: return "EQ";
case Elemwise::Mode::SHL: return "SHL";
case Elemwise::Mode::SHR: return "SHR";
case Elemwise::Mode::COND_LEQ_MOV: return "COND_LEQ_MOV";
case Elemwise::Mode::FUSE_MUL_ADD3: return "FUSE_MUL_ADD3";
case Elemwise::Mode::FUSE_MUL_ADD4: return "FUSE_MUL_ADD4";
case Elemwise::Mode::FUSE_ADD_RELU: return "FUSE_ADD_RELU";
case Elemwise::Mode::FUSE_ADD_SIGMOID: return "FUSE_ADD_SIGMOID";
case Elemwise::Mode::FUSE_ADD_TANH: return "FUSE_ADD_TANH";
case Elemwise::Mode::FAST_TANH: return "FAST_TANH";
case Elemwise::Mode::FAST_TANH_GRAD: return "FAST_TANH_GRAD";
case Elemwise::Mode::ROUND: return "ROUND";
case Elemwise::Mode::RMULH: return "RMULH";
case Elemwise::Mode::ATAN2: return "ATAN2";
case Elemwise::Mode::ERF: return "ERF";
case Elemwise::Mode::ERFINV: return "ERFINV";
case Elemwise::Mode::ERFC: return "ERFC";
case Elemwise::Mode::ERFCINV: return "ERFCINV";
case Elemwise::Mode::H_SWISH: return "H_SWISH";
case Elemwise::Mode::H_SWISH_GRAD: return "H_SWISH_GRAD";
case Elemwise::Mode::FUSE_ADD_H_SWISH: return "FUSE_ADD_H_SWISH";
case Elemwise::Mode::NOT: return "NOT";
case Elemwise::Mode::AND: return "AND";
case Elemwise::Mode::OR: return "OR";
case Elemwise::Mode::XOR: return "XOR";
case Elemwise::Mode::SILU: return "SILU";
case Elemwise::Mode::SILU_GRAD: return "SILU_GRAD";
case Elemwise::Mode::GELU: return "GELU";
case Elemwise::Mode::GELU_GRAD: return "GELU_GRAD";
case Elemwise::Mode::COND_LT_MOV: return "COND_LT_MOV";
case Elemwise::Mode::NEQ: return "NEQ";
case Elemwise::Mode::ISNAN: return "ISNAN";
case Elemwise::Mode::ISINF: return "ISINF";
            default:
                return "Elemwise::Mode::Unknown";
        }
    }
};
class ElemwiseMultiType : public OpDefImplBase<ElemwiseMultiType> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using Mode = ::megdnn::param::ElemwiseMultiType::Mode;
    Mode mode = ::megdnn::param::ElemwiseMultiType::Mode::FUSE_MUL_ADD3_INT16x32x32x32;
    ::megdnn::DType dtype;
    ElemwiseMultiType() = default;
    ElemwiseMultiType(Mode mode_, ::megdnn::DType dtype_, std::string scope_ = {}): mode(mode_), dtype(dtype_) { set_scope(scope_); }
    ElemwiseMultiType(::megdnn::param::ElemwiseMultiType packed_param_0, ::megdnn::DType dtype_): mode(packed_param_0.mode), dtype(dtype_) {}
    ::megdnn::param::ElemwiseMultiType param() const {
        return {mode};
    }
};


template <>
struct ToStringTrait<ElemwiseMultiType::Mode> {
    std::string operator()(ElemwiseMultiType::Mode e) const {
        switch (e) {
            case ElemwiseMultiType::Mode::FUSE_MUL_ADD3_INT16x32x32x32: return "FUSE_MUL_ADD3_INT16x32x32x32";
case ElemwiseMultiType::Mode::FUSE_MUL_ADD3_IXxF32xF32xI8: return "FUSE_MUL_ADD3_IXxF32xF32xI8";
case ElemwiseMultiType::Mode::ROUND_SHR_SATURATE_IXxI8xI8: return "ROUND_SHR_SATURATE_IXxI8xI8";
case ElemwiseMultiType::Mode::FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT16x16x16x8: return "FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT16x16x16x8";
case ElemwiseMultiType::Mode::FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT32x32x32x8: return "FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT32x32x32x8";
case ElemwiseMultiType::Mode::ROUND_SHR_SATURATE_IXxI8xI16: return "ROUND_SHR_SATURATE_IXxI8xI16";
case ElemwiseMultiType::Mode::QADD: return "QADD";
case ElemwiseMultiType::Mode::QFUSE_ADD_RELU: return "QFUSE_ADD_RELU";
case ElemwiseMultiType::Mode::QMUL: return "QMUL";
case ElemwiseMultiType::Mode::QMIN: return "QMIN";
case ElemwiseMultiType::Mode::QMAX: return "QMAX";
case ElemwiseMultiType::Mode::QSUB: return "QSUB";
case ElemwiseMultiType::Mode::QTRUE_DIV: return "QTRUE_DIV";
case ElemwiseMultiType::Mode::QFUSE_ADD_SIGMOID: return "QFUSE_ADD_SIGMOID";
case ElemwiseMultiType::Mode::QFUSE_ADD_TANH: return "QFUSE_ADD_TANH";
case ElemwiseMultiType::Mode::QRELU: return "QRELU";
case ElemwiseMultiType::Mode::QABS: return "QABS";
case ElemwiseMultiType::Mode::QSIGMOID: return "QSIGMOID";
case ElemwiseMultiType::Mode::QEXP: return "QEXP";
case ElemwiseMultiType::Mode::QTANH: return "QTANH";
case ElemwiseMultiType::Mode::QFUSE_MUL_ADD3: return "QFUSE_MUL_ADD3";
case ElemwiseMultiType::Mode::QFAST_TANH: return "QFAST_TANH";
case ElemwiseMultiType::Mode::QNEGATE: return "QNEGATE";
case ElemwiseMultiType::Mode::QACOS: return "QACOS";
case ElemwiseMultiType::Mode::QASIN: return "QASIN";
case ElemwiseMultiType::Mode::QCEIL: return "QCEIL";
case ElemwiseMultiType::Mode::QCOS: return "QCOS";
case ElemwiseMultiType::Mode::QEXPM1: return "QEXPM1";
case ElemwiseMultiType::Mode::QFLOOR: return "QFLOOR";
case ElemwiseMultiType::Mode::QLOG: return "QLOG";
case ElemwiseMultiType::Mode::QLOG1P: return "QLOG1P";
case ElemwiseMultiType::Mode::QSIN: return "QSIN";
case ElemwiseMultiType::Mode::QROUND: return "QROUND";
case ElemwiseMultiType::Mode::QERF: return "QERF";
case ElemwiseMultiType::Mode::QERFINV: return "QERFINV";
case ElemwiseMultiType::Mode::QERFC: return "QERFC";
case ElemwiseMultiType::Mode::QERFCINV: return "QERFCINV";
case ElemwiseMultiType::Mode::QABS_GRAD: return "QABS_GRAD";
case ElemwiseMultiType::Mode::QFLOOR_DIV: return "QFLOOR_DIV";
case ElemwiseMultiType::Mode::QMOD: return "QMOD";
case ElemwiseMultiType::Mode::QSIGMOID_GRAD: return "QSIGMOID_GRAD";
case ElemwiseMultiType::Mode::QSWITCH_GT0: return "QSWITCH_GT0";
case ElemwiseMultiType::Mode::QTANH_GRAD: return "QTANH_GRAD";
case ElemwiseMultiType::Mode::QLT: return "QLT";
case ElemwiseMultiType::Mode::QLEQ: return "QLEQ";
case ElemwiseMultiType::Mode::QEQ: return "QEQ";
case ElemwiseMultiType::Mode::QPOW: return "QPOW";
case ElemwiseMultiType::Mode::QLOG_SUM_EXP: return "QLOG_SUM_EXP";
case ElemwiseMultiType::Mode::QFAST_TANH_GRAD: return "QFAST_TANH_GRAD";
case ElemwiseMultiType::Mode::QATAN2: return "QATAN2";
case ElemwiseMultiType::Mode::QCOND_LEQ_MOV: return "QCOND_LEQ_MOV";
case ElemwiseMultiType::Mode::QH_SWISH: return "QH_SWISH";
case ElemwiseMultiType::Mode::QFUSE_ADD_H_SWISH: return "QFUSE_ADD_H_SWISH";
case ElemwiseMultiType::Mode::QH_SWISH_GRAD: return "QH_SWISH_GRAD";
case ElemwiseMultiType::Mode::FUSE_MUL_ADD3_INT16xF32xF32xF32: return "FUSE_MUL_ADD3_INT16xF32xF32xF32";
case ElemwiseMultiType::Mode::MUL_INT16xF32xF32: return "MUL_INT16xF32xF32";
case ElemwiseMultiType::Mode::FUSE_MUL_ADD3_UINT8xF32xF32xF32: return "FUSE_MUL_ADD3_UINT8xF32xF32xF32";
case ElemwiseMultiType::Mode::QCOND_LT_MOV: return "QCOND_LT_MOV";
case ElemwiseMultiType::Mode::EQ: return "EQ";
case ElemwiseMultiType::Mode::NEQ: return "NEQ";
case ElemwiseMultiType::Mode::LT: return "LT";
case ElemwiseMultiType::Mode::LEQ: return "LEQ";
case ElemwiseMultiType::Mode::ISNAN: return "ISNAN";
case ElemwiseMultiType::Mode::ISINF: return "ISINF";
            default:
                return "ElemwiseMultiType::Mode::Unknown";
        }
    }
};
class ExternOpr : public OpDefImplBase<ExternOpr> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    std::vector<std::vector<size_t>> output_shapes;
    std::string name;
    std::string data;
    size_t data_len;
    std::vector<::megdnn::DType> output_dtypes;
    ExternOpr() = default;
    ExternOpr(std::vector<std::vector<size_t>> output_shapes_, std::string name_, std::string data_, size_t data_len_, std::vector<::megdnn::DType> output_dtypes_, std::string scope_ = {}): output_shapes(output_shapes_), name(name_), data(data_), data_len(data_len_), output_dtypes(output_dtypes_) { set_scope(scope_); }
};

class Eye : public OpDefImplBase<Eye> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    int32_t k = 0;
    ::megdnn::DType dtype = megdnn::DType::from_enum(megdnn::DTypeEnum::Float32);
    ::mgb::CompNode comp_node;
    Eye() = default;
    Eye(int32_t k_, ::megdnn::DType dtype_, ::mgb::CompNode comp_node_, std::string scope_ = {}): k(k_), dtype(dtype_), comp_node(comp_node_) { set_scope(scope_); }
};

class FakeQuant : public OpDefImplBase<FakeQuant> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    int32_t qmin = -2147483648;
    int32_t qmax = 2147483647;
    FakeQuant() = default;
    FakeQuant(int32_t qmin_, int32_t qmax_, std::string scope_ = {}): qmin(qmin_), qmax(qmax_) { set_scope(scope_); }
    FakeQuant(::megdnn::param::FakeQuant packed_param_0): qmin(packed_param_0.qmin), qmax(packed_param_0.qmax) {}
    ::megdnn::param::FakeQuant param() const {
        return {qmin, qmax};
    }
};

class FastpathCopy : public OpDefImplBase<FastpathCopy> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    FastpathCopy() = default;
};

class GammaRNG : public OpDefImplBase<GammaRNG> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    uint64_t seed = 0;
    size_t handle;
    GammaRNG() = default;
    GammaRNG(uint64_t seed_, size_t handle_, std::string scope_ = {}): seed(seed_), handle(handle_) { set_scope(scope_); }
    GammaRNG(::megdnn::param::GammaRNG packed_param_0, size_t handle_): seed(packed_param_0.seed), handle(handle_) {}
    ::megdnn::param::GammaRNG param() const {
        return {seed};
    }
};

class GaussianRNG : public OpDefImplBase<GaussianRNG> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    uint64_t seed = 0;
    float mean = 0;
    float std = 1;
    ::megdnn::DType dtype = megdnn::DType::from_enum(megdnn::DTypeEnum::Float32);
    size_t handle;
    GaussianRNG() = default;
    GaussianRNG(uint64_t seed_, float mean_, float std_, ::megdnn::DType dtype_, size_t handle_, std::string scope_ = {}): seed(seed_), mean(mean_), std(std_), dtype(dtype_), handle(handle_) { set_scope(scope_); }
};

class GetVarShape : public OpDefImplBase<GetVarShape> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    int32_t axis = ::megdnn::param::OptionalAxisV1::INVALID_AXIS;
    GetVarShape() = default;
    GetVarShape(int32_t axis_, std::string scope_ = {}): axis(axis_) { set_scope(scope_); }
    GetVarShape(::megdnn::param::OptionalAxisV1 packed_param_0): axis(packed_param_0.axis) {}
    ::megdnn::param::OptionalAxisV1 param() const {
        return {axis};
    }
};

class GroupLocal : public OpDefImplBase<GroupLocal> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using Mode = ::megdnn::param::Convolution::Mode;
    using Sparse = ::megdnn::param::Convolution::Sparse;
    using Format = ::megdnn::param::Convolution::Format;
    using ComputeMode = ::megdnn::param::Convolution::ComputeMode;
    Mode mode = ::megdnn::param::Convolution::Mode::CROSS_CORRELATION;
    uint32_t pad_h = 0;
    uint32_t pad_w = 0;
    uint32_t stride_h = 1;
    uint32_t stride_w = 1;
    uint32_t dilate_h = 1;
    uint32_t dilate_w = 1;
    Sparse sparse = ::megdnn::param::Convolution::Sparse::DENSE;
    Format format = ::megdnn::param::Convolution::Format::NCHW;
    ComputeMode compute_mode = ::megdnn::param::Convolution::ComputeMode::DEFAULT;
    GroupLocal() = default;
    GroupLocal(Mode mode_, uint32_t pad_h_, uint32_t pad_w_, uint32_t stride_h_, uint32_t stride_w_, uint32_t dilate_h_, uint32_t dilate_w_, Sparse sparse_, Format format_, ComputeMode compute_mode_, std::string scope_ = {}): mode(mode_), pad_h(pad_h_), pad_w(pad_w_), stride_h(stride_h_), stride_w(stride_w_), dilate_h(dilate_h_), dilate_w(dilate_w_), sparse(sparse_), format(format_), compute_mode(compute_mode_) { set_scope(scope_); }
    GroupLocal(::megdnn::param::Convolution packed_param_0): mode(packed_param_0.mode), pad_h(packed_param_0.pad_h), pad_w(packed_param_0.pad_w), stride_h(packed_param_0.stride_h), stride_w(packed_param_0.stride_w), dilate_h(packed_param_0.dilate_h), dilate_w(packed_param_0.dilate_w), sparse(packed_param_0.sparse), format(packed_param_0.format), compute_mode(packed_param_0.compute_mode) {}
    ::megdnn::param::Convolution param() const {
        return {mode, pad_h, pad_w, stride_h, stride_w, dilate_h, dilate_w, sparse, format, compute_mode};
    }
};

class GroupNorm : public OpDefImplBase<GroupNorm> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using Format = ::megdnn::param::GroupNorm::Format;
    bool affine = true;
    float eps = 1e-5f;
    uint32_t group = 1;
    Format format = ::megdnn::param::GroupNorm::Format::NCHW;
    GroupNorm() = default;
    GroupNorm(bool affine_, float eps_, uint32_t group_, Format format_, std::string scope_ = {}): affine(affine_), eps(eps_), group(group_), format(format_) { set_scope(scope_); }
    GroupNorm(::megdnn::param::GroupNorm packed_param_0): affine(packed_param_0.affine), eps(packed_param_0.eps), group(packed_param_0.group), format(packed_param_0.format) {}
    ::megdnn::param::GroupNorm param() const {
        return {affine, eps, group, format};
    }
};

class Identity : public OpDefImplBase<Identity> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    Identity() = default;
};

class Images2Neibs : public OpDefImplBase<Images2Neibs> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    uint32_t pad_h = 0;
    uint32_t pad_w = 0;
    uint32_t stride_h = 1;
    uint32_t stride_w = 1;
    uint32_t dilate_h = 1;
    uint32_t dilate_w = 1;
    uint32_t window_h = 3;
    uint32_t window_w = 3;
    Images2Neibs() = default;
    Images2Neibs(uint32_t pad_h_, uint32_t pad_w_, uint32_t stride_h_, uint32_t stride_w_, uint32_t dilate_h_, uint32_t dilate_w_, uint32_t window_h_, uint32_t window_w_, std::string scope_ = {}): pad_h(pad_h_), pad_w(pad_w_), stride_h(stride_h_), stride_w(stride_w_), dilate_h(dilate_h_), dilate_w(dilate_w_), window_h(window_h_), window_w(window_w_) { set_scope(scope_); }
    Images2Neibs(::megdnn::param::Images2Neibs packed_param_0): pad_h(packed_param_0.pad_h), pad_w(packed_param_0.pad_w), stride_h(packed_param_0.stride_h), stride_w(packed_param_0.stride_w), dilate_h(packed_param_0.dilate_h), dilate_w(packed_param_0.dilate_w), window_h(packed_param_0.window_h), window_w(packed_param_0.window_w) {}
    ::megdnn::param::Images2Neibs param() const {
        return {pad_h, pad_w, stride_h, stride_w, dilate_h, dilate_w, window_h, window_w};
    }
};

class IncrMeshIndexing : public OpDefImplBase<IncrMeshIndexing> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    std::vector<std::tuple<int8_t, bool, bool, bool, bool>> items;
    IncrMeshIndexing() = default;
    IncrMeshIndexing(std::vector<std::tuple<int8_t, bool, bool, bool, bool>> items_, std::string scope_ = {}): items(items_) { set_scope(scope_); }
};

class IncrSubtensor : public OpDefImplBase<IncrSubtensor> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    std::vector<std::tuple<int8_t, bool, bool, bool, bool>> items;
    IncrSubtensor() = default;
    IncrSubtensor(std::vector<std::tuple<int8_t, bool, bool, bool, bool>> items_, std::string scope_ = {}): items(items_) { set_scope(scope_); }
};

class IndexingIncrMultiAxisVec : public OpDefImplBase<IndexingIncrMultiAxisVec> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    std::vector<std::tuple<int8_t, bool, bool, bool, bool>> items;
    IndexingIncrMultiAxisVec() = default;
    IndexingIncrMultiAxisVec(std::vector<std::tuple<int8_t, bool, bool, bool, bool>> items_, std::string scope_ = {}): items(items_) { set_scope(scope_); }
};

class IndexingMultiAxisVec : public OpDefImplBase<IndexingMultiAxisVec> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    std::vector<std::tuple<int8_t, bool, bool, bool, bool>> items;
    IndexingMultiAxisVec() = default;
    IndexingMultiAxisVec(std::vector<std::tuple<int8_t, bool, bool, bool, bool>> items_, std::string scope_ = {}): items(items_) { set_scope(scope_); }
};

class IndexingOneHot : public OpDefImplBase<IndexingOneHot> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    int32_t axis = 0;
    int32_t ndim;
    IndexingOneHot() = default;
    IndexingOneHot(int32_t axis_, int32_t ndim_, std::string scope_ = {}): axis(axis_), ndim(ndim_) { set_scope(scope_); }
    IndexingOneHot(::megdnn::param::Axis packed_param_0, int32_t ndim_): axis(packed_param_0.axis), ndim(ndim_) {}
    ::megdnn::param::Axis param() const {
        return {axis};
    }
};

class IndexingSetMultiAxisVec : public OpDefImplBase<IndexingSetMultiAxisVec> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    std::vector<std::tuple<int8_t, bool, bool, bool, bool>> items;
    IndexingSetMultiAxisVec() = default;
    IndexingSetMultiAxisVec(std::vector<std::tuple<int8_t, bool, bool, bool, bool>> items_, std::string scope_ = {}): items(items_) { set_scope(scope_); }
};

class IndexingSetOneHot : public OpDefImplBase<IndexingSetOneHot> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    int32_t axis = 0;
    int32_t ndim;
    IndexingSetOneHot() = default;
    IndexingSetOneHot(int32_t axis_, int32_t ndim_, std::string scope_ = {}): axis(axis_), ndim(ndim_) { set_scope(scope_); }
    IndexingSetOneHot(::megdnn::param::Axis packed_param_0, int32_t ndim_): axis(packed_param_0.axis), ndim(ndim_) {}
    ::megdnn::param::Axis param() const {
        return {axis};
    }
};

class InplaceAdd : public OpDefImplBase<InplaceAdd> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    InplaceAdd() = default;
    InplaceAdd(::megdnn::param::Empty) {}
    ::megdnn::param::Empty param() const {
        return {};
    }
};

class LAMBUpdate : public OpDefImplBase<LAMBUpdate> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    float beta_1 = 1.f;
    float beta_2 = 1.f;
    float step = 1.f;
    float lr = 1.f;
    float weight_decay = 1.f;
    float eps = 1.f;
    bool bias_correction = true;
    bool always_adapt = false;
    LAMBUpdate() = default;
    LAMBUpdate(float beta_1_, float beta_2_, float step_, float lr_, float weight_decay_, float eps_, bool bias_correction_, bool always_adapt_, std::string scope_ = {}): beta_1(beta_1_), beta_2(beta_2_), step(step_), lr(lr_), weight_decay(weight_decay_), eps(eps_), bias_correction(bias_correction_), always_adapt(always_adapt_) { set_scope(scope_); }
    LAMBUpdate(::megdnn::param::LAMBUpdate packed_param_0): beta_1(packed_param_0.beta_1), beta_2(packed_param_0.beta_2), step(packed_param_0.step), lr(packed_param_0.lr), weight_decay(packed_param_0.weight_decay), eps(packed_param_0.eps), bias_correction(packed_param_0.bias_correction), always_adapt(packed_param_0.always_adapt) {}
    ::megdnn::param::LAMBUpdate param() const {
        return {beta_1, beta_2, step, lr, weight_decay, eps, bias_correction, always_adapt};
    }
};

class LRN : public OpDefImplBase<LRN> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    uint32_t n = 5;
    float k = 2.f;
    float alpha = 1e-4f;
    float beta = 0.75f;
    LRN() = default;
    LRN(uint32_t n_, float k_, float alpha_, float beta_, std::string scope_ = {}): n(n_), k(k_), alpha(alpha_), beta(beta_) { set_scope(scope_); }
    LRN(::megdnn::param::LRN packed_param_0): n(packed_param_0.n), k(packed_param_0.k), alpha(packed_param_0.alpha), beta(packed_param_0.beta) {}
    ::megdnn::param::LRN param() const {
        return {n, k, alpha, beta};
    }
};

class LSQ : public OpDefImplBase<LSQ> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    int32_t qmin = -2147483648;
    int32_t qmax = 2147483647;
    LSQ() = default;
    LSQ(int32_t qmin_, int32_t qmax_, std::string scope_ = {}): qmin(qmin_), qmax(qmax_) { set_scope(scope_); }
    LSQ(::megdnn::param::LSQ packed_param_0): qmin(packed_param_0.qmin), qmax(packed_param_0.qmax) {}
    ::megdnn::param::LSQ param() const {
        return {qmin, qmax};
    }
};

class LSTM : public OpDefImplBase<LSTM> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using FwdMode = ::megdnn::param::LSTM::FwdMode;
    uint32_t num_layers = 1;
    bool bidirectional = false;
    bool bias = true;
    uint32_t hidden_size = 128;
    uint32_t proj_size = 0;
    float dropout = 0.f;
    FwdMode fwd_mode = ::megdnn::param::LSTM::FwdMode::TRAINING;
    LSTM() = default;
    LSTM(uint32_t num_layers_, bool bidirectional_, bool bias_, uint32_t hidden_size_, uint32_t proj_size_, float dropout_, FwdMode fwd_mode_, std::string scope_ = {}): num_layers(num_layers_), bidirectional(bidirectional_), bias(bias_), hidden_size(hidden_size_), proj_size(proj_size_), dropout(dropout_), fwd_mode(fwd_mode_) { set_scope(scope_); }
    LSTM(::megdnn::param::LSTM packed_param_0): num_layers(packed_param_0.num_layers), bidirectional(packed_param_0.bidirectional), bias(packed_param_0.bias), hidden_size(packed_param_0.hidden_size), proj_size(packed_param_0.proj_size), dropout(packed_param_0.dropout), fwd_mode(packed_param_0.fwd_mode) {}
    ::megdnn::param::LSTM param() const {
        return {num_layers, bidirectional, bias, hidden_size, proj_size, dropout, fwd_mode};
    }
};

class LSTMCell : public OpDefImplBase<LSTMCell> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    LSTMCell() = default;
    LSTMCell(::megdnn::param::Empty) {}
    ::megdnn::param::Empty param() const {
        return {};
    }
};

class LayerNorm : public OpDefImplBase<LayerNorm> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    bool affine = true;
    float eps = 1e-5f;
    uint64_t normalized_dim = 1;
    uint64_t normalized_size = 1;
    LayerNorm() = default;
    LayerNorm(bool affine_, float eps_, uint64_t normalized_dim_, uint64_t normalized_size_, std::string scope_ = {}): affine(affine_), eps(eps_), normalized_dim(normalized_dim_), normalized_size(normalized_size_) { set_scope(scope_); }
    LayerNorm(::megdnn::param::LayerNorm packed_param_0): affine(packed_param_0.affine), eps(packed_param_0.eps), normalized_dim(packed_param_0.normalized_dim), normalized_size(packed_param_0.normalized_size) {}
    ::megdnn::param::LayerNorm param() const {
        return {affine, eps, normalized_dim, normalized_size};
    }
};

class Linspace : public OpDefImplBase<Linspace> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    bool endpoint = true;
    ::mgb::CompNode comp_node;
    Linspace() = default;
    Linspace(bool endpoint_, ::mgb::CompNode comp_node_, std::string scope_ = {}): endpoint(endpoint_), comp_node(comp_node_) { set_scope(scope_); }
    Linspace(::megdnn::param::Linspace packed_param_0, ::mgb::CompNode comp_node_): endpoint(packed_param_0.endpoint), comp_node(comp_node_) {}
    ::megdnn::param::Linspace param() const {
        return {endpoint};
    }
};

class MagicMindRuntime : public OpDefImplBase<MagicMindRuntime> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    std::string buf;
    size_t buf_size;
    MagicMindRuntime() = default;
    MagicMindRuntime(std::string buf_, size_t buf_size_, std::string scope_ = {}): buf(buf_), buf_size(buf_size_) { set_scope(scope_); }
};

class MatrixInverse : public OpDefImplBase<MatrixInverse> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    MatrixInverse() = default;
    MatrixInverse(::megdnn::param::Empty) {}
    ::megdnn::param::Empty param() const {
        return {};
    }
};

class MatrixMul : public OpDefImplBase<MatrixMul> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using ComputeMode = ::megdnn::param::MatrixMul::ComputeMode;
    using Format = ::megdnn::param::MatrixMul::Format;
    using Strategy = ::megdnn::param::ExecutionPolicy::Strategy;
    bool transposeA = false;
    bool transposeB = false;
    ComputeMode compute_mode = ::megdnn::param::MatrixMul::ComputeMode::DEFAULT;
    Format format = ::megdnn::param::MatrixMul::Format::DEFAULT;
    Strategy strategy = static_cast<::megdnn::param::ExecutionPolicy::Strategy>(1);
    uint64_t workspace_limit = 18446744073709551615ull;
    uint32_t dimA;
    uint32_t dimB;
    MatrixMul() = default;
    MatrixMul(bool transposeA_, bool transposeB_, ComputeMode compute_mode_, Format format_, Strategy strategy_, uint64_t workspace_limit_, uint32_t dimA_, uint32_t dimB_, std::string scope_ = {}): transposeA(transposeA_), transposeB(transposeB_), compute_mode(compute_mode_), format(format_), strategy(strategy_), workspace_limit(workspace_limit_), dimA(dimA_), dimB(dimB_) {
        set_scope(scope_);
        mgb_assert(static_cast<uint32_t>(strategy) <= uint32_t(8));
    }
    MatrixMul(::megdnn::param::MatrixMul packed_param_0, ::megdnn::param::ExecutionPolicy packed_param_1, uint32_t dimA_, uint32_t dimB_): transposeA(packed_param_0.transposeA), transposeB(packed_param_0.transposeB), compute_mode(packed_param_0.compute_mode), format(packed_param_0.format), strategy(packed_param_1.strategy), workspace_limit(packed_param_1.workspace_limit), dimA(dimA_), dimB(dimB_) {
        mgb_assert(static_cast<uint32_t>(strategy) <= uint32_t(8));
    }
    ::megdnn::param::MatrixMul param() const {
        return {transposeA, transposeB, compute_mode, format};
    }
    ::megdnn::param::ExecutionPolicy policy() const {
        return {strategy, workspace_limit};
    }
};

class MeshGrid : public OpDefImplBase<MeshGrid> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    std::string indexing;
    MeshGrid() = default;
    MeshGrid(std::string indexing_, std::string scope_ = {}): indexing(indexing_) { set_scope(scope_); }
};

class MeshIndexing : public OpDefImplBase<MeshIndexing> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    std::vector<std::tuple<int8_t, bool, bool, bool, bool>> items;
    MeshIndexing() = default;
    MeshIndexing(std::vector<std::tuple<int8_t, bool, bool, bool, bool>> items_, std::string scope_ = {}): items(items_) { set_scope(scope_); }
};

class NMSKeep : public OpDefImplBase<NMSKeep> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    float iou_thresh;
    uint32_t max_output;
    NMSKeep() = default;
    NMSKeep(float iou_thresh_, uint32_t max_output_, std::string scope_ = {}): iou_thresh(iou_thresh_), max_output(max_output_) { set_scope(scope_); }
};

class NvOf : public OpDefImplBase<NvOf> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    uint32_t precision = 1;
    NvOf() = default;
    NvOf(uint32_t precision_, std::string scope_ = {}): precision(precision_) { set_scope(scope_); }
    NvOf(::megdnn::param::NvOf packed_param_0): precision(packed_param_0.precision) {}
    ::megdnn::param::NvOf param() const {
        return {precision};
    }
};

class Padding : public OpDefImplBase<Padding> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using PaddingMode = ::megdnn::param::Padding::PaddingMode;
    uint32_t front_offset_dim0 = 0;
    uint32_t front_offset_dim1 = 0;
    uint32_t front_offset_dim2 = 0;
    uint32_t front_offset_dim3 = 0;
    uint32_t front_offset_dim4 = 0;
    uint32_t front_offset_dim5 = 0;
    uint32_t front_offset_dim6 = 0;
    uint32_t back_offset_dim0 = 0;
    uint32_t back_offset_dim1 = 0;
    uint32_t back_offset_dim2 = 0;
    uint32_t back_offset_dim3 = 0;
    uint32_t back_offset_dim4 = 0;
    uint32_t back_offset_dim5 = 0;
    uint32_t back_offset_dim6 = 0;
    float padding_val = 0;
    PaddingMode padding_mode = ::megdnn::param::Padding::PaddingMode::CONSTANT;
    Padding() = default;
    Padding(uint32_t front_offset_dim0_, uint32_t front_offset_dim1_, uint32_t front_offset_dim2_, uint32_t front_offset_dim3_, uint32_t front_offset_dim4_, uint32_t front_offset_dim5_, uint32_t front_offset_dim6_, uint32_t back_offset_dim0_, uint32_t back_offset_dim1_, uint32_t back_offset_dim2_, uint32_t back_offset_dim3_, uint32_t back_offset_dim4_, uint32_t back_offset_dim5_, uint32_t back_offset_dim6_, float padding_val_, PaddingMode padding_mode_, std::string scope_ = {}): front_offset_dim0(front_offset_dim0_), front_offset_dim1(front_offset_dim1_), front_offset_dim2(front_offset_dim2_), front_offset_dim3(front_offset_dim3_), front_offset_dim4(front_offset_dim4_), front_offset_dim5(front_offset_dim5_), front_offset_dim6(front_offset_dim6_), back_offset_dim0(back_offset_dim0_), back_offset_dim1(back_offset_dim1_), back_offset_dim2(back_offset_dim2_), back_offset_dim3(back_offset_dim3_), back_offset_dim4(back_offset_dim4_), back_offset_dim5(back_offset_dim5_), back_offset_dim6(back_offset_dim6_), padding_val(padding_val_), padding_mode(padding_mode_) { set_scope(scope_); }
    Padding(::megdnn::param::Padding packed_param_0): front_offset_dim0(packed_param_0.front_offset_dim0), front_offset_dim1(packed_param_0.front_offset_dim1), front_offset_dim2(packed_param_0.front_offset_dim2), front_offset_dim3(packed_param_0.front_offset_dim3), front_offset_dim4(packed_param_0.front_offset_dim4), front_offset_dim5(packed_param_0.front_offset_dim5), front_offset_dim6(packed_param_0.front_offset_dim6), back_offset_dim0(packed_param_0.back_offset_dim0), back_offset_dim1(packed_param_0.back_offset_dim1), back_offset_dim2(packed_param_0.back_offset_dim2), back_offset_dim3(packed_param_0.back_offset_dim3), back_offset_dim4(packed_param_0.back_offset_dim4), back_offset_dim5(packed_param_0.back_offset_dim5), back_offset_dim6(packed_param_0.back_offset_dim6), padding_val(packed_param_0.padding_val), padding_mode(packed_param_0.padding_mode) {}
    ::megdnn::param::Padding param() const {
        return {front_offset_dim0, front_offset_dim1, front_offset_dim2, front_offset_dim3, front_offset_dim4, front_offset_dim5, front_offset_dim6, back_offset_dim0, back_offset_dim1, back_offset_dim2, back_offset_dim3, back_offset_dim4, back_offset_dim5, back_offset_dim6, padding_val, padding_mode};
    }
};

class ParamPackConcat : public OpDefImplBase<ParamPackConcat> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    std::vector<int32_t> offsets;
    ParamPackConcat() = default;
    ParamPackConcat(std::vector<int32_t> offsets_, std::string scope_ = {}): offsets(offsets_) { set_scope(scope_); }
};

class ParamPackSplit : public OpDefImplBase<ParamPackSplit> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    std::vector<int32_t> offsets;
    std::vector<std::vector<size_t>> shapes;
    ParamPackSplit() = default;
    ParamPackSplit(std::vector<int32_t> offsets_, std::vector<std::vector<size_t>> shapes_, std::string scope_ = {}): offsets(offsets_), shapes(shapes_) { set_scope(scope_); }
};

class PermutationRNG : public OpDefImplBase<PermutationRNG> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    uint64_t seed = 0;
    ::megdnn::DType dtype = megdnn::DType::from_enum(megdnn::DTypeEnum::Int32);
    size_t handle;
    PermutationRNG() = default;
    PermutationRNG(uint64_t seed_, ::megdnn::DType dtype_, size_t handle_, std::string scope_ = {}): seed(seed_), dtype(dtype_), handle(handle_) { set_scope(scope_); }
};

class PixelShuffle : public OpDefImplBase<PixelShuffle> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    int32_t factor;
    PixelShuffle() = default;
    PixelShuffle(int32_t factor_, std::string scope_ = {}): factor(factor_) { set_scope(scope_); }
};

class PixelShuffleBackward : public OpDefImplBase<PixelShuffleBackward> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    int32_t factor;
    PixelShuffleBackward() = default;
    PixelShuffleBackward(int32_t factor_, std::string scope_ = {}): factor(factor_) { set_scope(scope_); }
};

class PoissonRNG : public OpDefImplBase<PoissonRNG> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    uint64_t seed = 0;
    size_t handle;
    PoissonRNG() = default;
    PoissonRNG(uint64_t seed_, size_t handle_, std::string scope_ = {}): seed(seed_), handle(handle_) { set_scope(scope_); }
    PoissonRNG(::megdnn::param::PoissonRNG packed_param_0, size_t handle_): seed(packed_param_0.seed), handle(handle_) {}
    ::megdnn::param::PoissonRNG param() const {
        return {seed};
    }
};

class Pooling : public OpDefImplBase<Pooling> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using Mode = ::megdnn::param::Pooling::Mode;
    using Format = ::megdnn::param::Pooling::Format;
    using Strategy = ::megdnn::param::ExecutionPolicy::Strategy;
    Mode mode = ::megdnn::param::Pooling::Mode::MAX;
    uint32_t pad_h = 0;
    uint32_t pad_w = 0;
    uint32_t stride_h = 2;
    uint32_t stride_w = 2;
    uint32_t window_h = 2;
    uint32_t window_w = 2;
    Format format = ::megdnn::param::Pooling::Format::NCHW;
    Strategy strategy = static_cast<::megdnn::param::ExecutionPolicy::Strategy>(1);
    uint64_t workspace_limit = 18446744073709551615ull;
    Pooling() = default;
    Pooling(Mode mode_, uint32_t pad_h_, uint32_t pad_w_, uint32_t stride_h_, uint32_t stride_w_, uint32_t window_h_, uint32_t window_w_, Format format_, Strategy strategy_, uint64_t workspace_limit_, std::string scope_ = {}): mode(mode_), pad_h(pad_h_), pad_w(pad_w_), stride_h(stride_h_), stride_w(stride_w_), window_h(window_h_), window_w(window_w_), format(format_), strategy(strategy_), workspace_limit(workspace_limit_) {
        set_scope(scope_);
        mgb_assert(static_cast<uint32_t>(strategy) <= uint32_t(8));
    }
    Pooling(::megdnn::param::Pooling packed_param_0, ::megdnn::param::ExecutionPolicy packed_param_1): mode(packed_param_0.mode), pad_h(packed_param_0.pad_h), pad_w(packed_param_0.pad_w), stride_h(packed_param_0.stride_h), stride_w(packed_param_0.stride_w), window_h(packed_param_0.window_h), window_w(packed_param_0.window_w), format(packed_param_0.format), strategy(packed_param_1.strategy), workspace_limit(packed_param_1.workspace_limit) {
        mgb_assert(static_cast<uint32_t>(strategy) <= uint32_t(8));
    }
    ::megdnn::param::Pooling param() const {
        return {mode, pad_h, pad_w, stride_h, stride_w, window_h, window_w, format};
    }
    ::megdnn::param::ExecutionPolicy policy() const {
        return {strategy, workspace_limit};
    }
};

class RNN : public OpDefImplBase<RNN> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using NonlineMode = ::megdnn::param::RNN::NonlineMode;
    using FwdMode = ::megdnn::param::RNN::FwdMode;
    uint32_t num_layers = 1;
    bool bidirectional = false;
    bool bias = true;
    uint32_t hidden_size = 128;
    float dropout = 0.f;
    NonlineMode nonlineMode = ::megdnn::param::RNN::NonlineMode::IDENTITY;
    FwdMode fwd_mode = ::megdnn::param::RNN::FwdMode::TRAINING;
    RNN() = default;
    RNN(uint32_t num_layers_, bool bidirectional_, bool bias_, uint32_t hidden_size_, float dropout_, NonlineMode nonlineMode_, FwdMode fwd_mode_, std::string scope_ = {}): num_layers(num_layers_), bidirectional(bidirectional_), bias(bias_), hidden_size(hidden_size_), dropout(dropout_), nonlineMode(nonlineMode_), fwd_mode(fwd_mode_) { set_scope(scope_); }
    RNN(::megdnn::param::RNN packed_param_0): num_layers(packed_param_0.num_layers), bidirectional(packed_param_0.bidirectional), bias(packed_param_0.bias), hidden_size(packed_param_0.hidden_size), dropout(packed_param_0.dropout), nonlineMode(packed_param_0.nonlineMode), fwd_mode(packed_param_0.fwd_mode) {}
    ::megdnn::param::RNN param() const {
        return {num_layers, bidirectional, bias, hidden_size, dropout, nonlineMode, fwd_mode};
    }
};

class RNNCell : public OpDefImplBase<RNNCell> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using NonlineMode = ::megdnn::param::RNNCell::NonlineMode;
    NonlineMode nonlineMode = ::megdnn::param::RNNCell::NonlineMode::IDENTITY;
    RNNCell() = default;
    RNNCell(NonlineMode nonlineMode_, std::string scope_ = {}): nonlineMode(nonlineMode_) { set_scope(scope_); }
    RNNCell(::megdnn::param::RNNCell packed_param_0): nonlineMode(packed_param_0.nonlineMode) {}
    ::megdnn::param::RNNCell param() const {
        return {nonlineMode};
    }
};

class ROIAlign : public OpDefImplBase<ROIAlign> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using Mode = ::megdnn::param::ROIAlign::Mode;
    using Format = ::megdnn::param::ROIAlign::Format;
    Mode mode = ::megdnn::param::ROIAlign::Mode::MAX;
    Format format = ::megdnn::param::ROIAlign::Format::NCHW;
    float spatial_scale = 1.0;
    float offset = 0.0;
    uint32_t pooled_height = 1;
    uint32_t pooled_width = 1;
    uint32_t sample_height = 2;
    uint32_t sample_width = 2;
    ROIAlign() = default;
    ROIAlign(Mode mode_, Format format_, float spatial_scale_, float offset_, uint32_t pooled_height_, uint32_t pooled_width_, uint32_t sample_height_, uint32_t sample_width_, std::string scope_ = {}): mode(mode_), format(format_), spatial_scale(spatial_scale_), offset(offset_), pooled_height(pooled_height_), pooled_width(pooled_width_), sample_height(sample_height_), sample_width(sample_width_) { set_scope(scope_); }
    ROIAlign(::megdnn::param::ROIAlign packed_param_0): mode(packed_param_0.mode), format(packed_param_0.format), spatial_scale(packed_param_0.spatial_scale), offset(packed_param_0.offset), pooled_height(packed_param_0.pooled_height), pooled_width(packed_param_0.pooled_width), sample_height(packed_param_0.sample_height), sample_width(packed_param_0.sample_width) {}
    ::megdnn::param::ROIAlign param() const {
        return {mode, format, spatial_scale, offset, pooled_height, pooled_width, sample_height, sample_width};
    }
};

class ROIPooling : public OpDefImplBase<ROIPooling> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using Mode = ::megdnn::param::ROIPooling::Mode;
    Mode mode = ::megdnn::param::ROIPooling::Mode::MAX;
    float scale = 1.f;
    ROIPooling() = default;
    ROIPooling(Mode mode_, float scale_, std::string scope_ = {}): mode(mode_), scale(scale_) { set_scope(scope_); }
    ROIPooling(::megdnn::param::ROIPooling packed_param_0): mode(packed_param_0.mode), scale(packed_param_0.scale) {}
    ::megdnn::param::ROIPooling param() const {
        return {mode, scale};
    }
};

class Reduce : public OpDefImplBase<Reduce> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using Mode = ::megdnn::param::Reduce::Mode;
    using DataType = ::megdnn::param::Reduce::DataType;
    Mode mode = ::megdnn::param::Reduce::Mode::SUM;
    int32_t axis = 2147483647;
    DataType data_type = ::megdnn::param::Reduce::DataType::DEFAULT;
    bool keepdim = true;
    Reduce() = default;
    Reduce(Mode mode_, int32_t axis_, DataType data_type_, bool keepdim_, std::string scope_ = {}): mode(mode_), axis(axis_), data_type(data_type_), keepdim(keepdim_) { set_scope(scope_); }
    Reduce(::megdnn::param::Reduce packed_param_0, bool keepdim_): mode(packed_param_0.mode), axis(packed_param_0.axis), data_type(packed_param_0.data_type), keepdim(keepdim_) {}
    ::megdnn::param::Reduce param() const {
        return {mode, axis, data_type};
    }
};

class RegionRestrictedConvolution : public OpDefImplBase<RegionRestrictedConvolution> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using Mode = ::megdnn::param::Convolution::Mode;
    using Sparse = ::megdnn::param::Convolution::Sparse;
    using Format = ::megdnn::param::Convolution::Format;
    using ComputeMode = ::megdnn::param::Convolution::ComputeMode;
    Mode mode = ::megdnn::param::Convolution::Mode::CROSS_CORRELATION;
    uint32_t pad_h = 0;
    uint32_t pad_w = 0;
    uint32_t stride_h = 1;
    uint32_t stride_w = 1;
    uint32_t dilate_h = 1;
    uint32_t dilate_w = 1;
    Sparse sparse = ::megdnn::param::Convolution::Sparse::DENSE;
    Format format = ::megdnn::param::Convolution::Format::NCHW;
    ComputeMode compute_mode = ::megdnn::param::Convolution::ComputeMode::DEFAULT;
    RegionRestrictedConvolution() = default;
    RegionRestrictedConvolution(Mode mode_, uint32_t pad_h_, uint32_t pad_w_, uint32_t stride_h_, uint32_t stride_w_, uint32_t dilate_h_, uint32_t dilate_w_, Sparse sparse_, Format format_, ComputeMode compute_mode_, std::string scope_ = {}): mode(mode_), pad_h(pad_h_), pad_w(pad_w_), stride_h(stride_h_), stride_w(stride_w_), dilate_h(dilate_h_), dilate_w(dilate_w_), sparse(sparse_), format(format_), compute_mode(compute_mode_) { set_scope(scope_); }
    RegionRestrictedConvolution(::megdnn::param::Convolution packed_param_0): mode(packed_param_0.mode), pad_h(packed_param_0.pad_h), pad_w(packed_param_0.pad_w), stride_h(packed_param_0.stride_h), stride_w(packed_param_0.stride_w), dilate_h(packed_param_0.dilate_h), dilate_w(packed_param_0.dilate_w), sparse(packed_param_0.sparse), format(packed_param_0.format), compute_mode(packed_param_0.compute_mode) {}
    ::megdnn::param::Convolution param() const {
        return {mode, pad_h, pad_w, stride_h, stride_w, dilate_h, dilate_w, sparse, format, compute_mode};
    }
};

class RegionRestrictedConvolutionBackwardData : public OpDefImplBase<RegionRestrictedConvolutionBackwardData> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using Mode = ::megdnn::param::Convolution::Mode;
    using Sparse = ::megdnn::param::Convolution::Sparse;
    using Format = ::megdnn::param::Convolution::Format;
    using ComputeMode = ::megdnn::param::Convolution::ComputeMode;
    Mode mode = ::megdnn::param::Convolution::Mode::CROSS_CORRELATION;
    uint32_t pad_h = 0;
    uint32_t pad_w = 0;
    uint32_t stride_h = 1;
    uint32_t stride_w = 1;
    uint32_t dilate_h = 1;
    uint32_t dilate_w = 1;
    Sparse sparse = ::megdnn::param::Convolution::Sparse::DENSE;
    Format format = ::megdnn::param::Convolution::Format::NCHW;
    ComputeMode compute_mode = ::megdnn::param::Convolution::ComputeMode::DEFAULT;
    RegionRestrictedConvolutionBackwardData() = default;
    RegionRestrictedConvolutionBackwardData(Mode mode_, uint32_t pad_h_, uint32_t pad_w_, uint32_t stride_h_, uint32_t stride_w_, uint32_t dilate_h_, uint32_t dilate_w_, Sparse sparse_, Format format_, ComputeMode compute_mode_, std::string scope_ = {}): mode(mode_), pad_h(pad_h_), pad_w(pad_w_), stride_h(stride_h_), stride_w(stride_w_), dilate_h(dilate_h_), dilate_w(dilate_w_), sparse(sparse_), format(format_), compute_mode(compute_mode_) { set_scope(scope_); }
    RegionRestrictedConvolutionBackwardData(::megdnn::param::Convolution packed_param_0): mode(packed_param_0.mode), pad_h(packed_param_0.pad_h), pad_w(packed_param_0.pad_w), stride_h(packed_param_0.stride_h), stride_w(packed_param_0.stride_w), dilate_h(packed_param_0.dilate_h), dilate_w(packed_param_0.dilate_w), sparse(packed_param_0.sparse), format(packed_param_0.format), compute_mode(packed_param_0.compute_mode) {}
    ::megdnn::param::Convolution param() const {
        return {mode, pad_h, pad_w, stride_h, stride_w, dilate_h, dilate_w, sparse, format, compute_mode};
    }
};

class Remap : public OpDefImplBase<Remap> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using InterpolationMode = ::megdnn::param::Remap::InterpolationMode;
    using BorderMode = ::megdnn::param::Remap::BorderMode;
    using Format = ::megdnn::param::Remap::Format;
    InterpolationMode imode = ::megdnn::param::Remap::InterpolationMode::LINEAR;
    BorderMode border_type = ::megdnn::param::Remap::BorderMode::REPLICATE;
    Format format = ::megdnn::param::Remap::Format::NHWC;
    float scalar = 0.f;
    Remap() = default;
    Remap(InterpolationMode imode_, BorderMode border_type_, Format format_, float scalar_, std::string scope_ = {}): imode(imode_), border_type(border_type_), format(format_), scalar(scalar_) { set_scope(scope_); }
    Remap(::megdnn::param::Remap packed_param_0): imode(packed_param_0.imode), border_type(packed_param_0.border_type), format(packed_param_0.format), scalar(packed_param_0.scalar) {}
    ::megdnn::param::Remap param() const {
        return {imode, border_type, format, scalar};
    }
};

class RemoteRecv : public OpDefImplBase<RemoteRecv> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    std::string key;
    std::string addr;
    uint32_t port;
    uint32_t rank_from;
    ::mgb::CompNode cn;
    std::vector<int32_t> shape;
    ::megdnn::DType dtype;
    std::string backend;
    RemoteRecv() = default;
    RemoteRecv(std::string key_, std::string addr_, uint32_t port_, uint32_t rank_from_, ::mgb::CompNode cn_, std::vector<int32_t> shape_, ::megdnn::DType dtype_, std::string backend_, std::string scope_ = {}): key(key_), addr(addr_), port(port_), rank_from(rank_from_), cn(cn_), shape(shape_), dtype(dtype_), backend(backend_) { set_scope(scope_); }
};

class RemoteSend : public OpDefImplBase<RemoteSend> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    std::string key;
    std::string addr;
    uint32_t port;
    uint32_t rank_to;
    std::string backend;
    RemoteSend() = default;
    RemoteSend(std::string key_, std::string addr_, uint32_t port_, uint32_t rank_to_, std::string backend_, std::string scope_ = {}): key(key_), addr(addr_), port(port_), rank_to(rank_to_), backend(backend_) { set_scope(scope_); }
};

class RemoveAxis : public OpDefImplBase<RemoveAxis> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    std::vector<int32_t> axis;
    RemoveAxis() = default;
    RemoveAxis(std::vector<int32_t> axis_, std::string scope_ = {}): axis(axis_) { set_scope(scope_); }
};

class Reshape : public OpDefImplBase<Reshape> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    int32_t axis = ::megdnn::param::OptionalAxisV1::INVALID_AXIS;
    std::vector<int32_t> shape;
    Reshape() = default;
    Reshape(int32_t axis_, std::vector<int32_t> shape_, std::string scope_ = {}): axis(axis_), shape(shape_) { set_scope(scope_); }
    Reshape(::megdnn::param::OptionalAxisV1 packed_param_0, std::vector<int32_t> shape_): axis(packed_param_0.axis), shape(shape_) {}
    ::megdnn::param::OptionalAxisV1 param() const {
        return {axis};
    }
};

class Resize : public OpDefImplBase<Resize> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using InterpolationMode = ::megdnn::param::Resize::InterpolationMode;
    using Format = ::megdnn::param::Resize::Format;
    InterpolationMode imode = ::megdnn::param::Resize::InterpolationMode::LINEAR;
    Format format = ::megdnn::param::Resize::Format::NHWC;
    Resize() = default;
    Resize(InterpolationMode imode_, Format format_, std::string scope_ = {}): imode(imode_), format(format_) { set_scope(scope_); }
    Resize(::megdnn::param::Resize packed_param_0): imode(packed_param_0.imode), format(packed_param_0.format) {}
    ::megdnn::param::Resize param() const {
        return {imode, format};
    }
};

class SVD : public OpDefImplBase<SVD> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    bool full_matrices = false;
    bool compute_uv = true;
    SVD() = default;
    SVD(bool full_matrices_, bool compute_uv_, std::string scope_ = {}): full_matrices(full_matrices_), compute_uv(compute_uv_) { set_scope(scope_); }
    SVD(::megdnn::param::SVD packed_param_0): full_matrices(packed_param_0.full_matrices), compute_uv(packed_param_0.compute_uv) {}
    ::megdnn::param::SVD param() const {
        return {full_matrices, compute_uv};
    }
};

class SetMeshIndexing : public OpDefImplBase<SetMeshIndexing> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    std::vector<std::tuple<int8_t, bool, bool, bool, bool>> items;
    SetMeshIndexing() = default;
    SetMeshIndexing(std::vector<std::tuple<int8_t, bool, bool, bool, bool>> items_, std::string scope_ = {}): items(items_) { set_scope(scope_); }
};

class SetSubtensor : public OpDefImplBase<SetSubtensor> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    std::vector<std::tuple<int8_t, bool, bool, bool, bool>> items;
    SetSubtensor() = default;
    SetSubtensor(std::vector<std::tuple<int8_t, bool, bool, bool, bool>> items_, std::string scope_ = {}): items(items_) { set_scope(scope_); }
};

class ShuffleRNG : public OpDefImplBase<ShuffleRNG> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    uint64_t seed = 0;
    size_t handle;
    ShuffleRNG() = default;
    ShuffleRNG(uint64_t seed_, size_t handle_, std::string scope_ = {}): seed(seed_), handle(handle_) { set_scope(scope_); }
    ShuffleRNG(::megdnn::param::ShuffleRNG packed_param_0, size_t handle_): seed(packed_param_0.seed), handle(handle_) {}
    ::megdnn::param::ShuffleRNG param() const {
        return {seed};
    }
};

class SlidingWindowTranspose : public OpDefImplBase<SlidingWindowTranspose> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    uint32_t out_h = 0;
    uint32_t out_w = 0;
    uint32_t pad_h = 0;
    uint32_t pad_w = 0;
    uint32_t stride_h = 1;
    uint32_t stride_w = 1;
    uint32_t dilate_h = 1;
    uint32_t dilate_w = 1;
    uint32_t window_h = 3;
    uint32_t window_w = 3;
    SlidingWindowTranspose() = default;
    SlidingWindowTranspose(uint32_t out_h_, uint32_t out_w_, uint32_t pad_h_, uint32_t pad_w_, uint32_t stride_h_, uint32_t stride_w_, uint32_t dilate_h_, uint32_t dilate_w_, uint32_t window_h_, uint32_t window_w_, std::string scope_ = {}): out_h(out_h_), out_w(out_w_), pad_h(pad_h_), pad_w(pad_w_), stride_h(stride_h_), stride_w(stride_w_), dilate_h(dilate_h_), dilate_w(dilate_w_), window_h(window_h_), window_w(window_w_) { set_scope(scope_); }
    SlidingWindowTranspose(::megdnn::param::SlidingWindowTranspose packed_param_0): out_h(packed_param_0.out_h), out_w(packed_param_0.out_w), pad_h(packed_param_0.pad_h), pad_w(packed_param_0.pad_w), stride_h(packed_param_0.stride_h), stride_w(packed_param_0.stride_w), dilate_h(packed_param_0.dilate_h), dilate_w(packed_param_0.dilate_w), window_h(packed_param_0.window_h), window_w(packed_param_0.window_w) {}
    ::megdnn::param::SlidingWindowTranspose param() const {
        return {out_h, out_w, pad_h, pad_w, stride_h, stride_w, dilate_h, dilate_w, window_h, window_w};
    }
};

class Softmax : public OpDefImplBase<Softmax> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    int32_t axis = -1;
    Softmax() = default;
    Softmax(int32_t axis_, std::string scope_ = {}): axis(axis_) { set_scope(scope_); }
    Softmax(::megdnn::param::Softmax packed_param_0): axis(packed_param_0.axis) {}
    ::megdnn::param::Softmax param() const {
        return {axis};
    }
};

class Split : public OpDefImplBase<Split> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    int32_t axis;
    int32_t nsections;
    Split() = default;
    Split(int32_t axis_, int32_t nsections_, std::string scope_ = {}): axis(axis_), nsections(nsections_) { set_scope(scope_); }
    Split(::megdnn::param::Empty, int32_t axis_, int32_t nsections_): axis(axis_), nsections(nsections_) {}
    ::megdnn::param::Empty param() const {
        return {};
    }
};

class Subtensor : public OpDefImplBase<Subtensor> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    std::vector<std::tuple<int8_t, bool, bool, bool, bool>> items;
    Subtensor() = default;
    Subtensor(std::vector<std::tuple<int8_t, bool, bool, bool, bool>> items_, std::string scope_ = {}): items(items_) { set_scope(scope_); }
};

class TQT : public OpDefImplBase<TQT> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    int32_t qmin = -2147483648;
    int32_t qmax = 2147483647;
    TQT() = default;
    TQT(int32_t qmin_, int32_t qmax_, std::string scope_ = {}): qmin(qmin_), qmax(qmax_) { set_scope(scope_); }
    TQT(::megdnn::param::TQT packed_param_0): qmin(packed_param_0.qmin), qmax(packed_param_0.qmax) {}
    ::megdnn::param::TQT param() const {
        return {qmin, qmax};
    }
};

class TensorRTRuntime : public OpDefImplBase<TensorRTRuntime> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    std::string buf;
    size_t buf_size;
    TensorRTRuntime() = default;
    TensorRTRuntime(std::string buf_, size_t buf_size_, std::string scope_ = {}): buf(buf_), buf_size(buf_size_) { set_scope(scope_); }
};

class TopK : public OpDefImplBase<TopK> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using Mode = ::megdnn::param::TopK::Mode;
    Mode mode = ::megdnn::param::TopK::Mode::KTH_ONLY;
    TopK() = default;
    TopK(Mode mode_, std::string scope_ = {}): mode(mode_) { set_scope(scope_); }
    TopK(::megdnn::param::TopK packed_param_0): mode(packed_param_0.mode) {}
    ::megdnn::param::TopK param() const {
        return {mode};
    }
};

class TypeCvt : public OpDefImplBase<TypeCvt> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    ::megdnn::DType dtype;
    TypeCvt() = default;
    TypeCvt(::megdnn::DType dtype_, std::string scope_ = {}): dtype(dtype_) { set_scope(scope_); }
};

class UniformRNG : public OpDefImplBase<UniformRNG> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    uint64_t seed = 0;
    ::megdnn::DType dtype = megdnn::DType::from_enum(megdnn::DTypeEnum::Float32);
    size_t handle;
    UniformRNG() = default;
    UniformRNG(uint64_t seed_, ::megdnn::DType dtype_, size_t handle_, std::string scope_ = {}): seed(seed_), dtype(dtype_), handle(handle_) { set_scope(scope_); }
};

class WarpAffine : public OpDefImplBase<WarpAffine> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using InterpolationMode = ::megdnn::param::WarpAffine::InterpolationMode;
    using BorderMode = ::megdnn::param::WarpAffine::BorderMode;
    using Format = ::megdnn::param::WarpAffine::Format;
    InterpolationMode imode = ::megdnn::param::WarpAffine::InterpolationMode::LINEAR;
    BorderMode border_mode = ::megdnn::param::WarpAffine::BorderMode::REPLICATE;
    float border_val = .0f;
    Format format = ::megdnn::param::WarpAffine::Format::NHWC;
    WarpAffine() = default;
    WarpAffine(InterpolationMode imode_, BorderMode border_mode_, float border_val_, Format format_, std::string scope_ = {}): imode(imode_), border_mode(border_mode_), border_val(border_val_), format(format_) { set_scope(scope_); }
    WarpAffine(::megdnn::param::WarpAffine packed_param_0): imode(packed_param_0.imode), border_mode(packed_param_0.border_mode), border_val(packed_param_0.border_val), format(packed_param_0.format) {}
    ::megdnn::param::WarpAffine param() const {
        return {imode, border_mode, border_val, format};
    }
};

class WarpPerspective : public OpDefImplBase<WarpPerspective> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using InterpolationMode = ::megdnn::param::WarpPerspective::InterpolationMode;
    using BorderMode = ::megdnn::param::WarpPerspective::BorderMode;
    using Format = ::megdnn::param::WarpPerspective::Format;
    InterpolationMode imode = ::megdnn::param::WarpPerspective::InterpolationMode::LINEAR;
    BorderMode bmode = ::megdnn::param::WarpPerspective::BorderMode::REPLICATE;
    Format format = ::megdnn::param::WarpPerspective::Format::NCHW;
    float border_val = .0f;
    WarpPerspective() = default;
    WarpPerspective(InterpolationMode imode_, BorderMode bmode_, Format format_, float border_val_, std::string scope_ = {}): imode(imode_), bmode(bmode_), format(format_), border_val(border_val_) { set_scope(scope_); }
    WarpPerspective(::megdnn::param::WarpPerspective packed_param_0): imode(packed_param_0.imode), bmode(packed_param_0.bmode), format(packed_param_0.format), border_val(packed_param_0.border_val) {}
    ::megdnn::param::WarpPerspective param() const {
        return {imode, bmode, format, border_val};
    }
};

class WarpPerspectiveBackwardData : public OpDefImplBase<WarpPerspectiveBackwardData> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using InterpolationMode = ::megdnn::param::WarpPerspective::InterpolationMode;
    using BorderMode = ::megdnn::param::WarpPerspective::BorderMode;
    using Format = ::megdnn::param::WarpPerspective::Format;
    InterpolationMode imode = ::megdnn::param::WarpPerspective::InterpolationMode::LINEAR;
    BorderMode bmode = ::megdnn::param::WarpPerspective::BorderMode::REPLICATE;
    Format format = ::megdnn::param::WarpPerspective::Format::NCHW;
    float border_val = .0f;
    WarpPerspectiveBackwardData() = default;
    WarpPerspectiveBackwardData(InterpolationMode imode_, BorderMode bmode_, Format format_, float border_val_, std::string scope_ = {}): imode(imode_), bmode(bmode_), format(format_), border_val(border_val_) { set_scope(scope_); }
    WarpPerspectiveBackwardData(::megdnn::param::WarpPerspective packed_param_0): imode(packed_param_0.imode), bmode(packed_param_0.bmode), format(packed_param_0.format), border_val(packed_param_0.border_val) {}
    ::megdnn::param::WarpPerspective param() const {
        return {imode, bmode, format, border_val};
    }
};

class WarpPerspectiveBackwardMat : public OpDefImplBase<WarpPerspectiveBackwardMat> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using InterpolationMode = ::megdnn::param::WarpPerspective::InterpolationMode;
    using BorderMode = ::megdnn::param::WarpPerspective::BorderMode;
    using Format = ::megdnn::param::WarpPerspective::Format;
    InterpolationMode imode = ::megdnn::param::WarpPerspective::InterpolationMode::LINEAR;
    BorderMode bmode = ::megdnn::param::WarpPerspective::BorderMode::REPLICATE;
    Format format = ::megdnn::param::WarpPerspective::Format::NCHW;
    float border_val = .0f;
    WarpPerspectiveBackwardMat() = default;
    WarpPerspectiveBackwardMat(InterpolationMode imode_, BorderMode bmode_, Format format_, float border_val_, std::string scope_ = {}): imode(imode_), bmode(bmode_), format(format_), border_val(border_val_) { set_scope(scope_); }
    WarpPerspectiveBackwardMat(::megdnn::param::WarpPerspective packed_param_0): imode(packed_param_0.imode), bmode(packed_param_0.bmode), format(packed_param_0.format), border_val(packed_param_0.border_val) {}
    ::megdnn::param::WarpPerspective param() const {
        return {imode, bmode, format, border_val};
    }
};

// clang-format on
