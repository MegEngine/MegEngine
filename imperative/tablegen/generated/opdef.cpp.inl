// clang-format off
MGB_DYN_TYPE_OBJ_FINAL_IMPL(AdaptivePooling);

namespace {
size_t AdaptivePooling_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<AdaptivePooling>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.mode));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.format));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.shape));
    return val;
}
bool AdaptivePooling_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<AdaptivePooling>(),
         &&b_ = rhs_.cast_final_safe<AdaptivePooling>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.mode != b_.mode) return false;
    if (a_.format != b_.format) return false;
    if (a_.shape != b_.shape) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> AdaptivePooling_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<AdaptivePooling>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.mode){
    case AdaptivePooling::Mode::MAX:
        props_.emplace_back("mode", "MAX");
        break;
    case AdaptivePooling::Mode::AVERAGE:
        props_.emplace_back("mode", "AVERAGE");
        break;
    case AdaptivePooling::Mode::AVERAGE_COUNT_EXCLUDE_PADDING:
        props_.emplace_back("mode", "AVERAGE_COUNT_EXCLUDE_PADDING");
        break;
    default:
        props_.emplace_back("mode", "INVALID");
        break;
    }
    switch (op_.format){
    case AdaptivePooling::Format::NCHW:
        props_.emplace_back("format", "NCHW");
        break;
    case AdaptivePooling::Format::NHWC:
        props_.emplace_back("format", "NHWC");
        break;
    case AdaptivePooling::Format::NHWCD4:
        props_.emplace_back("format", "NHWCD4");
        break;
    case AdaptivePooling::Format::NCHW4:
        props_.emplace_back("format", "NCHW4");
        break;
    case AdaptivePooling::Format::NCHW8:
        props_.emplace_back("format", "NCHW8");
        break;
    case AdaptivePooling::Format::NCHW32:
        props_.emplace_back("format", "NCHW32");
        break;
    case AdaptivePooling::Format::NCHW88:
        props_.emplace_back("format", "NCHW88");
        break;
    case AdaptivePooling::Format::NCHW44:
        props_.emplace_back("format", "NCHW44");
        break;
    case AdaptivePooling::Format::NCHW44_DOT:
        props_.emplace_back("format", "NCHW44_DOT");
        break;
    case AdaptivePooling::Format::NCHW4_NCHW32:
        props_.emplace_back("format", "NCHW4_NCHW32");
        break;
    case AdaptivePooling::Format::NCHW32_NCHW4:
        props_.emplace_back("format", "NCHW32_NCHW4");
        break;
    case AdaptivePooling::Format::NCHW4_NCHW:
        props_.emplace_back("format", "NCHW4_NCHW");
        break;
    case AdaptivePooling::Format::NHWC_NCHW:
        props_.emplace_back("format", "NHWC_NCHW");
        break;
    case AdaptivePooling::Format::NHWC_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NHWC_NCHW4_IC_SMALL");
        break;
    case AdaptivePooling::Format::NCHW_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NCHW_NCHW4_IC_SMALL");
        break;
    case AdaptivePooling::Format::CHWN4:
        props_.emplace_back("format", "CHWN4");
        break;
    case AdaptivePooling::Format::NCHW64:
        props_.emplace_back("format", "NCHW64");
        break;
    case AdaptivePooling::Format::NCHW4_NHWC:
        props_.emplace_back("format", "NCHW4_NHWC");
        break;
    default:
        props_.emplace_back("format", "INVALID");
        break;
    }
    props_.emplace_back("shape", "{std::vector}");
    return props_;
}
std::string AdaptivePooling_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<AdaptivePooling>();
    static_cast<void>(op_);
    return "AdaptivePooling";
}
} // anonymous namespace
OP_TRAIT_REG(AdaptivePooling, AdaptivePooling)
    .hash(AdaptivePooling_hash_impl)
    .is_same_st(AdaptivePooling_is_same_st_impl)
    .props(AdaptivePooling_props_impl)
    .make_name(AdaptivePooling_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(AddAxis);

namespace {
size_t AddAxis_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<AddAxis>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.axis));
    return val;
}
bool AddAxis_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<AddAxis>(),
         &&b_ = rhs_.cast_final_safe<AddAxis>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.axis != b_.axis) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> AddAxis_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<AddAxis>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("axis", "{std::vector}");
    return props_;
}
std::string AddAxis_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<AddAxis>();
    static_cast<void>(op_);
    return "AddAxis";
}
} // anonymous namespace
OP_TRAIT_REG(AddAxis, AddAxis)
    .hash(AddAxis_hash_impl)
    .is_same_st(AddAxis_is_same_st_impl)
    .props(AddAxis_props_impl)
    .make_name(AddAxis_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Argmax);

namespace {
size_t Argmax_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Argmax>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.axis));
    return val;
}
bool Argmax_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Argmax>(),
         &&b_ = rhs_.cast_final_safe<Argmax>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.axis != b_.axis) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Argmax_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Argmax>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("axis", std::to_string(op_.axis));
    return props_;
}
std::string Argmax_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Argmax>();
    static_cast<void>(op_);
    return "Argmax";
}
} // anonymous namespace
OP_TRAIT_REG(Argmax, Argmax)
    .hash(Argmax_hash_impl)
    .is_same_st(Argmax_is_same_st_impl)
    .props(Argmax_props_impl)
    .make_name(Argmax_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Argmin);

namespace {
size_t Argmin_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Argmin>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.axis));
    return val;
}
bool Argmin_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Argmin>(),
         &&b_ = rhs_.cast_final_safe<Argmin>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.axis != b_.axis) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Argmin_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Argmin>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("axis", std::to_string(op_.axis));
    return props_;
}
std::string Argmin_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Argmin>();
    static_cast<void>(op_);
    return "Argmin";
}
} // anonymous namespace
OP_TRAIT_REG(Argmin, Argmin)
    .hash(Argmin_hash_impl)
    .is_same_st(Argmin_is_same_st_impl)
    .props(Argmin_props_impl)
    .make_name(Argmin_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Argsort);

namespace {
size_t Argsort_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Argsort>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.order));
    return val;
}
bool Argsort_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Argsort>(),
         &&b_ = rhs_.cast_final_safe<Argsort>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.order != b_.order) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Argsort_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Argsort>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.order){
    case Argsort::Order::ASCENDING:
        props_.emplace_back("order", "ASCENDING");
        break;
    case Argsort::Order::DESCENDING:
        props_.emplace_back("order", "DESCENDING");
        break;
    default:
        props_.emplace_back("order", "INVALID");
        break;
    }
    return props_;
}
std::string Argsort_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Argsort>();
    static_cast<void>(op_);
    return "Argsort";
}
} // anonymous namespace
OP_TRAIT_REG(Argsort, Argsort)
    .hash(Argsort_hash_impl)
    .is_same_st(Argsort_is_same_st_impl)
    .props(Argsort_props_impl)
    .make_name(Argsort_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(AssertEqual);

namespace {
size_t AssertEqual_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<AssertEqual>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.maxerr));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.verbose));
    return val;
}
bool AssertEqual_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<AssertEqual>(),
         &&b_ = rhs_.cast_final_safe<AssertEqual>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.maxerr != b_.maxerr) return false;
    if (a_.verbose != b_.verbose) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> AssertEqual_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<AssertEqual>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("maxerr", std::to_string(op_.maxerr));
    props_.emplace_back("verbose", std::to_string(op_.verbose));
    return props_;
}
std::string AssertEqual_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<AssertEqual>();
    static_cast<void>(op_);
    return "AssertEqual";
}
} // anonymous namespace
OP_TRAIT_REG(AssertEqual, AssertEqual)
    .hash(AssertEqual_hash_impl)
    .is_same_st(AssertEqual_is_same_st_impl)
    .props(AssertEqual_props_impl)
    .make_name(AssertEqual_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(AtlasRuntime);

namespace {
size_t AtlasRuntime_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<AtlasRuntime>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.buf));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.buf_size));
    return val;
}
bool AtlasRuntime_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<AtlasRuntime>(),
         &&b_ = rhs_.cast_final_safe<AtlasRuntime>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.buf != b_.buf) return false;
    if (a_.buf_size != b_.buf_size) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> AtlasRuntime_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<AtlasRuntime>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("buf", op_.buf);
    props_.emplace_back("buf_size", std::to_string(op_.buf_size));
    return props_;
}
std::string AtlasRuntime_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<AtlasRuntime>();
    static_cast<void>(op_);
    return "AtlasRuntime";
}
} // anonymous namespace
OP_TRAIT_REG(AtlasRuntime, AtlasRuntime)
    .hash(AtlasRuntime_hash_impl)
    .is_same_st(AtlasRuntime_is_same_st_impl)
    .props(AtlasRuntime_props_impl)
    .make_name(AtlasRuntime_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Barrier);

namespace {
size_t Barrier_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Barrier>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.comp_node));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.nr_outputs));
    return val;
}
bool Barrier_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Barrier>(),
         &&b_ = rhs_.cast_final_safe<Barrier>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.comp_node != b_.comp_node) return false;
    if (a_.nr_outputs != b_.nr_outputs) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Barrier_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Barrier>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("comp_node", op_.comp_node.to_string());
    props_.emplace_back("nr_outputs", std::to_string(op_.nr_outputs));
    return props_;
}
std::string Barrier_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Barrier>();
    static_cast<void>(op_);
    return "Barrier";
}
} // anonymous namespace
OP_TRAIT_REG(Barrier, Barrier)
    .hash(Barrier_hash_impl)
    .is_same_st(Barrier_is_same_st_impl)
    .props(Barrier_props_impl)
    .make_name(Barrier_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(BatchConvBias);

namespace {
size_t BatchConvBias_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<BatchConvBias>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.nonlineMode));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.mode));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pad_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pad_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dilate_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dilate_w));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.sparse));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.format));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.compute_mode));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.strategy));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.workspace_limit));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dtype.handle()));
    return val;
}
bool BatchConvBias_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<BatchConvBias>(),
         &&b_ = rhs_.cast_final_safe<BatchConvBias>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.nonlineMode != b_.nonlineMode) return false;
    if (a_.mode != b_.mode) return false;
    if (a_.pad_h != b_.pad_h) return false;
    if (a_.pad_w != b_.pad_w) return false;
    if (a_.stride_h != b_.stride_h) return false;
    if (a_.stride_w != b_.stride_w) return false;
    if (a_.dilate_h != b_.dilate_h) return false;
    if (a_.dilate_w != b_.dilate_w) return false;
    if (a_.sparse != b_.sparse) return false;
    if (a_.format != b_.format) return false;
    if (a_.compute_mode != b_.compute_mode) return false;
    if (a_.strategy != b_.strategy) return false;
    if (a_.workspace_limit != b_.workspace_limit) return false;
    if (a_.dtype != b_.dtype) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> BatchConvBias_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<BatchConvBias>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.nonlineMode){
    case BatchConvBias::NonlineMode::IDENTITY:
        props_.emplace_back("nonlineMode", "IDENTITY");
        break;
    case BatchConvBias::NonlineMode::RELU:
        props_.emplace_back("nonlineMode", "RELU");
        break;
    case BatchConvBias::NonlineMode::SIGMOID:
        props_.emplace_back("nonlineMode", "SIGMOID");
        break;
    case BatchConvBias::NonlineMode::H_SWISH:
        props_.emplace_back("nonlineMode", "H_SWISH");
        break;
    default:
        props_.emplace_back("nonlineMode", "INVALID");
        break;
    }
    switch (op_.mode){
    case BatchConvBias::Mode::CROSS_CORRELATION:
        props_.emplace_back("mode", "CROSS_CORRELATION");
        break;
    case BatchConvBias::Mode::CONVOLUTION:
        props_.emplace_back("mode", "CONVOLUTION");
        break;
    default:
        props_.emplace_back("mode", "INVALID");
        break;
    }
    props_.emplace_back("pad_h", std::to_string(op_.pad_h));
    props_.emplace_back("pad_w", std::to_string(op_.pad_w));
    props_.emplace_back("stride_h", std::to_string(op_.stride_h));
    props_.emplace_back("stride_w", std::to_string(op_.stride_w));
    props_.emplace_back("dilate_h", std::to_string(op_.dilate_h));
    props_.emplace_back("dilate_w", std::to_string(op_.dilate_w));
    switch (op_.sparse){
    case BatchConvBias::Sparse::DENSE:
        props_.emplace_back("sparse", "DENSE");
        break;
    case BatchConvBias::Sparse::GROUP:
        props_.emplace_back("sparse", "GROUP");
        break;
    default:
        props_.emplace_back("sparse", "INVALID");
        break;
    }
    switch (op_.format){
    case BatchConvBias::Format::NCHW:
        props_.emplace_back("format", "NCHW");
        break;
    case BatchConvBias::Format::NHWC:
        props_.emplace_back("format", "NHWC");
        break;
    case BatchConvBias::Format::NHWCD4:
        props_.emplace_back("format", "NHWCD4");
        break;
    case BatchConvBias::Format::NCHW4:
        props_.emplace_back("format", "NCHW4");
        break;
    case BatchConvBias::Format::NCHW8:
        props_.emplace_back("format", "NCHW8");
        break;
    case BatchConvBias::Format::NCHW32:
        props_.emplace_back("format", "NCHW32");
        break;
    case BatchConvBias::Format::NCHW88:
        props_.emplace_back("format", "NCHW88");
        break;
    case BatchConvBias::Format::NCHW44:
        props_.emplace_back("format", "NCHW44");
        break;
    case BatchConvBias::Format::NCHW44_DOT:
        props_.emplace_back("format", "NCHW44_DOT");
        break;
    case BatchConvBias::Format::NCHW4_NCHW32:
        props_.emplace_back("format", "NCHW4_NCHW32");
        break;
    case BatchConvBias::Format::NCHW32_NCHW4:
        props_.emplace_back("format", "NCHW32_NCHW4");
        break;
    case BatchConvBias::Format::NCHW4_NCHW:
        props_.emplace_back("format", "NCHW4_NCHW");
        break;
    case BatchConvBias::Format::NHWC_NCHW:
        props_.emplace_back("format", "NHWC_NCHW");
        break;
    case BatchConvBias::Format::NHWC_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NHWC_NCHW4_IC_SMALL");
        break;
    case BatchConvBias::Format::NCHW_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NCHW_NCHW4_IC_SMALL");
        break;
    case BatchConvBias::Format::CHWN4:
        props_.emplace_back("format", "CHWN4");
        break;
    case BatchConvBias::Format::NCHW64:
        props_.emplace_back("format", "NCHW64");
        break;
    case BatchConvBias::Format::NCHW4_NHWC:
        props_.emplace_back("format", "NCHW4_NHWC");
        break;
    default:
        props_.emplace_back("format", "INVALID");
        break;
    }
    switch (op_.compute_mode){
    case BatchConvBias::ComputeMode::DEFAULT:
        props_.emplace_back("compute_mode", "DEFAULT");
        break;
    case BatchConvBias::ComputeMode::FLOAT32:
        props_.emplace_back("compute_mode", "FLOAT32");
        break;
    default:
        props_.emplace_back("compute_mode", "INVALID");
        break;
    }
    switch (op_.strategy){
    case BatchConvBias::Strategy::HEURISTIC:
        props_.emplace_back("strategy", "HEURISTIC");
        break;
    case BatchConvBias::Strategy::PROFILE:
        props_.emplace_back("strategy", "PROFILE");
        break;
    case BatchConvBias::Strategy::REPRODUCIBLE:
        props_.emplace_back("strategy", "REPRODUCIBLE");
        break;
    case BatchConvBias::Strategy::OPTIMIZED:
        props_.emplace_back("strategy", "OPTIMIZED");
        break;
    default:
        props_.emplace_back("strategy", "INVALID");
        break;
    }
    props_.emplace_back("workspace_limit", std::to_string(op_.workspace_limit));
    props_.emplace_back("dtype", op_.dtype.name());
    return props_;
}
std::string BatchConvBias_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<BatchConvBias>();
    static_cast<void>(op_);
    return "BatchConvBias";
}
} // anonymous namespace
OP_TRAIT_REG(BatchConvBias, BatchConvBias)
    .hash(BatchConvBias_hash_impl)
    .is_same_st(BatchConvBias_is_same_st_impl)
    .props(BatchConvBias_props_impl)
    .make_name(BatchConvBias_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(BatchNorm);

namespace {
size_t BatchNorm_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<BatchNorm>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.param_dim));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.fwd_mode));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.epsilon));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.avg_factor));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.scale));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.bias));
    return val;
}
bool BatchNorm_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<BatchNorm>(),
         &&b_ = rhs_.cast_final_safe<BatchNorm>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.param_dim != b_.param_dim) return false;
    if (a_.fwd_mode != b_.fwd_mode) return false;
    if (a_.epsilon != b_.epsilon) return false;
    if (a_.avg_factor != b_.avg_factor) return false;
    if (a_.scale != b_.scale) return false;
    if (a_.bias != b_.bias) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> BatchNorm_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<BatchNorm>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.param_dim){
    case BatchNorm::ParamDim::DIM_11HW:
        props_.emplace_back("param_dim", "DIM_11HW");
        break;
    case BatchNorm::ParamDim::DIM_1CHW:
        props_.emplace_back("param_dim", "DIM_1CHW");
        break;
    case BatchNorm::ParamDim::DIM_1C11:
        props_.emplace_back("param_dim", "DIM_1C11");
        break;
    case BatchNorm::ParamDim::DIM_111C:
        props_.emplace_back("param_dim", "DIM_111C");
        break;
    default:
        props_.emplace_back("param_dim", "INVALID");
        break;
    }
    switch (op_.fwd_mode){
    case BatchNorm::FwdMode::TRAINING:
        props_.emplace_back("fwd_mode", "TRAINING");
        break;
    case BatchNorm::FwdMode::INFERENCE:
        props_.emplace_back("fwd_mode", "INFERENCE");
        break;
    default:
        props_.emplace_back("fwd_mode", "INVALID");
        break;
    }
    props_.emplace_back("epsilon", std::to_string(op_.epsilon));
    props_.emplace_back("avg_factor", std::to_string(op_.avg_factor));
    props_.emplace_back("scale", std::to_string(op_.scale));
    props_.emplace_back("bias", std::to_string(op_.bias));
    return props_;
}
std::string BatchNorm_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<BatchNorm>();
    static_cast<void>(op_);
    return "BatchNorm";
}
} // anonymous namespace
OP_TRAIT_REG(BatchNorm, BatchNorm)
    .hash(BatchNorm_hash_impl)
    .is_same_st(BatchNorm_is_same_st_impl)
    .props(BatchNorm_props_impl)
    .make_name(BatchNorm_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(BatchNormBackward);

namespace {
size_t BatchNormBackward_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<BatchNormBackward>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.param_dim));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.fwd_mode));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.epsilon));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.avg_factor));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.scale));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.bias));
    return val;
}
bool BatchNormBackward_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<BatchNormBackward>(),
         &&b_ = rhs_.cast_final_safe<BatchNormBackward>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.param_dim != b_.param_dim) return false;
    if (a_.fwd_mode != b_.fwd_mode) return false;
    if (a_.epsilon != b_.epsilon) return false;
    if (a_.avg_factor != b_.avg_factor) return false;
    if (a_.scale != b_.scale) return false;
    if (a_.bias != b_.bias) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> BatchNormBackward_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<BatchNormBackward>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.param_dim){
    case BatchNormBackward::ParamDim::DIM_11HW:
        props_.emplace_back("param_dim", "DIM_11HW");
        break;
    case BatchNormBackward::ParamDim::DIM_1CHW:
        props_.emplace_back("param_dim", "DIM_1CHW");
        break;
    case BatchNormBackward::ParamDim::DIM_1C11:
        props_.emplace_back("param_dim", "DIM_1C11");
        break;
    case BatchNormBackward::ParamDim::DIM_111C:
        props_.emplace_back("param_dim", "DIM_111C");
        break;
    default:
        props_.emplace_back("param_dim", "INVALID");
        break;
    }
    switch (op_.fwd_mode){
    case BatchNormBackward::FwdMode::TRAINING:
        props_.emplace_back("fwd_mode", "TRAINING");
        break;
    case BatchNormBackward::FwdMode::INFERENCE:
        props_.emplace_back("fwd_mode", "INFERENCE");
        break;
    default:
        props_.emplace_back("fwd_mode", "INVALID");
        break;
    }
    props_.emplace_back("epsilon", std::to_string(op_.epsilon));
    props_.emplace_back("avg_factor", std::to_string(op_.avg_factor));
    props_.emplace_back("scale", std::to_string(op_.scale));
    props_.emplace_back("bias", std::to_string(op_.bias));
    return props_;
}
std::string BatchNormBackward_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<BatchNormBackward>();
    static_cast<void>(op_);
    return "BatchNormBackward";
}
} // anonymous namespace
OP_TRAIT_REG(BatchNormBackward, BatchNormBackward)
    .hash(BatchNormBackward_hash_impl)
    .is_same_st(BatchNormBackward_is_same_st_impl)
    .props(BatchNormBackward_props_impl)
    .make_name(BatchNormBackward_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(BatchedIncrMeshIndexing);

namespace {
size_t BatchedIncrMeshIndexing_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<BatchedIncrMeshIndexing>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.items));
    return val;
}
bool BatchedIncrMeshIndexing_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<BatchedIncrMeshIndexing>(),
         &&b_ = rhs_.cast_final_safe<BatchedIncrMeshIndexing>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.items != b_.items) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> BatchedIncrMeshIndexing_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<BatchedIncrMeshIndexing>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("items", "{std::vector}");
    return props_;
}
std::string BatchedIncrMeshIndexing_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<BatchedIncrMeshIndexing>();
    static_cast<void>(op_);
    return "BatchedIncrMeshIndexing";
}
} // anonymous namespace
OP_TRAIT_REG(BatchedIncrMeshIndexing, BatchedIncrMeshIndexing)
    .hash(BatchedIncrMeshIndexing_hash_impl)
    .is_same_st(BatchedIncrMeshIndexing_is_same_st_impl)
    .props(BatchedIncrMeshIndexing_props_impl)
    .make_name(BatchedIncrMeshIndexing_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(BatchedMatrixMul);

namespace {
size_t BatchedMatrixMul_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<BatchedMatrixMul>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.transposeA));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.transposeB));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.compute_mode));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.format));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.strategy));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.workspace_limit));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dimA));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dimB));
    return val;
}
bool BatchedMatrixMul_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<BatchedMatrixMul>(),
         &&b_ = rhs_.cast_final_safe<BatchedMatrixMul>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.transposeA != b_.transposeA) return false;
    if (a_.transposeB != b_.transposeB) return false;
    if (a_.compute_mode != b_.compute_mode) return false;
    if (a_.format != b_.format) return false;
    if (a_.strategy != b_.strategy) return false;
    if (a_.workspace_limit != b_.workspace_limit) return false;
    if (a_.dimA != b_.dimA) return false;
    if (a_.dimB != b_.dimB) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> BatchedMatrixMul_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<BatchedMatrixMul>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("transposeA", std::to_string(op_.transposeA));
    props_.emplace_back("transposeB", std::to_string(op_.transposeB));
    switch (op_.compute_mode){
    case BatchedMatrixMul::ComputeMode::DEFAULT:
        props_.emplace_back("compute_mode", "DEFAULT");
        break;
    case BatchedMatrixMul::ComputeMode::FLOAT32:
        props_.emplace_back("compute_mode", "FLOAT32");
        break;
    default:
        props_.emplace_back("compute_mode", "INVALID");
        break;
    }
    switch (op_.format){
    case BatchedMatrixMul::Format::DEFAULT:
        props_.emplace_back("format", "DEFAULT");
        break;
    case BatchedMatrixMul::Format::MK4:
        props_.emplace_back("format", "MK4");
        break;
    case BatchedMatrixMul::Format::MK8:
        props_.emplace_back("format", "MK8");
        break;
    case BatchedMatrixMul::Format::MK4_DOT:
        props_.emplace_back("format", "MK4_DOT");
        break;
    case BatchedMatrixMul::Format::N32K4_DOT:
        props_.emplace_back("format", "N32K4_DOT");
        break;
    default:
        props_.emplace_back("format", "INVALID");
        break;
    }
    switch (op_.strategy){
    case BatchedMatrixMul::Strategy::HEURISTIC:
        props_.emplace_back("strategy", "HEURISTIC");
        break;
    case BatchedMatrixMul::Strategy::PROFILE:
        props_.emplace_back("strategy", "PROFILE");
        break;
    case BatchedMatrixMul::Strategy::REPRODUCIBLE:
        props_.emplace_back("strategy", "REPRODUCIBLE");
        break;
    case BatchedMatrixMul::Strategy::OPTIMIZED:
        props_.emplace_back("strategy", "OPTIMIZED");
        break;
    default:
        props_.emplace_back("strategy", "INVALID");
        break;
    }
    props_.emplace_back("workspace_limit", std::to_string(op_.workspace_limit));
    props_.emplace_back("dimA", std::to_string(op_.dimA));
    props_.emplace_back("dimB", std::to_string(op_.dimB));
    return props_;
}
std::string BatchedMatrixMul_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<BatchedMatrixMul>();
    static_cast<void>(op_);
    return "BatchedMatrixMul";
}
} // anonymous namespace
OP_TRAIT_REG(BatchedMatrixMul, BatchedMatrixMul)
    .hash(BatchedMatrixMul_hash_impl)
    .is_same_st(BatchedMatrixMul_is_same_st_impl)
    .props(BatchedMatrixMul_props_impl)
    .make_name(BatchedMatrixMul_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(BatchedMeshIndexing);

namespace {
size_t BatchedMeshIndexing_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<BatchedMeshIndexing>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.items));
    return val;
}
bool BatchedMeshIndexing_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<BatchedMeshIndexing>(),
         &&b_ = rhs_.cast_final_safe<BatchedMeshIndexing>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.items != b_.items) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> BatchedMeshIndexing_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<BatchedMeshIndexing>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("items", "{std::vector}");
    return props_;
}
std::string BatchedMeshIndexing_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<BatchedMeshIndexing>();
    static_cast<void>(op_);
    return "BatchedMeshIndexing";
}
} // anonymous namespace
OP_TRAIT_REG(BatchedMeshIndexing, BatchedMeshIndexing)
    .hash(BatchedMeshIndexing_hash_impl)
    .is_same_st(BatchedMeshIndexing_is_same_st_impl)
    .props(BatchedMeshIndexing_props_impl)
    .make_name(BatchedMeshIndexing_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(BatchedSetMeshIndexing);

namespace {
size_t BatchedSetMeshIndexing_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<BatchedSetMeshIndexing>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.items));
    return val;
}
bool BatchedSetMeshIndexing_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<BatchedSetMeshIndexing>(),
         &&b_ = rhs_.cast_final_safe<BatchedSetMeshIndexing>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.items != b_.items) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> BatchedSetMeshIndexing_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<BatchedSetMeshIndexing>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("items", "{std::vector}");
    return props_;
}
std::string BatchedSetMeshIndexing_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<BatchedSetMeshIndexing>();
    static_cast<void>(op_);
    return "BatchedSetMeshIndexing";
}
} // anonymous namespace
OP_TRAIT_REG(BatchedSetMeshIndexing, BatchedSetMeshIndexing)
    .hash(BatchedSetMeshIndexing_hash_impl)
    .is_same_st(BatchedSetMeshIndexing_is_same_st_impl)
    .props(BatchedSetMeshIndexing_props_impl)
    .make_name(BatchedSetMeshIndexing_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(BetaRNG);

namespace {
size_t BetaRNG_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<BetaRNG>();
    static_cast<void>(op_);

    return mgb::hash_pair_combine(
      mgb::hash(op_.dyn_typeinfo()),
      mgb::hash(op_.handle)
      );
  }
bool BetaRNG_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<BetaRNG>(),
         &&b_ = rhs_.cast_final_safe<BetaRNG>();
    static_cast<void>(a_);
    static_cast<void>(b_);
return a_.handle == b_.handle;}
std::vector<std::pair<const char*, std::string>> BetaRNG_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<BetaRNG>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("seed", std::to_string(op_.seed));
    props_.emplace_back("handle", std::to_string(op_.handle));
    return props_;
}
std::string BetaRNG_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<BetaRNG>();
    static_cast<void>(op_);
    return "BetaRNG";
}
} // anonymous namespace
OP_TRAIT_REG(BetaRNG, BetaRNG)
    .hash(BetaRNG_hash_impl)
    .is_same_st(BetaRNG_is_same_st_impl)
    .props(BetaRNG_props_impl)
    .make_name(BetaRNG_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Borrow);

namespace {
size_t Borrow_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Borrow>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.comp_node));
    return val;
}
bool Borrow_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Borrow>(),
         &&b_ = rhs_.cast_final_safe<Borrow>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.comp_node != b_.comp_node) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Borrow_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Borrow>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("comp_node", op_.comp_node.to_string());
    return props_;
}
std::string Borrow_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Borrow>();
    static_cast<void>(op_);
    return "Borrow";
}
} // anonymous namespace
OP_TRAIT_REG(Borrow, Borrow)
    .hash(Borrow_hash_impl)
    .is_same_st(Borrow_is_same_st_impl)
    .props(Borrow_props_impl)
    .make_name(Borrow_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Broadcast);

namespace {
size_t Broadcast_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Broadcast>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.shape));
    return val;
}
bool Broadcast_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Broadcast>(),
         &&b_ = rhs_.cast_final_safe<Broadcast>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.shape != b_.shape) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Broadcast_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Broadcast>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("shape", "{std::vector}");
    return props_;
}
std::string Broadcast_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Broadcast>();
    static_cast<void>(op_);
    return "Broadcast";
}
} // anonymous namespace
OP_TRAIT_REG(Broadcast, Broadcast)
    .hash(Broadcast_hash_impl)
    .is_same_st(Broadcast_is_same_st_impl)
    .props(Broadcast_props_impl)
    .make_name(Broadcast_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(CambriconRuntime);

namespace {
size_t CambriconRuntime_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<CambriconRuntime>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.buf));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.buf_size));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.symbol));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.tensor_dim_mutable));
    return val;
}
bool CambriconRuntime_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<CambriconRuntime>(),
         &&b_ = rhs_.cast_final_safe<CambriconRuntime>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.buf != b_.buf) return false;
    if (a_.buf_size != b_.buf_size) return false;
    if (a_.symbol != b_.symbol) return false;
    if (a_.tensor_dim_mutable != b_.tensor_dim_mutable) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> CambriconRuntime_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<CambriconRuntime>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("buf", op_.buf);
    props_.emplace_back("buf_size", std::to_string(op_.buf_size));
    props_.emplace_back("symbol", op_.symbol);
    props_.emplace_back("tensor_dim_mutable", std::to_string(op_.tensor_dim_mutable));
    return props_;
}
std::string CambriconRuntime_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<CambriconRuntime>();
    static_cast<void>(op_);
    return "CambriconRuntime";
}
} // anonymous namespace
OP_TRAIT_REG(CambriconRuntime, CambriconRuntime)
    .hash(CambriconRuntime_hash_impl)
    .is_same_st(CambriconRuntime_is_same_st_impl)
    .props(CambriconRuntime_props_impl)
    .make_name(CambriconRuntime_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(CheckNonFinite);

namespace {
size_t CheckNonFinite_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<CheckNonFinite>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.scale));
    return val;
}
bool CheckNonFinite_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<CheckNonFinite>(),
         &&b_ = rhs_.cast_final_safe<CheckNonFinite>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.scale != b_.scale) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> CheckNonFinite_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<CheckNonFinite>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("scale", std::to_string(op_.scale));
    return props_;
}
std::string CheckNonFinite_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<CheckNonFinite>();
    static_cast<void>(op_);
    return "CheckNonFinite";
}
} // anonymous namespace
OP_TRAIT_REG(CheckNonFinite, CheckNonFinite)
    .hash(CheckNonFinite_hash_impl)
    .is_same_st(CheckNonFinite_is_same_st_impl)
    .props(CheckNonFinite_props_impl)
    .make_name(CheckNonFinite_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(CollectiveComm);

namespace {
size_t CollectiveComm_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<CollectiveComm>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.mode));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.key));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.nr_devices));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.rank));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.is_root));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.local_grad));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.addr));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.port));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dtype.handle()));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.backend));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.comp_node));
    return val;
}
bool CollectiveComm_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<CollectiveComm>(),
         &&b_ = rhs_.cast_final_safe<CollectiveComm>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.mode != b_.mode) return false;
    if (a_.key != b_.key) return false;
    if (a_.nr_devices != b_.nr_devices) return false;
    if (a_.rank != b_.rank) return false;
    if (a_.is_root != b_.is_root) return false;
    if (a_.local_grad != b_.local_grad) return false;
    if (a_.addr != b_.addr) return false;
    if (a_.port != b_.port) return false;
    if (a_.dtype != b_.dtype) return false;
    if (a_.backend != b_.backend) return false;
    if (a_.comp_node != b_.comp_node) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> CollectiveComm_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<CollectiveComm>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.mode){
    case CollectiveComm::Mode::REDUCE_SUM:
        props_.emplace_back("mode", "REDUCE_SUM");
        break;
    case CollectiveComm::Mode::BROADCAST:
        props_.emplace_back("mode", "BROADCAST");
        break;
    case CollectiveComm::Mode::ALL_GATHER:
        props_.emplace_back("mode", "ALL_GATHER");
        break;
    case CollectiveComm::Mode::REDUCE_SCATTER_SUM:
        props_.emplace_back("mode", "REDUCE_SCATTER_SUM");
        break;
    case CollectiveComm::Mode::ALL_REDUCE_SUM:
        props_.emplace_back("mode", "ALL_REDUCE_SUM");
        break;
    case CollectiveComm::Mode::ALL_REDUCE_MAX:
        props_.emplace_back("mode", "ALL_REDUCE_MAX");
        break;
    case CollectiveComm::Mode::ALL_REDUCE_MIN:
        props_.emplace_back("mode", "ALL_REDUCE_MIN");
        break;
    case CollectiveComm::Mode::ALL_REDUCE_PROD:
        props_.emplace_back("mode", "ALL_REDUCE_PROD");
        break;
    case CollectiveComm::Mode::GATHER:
        props_.emplace_back("mode", "GATHER");
        break;
    case CollectiveComm::Mode::SCATTER:
        props_.emplace_back("mode", "SCATTER");
        break;
    case CollectiveComm::Mode::ALL_TO_ALL:
        props_.emplace_back("mode", "ALL_TO_ALL");
        break;
    default:
        props_.emplace_back("mode", "INVALID");
        break;
    }
    props_.emplace_back("key", op_.key);
    props_.emplace_back("nr_devices", std::to_string(op_.nr_devices));
    props_.emplace_back("rank", std::to_string(op_.rank));
    props_.emplace_back("is_root", std::to_string(op_.is_root));
    props_.emplace_back("local_grad", std::to_string(op_.local_grad));
    props_.emplace_back("addr", op_.addr);
    props_.emplace_back("port", std::to_string(op_.port));
    props_.emplace_back("dtype", op_.dtype.name());
    props_.emplace_back("backend", op_.backend);
    props_.emplace_back("comp_node", op_.comp_node);
    return props_;
}
std::string CollectiveComm_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<CollectiveComm>();
    static_cast<void>(op_);
    return "CollectiveComm";
}
} // anonymous namespace
OP_TRAIT_REG(CollectiveComm, CollectiveComm)
    .hash(CollectiveComm_hash_impl)
    .is_same_st(CollectiveComm_is_same_st_impl)
    .props(CollectiveComm_props_impl)
    .make_name(CollectiveComm_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Concat);

namespace {
size_t Concat_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Concat>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.axis));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.comp_node));
    return val;
}
bool Concat_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Concat>(),
         &&b_ = rhs_.cast_final_safe<Concat>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.axis != b_.axis) return false;
    if (a_.comp_node != b_.comp_node) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Concat_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Concat>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("axis", std::to_string(op_.axis));
    props_.emplace_back("comp_node", op_.comp_node.to_string());
    return props_;
}
std::string Concat_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Concat>();
    static_cast<void>(op_);
    return "Concat";
}
} // anonymous namespace
OP_TRAIT_REG(Concat, Concat)
    .hash(Concat_hash_impl)
    .is_same_st(Concat_is_same_st_impl)
    .props(Concat_props_impl)
    .make_name(Concat_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(CondTake);

namespace {
size_t CondTake_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<CondTake>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    return val;
}
bool CondTake_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<CondTake>(),
         &&b_ = rhs_.cast_final_safe<CondTake>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    return true;
}
std::vector<std::pair<const char*, std::string>> CondTake_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<CondTake>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    return props_;
}
std::string CondTake_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<CondTake>();
    static_cast<void>(op_);
    return "CondTake";
}
} // anonymous namespace
OP_TRAIT_REG(CondTake, CondTake)
    .hash(CondTake_hash_impl)
    .is_same_st(CondTake_is_same_st_impl)
    .props(CondTake_props_impl)
    .make_name(CondTake_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(ConvBias);

namespace {
size_t ConvBias_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<ConvBias>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.nonlineMode));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.mode));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.sparse));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.format));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pad_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pad_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dilate_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dilate_w));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.compute_mode));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.strategy));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.workspace_limit));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dtype.handle()));
    return val;
}
bool ConvBias_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<ConvBias>(),
         &&b_ = rhs_.cast_final_safe<ConvBias>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.nonlineMode != b_.nonlineMode) return false;
    if (a_.mode != b_.mode) return false;
    if (a_.sparse != b_.sparse) return false;
    if (a_.format != b_.format) return false;
    if (a_.pad_h != b_.pad_h) return false;
    if (a_.pad_w != b_.pad_w) return false;
    if (a_.stride_h != b_.stride_h) return false;
    if (a_.stride_w != b_.stride_w) return false;
    if (a_.dilate_h != b_.dilate_h) return false;
    if (a_.dilate_w != b_.dilate_w) return false;
    if (a_.compute_mode != b_.compute_mode) return false;
    if (a_.strategy != b_.strategy) return false;
    if (a_.workspace_limit != b_.workspace_limit) return false;
    if (a_.dtype != b_.dtype) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> ConvBias_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<ConvBias>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.nonlineMode){
    case ConvBias::NonlineMode::IDENTITY:
        props_.emplace_back("nonlineMode", "IDENTITY");
        break;
    case ConvBias::NonlineMode::RELU:
        props_.emplace_back("nonlineMode", "RELU");
        break;
    case ConvBias::NonlineMode::SIGMOID:
        props_.emplace_back("nonlineMode", "SIGMOID");
        break;
    case ConvBias::NonlineMode::H_SWISH:
        props_.emplace_back("nonlineMode", "H_SWISH");
        break;
    default:
        props_.emplace_back("nonlineMode", "INVALID");
        break;
    }
    switch (op_.mode){
    case ConvBias::Mode::CROSS_CORRELATION:
        props_.emplace_back("mode", "CROSS_CORRELATION");
        break;
    case ConvBias::Mode::CONVOLUTION:
        props_.emplace_back("mode", "CONVOLUTION");
        break;
    default:
        props_.emplace_back("mode", "INVALID");
        break;
    }
    switch (op_.sparse){
    case ConvBias::Sparse::DENSE:
        props_.emplace_back("sparse", "DENSE");
        break;
    case ConvBias::Sparse::GROUP:
        props_.emplace_back("sparse", "GROUP");
        break;
    default:
        props_.emplace_back("sparse", "INVALID");
        break;
    }
    switch (op_.format){
    case ConvBias::Format::NCHW:
        props_.emplace_back("format", "NCHW");
        break;
    case ConvBias::Format::NHWC:
        props_.emplace_back("format", "NHWC");
        break;
    case ConvBias::Format::NHWCD4:
        props_.emplace_back("format", "NHWCD4");
        break;
    case ConvBias::Format::NCHW4:
        props_.emplace_back("format", "NCHW4");
        break;
    case ConvBias::Format::NCHW8:
        props_.emplace_back("format", "NCHW8");
        break;
    case ConvBias::Format::NCHW32:
        props_.emplace_back("format", "NCHW32");
        break;
    case ConvBias::Format::NCHW88:
        props_.emplace_back("format", "NCHW88");
        break;
    case ConvBias::Format::NCHW44:
        props_.emplace_back("format", "NCHW44");
        break;
    case ConvBias::Format::NCHW44_DOT:
        props_.emplace_back("format", "NCHW44_DOT");
        break;
    case ConvBias::Format::NCHW4_NCHW32:
        props_.emplace_back("format", "NCHW4_NCHW32");
        break;
    case ConvBias::Format::NCHW32_NCHW4:
        props_.emplace_back("format", "NCHW32_NCHW4");
        break;
    case ConvBias::Format::NCHW4_NCHW:
        props_.emplace_back("format", "NCHW4_NCHW");
        break;
    case ConvBias::Format::NHWC_NCHW:
        props_.emplace_back("format", "NHWC_NCHW");
        break;
    case ConvBias::Format::NHWC_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NHWC_NCHW4_IC_SMALL");
        break;
    case ConvBias::Format::NCHW_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NCHW_NCHW4_IC_SMALL");
        break;
    case ConvBias::Format::CHWN4:
        props_.emplace_back("format", "CHWN4");
        break;
    case ConvBias::Format::NCHW64:
        props_.emplace_back("format", "NCHW64");
        break;
    case ConvBias::Format::NCHW4_NHWC:
        props_.emplace_back("format", "NCHW4_NHWC");
        break;
    default:
        props_.emplace_back("format", "INVALID");
        break;
    }
    props_.emplace_back("pad_h", std::to_string(op_.pad_h));
    props_.emplace_back("pad_w", std::to_string(op_.pad_w));
    props_.emplace_back("stride_h", std::to_string(op_.stride_h));
    props_.emplace_back("stride_w", std::to_string(op_.stride_w));
    props_.emplace_back("dilate_h", std::to_string(op_.dilate_h));
    props_.emplace_back("dilate_w", std::to_string(op_.dilate_w));
    switch (op_.compute_mode){
    case ConvBias::ComputeMode::DEFAULT:
        props_.emplace_back("compute_mode", "DEFAULT");
        break;
    case ConvBias::ComputeMode::FLOAT32:
        props_.emplace_back("compute_mode", "FLOAT32");
        break;
    default:
        props_.emplace_back("compute_mode", "INVALID");
        break;
    }
    switch (op_.strategy){
    case ConvBias::Strategy::HEURISTIC:
        props_.emplace_back("strategy", "HEURISTIC");
        break;
    case ConvBias::Strategy::PROFILE:
        props_.emplace_back("strategy", "PROFILE");
        break;
    case ConvBias::Strategy::REPRODUCIBLE:
        props_.emplace_back("strategy", "REPRODUCIBLE");
        break;
    case ConvBias::Strategy::OPTIMIZED:
        props_.emplace_back("strategy", "OPTIMIZED");
        break;
    default:
        props_.emplace_back("strategy", "INVALID");
        break;
    }
    props_.emplace_back("workspace_limit", std::to_string(op_.workspace_limit));
    props_.emplace_back("dtype", op_.dtype.name());
    return props_;
}
std::string ConvBias_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<ConvBias>();
    static_cast<void>(op_);
    return "ConvBias";
}
} // anonymous namespace
OP_TRAIT_REG(ConvBias, ConvBias)
    .hash(ConvBias_hash_impl)
    .is_same_st(ConvBias_is_same_st_impl)
    .props(ConvBias_props_impl)
    .make_name(ConvBias_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Convolution);

namespace {
size_t Convolution_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Convolution>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.mode));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pad_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pad_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dilate_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dilate_w));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.sparse));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.format));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.compute_mode));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.strategy));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.workspace_limit));
    return val;
}
bool Convolution_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Convolution>(),
         &&b_ = rhs_.cast_final_safe<Convolution>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.mode != b_.mode) return false;
    if (a_.pad_h != b_.pad_h) return false;
    if (a_.pad_w != b_.pad_w) return false;
    if (a_.stride_h != b_.stride_h) return false;
    if (a_.stride_w != b_.stride_w) return false;
    if (a_.dilate_h != b_.dilate_h) return false;
    if (a_.dilate_w != b_.dilate_w) return false;
    if (a_.sparse != b_.sparse) return false;
    if (a_.format != b_.format) return false;
    if (a_.compute_mode != b_.compute_mode) return false;
    if (a_.strategy != b_.strategy) return false;
    if (a_.workspace_limit != b_.workspace_limit) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Convolution_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Convolution>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.mode){
    case Convolution::Mode::CROSS_CORRELATION:
        props_.emplace_back("mode", "CROSS_CORRELATION");
        break;
    case Convolution::Mode::CONVOLUTION:
        props_.emplace_back("mode", "CONVOLUTION");
        break;
    default:
        props_.emplace_back("mode", "INVALID");
        break;
    }
    props_.emplace_back("pad_h", std::to_string(op_.pad_h));
    props_.emplace_back("pad_w", std::to_string(op_.pad_w));
    props_.emplace_back("stride_h", std::to_string(op_.stride_h));
    props_.emplace_back("stride_w", std::to_string(op_.stride_w));
    props_.emplace_back("dilate_h", std::to_string(op_.dilate_h));
    props_.emplace_back("dilate_w", std::to_string(op_.dilate_w));
    switch (op_.sparse){
    case Convolution::Sparse::DENSE:
        props_.emplace_back("sparse", "DENSE");
        break;
    case Convolution::Sparse::GROUP:
        props_.emplace_back("sparse", "GROUP");
        break;
    default:
        props_.emplace_back("sparse", "INVALID");
        break;
    }
    switch (op_.format){
    case Convolution::Format::NCHW:
        props_.emplace_back("format", "NCHW");
        break;
    case Convolution::Format::NHWC:
        props_.emplace_back("format", "NHWC");
        break;
    case Convolution::Format::NHWCD4:
        props_.emplace_back("format", "NHWCD4");
        break;
    case Convolution::Format::NCHW4:
        props_.emplace_back("format", "NCHW4");
        break;
    case Convolution::Format::NCHW8:
        props_.emplace_back("format", "NCHW8");
        break;
    case Convolution::Format::NCHW32:
        props_.emplace_back("format", "NCHW32");
        break;
    case Convolution::Format::NCHW88:
        props_.emplace_back("format", "NCHW88");
        break;
    case Convolution::Format::NCHW44:
        props_.emplace_back("format", "NCHW44");
        break;
    case Convolution::Format::NCHW44_DOT:
        props_.emplace_back("format", "NCHW44_DOT");
        break;
    case Convolution::Format::NCHW4_NCHW32:
        props_.emplace_back("format", "NCHW4_NCHW32");
        break;
    case Convolution::Format::NCHW32_NCHW4:
        props_.emplace_back("format", "NCHW32_NCHW4");
        break;
    case Convolution::Format::NCHW4_NCHW:
        props_.emplace_back("format", "NCHW4_NCHW");
        break;
    case Convolution::Format::NHWC_NCHW:
        props_.emplace_back("format", "NHWC_NCHW");
        break;
    case Convolution::Format::NHWC_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NHWC_NCHW4_IC_SMALL");
        break;
    case Convolution::Format::NCHW_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NCHW_NCHW4_IC_SMALL");
        break;
    case Convolution::Format::CHWN4:
        props_.emplace_back("format", "CHWN4");
        break;
    case Convolution::Format::NCHW64:
        props_.emplace_back("format", "NCHW64");
        break;
    case Convolution::Format::NCHW4_NHWC:
        props_.emplace_back("format", "NCHW4_NHWC");
        break;
    default:
        props_.emplace_back("format", "INVALID");
        break;
    }
    switch (op_.compute_mode){
    case Convolution::ComputeMode::DEFAULT:
        props_.emplace_back("compute_mode", "DEFAULT");
        break;
    case Convolution::ComputeMode::FLOAT32:
        props_.emplace_back("compute_mode", "FLOAT32");
        break;
    default:
        props_.emplace_back("compute_mode", "INVALID");
        break;
    }
    switch (op_.strategy){
    case Convolution::Strategy::HEURISTIC:
        props_.emplace_back("strategy", "HEURISTIC");
        break;
    case Convolution::Strategy::PROFILE:
        props_.emplace_back("strategy", "PROFILE");
        break;
    case Convolution::Strategy::REPRODUCIBLE:
        props_.emplace_back("strategy", "REPRODUCIBLE");
        break;
    case Convolution::Strategy::OPTIMIZED:
        props_.emplace_back("strategy", "OPTIMIZED");
        break;
    default:
        props_.emplace_back("strategy", "INVALID");
        break;
    }
    props_.emplace_back("workspace_limit", std::to_string(op_.workspace_limit));
    return props_;
}
std::string Convolution_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Convolution>();
    static_cast<void>(op_);
    return "Convolution";
}
} // anonymous namespace
OP_TRAIT_REG(Convolution, Convolution)
    .hash(Convolution_hash_impl)
    .is_same_st(Convolution_is_same_st_impl)
    .props(Convolution_props_impl)
    .make_name(Convolution_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Convolution3D);

namespace {
size_t Convolution3D_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Convolution3D>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.mode));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pad_d));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pad_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pad_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride_d));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dilate_d));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dilate_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dilate_w));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.sparse));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.data_type));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.format));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.strategy));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.workspace_limit));
    return val;
}
bool Convolution3D_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Convolution3D>(),
         &&b_ = rhs_.cast_final_safe<Convolution3D>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.mode != b_.mode) return false;
    if (a_.pad_d != b_.pad_d) return false;
    if (a_.pad_h != b_.pad_h) return false;
    if (a_.pad_w != b_.pad_w) return false;
    if (a_.stride_d != b_.stride_d) return false;
    if (a_.stride_h != b_.stride_h) return false;
    if (a_.stride_w != b_.stride_w) return false;
    if (a_.dilate_d != b_.dilate_d) return false;
    if (a_.dilate_h != b_.dilate_h) return false;
    if (a_.dilate_w != b_.dilate_w) return false;
    if (a_.sparse != b_.sparse) return false;
    if (a_.data_type != b_.data_type) return false;
    if (a_.format != b_.format) return false;
    if (a_.strategy != b_.strategy) return false;
    if (a_.workspace_limit != b_.workspace_limit) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Convolution3D_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Convolution3D>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.mode){
    case Convolution3D::Mode::CROSS_CORRELATION:
        props_.emplace_back("mode", "CROSS_CORRELATION");
        break;
    case Convolution3D::Mode::CONVOLUTION:
        props_.emplace_back("mode", "CONVOLUTION");
        break;
    default:
        props_.emplace_back("mode", "INVALID");
        break;
    }
    props_.emplace_back("pad_d", std::to_string(op_.pad_d));
    props_.emplace_back("pad_h", std::to_string(op_.pad_h));
    props_.emplace_back("pad_w", std::to_string(op_.pad_w));
    props_.emplace_back("stride_d", std::to_string(op_.stride_d));
    props_.emplace_back("stride_h", std::to_string(op_.stride_h));
    props_.emplace_back("stride_w", std::to_string(op_.stride_w));
    props_.emplace_back("dilate_d", std::to_string(op_.dilate_d));
    props_.emplace_back("dilate_h", std::to_string(op_.dilate_h));
    props_.emplace_back("dilate_w", std::to_string(op_.dilate_w));
    switch (op_.sparse){
    case Convolution3D::Sparse::DENSE:
        props_.emplace_back("sparse", "DENSE");
        break;
    case Convolution3D::Sparse::GROUP:
        props_.emplace_back("sparse", "GROUP");
        break;
    default:
        props_.emplace_back("sparse", "INVALID");
        break;
    }
    switch (op_.data_type){
    case Convolution3D::DataType::FLOAT:
        props_.emplace_back("data_type", "FLOAT");
        break;
    case Convolution3D::DataType::FLOAT_IO16xC32:
        props_.emplace_back("data_type", "FLOAT_IO16xC32");
        break;
    default:
        props_.emplace_back("data_type", "INVALID");
        break;
    }
    switch (op_.format){
    case Convolution3D::Format::NCDHW:
        props_.emplace_back("format", "NCDHW");
        break;
    case Convolution3D::Format::NDHWC:
        props_.emplace_back("format", "NDHWC");
        break;
    default:
        props_.emplace_back("format", "INVALID");
        break;
    }
    switch (op_.strategy){
    case Convolution3D::Strategy::HEURISTIC:
        props_.emplace_back("strategy", "HEURISTIC");
        break;
    case Convolution3D::Strategy::PROFILE:
        props_.emplace_back("strategy", "PROFILE");
        break;
    case Convolution3D::Strategy::REPRODUCIBLE:
        props_.emplace_back("strategy", "REPRODUCIBLE");
        break;
    case Convolution3D::Strategy::OPTIMIZED:
        props_.emplace_back("strategy", "OPTIMIZED");
        break;
    default:
        props_.emplace_back("strategy", "INVALID");
        break;
    }
    props_.emplace_back("workspace_limit", std::to_string(op_.workspace_limit));
    return props_;
}
std::string Convolution3D_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Convolution3D>();
    static_cast<void>(op_);
    return "Convolution3D";
}
} // anonymous namespace
OP_TRAIT_REG(Convolution3D, Convolution3D)
    .hash(Convolution3D_hash_impl)
    .is_same_st(Convolution3D_is_same_st_impl)
    .props(Convolution3D_props_impl)
    .make_name(Convolution3D_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Convolution3DBackwardData);

namespace {
size_t Convolution3DBackwardData_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Convolution3DBackwardData>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.mode));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pad_d));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pad_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pad_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride_d));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dilate_d));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dilate_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dilate_w));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.sparse));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.data_type));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.format));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.strategy));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.workspace_limit));
    return val;
}
bool Convolution3DBackwardData_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Convolution3DBackwardData>(),
         &&b_ = rhs_.cast_final_safe<Convolution3DBackwardData>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.mode != b_.mode) return false;
    if (a_.pad_d != b_.pad_d) return false;
    if (a_.pad_h != b_.pad_h) return false;
    if (a_.pad_w != b_.pad_w) return false;
    if (a_.stride_d != b_.stride_d) return false;
    if (a_.stride_h != b_.stride_h) return false;
    if (a_.stride_w != b_.stride_w) return false;
    if (a_.dilate_d != b_.dilate_d) return false;
    if (a_.dilate_h != b_.dilate_h) return false;
    if (a_.dilate_w != b_.dilate_w) return false;
    if (a_.sparse != b_.sparse) return false;
    if (a_.data_type != b_.data_type) return false;
    if (a_.format != b_.format) return false;
    if (a_.strategy != b_.strategy) return false;
    if (a_.workspace_limit != b_.workspace_limit) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Convolution3DBackwardData_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Convolution3DBackwardData>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.mode){
    case Convolution3DBackwardData::Mode::CROSS_CORRELATION:
        props_.emplace_back("mode", "CROSS_CORRELATION");
        break;
    case Convolution3DBackwardData::Mode::CONVOLUTION:
        props_.emplace_back("mode", "CONVOLUTION");
        break;
    default:
        props_.emplace_back("mode", "INVALID");
        break;
    }
    props_.emplace_back("pad_d", std::to_string(op_.pad_d));
    props_.emplace_back("pad_h", std::to_string(op_.pad_h));
    props_.emplace_back("pad_w", std::to_string(op_.pad_w));
    props_.emplace_back("stride_d", std::to_string(op_.stride_d));
    props_.emplace_back("stride_h", std::to_string(op_.stride_h));
    props_.emplace_back("stride_w", std::to_string(op_.stride_w));
    props_.emplace_back("dilate_d", std::to_string(op_.dilate_d));
    props_.emplace_back("dilate_h", std::to_string(op_.dilate_h));
    props_.emplace_back("dilate_w", std::to_string(op_.dilate_w));
    switch (op_.sparse){
    case Convolution3DBackwardData::Sparse::DENSE:
        props_.emplace_back("sparse", "DENSE");
        break;
    case Convolution3DBackwardData::Sparse::GROUP:
        props_.emplace_back("sparse", "GROUP");
        break;
    default:
        props_.emplace_back("sparse", "INVALID");
        break;
    }
    switch (op_.data_type){
    case Convolution3DBackwardData::DataType::FLOAT:
        props_.emplace_back("data_type", "FLOAT");
        break;
    case Convolution3DBackwardData::DataType::FLOAT_IO16xC32:
        props_.emplace_back("data_type", "FLOAT_IO16xC32");
        break;
    default:
        props_.emplace_back("data_type", "INVALID");
        break;
    }
    switch (op_.format){
    case Convolution3DBackwardData::Format::NCDHW:
        props_.emplace_back("format", "NCDHW");
        break;
    case Convolution3DBackwardData::Format::NDHWC:
        props_.emplace_back("format", "NDHWC");
        break;
    default:
        props_.emplace_back("format", "INVALID");
        break;
    }
    switch (op_.strategy){
    case Convolution3DBackwardData::Strategy::HEURISTIC:
        props_.emplace_back("strategy", "HEURISTIC");
        break;
    case Convolution3DBackwardData::Strategy::PROFILE:
        props_.emplace_back("strategy", "PROFILE");
        break;
    case Convolution3DBackwardData::Strategy::REPRODUCIBLE:
        props_.emplace_back("strategy", "REPRODUCIBLE");
        break;
    case Convolution3DBackwardData::Strategy::OPTIMIZED:
        props_.emplace_back("strategy", "OPTIMIZED");
        break;
    default:
        props_.emplace_back("strategy", "INVALID");
        break;
    }
    props_.emplace_back("workspace_limit", std::to_string(op_.workspace_limit));
    return props_;
}
std::string Convolution3DBackwardData_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Convolution3DBackwardData>();
    static_cast<void>(op_);
    return "Convolution3DBackwardData";
}
} // anonymous namespace
OP_TRAIT_REG(Convolution3DBackwardData, Convolution3DBackwardData)
    .hash(Convolution3DBackwardData_hash_impl)
    .is_same_st(Convolution3DBackwardData_is_same_st_impl)
    .props(Convolution3DBackwardData_props_impl)
    .make_name(Convolution3DBackwardData_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(ConvolutionBackwardData);

namespace {
size_t ConvolutionBackwardData_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<ConvolutionBackwardData>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.mode));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pad_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pad_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dilate_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dilate_w));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.sparse));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.format));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.compute_mode));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.strategy));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.workspace_limit));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dtype.handle()));
    return val;
}
bool ConvolutionBackwardData_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<ConvolutionBackwardData>(),
         &&b_ = rhs_.cast_final_safe<ConvolutionBackwardData>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.mode != b_.mode) return false;
    if (a_.pad_h != b_.pad_h) return false;
    if (a_.pad_w != b_.pad_w) return false;
    if (a_.stride_h != b_.stride_h) return false;
    if (a_.stride_w != b_.stride_w) return false;
    if (a_.dilate_h != b_.dilate_h) return false;
    if (a_.dilate_w != b_.dilate_w) return false;
    if (a_.sparse != b_.sparse) return false;
    if (a_.format != b_.format) return false;
    if (a_.compute_mode != b_.compute_mode) return false;
    if (a_.strategy != b_.strategy) return false;
    if (a_.workspace_limit != b_.workspace_limit) return false;
    if (a_.dtype != b_.dtype) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> ConvolutionBackwardData_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<ConvolutionBackwardData>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.mode){
    case ConvolutionBackwardData::Mode::CROSS_CORRELATION:
        props_.emplace_back("mode", "CROSS_CORRELATION");
        break;
    case ConvolutionBackwardData::Mode::CONVOLUTION:
        props_.emplace_back("mode", "CONVOLUTION");
        break;
    default:
        props_.emplace_back("mode", "INVALID");
        break;
    }
    props_.emplace_back("pad_h", std::to_string(op_.pad_h));
    props_.emplace_back("pad_w", std::to_string(op_.pad_w));
    props_.emplace_back("stride_h", std::to_string(op_.stride_h));
    props_.emplace_back("stride_w", std::to_string(op_.stride_w));
    props_.emplace_back("dilate_h", std::to_string(op_.dilate_h));
    props_.emplace_back("dilate_w", std::to_string(op_.dilate_w));
    switch (op_.sparse){
    case ConvolutionBackwardData::Sparse::DENSE:
        props_.emplace_back("sparse", "DENSE");
        break;
    case ConvolutionBackwardData::Sparse::GROUP:
        props_.emplace_back("sparse", "GROUP");
        break;
    default:
        props_.emplace_back("sparse", "INVALID");
        break;
    }
    switch (op_.format){
    case ConvolutionBackwardData::Format::NCHW:
        props_.emplace_back("format", "NCHW");
        break;
    case ConvolutionBackwardData::Format::NHWC:
        props_.emplace_back("format", "NHWC");
        break;
    case ConvolutionBackwardData::Format::NHWCD4:
        props_.emplace_back("format", "NHWCD4");
        break;
    case ConvolutionBackwardData::Format::NCHW4:
        props_.emplace_back("format", "NCHW4");
        break;
    case ConvolutionBackwardData::Format::NCHW8:
        props_.emplace_back("format", "NCHW8");
        break;
    case ConvolutionBackwardData::Format::NCHW32:
        props_.emplace_back("format", "NCHW32");
        break;
    case ConvolutionBackwardData::Format::NCHW88:
        props_.emplace_back("format", "NCHW88");
        break;
    case ConvolutionBackwardData::Format::NCHW44:
        props_.emplace_back("format", "NCHW44");
        break;
    case ConvolutionBackwardData::Format::NCHW44_DOT:
        props_.emplace_back("format", "NCHW44_DOT");
        break;
    case ConvolutionBackwardData::Format::NCHW4_NCHW32:
        props_.emplace_back("format", "NCHW4_NCHW32");
        break;
    case ConvolutionBackwardData::Format::NCHW32_NCHW4:
        props_.emplace_back("format", "NCHW32_NCHW4");
        break;
    case ConvolutionBackwardData::Format::NCHW4_NCHW:
        props_.emplace_back("format", "NCHW4_NCHW");
        break;
    case ConvolutionBackwardData::Format::NHWC_NCHW:
        props_.emplace_back("format", "NHWC_NCHW");
        break;
    case ConvolutionBackwardData::Format::NHWC_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NHWC_NCHW4_IC_SMALL");
        break;
    case ConvolutionBackwardData::Format::NCHW_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NCHW_NCHW4_IC_SMALL");
        break;
    case ConvolutionBackwardData::Format::CHWN4:
        props_.emplace_back("format", "CHWN4");
        break;
    case ConvolutionBackwardData::Format::NCHW64:
        props_.emplace_back("format", "NCHW64");
        break;
    case ConvolutionBackwardData::Format::NCHW4_NHWC:
        props_.emplace_back("format", "NCHW4_NHWC");
        break;
    default:
        props_.emplace_back("format", "INVALID");
        break;
    }
    switch (op_.compute_mode){
    case ConvolutionBackwardData::ComputeMode::DEFAULT:
        props_.emplace_back("compute_mode", "DEFAULT");
        break;
    case ConvolutionBackwardData::ComputeMode::FLOAT32:
        props_.emplace_back("compute_mode", "FLOAT32");
        break;
    default:
        props_.emplace_back("compute_mode", "INVALID");
        break;
    }
    switch (op_.strategy){
    case ConvolutionBackwardData::Strategy::HEURISTIC:
        props_.emplace_back("strategy", "HEURISTIC");
        break;
    case ConvolutionBackwardData::Strategy::PROFILE:
        props_.emplace_back("strategy", "PROFILE");
        break;
    case ConvolutionBackwardData::Strategy::REPRODUCIBLE:
        props_.emplace_back("strategy", "REPRODUCIBLE");
        break;
    case ConvolutionBackwardData::Strategy::OPTIMIZED:
        props_.emplace_back("strategy", "OPTIMIZED");
        break;
    default:
        props_.emplace_back("strategy", "INVALID");
        break;
    }
    props_.emplace_back("workspace_limit", std::to_string(op_.workspace_limit));
    props_.emplace_back("dtype", op_.dtype.name());
    return props_;
}
std::string ConvolutionBackwardData_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<ConvolutionBackwardData>();
    static_cast<void>(op_);
    return "ConvolutionBackwardData";
}
} // anonymous namespace
OP_TRAIT_REG(ConvolutionBackwardData, ConvolutionBackwardData)
    .hash(ConvolutionBackwardData_hash_impl)
    .is_same_st(ConvolutionBackwardData_is_same_st_impl)
    .props(ConvolutionBackwardData_props_impl)
    .make_name(ConvolutionBackwardData_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Copy);

namespace {
size_t Copy_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Copy>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.comp_node));
    return val;
}
bool Copy_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Copy>(),
         &&b_ = rhs_.cast_final_safe<Copy>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.comp_node != b_.comp_node) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Copy_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Copy>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("comp_node", op_.comp_node.to_string());
    return props_;
}
std::string Copy_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Copy>();
    static_cast<void>(op_);
    return "Copy";
}
} // anonymous namespace
OP_TRAIT_REG(Copy, Copy)
    .hash(Copy_hash_impl)
    .is_same_st(Copy_is_same_st_impl)
    .props(Copy_props_impl)
    .make_name(Copy_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Correlation);

namespace {
size_t Correlation_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Correlation>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.format));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.kernel_size));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.max_displacement));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride1));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride2));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pad_size));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.is_multiply));
    return val;
}
bool Correlation_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Correlation>(),
         &&b_ = rhs_.cast_final_safe<Correlation>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.format != b_.format) return false;
    if (a_.kernel_size != b_.kernel_size) return false;
    if (a_.max_displacement != b_.max_displacement) return false;
    if (a_.stride1 != b_.stride1) return false;
    if (a_.stride2 != b_.stride2) return false;
    if (a_.pad_size != b_.pad_size) return false;
    if (a_.is_multiply != b_.is_multiply) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Correlation_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Correlation>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.format){
    case Correlation::Format::NCHW:
        props_.emplace_back("format", "NCHW");
        break;
    case Correlation::Format::NHWC:
        props_.emplace_back("format", "NHWC");
        break;
    case Correlation::Format::NHWCD4:
        props_.emplace_back("format", "NHWCD4");
        break;
    case Correlation::Format::NCHW4:
        props_.emplace_back("format", "NCHW4");
        break;
    case Correlation::Format::NCHW8:
        props_.emplace_back("format", "NCHW8");
        break;
    case Correlation::Format::NCHW32:
        props_.emplace_back("format", "NCHW32");
        break;
    case Correlation::Format::NCHW88:
        props_.emplace_back("format", "NCHW88");
        break;
    case Correlation::Format::NCHW44:
        props_.emplace_back("format", "NCHW44");
        break;
    case Correlation::Format::NCHW44_DOT:
        props_.emplace_back("format", "NCHW44_DOT");
        break;
    case Correlation::Format::NCHW_WINOGRAD:
        props_.emplace_back("format", "NCHW_WINOGRAD");
        break;
    case Correlation::Format::NCHW88_WINOGRAD:
        props_.emplace_back("format", "NCHW88_WINOGRAD");
        break;
    case Correlation::Format::NCHW44_WINOGRAD:
        props_.emplace_back("format", "NCHW44_WINOGRAD");
        break;
    case Correlation::Format::NCHW4_NCHW32:
        props_.emplace_back("format", "NCHW4_NCHW32");
        break;
    case Correlation::Format::NCHW32_NCHW4:
        props_.emplace_back("format", "NCHW32_NCHW4");
        break;
    case Correlation::Format::NCHW4_NCHW:
        props_.emplace_back("format", "NCHW4_NCHW");
        break;
    case Correlation::Format::NHWC_NCHW:
        props_.emplace_back("format", "NHWC_NCHW");
        break;
    case Correlation::Format::NHWC_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NHWC_NCHW4_IC_SMALL");
        break;
    case Correlation::Format::NCHW_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NCHW_NCHW4_IC_SMALL");
        break;
    case Correlation::Format::CHWN4:
        props_.emplace_back("format", "CHWN4");
        break;
    case Correlation::Format::NCHW4_NHWC:
        props_.emplace_back("format", "NCHW4_NHWC");
        break;
    default:
        props_.emplace_back("format", "INVALID");
        break;
    }
    props_.emplace_back("kernel_size", std::to_string(op_.kernel_size));
    props_.emplace_back("max_displacement", std::to_string(op_.max_displacement));
    props_.emplace_back("stride1", std::to_string(op_.stride1));
    props_.emplace_back("stride2", std::to_string(op_.stride2));
    props_.emplace_back("pad_size", std::to_string(op_.pad_size));
    props_.emplace_back("is_multiply", std::to_string(op_.is_multiply));
    return props_;
}
std::string Correlation_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Correlation>();
    static_cast<void>(op_);
    return "Correlation";
}
} // anonymous namespace
OP_TRAIT_REG(Correlation, Correlation)
    .hash(Correlation_hash_impl)
    .is_same_st(Correlation_is_same_st_impl)
    .props(Correlation_props_impl)
    .make_name(Correlation_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Cumsum);

namespace {
size_t Cumsum_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Cumsum>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.axis));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.exclusive));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.reverse));
    return val;
}
bool Cumsum_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Cumsum>(),
         &&b_ = rhs_.cast_final_safe<Cumsum>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.axis != b_.axis) return false;
    if (a_.exclusive != b_.exclusive) return false;
    if (a_.reverse != b_.reverse) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Cumsum_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Cumsum>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("axis", std::to_string(op_.axis));
    props_.emplace_back("exclusive", std::to_string(op_.exclusive));
    props_.emplace_back("reverse", std::to_string(op_.reverse));
    return props_;
}
std::string Cumsum_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Cumsum>();
    static_cast<void>(op_);
    return "Cumsum";
}
} // anonymous namespace
OP_TRAIT_REG(Cumsum, Cumsum)
    .hash(Cumsum_hash_impl)
    .is_same_st(Cumsum_is_same_st_impl)
    .props(Cumsum_props_impl)
    .make_name(Cumsum_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(CvtColor);

namespace {
size_t CvtColor_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<CvtColor>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.mode));
    return val;
}
bool CvtColor_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<CvtColor>(),
         &&b_ = rhs_.cast_final_safe<CvtColor>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.mode != b_.mode) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> CvtColor_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<CvtColor>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.mode){
    case CvtColor::Mode::RGB2GRAY:
        props_.emplace_back("mode", "RGB2GRAY");
        break;
    case CvtColor::Mode::RGB2YUV:
        props_.emplace_back("mode", "RGB2YUV");
        break;
    case CvtColor::Mode::YUV2RGB:
        props_.emplace_back("mode", "YUV2RGB");
        break;
    case CvtColor::Mode::GRAY2RGB:
        props_.emplace_back("mode", "GRAY2RGB");
        break;
    case CvtColor::Mode::RGBA2RGB:
        props_.emplace_back("mode", "RGBA2RGB");
        break;
    case CvtColor::Mode::RGBA2BGR:
        props_.emplace_back("mode", "RGBA2BGR");
        break;
    case CvtColor::Mode::RGBA2GRAY:
        props_.emplace_back("mode", "RGBA2GRAY");
        break;
    case CvtColor::Mode::RGB2BGR:
        props_.emplace_back("mode", "RGB2BGR");
        break;
    case CvtColor::Mode::BGR2GRAY:
        props_.emplace_back("mode", "BGR2GRAY");
        break;
    case CvtColor::Mode::BGR2RGB:
        props_.emplace_back("mode", "BGR2RGB");
        break;
    case CvtColor::Mode::YUV2GRAY_NV21:
        props_.emplace_back("mode", "YUV2GRAY_NV21");
        break;
    case CvtColor::Mode::YUV2RGB_NV21:
        props_.emplace_back("mode", "YUV2RGB_NV21");
        break;
    case CvtColor::Mode::YUV2BGR_NV21:
        props_.emplace_back("mode", "YUV2BGR_NV21");
        break;
    case CvtColor::Mode::YUV2GRAY_NV12:
        props_.emplace_back("mode", "YUV2GRAY_NV12");
        break;
    case CvtColor::Mode::YUV2RGB_NV12:
        props_.emplace_back("mode", "YUV2RGB_NV12");
        break;
    case CvtColor::Mode::YUV2BGR_NV12:
        props_.emplace_back("mode", "YUV2BGR_NV12");
        break;
    case CvtColor::Mode::YUV2GRAY_YV12:
        props_.emplace_back("mode", "YUV2GRAY_YV12");
        break;
    case CvtColor::Mode::YUV2RGB_YV12:
        props_.emplace_back("mode", "YUV2RGB_YV12");
        break;
    case CvtColor::Mode::YUV2BGR_YV12:
        props_.emplace_back("mode", "YUV2BGR_YV12");
        break;
    case CvtColor::Mode::YUV2GRAY_YU12:
        props_.emplace_back("mode", "YUV2GRAY_YU12");
        break;
    case CvtColor::Mode::YUV2RGB_YU12:
        props_.emplace_back("mode", "YUV2RGB_YU12");
        break;
    case CvtColor::Mode::YUV2BGR_YU12:
        props_.emplace_back("mode", "YUV2BGR_YU12");
        break;
    case CvtColor::Mode::YCrCb2RGB:
        props_.emplace_back("mode", "YCrCb2RGB");
        break;
    case CvtColor::Mode::YCrCb2BGR:
        props_.emplace_back("mode", "YCrCb2BGR");
        break;
    case CvtColor::Mode::BT601_YUV2RGB_NV21:
        props_.emplace_back("mode", "BT601_YUV2RGB_NV21");
        break;
    case CvtColor::Mode::BT601_YUV2BGR_NV21:
        props_.emplace_back("mode", "BT601_YUV2BGR_NV21");
        break;
    case CvtColor::Mode::BT601_YUV2RGB_NV12:
        props_.emplace_back("mode", "BT601_YUV2RGB_NV12");
        break;
    case CvtColor::Mode::BT601_YUV2BGR_NV12:
        props_.emplace_back("mode", "BT601_YUV2BGR_NV12");
        break;
    case CvtColor::Mode::BT601_YUV2RGB_YV12:
        props_.emplace_back("mode", "BT601_YUV2RGB_YV12");
        break;
    case CvtColor::Mode::BT601_YUV2BGR_YV12:
        props_.emplace_back("mode", "BT601_YUV2BGR_YV12");
        break;
    case CvtColor::Mode::BT601_YUV2RGB_YU12:
        props_.emplace_back("mode", "BT601_YUV2RGB_YU12");
        break;
    case CvtColor::Mode::BT601_YUV2BGR_YU12:
        props_.emplace_back("mode", "BT601_YUV2BGR_YU12");
        break;
    default:
        props_.emplace_back("mode", "INVALID");
        break;
    }
    return props_;
}
std::string CvtColor_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<CvtColor>();
    static_cast<void>(op_);
    return "CvtColor";
}
} // anonymous namespace
OP_TRAIT_REG(CvtColor, CvtColor)
    .hash(CvtColor_hash_impl)
    .is_same_st(CvtColor_is_same_st_impl)
    .props(CvtColor_props_impl)
    .make_name(CvtColor_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(DeformableConv);

namespace {
size_t DeformableConv_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<DeformableConv>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.mode));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pad_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pad_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dilate_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dilate_w));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.sparse));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.format));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.compute_mode));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.strategy));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.workspace_limit));
    return val;
}
bool DeformableConv_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<DeformableConv>(),
         &&b_ = rhs_.cast_final_safe<DeformableConv>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.mode != b_.mode) return false;
    if (a_.pad_h != b_.pad_h) return false;
    if (a_.pad_w != b_.pad_w) return false;
    if (a_.stride_h != b_.stride_h) return false;
    if (a_.stride_w != b_.stride_w) return false;
    if (a_.dilate_h != b_.dilate_h) return false;
    if (a_.dilate_w != b_.dilate_w) return false;
    if (a_.sparse != b_.sparse) return false;
    if (a_.format != b_.format) return false;
    if (a_.compute_mode != b_.compute_mode) return false;
    if (a_.strategy != b_.strategy) return false;
    if (a_.workspace_limit != b_.workspace_limit) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> DeformableConv_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<DeformableConv>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.mode){
    case DeformableConv::Mode::CROSS_CORRELATION:
        props_.emplace_back("mode", "CROSS_CORRELATION");
        break;
    case DeformableConv::Mode::CONVOLUTION:
        props_.emplace_back("mode", "CONVOLUTION");
        break;
    default:
        props_.emplace_back("mode", "INVALID");
        break;
    }
    props_.emplace_back("pad_h", std::to_string(op_.pad_h));
    props_.emplace_back("pad_w", std::to_string(op_.pad_w));
    props_.emplace_back("stride_h", std::to_string(op_.stride_h));
    props_.emplace_back("stride_w", std::to_string(op_.stride_w));
    props_.emplace_back("dilate_h", std::to_string(op_.dilate_h));
    props_.emplace_back("dilate_w", std::to_string(op_.dilate_w));
    switch (op_.sparse){
    case DeformableConv::Sparse::DENSE:
        props_.emplace_back("sparse", "DENSE");
        break;
    case DeformableConv::Sparse::GROUP:
        props_.emplace_back("sparse", "GROUP");
        break;
    default:
        props_.emplace_back("sparse", "INVALID");
        break;
    }
    switch (op_.format){
    case DeformableConv::Format::NCHW:
        props_.emplace_back("format", "NCHW");
        break;
    case DeformableConv::Format::NHWC:
        props_.emplace_back("format", "NHWC");
        break;
    case DeformableConv::Format::NHWCD4:
        props_.emplace_back("format", "NHWCD4");
        break;
    case DeformableConv::Format::NCHW4:
        props_.emplace_back("format", "NCHW4");
        break;
    case DeformableConv::Format::NCHW8:
        props_.emplace_back("format", "NCHW8");
        break;
    case DeformableConv::Format::NCHW32:
        props_.emplace_back("format", "NCHW32");
        break;
    case DeformableConv::Format::NCHW88:
        props_.emplace_back("format", "NCHW88");
        break;
    case DeformableConv::Format::NCHW44:
        props_.emplace_back("format", "NCHW44");
        break;
    case DeformableConv::Format::NCHW44_DOT:
        props_.emplace_back("format", "NCHW44_DOT");
        break;
    case DeformableConv::Format::NCHW4_NCHW32:
        props_.emplace_back("format", "NCHW4_NCHW32");
        break;
    case DeformableConv::Format::NCHW32_NCHW4:
        props_.emplace_back("format", "NCHW32_NCHW4");
        break;
    case DeformableConv::Format::NCHW4_NCHW:
        props_.emplace_back("format", "NCHW4_NCHW");
        break;
    case DeformableConv::Format::NHWC_NCHW:
        props_.emplace_back("format", "NHWC_NCHW");
        break;
    case DeformableConv::Format::NHWC_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NHWC_NCHW4_IC_SMALL");
        break;
    case DeformableConv::Format::NCHW_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NCHW_NCHW4_IC_SMALL");
        break;
    case DeformableConv::Format::CHWN4:
        props_.emplace_back("format", "CHWN4");
        break;
    case DeformableConv::Format::NCHW64:
        props_.emplace_back("format", "NCHW64");
        break;
    case DeformableConv::Format::NCHW4_NHWC:
        props_.emplace_back("format", "NCHW4_NHWC");
        break;
    default:
        props_.emplace_back("format", "INVALID");
        break;
    }
    switch (op_.compute_mode){
    case DeformableConv::ComputeMode::DEFAULT:
        props_.emplace_back("compute_mode", "DEFAULT");
        break;
    case DeformableConv::ComputeMode::FLOAT32:
        props_.emplace_back("compute_mode", "FLOAT32");
        break;
    default:
        props_.emplace_back("compute_mode", "INVALID");
        break;
    }
    switch (op_.strategy){
    case DeformableConv::Strategy::HEURISTIC:
        props_.emplace_back("strategy", "HEURISTIC");
        break;
    case DeformableConv::Strategy::PROFILE:
        props_.emplace_back("strategy", "PROFILE");
        break;
    case DeformableConv::Strategy::REPRODUCIBLE:
        props_.emplace_back("strategy", "REPRODUCIBLE");
        break;
    case DeformableConv::Strategy::OPTIMIZED:
        props_.emplace_back("strategy", "OPTIMIZED");
        break;
    default:
        props_.emplace_back("strategy", "INVALID");
        break;
    }
    props_.emplace_back("workspace_limit", std::to_string(op_.workspace_limit));
    return props_;
}
std::string DeformableConv_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<DeformableConv>();
    static_cast<void>(op_);
    return "DeformableConv";
}
} // anonymous namespace
OP_TRAIT_REG(DeformableConv, DeformableConv)
    .hash(DeformableConv_hash_impl)
    .is_same_st(DeformableConv_is_same_st_impl)
    .props(DeformableConv_props_impl)
    .make_name(DeformableConv_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(DeformablePSROIPooling);

namespace {
size_t DeformablePSROIPooling_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<DeformablePSROIPooling>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.no_trans));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.spatial_scale));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.trans_std));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pooled_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pooled_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.part_size));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.sample_per_part));
    return val;
}
bool DeformablePSROIPooling_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<DeformablePSROIPooling>(),
         &&b_ = rhs_.cast_final_safe<DeformablePSROIPooling>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.no_trans != b_.no_trans) return false;
    if (a_.spatial_scale != b_.spatial_scale) return false;
    if (a_.trans_std != b_.trans_std) return false;
    if (a_.pooled_h != b_.pooled_h) return false;
    if (a_.pooled_w != b_.pooled_w) return false;
    if (a_.part_size != b_.part_size) return false;
    if (a_.sample_per_part != b_.sample_per_part) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> DeformablePSROIPooling_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<DeformablePSROIPooling>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("no_trans", std::to_string(op_.no_trans));
    props_.emplace_back("spatial_scale", std::to_string(op_.spatial_scale));
    props_.emplace_back("trans_std", std::to_string(op_.trans_std));
    props_.emplace_back("pooled_h", std::to_string(op_.pooled_h));
    props_.emplace_back("pooled_w", std::to_string(op_.pooled_w));
    props_.emplace_back("part_size", std::to_string(op_.part_size));
    props_.emplace_back("sample_per_part", std::to_string(op_.sample_per_part));
    return props_;
}
std::string DeformablePSROIPooling_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<DeformablePSROIPooling>();
    static_cast<void>(op_);
    return "DeformablePSROIPooling";
}
} // anonymous namespace
OP_TRAIT_REG(DeformablePSROIPooling, DeformablePSROIPooling)
    .hash(DeformablePSROIPooling_hash_impl)
    .is_same_st(DeformablePSROIPooling_is_same_st_impl)
    .props(DeformablePSROIPooling_props_impl)
    .make_name(DeformablePSROIPooling_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Diag);

namespace {
size_t Diag_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Diag>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.k));
    return val;
}
bool Diag_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Diag>(),
         &&b_ = rhs_.cast_final_safe<Diag>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.k != b_.k) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Diag_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Diag>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("k", std::to_string(op_.k));
    return props_;
}
std::string Diag_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Diag>();
    static_cast<void>(op_);
    return "Diag";
}
} // anonymous namespace
OP_TRAIT_REG(Diag, Diag)
    .hash(Diag_hash_impl)
    .is_same_st(Diag_is_same_st_impl)
    .props(Diag_props_impl)
    .make_name(Diag_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Dimshuffle);

namespace {
size_t Dimshuffle_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Dimshuffle>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pattern));
    return val;
}
bool Dimshuffle_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Dimshuffle>(),
         &&b_ = rhs_.cast_final_safe<Dimshuffle>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.pattern != b_.pattern) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Dimshuffle_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Dimshuffle>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("pattern", "{std::vector}");
    return props_;
}
std::string Dimshuffle_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Dimshuffle>();
    static_cast<void>(op_);
    return "Dimshuffle";
}
} // anonymous namespace
OP_TRAIT_REG(Dimshuffle, Dimshuffle)
    .hash(Dimshuffle_hash_impl)
    .is_same_st(Dimshuffle_is_same_st_impl)
    .props(Dimshuffle_props_impl)
    .make_name(Dimshuffle_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Dot);

namespace {
size_t Dot_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Dot>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    return val;
}
bool Dot_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Dot>(),
         &&b_ = rhs_.cast_final_safe<Dot>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    return true;
}
std::vector<std::pair<const char*, std::string>> Dot_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Dot>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    return props_;
}
std::string Dot_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Dot>();
    static_cast<void>(op_);
    return "Dot";
}
} // anonymous namespace
OP_TRAIT_REG(Dot, Dot)
    .hash(Dot_hash_impl)
    .is_same_st(Dot_is_same_st_impl)
    .props(Dot_props_impl)
    .make_name(Dot_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Dropout);

namespace {
size_t Dropout_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Dropout>();
    static_cast<void>(op_);

    return mgb::hash_pair_combine(
      mgb::hash(op_.dyn_typeinfo()),
      mgb::hash_pair_combine(
        mgb::hash(op_.drop_prob),
        mgb::hash(op_.handle))
      );
  }
bool Dropout_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Dropout>(),
         &&b_ = rhs_.cast_final_safe<Dropout>();
    static_cast<void>(a_);
    static_cast<void>(b_);
return a_.handle == b_.handle && a_.drop_prob == b_.drop_prob;}
std::vector<std::pair<const char*, std::string>> Dropout_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Dropout>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("drop_prob", std::to_string(op_.drop_prob));
    props_.emplace_back("seed", std::to_string(op_.seed));
    props_.emplace_back("handle", std::to_string(op_.handle));
    return props_;
}
std::string Dropout_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Dropout>();
    static_cast<void>(op_);
    return "Dropout";
}
} // anonymous namespace
OP_TRAIT_REG(Dropout, Dropout)
    .hash(Dropout_hash_impl)
    .is_same_st(Dropout_is_same_st_impl)
    .props(Dropout_props_impl)
    .make_name(Dropout_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Elemwise);

namespace {
size_t Elemwise_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Elemwise>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.mode));
    return val;
}
bool Elemwise_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Elemwise>(),
         &&b_ = rhs_.cast_final_safe<Elemwise>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.mode != b_.mode) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Elemwise_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Elemwise>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.mode){
    case Elemwise::Mode::RELU:
        props_.emplace_back("mode", "RELU");
        break;
    case Elemwise::Mode::ABS:
        props_.emplace_back("mode", "ABS");
        break;
    case Elemwise::Mode::ACOS:
        props_.emplace_back("mode", "ACOS");
        break;
    case Elemwise::Mode::ASIN:
        props_.emplace_back("mode", "ASIN");
        break;
    case Elemwise::Mode::CEIL:
        props_.emplace_back("mode", "CEIL");
        break;
    case Elemwise::Mode::COS:
        props_.emplace_back("mode", "COS");
        break;
    case Elemwise::Mode::EXP:
        props_.emplace_back("mode", "EXP");
        break;
    case Elemwise::Mode::EXPM1:
        props_.emplace_back("mode", "EXPM1");
        break;
    case Elemwise::Mode::FLOOR:
        props_.emplace_back("mode", "FLOOR");
        break;
    case Elemwise::Mode::LOG:
        props_.emplace_back("mode", "LOG");
        break;
    case Elemwise::Mode::LOG1P:
        props_.emplace_back("mode", "LOG1P");
        break;
    case Elemwise::Mode::NEGATE:
        props_.emplace_back("mode", "NEGATE");
        break;
    case Elemwise::Mode::SIGMOID:
        props_.emplace_back("mode", "SIGMOID");
        break;
    case Elemwise::Mode::SIN:
        props_.emplace_back("mode", "SIN");
        break;
    case Elemwise::Mode::TANH:
        props_.emplace_back("mode", "TANH");
        break;
    case Elemwise::Mode::ABS_GRAD:
        props_.emplace_back("mode", "ABS_GRAD");
        break;
    case Elemwise::Mode::ADD:
        props_.emplace_back("mode", "ADD");
        break;
    case Elemwise::Mode::FLOOR_DIV:
        props_.emplace_back("mode", "FLOOR_DIV");
        break;
    case Elemwise::Mode::MAX:
        props_.emplace_back("mode", "MAX");
        break;
    case Elemwise::Mode::MIN:
        props_.emplace_back("mode", "MIN");
        break;
    case Elemwise::Mode::MOD:
        props_.emplace_back("mode", "MOD");
        break;
    case Elemwise::Mode::MUL:
        props_.emplace_back("mode", "MUL");
        break;
    case Elemwise::Mode::POW:
        props_.emplace_back("mode", "POW");
        break;
    case Elemwise::Mode::SIGMOID_GRAD:
        props_.emplace_back("mode", "SIGMOID_GRAD");
        break;
    case Elemwise::Mode::SUB:
        props_.emplace_back("mode", "SUB");
        break;
    case Elemwise::Mode::SWITCH_GT0:
        props_.emplace_back("mode", "SWITCH_GT0");
        break;
    case Elemwise::Mode::TANH_GRAD:
        props_.emplace_back("mode", "TANH_GRAD");
        break;
    case Elemwise::Mode::TRUE_DIV:
        props_.emplace_back("mode", "TRUE_DIV");
        break;
    case Elemwise::Mode::LOG_SUM_EXP:
        props_.emplace_back("mode", "LOG_SUM_EXP");
        break;
    case Elemwise::Mode::LT:
        props_.emplace_back("mode", "LT");
        break;
    case Elemwise::Mode::LEQ:
        props_.emplace_back("mode", "LEQ");
        break;
    case Elemwise::Mode::EQ:
        props_.emplace_back("mode", "EQ");
        break;
    case Elemwise::Mode::SHL:
        props_.emplace_back("mode", "SHL");
        break;
    case Elemwise::Mode::SHR:
        props_.emplace_back("mode", "SHR");
        break;
    case Elemwise::Mode::COND_LEQ_MOV:
        props_.emplace_back("mode", "COND_LEQ_MOV");
        break;
    case Elemwise::Mode::FUSE_MUL_ADD3:
        props_.emplace_back("mode", "FUSE_MUL_ADD3");
        break;
    case Elemwise::Mode::FUSE_MUL_ADD4:
        props_.emplace_back("mode", "FUSE_MUL_ADD4");
        break;
    case Elemwise::Mode::FUSE_ADD_RELU:
        props_.emplace_back("mode", "FUSE_ADD_RELU");
        break;
    case Elemwise::Mode::FUSE_ADD_SIGMOID:
        props_.emplace_back("mode", "FUSE_ADD_SIGMOID");
        break;
    case Elemwise::Mode::FUSE_ADD_TANH:
        props_.emplace_back("mode", "FUSE_ADD_TANH");
        break;
    case Elemwise::Mode::FAST_TANH:
        props_.emplace_back("mode", "FAST_TANH");
        break;
    case Elemwise::Mode::FAST_TANH_GRAD:
        props_.emplace_back("mode", "FAST_TANH_GRAD");
        break;
    case Elemwise::Mode::ROUND:
        props_.emplace_back("mode", "ROUND");
        break;
    case Elemwise::Mode::RMULH:
        props_.emplace_back("mode", "RMULH");
        break;
    case Elemwise::Mode::ATAN2:
        props_.emplace_back("mode", "ATAN2");
        break;
    case Elemwise::Mode::ERF:
        props_.emplace_back("mode", "ERF");
        break;
    case Elemwise::Mode::ERFINV:
        props_.emplace_back("mode", "ERFINV");
        break;
    case Elemwise::Mode::ERFC:
        props_.emplace_back("mode", "ERFC");
        break;
    case Elemwise::Mode::ERFCINV:
        props_.emplace_back("mode", "ERFCINV");
        break;
    case Elemwise::Mode::H_SWISH:
        props_.emplace_back("mode", "H_SWISH");
        break;
    case Elemwise::Mode::H_SWISH_GRAD:
        props_.emplace_back("mode", "H_SWISH_GRAD");
        break;
    case Elemwise::Mode::FUSE_ADD_H_SWISH:
        props_.emplace_back("mode", "FUSE_ADD_H_SWISH");
        break;
    case Elemwise::Mode::NOT:
        props_.emplace_back("mode", "NOT");
        break;
    case Elemwise::Mode::AND:
        props_.emplace_back("mode", "AND");
        break;
    case Elemwise::Mode::OR:
        props_.emplace_back("mode", "OR");
        break;
    case Elemwise::Mode::XOR:
        props_.emplace_back("mode", "XOR");
        break;
    case Elemwise::Mode::SILU:
        props_.emplace_back("mode", "SILU");
        break;
    case Elemwise::Mode::SILU_GRAD:
        props_.emplace_back("mode", "SILU_GRAD");
        break;
    case Elemwise::Mode::GELU:
        props_.emplace_back("mode", "GELU");
        break;
    case Elemwise::Mode::GELU_GRAD:
        props_.emplace_back("mode", "GELU_GRAD");
        break;
    case Elemwise::Mode::COND_LT_MOV:
        props_.emplace_back("mode", "COND_LT_MOV");
        break;
    case Elemwise::Mode::NEQ:
        props_.emplace_back("mode", "NEQ");
        break;
    case Elemwise::Mode::ISNAN:
        props_.emplace_back("mode", "ISNAN");
        break;
    case Elemwise::Mode::ISINF:
        props_.emplace_back("mode", "ISINF");
        break;
    case Elemwise::Mode::SINH:
        props_.emplace_back("mode", "SINH");
        break;
    case Elemwise::Mode::COSH:
        props_.emplace_back("mode", "COSH");
        break;
    case Elemwise::Mode::ASINH:
        props_.emplace_back("mode", "ASINH");
        break;
    case Elemwise::Mode::ACOSH:
        props_.emplace_back("mode", "ACOSH");
        break;
    case Elemwise::Mode::ATANH:
        props_.emplace_back("mode", "ATANH");
        break;
    case Elemwise::Mode::TAN:
        props_.emplace_back("mode", "TAN");
        break;
    case Elemwise::Mode::ASINH_GRAD:
        props_.emplace_back("mode", "ASINH_GRAD");
        break;
    case Elemwise::Mode::ACOSH_GRAD:
        props_.emplace_back("mode", "ACOSH_GRAD");
        break;
    case Elemwise::Mode::ATANH_GRAD:
        props_.emplace_back("mode", "ATANH_GRAD");
        break;
    case Elemwise::Mode::PRELU:
        props_.emplace_back("mode", "PRELU");
        break;
    case Elemwise::Mode::CLIP:
        props_.emplace_back("mode", "CLIP");
        break;
    case Elemwise::Mode::PRELU_GRAD:
        props_.emplace_back("mode", "PRELU_GRAD");
        break;
    case Elemwise::Mode::SOFTPLUS:
        props_.emplace_back("mode", "SOFTPLUS");
        break;
    case Elemwise::Mode::SOFTPLUS_GRAD:
        props_.emplace_back("mode", "SOFTPLUS_GRAD");
        break;
    case Elemwise::Mode::RELU6:
        props_.emplace_back("mode", "RELU6");
        break;
    case Elemwise::Mode::RELU6_GRAD:
        props_.emplace_back("mode", "RELU6_GRAD");
        break;
    case Elemwise::Mode::HSIGMOID:
        props_.emplace_back("mode", "HSIGMOID");
        break;
    case Elemwise::Mode::HSIGMOID_GRAD:
        props_.emplace_back("mode", "HSIGMOID_GRAD");
        break;
    case Elemwise::Mode::LOGSIGMOID:
        props_.emplace_back("mode", "LOGSIGMOID");
        break;
    case Elemwise::Mode::SQRT:
        props_.emplace_back("mode", "SQRT");
        break;
    case Elemwise::Mode::SQUARE:
        props_.emplace_back("mode", "SQUARE");
        break;
    case Elemwise::Mode::SIGN:
        props_.emplace_back("mode", "SIGN");
        break;
    default:
        props_.emplace_back("mode", "INVALID");
        break;
    }
    return props_;
}
std::string Elemwise_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Elemwise>();
    static_cast<void>(op_);

    return to_string(op_.mode);
  }
} // anonymous namespace
OP_TRAIT_REG(Elemwise, Elemwise)
    .hash(Elemwise_hash_impl)
    .is_same_st(Elemwise_is_same_st_impl)
    .props(Elemwise_props_impl)
    .make_name(Elemwise_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(ElemwiseMultiType);

namespace {
size_t ElemwiseMultiType_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<ElemwiseMultiType>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.mode));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dtype.handle()));
    return val;
}
bool ElemwiseMultiType_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<ElemwiseMultiType>(),
         &&b_ = rhs_.cast_final_safe<ElemwiseMultiType>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.mode != b_.mode) return false;
    if (a_.dtype != b_.dtype) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> ElemwiseMultiType_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<ElemwiseMultiType>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.mode){
    case ElemwiseMultiType::Mode::FUSE_MUL_ADD3_INT16x32x32x32:
        props_.emplace_back("mode", "FUSE_MUL_ADD3_INT16x32x32x32");
        break;
    case ElemwiseMultiType::Mode::FUSE_MUL_ADD3_IXxF32xF32xI8:
        props_.emplace_back("mode", "FUSE_MUL_ADD3_IXxF32xF32xI8");
        break;
    case ElemwiseMultiType::Mode::ROUND_SHR_SATURATE_IXxI8xI8:
        props_.emplace_back("mode", "ROUND_SHR_SATURATE_IXxI8xI8");
        break;
    case ElemwiseMultiType::Mode::FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT16x16x16x8:
        props_.emplace_back("mode", "FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT16x16x16x8");
        break;
    case ElemwiseMultiType::Mode::FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT32x32x32x8:
        props_.emplace_back("mode", "FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT32x32x32x8");
        break;
    case ElemwiseMultiType::Mode::ROUND_SHR_SATURATE_IXxI8xI16:
        props_.emplace_back("mode", "ROUND_SHR_SATURATE_IXxI8xI16");
        break;
    case ElemwiseMultiType::Mode::QADD:
        props_.emplace_back("mode", "QADD");
        break;
    case ElemwiseMultiType::Mode::QFUSE_ADD_RELU:
        props_.emplace_back("mode", "QFUSE_ADD_RELU");
        break;
    case ElemwiseMultiType::Mode::QMUL:
        props_.emplace_back("mode", "QMUL");
        break;
    case ElemwiseMultiType::Mode::QMIN:
        props_.emplace_back("mode", "QMIN");
        break;
    case ElemwiseMultiType::Mode::QMAX:
        props_.emplace_back("mode", "QMAX");
        break;
    case ElemwiseMultiType::Mode::QSUB:
        props_.emplace_back("mode", "QSUB");
        break;
    case ElemwiseMultiType::Mode::QTRUE_DIV:
        props_.emplace_back("mode", "QTRUE_DIV");
        break;
    case ElemwiseMultiType::Mode::QFUSE_ADD_SIGMOID:
        props_.emplace_back("mode", "QFUSE_ADD_SIGMOID");
        break;
    case ElemwiseMultiType::Mode::QFUSE_ADD_TANH:
        props_.emplace_back("mode", "QFUSE_ADD_TANH");
        break;
    case ElemwiseMultiType::Mode::QRELU:
        props_.emplace_back("mode", "QRELU");
        break;
    case ElemwiseMultiType::Mode::QABS:
        props_.emplace_back("mode", "QABS");
        break;
    case ElemwiseMultiType::Mode::QSIGMOID:
        props_.emplace_back("mode", "QSIGMOID");
        break;
    case ElemwiseMultiType::Mode::QEXP:
        props_.emplace_back("mode", "QEXP");
        break;
    case ElemwiseMultiType::Mode::QTANH:
        props_.emplace_back("mode", "QTANH");
        break;
    case ElemwiseMultiType::Mode::QFUSE_MUL_ADD3:
        props_.emplace_back("mode", "QFUSE_MUL_ADD3");
        break;
    case ElemwiseMultiType::Mode::QFAST_TANH:
        props_.emplace_back("mode", "QFAST_TANH");
        break;
    case ElemwiseMultiType::Mode::QNEGATE:
        props_.emplace_back("mode", "QNEGATE");
        break;
    case ElemwiseMultiType::Mode::QACOS:
        props_.emplace_back("mode", "QACOS");
        break;
    case ElemwiseMultiType::Mode::QASIN:
        props_.emplace_back("mode", "QASIN");
        break;
    case ElemwiseMultiType::Mode::QCEIL:
        props_.emplace_back("mode", "QCEIL");
        break;
    case ElemwiseMultiType::Mode::QCOS:
        props_.emplace_back("mode", "QCOS");
        break;
    case ElemwiseMultiType::Mode::QEXPM1:
        props_.emplace_back("mode", "QEXPM1");
        break;
    case ElemwiseMultiType::Mode::QFLOOR:
        props_.emplace_back("mode", "QFLOOR");
        break;
    case ElemwiseMultiType::Mode::QLOG:
        props_.emplace_back("mode", "QLOG");
        break;
    case ElemwiseMultiType::Mode::QLOG1P:
        props_.emplace_back("mode", "QLOG1P");
        break;
    case ElemwiseMultiType::Mode::QSIN:
        props_.emplace_back("mode", "QSIN");
        break;
    case ElemwiseMultiType::Mode::QROUND:
        props_.emplace_back("mode", "QROUND");
        break;
    case ElemwiseMultiType::Mode::QERF:
        props_.emplace_back("mode", "QERF");
        break;
    case ElemwiseMultiType::Mode::QERFINV:
        props_.emplace_back("mode", "QERFINV");
        break;
    case ElemwiseMultiType::Mode::QERFC:
        props_.emplace_back("mode", "QERFC");
        break;
    case ElemwiseMultiType::Mode::QERFCINV:
        props_.emplace_back("mode", "QERFCINV");
        break;
    case ElemwiseMultiType::Mode::QABS_GRAD:
        props_.emplace_back("mode", "QABS_GRAD");
        break;
    case ElemwiseMultiType::Mode::QFLOOR_DIV:
        props_.emplace_back("mode", "QFLOOR_DIV");
        break;
    case ElemwiseMultiType::Mode::QMOD:
        props_.emplace_back("mode", "QMOD");
        break;
    case ElemwiseMultiType::Mode::QSIGMOID_GRAD:
        props_.emplace_back("mode", "QSIGMOID_GRAD");
        break;
    case ElemwiseMultiType::Mode::QSWITCH_GT0:
        props_.emplace_back("mode", "QSWITCH_GT0");
        break;
    case ElemwiseMultiType::Mode::QTANH_GRAD:
        props_.emplace_back("mode", "QTANH_GRAD");
        break;
    case ElemwiseMultiType::Mode::QLT:
        props_.emplace_back("mode", "QLT");
        break;
    case ElemwiseMultiType::Mode::QLEQ:
        props_.emplace_back("mode", "QLEQ");
        break;
    case ElemwiseMultiType::Mode::QEQ:
        props_.emplace_back("mode", "QEQ");
        break;
    case ElemwiseMultiType::Mode::QPOW:
        props_.emplace_back("mode", "QPOW");
        break;
    case ElemwiseMultiType::Mode::QLOG_SUM_EXP:
        props_.emplace_back("mode", "QLOG_SUM_EXP");
        break;
    case ElemwiseMultiType::Mode::QFAST_TANH_GRAD:
        props_.emplace_back("mode", "QFAST_TANH_GRAD");
        break;
    case ElemwiseMultiType::Mode::QATAN2:
        props_.emplace_back("mode", "QATAN2");
        break;
    case ElemwiseMultiType::Mode::QCOND_LEQ_MOV:
        props_.emplace_back("mode", "QCOND_LEQ_MOV");
        break;
    case ElemwiseMultiType::Mode::QH_SWISH:
        props_.emplace_back("mode", "QH_SWISH");
        break;
    case ElemwiseMultiType::Mode::QFUSE_ADD_H_SWISH:
        props_.emplace_back("mode", "QFUSE_ADD_H_SWISH");
        break;
    case ElemwiseMultiType::Mode::QH_SWISH_GRAD:
        props_.emplace_back("mode", "QH_SWISH_GRAD");
        break;
    case ElemwiseMultiType::Mode::FUSE_MUL_ADD3_INT16xF32xF32xF32:
        props_.emplace_back("mode", "FUSE_MUL_ADD3_INT16xF32xF32xF32");
        break;
    case ElemwiseMultiType::Mode::MUL_INT16xF32xF32:
        props_.emplace_back("mode", "MUL_INT16xF32xF32");
        break;
    case ElemwiseMultiType::Mode::FUSE_MUL_ADD3_UINT8xF32xF32xF32:
        props_.emplace_back("mode", "FUSE_MUL_ADD3_UINT8xF32xF32xF32");
        break;
    case ElemwiseMultiType::Mode::QCOND_LT_MOV:
        props_.emplace_back("mode", "QCOND_LT_MOV");
        break;
    case ElemwiseMultiType::Mode::EQ:
        props_.emplace_back("mode", "EQ");
        break;
    case ElemwiseMultiType::Mode::NEQ:
        props_.emplace_back("mode", "NEQ");
        break;
    case ElemwiseMultiType::Mode::LT:
        props_.emplace_back("mode", "LT");
        break;
    case ElemwiseMultiType::Mode::LEQ:
        props_.emplace_back("mode", "LEQ");
        break;
    case ElemwiseMultiType::Mode::ISNAN:
        props_.emplace_back("mode", "ISNAN");
        break;
    case ElemwiseMultiType::Mode::ISINF:
        props_.emplace_back("mode", "ISINF");
        break;
    default:
        props_.emplace_back("mode", "INVALID");
        break;
    }
    props_.emplace_back("dtype", op_.dtype.name());
    return props_;
}
std::string ElemwiseMultiType_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<ElemwiseMultiType>();
    static_cast<void>(op_);

    return to_string(op_.mode);
  }
} // anonymous namespace
OP_TRAIT_REG(ElemwiseMultiType, ElemwiseMultiType)
    .hash(ElemwiseMultiType_hash_impl)
    .is_same_st(ElemwiseMultiType_is_same_st_impl)
    .props(ElemwiseMultiType_props_impl)
    .make_name(ElemwiseMultiType_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(ExternOpr);

namespace {
size_t ExternOpr_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<ExternOpr>();
    static_cast<void>(op_);

    return mgb::hash_pair_combine(
      mgb::hash(op_.dyn_typeinfo()),
      mgb::hash_pair_combine(
        mgb::hash(op_.name),
        mgb::hash(op_.data))
      );
  }
bool ExternOpr_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<ExternOpr>(),
         &&b_ = rhs_.cast_final_safe<ExternOpr>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.output_shapes != b_.output_shapes) return false;
    if (a_.name != b_.name) return false;
    if (a_.data != b_.data) return false;
    if (a_.data_len != b_.data_len) return false;
    if (a_.output_dtypes != b_.output_dtypes) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> ExternOpr_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<ExternOpr>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("output_shapes", "{std::vector}");
    props_.emplace_back("name", op_.name);
    props_.emplace_back("data", op_.data);
    props_.emplace_back("data_len", std::to_string(op_.data_len));
    props_.emplace_back("output_dtypes", "{std::vector}");
    return props_;
}
std::string ExternOpr_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<ExternOpr>();
    static_cast<void>(op_);
    return "ExternOpr";
}
} // anonymous namespace
OP_TRAIT_REG(ExternOpr, ExternOpr)
    .hash(ExternOpr_hash_impl)
    .is_same_st(ExternOpr_is_same_st_impl)
    .props(ExternOpr_props_impl)
    .make_name(ExternOpr_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Eye);

namespace {
size_t Eye_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Eye>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.k));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dtype.handle()));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.comp_node));
    return val;
}
bool Eye_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Eye>(),
         &&b_ = rhs_.cast_final_safe<Eye>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.k != b_.k) return false;
    if (a_.dtype != b_.dtype) return false;
    if (a_.comp_node != b_.comp_node) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Eye_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Eye>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("k", std::to_string(op_.k));
    props_.emplace_back("dtype", op_.dtype.name());
    props_.emplace_back("comp_node", op_.comp_node.to_string());
    return props_;
}
std::string Eye_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Eye>();
    static_cast<void>(op_);
    return "Eye";
}
} // anonymous namespace
OP_TRAIT_REG(Eye, Eye)
    .hash(Eye_hash_impl)
    .is_same_st(Eye_is_same_st_impl)
    .props(Eye_props_impl)
    .make_name(Eye_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(FakeQuant);

namespace {
size_t FakeQuant_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<FakeQuant>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.qmin));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.qmax));
    return val;
}
bool FakeQuant_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<FakeQuant>(),
         &&b_ = rhs_.cast_final_safe<FakeQuant>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.qmin != b_.qmin) return false;
    if (a_.qmax != b_.qmax) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> FakeQuant_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<FakeQuant>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("qmin", std::to_string(op_.qmin));
    props_.emplace_back("qmax", std::to_string(op_.qmax));
    return props_;
}
std::string FakeQuant_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<FakeQuant>();
    static_cast<void>(op_);
    return "FakeQuant";
}
} // anonymous namespace
OP_TRAIT_REG(FakeQuant, FakeQuant)
    .hash(FakeQuant_hash_impl)
    .is_same_st(FakeQuant_is_same_st_impl)
    .props(FakeQuant_props_impl)
    .make_name(FakeQuant_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(FastpathCopy);

namespace {
size_t FastpathCopy_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<FastpathCopy>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    return val;
}
bool FastpathCopy_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<FastpathCopy>(),
         &&b_ = rhs_.cast_final_safe<FastpathCopy>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    return true;
}
std::vector<std::pair<const char*, std::string>> FastpathCopy_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<FastpathCopy>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    return props_;
}
std::string FastpathCopy_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<FastpathCopy>();
    static_cast<void>(op_);
    return "FastpathCopy";
}
} // anonymous namespace
OP_TRAIT_REG(FastpathCopy, FastpathCopy)
    .hash(FastpathCopy_hash_impl)
    .is_same_st(FastpathCopy_is_same_st_impl)
    .props(FastpathCopy_props_impl)
    .make_name(FastpathCopy_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Fill);

namespace {
size_t Fill_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Fill>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.value));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dtype.handle()));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.comp_node));
    return val;
}
bool Fill_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Fill>(),
         &&b_ = rhs_.cast_final_safe<Fill>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.value != b_.value) return false;
    if (a_.dtype != b_.dtype) return false;
    if (a_.comp_node != b_.comp_node) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Fill_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Fill>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("value", std::to_string(op_.value));
    props_.emplace_back("dtype", op_.dtype.name());
    props_.emplace_back("comp_node", op_.comp_node.to_string());
    return props_;
}
std::string Fill_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Fill>();
    static_cast<void>(op_);
    return "Fill";
}
} // anonymous namespace
OP_TRAIT_REG(Fill, Fill)
    .hash(Fill_hash_impl)
    .is_same_st(Fill_is_same_st_impl)
    .props(Fill_props_impl)
    .make_name(Fill_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(FillLike);

namespace {
size_t FillLike_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<FillLike>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.value));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.comp_node));
    return val;
}
bool FillLike_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<FillLike>(),
         &&b_ = rhs_.cast_final_safe<FillLike>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.value != b_.value) return false;
    if (a_.comp_node != b_.comp_node) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> FillLike_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<FillLike>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("value", std::to_string(op_.value));
    props_.emplace_back("comp_node", op_.comp_node.to_string());
    return props_;
}
std::string FillLike_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<FillLike>();
    static_cast<void>(op_);
    return "FillLike";
}
} // anonymous namespace
OP_TRAIT_REG(FillLike, FillLike)
    .hash(FillLike_hash_impl)
    .is_same_st(FillLike_is_same_st_impl)
    .props(FillLike_props_impl)
    .make_name(FillLike_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(GammaRNG);

namespace {
size_t GammaRNG_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<GammaRNG>();
    static_cast<void>(op_);

    return mgb::hash_pair_combine(
      mgb::hash(op_.dyn_typeinfo()),
      mgb::hash(op_.handle)
      );
  }
bool GammaRNG_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<GammaRNG>(),
         &&b_ = rhs_.cast_final_safe<GammaRNG>();
    static_cast<void>(a_);
    static_cast<void>(b_);
return a_.handle == b_.handle;}
std::vector<std::pair<const char*, std::string>> GammaRNG_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<GammaRNG>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("seed", std::to_string(op_.seed));
    props_.emplace_back("handle", std::to_string(op_.handle));
    return props_;
}
std::string GammaRNG_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<GammaRNG>();
    static_cast<void>(op_);
    return "GammaRNG";
}
} // anonymous namespace
OP_TRAIT_REG(GammaRNG, GammaRNG)
    .hash(GammaRNG_hash_impl)
    .is_same_st(GammaRNG_is_same_st_impl)
    .props(GammaRNG_props_impl)
    .make_name(GammaRNG_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(GaussianRNG);

namespace {
size_t GaussianRNG_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<GaussianRNG>();
    static_cast<void>(op_);

    return mgb::hash_pair_combine(
      mgb::hash(op_.dyn_typeinfo()),
      mgb::hash_pair_combine(
        mgb::hash(op_.handle),
        mgb::hash_pair_combine(
          mgb::hash(op_.mean),
          mgb::hash_pair_combine(
            mgb::hash(op_.std),
            mgb::hash(op_.dtype.enumv())
          )
        )
      )
    );
  }
bool GaussianRNG_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<GaussianRNG>(),
         &&b_ = rhs_.cast_final_safe<GaussianRNG>();
    static_cast<void>(a_);
    static_cast<void>(b_);
return a_.handle == b_.handle && a_.mean == b_.mean && a_.std == b_.std && a_.dtype == b_.dtype;}
std::vector<std::pair<const char*, std::string>> GaussianRNG_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<GaussianRNG>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("seed", std::to_string(op_.seed));
    props_.emplace_back("mean", std::to_string(op_.mean));
    props_.emplace_back("std", std::to_string(op_.std));
    props_.emplace_back("dtype", op_.dtype.name());
    props_.emplace_back("handle", std::to_string(op_.handle));
    return props_;
}
std::string GaussianRNG_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<GaussianRNG>();
    static_cast<void>(op_);
    return "GaussianRNG";
}
} // anonymous namespace
OP_TRAIT_REG(GaussianRNG, GaussianRNG)
    .hash(GaussianRNG_hash_impl)
    .is_same_st(GaussianRNG_is_same_st_impl)
    .props(GaussianRNG_props_impl)
    .make_name(GaussianRNG_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(GeneralNorm);

namespace {
size_t GeneralNorm_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<GeneralNorm>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.affine));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.eps));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.axis_start));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.axis_end));
    return val;
}
bool GeneralNorm_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<GeneralNorm>(),
         &&b_ = rhs_.cast_final_safe<GeneralNorm>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.affine != b_.affine) return false;
    if (a_.eps != b_.eps) return false;
    if (a_.axis_start != b_.axis_start) return false;
    if (a_.axis_end != b_.axis_end) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> GeneralNorm_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<GeneralNorm>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("affine", std::to_string(op_.affine));
    props_.emplace_back("eps", std::to_string(op_.eps));
    props_.emplace_back("axis_start", std::to_string(op_.axis_start));
    props_.emplace_back("axis_end", std::to_string(op_.axis_end));
    return props_;
}
std::string GeneralNorm_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<GeneralNorm>();
    static_cast<void>(op_);
    return "GeneralNorm";
}
} // anonymous namespace
OP_TRAIT_REG(GeneralNorm, GeneralNorm)
    .hash(GeneralNorm_hash_impl)
    .is_same_st(GeneralNorm_is_same_st_impl)
    .props(GeneralNorm_props_impl)
    .make_name(GeneralNorm_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(GetVarShape);

namespace {
size_t GetVarShape_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<GetVarShape>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.axis));
    return val;
}
bool GetVarShape_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<GetVarShape>(),
         &&b_ = rhs_.cast_final_safe<GetVarShape>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.axis != b_.axis) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> GetVarShape_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<GetVarShape>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("axis", std::to_string(op_.axis));
    return props_;
}
std::string GetVarShape_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<GetVarShape>();
    static_cast<void>(op_);
    return "GetVarShape";
}
} // anonymous namespace
OP_TRAIT_REG(GetVarShape, GetVarShape)
    .hash(GetVarShape_hash_impl)
    .is_same_st(GetVarShape_is_same_st_impl)
    .props(GetVarShape_props_impl)
    .make_name(GetVarShape_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(GroupLocal);

namespace {
size_t GroupLocal_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<GroupLocal>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.mode));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pad_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pad_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dilate_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dilate_w));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.sparse));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.format));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.compute_mode));
    return val;
}
bool GroupLocal_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<GroupLocal>(),
         &&b_ = rhs_.cast_final_safe<GroupLocal>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.mode != b_.mode) return false;
    if (a_.pad_h != b_.pad_h) return false;
    if (a_.pad_w != b_.pad_w) return false;
    if (a_.stride_h != b_.stride_h) return false;
    if (a_.stride_w != b_.stride_w) return false;
    if (a_.dilate_h != b_.dilate_h) return false;
    if (a_.dilate_w != b_.dilate_w) return false;
    if (a_.sparse != b_.sparse) return false;
    if (a_.format != b_.format) return false;
    if (a_.compute_mode != b_.compute_mode) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> GroupLocal_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<GroupLocal>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.mode){
    case GroupLocal::Mode::CROSS_CORRELATION:
        props_.emplace_back("mode", "CROSS_CORRELATION");
        break;
    case GroupLocal::Mode::CONVOLUTION:
        props_.emplace_back("mode", "CONVOLUTION");
        break;
    default:
        props_.emplace_back("mode", "INVALID");
        break;
    }
    props_.emplace_back("pad_h", std::to_string(op_.pad_h));
    props_.emplace_back("pad_w", std::to_string(op_.pad_w));
    props_.emplace_back("stride_h", std::to_string(op_.stride_h));
    props_.emplace_back("stride_w", std::to_string(op_.stride_w));
    props_.emplace_back("dilate_h", std::to_string(op_.dilate_h));
    props_.emplace_back("dilate_w", std::to_string(op_.dilate_w));
    switch (op_.sparse){
    case GroupLocal::Sparse::DENSE:
        props_.emplace_back("sparse", "DENSE");
        break;
    case GroupLocal::Sparse::GROUP:
        props_.emplace_back("sparse", "GROUP");
        break;
    default:
        props_.emplace_back("sparse", "INVALID");
        break;
    }
    switch (op_.format){
    case GroupLocal::Format::NCHW:
        props_.emplace_back("format", "NCHW");
        break;
    case GroupLocal::Format::NHWC:
        props_.emplace_back("format", "NHWC");
        break;
    case GroupLocal::Format::NHWCD4:
        props_.emplace_back("format", "NHWCD4");
        break;
    case GroupLocal::Format::NCHW4:
        props_.emplace_back("format", "NCHW4");
        break;
    case GroupLocal::Format::NCHW8:
        props_.emplace_back("format", "NCHW8");
        break;
    case GroupLocal::Format::NCHW32:
        props_.emplace_back("format", "NCHW32");
        break;
    case GroupLocal::Format::NCHW88:
        props_.emplace_back("format", "NCHW88");
        break;
    case GroupLocal::Format::NCHW44:
        props_.emplace_back("format", "NCHW44");
        break;
    case GroupLocal::Format::NCHW44_DOT:
        props_.emplace_back("format", "NCHW44_DOT");
        break;
    case GroupLocal::Format::NCHW4_NCHW32:
        props_.emplace_back("format", "NCHW4_NCHW32");
        break;
    case GroupLocal::Format::NCHW32_NCHW4:
        props_.emplace_back("format", "NCHW32_NCHW4");
        break;
    case GroupLocal::Format::NCHW4_NCHW:
        props_.emplace_back("format", "NCHW4_NCHW");
        break;
    case GroupLocal::Format::NHWC_NCHW:
        props_.emplace_back("format", "NHWC_NCHW");
        break;
    case GroupLocal::Format::NHWC_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NHWC_NCHW4_IC_SMALL");
        break;
    case GroupLocal::Format::NCHW_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NCHW_NCHW4_IC_SMALL");
        break;
    case GroupLocal::Format::CHWN4:
        props_.emplace_back("format", "CHWN4");
        break;
    case GroupLocal::Format::NCHW64:
        props_.emplace_back("format", "NCHW64");
        break;
    case GroupLocal::Format::NCHW4_NHWC:
        props_.emplace_back("format", "NCHW4_NHWC");
        break;
    default:
        props_.emplace_back("format", "INVALID");
        break;
    }
    switch (op_.compute_mode){
    case GroupLocal::ComputeMode::DEFAULT:
        props_.emplace_back("compute_mode", "DEFAULT");
        break;
    case GroupLocal::ComputeMode::FLOAT32:
        props_.emplace_back("compute_mode", "FLOAT32");
        break;
    default:
        props_.emplace_back("compute_mode", "INVALID");
        break;
    }
    return props_;
}
std::string GroupLocal_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<GroupLocal>();
    static_cast<void>(op_);
    return "GroupLocal";
}
} // anonymous namespace
OP_TRAIT_REG(GroupLocal, GroupLocal)
    .hash(GroupLocal_hash_impl)
    .is_same_st(GroupLocal_is_same_st_impl)
    .props(GroupLocal_props_impl)
    .make_name(GroupLocal_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(GroupNorm);

namespace {
size_t GroupNorm_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<GroupNorm>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.affine));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.eps));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.group));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.format));
    return val;
}
bool GroupNorm_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<GroupNorm>(),
         &&b_ = rhs_.cast_final_safe<GroupNorm>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.affine != b_.affine) return false;
    if (a_.eps != b_.eps) return false;
    if (a_.group != b_.group) return false;
    if (a_.format != b_.format) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> GroupNorm_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<GroupNorm>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("affine", std::to_string(op_.affine));
    props_.emplace_back("eps", std::to_string(op_.eps));
    props_.emplace_back("group", std::to_string(op_.group));
    switch (op_.format){
    case GroupNorm::Format::NCHW:
        props_.emplace_back("format", "NCHW");
        break;
    case GroupNorm::Format::NHWC:
        props_.emplace_back("format", "NHWC");
        break;
    case GroupNorm::Format::NHWCD4:
        props_.emplace_back("format", "NHWCD4");
        break;
    case GroupNorm::Format::NCHW4:
        props_.emplace_back("format", "NCHW4");
        break;
    case GroupNorm::Format::NCHW8:
        props_.emplace_back("format", "NCHW8");
        break;
    case GroupNorm::Format::NCHW32:
        props_.emplace_back("format", "NCHW32");
        break;
    case GroupNorm::Format::NCHW88:
        props_.emplace_back("format", "NCHW88");
        break;
    case GroupNorm::Format::NCHW44:
        props_.emplace_back("format", "NCHW44");
        break;
    case GroupNorm::Format::NCHW44_DOT:
        props_.emplace_back("format", "NCHW44_DOT");
        break;
    case GroupNorm::Format::NCHW4_NCHW32:
        props_.emplace_back("format", "NCHW4_NCHW32");
        break;
    case GroupNorm::Format::NCHW32_NCHW4:
        props_.emplace_back("format", "NCHW32_NCHW4");
        break;
    case GroupNorm::Format::NCHW4_NCHW:
        props_.emplace_back("format", "NCHW4_NCHW");
        break;
    case GroupNorm::Format::NHWC_NCHW:
        props_.emplace_back("format", "NHWC_NCHW");
        break;
    case GroupNorm::Format::NHWC_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NHWC_NCHW4_IC_SMALL");
        break;
    case GroupNorm::Format::NCHW_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NCHW_NCHW4_IC_SMALL");
        break;
    case GroupNorm::Format::CHWN4:
        props_.emplace_back("format", "CHWN4");
        break;
    case GroupNorm::Format::NCHW64:
        props_.emplace_back("format", "NCHW64");
        break;
    case GroupNorm::Format::NCHW4_NHWC:
        props_.emplace_back("format", "NCHW4_NHWC");
        break;
    default:
        props_.emplace_back("format", "INVALID");
        break;
    }
    return props_;
}
std::string GroupNorm_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<GroupNorm>();
    static_cast<void>(op_);
    return "GroupNorm";
}
} // anonymous namespace
OP_TRAIT_REG(GroupNorm, GroupNorm)
    .hash(GroupNorm_hash_impl)
    .is_same_st(GroupNorm_is_same_st_impl)
    .props(GroupNorm_props_impl)
    .make_name(GroupNorm_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Identity);

namespace {
size_t Identity_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Identity>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    return val;
}
bool Identity_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Identity>(),
         &&b_ = rhs_.cast_final_safe<Identity>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    return true;
}
std::vector<std::pair<const char*, std::string>> Identity_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Identity>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    return props_;
}
std::string Identity_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Identity>();
    static_cast<void>(op_);
    return "Identity";
}
} // anonymous namespace
OP_TRAIT_REG(Identity, Identity)
    .hash(Identity_hash_impl)
    .is_same_st(Identity_is_same_st_impl)
    .props(Identity_props_impl)
    .make_name(Identity_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Images2Neibs);

namespace {
size_t Images2Neibs_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Images2Neibs>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pad_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pad_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dilate_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dilate_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.window_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.window_w));
    return val;
}
bool Images2Neibs_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Images2Neibs>(),
         &&b_ = rhs_.cast_final_safe<Images2Neibs>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.pad_h != b_.pad_h) return false;
    if (a_.pad_w != b_.pad_w) return false;
    if (a_.stride_h != b_.stride_h) return false;
    if (a_.stride_w != b_.stride_w) return false;
    if (a_.dilate_h != b_.dilate_h) return false;
    if (a_.dilate_w != b_.dilate_w) return false;
    if (a_.window_h != b_.window_h) return false;
    if (a_.window_w != b_.window_w) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Images2Neibs_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Images2Neibs>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("pad_h", std::to_string(op_.pad_h));
    props_.emplace_back("pad_w", std::to_string(op_.pad_w));
    props_.emplace_back("stride_h", std::to_string(op_.stride_h));
    props_.emplace_back("stride_w", std::to_string(op_.stride_w));
    props_.emplace_back("dilate_h", std::to_string(op_.dilate_h));
    props_.emplace_back("dilate_w", std::to_string(op_.dilate_w));
    props_.emplace_back("window_h", std::to_string(op_.window_h));
    props_.emplace_back("window_w", std::to_string(op_.window_w));
    return props_;
}
std::string Images2Neibs_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Images2Neibs>();
    static_cast<void>(op_);
    return "Images2Neibs";
}
} // anonymous namespace
OP_TRAIT_REG(Images2Neibs, Images2Neibs)
    .hash(Images2Neibs_hash_impl)
    .is_same_st(Images2Neibs_is_same_st_impl)
    .props(Images2Neibs_props_impl)
    .make_name(Images2Neibs_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(IncrMeshIndexing);

namespace {
size_t IncrMeshIndexing_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<IncrMeshIndexing>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.items));
    return val;
}
bool IncrMeshIndexing_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<IncrMeshIndexing>(),
         &&b_ = rhs_.cast_final_safe<IncrMeshIndexing>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.items != b_.items) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> IncrMeshIndexing_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<IncrMeshIndexing>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("items", "{std::vector}");
    return props_;
}
std::string IncrMeshIndexing_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<IncrMeshIndexing>();
    static_cast<void>(op_);
    return "IncrMeshIndexing";
}
} // anonymous namespace
OP_TRAIT_REG(IncrMeshIndexing, IncrMeshIndexing)
    .hash(IncrMeshIndexing_hash_impl)
    .is_same_st(IncrMeshIndexing_is_same_st_impl)
    .props(IncrMeshIndexing_props_impl)
    .make_name(IncrMeshIndexing_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(IncrSubtensor);

namespace {
size_t IncrSubtensor_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<IncrSubtensor>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.items));
    return val;
}
bool IncrSubtensor_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<IncrSubtensor>(),
         &&b_ = rhs_.cast_final_safe<IncrSubtensor>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.items != b_.items) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> IncrSubtensor_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<IncrSubtensor>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("items", "{std::vector}");
    return props_;
}
std::string IncrSubtensor_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<IncrSubtensor>();
    static_cast<void>(op_);
    return "IncrSubtensor";
}
} // anonymous namespace
OP_TRAIT_REG(IncrSubtensor, IncrSubtensor)
    .hash(IncrSubtensor_hash_impl)
    .is_same_st(IncrSubtensor_is_same_st_impl)
    .props(IncrSubtensor_props_impl)
    .make_name(IncrSubtensor_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(IndexingIncrMultiAxisVec);

namespace {
size_t IndexingIncrMultiAxisVec_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<IndexingIncrMultiAxisVec>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.items));
    return val;
}
bool IndexingIncrMultiAxisVec_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<IndexingIncrMultiAxisVec>(),
         &&b_ = rhs_.cast_final_safe<IndexingIncrMultiAxisVec>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.items != b_.items) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> IndexingIncrMultiAxisVec_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<IndexingIncrMultiAxisVec>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("items", "{std::vector}");
    return props_;
}
std::string IndexingIncrMultiAxisVec_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<IndexingIncrMultiAxisVec>();
    static_cast<void>(op_);
    return "IndexingIncrMultiAxisVec";
}
} // anonymous namespace
OP_TRAIT_REG(IndexingIncrMultiAxisVec, IndexingIncrMultiAxisVec)
    .hash(IndexingIncrMultiAxisVec_hash_impl)
    .is_same_st(IndexingIncrMultiAxisVec_is_same_st_impl)
    .props(IndexingIncrMultiAxisVec_props_impl)
    .make_name(IndexingIncrMultiAxisVec_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(IndexingMultiAxisVec);

namespace {
size_t IndexingMultiAxisVec_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<IndexingMultiAxisVec>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.items));
    return val;
}
bool IndexingMultiAxisVec_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<IndexingMultiAxisVec>(),
         &&b_ = rhs_.cast_final_safe<IndexingMultiAxisVec>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.items != b_.items) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> IndexingMultiAxisVec_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<IndexingMultiAxisVec>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("items", "{std::vector}");
    return props_;
}
std::string IndexingMultiAxisVec_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<IndexingMultiAxisVec>();
    static_cast<void>(op_);
    return "IndexingMultiAxisVec";
}
} // anonymous namespace
OP_TRAIT_REG(IndexingMultiAxisVec, IndexingMultiAxisVec)
    .hash(IndexingMultiAxisVec_hash_impl)
    .is_same_st(IndexingMultiAxisVec_is_same_st_impl)
    .props(IndexingMultiAxisVec_props_impl)
    .make_name(IndexingMultiAxisVec_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(IndexingOneHot);

namespace {
size_t IndexingOneHot_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<IndexingOneHot>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.axis));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.ndim));
    return val;
}
bool IndexingOneHot_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<IndexingOneHot>(),
         &&b_ = rhs_.cast_final_safe<IndexingOneHot>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.axis != b_.axis) return false;
    if (a_.ndim != b_.ndim) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> IndexingOneHot_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<IndexingOneHot>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("axis", std::to_string(op_.axis));
    props_.emplace_back("ndim", std::to_string(op_.ndim));
    return props_;
}
std::string IndexingOneHot_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<IndexingOneHot>();
    static_cast<void>(op_);
    return "IndexingOneHot";
}
} // anonymous namespace
OP_TRAIT_REG(IndexingOneHot, IndexingOneHot)
    .hash(IndexingOneHot_hash_impl)
    .is_same_st(IndexingOneHot_is_same_st_impl)
    .props(IndexingOneHot_props_impl)
    .make_name(IndexingOneHot_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(IndexingSetMultiAxisVec);

namespace {
size_t IndexingSetMultiAxisVec_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<IndexingSetMultiAxisVec>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.items));
    return val;
}
bool IndexingSetMultiAxisVec_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<IndexingSetMultiAxisVec>(),
         &&b_ = rhs_.cast_final_safe<IndexingSetMultiAxisVec>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.items != b_.items) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> IndexingSetMultiAxisVec_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<IndexingSetMultiAxisVec>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("items", "{std::vector}");
    return props_;
}
std::string IndexingSetMultiAxisVec_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<IndexingSetMultiAxisVec>();
    static_cast<void>(op_);
    return "IndexingSetMultiAxisVec";
}
} // anonymous namespace
OP_TRAIT_REG(IndexingSetMultiAxisVec, IndexingSetMultiAxisVec)
    .hash(IndexingSetMultiAxisVec_hash_impl)
    .is_same_st(IndexingSetMultiAxisVec_is_same_st_impl)
    .props(IndexingSetMultiAxisVec_props_impl)
    .make_name(IndexingSetMultiAxisVec_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(IndexingSetOneHot);

namespace {
size_t IndexingSetOneHot_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<IndexingSetOneHot>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.axis));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.ndim));
    return val;
}
bool IndexingSetOneHot_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<IndexingSetOneHot>(),
         &&b_ = rhs_.cast_final_safe<IndexingSetOneHot>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.axis != b_.axis) return false;
    if (a_.ndim != b_.ndim) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> IndexingSetOneHot_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<IndexingSetOneHot>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("axis", std::to_string(op_.axis));
    props_.emplace_back("ndim", std::to_string(op_.ndim));
    return props_;
}
std::string IndexingSetOneHot_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<IndexingSetOneHot>();
    static_cast<void>(op_);
    return "IndexingSetOneHot";
}
} // anonymous namespace
OP_TRAIT_REG(IndexingSetOneHot, IndexingSetOneHot)
    .hash(IndexingSetOneHot_hash_impl)
    .is_same_st(IndexingSetOneHot_is_same_st_impl)
    .props(IndexingSetOneHot_props_impl)
    .make_name(IndexingSetOneHot_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(InplaceAdd);

namespace {
size_t InplaceAdd_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<InplaceAdd>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    return val;
}
bool InplaceAdd_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<InplaceAdd>(),
         &&b_ = rhs_.cast_final_safe<InplaceAdd>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    return true;
}
std::vector<std::pair<const char*, std::string>> InplaceAdd_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<InplaceAdd>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    return props_;
}
std::string InplaceAdd_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<InplaceAdd>();
    static_cast<void>(op_);
    return "InplaceAdd";
}
} // anonymous namespace
OP_TRAIT_REG(InplaceAdd, InplaceAdd)
    .hash(InplaceAdd_hash_impl)
    .is_same_st(InplaceAdd_is_same_st_impl)
    .props(InplaceAdd_props_impl)
    .make_name(InplaceAdd_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(InstanceNorm);

namespace {
size_t InstanceNorm_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<InstanceNorm>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.affine));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.eps));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.group));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.format));
    return val;
}
bool InstanceNorm_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<InstanceNorm>(),
         &&b_ = rhs_.cast_final_safe<InstanceNorm>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.affine != b_.affine) return false;
    if (a_.eps != b_.eps) return false;
    if (a_.group != b_.group) return false;
    if (a_.format != b_.format) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> InstanceNorm_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<InstanceNorm>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("affine", std::to_string(op_.affine));
    props_.emplace_back("eps", std::to_string(op_.eps));
    props_.emplace_back("group", std::to_string(op_.group));
    switch (op_.format){
    case InstanceNorm::Format::NCHW:
        props_.emplace_back("format", "NCHW");
        break;
    case InstanceNorm::Format::NHWC:
        props_.emplace_back("format", "NHWC");
        break;
    case InstanceNorm::Format::NHWCD4:
        props_.emplace_back("format", "NHWCD4");
        break;
    case InstanceNorm::Format::NCHW4:
        props_.emplace_back("format", "NCHW4");
        break;
    case InstanceNorm::Format::NCHW8:
        props_.emplace_back("format", "NCHW8");
        break;
    case InstanceNorm::Format::NCHW32:
        props_.emplace_back("format", "NCHW32");
        break;
    case InstanceNorm::Format::NCHW88:
        props_.emplace_back("format", "NCHW88");
        break;
    case InstanceNorm::Format::NCHW44:
        props_.emplace_back("format", "NCHW44");
        break;
    case InstanceNorm::Format::NCHW44_DOT:
        props_.emplace_back("format", "NCHW44_DOT");
        break;
    case InstanceNorm::Format::NCHW4_NCHW32:
        props_.emplace_back("format", "NCHW4_NCHW32");
        break;
    case InstanceNorm::Format::NCHW32_NCHW4:
        props_.emplace_back("format", "NCHW32_NCHW4");
        break;
    case InstanceNorm::Format::NCHW4_NCHW:
        props_.emplace_back("format", "NCHW4_NCHW");
        break;
    case InstanceNorm::Format::NHWC_NCHW:
        props_.emplace_back("format", "NHWC_NCHW");
        break;
    case InstanceNorm::Format::NHWC_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NHWC_NCHW4_IC_SMALL");
        break;
    case InstanceNorm::Format::NCHW_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NCHW_NCHW4_IC_SMALL");
        break;
    case InstanceNorm::Format::CHWN4:
        props_.emplace_back("format", "CHWN4");
        break;
    case InstanceNorm::Format::NCHW64:
        props_.emplace_back("format", "NCHW64");
        break;
    case InstanceNorm::Format::NCHW4_NHWC:
        props_.emplace_back("format", "NCHW4_NHWC");
        break;
    default:
        props_.emplace_back("format", "INVALID");
        break;
    }
    return props_;
}
std::string InstanceNorm_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<InstanceNorm>();
    static_cast<void>(op_);
    return "InstanceNorm";
}
} // anonymous namespace
OP_TRAIT_REG(InstanceNorm, InstanceNorm)
    .hash(InstanceNorm_hash_impl)
    .is_same_st(InstanceNorm_is_same_st_impl)
    .props(InstanceNorm_props_impl)
    .make_name(InstanceNorm_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(LAMBUpdate);

namespace {
size_t LAMBUpdate_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<LAMBUpdate>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.beta_1));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.beta_2));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.step));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.lr));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.weight_decay));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.eps));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.bias_correction));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.always_adapt));
    return val;
}
bool LAMBUpdate_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<LAMBUpdate>(),
         &&b_ = rhs_.cast_final_safe<LAMBUpdate>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.beta_1 != b_.beta_1) return false;
    if (a_.beta_2 != b_.beta_2) return false;
    if (a_.step != b_.step) return false;
    if (a_.lr != b_.lr) return false;
    if (a_.weight_decay != b_.weight_decay) return false;
    if (a_.eps != b_.eps) return false;
    if (a_.bias_correction != b_.bias_correction) return false;
    if (a_.always_adapt != b_.always_adapt) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> LAMBUpdate_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<LAMBUpdate>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("beta_1", std::to_string(op_.beta_1));
    props_.emplace_back("beta_2", std::to_string(op_.beta_2));
    props_.emplace_back("step", std::to_string(op_.step));
    props_.emplace_back("lr", std::to_string(op_.lr));
    props_.emplace_back("weight_decay", std::to_string(op_.weight_decay));
    props_.emplace_back("eps", std::to_string(op_.eps));
    props_.emplace_back("bias_correction", std::to_string(op_.bias_correction));
    props_.emplace_back("always_adapt", std::to_string(op_.always_adapt));
    return props_;
}
std::string LAMBUpdate_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<LAMBUpdate>();
    static_cast<void>(op_);
    return "LAMBUpdate";
}
} // anonymous namespace
OP_TRAIT_REG(LAMBUpdate, LAMBUpdate)
    .hash(LAMBUpdate_hash_impl)
    .is_same_st(LAMBUpdate_is_same_st_impl)
    .props(LAMBUpdate_props_impl)
    .make_name(LAMBUpdate_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(LRN);

namespace {
size_t LRN_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<LRN>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.n));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.k));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.alpha));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.beta));
    return val;
}
bool LRN_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<LRN>(),
         &&b_ = rhs_.cast_final_safe<LRN>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.n != b_.n) return false;
    if (a_.k != b_.k) return false;
    if (a_.alpha != b_.alpha) return false;
    if (a_.beta != b_.beta) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> LRN_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<LRN>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("n", std::to_string(op_.n));
    props_.emplace_back("k", std::to_string(op_.k));
    props_.emplace_back("alpha", std::to_string(op_.alpha));
    props_.emplace_back("beta", std::to_string(op_.beta));
    return props_;
}
std::string LRN_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<LRN>();
    static_cast<void>(op_);
    return "LRN";
}
} // anonymous namespace
OP_TRAIT_REG(LRN, LRN)
    .hash(LRN_hash_impl)
    .is_same_st(LRN_is_same_st_impl)
    .props(LRN_props_impl)
    .make_name(LRN_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(LSQ);

namespace {
size_t LSQ_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<LSQ>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.qmin));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.qmax));
    return val;
}
bool LSQ_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<LSQ>(),
         &&b_ = rhs_.cast_final_safe<LSQ>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.qmin != b_.qmin) return false;
    if (a_.qmax != b_.qmax) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> LSQ_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<LSQ>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("qmin", std::to_string(op_.qmin));
    props_.emplace_back("qmax", std::to_string(op_.qmax));
    return props_;
}
std::string LSQ_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<LSQ>();
    static_cast<void>(op_);
    return "LSQ";
}
} // anonymous namespace
OP_TRAIT_REG(LSQ, LSQ)
    .hash(LSQ_hash_impl)
    .is_same_st(LSQ_is_same_st_impl)
    .props(LSQ_props_impl)
    .make_name(LSQ_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(LSTM);

namespace {
size_t LSTM_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<LSTM>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.num_layers));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.bidirectional));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.bias));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.hidden_size));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.proj_size));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dropout));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.fwd_mode));
    return val;
}
bool LSTM_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<LSTM>(),
         &&b_ = rhs_.cast_final_safe<LSTM>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.num_layers != b_.num_layers) return false;
    if (a_.bidirectional != b_.bidirectional) return false;
    if (a_.bias != b_.bias) return false;
    if (a_.hidden_size != b_.hidden_size) return false;
    if (a_.proj_size != b_.proj_size) return false;
    if (a_.dropout != b_.dropout) return false;
    if (a_.fwd_mode != b_.fwd_mode) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> LSTM_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<LSTM>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("num_layers", std::to_string(op_.num_layers));
    props_.emplace_back("bidirectional", std::to_string(op_.bidirectional));
    props_.emplace_back("bias", std::to_string(op_.bias));
    props_.emplace_back("hidden_size", std::to_string(op_.hidden_size));
    props_.emplace_back("proj_size", std::to_string(op_.proj_size));
    props_.emplace_back("dropout", std::to_string(op_.dropout));
    switch (op_.fwd_mode){
    case LSTM::FwdMode::TRAINING:
        props_.emplace_back("fwd_mode", "TRAINING");
        break;
    case LSTM::FwdMode::INFERENCE:
        props_.emplace_back("fwd_mode", "INFERENCE");
        break;
    default:
        props_.emplace_back("fwd_mode", "INVALID");
        break;
    }
    return props_;
}
std::string LSTM_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<LSTM>();
    static_cast<void>(op_);
    return "LSTM";
}
} // anonymous namespace
OP_TRAIT_REG(LSTM, LSTM)
    .hash(LSTM_hash_impl)
    .is_same_st(LSTM_is_same_st_impl)
    .props(LSTM_props_impl)
    .make_name(LSTM_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(LSTMCell);

namespace {
size_t LSTMCell_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<LSTMCell>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    return val;
}
bool LSTMCell_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<LSTMCell>(),
         &&b_ = rhs_.cast_final_safe<LSTMCell>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    return true;
}
std::vector<std::pair<const char*, std::string>> LSTMCell_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<LSTMCell>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    return props_;
}
std::string LSTMCell_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<LSTMCell>();
    static_cast<void>(op_);
    return "LSTMCell";
}
} // anonymous namespace
OP_TRAIT_REG(LSTMCell, LSTMCell)
    .hash(LSTMCell_hash_impl)
    .is_same_st(LSTMCell_is_same_st_impl)
    .props(LSTMCell_props_impl)
    .make_name(LSTMCell_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(LayerNorm);

namespace {
size_t LayerNorm_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<LayerNorm>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.affine));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.eps));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.normalized_dim));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.normalized_size));
    return val;
}
bool LayerNorm_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<LayerNorm>(),
         &&b_ = rhs_.cast_final_safe<LayerNorm>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.affine != b_.affine) return false;
    if (a_.eps != b_.eps) return false;
    if (a_.normalized_dim != b_.normalized_dim) return false;
    if (a_.normalized_size != b_.normalized_size) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> LayerNorm_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<LayerNorm>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("affine", std::to_string(op_.affine));
    props_.emplace_back("eps", std::to_string(op_.eps));
    props_.emplace_back("normalized_dim", std::to_string(op_.normalized_dim));
    props_.emplace_back("normalized_size", std::to_string(op_.normalized_size));
    return props_;
}
std::string LayerNorm_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<LayerNorm>();
    static_cast<void>(op_);
    return "LayerNorm";
}
} // anonymous namespace
OP_TRAIT_REG(LayerNorm, LayerNorm)
    .hash(LayerNorm_hash_impl)
    .is_same_st(LayerNorm_is_same_st_impl)
    .props(LayerNorm_props_impl)
    .make_name(LayerNorm_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Linspace);

namespace {
size_t Linspace_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Linspace>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.endpoint));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.comp_node));
    return val;
}
bool Linspace_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Linspace>(),
         &&b_ = rhs_.cast_final_safe<Linspace>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.endpoint != b_.endpoint) return false;
    if (a_.comp_node != b_.comp_node) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Linspace_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Linspace>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("endpoint", std::to_string(op_.endpoint));
    props_.emplace_back("comp_node", op_.comp_node.to_string());
    return props_;
}
std::string Linspace_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Linspace>();
    static_cast<void>(op_);
    return "Linspace";
}
} // anonymous namespace
OP_TRAIT_REG(Linspace, Linspace)
    .hash(Linspace_hash_impl)
    .is_same_st(Linspace_is_same_st_impl)
    .props(Linspace_props_impl)
    .make_name(Linspace_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(MagicMindRuntime);

namespace {
size_t MagicMindRuntime_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<MagicMindRuntime>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.buf));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.buf_size));
    return val;
}
bool MagicMindRuntime_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<MagicMindRuntime>(),
         &&b_ = rhs_.cast_final_safe<MagicMindRuntime>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.buf != b_.buf) return false;
    if (a_.buf_size != b_.buf_size) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> MagicMindRuntime_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<MagicMindRuntime>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("buf", op_.buf);
    props_.emplace_back("buf_size", std::to_string(op_.buf_size));
    return props_;
}
std::string MagicMindRuntime_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<MagicMindRuntime>();
    static_cast<void>(op_);
    return "MagicMindRuntime";
}
} // anonymous namespace
OP_TRAIT_REG(MagicMindRuntime, MagicMindRuntime)
    .hash(MagicMindRuntime_hash_impl)
    .is_same_st(MagicMindRuntime_is_same_st_impl)
    .props(MagicMindRuntime_props_impl)
    .make_name(MagicMindRuntime_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(MaskedFill);

namespace {
size_t MaskedFill_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<MaskedFill>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.value));
    return val;
}
bool MaskedFill_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<MaskedFill>(),
         &&b_ = rhs_.cast_final_safe<MaskedFill>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.value != b_.value) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> MaskedFill_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<MaskedFill>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("value", std::to_string(op_.value));
    return props_;
}
std::string MaskedFill_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<MaskedFill>();
    static_cast<void>(op_);
    return "MaskedFill";
}
} // anonymous namespace
OP_TRAIT_REG(MaskedFill, MaskedFill)
    .hash(MaskedFill_hash_impl)
    .is_same_st(MaskedFill_is_same_st_impl)
    .props(MaskedFill_props_impl)
    .make_name(MaskedFill_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(MatrixInverse);

namespace {
size_t MatrixInverse_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<MatrixInverse>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    return val;
}
bool MatrixInverse_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<MatrixInverse>(),
         &&b_ = rhs_.cast_final_safe<MatrixInverse>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    return true;
}
std::vector<std::pair<const char*, std::string>> MatrixInverse_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<MatrixInverse>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    return props_;
}
std::string MatrixInverse_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<MatrixInverse>();
    static_cast<void>(op_);
    return "MatrixInverse";
}
} // anonymous namespace
OP_TRAIT_REG(MatrixInverse, MatrixInverse)
    .hash(MatrixInverse_hash_impl)
    .is_same_st(MatrixInverse_is_same_st_impl)
    .props(MatrixInverse_props_impl)
    .make_name(MatrixInverse_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(MatrixMul);

namespace {
size_t MatrixMul_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<MatrixMul>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.transposeA));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.transposeB));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.compute_mode));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.format));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.strategy));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.workspace_limit));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dimA));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dimB));
    return val;
}
bool MatrixMul_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<MatrixMul>(),
         &&b_ = rhs_.cast_final_safe<MatrixMul>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.transposeA != b_.transposeA) return false;
    if (a_.transposeB != b_.transposeB) return false;
    if (a_.compute_mode != b_.compute_mode) return false;
    if (a_.format != b_.format) return false;
    if (a_.strategy != b_.strategy) return false;
    if (a_.workspace_limit != b_.workspace_limit) return false;
    if (a_.dimA != b_.dimA) return false;
    if (a_.dimB != b_.dimB) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> MatrixMul_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<MatrixMul>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("transposeA", std::to_string(op_.transposeA));
    props_.emplace_back("transposeB", std::to_string(op_.transposeB));
    switch (op_.compute_mode){
    case MatrixMul::ComputeMode::DEFAULT:
        props_.emplace_back("compute_mode", "DEFAULT");
        break;
    case MatrixMul::ComputeMode::FLOAT32:
        props_.emplace_back("compute_mode", "FLOAT32");
        break;
    default:
        props_.emplace_back("compute_mode", "INVALID");
        break;
    }
    switch (op_.format){
    case MatrixMul::Format::DEFAULT:
        props_.emplace_back("format", "DEFAULT");
        break;
    case MatrixMul::Format::MK4:
        props_.emplace_back("format", "MK4");
        break;
    case MatrixMul::Format::MK8:
        props_.emplace_back("format", "MK8");
        break;
    case MatrixMul::Format::MK4_DOT:
        props_.emplace_back("format", "MK4_DOT");
        break;
    case MatrixMul::Format::N32K4_DOT:
        props_.emplace_back("format", "N32K4_DOT");
        break;
    default:
        props_.emplace_back("format", "INVALID");
        break;
    }
    switch (op_.strategy){
    case MatrixMul::Strategy::HEURISTIC:
        props_.emplace_back("strategy", "HEURISTIC");
        break;
    case MatrixMul::Strategy::PROFILE:
        props_.emplace_back("strategy", "PROFILE");
        break;
    case MatrixMul::Strategy::REPRODUCIBLE:
        props_.emplace_back("strategy", "REPRODUCIBLE");
        break;
    case MatrixMul::Strategy::OPTIMIZED:
        props_.emplace_back("strategy", "OPTIMIZED");
        break;
    default:
        props_.emplace_back("strategy", "INVALID");
        break;
    }
    props_.emplace_back("workspace_limit", std::to_string(op_.workspace_limit));
    props_.emplace_back("dimA", std::to_string(op_.dimA));
    props_.emplace_back("dimB", std::to_string(op_.dimB));
    return props_;
}
std::string MatrixMul_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<MatrixMul>();
    static_cast<void>(op_);
    return "MatrixMul";
}
} // anonymous namespace
OP_TRAIT_REG(MatrixMul, MatrixMul)
    .hash(MatrixMul_hash_impl)
    .is_same_st(MatrixMul_is_same_st_impl)
    .props(MatrixMul_props_impl)
    .make_name(MatrixMul_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(MeshGrid);

namespace {
size_t MeshGrid_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<MeshGrid>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.indexing));
    return val;
}
bool MeshGrid_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<MeshGrid>(),
         &&b_ = rhs_.cast_final_safe<MeshGrid>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.indexing != b_.indexing) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> MeshGrid_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<MeshGrid>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("indexing", op_.indexing);
    return props_;
}
std::string MeshGrid_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<MeshGrid>();
    static_cast<void>(op_);
    return "MeshGrid";
}
} // anonymous namespace
OP_TRAIT_REG(MeshGrid, MeshGrid)
    .hash(MeshGrid_hash_impl)
    .is_same_st(MeshGrid_is_same_st_impl)
    .props(MeshGrid_props_impl)
    .make_name(MeshGrid_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(MeshIndexing);

namespace {
size_t MeshIndexing_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<MeshIndexing>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.items));
    return val;
}
bool MeshIndexing_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<MeshIndexing>(),
         &&b_ = rhs_.cast_final_safe<MeshIndexing>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.items != b_.items) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> MeshIndexing_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<MeshIndexing>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("items", "{std::vector}");
    return props_;
}
std::string MeshIndexing_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<MeshIndexing>();
    static_cast<void>(op_);
    return "MeshIndexing";
}
} // anonymous namespace
OP_TRAIT_REG(MeshIndexing, MeshIndexing)
    .hash(MeshIndexing_hash_impl)
    .is_same_st(MeshIndexing_is_same_st_impl)
    .props(MeshIndexing_props_impl)
    .make_name(MeshIndexing_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(MultiHeadAttn);

namespace {
size_t MultiHeadAttn_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<MultiHeadAttn>();
    static_cast<void>(op_);

    return mgb::hash_pair_combine(
      mgb::hash(op_.dyn_typeinfo()),
      mgb::hash_pair_combine(
        mgb::hash(op_.handle),
        mgb::hash_pair_combine(
          mgb::hash(op_.num_heads),
          mgb::hash_pair_combine(
            mgb::hash(op_.embeding_size),
            mgb::hash_pair_combine(
              mgb::hash(op_.k_size),
              mgb::hash_pair_combine(
                mgb::hash(op_.v_size),
                mgb::hash_pair_combine(
                  mgb::hash(op_.qproj_size),
                  mgb::hash_pair_combine(
                    mgb::hash(op_.kproj_size),
                    mgb::hash_pair_combine(
                      mgb::hash(op_.vproj_size),
                      mgb::hash_pair_combine(
                        mgb::hash(op_.oproj_size),
                        mgb::hash_pair_combine(
                          mgb::hash(op_.qbias),
                          mgb::hash_pair_combine(
                            mgb::hash(op_.kbias),
                            mgb::hash_pair_combine(
                              mgb::hash(op_.vbias),
                              mgb::hash_pair_combine(
                                mgb::hash(op_.obias),
                                mgb::hash_pair_combine(
                                  mgb::hash(op_.sm_scaler),
                                  mgb::hash_pair_combine(
                                    mgb::hash(op_.input_order),
                                    mgb::hash_pair_combine(
                                      mgb::hash(op_.attn_mask_type),
                                      mgb::hash_pair_combine(
                                        mgb::hash(op_.tensor_combination_type),
                                        mgb::hash_pair_combine(
                                          mgb::hash(op_.add_zero_attn),
                                          mgb::hash_pair_combine(
                                            mgb::hash(op_.need_weights),
                                            mgb::hash_pair_combine(
                                              mgb::hash(op_.reslink),
                                              mgb::hash_pair_combine(
                                                mgb::hash(op_.training),
                                                mgb::hash_pair_combine(
                                                  mgb::hash(op_.attn_prob),
                                                  mgb::hash(op_.out_prob))
                                                )
                                              )
                                            )
                                          )
                                        )
                                      )
                                    )
                                  )
                                )
                              )
                            )
                          )
                        )
                      )
                    )
                  )
                )
              )
            )
          )
        )
      );
  }
bool MultiHeadAttn_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<MultiHeadAttn>(),
         &&b_ = rhs_.cast_final_safe<MultiHeadAttn>();
    static_cast<void>(a_);
    static_cast<void>(b_);
return a_.handle == b_.handle && a_.num_heads == b_.num_heads && a_.embeding_size == b_.embeding_size && a_.k_size == b_.k_size && a_.v_size == b_.v_size && a_.qproj_size == b_.qproj_size && a_.kproj_size == b_.kproj_size && a_.vproj_size == b_.vproj_size && a_.oproj_size == b_.oproj_size && a_.qbias == b_.qbias && a_.kbias == b_.kbias && a_.vbias == b_.vbias && a_.obias == b_.obias && a_.sm_scaler == b_.sm_scaler && a_.input_order == b_.input_order && a_.reslink == b_.reslink && a_.training == b_.training && a_.need_weights == b_.need_weights && a_.attn_mask_type == b_.attn_mask_type && a_.add_zero_attn == b_.add_zero_attn && a_.tensor_combination_type == b_.tensor_combination_type && a_.attn_prob == b_.attn_prob && a_.out_prob == b_.out_prob;}
std::vector<std::pair<const char*, std::string>> MultiHeadAttn_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<MultiHeadAttn>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("num_heads", std::to_string(op_.num_heads));
    props_.emplace_back("embeding_size", std::to_string(op_.embeding_size));
    props_.emplace_back("k_size", std::to_string(op_.k_size));
    props_.emplace_back("v_size", std::to_string(op_.v_size));
    props_.emplace_back("qproj_size", std::to_string(op_.qproj_size));
    props_.emplace_back("kproj_size", std::to_string(op_.kproj_size));
    props_.emplace_back("vproj_size", std::to_string(op_.vproj_size));
    props_.emplace_back("oproj_size", std::to_string(op_.oproj_size));
    props_.emplace_back("qbias", std::to_string(op_.qbias));
    props_.emplace_back("kbias", std::to_string(op_.kbias));
    props_.emplace_back("vbias", std::to_string(op_.vbias));
    props_.emplace_back("obias", std::to_string(op_.obias));
    props_.emplace_back("sm_scaler", std::to_string(op_.sm_scaler));
    props_.emplace_back("input_order", std::to_string(op_.input_order));
    switch (op_.attn_mask_type){
    case MultiHeadAttn::AttnMaskType::NO_MASK:
        props_.emplace_back("attn_mask_type", "NO_MASK");
        break;
    case MultiHeadAttn::AttnMaskType::DEFAULT_MASK:
        props_.emplace_back("attn_mask_type", "DEFAULT_MASK");
        break;
    case MultiHeadAttn::AttnMaskType::CUDNN_STYLE_MASK:
        props_.emplace_back("attn_mask_type", "CUDNN_STYLE_MASK");
        break;
    case MultiHeadAttn::AttnMaskType::USER_DEFINED_MASK:
        props_.emplace_back("attn_mask_type", "USER_DEFINED_MASK");
        break;
    default:
        props_.emplace_back("attn_mask_type", "INVALID");
        break;
    }
    switch (op_.tensor_combination_type){
    case MultiHeadAttn::TensorCombinationType::NONE:
        props_.emplace_back("tensor_combination_type", "NONE");
        break;
    case MultiHeadAttn::TensorCombinationType::ONLY_MASK:
        props_.emplace_back("tensor_combination_type", "ONLY_MASK");
        break;
    case MultiHeadAttn::TensorCombinationType::ONLY_BIASKV:
        props_.emplace_back("tensor_combination_type", "ONLY_BIASKV");
        break;
    case MultiHeadAttn::TensorCombinationType::ALL:
        props_.emplace_back("tensor_combination_type", "ALL");
        break;
    default:
        props_.emplace_back("tensor_combination_type", "INVALID");
        break;
    }
    props_.emplace_back("add_zero_attn", std::to_string(op_.add_zero_attn));
    props_.emplace_back("need_weights", std::to_string(op_.need_weights));
    props_.emplace_back("reslink", std::to_string(op_.reslink));
    props_.emplace_back("training", std::to_string(op_.training));
    props_.emplace_back("seed", std::to_string(op_.seed));
    props_.emplace_back("attn_prob", std::to_string(op_.attn_prob));
    props_.emplace_back("out_prob", std::to_string(op_.out_prob));
    props_.emplace_back("handle", std::to_string(op_.handle));
    return props_;
}
std::string MultiHeadAttn_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<MultiHeadAttn>();
    static_cast<void>(op_);
    return "MultiHeadAttn";
}
} // anonymous namespace
OP_TRAIT_REG(MultiHeadAttn, MultiHeadAttn)
    .hash(MultiHeadAttn_hash_impl)
    .is_same_st(MultiHeadAttn_is_same_st_impl)
    .props(MultiHeadAttn_props_impl)
    .make_name(MultiHeadAttn_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(NMSKeep);

namespace {
size_t NMSKeep_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<NMSKeep>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.iou_thresh));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.max_output));
    return val;
}
bool NMSKeep_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<NMSKeep>(),
         &&b_ = rhs_.cast_final_safe<NMSKeep>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.iou_thresh != b_.iou_thresh) return false;
    if (a_.max_output != b_.max_output) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> NMSKeep_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<NMSKeep>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("iou_thresh", std::to_string(op_.iou_thresh));
    props_.emplace_back("max_output", std::to_string(op_.max_output));
    return props_;
}
std::string NMSKeep_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<NMSKeep>();
    static_cast<void>(op_);
    return "NMSKeep";
}
} // anonymous namespace
OP_TRAIT_REG(NMSKeep, NMSKeep)
    .hash(NMSKeep_hash_impl)
    .is_same_st(NMSKeep_is_same_st_impl)
    .props(NMSKeep_props_impl)
    .make_name(NMSKeep_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(NvOf);

namespace {
size_t NvOf_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<NvOf>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.precision));
    return val;
}
bool NvOf_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<NvOf>(),
         &&b_ = rhs_.cast_final_safe<NvOf>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.precision != b_.precision) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> NvOf_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<NvOf>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("precision", std::to_string(op_.precision));
    return props_;
}
std::string NvOf_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<NvOf>();
    static_cast<void>(op_);
    return "NvOf";
}
} // anonymous namespace
OP_TRAIT_REG(NvOf, NvOf)
    .hash(NvOf_hash_impl)
    .is_same_st(NvOf_is_same_st_impl)
    .props(NvOf_props_impl)
    .make_name(NvOf_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Padding);

namespace {
size_t Padding_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Padding>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.front_offset_dim0));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.front_offset_dim1));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.front_offset_dim2));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.front_offset_dim3));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.front_offset_dim4));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.front_offset_dim5));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.front_offset_dim6));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.back_offset_dim0));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.back_offset_dim1));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.back_offset_dim2));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.back_offset_dim3));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.back_offset_dim4));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.back_offset_dim5));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.back_offset_dim6));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.padding_val));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.padding_mode));
    return val;
}
bool Padding_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Padding>(),
         &&b_ = rhs_.cast_final_safe<Padding>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.front_offset_dim0 != b_.front_offset_dim0) return false;
    if (a_.front_offset_dim1 != b_.front_offset_dim1) return false;
    if (a_.front_offset_dim2 != b_.front_offset_dim2) return false;
    if (a_.front_offset_dim3 != b_.front_offset_dim3) return false;
    if (a_.front_offset_dim4 != b_.front_offset_dim4) return false;
    if (a_.front_offset_dim5 != b_.front_offset_dim5) return false;
    if (a_.front_offset_dim6 != b_.front_offset_dim6) return false;
    if (a_.back_offset_dim0 != b_.back_offset_dim0) return false;
    if (a_.back_offset_dim1 != b_.back_offset_dim1) return false;
    if (a_.back_offset_dim2 != b_.back_offset_dim2) return false;
    if (a_.back_offset_dim3 != b_.back_offset_dim3) return false;
    if (a_.back_offset_dim4 != b_.back_offset_dim4) return false;
    if (a_.back_offset_dim5 != b_.back_offset_dim5) return false;
    if (a_.back_offset_dim6 != b_.back_offset_dim6) return false;
    if (a_.padding_val != b_.padding_val) return false;
    if (a_.padding_mode != b_.padding_mode) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Padding_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Padding>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("front_offset_dim0", std::to_string(op_.front_offset_dim0));
    props_.emplace_back("front_offset_dim1", std::to_string(op_.front_offset_dim1));
    props_.emplace_back("front_offset_dim2", std::to_string(op_.front_offset_dim2));
    props_.emplace_back("front_offset_dim3", std::to_string(op_.front_offset_dim3));
    props_.emplace_back("front_offset_dim4", std::to_string(op_.front_offset_dim4));
    props_.emplace_back("front_offset_dim5", std::to_string(op_.front_offset_dim5));
    props_.emplace_back("front_offset_dim6", std::to_string(op_.front_offset_dim6));
    props_.emplace_back("back_offset_dim0", std::to_string(op_.back_offset_dim0));
    props_.emplace_back("back_offset_dim1", std::to_string(op_.back_offset_dim1));
    props_.emplace_back("back_offset_dim2", std::to_string(op_.back_offset_dim2));
    props_.emplace_back("back_offset_dim3", std::to_string(op_.back_offset_dim3));
    props_.emplace_back("back_offset_dim4", std::to_string(op_.back_offset_dim4));
    props_.emplace_back("back_offset_dim5", std::to_string(op_.back_offset_dim5));
    props_.emplace_back("back_offset_dim6", std::to_string(op_.back_offset_dim6));
    props_.emplace_back("padding_val", std::to_string(op_.padding_val));
    switch (op_.padding_mode){
    case Padding::PaddingMode::REPLICATE:
        props_.emplace_back("padding_mode", "REPLICATE");
        break;
    case Padding::PaddingMode::REFLECT:
        props_.emplace_back("padding_mode", "REFLECT");
        break;
    case Padding::PaddingMode::CONSTANT:
        props_.emplace_back("padding_mode", "CONSTANT");
        break;
    default:
        props_.emplace_back("padding_mode", "INVALID");
        break;
    }
    return props_;
}
std::string Padding_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Padding>();
    static_cast<void>(op_);
    return "Padding";
}
} // anonymous namespace
OP_TRAIT_REG(Padding, Padding)
    .hash(Padding_hash_impl)
    .is_same_st(Padding_is_same_st_impl)
    .props(Padding_props_impl)
    .make_name(Padding_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(ParamPackConcat);

namespace {
size_t ParamPackConcat_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<ParamPackConcat>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.offsets));
    return val;
}
bool ParamPackConcat_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<ParamPackConcat>(),
         &&b_ = rhs_.cast_final_safe<ParamPackConcat>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.offsets != b_.offsets) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> ParamPackConcat_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<ParamPackConcat>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("offsets", "{std::vector}");
    return props_;
}
std::string ParamPackConcat_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<ParamPackConcat>();
    static_cast<void>(op_);
    return "ParamPackConcat";
}
} // anonymous namespace
OP_TRAIT_REG(ParamPackConcat, ParamPackConcat)
    .hash(ParamPackConcat_hash_impl)
    .is_same_st(ParamPackConcat_is_same_st_impl)
    .props(ParamPackConcat_props_impl)
    .make_name(ParamPackConcat_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(ParamPackSplit);

namespace {
size_t ParamPackSplit_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<ParamPackSplit>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.offsets));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.shapes));
    return val;
}
bool ParamPackSplit_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<ParamPackSplit>(),
         &&b_ = rhs_.cast_final_safe<ParamPackSplit>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.offsets != b_.offsets) return false;
    if (a_.shapes != b_.shapes) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> ParamPackSplit_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<ParamPackSplit>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("offsets", "{std::vector}");
    props_.emplace_back("shapes", "{std::vector}");
    return props_;
}
std::string ParamPackSplit_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<ParamPackSplit>();
    static_cast<void>(op_);
    return "ParamPackSplit";
}
} // anonymous namespace
OP_TRAIT_REG(ParamPackSplit, ParamPackSplit)
    .hash(ParamPackSplit_hash_impl)
    .is_same_st(ParamPackSplit_is_same_st_impl)
    .props(ParamPackSplit_props_impl)
    .make_name(ParamPackSplit_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(PermutationRNG);

namespace {
size_t PermutationRNG_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<PermutationRNG>();
    static_cast<void>(op_);

    return mgb::hash_pair_combine(
      mgb::hash(op_.dyn_typeinfo()),
      mgb::hash_pair_combine(
        mgb::hash(op_.handle),
        mgb::hash(op_.dtype.enumv())
      )
    );
  }
bool PermutationRNG_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<PermutationRNG>(),
         &&b_ = rhs_.cast_final_safe<PermutationRNG>();
    static_cast<void>(a_);
    static_cast<void>(b_);
return a_.handle == b_.handle && a_.dtype == b_.dtype;}
std::vector<std::pair<const char*, std::string>> PermutationRNG_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<PermutationRNG>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("seed", std::to_string(op_.seed));
    props_.emplace_back("dtype", op_.dtype.name());
    props_.emplace_back("handle", std::to_string(op_.handle));
    return props_;
}
std::string PermutationRNG_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<PermutationRNG>();
    static_cast<void>(op_);
    return "PermutationRNG";
}
} // anonymous namespace
OP_TRAIT_REG(PermutationRNG, PermutationRNG)
    .hash(PermutationRNG_hash_impl)
    .is_same_st(PermutationRNG_is_same_st_impl)
    .props(PermutationRNG_props_impl)
    .make_name(PermutationRNG_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(PixelShuffle);

namespace {
size_t PixelShuffle_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<PixelShuffle>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.factor));
    return val;
}
bool PixelShuffle_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<PixelShuffle>(),
         &&b_ = rhs_.cast_final_safe<PixelShuffle>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.factor != b_.factor) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> PixelShuffle_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<PixelShuffle>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("factor", std::to_string(op_.factor));
    return props_;
}
std::string PixelShuffle_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<PixelShuffle>();
    static_cast<void>(op_);
    return "PixelShuffle";
}
} // anonymous namespace
OP_TRAIT_REG(PixelShuffle, PixelShuffle)
    .hash(PixelShuffle_hash_impl)
    .is_same_st(PixelShuffle_is_same_st_impl)
    .props(PixelShuffle_props_impl)
    .make_name(PixelShuffle_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(PixelShuffleBackward);

namespace {
size_t PixelShuffleBackward_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<PixelShuffleBackward>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.factor));
    return val;
}
bool PixelShuffleBackward_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<PixelShuffleBackward>(),
         &&b_ = rhs_.cast_final_safe<PixelShuffleBackward>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.factor != b_.factor) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> PixelShuffleBackward_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<PixelShuffleBackward>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("factor", std::to_string(op_.factor));
    return props_;
}
std::string PixelShuffleBackward_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<PixelShuffleBackward>();
    static_cast<void>(op_);
    return "PixelShuffleBackward";
}
} // anonymous namespace
OP_TRAIT_REG(PixelShuffleBackward, PixelShuffleBackward)
    .hash(PixelShuffleBackward_hash_impl)
    .is_same_st(PixelShuffleBackward_is_same_st_impl)
    .props(PixelShuffleBackward_props_impl)
    .make_name(PixelShuffleBackward_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(PoissonRNG);

namespace {
size_t PoissonRNG_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<PoissonRNG>();
    static_cast<void>(op_);

    return mgb::hash_pair_combine(
      mgb::hash(op_.dyn_typeinfo()),
      mgb::hash(op_.handle)
      );
  }
bool PoissonRNG_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<PoissonRNG>(),
         &&b_ = rhs_.cast_final_safe<PoissonRNG>();
    static_cast<void>(a_);
    static_cast<void>(b_);
return a_.handle == b_.handle;}
std::vector<std::pair<const char*, std::string>> PoissonRNG_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<PoissonRNG>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("seed", std::to_string(op_.seed));
    props_.emplace_back("handle", std::to_string(op_.handle));
    return props_;
}
std::string PoissonRNG_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<PoissonRNG>();
    static_cast<void>(op_);
    return "PoissonRNG";
}
} // anonymous namespace
OP_TRAIT_REG(PoissonRNG, PoissonRNG)
    .hash(PoissonRNG_hash_impl)
    .is_same_st(PoissonRNG_is_same_st_impl)
    .props(PoissonRNG_props_impl)
    .make_name(PoissonRNG_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Pooling);

namespace {
size_t Pooling_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Pooling>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.mode));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pad_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pad_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.window_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.window_w));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.format));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.strategy));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.workspace_limit));
    return val;
}
bool Pooling_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Pooling>(),
         &&b_ = rhs_.cast_final_safe<Pooling>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.mode != b_.mode) return false;
    if (a_.pad_h != b_.pad_h) return false;
    if (a_.pad_w != b_.pad_w) return false;
    if (a_.stride_h != b_.stride_h) return false;
    if (a_.stride_w != b_.stride_w) return false;
    if (a_.window_h != b_.window_h) return false;
    if (a_.window_w != b_.window_w) return false;
    if (a_.format != b_.format) return false;
    if (a_.strategy != b_.strategy) return false;
    if (a_.workspace_limit != b_.workspace_limit) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Pooling_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Pooling>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.mode){
    case Pooling::Mode::MAX:
        props_.emplace_back("mode", "MAX");
        break;
    case Pooling::Mode::AVERAGE:
        props_.emplace_back("mode", "AVERAGE");
        break;
    case Pooling::Mode::AVERAGE_COUNT_EXCLUDE_PADDING:
        props_.emplace_back("mode", "AVERAGE_COUNT_EXCLUDE_PADDING");
        break;
    default:
        props_.emplace_back("mode", "INVALID");
        break;
    }
    props_.emplace_back("pad_h", std::to_string(op_.pad_h));
    props_.emplace_back("pad_w", std::to_string(op_.pad_w));
    props_.emplace_back("stride_h", std::to_string(op_.stride_h));
    props_.emplace_back("stride_w", std::to_string(op_.stride_w));
    props_.emplace_back("window_h", std::to_string(op_.window_h));
    props_.emplace_back("window_w", std::to_string(op_.window_w));
    switch (op_.format){
    case Pooling::Format::NCHW:
        props_.emplace_back("format", "NCHW");
        break;
    case Pooling::Format::NHWC:
        props_.emplace_back("format", "NHWC");
        break;
    case Pooling::Format::NHWCD4:
        props_.emplace_back("format", "NHWCD4");
        break;
    case Pooling::Format::NCHW4:
        props_.emplace_back("format", "NCHW4");
        break;
    case Pooling::Format::NCHW8:
        props_.emplace_back("format", "NCHW8");
        break;
    case Pooling::Format::NCHW32:
        props_.emplace_back("format", "NCHW32");
        break;
    case Pooling::Format::NCHW88:
        props_.emplace_back("format", "NCHW88");
        break;
    case Pooling::Format::NCHW44:
        props_.emplace_back("format", "NCHW44");
        break;
    case Pooling::Format::NCHW44_DOT:
        props_.emplace_back("format", "NCHW44_DOT");
        break;
    case Pooling::Format::NCHW4_NCHW32:
        props_.emplace_back("format", "NCHW4_NCHW32");
        break;
    case Pooling::Format::NCHW32_NCHW4:
        props_.emplace_back("format", "NCHW32_NCHW4");
        break;
    case Pooling::Format::NCHW4_NCHW:
        props_.emplace_back("format", "NCHW4_NCHW");
        break;
    case Pooling::Format::NHWC_NCHW:
        props_.emplace_back("format", "NHWC_NCHW");
        break;
    case Pooling::Format::NHWC_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NHWC_NCHW4_IC_SMALL");
        break;
    case Pooling::Format::NCHW_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NCHW_NCHW4_IC_SMALL");
        break;
    case Pooling::Format::CHWN4:
        props_.emplace_back("format", "CHWN4");
        break;
    case Pooling::Format::NCHW64:
        props_.emplace_back("format", "NCHW64");
        break;
    case Pooling::Format::NCHW4_NHWC:
        props_.emplace_back("format", "NCHW4_NHWC");
        break;
    default:
        props_.emplace_back("format", "INVALID");
        break;
    }
    switch (op_.strategy){
    case Pooling::Strategy::HEURISTIC:
        props_.emplace_back("strategy", "HEURISTIC");
        break;
    case Pooling::Strategy::PROFILE:
        props_.emplace_back("strategy", "PROFILE");
        break;
    case Pooling::Strategy::REPRODUCIBLE:
        props_.emplace_back("strategy", "REPRODUCIBLE");
        break;
    case Pooling::Strategy::OPTIMIZED:
        props_.emplace_back("strategy", "OPTIMIZED");
        break;
    default:
        props_.emplace_back("strategy", "INVALID");
        break;
    }
    props_.emplace_back("workspace_limit", std::to_string(op_.workspace_limit));
    return props_;
}
std::string Pooling_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Pooling>();
    static_cast<void>(op_);
    return "Pooling";
}
} // anonymous namespace
OP_TRAIT_REG(Pooling, Pooling)
    .hash(Pooling_hash_impl)
    .is_same_st(Pooling_is_same_st_impl)
    .props(Pooling_props_impl)
    .make_name(Pooling_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(RNN);

namespace {
size_t RNN_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<RNN>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.num_layers));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.bidirectional));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.bias));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.hidden_size));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dropout));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.nonlineMode));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.fwd_mode));
    return val;
}
bool RNN_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<RNN>(),
         &&b_ = rhs_.cast_final_safe<RNN>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.num_layers != b_.num_layers) return false;
    if (a_.bidirectional != b_.bidirectional) return false;
    if (a_.bias != b_.bias) return false;
    if (a_.hidden_size != b_.hidden_size) return false;
    if (a_.dropout != b_.dropout) return false;
    if (a_.nonlineMode != b_.nonlineMode) return false;
    if (a_.fwd_mode != b_.fwd_mode) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> RNN_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<RNN>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("num_layers", std::to_string(op_.num_layers));
    props_.emplace_back("bidirectional", std::to_string(op_.bidirectional));
    props_.emplace_back("bias", std::to_string(op_.bias));
    props_.emplace_back("hidden_size", std::to_string(op_.hidden_size));
    props_.emplace_back("dropout", std::to_string(op_.dropout));
    switch (op_.nonlineMode){
    case RNN::NonlineMode::IDENTITY:
        props_.emplace_back("nonlineMode", "IDENTITY");
        break;
    case RNN::NonlineMode::RELU:
        props_.emplace_back("nonlineMode", "RELU");
        break;
    case RNN::NonlineMode::TANH:
        props_.emplace_back("nonlineMode", "TANH");
        break;
    default:
        props_.emplace_back("nonlineMode", "INVALID");
        break;
    }
    switch (op_.fwd_mode){
    case RNN::FwdMode::TRAINING:
        props_.emplace_back("fwd_mode", "TRAINING");
        break;
    case RNN::FwdMode::INFERENCE:
        props_.emplace_back("fwd_mode", "INFERENCE");
        break;
    default:
        props_.emplace_back("fwd_mode", "INVALID");
        break;
    }
    return props_;
}
std::string RNN_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<RNN>();
    static_cast<void>(op_);
    return "RNN";
}
} // anonymous namespace
OP_TRAIT_REG(RNN, RNN)
    .hash(RNN_hash_impl)
    .is_same_st(RNN_is_same_st_impl)
    .props(RNN_props_impl)
    .make_name(RNN_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(RNNCell);

namespace {
size_t RNNCell_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<RNNCell>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.nonlineMode));
    return val;
}
bool RNNCell_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<RNNCell>(),
         &&b_ = rhs_.cast_final_safe<RNNCell>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.nonlineMode != b_.nonlineMode) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> RNNCell_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<RNNCell>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.nonlineMode){
    case RNNCell::NonlineMode::IDENTITY:
        props_.emplace_back("nonlineMode", "IDENTITY");
        break;
    case RNNCell::NonlineMode::RELU:
        props_.emplace_back("nonlineMode", "RELU");
        break;
    case RNNCell::NonlineMode::TANH:
        props_.emplace_back("nonlineMode", "TANH");
        break;
    default:
        props_.emplace_back("nonlineMode", "INVALID");
        break;
    }
    return props_;
}
std::string RNNCell_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<RNNCell>();
    static_cast<void>(op_);
    return "RNNCell";
}
} // anonymous namespace
OP_TRAIT_REG(RNNCell, RNNCell)
    .hash(RNNCell_hash_impl)
    .is_same_st(RNNCell_is_same_st_impl)
    .props(RNNCell_props_impl)
    .make_name(RNNCell_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(ROIAlign);

namespace {
size_t ROIAlign_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<ROIAlign>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.mode));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.format));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.spatial_scale));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.offset));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pooled_height));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pooled_width));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.sample_height));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.sample_width));
    return val;
}
bool ROIAlign_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<ROIAlign>(),
         &&b_ = rhs_.cast_final_safe<ROIAlign>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.mode != b_.mode) return false;
    if (a_.format != b_.format) return false;
    if (a_.spatial_scale != b_.spatial_scale) return false;
    if (a_.offset != b_.offset) return false;
    if (a_.pooled_height != b_.pooled_height) return false;
    if (a_.pooled_width != b_.pooled_width) return false;
    if (a_.sample_height != b_.sample_height) return false;
    if (a_.sample_width != b_.sample_width) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> ROIAlign_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<ROIAlign>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.mode){
    case ROIAlign::Mode::MAX:
        props_.emplace_back("mode", "MAX");
        break;
    case ROIAlign::Mode::AVERAGE:
        props_.emplace_back("mode", "AVERAGE");
        break;
    default:
        props_.emplace_back("mode", "INVALID");
        break;
    }
    switch (op_.format){
    case ROIAlign::Format::NCHW:
        props_.emplace_back("format", "NCHW");
        break;
    case ROIAlign::Format::NHWC:
        props_.emplace_back("format", "NHWC");
        break;
    case ROIAlign::Format::NHWCD4:
        props_.emplace_back("format", "NHWCD4");
        break;
    case ROIAlign::Format::NCHW4:
        props_.emplace_back("format", "NCHW4");
        break;
    case ROIAlign::Format::NCHW8:
        props_.emplace_back("format", "NCHW8");
        break;
    case ROIAlign::Format::NCHW32:
        props_.emplace_back("format", "NCHW32");
        break;
    case ROIAlign::Format::NCHW88:
        props_.emplace_back("format", "NCHW88");
        break;
    case ROIAlign::Format::NCHW44:
        props_.emplace_back("format", "NCHW44");
        break;
    case ROIAlign::Format::NCHW44_DOT:
        props_.emplace_back("format", "NCHW44_DOT");
        break;
    case ROIAlign::Format::NCHW4_NCHW32:
        props_.emplace_back("format", "NCHW4_NCHW32");
        break;
    case ROIAlign::Format::NCHW32_NCHW4:
        props_.emplace_back("format", "NCHW32_NCHW4");
        break;
    case ROIAlign::Format::NCHW4_NCHW:
        props_.emplace_back("format", "NCHW4_NCHW");
        break;
    case ROIAlign::Format::NHWC_NCHW:
        props_.emplace_back("format", "NHWC_NCHW");
        break;
    case ROIAlign::Format::NHWC_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NHWC_NCHW4_IC_SMALL");
        break;
    case ROIAlign::Format::NCHW_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NCHW_NCHW4_IC_SMALL");
        break;
    case ROIAlign::Format::CHWN4:
        props_.emplace_back("format", "CHWN4");
        break;
    case ROIAlign::Format::NCHW64:
        props_.emplace_back("format", "NCHW64");
        break;
    case ROIAlign::Format::NCHW4_NHWC:
        props_.emplace_back("format", "NCHW4_NHWC");
        break;
    default:
        props_.emplace_back("format", "INVALID");
        break;
    }
    props_.emplace_back("spatial_scale", std::to_string(op_.spatial_scale));
    props_.emplace_back("offset", std::to_string(op_.offset));
    props_.emplace_back("pooled_height", std::to_string(op_.pooled_height));
    props_.emplace_back("pooled_width", std::to_string(op_.pooled_width));
    props_.emplace_back("sample_height", std::to_string(op_.sample_height));
    props_.emplace_back("sample_width", std::to_string(op_.sample_width));
    return props_;
}
std::string ROIAlign_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<ROIAlign>();
    static_cast<void>(op_);
    return "ROIAlign";
}
} // anonymous namespace
OP_TRAIT_REG(ROIAlign, ROIAlign)
    .hash(ROIAlign_hash_impl)
    .is_same_st(ROIAlign_is_same_st_impl)
    .props(ROIAlign_props_impl)
    .make_name(ROIAlign_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(ROIPooling);

namespace {
size_t ROIPooling_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<ROIPooling>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.mode));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.scale));
    return val;
}
bool ROIPooling_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<ROIPooling>(),
         &&b_ = rhs_.cast_final_safe<ROIPooling>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.mode != b_.mode) return false;
    if (a_.scale != b_.scale) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> ROIPooling_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<ROIPooling>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.mode){
    case ROIPooling::Mode::MAX:
        props_.emplace_back("mode", "MAX");
        break;
    case ROIPooling::Mode::AVERAGE:
        props_.emplace_back("mode", "AVERAGE");
        break;
    default:
        props_.emplace_back("mode", "INVALID");
        break;
    }
    props_.emplace_back("scale", std::to_string(op_.scale));
    return props_;
}
std::string ROIPooling_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<ROIPooling>();
    static_cast<void>(op_);
    return "ROIPooling";
}
} // anonymous namespace
OP_TRAIT_REG(ROIPooling, ROIPooling)
    .hash(ROIPooling_hash_impl)
    .is_same_st(ROIPooling_is_same_st_impl)
    .props(ROIPooling_props_impl)
    .make_name(ROIPooling_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Reduce);

namespace {
size_t Reduce_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Reduce>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.mode));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.axis));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.data_type));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.keepdim));
    return val;
}
bool Reduce_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Reduce>(),
         &&b_ = rhs_.cast_final_safe<Reduce>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.mode != b_.mode) return false;
    if (a_.axis != b_.axis) return false;
    if (a_.data_type != b_.data_type) return false;
    if (a_.keepdim != b_.keepdim) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Reduce_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Reduce>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.mode){
    case Reduce::Mode::SUM:
        props_.emplace_back("mode", "SUM");
        break;
    case Reduce::Mode::SUM_SQR:
        props_.emplace_back("mode", "SUM_SQR");
        break;
    case Reduce::Mode::PRODUCT:
        props_.emplace_back("mode", "PRODUCT");
        break;
    case Reduce::Mode::MIN:
        props_.emplace_back("mode", "MIN");
        break;
    case Reduce::Mode::MAX:
        props_.emplace_back("mode", "MAX");
        break;
    case Reduce::Mode::MEAN:
        props_.emplace_back("mode", "MEAN");
        break;
    default:
        props_.emplace_back("mode", "INVALID");
        break;
    }
    props_.emplace_back("axis", std::to_string(op_.axis));
    switch (op_.data_type){
    case Reduce::DataType::DEFAULT:
        props_.emplace_back("data_type", "DEFAULT");
        break;
    case Reduce::DataType::FLOAT_IO16xC32:
        props_.emplace_back("data_type", "FLOAT_IO16xC32");
        break;
    case Reduce::DataType::FLOAT_O32xC32:
        props_.emplace_back("data_type", "FLOAT_O32xC32");
        break;
    case Reduce::DataType::FLOAT_O16xC32:
        props_.emplace_back("data_type", "FLOAT_O16xC32");
        break;
    case Reduce::DataType::QUINT_I8xO32:
        props_.emplace_back("data_type", "QUINT_I8xO32");
        break;
    case Reduce::DataType::QINT_I8xO32:
        props_.emplace_back("data_type", "QINT_I8xO32");
        break;
    default:
        props_.emplace_back("data_type", "INVALID");
        break;
    }
    props_.emplace_back("keepdim", std::to_string(op_.keepdim));
    return props_;
}
std::string Reduce_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Reduce>();
    static_cast<void>(op_);
    return "Reduce";
}
} // anonymous namespace
OP_TRAIT_REG(Reduce, Reduce)
    .hash(Reduce_hash_impl)
    .is_same_st(Reduce_is_same_st_impl)
    .props(Reduce_props_impl)
    .make_name(Reduce_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(RegionRestrictedConvolution);

namespace {
size_t RegionRestrictedConvolution_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<RegionRestrictedConvolution>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.mode));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pad_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pad_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dilate_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dilate_w));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.sparse));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.format));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.compute_mode));
    return val;
}
bool RegionRestrictedConvolution_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<RegionRestrictedConvolution>(),
         &&b_ = rhs_.cast_final_safe<RegionRestrictedConvolution>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.mode != b_.mode) return false;
    if (a_.pad_h != b_.pad_h) return false;
    if (a_.pad_w != b_.pad_w) return false;
    if (a_.stride_h != b_.stride_h) return false;
    if (a_.stride_w != b_.stride_w) return false;
    if (a_.dilate_h != b_.dilate_h) return false;
    if (a_.dilate_w != b_.dilate_w) return false;
    if (a_.sparse != b_.sparse) return false;
    if (a_.format != b_.format) return false;
    if (a_.compute_mode != b_.compute_mode) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> RegionRestrictedConvolution_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<RegionRestrictedConvolution>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.mode){
    case RegionRestrictedConvolution::Mode::CROSS_CORRELATION:
        props_.emplace_back("mode", "CROSS_CORRELATION");
        break;
    case RegionRestrictedConvolution::Mode::CONVOLUTION:
        props_.emplace_back("mode", "CONVOLUTION");
        break;
    default:
        props_.emplace_back("mode", "INVALID");
        break;
    }
    props_.emplace_back("pad_h", std::to_string(op_.pad_h));
    props_.emplace_back("pad_w", std::to_string(op_.pad_w));
    props_.emplace_back("stride_h", std::to_string(op_.stride_h));
    props_.emplace_back("stride_w", std::to_string(op_.stride_w));
    props_.emplace_back("dilate_h", std::to_string(op_.dilate_h));
    props_.emplace_back("dilate_w", std::to_string(op_.dilate_w));
    switch (op_.sparse){
    case RegionRestrictedConvolution::Sparse::DENSE:
        props_.emplace_back("sparse", "DENSE");
        break;
    case RegionRestrictedConvolution::Sparse::GROUP:
        props_.emplace_back("sparse", "GROUP");
        break;
    default:
        props_.emplace_back("sparse", "INVALID");
        break;
    }
    switch (op_.format){
    case RegionRestrictedConvolution::Format::NCHW:
        props_.emplace_back("format", "NCHW");
        break;
    case RegionRestrictedConvolution::Format::NHWC:
        props_.emplace_back("format", "NHWC");
        break;
    case RegionRestrictedConvolution::Format::NHWCD4:
        props_.emplace_back("format", "NHWCD4");
        break;
    case RegionRestrictedConvolution::Format::NCHW4:
        props_.emplace_back("format", "NCHW4");
        break;
    case RegionRestrictedConvolution::Format::NCHW8:
        props_.emplace_back("format", "NCHW8");
        break;
    case RegionRestrictedConvolution::Format::NCHW32:
        props_.emplace_back("format", "NCHW32");
        break;
    case RegionRestrictedConvolution::Format::NCHW88:
        props_.emplace_back("format", "NCHW88");
        break;
    case RegionRestrictedConvolution::Format::NCHW44:
        props_.emplace_back("format", "NCHW44");
        break;
    case RegionRestrictedConvolution::Format::NCHW44_DOT:
        props_.emplace_back("format", "NCHW44_DOT");
        break;
    case RegionRestrictedConvolution::Format::NCHW4_NCHW32:
        props_.emplace_back("format", "NCHW4_NCHW32");
        break;
    case RegionRestrictedConvolution::Format::NCHW32_NCHW4:
        props_.emplace_back("format", "NCHW32_NCHW4");
        break;
    case RegionRestrictedConvolution::Format::NCHW4_NCHW:
        props_.emplace_back("format", "NCHW4_NCHW");
        break;
    case RegionRestrictedConvolution::Format::NHWC_NCHW:
        props_.emplace_back("format", "NHWC_NCHW");
        break;
    case RegionRestrictedConvolution::Format::NHWC_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NHWC_NCHW4_IC_SMALL");
        break;
    case RegionRestrictedConvolution::Format::NCHW_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NCHW_NCHW4_IC_SMALL");
        break;
    case RegionRestrictedConvolution::Format::CHWN4:
        props_.emplace_back("format", "CHWN4");
        break;
    case RegionRestrictedConvolution::Format::NCHW64:
        props_.emplace_back("format", "NCHW64");
        break;
    case RegionRestrictedConvolution::Format::NCHW4_NHWC:
        props_.emplace_back("format", "NCHW4_NHWC");
        break;
    default:
        props_.emplace_back("format", "INVALID");
        break;
    }
    switch (op_.compute_mode){
    case RegionRestrictedConvolution::ComputeMode::DEFAULT:
        props_.emplace_back("compute_mode", "DEFAULT");
        break;
    case RegionRestrictedConvolution::ComputeMode::FLOAT32:
        props_.emplace_back("compute_mode", "FLOAT32");
        break;
    default:
        props_.emplace_back("compute_mode", "INVALID");
        break;
    }
    return props_;
}
std::string RegionRestrictedConvolution_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<RegionRestrictedConvolution>();
    static_cast<void>(op_);
    return "RegionRestrictedConvolution";
}
} // anonymous namespace
OP_TRAIT_REG(RegionRestrictedConvolution, RegionRestrictedConvolution)
    .hash(RegionRestrictedConvolution_hash_impl)
    .is_same_st(RegionRestrictedConvolution_is_same_st_impl)
    .props(RegionRestrictedConvolution_props_impl)
    .make_name(RegionRestrictedConvolution_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(RegionRestrictedConvolutionBackwardData);

namespace {
size_t RegionRestrictedConvolutionBackwardData_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<RegionRestrictedConvolutionBackwardData>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.mode));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pad_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pad_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dilate_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dilate_w));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.sparse));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.format));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.compute_mode));
    return val;
}
bool RegionRestrictedConvolutionBackwardData_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<RegionRestrictedConvolutionBackwardData>(),
         &&b_ = rhs_.cast_final_safe<RegionRestrictedConvolutionBackwardData>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.mode != b_.mode) return false;
    if (a_.pad_h != b_.pad_h) return false;
    if (a_.pad_w != b_.pad_w) return false;
    if (a_.stride_h != b_.stride_h) return false;
    if (a_.stride_w != b_.stride_w) return false;
    if (a_.dilate_h != b_.dilate_h) return false;
    if (a_.dilate_w != b_.dilate_w) return false;
    if (a_.sparse != b_.sparse) return false;
    if (a_.format != b_.format) return false;
    if (a_.compute_mode != b_.compute_mode) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> RegionRestrictedConvolutionBackwardData_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<RegionRestrictedConvolutionBackwardData>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.mode){
    case RegionRestrictedConvolutionBackwardData::Mode::CROSS_CORRELATION:
        props_.emplace_back("mode", "CROSS_CORRELATION");
        break;
    case RegionRestrictedConvolutionBackwardData::Mode::CONVOLUTION:
        props_.emplace_back("mode", "CONVOLUTION");
        break;
    default:
        props_.emplace_back("mode", "INVALID");
        break;
    }
    props_.emplace_back("pad_h", std::to_string(op_.pad_h));
    props_.emplace_back("pad_w", std::to_string(op_.pad_w));
    props_.emplace_back("stride_h", std::to_string(op_.stride_h));
    props_.emplace_back("stride_w", std::to_string(op_.stride_w));
    props_.emplace_back("dilate_h", std::to_string(op_.dilate_h));
    props_.emplace_back("dilate_w", std::to_string(op_.dilate_w));
    switch (op_.sparse){
    case RegionRestrictedConvolutionBackwardData::Sparse::DENSE:
        props_.emplace_back("sparse", "DENSE");
        break;
    case RegionRestrictedConvolutionBackwardData::Sparse::GROUP:
        props_.emplace_back("sparse", "GROUP");
        break;
    default:
        props_.emplace_back("sparse", "INVALID");
        break;
    }
    switch (op_.format){
    case RegionRestrictedConvolutionBackwardData::Format::NCHW:
        props_.emplace_back("format", "NCHW");
        break;
    case RegionRestrictedConvolutionBackwardData::Format::NHWC:
        props_.emplace_back("format", "NHWC");
        break;
    case RegionRestrictedConvolutionBackwardData::Format::NHWCD4:
        props_.emplace_back("format", "NHWCD4");
        break;
    case RegionRestrictedConvolutionBackwardData::Format::NCHW4:
        props_.emplace_back("format", "NCHW4");
        break;
    case RegionRestrictedConvolutionBackwardData::Format::NCHW8:
        props_.emplace_back("format", "NCHW8");
        break;
    case RegionRestrictedConvolutionBackwardData::Format::NCHW32:
        props_.emplace_back("format", "NCHW32");
        break;
    case RegionRestrictedConvolutionBackwardData::Format::NCHW88:
        props_.emplace_back("format", "NCHW88");
        break;
    case RegionRestrictedConvolutionBackwardData::Format::NCHW44:
        props_.emplace_back("format", "NCHW44");
        break;
    case RegionRestrictedConvolutionBackwardData::Format::NCHW44_DOT:
        props_.emplace_back("format", "NCHW44_DOT");
        break;
    case RegionRestrictedConvolutionBackwardData::Format::NCHW4_NCHW32:
        props_.emplace_back("format", "NCHW4_NCHW32");
        break;
    case RegionRestrictedConvolutionBackwardData::Format::NCHW32_NCHW4:
        props_.emplace_back("format", "NCHW32_NCHW4");
        break;
    case RegionRestrictedConvolutionBackwardData::Format::NCHW4_NCHW:
        props_.emplace_back("format", "NCHW4_NCHW");
        break;
    case RegionRestrictedConvolutionBackwardData::Format::NHWC_NCHW:
        props_.emplace_back("format", "NHWC_NCHW");
        break;
    case RegionRestrictedConvolutionBackwardData::Format::NHWC_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NHWC_NCHW4_IC_SMALL");
        break;
    case RegionRestrictedConvolutionBackwardData::Format::NCHW_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NCHW_NCHW4_IC_SMALL");
        break;
    case RegionRestrictedConvolutionBackwardData::Format::CHWN4:
        props_.emplace_back("format", "CHWN4");
        break;
    case RegionRestrictedConvolutionBackwardData::Format::NCHW64:
        props_.emplace_back("format", "NCHW64");
        break;
    case RegionRestrictedConvolutionBackwardData::Format::NCHW4_NHWC:
        props_.emplace_back("format", "NCHW4_NHWC");
        break;
    default:
        props_.emplace_back("format", "INVALID");
        break;
    }
    switch (op_.compute_mode){
    case RegionRestrictedConvolutionBackwardData::ComputeMode::DEFAULT:
        props_.emplace_back("compute_mode", "DEFAULT");
        break;
    case RegionRestrictedConvolutionBackwardData::ComputeMode::FLOAT32:
        props_.emplace_back("compute_mode", "FLOAT32");
        break;
    default:
        props_.emplace_back("compute_mode", "INVALID");
        break;
    }
    return props_;
}
std::string RegionRestrictedConvolutionBackwardData_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<RegionRestrictedConvolutionBackwardData>();
    static_cast<void>(op_);
    return "RegionRestrictedConvolutionBackwardData";
}
} // anonymous namespace
OP_TRAIT_REG(RegionRestrictedConvolutionBackwardData, RegionRestrictedConvolutionBackwardData)
    .hash(RegionRestrictedConvolutionBackwardData_hash_impl)
    .is_same_st(RegionRestrictedConvolutionBackwardData_is_same_st_impl)
    .props(RegionRestrictedConvolutionBackwardData_props_impl)
    .make_name(RegionRestrictedConvolutionBackwardData_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Remap);

namespace {
size_t Remap_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Remap>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.imode));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.border_type));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.format));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.scalar));
    return val;
}
bool Remap_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Remap>(),
         &&b_ = rhs_.cast_final_safe<Remap>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.imode != b_.imode) return false;
    if (a_.border_type != b_.border_type) return false;
    if (a_.format != b_.format) return false;
    if (a_.scalar != b_.scalar) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Remap_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Remap>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.imode){
    case Remap::InterpolationMode::NEAREST:
        props_.emplace_back("imode", "NEAREST");
        break;
    case Remap::InterpolationMode::LINEAR:
        props_.emplace_back("imode", "LINEAR");
        break;
    case Remap::InterpolationMode::AREA:
        props_.emplace_back("imode", "AREA");
        break;
    case Remap::InterpolationMode::CUBIC:
        props_.emplace_back("imode", "CUBIC");
        break;
    case Remap::InterpolationMode::LANCZOS4:
        props_.emplace_back("imode", "LANCZOS4");
        break;
    default:
        props_.emplace_back("imode", "INVALID");
        break;
    }
    switch (op_.border_type){
    case Remap::BorderMode::REPLICATE:
        props_.emplace_back("border_type", "REPLICATE");
        break;
    case Remap::BorderMode::REFLECT:
        props_.emplace_back("border_type", "REFLECT");
        break;
    case Remap::BorderMode::REFLECT_101:
        props_.emplace_back("border_type", "REFLECT_101");
        break;
    case Remap::BorderMode::WRAP:
        props_.emplace_back("border_type", "WRAP");
        break;
    case Remap::BorderMode::CONSTANT:
        props_.emplace_back("border_type", "CONSTANT");
        break;
    case Remap::BorderMode::TRANSPARENT:
        props_.emplace_back("border_type", "TRANSPARENT");
        break;
    case Remap::BorderMode::ISOLATED:
        props_.emplace_back("border_type", "ISOLATED");
        break;
    default:
        props_.emplace_back("border_type", "INVALID");
        break;
    }
    switch (op_.format){
    case Remap::Format::NCHW:
        props_.emplace_back("format", "NCHW");
        break;
    case Remap::Format::NHWC:
        props_.emplace_back("format", "NHWC");
        break;
    case Remap::Format::NHWCD4:
        props_.emplace_back("format", "NHWCD4");
        break;
    case Remap::Format::NCHW4:
        props_.emplace_back("format", "NCHW4");
        break;
    case Remap::Format::NCHW8:
        props_.emplace_back("format", "NCHW8");
        break;
    case Remap::Format::NCHW32:
        props_.emplace_back("format", "NCHW32");
        break;
    case Remap::Format::NCHW88:
        props_.emplace_back("format", "NCHW88");
        break;
    case Remap::Format::NCHW44:
        props_.emplace_back("format", "NCHW44");
        break;
    case Remap::Format::NCHW44_DOT:
        props_.emplace_back("format", "NCHW44_DOT");
        break;
    case Remap::Format::NCHW4_NCHW32:
        props_.emplace_back("format", "NCHW4_NCHW32");
        break;
    case Remap::Format::NCHW32_NCHW4:
        props_.emplace_back("format", "NCHW32_NCHW4");
        break;
    case Remap::Format::NCHW4_NCHW:
        props_.emplace_back("format", "NCHW4_NCHW");
        break;
    case Remap::Format::NHWC_NCHW:
        props_.emplace_back("format", "NHWC_NCHW");
        break;
    case Remap::Format::NHWC_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NHWC_NCHW4_IC_SMALL");
        break;
    case Remap::Format::NCHW_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NCHW_NCHW4_IC_SMALL");
        break;
    case Remap::Format::CHWN4:
        props_.emplace_back("format", "CHWN4");
        break;
    case Remap::Format::NCHW64:
        props_.emplace_back("format", "NCHW64");
        break;
    case Remap::Format::NCHW4_NHWC:
        props_.emplace_back("format", "NCHW4_NHWC");
        break;
    default:
        props_.emplace_back("format", "INVALID");
        break;
    }
    props_.emplace_back("scalar", std::to_string(op_.scalar));
    return props_;
}
std::string Remap_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Remap>();
    static_cast<void>(op_);
    return "Remap";
}
} // anonymous namespace
OP_TRAIT_REG(Remap, Remap)
    .hash(Remap_hash_impl)
    .is_same_st(Remap_is_same_st_impl)
    .props(Remap_props_impl)
    .make_name(Remap_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(RemoteRecv);

namespace {
size_t RemoteRecv_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<RemoteRecv>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.key));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.addr));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.port));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.rank_from));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.cn));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.shape));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dtype.handle()));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.backend));
    return val;
}
bool RemoteRecv_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<RemoteRecv>(),
         &&b_ = rhs_.cast_final_safe<RemoteRecv>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.key != b_.key) return false;
    if (a_.addr != b_.addr) return false;
    if (a_.port != b_.port) return false;
    if (a_.rank_from != b_.rank_from) return false;
    if (a_.cn != b_.cn) return false;
    if (a_.shape != b_.shape) return false;
    if (a_.dtype != b_.dtype) return false;
    if (a_.backend != b_.backend) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> RemoteRecv_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<RemoteRecv>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("key", op_.key);
    props_.emplace_back("addr", op_.addr);
    props_.emplace_back("port", std::to_string(op_.port));
    props_.emplace_back("rank_from", std::to_string(op_.rank_from));
    props_.emplace_back("cn", op_.cn.to_string());
    props_.emplace_back("shape", "{std::vector}");
    props_.emplace_back("dtype", op_.dtype.name());
    props_.emplace_back("backend", op_.backend);
    return props_;
}
std::string RemoteRecv_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<RemoteRecv>();
    static_cast<void>(op_);
    return "RemoteRecv";
}
} // anonymous namespace
OP_TRAIT_REG(RemoteRecv, RemoteRecv)
    .hash(RemoteRecv_hash_impl)
    .is_same_st(RemoteRecv_is_same_st_impl)
    .props(RemoteRecv_props_impl)
    .make_name(RemoteRecv_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(RemoteSend);

namespace {
size_t RemoteSend_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<RemoteSend>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.key));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.addr));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.port));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.rank_to));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.backend));
    return val;
}
bool RemoteSend_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<RemoteSend>(),
         &&b_ = rhs_.cast_final_safe<RemoteSend>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.key != b_.key) return false;
    if (a_.addr != b_.addr) return false;
    if (a_.port != b_.port) return false;
    if (a_.rank_to != b_.rank_to) return false;
    if (a_.backend != b_.backend) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> RemoteSend_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<RemoteSend>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("key", op_.key);
    props_.emplace_back("addr", op_.addr);
    props_.emplace_back("port", std::to_string(op_.port));
    props_.emplace_back("rank_to", std::to_string(op_.rank_to));
    props_.emplace_back("backend", op_.backend);
    return props_;
}
std::string RemoteSend_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<RemoteSend>();
    static_cast<void>(op_);
    return "RemoteSend";
}
} // anonymous namespace
OP_TRAIT_REG(RemoteSend, RemoteSend)
    .hash(RemoteSend_hash_impl)
    .is_same_st(RemoteSend_is_same_st_impl)
    .props(RemoteSend_props_impl)
    .make_name(RemoteSend_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(RemoveAxis);

namespace {
size_t RemoveAxis_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<RemoveAxis>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.axis));
    return val;
}
bool RemoveAxis_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<RemoveAxis>(),
         &&b_ = rhs_.cast_final_safe<RemoveAxis>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.axis != b_.axis) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> RemoveAxis_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<RemoveAxis>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("axis", "{std::vector}");
    return props_;
}
std::string RemoveAxis_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<RemoveAxis>();
    static_cast<void>(op_);
    return "RemoveAxis";
}
} // anonymous namespace
OP_TRAIT_REG(RemoveAxis, RemoveAxis)
    .hash(RemoveAxis_hash_impl)
    .is_same_st(RemoveAxis_is_same_st_impl)
    .props(RemoveAxis_props_impl)
    .make_name(RemoveAxis_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Reshape);

namespace {
size_t Reshape_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Reshape>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.axis));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.shape));
    return val;
}
bool Reshape_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Reshape>(),
         &&b_ = rhs_.cast_final_safe<Reshape>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.axis != b_.axis) return false;
    if (a_.shape != b_.shape) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Reshape_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Reshape>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("axis", std::to_string(op_.axis));
    props_.emplace_back("shape", "{std::vector}");
    return props_;
}
std::string Reshape_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Reshape>();
    static_cast<void>(op_);
    return "Reshape";
}
} // anonymous namespace
OP_TRAIT_REG(Reshape, Reshape)
    .hash(Reshape_hash_impl)
    .is_same_st(Reshape_is_same_st_impl)
    .props(Reshape_props_impl)
    .make_name(Reshape_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Resize);

namespace {
size_t Resize_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Resize>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.imode));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.format));
    return val;
}
bool Resize_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Resize>(),
         &&b_ = rhs_.cast_final_safe<Resize>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.imode != b_.imode) return false;
    if (a_.format != b_.format) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Resize_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Resize>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.imode){
    case Resize::InterpolationMode::NEAREST:
        props_.emplace_back("imode", "NEAREST");
        break;
    case Resize::InterpolationMode::LINEAR:
        props_.emplace_back("imode", "LINEAR");
        break;
    case Resize::InterpolationMode::AREA:
        props_.emplace_back("imode", "AREA");
        break;
    case Resize::InterpolationMode::CUBIC:
        props_.emplace_back("imode", "CUBIC");
        break;
    case Resize::InterpolationMode::LANCZOS4:
        props_.emplace_back("imode", "LANCZOS4");
        break;
    default:
        props_.emplace_back("imode", "INVALID");
        break;
    }
    switch (op_.format){
    case Resize::Format::NCHW:
        props_.emplace_back("format", "NCHW");
        break;
    case Resize::Format::NHWC:
        props_.emplace_back("format", "NHWC");
        break;
    case Resize::Format::NHWCD4:
        props_.emplace_back("format", "NHWCD4");
        break;
    case Resize::Format::NCHW4:
        props_.emplace_back("format", "NCHW4");
        break;
    case Resize::Format::NCHW8:
        props_.emplace_back("format", "NCHW8");
        break;
    case Resize::Format::NCHW32:
        props_.emplace_back("format", "NCHW32");
        break;
    case Resize::Format::NCHW88:
        props_.emplace_back("format", "NCHW88");
        break;
    case Resize::Format::NCHW44:
        props_.emplace_back("format", "NCHW44");
        break;
    case Resize::Format::NCHW44_DOT:
        props_.emplace_back("format", "NCHW44_DOT");
        break;
    case Resize::Format::NCHW4_NCHW32:
        props_.emplace_back("format", "NCHW4_NCHW32");
        break;
    case Resize::Format::NCHW32_NCHW4:
        props_.emplace_back("format", "NCHW32_NCHW4");
        break;
    case Resize::Format::NCHW4_NCHW:
        props_.emplace_back("format", "NCHW4_NCHW");
        break;
    case Resize::Format::NHWC_NCHW:
        props_.emplace_back("format", "NHWC_NCHW");
        break;
    case Resize::Format::NHWC_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NHWC_NCHW4_IC_SMALL");
        break;
    case Resize::Format::NCHW_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NCHW_NCHW4_IC_SMALL");
        break;
    case Resize::Format::CHWN4:
        props_.emplace_back("format", "CHWN4");
        break;
    case Resize::Format::NCHW64:
        props_.emplace_back("format", "NCHW64");
        break;
    case Resize::Format::NCHW4_NHWC:
        props_.emplace_back("format", "NCHW4_NHWC");
        break;
    default:
        props_.emplace_back("format", "INVALID");
        break;
    }
    return props_;
}
std::string Resize_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Resize>();
    static_cast<void>(op_);
    return "Resize";
}
} // anonymous namespace
OP_TRAIT_REG(Resize, Resize)
    .hash(Resize_hash_impl)
    .is_same_st(Resize_is_same_st_impl)
    .props(Resize_props_impl)
    .make_name(Resize_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Resize3D);

namespace {
size_t Resize3D_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Resize3D>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.imode));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.format));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.align_corners));
    return val;
}
bool Resize3D_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Resize3D>(),
         &&b_ = rhs_.cast_final_safe<Resize3D>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.imode != b_.imode) return false;
    if (a_.format != b_.format) return false;
    if (a_.align_corners != b_.align_corners) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Resize3D_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Resize3D>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.imode){
    case Resize3D::InterpolationMode::NEAREST:
        props_.emplace_back("imode", "NEAREST");
        break;
    case Resize3D::InterpolationMode::LINEAR:
        props_.emplace_back("imode", "LINEAR");
        break;
    case Resize3D::InterpolationMode::AREA:
        props_.emplace_back("imode", "AREA");
        break;
    case Resize3D::InterpolationMode::CUBIC:
        props_.emplace_back("imode", "CUBIC");
        break;
    case Resize3D::InterpolationMode::LANCZOS4:
        props_.emplace_back("imode", "LANCZOS4");
        break;
    default:
        props_.emplace_back("imode", "INVALID");
        break;
    }
    switch (op_.format){
    case Resize3D::Format::NCDHW:
        props_.emplace_back("format", "NCDHW");
        break;
    case Resize3D::Format::NDHWC:
        props_.emplace_back("format", "NDHWC");
        break;
    default:
        props_.emplace_back("format", "INVALID");
        break;
    }
    props_.emplace_back("align_corners", std::to_string(op_.align_corners));
    return props_;
}
std::string Resize3D_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Resize3D>();
    static_cast<void>(op_);
    return "Resize3D";
}
} // anonymous namespace
OP_TRAIT_REG(Resize3D, Resize3D)
    .hash(Resize3D_hash_impl)
    .is_same_st(Resize3D_is_same_st_impl)
    .props(Resize3D_props_impl)
    .make_name(Resize3D_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(SVD);

namespace {
size_t SVD_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<SVD>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.full_matrices));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.compute_uv));
    return val;
}
bool SVD_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<SVD>(),
         &&b_ = rhs_.cast_final_safe<SVD>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.full_matrices != b_.full_matrices) return false;
    if (a_.compute_uv != b_.compute_uv) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> SVD_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<SVD>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("full_matrices", std::to_string(op_.full_matrices));
    props_.emplace_back("compute_uv", std::to_string(op_.compute_uv));
    return props_;
}
std::string SVD_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<SVD>();
    static_cast<void>(op_);
    return "SVD";
}
} // anonymous namespace
OP_TRAIT_REG(SVD, SVD)
    .hash(SVD_hash_impl)
    .is_same_st(SVD_is_same_st_impl)
    .props(SVD_props_impl)
    .make_name(SVD_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(SetMeshIndexing);

namespace {
size_t SetMeshIndexing_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<SetMeshIndexing>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.items));
    return val;
}
bool SetMeshIndexing_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<SetMeshIndexing>(),
         &&b_ = rhs_.cast_final_safe<SetMeshIndexing>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.items != b_.items) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> SetMeshIndexing_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<SetMeshIndexing>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("items", "{std::vector}");
    return props_;
}
std::string SetMeshIndexing_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<SetMeshIndexing>();
    static_cast<void>(op_);
    return "SetMeshIndexing";
}
} // anonymous namespace
OP_TRAIT_REG(SetMeshIndexing, SetMeshIndexing)
    .hash(SetMeshIndexing_hash_impl)
    .is_same_st(SetMeshIndexing_is_same_st_impl)
    .props(SetMeshIndexing_props_impl)
    .make_name(SetMeshIndexing_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(SetSubtensor);

namespace {
size_t SetSubtensor_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<SetSubtensor>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.items));
    return val;
}
bool SetSubtensor_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<SetSubtensor>(),
         &&b_ = rhs_.cast_final_safe<SetSubtensor>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.items != b_.items) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> SetSubtensor_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<SetSubtensor>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("items", "{std::vector}");
    return props_;
}
std::string SetSubtensor_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<SetSubtensor>();
    static_cast<void>(op_);
    return "SetSubtensor";
}
} // anonymous namespace
OP_TRAIT_REG(SetSubtensor, SetSubtensor)
    .hash(SetSubtensor_hash_impl)
    .is_same_st(SetSubtensor_is_same_st_impl)
    .props(SetSubtensor_props_impl)
    .make_name(SetSubtensor_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(ShuffleRNG);

namespace {
size_t ShuffleRNG_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<ShuffleRNG>();
    static_cast<void>(op_);

    return mgb::hash_pair_combine(
      mgb::hash(op_.dyn_typeinfo()),
      mgb::hash(op_.handle)
      );
  }
bool ShuffleRNG_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<ShuffleRNG>(),
         &&b_ = rhs_.cast_final_safe<ShuffleRNG>();
    static_cast<void>(a_);
    static_cast<void>(b_);
return a_.handle == b_.handle;}
std::vector<std::pair<const char*, std::string>> ShuffleRNG_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<ShuffleRNG>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("seed", std::to_string(op_.seed));
    props_.emplace_back("handle", std::to_string(op_.handle));
    return props_;
}
std::string ShuffleRNG_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<ShuffleRNG>();
    static_cast<void>(op_);
    return "ShuffleRNG";
}
} // anonymous namespace
OP_TRAIT_REG(ShuffleRNG, ShuffleRNG)
    .hash(ShuffleRNG_hash_impl)
    .is_same_st(ShuffleRNG_is_same_st_impl)
    .props(ShuffleRNG_props_impl)
    .make_name(ShuffleRNG_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(SlidingWindowTranspose);

namespace {
size_t SlidingWindowTranspose_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<SlidingWindowTranspose>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.out_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.out_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pad_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.pad_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.stride_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dilate_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dilate_w));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.window_h));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.window_w));
    return val;
}
bool SlidingWindowTranspose_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<SlidingWindowTranspose>(),
         &&b_ = rhs_.cast_final_safe<SlidingWindowTranspose>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.out_h != b_.out_h) return false;
    if (a_.out_w != b_.out_w) return false;
    if (a_.pad_h != b_.pad_h) return false;
    if (a_.pad_w != b_.pad_w) return false;
    if (a_.stride_h != b_.stride_h) return false;
    if (a_.stride_w != b_.stride_w) return false;
    if (a_.dilate_h != b_.dilate_h) return false;
    if (a_.dilate_w != b_.dilate_w) return false;
    if (a_.window_h != b_.window_h) return false;
    if (a_.window_w != b_.window_w) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> SlidingWindowTranspose_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<SlidingWindowTranspose>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("out_h", std::to_string(op_.out_h));
    props_.emplace_back("out_w", std::to_string(op_.out_w));
    props_.emplace_back("pad_h", std::to_string(op_.pad_h));
    props_.emplace_back("pad_w", std::to_string(op_.pad_w));
    props_.emplace_back("stride_h", std::to_string(op_.stride_h));
    props_.emplace_back("stride_w", std::to_string(op_.stride_w));
    props_.emplace_back("dilate_h", std::to_string(op_.dilate_h));
    props_.emplace_back("dilate_w", std::to_string(op_.dilate_w));
    props_.emplace_back("window_h", std::to_string(op_.window_h));
    props_.emplace_back("window_w", std::to_string(op_.window_w));
    return props_;
}
std::string SlidingWindowTranspose_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<SlidingWindowTranspose>();
    static_cast<void>(op_);
    return "SlidingWindowTranspose";
}
} // anonymous namespace
OP_TRAIT_REG(SlidingWindowTranspose, SlidingWindowTranspose)
    .hash(SlidingWindowTranspose_hash_impl)
    .is_same_st(SlidingWindowTranspose_is_same_st_impl)
    .props(SlidingWindowTranspose_props_impl)
    .make_name(SlidingWindowTranspose_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Softmax);

namespace {
size_t Softmax_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Softmax>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.axis));
    return val;
}
bool Softmax_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Softmax>(),
         &&b_ = rhs_.cast_final_safe<Softmax>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.axis != b_.axis) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Softmax_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Softmax>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("axis", std::to_string(op_.axis));
    return props_;
}
std::string Softmax_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Softmax>();
    static_cast<void>(op_);
    return "Softmax";
}
} // anonymous namespace
OP_TRAIT_REG(Softmax, Softmax)
    .hash(Softmax_hash_impl)
    .is_same_st(Softmax_is_same_st_impl)
    .props(Softmax_props_impl)
    .make_name(Softmax_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Split);

namespace {
size_t Split_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Split>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.axis));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.nsections));
    return val;
}
bool Split_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Split>(),
         &&b_ = rhs_.cast_final_safe<Split>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.axis != b_.axis) return false;
    if (a_.nsections != b_.nsections) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Split_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Split>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("axis", std::to_string(op_.axis));
    props_.emplace_back("nsections", std::to_string(op_.nsections));
    return props_;
}
std::string Split_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Split>();
    static_cast<void>(op_);
    return "Split";
}
} // anonymous namespace
OP_TRAIT_REG(Split, Split)
    .hash(Split_hash_impl)
    .is_same_st(Split_is_same_st_impl)
    .props(Split_props_impl)
    .make_name(Split_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Stack);

namespace {
size_t Stack_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Stack>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.axis));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.comp_node));
    return val;
}
bool Stack_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Stack>(),
         &&b_ = rhs_.cast_final_safe<Stack>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.axis != b_.axis) return false;
    if (a_.comp_node != b_.comp_node) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Stack_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Stack>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("axis", std::to_string(op_.axis));
    props_.emplace_back("comp_node", op_.comp_node.to_string());
    return props_;
}
std::string Stack_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Stack>();
    static_cast<void>(op_);
    return "Stack";
}
} // anonymous namespace
OP_TRAIT_REG(Stack, Stack)
    .hash(Stack_hash_impl)
    .is_same_st(Stack_is_same_st_impl)
    .props(Stack_props_impl)
    .make_name(Stack_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Subtensor);

namespace {
size_t Subtensor_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Subtensor>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.items));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.slice_items));
    return val;
}
bool Subtensor_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<Subtensor>(),
         &&b_ = rhs_.cast_final_safe<Subtensor>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.items != b_.items) return false;
    if (a_.slice_items != b_.slice_items) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> Subtensor_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Subtensor>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("items", "{std::vector}");
    props_.emplace_back("slice_items", "{std::vector}");
    return props_;
}
std::string Subtensor_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<Subtensor>();
    static_cast<void>(op_);
    return "Subtensor";
}
} // anonymous namespace
OP_TRAIT_REG(Subtensor, Subtensor)
    .hash(Subtensor_hash_impl)
    .is_same_st(Subtensor_is_same_st_impl)
    .props(Subtensor_props_impl)
    .make_name(Subtensor_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(TQT);

namespace {
size_t TQT_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<TQT>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.qmin));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.qmax));
    return val;
}
bool TQT_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<TQT>(),
         &&b_ = rhs_.cast_final_safe<TQT>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.qmin != b_.qmin) return false;
    if (a_.qmax != b_.qmax) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> TQT_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<TQT>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("qmin", std::to_string(op_.qmin));
    props_.emplace_back("qmax", std::to_string(op_.qmax));
    return props_;
}
std::string TQT_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<TQT>();
    static_cast<void>(op_);
    return "TQT";
}
} // anonymous namespace
OP_TRAIT_REG(TQT, TQT)
    .hash(TQT_hash_impl)
    .is_same_st(TQT_is_same_st_impl)
    .props(TQT_props_impl)
    .make_name(TQT_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(TensorRTRuntime);

namespace {
size_t TensorRTRuntime_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<TensorRTRuntime>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.buf));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.buf_size));
    return val;
}
bool TensorRTRuntime_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<TensorRTRuntime>(),
         &&b_ = rhs_.cast_final_safe<TensorRTRuntime>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.buf != b_.buf) return false;
    if (a_.buf_size != b_.buf_size) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> TensorRTRuntime_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<TensorRTRuntime>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("buf", op_.buf);
    props_.emplace_back("buf_size", std::to_string(op_.buf_size));
    return props_;
}
std::string TensorRTRuntime_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<TensorRTRuntime>();
    static_cast<void>(op_);
    return "TensorRTRuntime";
}
} // anonymous namespace
OP_TRAIT_REG(TensorRTRuntime, TensorRTRuntime)
    .hash(TensorRTRuntime_hash_impl)
    .is_same_st(TensorRTRuntime_is_same_st_impl)
    .props(TensorRTRuntime_props_impl)
    .make_name(TensorRTRuntime_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(TopK);

namespace {
size_t TopK_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<TopK>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.mode));
    return val;
}
bool TopK_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<TopK>(),
         &&b_ = rhs_.cast_final_safe<TopK>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.mode != b_.mode) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> TopK_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<TopK>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.mode){
    case TopK::Mode::KTH_ONLY:
        props_.emplace_back("mode", "KTH_ONLY");
        break;
    case TopK::Mode::VALUE_IDX_NOSORT:
        props_.emplace_back("mode", "VALUE_IDX_NOSORT");
        break;
    case TopK::Mode::VALUE_IDX_SORTED:
        props_.emplace_back("mode", "VALUE_IDX_SORTED");
        break;
    default:
        props_.emplace_back("mode", "INVALID");
        break;
    }
    return props_;
}
std::string TopK_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<TopK>();
    static_cast<void>(op_);
    return "TopK";
}
} // anonymous namespace
OP_TRAIT_REG(TopK, TopK)
    .hash(TopK_hash_impl)
    .is_same_st(TopK_is_same_st_impl)
    .props(TopK_props_impl)
    .make_name(TopK_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(TypeCvt);

namespace {
size_t TypeCvt_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<TypeCvt>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::hash(op_.dtype.handle()));
    return val;
}
bool TypeCvt_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<TypeCvt>(),
         &&b_ = rhs_.cast_final_safe<TypeCvt>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.dtype != b_.dtype) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> TypeCvt_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<TypeCvt>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("dtype", op_.dtype.name());
    return props_;
}
std::string TypeCvt_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<TypeCvt>();
    static_cast<void>(op_);
    return "TypeCvt";
}
} // anonymous namespace
OP_TRAIT_REG(TypeCvt, TypeCvt)
    .hash(TypeCvt_hash_impl)
    .is_same_st(TypeCvt_is_same_st_impl)
    .props(TypeCvt_props_impl)
    .make_name(TypeCvt_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(UniformRNG);

namespace {
size_t UniformRNG_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<UniformRNG>();
    static_cast<void>(op_);

    return mgb::hash_pair_combine(
      mgb::hash(op_.dyn_typeinfo()),
      mgb::hash_pair_combine(
        mgb::hash(op_.handle),
        mgb::hash(op_.dtype.enumv())
      )
    );
  }
bool UniformRNG_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<UniformRNG>(),
         &&b_ = rhs_.cast_final_safe<UniformRNG>();
    static_cast<void>(a_);
    static_cast<void>(b_);
return a_.handle == b_.handle && a_.dtype == b_.dtype;}
std::vector<std::pair<const char*, std::string>> UniformRNG_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<UniformRNG>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    props_.emplace_back("seed", std::to_string(op_.seed));
    props_.emplace_back("dtype", op_.dtype.name());
    props_.emplace_back("handle", std::to_string(op_.handle));
    return props_;
}
std::string UniformRNG_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<UniformRNG>();
    static_cast<void>(op_);
    return "UniformRNG";
}
} // anonymous namespace
OP_TRAIT_REG(UniformRNG, UniformRNG)
    .hash(UniformRNG_hash_impl)
    .is_same_st(UniformRNG_is_same_st_impl)
    .props(UniformRNG_props_impl)
    .make_name(UniformRNG_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(WarpAffine);

namespace {
size_t WarpAffine_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<WarpAffine>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.imode));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.border_mode));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.border_val));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.format));
    return val;
}
bool WarpAffine_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<WarpAffine>(),
         &&b_ = rhs_.cast_final_safe<WarpAffine>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.imode != b_.imode) return false;
    if (a_.border_mode != b_.border_mode) return false;
    if (a_.border_val != b_.border_val) return false;
    if (a_.format != b_.format) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> WarpAffine_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<WarpAffine>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.imode){
    case WarpAffine::InterpolationMode::NEAREST:
        props_.emplace_back("imode", "NEAREST");
        break;
    case WarpAffine::InterpolationMode::LINEAR:
        props_.emplace_back("imode", "LINEAR");
        break;
    case WarpAffine::InterpolationMode::AREA:
        props_.emplace_back("imode", "AREA");
        break;
    case WarpAffine::InterpolationMode::CUBIC:
        props_.emplace_back("imode", "CUBIC");
        break;
    case WarpAffine::InterpolationMode::LANCZOS4:
        props_.emplace_back("imode", "LANCZOS4");
        break;
    default:
        props_.emplace_back("imode", "INVALID");
        break;
    }
    switch (op_.border_mode){
    case WarpAffine::BorderMode::REPLICATE:
        props_.emplace_back("border_mode", "REPLICATE");
        break;
    case WarpAffine::BorderMode::REFLECT:
        props_.emplace_back("border_mode", "REFLECT");
        break;
    case WarpAffine::BorderMode::REFLECT_101:
        props_.emplace_back("border_mode", "REFLECT_101");
        break;
    case WarpAffine::BorderMode::WRAP:
        props_.emplace_back("border_mode", "WRAP");
        break;
    case WarpAffine::BorderMode::CONSTANT:
        props_.emplace_back("border_mode", "CONSTANT");
        break;
    case WarpAffine::BorderMode::TRANSPARENT:
        props_.emplace_back("border_mode", "TRANSPARENT");
        break;
    case WarpAffine::BorderMode::ISOLATED:
        props_.emplace_back("border_mode", "ISOLATED");
        break;
    default:
        props_.emplace_back("border_mode", "INVALID");
        break;
    }
    props_.emplace_back("border_val", std::to_string(op_.border_val));
    switch (op_.format){
    case WarpAffine::Format::NCHW:
        props_.emplace_back("format", "NCHW");
        break;
    case WarpAffine::Format::NHWC:
        props_.emplace_back("format", "NHWC");
        break;
    case WarpAffine::Format::NHWCD4:
        props_.emplace_back("format", "NHWCD4");
        break;
    case WarpAffine::Format::NCHW4:
        props_.emplace_back("format", "NCHW4");
        break;
    case WarpAffine::Format::NCHW8:
        props_.emplace_back("format", "NCHW8");
        break;
    case WarpAffine::Format::NCHW32:
        props_.emplace_back("format", "NCHW32");
        break;
    case WarpAffine::Format::NCHW88:
        props_.emplace_back("format", "NCHW88");
        break;
    case WarpAffine::Format::NCHW44:
        props_.emplace_back("format", "NCHW44");
        break;
    case WarpAffine::Format::NCHW44_DOT:
        props_.emplace_back("format", "NCHW44_DOT");
        break;
    case WarpAffine::Format::NCHW4_NCHW32:
        props_.emplace_back("format", "NCHW4_NCHW32");
        break;
    case WarpAffine::Format::NCHW32_NCHW4:
        props_.emplace_back("format", "NCHW32_NCHW4");
        break;
    case WarpAffine::Format::NCHW4_NCHW:
        props_.emplace_back("format", "NCHW4_NCHW");
        break;
    case WarpAffine::Format::NHWC_NCHW:
        props_.emplace_back("format", "NHWC_NCHW");
        break;
    case WarpAffine::Format::NHWC_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NHWC_NCHW4_IC_SMALL");
        break;
    case WarpAffine::Format::NCHW_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NCHW_NCHW4_IC_SMALL");
        break;
    case WarpAffine::Format::CHWN4:
        props_.emplace_back("format", "CHWN4");
        break;
    case WarpAffine::Format::NCHW64:
        props_.emplace_back("format", "NCHW64");
        break;
    case WarpAffine::Format::NCHW4_NHWC:
        props_.emplace_back("format", "NCHW4_NHWC");
        break;
    default:
        props_.emplace_back("format", "INVALID");
        break;
    }
    return props_;
}
std::string WarpAffine_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<WarpAffine>();
    static_cast<void>(op_);
    return "WarpAffine";
}
} // anonymous namespace
OP_TRAIT_REG(WarpAffine, WarpAffine)
    .hash(WarpAffine_hash_impl)
    .is_same_st(WarpAffine_is_same_st_impl)
    .props(WarpAffine_props_impl)
    .make_name(WarpAffine_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(WarpPerspective);

namespace {
size_t WarpPerspective_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<WarpPerspective>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.imode));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.bmode));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.format));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.border_val));
    return val;
}
bool WarpPerspective_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<WarpPerspective>(),
         &&b_ = rhs_.cast_final_safe<WarpPerspective>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.imode != b_.imode) return false;
    if (a_.bmode != b_.bmode) return false;
    if (a_.format != b_.format) return false;
    if (a_.border_val != b_.border_val) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> WarpPerspective_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<WarpPerspective>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.imode){
    case WarpPerspective::InterpolationMode::NEAREST:
        props_.emplace_back("imode", "NEAREST");
        break;
    case WarpPerspective::InterpolationMode::LINEAR:
        props_.emplace_back("imode", "LINEAR");
        break;
    case WarpPerspective::InterpolationMode::AREA:
        props_.emplace_back("imode", "AREA");
        break;
    case WarpPerspective::InterpolationMode::CUBIC:
        props_.emplace_back("imode", "CUBIC");
        break;
    case WarpPerspective::InterpolationMode::LANCZOS4:
        props_.emplace_back("imode", "LANCZOS4");
        break;
    default:
        props_.emplace_back("imode", "INVALID");
        break;
    }
    switch (op_.bmode){
    case WarpPerspective::BorderMode::REPLICATE:
        props_.emplace_back("bmode", "REPLICATE");
        break;
    case WarpPerspective::BorderMode::REFLECT:
        props_.emplace_back("bmode", "REFLECT");
        break;
    case WarpPerspective::BorderMode::REFLECT_101:
        props_.emplace_back("bmode", "REFLECT_101");
        break;
    case WarpPerspective::BorderMode::WRAP:
        props_.emplace_back("bmode", "WRAP");
        break;
    case WarpPerspective::BorderMode::CONSTANT:
        props_.emplace_back("bmode", "CONSTANT");
        break;
    case WarpPerspective::BorderMode::TRANSPARENT:
        props_.emplace_back("bmode", "TRANSPARENT");
        break;
    case WarpPerspective::BorderMode::ISOLATED:
        props_.emplace_back("bmode", "ISOLATED");
        break;
    default:
        props_.emplace_back("bmode", "INVALID");
        break;
    }
    switch (op_.format){
    case WarpPerspective::Format::NCHW:
        props_.emplace_back("format", "NCHW");
        break;
    case WarpPerspective::Format::NHWC:
        props_.emplace_back("format", "NHWC");
        break;
    case WarpPerspective::Format::NHWCD4:
        props_.emplace_back("format", "NHWCD4");
        break;
    case WarpPerspective::Format::NCHW4:
        props_.emplace_back("format", "NCHW4");
        break;
    case WarpPerspective::Format::NCHW8:
        props_.emplace_back("format", "NCHW8");
        break;
    case WarpPerspective::Format::NCHW32:
        props_.emplace_back("format", "NCHW32");
        break;
    case WarpPerspective::Format::NCHW88:
        props_.emplace_back("format", "NCHW88");
        break;
    case WarpPerspective::Format::NCHW44:
        props_.emplace_back("format", "NCHW44");
        break;
    case WarpPerspective::Format::NCHW44_DOT:
        props_.emplace_back("format", "NCHW44_DOT");
        break;
    case WarpPerspective::Format::NCHW4_NCHW32:
        props_.emplace_back("format", "NCHW4_NCHW32");
        break;
    case WarpPerspective::Format::NCHW32_NCHW4:
        props_.emplace_back("format", "NCHW32_NCHW4");
        break;
    case WarpPerspective::Format::NCHW4_NCHW:
        props_.emplace_back("format", "NCHW4_NCHW");
        break;
    case WarpPerspective::Format::NHWC_NCHW:
        props_.emplace_back("format", "NHWC_NCHW");
        break;
    case WarpPerspective::Format::NHWC_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NHWC_NCHW4_IC_SMALL");
        break;
    case WarpPerspective::Format::NCHW_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NCHW_NCHW4_IC_SMALL");
        break;
    case WarpPerspective::Format::CHWN4:
        props_.emplace_back("format", "CHWN4");
        break;
    case WarpPerspective::Format::NCHW64:
        props_.emplace_back("format", "NCHW64");
        break;
    case WarpPerspective::Format::NCHW4_NHWC:
        props_.emplace_back("format", "NCHW4_NHWC");
        break;
    default:
        props_.emplace_back("format", "INVALID");
        break;
    }
    props_.emplace_back("border_val", std::to_string(op_.border_val));
    return props_;
}
std::string WarpPerspective_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<WarpPerspective>();
    static_cast<void>(op_);
    return "WarpPerspective";
}
} // anonymous namespace
OP_TRAIT_REG(WarpPerspective, WarpPerspective)
    .hash(WarpPerspective_hash_impl)
    .is_same_st(WarpPerspective_is_same_st_impl)
    .props(WarpPerspective_props_impl)
    .make_name(WarpPerspective_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(WarpPerspectiveBackwardData);

namespace {
size_t WarpPerspectiveBackwardData_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<WarpPerspectiveBackwardData>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.imode));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.bmode));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.format));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.border_val));
    return val;
}
bool WarpPerspectiveBackwardData_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<WarpPerspectiveBackwardData>(),
         &&b_ = rhs_.cast_final_safe<WarpPerspectiveBackwardData>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.imode != b_.imode) return false;
    if (a_.bmode != b_.bmode) return false;
    if (a_.format != b_.format) return false;
    if (a_.border_val != b_.border_val) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> WarpPerspectiveBackwardData_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<WarpPerspectiveBackwardData>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.imode){
    case WarpPerspectiveBackwardData::InterpolationMode::NEAREST:
        props_.emplace_back("imode", "NEAREST");
        break;
    case WarpPerspectiveBackwardData::InterpolationMode::LINEAR:
        props_.emplace_back("imode", "LINEAR");
        break;
    case WarpPerspectiveBackwardData::InterpolationMode::AREA:
        props_.emplace_back("imode", "AREA");
        break;
    case WarpPerspectiveBackwardData::InterpolationMode::CUBIC:
        props_.emplace_back("imode", "CUBIC");
        break;
    case WarpPerspectiveBackwardData::InterpolationMode::LANCZOS4:
        props_.emplace_back("imode", "LANCZOS4");
        break;
    default:
        props_.emplace_back("imode", "INVALID");
        break;
    }
    switch (op_.bmode){
    case WarpPerspectiveBackwardData::BorderMode::REPLICATE:
        props_.emplace_back("bmode", "REPLICATE");
        break;
    case WarpPerspectiveBackwardData::BorderMode::REFLECT:
        props_.emplace_back("bmode", "REFLECT");
        break;
    case WarpPerspectiveBackwardData::BorderMode::REFLECT_101:
        props_.emplace_back("bmode", "REFLECT_101");
        break;
    case WarpPerspectiveBackwardData::BorderMode::WRAP:
        props_.emplace_back("bmode", "WRAP");
        break;
    case WarpPerspectiveBackwardData::BorderMode::CONSTANT:
        props_.emplace_back("bmode", "CONSTANT");
        break;
    case WarpPerspectiveBackwardData::BorderMode::TRANSPARENT:
        props_.emplace_back("bmode", "TRANSPARENT");
        break;
    case WarpPerspectiveBackwardData::BorderMode::ISOLATED:
        props_.emplace_back("bmode", "ISOLATED");
        break;
    default:
        props_.emplace_back("bmode", "INVALID");
        break;
    }
    switch (op_.format){
    case WarpPerspectiveBackwardData::Format::NCHW:
        props_.emplace_back("format", "NCHW");
        break;
    case WarpPerspectiveBackwardData::Format::NHWC:
        props_.emplace_back("format", "NHWC");
        break;
    case WarpPerspectiveBackwardData::Format::NHWCD4:
        props_.emplace_back("format", "NHWCD4");
        break;
    case WarpPerspectiveBackwardData::Format::NCHW4:
        props_.emplace_back("format", "NCHW4");
        break;
    case WarpPerspectiveBackwardData::Format::NCHW8:
        props_.emplace_back("format", "NCHW8");
        break;
    case WarpPerspectiveBackwardData::Format::NCHW32:
        props_.emplace_back("format", "NCHW32");
        break;
    case WarpPerspectiveBackwardData::Format::NCHW88:
        props_.emplace_back("format", "NCHW88");
        break;
    case WarpPerspectiveBackwardData::Format::NCHW44:
        props_.emplace_back("format", "NCHW44");
        break;
    case WarpPerspectiveBackwardData::Format::NCHW44_DOT:
        props_.emplace_back("format", "NCHW44_DOT");
        break;
    case WarpPerspectiveBackwardData::Format::NCHW4_NCHW32:
        props_.emplace_back("format", "NCHW4_NCHW32");
        break;
    case WarpPerspectiveBackwardData::Format::NCHW32_NCHW4:
        props_.emplace_back("format", "NCHW32_NCHW4");
        break;
    case WarpPerspectiveBackwardData::Format::NCHW4_NCHW:
        props_.emplace_back("format", "NCHW4_NCHW");
        break;
    case WarpPerspectiveBackwardData::Format::NHWC_NCHW:
        props_.emplace_back("format", "NHWC_NCHW");
        break;
    case WarpPerspectiveBackwardData::Format::NHWC_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NHWC_NCHW4_IC_SMALL");
        break;
    case WarpPerspectiveBackwardData::Format::NCHW_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NCHW_NCHW4_IC_SMALL");
        break;
    case WarpPerspectiveBackwardData::Format::CHWN4:
        props_.emplace_back("format", "CHWN4");
        break;
    case WarpPerspectiveBackwardData::Format::NCHW64:
        props_.emplace_back("format", "NCHW64");
        break;
    case WarpPerspectiveBackwardData::Format::NCHW4_NHWC:
        props_.emplace_back("format", "NCHW4_NHWC");
        break;
    default:
        props_.emplace_back("format", "INVALID");
        break;
    }
    props_.emplace_back("border_val", std::to_string(op_.border_val));
    return props_;
}
std::string WarpPerspectiveBackwardData_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<WarpPerspectiveBackwardData>();
    static_cast<void>(op_);
    return "WarpPerspectiveBackwardData";
}
} // anonymous namespace
OP_TRAIT_REG(WarpPerspectiveBackwardData, WarpPerspectiveBackwardData)
    .hash(WarpPerspectiveBackwardData_hash_impl)
    .is_same_st(WarpPerspectiveBackwardData_is_same_st_impl)
    .props(WarpPerspectiveBackwardData_props_impl)
    .make_name(WarpPerspectiveBackwardData_make_name_impl);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(WarpPerspectiveBackwardMat);

namespace {
size_t WarpPerspectiveBackwardMat_hash_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<WarpPerspectiveBackwardMat>();
    static_cast<void>(op_);
    size_t val = mgb::hash(op_.dyn_typeinfo());
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.imode));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.bmode));
    val = mgb::hash_pair_combine(val, mgb::enumhash()(op_.format));
    val = mgb::hash_pair_combine(val, mgb::hash(op_.border_val));
    return val;
}
bool WarpPerspectiveBackwardMat_is_same_st_impl(const OpDef& lhs_, const OpDef& rhs_) {
    auto &&a_ = lhs_.cast_final_safe<WarpPerspectiveBackwardMat>(),
         &&b_ = rhs_.cast_final_safe<WarpPerspectiveBackwardMat>();
    static_cast<void>(a_);
    static_cast<void>(b_);
    if (a_.imode != b_.imode) return false;
    if (a_.bmode != b_.bmode) return false;
    if (a_.format != b_.format) return false;
    if (a_.border_val != b_.border_val) return false;
    return true;
}
std::vector<std::pair<const char*, std::string>> WarpPerspectiveBackwardMat_props_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<WarpPerspectiveBackwardMat>();
    static_cast<void>(op_);
    std::vector<std::pair<const char*, std::string>> props_;
    switch (op_.imode){
    case WarpPerspectiveBackwardMat::InterpolationMode::NEAREST:
        props_.emplace_back("imode", "NEAREST");
        break;
    case WarpPerspectiveBackwardMat::InterpolationMode::LINEAR:
        props_.emplace_back("imode", "LINEAR");
        break;
    case WarpPerspectiveBackwardMat::InterpolationMode::AREA:
        props_.emplace_back("imode", "AREA");
        break;
    case WarpPerspectiveBackwardMat::InterpolationMode::CUBIC:
        props_.emplace_back("imode", "CUBIC");
        break;
    case WarpPerspectiveBackwardMat::InterpolationMode::LANCZOS4:
        props_.emplace_back("imode", "LANCZOS4");
        break;
    default:
        props_.emplace_back("imode", "INVALID");
        break;
    }
    switch (op_.bmode){
    case WarpPerspectiveBackwardMat::BorderMode::REPLICATE:
        props_.emplace_back("bmode", "REPLICATE");
        break;
    case WarpPerspectiveBackwardMat::BorderMode::REFLECT:
        props_.emplace_back("bmode", "REFLECT");
        break;
    case WarpPerspectiveBackwardMat::BorderMode::REFLECT_101:
        props_.emplace_back("bmode", "REFLECT_101");
        break;
    case WarpPerspectiveBackwardMat::BorderMode::WRAP:
        props_.emplace_back("bmode", "WRAP");
        break;
    case WarpPerspectiveBackwardMat::BorderMode::CONSTANT:
        props_.emplace_back("bmode", "CONSTANT");
        break;
    case WarpPerspectiveBackwardMat::BorderMode::TRANSPARENT:
        props_.emplace_back("bmode", "TRANSPARENT");
        break;
    case WarpPerspectiveBackwardMat::BorderMode::ISOLATED:
        props_.emplace_back("bmode", "ISOLATED");
        break;
    default:
        props_.emplace_back("bmode", "INVALID");
        break;
    }
    switch (op_.format){
    case WarpPerspectiveBackwardMat::Format::NCHW:
        props_.emplace_back("format", "NCHW");
        break;
    case WarpPerspectiveBackwardMat::Format::NHWC:
        props_.emplace_back("format", "NHWC");
        break;
    case WarpPerspectiveBackwardMat::Format::NHWCD4:
        props_.emplace_back("format", "NHWCD4");
        break;
    case WarpPerspectiveBackwardMat::Format::NCHW4:
        props_.emplace_back("format", "NCHW4");
        break;
    case WarpPerspectiveBackwardMat::Format::NCHW8:
        props_.emplace_back("format", "NCHW8");
        break;
    case WarpPerspectiveBackwardMat::Format::NCHW32:
        props_.emplace_back("format", "NCHW32");
        break;
    case WarpPerspectiveBackwardMat::Format::NCHW88:
        props_.emplace_back("format", "NCHW88");
        break;
    case WarpPerspectiveBackwardMat::Format::NCHW44:
        props_.emplace_back("format", "NCHW44");
        break;
    case WarpPerspectiveBackwardMat::Format::NCHW44_DOT:
        props_.emplace_back("format", "NCHW44_DOT");
        break;
    case WarpPerspectiveBackwardMat::Format::NCHW4_NCHW32:
        props_.emplace_back("format", "NCHW4_NCHW32");
        break;
    case WarpPerspectiveBackwardMat::Format::NCHW32_NCHW4:
        props_.emplace_back("format", "NCHW32_NCHW4");
        break;
    case WarpPerspectiveBackwardMat::Format::NCHW4_NCHW:
        props_.emplace_back("format", "NCHW4_NCHW");
        break;
    case WarpPerspectiveBackwardMat::Format::NHWC_NCHW:
        props_.emplace_back("format", "NHWC_NCHW");
        break;
    case WarpPerspectiveBackwardMat::Format::NHWC_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NHWC_NCHW4_IC_SMALL");
        break;
    case WarpPerspectiveBackwardMat::Format::NCHW_NCHW4_IC_SMALL:
        props_.emplace_back("format", "NCHW_NCHW4_IC_SMALL");
        break;
    case WarpPerspectiveBackwardMat::Format::CHWN4:
        props_.emplace_back("format", "CHWN4");
        break;
    case WarpPerspectiveBackwardMat::Format::NCHW64:
        props_.emplace_back("format", "NCHW64");
        break;
    case WarpPerspectiveBackwardMat::Format::NCHW4_NHWC:
        props_.emplace_back("format", "NCHW4_NHWC");
        break;
    default:
        props_.emplace_back("format", "INVALID");
        break;
    }
    props_.emplace_back("border_val", std::to_string(op_.border_val));
    return props_;
}
std::string WarpPerspectiveBackwardMat_make_name_impl(const OpDef& def_) {
    auto&& op_ = def_.cast_final_safe<WarpPerspectiveBackwardMat>();
    static_cast<void>(op_);
    return "WarpPerspectiveBackwardMat";
}
} // anonymous namespace
OP_TRAIT_REG(WarpPerspectiveBackwardMat, WarpPerspectiveBackwardMat)
    .hash(WarpPerspectiveBackwardMat_hash_impl)
    .is_same_st(WarpPerspectiveBackwardMat_is_same_st_impl)
    .props(WarpPerspectiveBackwardMat_props_impl)
    .make_name(WarpPerspectiveBackwardMat_make_name_impl);

// clang-format on
