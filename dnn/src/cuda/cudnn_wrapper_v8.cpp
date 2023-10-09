#include "src/cuda/cudnn_wrapper_v8.h"
#if CUDNN_VERSION >= 8020 && 0  // FIXME(hc): need fix
#include "src/cuda/cudnn_wrapper.h"

#include "src/common/utils.h"
#include "src/cuda/utils.h"

#include "src/cuda/conv_bias/helper.h"

#include "cudnn_frontend_EngineConfigGenerator.h"

#include "megdnn/algorithm_cache.h"

using namespace megdnn;
using namespace cuda;

// helper functions for underlying descriptors
namespace {
cudnnDataType_t get_cudnn_data_type(DType type) {
    switch (type.enumv()) {
        case DTypeEnum::Float32:
            return CUDNN_DATA_FLOAT;
        case DTypeEnum::Float16:
            return CUDNN_DATA_HALF;
        case DTypeEnum::Int32:
        case DTypeEnum::QuantizedS32:
            return CUDNN_DATA_INT32;
        case DTypeEnum::QuantizedS8:
        case DTypeEnum::Int8:
            return CUDNN_DATA_INT8;
        default:
            megdnn_throw("dtype must be float16/float32/int8/qint8/int32/qint32");
    }
}

cudnnDataType_t get_compute_type(
        DType type, param::Convolution::ComputeMode comp_mode) {
    if (type.enumv() == DTypeEnum::Float32) {
        return CUDNN_DATA_FLOAT;
    } else if (type.enumv() == DTypeEnum::Float16) {
        return get_compute_type_fp16(comp_mode);
    } else if (
            type.category() == DTypeCategory::INT ||
            type.category() == DTypeCategory::QUANTIZED) {
        return CUDNN_DATA_INT32;
    } else {
        megdnn_throw("unsupported compute type for convolution");
    }
}

using Format = param::Convolution::Format;
using IntArrayRef = SmallVector<int64_t>;
std::pair<IntArrayRef, IntArrayRef> get_shape_and_stride(
        const TensorLayout& layout, const Format format, int64_t nr_group) {
    // DENSE: n, c, h, w
    //        n, k, p, q; ndim = 4
    // GROUP: n, g, c, h, w
    //        n, g, k, p, q; ndim = 5
    static constexpr size_t CUDNN_NDIM = 4;
    size_t cudnn_ndim = CUDNN_NDIM;
    if (nr_group > 1)
        cudnn_ndim += 1;
    IntArrayRef shape(cudnn_ndim);
    IntArrayRef stride(cudnn_ndim);

    if (format == Format::NCHW4 || format == Format::NCHW32)
        megdnn_assert_eq_size_t(layout.ndim, 5_z);
    else
        megdnn_assert_eq_size_t(layout.ndim, 4_z);

    size_t c_pos, spatial_pos;
    if (format == Format::NCHW || format == Format::NCHW4 || format == Format::NCHW32) {
        c_pos = 1;
        spatial_pos = 2;
    } else {
        megdnn_assert(format == Format::NHWC);
        c_pos = 3;
        spatial_pos = 1;
    }
    int64_t vector_count, vector_dimension;
    std::tie(vector_count, vector_dimension) = get_vector_count_and_dimension(format);

    size_t out_c_pos = nr_group == 1 ? 1 : 2;
    size_t out_spatial_pos = nr_group == 1 ? 2 : 3;
    // For NCHW4 and NCHW32 we still compute standard strides here to input to cuDNN
    // functions. We will manually scale by resizeFactor in the cpu ref.
    shape[0] = layout[0];
    if (nr_group > 1)
        shape[1] = nr_group;
    shape[out_c_pos] = layout[c_pos] / nr_group;
    shape[out_spatial_pos] = layout[spatial_pos];
    shape[out_spatial_pos + 1] = layout[spatial_pos + 1];
    if (c_pos == 1) {
        stride[cudnn_ndim - 1] = 1;
        for (int i = cudnn_ndim - 2; i >= 0; --i) {
            stride[i] = stride[i + 1] * shape[i + 1];
        }
    } else {
        megdnn_assert(c_pos == 3);  // Here we assume that the format is NHWC
        stride[out_c_pos] = 1;
        if (nr_group > 1)
            stride[1] = shape[out_c_pos] * stride[out_c_pos];
        stride[out_spatial_pos + 1] = stride[1] * shape[1];
        stride[out_spatial_pos] =
                stride[out_spatial_pos + 1] * shape[out_spatial_pos + 1];
        stride[0] = stride[out_spatial_pos] * shape[out_spatial_pos];
    }
    return {shape, stride};
}

/* --------------- make cudnn-frontend tensor descriptor --------------- */
auto make_tensor_descriptor(
        int64_t id, uint8_t alignment, const TensorLayout& layout, const Format format,
        int64_t nr_group, bool is_virtual = false) {
    int64_t vector_count, vector_dimension;
    std::tie(vector_count, vector_dimension) = get_vector_count_and_dimension(format);
    IntArrayRef shape, stride;
    std::tie(shape, stride) = get_shape_and_stride(layout, format, nr_group);
    return cudnn_frontend::TensorBuilder()
            .setDim(shape.size(), shape.data())
            .setStrides(stride.size(), stride.data())
            .setId(id)
            .setAlignment(alignment)
            .setDataType(get_cudnn_data_type(layout.dtype))
            .setVirtual(is_virtual)
            .setVectorCountAndDimension(vector_count, vector_dimension)
            .build();
}

/* --------------- make cudnn-frontend filter descriptor --------------- */
template <typename FilterMeta>
cudnn_frontend::Tensor make_filter_descriptor(uint8_t alignment, const FilterMeta& fm) {
    // DENSE: k, c, r, s; ndim = 4
    // GROUP: g, k, c, r, s; ndim = 5
    // generate shape and stride
    static constexpr size_t CUDNN_NDIM = 4;
    size_t cudnn_ndim = CUDNN_NDIM;
    if (fm.group > 1)
        cudnn_ndim += 1;
    IntArrayRef shape(cudnn_ndim), stride(cudnn_ndim);
    auto format = fm.format;
    int64_t vector_count, vector_dimension;
    std::tie(vector_count, vector_dimension) = get_vector_count_and_dimension(format);

    int64_t group = fm.group;
    size_t out_ch_pos = group == 1 ? 0 : 1;
    size_t in_ch_pos = group == 1 ? 1 : 2;
    size_t filter_start = group == 1 ? 2 : 3;
    if (group > 1)
        shape[0] = group;
    shape[out_ch_pos] = fm.ocpg;
    shape[in_ch_pos] = fm.icpg / vector_count;
    shape[filter_start] = fm.spatial[0];
    shape[filter_start + 1] = fm.spatial[1];
    if (format == Format::NCHW || format == Format::NCHW4 || format == Format::NCHW32) {
        stride[cudnn_ndim - 1] = 1;
        for (int i = cudnn_ndim - 2; i >= 0; --i) {
            stride[i] = stride[i + 1] * shape[i + 1];
        }
    } else {
        megdnn_assert(
                format == Format::NHWC);  // Here we assume that the format is NHWC
        stride[in_ch_pos] = 1;
        stride[filter_start + 1] = stride[in_ch_pos] * shape[in_ch_pos];
        stride[filter_start] = stride[filter_start + 1] * shape[filter_start + 1];
        stride[out_ch_pos] = stride[filter_start] * shape[filter_start];
        if (group > 1)
            stride[0] = stride[out_ch_pos] * shape[out_ch_pos];
    }
    return cudnn_frontend::TensorBuilder()
            .setDim(shape.size(), shape.data())
            .setStrides(stride.size(), stride.data())
            .setId('w')  // weight descriptor
            .setAlignment(alignment)
            .setDataType(get_cudnn_data_type(fm.dtype))
            .setVectorCountAndDimension(vector_count, vector_dimension)
            .build();
}

/* --------------- make cudnn-frontend conv descriptor --------------- */
template <typename Param>
cudnn_frontend::ConvDesc_v8 make_conv_descriptor(
        cudnnDataType_t data_type, const Param& param) {
    IntArrayRef padding = {param.pad_h, param.pad_w};
    IntArrayRef stride = {param.stride_h, param.stride_w};
    IntArrayRef dilation = {param.dilate_h, param.dilate_w};
    uint64_t conv_dim = stride.size();
    cudnnConvolutionMode_t mode;
    switch (param.mode) {
        case Param::Mode::CROSS_CORRELATION:
            mode = CUDNN_CROSS_CORRELATION;
            break;
        case Param::Mode::CONVOLUTION:
            mode = CUDNN_CONVOLUTION;
            break;
        default:
            megdnn_throw("conv mode must be conv or xcorr.");
    }
    return cudnn_frontend::ConvDescBuilder()
            .setDataType(data_type)
            .setMathMode(mode)
            .setNDims(conv_dim)
            .setStrides(conv_dim, stride.data())
            .setPrePadding(conv_dim, padding.data())
            .setPostPadding(conv_dim, padding.data())
            .setDilation(conv_dim, dilation.data())
            .build();
}

/* --------------- make cudnn-frontend activation descriptor --------------- */
auto make_activation_descriptor(
        DType data_type, const param::ConvBias::NonlineMode nonline_mode) {
    cudnnPointwiseMode_t mode;
    using NonlineMode = param::ConvBias::NonlineMode;
    switch (nonline_mode) {
        case NonlineMode::RELU:
            mode = CUDNN_POINTWISE_RELU_FWD;
            break;
        case NonlineMode::SIGMOID:
            mode = CUDNN_POINTWISE_SIGMOID_FWD;
            break;
        default:
            megdnn_throw("unsupported non linear mode");
    }
    return cudnn_frontend::PointWiseDescBuilder()
            .setMode(mode)
            .setMathPrecision(get_cudnn_data_type(data_type))
            .build();
}

// high-level api for convolution execution
struct StaticData {
    using Key = megdnn::AlgorithmCache::Key;
    using KeyStorage = megdnn::AlgorithmCache::KeyStorage;
    using KeyHash = megdnn::AlgorithmCache::Hash;
    using Result = cudnn_frontend::ExecutionPlan;
    using CudnnFrontendExecutionPlanCache =
            std::unordered_map<KeyStorage, Result, KeyHash>;
    CudnnFrontendExecutionPlanCache cache;
#if __DEPLOY_ON_XP_SP2__
    size_t cache_mutex;
#else
    std::mutex cache_mutex;
#endif
    cudnnBackendHeurMode_t heur_mode = CUDNN_HEUR_MODE_INSTANT;
    bool deterministic = true;
};

StaticData& static_data() {
    static StaticData inst;
    return inst;
}

template <typename Opr>
struct CudnnBackendOpTypeTrait;

template <>
struct CudnnBackendOpTypeTrait<ConvolutionForward> {
    static constexpr cudnnBackendDescriptorType_t OPERATION =
            CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
};

template <>
struct CudnnBackendOpTypeTrait<ConvolutionBackwardData> {
    static constexpr cudnnBackendDescriptorType_t OPERATION =
            CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR;
};

template <>
struct CudnnBackendOpTypeTrait<ConvolutionBackwardFilter> {
    static constexpr cudnnBackendDescriptorType_t OPERATION =
            CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR;
};

auto build_opgraph(
        const cudnnHandle_t& handle, const cudnnBackendDescriptorType_t operation,
        const cudnn_frontend::Tensor& x, const cudnn_frontend::Tensor& y,
        const cudnn_frontend::Tensor& w, const cudnn_frontend::ConvDesc_v8& conv_desc) {
    auto op = cudnn_frontend::OperationBuilder(operation)
                      .setxDesc(x)
                      .setyDesc(y)
                      .setwDesc(w)
                      .setcDesc(conv_desc)
                      .build();
    std::array<cudnn_frontend::Operation const*, 1> ops = {&op};
    auto op_graph = cudnn_frontend::OperationGraphBuilder()
                            .setHandle(handle)
                            .setOperationGraph(1, ops.data())
                            .build();
    return op_graph;
}

auto build_opgraph_fused(
        const cudnnHandle_t& handle, const cudnn_frontend::Tensor& x,
        const cudnn_frontend::Tensor& y, const cudnn_frontend::Tensor& w,
        const cudnn_frontend::Tensor& b, const cudnn_frontend::Tensor& z,
        const cudnn_frontend::Tensor& after_add,
        const cudnn_frontend::Tensor& after_bias,
        const cudnn_frontend::Tensor& after_conv,
        const cudnn_frontend::ConvDesc_v8& conv_desc,
        const cudnn_frontend::PointWiseDesc_v8& act_desc, float alpha, float beta) {
    const auto precision = CUDNN_DATA_FLOAT;

    // add z
    auto add_desc1 = cudnn_frontend::PointWiseDescBuilder()
                             .setMode(CUDNN_POINTWISE_ADD)
                             .setMathPrecision(precision)
                             .build();
    // add bias
    auto add_desc2 = cudnn_frontend::PointWiseDescBuilder()
                             .setMode(CUDNN_POINTWISE_ADD)
                             .setMathPrecision(precision)
                             .build();

    // create conv node
    auto conv_op = cudnn_frontend::OperationBuilder(
                           CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(x)
                           .setyDesc(after_conv)
                           .setwDesc(w)
                           .setcDesc(conv_desc)
                           .build();

    // create add z node
    auto add_op1 = cudnn_frontend::OperationBuilder(
                           CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(conv_op.getOutputTensor())
                           .setbDesc(z)
                           .setyDesc(after_add)
                           .setpwDesc(add_desc1)
                           .setAlpha(alpha)
                           .setAlpha2(beta)
                           .build();

    // create add bias node
    auto add_op2 = cudnn_frontend::OperationBuilder(
                           CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(add_op1.getOutputTensor())
                           .setbDesc(b)
                           .setyDesc(after_bias)
                           .setpwDesc(add_desc2)
                           .build();

    // create act node
    auto act_op = cudnn_frontend::OperationBuilder(
                          CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(add_op2.getOutputTensor())
                          .setyDesc(y)
                          .setpwDesc(act_desc)
                          .build();

    std::array<cudnn_frontend::Operation const*, 4> ops = {
            &conv_op, &add_op1, &add_op2, &act_op};

    auto op_graph = cudnn_frontend::OperationGraphBuilder()
                            .setHandle(handle)
                            .setOperationGraph(ops.size(), ops.data())
                            .build();
    return op_graph;
}

auto build_opgraph_fused_nonactivation(
        const cudnnHandle_t& handle, const cudnn_frontend::Tensor& x,
        const cudnn_frontend::Tensor& y, const cudnn_frontend::Tensor& w,
        const cudnn_frontend::Tensor& b, const cudnn_frontend::Tensor& z,
        const cudnn_frontend::Tensor& after_add,
        const cudnn_frontend::Tensor& after_conv,
        const cudnn_frontend::ConvDesc_v8& conv_desc, float alpha, float beta) {
    const auto precision = CUDNN_DATA_FLOAT;

    // add z
    auto add_desc1 = cudnn_frontend::PointWiseDescBuilder()
                             .setMode(CUDNN_POINTWISE_ADD)
                             .setMathPrecision(precision)
                             .build();
    // add bias
    auto add_desc2 = cudnn_frontend::PointWiseDescBuilder()
                             .setMode(CUDNN_POINTWISE_ADD)
                             .setMathPrecision(precision)
                             .build();

    // create conv node
    auto conv_op = cudnn_frontend::OperationBuilder(
                           CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(x)
                           .setyDesc(after_conv)
                           .setwDesc(w)
                           .setcDesc(conv_desc)
                           .build();

    // create add z node
    auto add_op1 = cudnn_frontend::OperationBuilder(
                           CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(conv_op.getOutputTensor())
                           .setbDesc(z)
                           .setyDesc(after_add)
                           .setpwDesc(add_desc1)
                           .setAlpha(alpha)
                           .setAlpha2(beta)
                           .build();

    // create add bias node
    auto add_op2 = cudnn_frontend::OperationBuilder(
                           CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(add_op1.getOutputTensor())
                           .setbDesc(b)
                           .setyDesc(y)
                           .setpwDesc(add_desc2)
                           .build();

    std::array<cudnn_frontend::Operation const*, 3> ops = {
            &conv_op, &add_op1, &add_op2};

    auto op_graph = cudnn_frontend::OperationGraphBuilder()
                            .setHandle(handle)
                            .setOperationGraph(ops.size(), ops.data())
                            .build();
    return op_graph;
}

void filter_engine_configs(
        cudnn_frontend::EngineConfigList& from, cudnn_frontend::EngineConfigList& to,
        bool deterministic) {
    auto filter = [&deterministic](cudnnBackendDescriptor_t c) {
        if (deterministic) {
            if (cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(
                        c)) {
                return true;
            }
        }
        if (cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(
                    c)) {
            return true;
        }
        return false;
    };
    cudnn_frontend::filter(from, to, filter);
}
};  // namespace

/* --------- get heuristic plan from megdnn opr -------- */
template <typename Opr>
cudnn_frontend::ExecutionPlan* megdnn::cuda::get_heuristic_plan_from_opr(
        const Opr* opr, const TensorLayout& x, const TensorLayout& y,
        const TensorLayout& w, const TensorLayout& b, const TensorLayout& z,
        const typename Opr::CanonizedFilterMeta& fm) {
    auto&& param = opr->param();
    TensorLayoutArray layouts{x, y, w};
    auto key = StaticData::Key{opr->handle(),  opr->get_opr_type(),
                               layouts.data(), layouts.size(),
                               &param,         sizeof(param)}
                       .build_key_storage();
    auto& cache = static_data().cache;
    {
        MEGDNN_LOCK_GUARD(static_data().cache_mutex);
        auto iter = cache.find(key);
        if (iter != cache.end()) {
            return &iter->second;
        }
    }

    size_t aligned = 16;
    uint8_t alignment = std::min(opr->handle()->alignment_requirement(), aligned);
    auto&& handle = cudnn_handle(opr->handle());
    auto&& x_desc = make_tensor_descriptor('x', alignment, x, fm.format, fm.group);
    auto&& y_desc = make_tensor_descriptor('y', alignment, y, fm.format, fm.group);
    auto&& w_desc = make_filter_descriptor(alignment, fm);
    auto compute_type = get_compute_type(x.dtype, param.compute_mode);
    auto&& conv_desc = make_conv_descriptor(compute_type, param);
    constexpr auto operation = CudnnBackendOpTypeTrait<Opr>::OPERATION;
    auto op_graph = build_opgraph(handle, operation, x_desc, y_desc, w_desc, conv_desc);
    auto deterministic = static_data().deterministic;
    auto heur_mode = static_data().heur_mode;
    auto heurgen_method = [&deterministic,
                           &heur_mode](cudnn_frontend::OperationGraph& op_graph)
            -> cudnn_frontend::EngineConfigList {
        auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                                  .setOperationGraph(op_graph)
                                  .setHeurMode(heur_mode)
                                  .build();
        auto& engine_configs =
                heuristics.getEngineConfig(heuristics.getEngineConfigCount());
        cudnn_frontend::EngineConfigList filtered_configs;
        filter_engine_configs(engine_configs, filtered_configs, deterministic);
        return filtered_configs;
    };

    auto fallback_method = [&deterministic, &heur_mode,
                            &operation](cudnn_frontend::OperationGraph& op_graph)
            -> cudnn_frontend::EngineConfigList {
        auto fallback = cudnn_frontend::EngineFallbackListBuilder()
                                .setOperationGraph(op_graph)
                                .setOperation(operation)
                                .build();
        auto& fallback_list = fallback.getFallbackList();
        cudnn_frontend::EngineConfigList filtered_configs;
        filter_engine_configs(fallback_list, filtered_configs, deterministic);
        return filtered_configs;
    };

    std::array<cudnn_frontend::GeneratorSource const, 2> sources = {
            heurgen_method, fallback_method};

    cudnn_frontend::EngineConfigGenerator generator(sources.size(), sources.data());
    auto configs = generator.generate_engine_config(op_graph);

    for (auto& config : configs) {
        try {
            auto plan = cudnn_frontend::ExecutionPlanBuilder()
                                .setHandle(handle)
                                .setEngineConfig(config)
                                .build();
            auto workspace_size = plan.getWorkspaceSize();
            MEGDNN_MARK_USED_VAR(workspace_size);
            MEGDNN_LOCK_GUARD(static_data().cache_mutex);
            auto insert = cache.insert(std::make_pair(key, std::move(plan)));
            return &insert.first->second;
        } catch (cudnn_frontend::cudnnException& e) {
            continue;
        }
    }
    return nullptr;
}

#define INST(_Opr)                                                                     \
    template cudnn_frontend::ExecutionPlan* megdnn::cuda::get_heuristic_plan_from_opr( \
            const _Opr* opr, const TensorLayout& x, const TensorLayout& y,             \
            const TensorLayout& w, const TensorLayout& b, const TensorLayout& z,       \
            const typename _Opr::CanonizedFilterMeta& fm);

INST(ConvolutionForward);
INST(ConvolutionBackwardData);
INST(ConvolutionBackwardFilter);

/* --------- get heuristic plan from conv_bias opr -------- */
template <>
cudnn_frontend::ExecutionPlan* megdnn::cuda::get_heuristic_plan_from_opr(
        const ConvBiasForward* opr, const TensorLayout& x, const TensorLayout& y,
        const TensorLayout& w, const TensorLayout& b, const TensorLayout& z,
        const typename ConvBiasForward::CanonizedFilterMeta& fm) {
    auto&& param = opr->param();
    TensorLayoutArray layouts{x, y, w, b, z};
    auto key = StaticData::Key{opr->handle(),  opr->get_opr_type(),
                               layouts.data(), layouts.size(),
                               &param,         sizeof(param)}
                       .build_key_storage();
    auto& cache = static_data().cache;
    {
        MEGDNN_LOCK_GUARD(static_data().cache_mutex);
        auto iter = cache.find(key);
        if (iter != cache.end()) {
            return &iter->second;
        }
    }

    size_t aligned = 16;
    uint8_t alignment = std::min(opr->handle()->alignment_requirement(), aligned);
    auto&& handle = cudnn_handle(opr->handle());
    auto&& x_desc = make_tensor_descriptor('x', alignment, x, fm.format, fm.group);
    auto&& y_desc = make_tensor_descriptor('y', alignment, y, fm.format, fm.group);
    auto&& w_desc = make_filter_descriptor(alignment, fm);
    auto&& z_desc = make_tensor_descriptor('z', alignment, y, fm.format, fm.group);
    auto&& b_desc = make_tensor_descriptor('b', alignment, b, Format::NCHW, fm.group);
    auto&& after_conv =
            make_tensor_descriptor('C', alignment, y, fm.format, fm.group, true);
    auto&& after_add =
            make_tensor_descriptor('A', alignment, y, fm.format, fm.group, true);
    auto&& after_bias =
            make_tensor_descriptor('B', alignment, y, fm.format, fm.group, true);
    auto compute_type = get_compute_type(x.dtype, param.compute_mode);
    auto&& conv_desc = make_conv_descriptor(compute_type, param);
    float alpha, beta;
    std::tie(alpha, beta) =
            conv_bias::cudnn_get_conv_bias_act_scale_param(x, y, w, b, z);
    // Because the OperationGraph has no public copy constructor and default
    // constructor, here we use a lambda function to bypass the compile error.
    auto get_op_graph = [&]() {
        if (param.nonlineMode == param::ConvBias::NonlineMode::IDENTITY) {
            return build_opgraph_fused_nonactivation(
                    handle, x_desc, y_desc, w_desc, b_desc, z_desc, after_add,
                    after_conv, conv_desc, alpha, beta);
        } else {
            auto&& act_desc =
                    make_activation_descriptor(dtype::Float32(), param.nonlineMode);
            return build_opgraph_fused(
                    handle, x_desc, y_desc, w_desc, b_desc, z_desc, after_add,
                    after_bias, after_conv, conv_desc, act_desc, alpha, beta);
        }
    };
    auto op_graph = get_op_graph();
    auto deterministic = static_data().deterministic;
    auto heur_mode = static_data().heur_mode;
    auto heurgen_method = [&deterministic,
                           &heur_mode](cudnn_frontend::OperationGraph& op_graph)
            -> cudnn_frontend::EngineConfigList {
        auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                                  .setOperationGraph(op_graph)
                                  .setHeurMode(heur_mode)
                                  .build();
        auto& engine_configs =
                heuristics.getEngineConfig(heuristics.getEngineConfigCount());
        cudnn_frontend::EngineConfigList filtered_configs;
        filter_engine_configs(engine_configs, filtered_configs, deterministic);
        return filtered_configs;
    };

    std::array<cudnn_frontend::GeneratorSource const, 1> sources = {heurgen_method};

    cudnn_frontend::EngineConfigGenerator generator(sources.size(), sources.data());
    auto configs = generator.generate_engine_config(op_graph);

    for (auto& config : configs) {
        try {
            auto plan = cudnn_frontend::ExecutionPlanBuilder()
                                .setHandle(handle)
                                .setEngineConfig(config)
                                .build();
            auto workspace_size = plan.getWorkspaceSize();
            MEGDNN_MARK_USED_VAR(workspace_size);
            MEGDNN_LOCK_GUARD(static_data().cache_mutex);
            auto insert = cache.insert(std::make_pair(key, std::move(plan)));
            return &insert.first->second;
        } catch (cudnn_frontend::cudnnException& e) {
            continue;
        }
    }
    return nullptr;
}

/* ------ impl for running a single conv ----- */
void megdnn::cuda::run_single_conv_with_plan(
        const cudnnHandle_t& handle, const cudnn_frontend::ExecutionPlan& plan,
        const TensorND& x, const TensorND& y, const TensorND& w,
        const Workspace& workspace) {
    size_t workspace_size = plan.getWorkspaceSize();
    megdnn_assert(
            workspace.size >= workspace_size,
            "workspace does not meet the requirement of execution "
            "plan(got:%zu,expected:%zu)",
            workspace.size, workspace_size);
    void* data_ptrs[] = {x.raw_ptr(), y.raw_ptr(), w.raw_ptr()};
    int64_t uids[] = {'x', 'y', 'w'};
    auto variant_pack = cudnn_frontend::VariantPackBuilder()
                                .setWorkspacePointer(workspace.raw_ptr)
                                .setDataPointers(3, data_ptrs)
                                .setUids(3, uids)
                                .build();
    cudnn_check(cudnnBackendExecute(
            handle, plan.get_raw_desc(), variant_pack.get_raw_desc()));
}

/* ------ impl for running a fused conv bias activation ----- */
void megdnn::cuda::run_conv_bias_act_with_plan(
        const cudnnHandle_t& handle, const cudnn_frontend::ExecutionPlan& plan,
        const TensorND& x, const TensorND& y, const TensorND& w, const TensorND& b,
        const TensorND& z, const Workspace& workspace) {
    size_t workspace_size = plan.getWorkspaceSize();
    megdnn_assert(
            workspace.size >= workspace_size,
            "workspace does not meet the requirement of execution "
            "plan(got:%zu,expected:%zu)",
            workspace.size, workspace_size);
    void* z_ptr = z.layout.ndim == 0 ? nullptr : z.raw_ptr();
    void* data_ptrs[] = {x.raw_ptr(), y.raw_ptr(), w.raw_ptr(), z_ptr, b.raw_ptr()};
    int64_t uids[] = {'x', 'y', 'w', 'z', 'b'};
    auto variant_pack = cudnn_frontend::VariantPackBuilder()
                                .setWorkspacePointer(workspace.raw_ptr)
                                .setDataPointers(5, data_ptrs)
                                .setUids(5, uids)
                                .build();
    cudnn_check(cudnnBackendExecute(
            handle, plan.get_raw_desc(), variant_pack.get_raw_desc()));
}

#endif
// vim: syntax=cpp.doxygen
