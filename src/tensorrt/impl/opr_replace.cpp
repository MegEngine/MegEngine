/**
 * \file src/tensorrt/impl/opr_replace.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <cstring>
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/utils/arith_helper.h"
#include "megbrain/opr/nn_int.h"
#include "megbrain/dtype.h"

#if MGB_ENABLE_TENSOR_RT
#include "megbrain/tensorrt/opr_replace.h"
#include "megbrain/tensorrt/tensorrt_opr.h"
#include "megbrain/tensorrt/tensorrt_engine_cache.h"
#include "megbrain/gopt/basic_arith.h"
#include "megbrain/gopt/inference.h"
#include "megbrain/gopt/misc.h"

using namespace mgb;
using namespace gopt;
using namespace cg;

template <typename T>
using TensorRTUniquePtr = opr::intl::TensorRTUniquePtr<T>;

namespace {
nvinfer1::DataType mgb_dtype_to_trt_dtype(DType dtype) {
    switch (dtype.enumv()) {
        case DTypeEnum::Float32:
            return nvinfer1::DataType::kFLOAT;
        case DTypeEnum::Float16:
            return nvinfer1::DataType::kHALF;
        case DTypeEnum::QuantizedS8:
            return nvinfer1::DataType::kINT8;
        case DTypeEnum::Int32:
            return nvinfer1::DataType::kINT32;
        default:
            mgb_throw(
                    InternalError,
                    "invalid data type which is not supported in TensorRT: %s",
                    dtype.name());
    }
}
}

class TensorRTReplacePass::Impl final {
    static constexpr size_t OPR_FAIL_LOG_NUM = 10;
    static constexpr float i8_max = std::numeric_limits<int8_t>::max();
    using TensorRTGraphFeatureBits = opr::intl::TensorRTGraphFeatureBits;
    using ConvFormat = opr::Convolution::Param::Format;
    using ExtraDep = ThinHashMap<OperatorNodeBase*, VarNodeArray>;

    const Pass& m_pass;
    OptState& m_opt_state;
    SubGraph::Rewriter m_rewriter;

    struct TensorRTGraph {
        using Callback = cg::DepOprIter::Callback;
        nvinfer1::IBuilder* builder;
        nvinfer1::INetworkDefinition* network;
        ThinHashSet<VarNode*> inputs;
        ThinHashSet<VarNode*> outputs;
        // is used for mapping output varnode in original computing graph to
        // output varnode of TensorRTOpr
        ThinHashMap<VarNode*, size_t> output2idx;
        // mark input and output tensor as nchw4 format, we should insert
        // dimshuffle and typecvt to make the TensorRTOpr's inputs and outputs
        // match with those of non fused operators.
        ThinHashSet<VarNode*> mark_input_varnode_nchw4;
        ThinHashSet<VarNode*> mark_output_varnode_nchw4;
        VarNodeArray trt_inputs;
        VarNodeArray trt_outputs;
        // Every tensor rt graph should own a map from var node to infer tensor.
        // Because a var node can belong to two different tensor rt subgraph
        ThinHashMap<VarNode*, nvinfer1::ITensor*> varnode2itensor;
        TensorRTGraphFeatureBits feature_bits;
        TensorRTGraph(TensorRTGraphFeatureBits feature_bits =
                              TensorRTGraphFeatureBits::NCHW_FLOAT)
                : builder{nvinfer1::createInferBuilder(
                          opr::TensorRTOpr::Logger::instance())},
                  network{nullptr},
                  feature_bits{feature_bits} {}
        void mark_varnode_format_nchw4();
    };

    struct FailInfo {
        OperatorNodeBase* opr;
        std::string fail_msg;
    };

    class HostTensorKeeper : public UserDataContainer::UserData {
        MGB_TYPEINFO_OBJ_DECL;

    public:
        std::vector<HostTensorND> htr;
    };

    std::unique_ptr<ConstVarPropogate> m_const_var_propogate;
    std::vector<std::shared_ptr<TensorRTGraph>> m_tensorrt_graphs;
    // use ThinHashMap instead of std::unordered_map
    ThinHashMap<OperatorNodeBase*, size_t> m_graph_map;
    ThinHashMap<OperatorNodeBase*, nvinfer1::IConvolutionLayer*>
            m_opr2convlayer;
    ThinHashMap<OperatorNodeBase*, nvinfer1::IDeconvolutionLayer*>
            m_opr2deconvlayer;

    size_t m_opr_num;
    size_t m_opr_fail_num;
    std::vector<FailInfo> m_opr_fail;

    struct OprTrait {
        // judge if supported, not exist means not support
        thin_function<Maybe<std::string>(OperatorNodeBase*)>
                get_replace_fail_msg;
        // replace opr by trt opr, ditto
        thin_function<void(nvinfer1::INetworkDefinition*, OperatorNodeBase*)>
                add_to_nvinfer;
    };
    ThinHashMap<Typeinfo*, OprTrait> m_opr_trait;
    // Find parent conv of elemwise ADD opr.
    VarNodeArray find_parent_conv(OperatorNodeBase* opr);
    // Make a trt tensor for Varnode var and add it as input of trt buffer.
    // Return false if a tensor of var is previously made and added.
    // True if var is encountered for the first time.
    bool check_input(VarNode* var, OperatorNodeBase* opr,
                     mgb::SmallVector<nvinfer1::DimensionType> dimtypes = {});
    HostTensorND get_value(VarNode* var, ConvFormat format = ConvFormat::NCHW);
    void set_itensor_dynamic_range(VarNode* var, OperatorNodeBase* opr);
    float get_scale(DType data_type);
    // Check whether an operator is a quantized operator. If an operator is a
    // quantized operator, this operator can be fused into a quantized TensorRT
    // subgraph
    bool is_quantized_int8_operator(OperatorNodeBase* opr);
    Maybe<std::string> has_fail_msg(OperatorNodeBase* opr);
    static nvinfer1::ITensor& replace(nvinfer1::INetworkDefinition* newtwork,
                                      nvinfer1::ITensor& pre_output,
                                      OperatorNodeBase* opr);
    void update_graph();
    void mark_varnode_format_nchw4();
    void detect_replace();

public:
    Impl(const Pass& pass, OptState& opt_state)
            : m_pass{pass},
              m_opt_state{opt_state},
              m_rewriter{opt_state.graph().make_rewriter()},
              m_const_var_propogate{std::make_unique<ConstVarPropogate>(
                      ConstVarType::IMMUTABLE_AND_PARAM)} {
#define REPLACE_FAIL_MSG_EPILOGUE                                       \
    {                                                                   \
        auto&& mgr = opr->owner_graph()->static_infer_manager();        \
        auto&& shp = mgr.infer_shape_fallible(opr->output(0));          \
        if (!shp)                                                       \
            return "Unsupported opr, because operator shape cannot be " \
                   "inferred at compile time.";                         \
        else                                                            \
            return None;                                                \
    }
        m_opr_trait[opr::Elemwise::typeinfo()].get_replace_fail_msg =
                [](OperatorNodeBase* opr) -> Maybe<std::string> {
            bool has_scalar = false;
            for (auto&& inp : opr->input()) {
                if (inp->shape().is_scalar()) {
                    has_scalar = true;
                    break;
                }
            }
            if (has_scalar)
                return "Elemwise with scalar input is not supported.";
            if (opr->input(0)->dtype().enumv() != DTypeEnum::QuantizedS8 &&
                opr->input(0)->dtype() != dtype::Float32()) {
                return "Unsupported data type.";
            }
            using Mode = opr::Elemwise::Mode;
            static const ThinHashSet<Mode> supported_modes {
#if NV_TENSOR_RT_VERSION >= 5105
                Mode::SIN, Mode::COS, Mode::ASIN, Mode::ACOS, Mode::CEIL,
                        Mode::FLOOR,
#endif
                        Mode::EXP, Mode::LOG, Mode::ABS,

                        Mode::RELU, Mode::SIGMOID, Mode::TANH, Mode::ADD,
                        Mode::MUL, Mode::MIN, Mode::MAX, Mode::SUB,
                        Mode::TRUE_DIV, Mode::POW, Mode::FUSE_ADD_RELU,
                        Mode::FUSE_ADD_TANH, Mode::FUSE_ADD_SIGMOID
            };
            auto mode = opr->cast_final_safe<opr::Elemwise>().param().mode;
            if (!supported_modes.count(mode)) {
                return "Unsupported Elemwise mode.";
            }
#if NV_TENSOR_RT_VERSION >= 6001
            if (opr->input(0)->dtype().enumv() == DTypeEnum::QuantizedS8) {
                TensorShapeArray inps;
                for (auto&& inp : opr->input()) {
                    inps.push_back(inp->shape());
                }
                TensorShape brdcast;
                megdnn::Elemwise::deduce_shape(inps, brdcast);
                if (brdcast.ndim < 4) {
                    return "Elemwise with QuantizedS8 data type must have more "
                           "than 4 dimensions. Less than 3 dimensions is not "
                           "supported since trt6.0.";
                }
            }
#endif
            REPLACE_FAIL_MSG_EPILOGUE;
        };

        m_opr_trait[opr::ElemwiseMultiType::typeinfo()].get_replace_fail_msg =
                [](OperatorNodeBase* opr) -> Maybe<std::string> {
            bool has_scalar = false;
            for (auto&& inp : opr->input()) {
                if (inp->shape().is_scalar()) {
                    has_scalar = true;
                    break;
                }
            }
            if (has_scalar)
                return "ElemwiseMultiType with scalar input is not supported.";

            for (auto&& inp : opr->input()) {
                if (inp->dtype().enumv() != DTypeEnum::QuantizedS8)
                    return "Unsupported data type.";
            }
            if (opr->output(0)->dtype().enumv() != DTypeEnum::QuantizedS8)
                return "Unsupported data type.";
            using Mode = opr::ElemwiseMultiType::Mode;
            auto mode =
                    opr->cast_final_safe<opr::ElemwiseMultiType>().param().mode;
            if (mode != Mode::QFUSE_ADD_RELU && mode != Mode::QADD) {
                return "Unsupported ElemwiseMultiType mode.";
            }
            REPLACE_FAIL_MSG_EPILOGUE;
        };

        m_opr_trait[opr::Convolution::typeinfo()].get_replace_fail_msg =
                [this](OperatorNodeBase* opr) -> Maybe<std::string> {
            if (opr->input(0)->dtype() != dtype::Float32())
                return "Non-Float32 convolution is not supported.";
            if (!m_const_var_propogate->is_const(opr->input(1)))
                return "Weights not constant. Not replaceable in TRT.";
            auto&& param = opr->cast_final_safe<opr::Convolution>().param();
            if (param.format != ConvFormat::NCHW)
                return "TensorRT replace pass only support NCHW format "
                       "convolution.";
            if (param.mode == opr::Convolution::Param::Mode::CONVOLUTION)
                return "TensorRT does not support non cross correlation "
                       "convolution.";
            REPLACE_FAIL_MSG_EPILOGUE;
        };

        m_opr_trait[opr::ConvBias::typeinfo()].get_replace_fail_msg =
                [this](OperatorNodeBase* opr) -> Maybe<std::string> {
            if (opr->input(0)->dtype() != dtype::Float32() &&
                opr->input(0)->dtype().enumv() != DTypeEnum::QuantizedS8)
                return "Convolution is only supported for float32 or qint8.";
            if (!m_const_var_propogate->is_const(opr->input(1)))
                return "Weights not constant. Not replaceable in TRT.";
            if (opr->input().size() >= 3) {
                if (!m_const_var_propogate->is_const(opr->input(2)))
                    return "Bias not constant. Not replaceable in TRT.";
            }
            auto&& param = opr->cast_final_safe<opr::ConvBias>().param();
            if (param.format != ConvFormat::NCHW &&
                param.format != ConvFormat::NCHW4)
                return "TensorRT replace pass only support NCHW format "
                       "convolution.";
            if (param.mode == opr::ConvBias::Param::Mode::CONVOLUTION)
                return "TensorRT does not support non cross correlation "
                       "convolution.";
            REPLACE_FAIL_MSG_EPILOGUE;
        };

        m_opr_trait[opr::ConvolutionBackwardData::typeinfo()]
                .get_replace_fail_msg =
                [this](OperatorNodeBase* opr) -> Maybe<std::string> {
            if (opr->input(0)->dtype() != dtype::Float32())
                return "Non-Float32 Deconvolution is not supported.";
            if (!m_const_var_propogate->is_const(opr->input(0)))
                return "Weights not constant. Not replaceable in TRT.";
            auto&& param = opr->cast_final_safe<opr::ConvolutionBackwardData>().param();
            if (param.dilate_h != 1 || param.dilate_w != 1)
                return "TensorRT does not support dilation deconvolution.";
            if (param.format != ConvFormat::NCHW)
                return "TensorRT replace pass only support NCHW format deconv.";
            if (param.mode == opr::ConvBias::Param::Mode::CONVOLUTION)
                return "TensorRT does not support non cross correlation "
                       "deconvolution.";
            REPLACE_FAIL_MSG_EPILOGUE;
        };

        m_opr_trait[opr::Pooling::typeinfo()].get_replace_fail_msg =
                [](OperatorNodeBase* opr) -> Maybe<std::string> {
            auto pool = opr->try_cast_final<opr::Pooling>();
            auto&& param = pool->param();
            if (param.format != opr::Pooling::Param::Format::NCHW &&
                param.format != opr::Pooling::Param::Format::NCHW4)
                return "Pooling is only supported for NCHW and NCHW4";
            REPLACE_FAIL_MSG_EPILOGUE;
        };

        m_opr_trait[opr::Concat::typeinfo()].get_replace_fail_msg =
                [](OperatorNodeBase* opr) -> Maybe<std::string> {
            if (opr->input(0)->dtype() != dtype::Float32() &&
                opr->input(0)->dtype().enumv() != DTypeEnum::QuantizedS8) {
                return "Concat only support float32 and quantized int8.";
            }
            // TODO: TensorRT only supports concat on channel dimension,
            // we can set nvinfer1::DimensionType to kCHANNEL to support
            // concat on other dimension
            if (!(opr->input(0)->shape().ndim == 4 &&
                  opr->cast_final_safe<opr::Concat>().param().axis == 1)) {
                return "Concat only support input is NCHW and axis is 1.";
            }
            REPLACE_FAIL_MSG_EPILOGUE;
        };

        m_opr_trait[opr::MatrixMul::typeinfo()].get_replace_fail_msg =
                [](OperatorNodeBase* opr) -> Maybe<std::string> {
            if (opr->input(0)->dtype() != dtype::Float32())
                return "Non-Float32 MatrixMul is not supported.";
            REPLACE_FAIL_MSG_EPILOGUE;
        };

        m_opr_trait[opr::BatchedMatrixMul::typeinfo()].get_replace_fail_msg =
                [](OperatorNodeBase* opr) -> Maybe<std::string> {
            if (opr->input(0)->dtype() != dtype::Float32())
                return "Non-Float32 MatrixMul is not supported.";
            REPLACE_FAIL_MSG_EPILOGUE;
        };

        m_opr_trait[opr::PowC::typeinfo()].get_replace_fail_msg =
                [](OperatorNodeBase* opr) -> Maybe<std::string> {
            if (opr->input(0)->dtype() != dtype::Float32())
                return "Non-Float32 PowC is not supported.";
            if (opr->input(0)->shape().ndim < 3)
                return "Dimensions of input should be greater than or equal to "
                       "3.";
            REPLACE_FAIL_MSG_EPILOGUE;
        };
#undef REPLACE_FAIL_MSG_EPILOGUE

        // megdnn convolution opr on cuda backend does not support quantized
        // dtype, so we assume that megbrain int8 network for converting to fine
        // grained TensorRT subgraph does not include convolution operator with
        // quantized int8 data type
        m_opr_trait[opr::Convolution::typeinfo()]
                .add_to_nvinfer = [this](nvinfer1::INetworkDefinition* net,
                                         OperatorNodeBase* opr) {
            auto&& varnode2itensor =
                    m_tensorrt_graphs[m_graph_map[opr] - 1]->varnode2itensor;
            VarNode* input = opr->input(0);
            VarNode* kernel = opr->input(1);
            check_input(input, opr);
            nvinfer1::Weights wt_kernel{
                    nvinfer1::DataType::kFLOAT, get_value(kernel).raw_ptr(),
                    static_cast<int64_t>(kernel->shape().total_nr_elems())};
            nvinfer1::Weights wt_bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
            auto&& param = opr->cast_final_safe<opr::Convolution>().param();
            mgb_assert(
                    param.format == megdnn::param::Convolution::Format::NCHW &&
                            param.mode == megdnn::param::Convolution::Mode::
                                                  CROSS_CORRELATION,
                    "conv param is not supported by TensorRT");
            size_t group_offset = 0;
            if (param.sparse == megdnn::param::Convolution::Sparse::GROUP) {
                group_offset = 1;
            } else {
                mgb_assert(param.sparse ==
                                   megdnn::param::Convolution::Sparse::DENSE,
                           "param.sparse should be GROUP or DENSE");
            }
            auto conv = net->addConvolution(
                    *varnode2itensor[input], opr->output(0)->shape()[1],
                    nvinfer1::DimsHW{
                            static_cast<int>(kernel->shape()[group_offset + 2]),
                            static_cast<int>(
                                    kernel->shape()[group_offset + 3])},
                    wt_kernel, wt_bias);
            mgb_assert(conv, "construct network failed");
            std::string layer_name = "TRT_CONV:" + opr->name();
            conv->setName(layer_name.c_str());
            conv->setStride(nvinfer1::DimsHW{static_cast<int>(param.stride_h),
                                             static_cast<int>(param.stride_w)});
            conv->setPadding(nvinfer1::DimsHW{static_cast<int>(param.pad_h),
                                              static_cast<int>(param.pad_w)});
            conv->setDilation(
                    nvinfer1::DimsHW{static_cast<int>(param.dilate_h),
                                     static_cast<int>(param.dilate_w)});
            if (group_offset > 0)
                conv->setNbGroups(static_cast<int>(kernel->shape()[0]));
            m_opr2convlayer[opr] = conv;
            std::string output_name = "TRT_O:" + opr->output()[0]->name();
            conv->getOutput(0)->setName(output_name.c_str());
            varnode2itensor[opr->output(0)] = conv->getOutput(0);
        };

        // support floating point data type and quantized data type
        m_opr_trait[opr::ConvBiasForward::typeinfo()]
                .add_to_nvinfer = [this](nvinfer1::INetworkDefinition* net,
                                         OperatorNodeBase* opr) {
            auto&& varnode2itensor =
                    m_tensorrt_graphs[m_graph_map[opr] - 1]->varnode2itensor;
            using Param = opr::ConvBias::Param;
            using NonlineMode = Param::NonlineMode;
            using Sparse = Param::Sparse;
            using Format = Param::Format;
            auto conv_bias = try_cast_as_op<opr::ConvBias>(opr);
            auto&& param = conv_bias->param();
            mgb_assert(param.mode == Param::Mode::CROSS_CORRELATION,
                       "Trt only support CROSS_CORRELATION convolution.");
            bool is_format_nchw4 = param.format == Format::NCHW4;
            bool is_qint8 = is_quantized_int8_operator(opr);
            if (is_format_nchw4)
                mgb_assert(is_qint8);
            // set kernel and bias
            VarNode* input = conv_bias->input(0);
            VarNode* kernel = conv_bias->input(1);
            check_input(input, opr);
            nvinfer1::Weights wt_kernel{
                    nvinfer1::DataType::kFLOAT,
                    get_value(kernel, param.format).raw_ptr(),
                    static_cast<int64_t>(kernel->shape().total_nr_elems())};
            nvinfer1::Weights wt_bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
            if (conv_bias->input().size() >= 3) {
                VarNode* bias = conv_bias->input(2);
                wt_bias.values = get_value(bias, param.format).raw_ptr();
                wt_bias.count =
                        static_cast<int64_t>(bias->shape().total_nr_elems());
            }

            // determine conv shape
            int co = 0;
            int sh = param.stride_h, sw = param.stride_w, ph = param.pad_h,
                pw = param.pad_w, dh = param.dilate_h, dw = param.dilate_w;
            size_t group_offset = 0;
            int groups = 1;
            if (param.sparse == Sparse::GROUP) {
                groups = kernel->shape()[0];
                group_offset = 1;
            } else {
                mgb_assert(param.sparse == Sparse::DENSE,
                           "sparse should be GROUP or DENSE");
            }
            int fh = kernel->shape()[group_offset + 2],
                fw = kernel->shape()[group_offset + 3];
            if (param.format == Format::NCHW) {
                mgb_assert(conv_bias->input(0)->dtype() == dtype::Float32(),
                           "conv bias only support Float32 with NCHW format");
                co = conv_bias->output(0)->shape()[1];
            } else if (param.format == Format::NCHW4) {
                mgb_assert(
                        conv_bias->input(0)->dtype().enumv() ==
                                        DTypeEnum::QuantizedS8 &&
                                conv_bias->output(0)->dtype().enumv() ==
                                        DTypeEnum::QuantizedS8,
                        "conv bias only support QuantizedS8 with NCHW4 format");
                co = conv_bias->output(0)->shape()[1] * 4;
            }
            mgb_assert(co > 0);

            // process conv
            auto conv = net->addConvolution(*varnode2itensor[input], co,
                                            nvinfer1::DimsHW{fh, fw}, wt_kernel,
                                            wt_bias);
            mgb_assert(conv, "construct network failed");
            std::string layer_name = "TRT_CONV:" + conv_bias->name();
            conv->setName(layer_name.c_str());
            conv->setStride(nvinfer1::DimsHW{sh, sw});
            conv->setPadding(nvinfer1::DimsHW{ph, pw});
            conv->setDilation(nvinfer1::DimsHW{dh, dw});

            if (group_offset > 0)
                conv->setNbGroups(groups);
            std::string output_name = "TRT_O:" + conv_bias->output(0)->name();
            conv->getOutput(0)->setName(output_name.c_str());
            varnode2itensor[conv_bias->output(0)] = conv->getOutput(0);
            if (is_qint8)
                set_itensor_dynamic_range(conv_bias->output(0), conv_bias);

            // process short cut add
            if (conv_bias->input().size() >= 4) {
                check_input(conv_bias->input(3), opr);
                auto add = net->addElementWise(
                        *varnode2itensor[conv_bias->output(0)],
                        *varnode2itensor[conv_bias->input(3)],
                        nvinfer1::ElementWiseOperation::kSUM);
                mgb_assert(add, "construct network failed");
                std::string layer_name = "TRT_ELEM:" + conv_bias->name();
                add->setName(layer_name.c_str());
                std::string output_name =
                        "TRT_O:" + conv_bias->output(0)->name() +
                        "_shortcut_add";
                add->getOutput(0)->setName(output_name.c_str());
                varnode2itensor[conv_bias->output(0)] = add->getOutput(0);
                if (is_qint8)
                    set_itensor_dynamic_range(conv_bias->output(0), conv_bias);
            }

            // process activation
            if (param.nonlineMode != Param::NonlineMode::IDENTITY) {
                nvinfer1::ActivationType act_type =
                        param.nonlineMode == NonlineMode::RELU
                                ? nvinfer1::ActivationType::kRELU
                                : nvinfer1::ActivationType::kSIGMOID;
                auto act = net->addActivation(
                        *varnode2itensor[conv_bias->output(0)], act_type);
                mgb_assert(act, "construct network failed");
                std::string layer_name =
                        "TRT_ACTV:" + conv_bias->name();
                act->setName(layer_name.c_str());
                std::string output_name =
                        "TRT_O:" + conv_bias->output(0)->name() + "_act";
                act->getOutput(0)->setName(output_name.c_str());
                varnode2itensor[conv_bias->output(0)] = act->getOutput(0);
                if (is_qint8)
                    set_itensor_dynamic_range(conv_bias->output(0), conv_bias);

            }
        };

        // megbrain deconvolution operator does not support quantized data type
        m_opr_trait[opr::ConvolutionBackwardData::typeinfo()]
                .add_to_nvinfer = [this](nvinfer1::INetworkDefinition* net,
                                         OperatorNodeBase* opr) {
            auto&& varnode2itensor =
                    m_tensorrt_graphs[m_graph_map[opr] - 1]->varnode2itensor;
            VarNode* kernel = opr->input(0);
            VarNode* input = opr->input(1);
            check_input(input, opr);
            nvinfer1::Weights wt_kernel{
                    nvinfer1::DataType::kFLOAT, get_value(kernel).raw_ptr(),
                    static_cast<int64_t>(kernel->shape().total_nr_elems())};
            nvinfer1::Weights wt_bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
            auto&& param = opr->cast_final_safe<opr::ConvolutionBackwardData>()
                                   .param();
            mgb_assert(
                    param.format == megdnn::param::Convolution::Format::NCHW &&
                            param.mode == megdnn::param::Convolution::Mode::
                                                  CROSS_CORRELATION &&
                            param.dilate_h == 1 && param.dilate_w == 1,
                    "conv param is not supported by TensorRT");
            size_t group_offset = 0;
            if (param.sparse == megdnn::param::Convolution::Sparse::GROUP) {
                group_offset = 1;
            } else {
                mgb_assert(param.sparse ==
                                   megdnn::param::Convolution::Sparse::DENSE,
                           "param.sparse should be GROUP or DENSE");
            }

            auto deconv = net->addDeconvolution(
                    *varnode2itensor[input], opr->output(0)->shape()[1],
                    nvinfer1::DimsHW{
                            static_cast<int>(kernel->shape()[group_offset + 2]),
                            static_cast<int>(
                                    kernel->shape()[group_offset + 3])},
                    wt_kernel, wt_bias);
            mgb_assert(deconv, "construct network failed");
            std::string layer_name = "TRT_DCON:" + opr->name();
            deconv->setName(layer_name.c_str());
            deconv->setStride(
                    nvinfer1::DimsHW{static_cast<int>(param.stride_h),
                                     static_cast<int>(param.stride_w)});
            deconv->setPadding(nvinfer1::DimsHW{static_cast<int>(param.pad_h),
                                                static_cast<int>(param.pad_w)});

            if (group_offset > 0)
                deconv->setNbGroups(static_cast<int>(kernel->shape()[0]));
            m_opr2deconvlayer[opr] = deconv;
            std::string output_name = "TRT_O:" + opr->output()[0]->name();
            deconv->getOutput(0)->setName(output_name.c_str());
            varnode2itensor[opr->output(0)] = deconv->getOutput(0);
        };

        // support floating point data type and quantized data type
        m_opr_trait[opr::Pooling::typeinfo()]
                .add_to_nvinfer = [this](nvinfer1::INetworkDefinition* net,
                                         OperatorNodeBase* opr) {
            auto&& varnode2itensor =
                    m_tensorrt_graphs[m_graph_map[opr] - 1]->varnode2itensor;
            using Param = opr::Pooling::Param;
            using Mode = Param::Mode;
            using Format = Param::Format;
            static ThinHashMap<Mode, nvinfer1::PoolingType> pooling_type_map = {
                    {Mode::MAX, nvinfer1::PoolingType::kMAX},
                    {Mode::AVERAGE, nvinfer1::PoolingType::kAVERAGE},
                    {Mode::AVERAGE_COUNT_EXCLUDE_PADDING,
                     nvinfer1::PoolingType::kAVERAGE}};
            auto&& param = opr->cast_final_safe<opr::Pooling>().param();
            check_input(opr->input(0), opr);
            auto pool = net->addPooling(
                    *varnode2itensor[opr->input(0)],
                    pooling_type_map.at(param.mode),
                    nvinfer1::DimsHW{static_cast<int>(param.window_h),
                                     static_cast<int>(param.window_w)});
            mgb_assert(pool, "construct network failed");
            std::string layer_name = "TRT_POOL:" + opr->name();
            pool->setName(layer_name.c_str());
            pool->setPadding(nvinfer1::DimsHW{static_cast<int>(param.pad_h),
                                              static_cast<int>(param.pad_w)});
            pool->setStride(nvinfer1::DimsHW{static_cast<int>(param.stride_h),
                                             static_cast<int>(param.stride_w)});
            //! According to the documentation of TensorRT, the default value of exclusive is true.
            //! So we need to set exclusive to false when pooling mode is average
            if (param.mode == Mode::AVERAGE_COUNT_EXCLUDE_PADDING)
                pool->setAverageCountExcludesPadding(true);
            else if (param.mode == Mode::AVERAGE)
                pool->setAverageCountExcludesPadding(false);
            std::string output_name = "TRT_O:" + opr->output()[0]->name();
            pool->getOutput(0)->setName(output_name.c_str());
            varnode2itensor[opr->output(0)] = pool->getOutput(0);
            if (param.format == Format::NCHW4) {
                mgb_assert(opr->input(0)->dtype().enumv() ==
                                   DTypeEnum::QuantizedS8,
                           "Pooling with NCHW4 format should use quantized "
                           "int8 data type");
                set_itensor_dynamic_range(opr->output(0), opr);
            }
        };

        m_opr_trait[opr::Concat::typeinfo()].add_to_nvinfer =
                [this](nvinfer1::INetworkDefinition* net,
                       OperatorNodeBase* opr) {
                    auto&& varnode2itensor =
                            m_tensorrt_graphs[m_graph_map[opr] - 1]
                                    ->varnode2itensor;
                    size_t input_size = opr->input().size();
                    std::unique_ptr<nvinfer1::ITensor* []> input_tensors(
                            new nvinfer1::ITensor*[input_size]);
                    for (size_t i = 0; i < input_size; ++i) {
                        check_input(opr->input(i), opr);
                        input_tensors[i] = varnode2itensor[opr->input(i)];
                    }
                    auto concat = net->addConcatenation(
                            input_tensors.get(), static_cast<int>(input_size));
                    mgb_assert(concat, "construct Concatenation layer failed!");
                    std::string layer_name = "TRT_CCAT:" + opr->name();
                    concat->setName(layer_name.c_str());

                    int axis = opr->cast_final_safe<opr::Concat>().param().axis;
                    concat->setAxis(axis);
                    std::string output_name =
                            "TRT_O:" + opr->output()[0]->name();
                    concat->getOutput(0)->setName(output_name.c_str());
                    varnode2itensor[opr->output(0)] = concat->getOutput(0);
                    if (is_quantized_int8_operator(opr)) {
                        set_itensor_dynamic_range(opr->output(0), opr);
                    }
                };

        // support floating point data type and quantized data type
        m_opr_trait[opr::Elemwise::typeinfo()]
                .add_to_nvinfer = [this](nvinfer1::INetworkDefinition* net,
                                         OperatorNodeBase* opr) {
            auto&& varnode2itensor =
                    m_tensorrt_graphs[m_graph_map[opr] - 1]->varnode2itensor;
            using Mode = opr::Elemwise::Mode;
            auto mode = opr->cast_final_safe<opr::Elemwise>().param().mode;
            auto get_dimtype = [&](int ndim) {
                SmallVector<nvinfer1::DimensionType> dimtypes(ndim);
                for (int i = 0; i < ndim; i++) {
                    dimtypes[i] = nvinfer1::DimensionType::kSPATIAL;
                }
                return dimtypes;
            };
            auto on_elemwise_arity_unary =
                    [this, &varnode2itensor, &net, &opr,
                     &get_dimtype](nvinfer1::UnaryOperation unary_op) {
                        size_t tensor_ndim = opr->input(0)->shape().ndim;
                        check_input(opr->input(0), opr,
                                    get_dimtype(tensor_ndim));
                        auto unary = net->addUnary(
                                *varnode2itensor[opr->input(0)], unary_op);
                        mgb_assert(unary, "construct network failed");
                        std::string layer_name = "TRT_UNARY:" + opr->name();
                        unary->setName(layer_name.c_str());
                        std::string output_name =
                                "TRT_O:" + opr->output()[0]->name();
                        unary->getOutput(0)->setName(output_name.c_str());
                        varnode2itensor[opr->output(0)] = unary->getOutput(0);
                    };
            auto on_elemwise_arity_activation =
                    [this, &varnode2itensor, &net, &opr,
                     &get_dimtype](nvinfer1::ActivationType act_type) {
                        size_t tensor_ndim = opr->input(0)->shape().ndim;
                        check_input(opr->input(0), opr,
                                    get_dimtype(tensor_ndim));
                        auto act = net->addActivation(
                                *varnode2itensor[opr->input(0)], act_type);
                        mgb_assert(act, "construct network failed");
                        std::string layer_name = "TRT_ACTV:" + opr->name();
                        act->setName(layer_name.c_str());
                        std::string output_name =
                                "TRT_O:" + opr->output()[0]->name();
                        act->getOutput(0)->setName(output_name.c_str());
                        varnode2itensor[opr->output(0)] = act->getOutput(0);
                    };
            auto on_elemwise_arity_binary = [this, &varnode2itensor, &net, &opr,
                                             &get_dimtype](
                                                    nvinfer1::
                                                            ElementWiseOperation
                                                                    elem_op) {
                size_t ndim0 = opr->input(0)->shape().ndim,
                       ndim1 = opr->input(1)->shape().ndim;
                mgb_assert(ndim0 == ndim1);
                size_t tensor_ndim = ndim0;
                bool inp0_new = check_input(opr->input(0), opr,
                                            get_dimtype(tensor_ndim));
                bool inp1_new = check_input(opr->input(1), opr,
                                            get_dimtype(tensor_ndim));
                if (inp0_new && inp1_new) {
                    mgb_log_warn(
                            "Both operands of Elemwise are newly prepared. "
                            "This is rare. "
                            "Please check. opr=%s inputs=%s",
                            opr->cname(),
                            cg::dump_var_info(opr->input()).c_str());
                }
                auto dims0 = varnode2itensor[opr->input(0)]->getDimensions(),
                     dims1 = varnode2itensor[opr->input(1)]->getDimensions();
                mgb_throw_if(dims0.nbDims != dims1.nbDims, AssertionError,
                             "Input dimensions of two input tensors must be "
                             "equal (got: %d, %d).",
                             dims0.nbDims, dims1.nbDims);
                auto elem = net->addElementWise(*varnode2itensor[opr->input(0)],
                                                *varnode2itensor[opr->input(1)],
                                                elem_op);
                mgb_assert(elem, "construct network failed");
                std::string layer_name = "TRT_ELEM:" + opr->name();
                elem->setName(layer_name.c_str());
                std::string output_name = "TRT_O:" + opr->output()[0]->name();
                elem->getOutput(0)->setName(output_name.c_str());
                varnode2itensor[opr->output(0)] = elem->getOutput(0);
            };
            switch (mode) {
#define cb(mode)                                                    \
    case Mode::mode:                                                \
        on_elemwise_arity_unary(nvinfer1::UnaryOperation::k##mode); \
        break;
#if NV_TENSOR_RT_VERSION >= 5105
#define MGB_FOREACH_UNARY_OPERATION(cb) \
    cb(EXP) cb(LOG) cb(ABS) cb(SIN) cb(COS) cb(ASIN) cb(ACOS) cb(CEIL) cb(FLOOR)
#else
#define MGB_FOREACH_UNARY_OPERATION(cb) cb(EXP) cb(LOG) cb(ABS)
#endif
                MGB_FOREACH_UNARY_OPERATION(cb)
#undef cb
#undef MGB_FOREACH_UNARY_OPERATION
#define cb(mode)                                                         \
    case Mode::mode:                                                     \
        on_elemwise_arity_activation(nvinfer1::ActivationType::k##mode); \
        break;
#define MGB_FOREACH_ACTIVATION_TYPE(cb) cb(RELU) cb(SIGMOID) cb(TANH)
                MGB_FOREACH_ACTIVATION_TYPE(cb)
#undef cb
#undef MGB_FOREACH_ACTIVATION_TYPE
                case Mode::ADD: {
                    VarNode *opr_var, *bias_var;
                    VarNodeArray result = find_parent_conv(opr);
                    if (result.size() > 0) {
                        opr_var = result[0];
                        bias_var = result[1];
                        nvinfer1::Weights wt_bias{
                                nvinfer1::DataType::kFLOAT,
                                get_value(bias_var).raw_ptr(),
                                static_cast<int64_t>(
                                        bias_var->shape().total_nr_elems())};
                        if (opr_var->owner_opr()
                                    ->same_type<opr::Convolution>()) {
                            m_opr2convlayer[opr_var->owner_opr()]
                                    ->setBiasWeights(wt_bias);
                        } else if (
                                opr_var->owner_opr()
                                        ->same_type<
                                                opr::ConvolutionBackwardData>()) {
                            m_opr2deconvlayer[opr_var->owner_opr()]
                                    ->setBiasWeights(wt_bias);
                        }
                        varnode2itensor[opr->output(0)] =
                                varnode2itensor[result[2]];
                        break;
                    }
                    on_elemwise_arity_binary(
                            nvinfer1::ElementWiseOperation::kSUM);
                    break;
                }
                case Mode::MUL:
                    on_elemwise_arity_binary(
                            nvinfer1::ElementWiseOperation::kPROD);
                    break;
                case Mode::MIN:
                    on_elemwise_arity_binary(
                            nvinfer1::ElementWiseOperation::kMIN);
                    break;
                case Mode::MAX:
                    on_elemwise_arity_binary(
                            nvinfer1::ElementWiseOperation::kMAX);
                    break;
                case Mode::SUB:
                    on_elemwise_arity_binary(
                            nvinfer1::ElementWiseOperation::kSUB);
                    break;
                case Mode::TRUE_DIV:
                    on_elemwise_arity_binary(
                            nvinfer1::ElementWiseOperation::kDIV);
                    break;
                case Mode::POW:
                    on_elemwise_arity_binary(
                            nvinfer1::ElementWiseOperation::kPOW);
                    break;
                case Mode::FUSE_ADD_RELU: {
                    on_elemwise_arity_binary(
                            nvinfer1::ElementWiseOperation::kSUM);
                    if (is_quantized_int8_operator(opr))
                        set_itensor_dynamic_range(opr->output(0), opr);
                    auto act =
                            net->addActivation(*varnode2itensor[opr->output(0)],
                                               nvinfer1::ActivationType::kRELU);
                    mgb_assert(act, "construct network failed");
                    std::string layer_name = "TRT_ACTV:" + opr->name();
                    act->setName(layer_name.c_str());
                    std::string output_name =
                            "TRT_O:" + opr->output()[0]->name();
                    act->getOutput(0)->setName(output_name.c_str());
                    varnode2itensor[opr->output(0)] = act->getOutput(0);
                    break;
                }
                case Mode::FUSE_ADD_SIGMOID: {
                    on_elemwise_arity_binary(
                            nvinfer1::ElementWiseOperation::kSUM);
                    if (is_quantized_int8_operator(opr))
                        set_itensor_dynamic_range(opr->output(0), opr);
                    auto act = net->addActivation(
                            *varnode2itensor[opr->output(0)],
                            nvinfer1::ActivationType::kSIGMOID);
                    mgb_assert(act, "construct network failed");
                    std::string layer_name = "TRT_ACTV:" + opr->name();
                    act->setName(layer_name.c_str());
                    std::string output_name =
                            "TRT_O:" + opr->output()[0]->name();
                    act->getOutput(0)->setName(output_name.c_str());
                    varnode2itensor[opr->output(0)] = act->getOutput(0);
                    break;
                }
                case Mode::FUSE_ADD_TANH: {
                    on_elemwise_arity_binary(
                            nvinfer1::ElementWiseOperation::kSUM);
                    if (is_quantized_int8_operator(opr))
                        set_itensor_dynamic_range(opr->output(0), opr);
                    auto act =
                            net->addActivation(*varnode2itensor[opr->output(0)],
                                               nvinfer1::ActivationType::kTANH);
                    mgb_assert(act, "construct network failed");
                    std::string layer_name = "TRT_ACTV:" + opr->name();
                    act->setName(layer_name.c_str());
                    std::string output_name =
                            "TRT_O:" + opr->output()[0]->name();
                    act->getOutput(0)->setName(output_name.c_str());
                    varnode2itensor[opr->output(0)] = act->getOutput(0);
                    break;
                }
                default:
                    mgb_assert(false, "Unsupported elemwise mode.");
            }
            if (is_quantized_int8_operator(opr))
                set_itensor_dynamic_range(opr->output(0), opr);
        };

        m_opr_trait[opr::ElemwiseMultiType::typeinfo()]
                .add_to_nvinfer = [this](nvinfer1::INetworkDefinition* net,
                                         OperatorNodeBase* opr) {
            auto&& varnode2itensor =
                    m_tensorrt_graphs[m_graph_map[opr] - 1]->varnode2itensor;
            size_t ndim0 = opr->input(0)->shape().ndim,
                   ndim1 = opr->input(1)->shape().ndim;
            mgb_assert(ndim0 == ndim1);
            size_t tensor_ndim = ndim0;
            using Mode = opr::ElemwiseMultiType::Mode;
            SmallVector<nvinfer1::DimensionType> dimtypes(tensor_ndim);
            for (size_t  i = 0; i < tensor_ndim; i++) {
                dimtypes[i] = nvinfer1::DimensionType::kSPATIAL;
            }
            auto mode =
                    opr->cast_final_safe<opr::ElemwiseMultiType>().param().mode;
            mgb_assert(mode == Mode::QADD || mode == Mode::QFUSE_ADD_RELU,
                       "Only QADD and QFUSE_ADD_RELU are supported on CUDA.");
            mgb_assert(
                    opr->output(0)->dtype().enumv() == DTypeEnum::QuantizedS8,
                    "output data type %s is not supported",
                    opr->output(0)->dtype().name());
            check_input(opr->input(0), opr, dimtypes);
            check_input(opr->input(1), opr, dimtypes);
            auto dims0 = varnode2itensor[opr->input(0)]->getDimensions(),
                 dims1 = varnode2itensor[opr->input(1)]->getDimensions();
            mgb_throw_if(dims0.nbDims != dims1.nbDims, AssertionError,
                         "Input dimensions of two input tensors must be "
                         "equal (got: %d, %d).",
                         dims0.nbDims, dims1.nbDims);
            auto elem =
                    net->addElementWise(*varnode2itensor[opr->input(0)],
                                        *varnode2itensor[opr->input(1)],
                                        nvinfer1::ElementWiseOperation::kSUM);
            mgb_assert(elem, "construct network failed");
            std::string layer_name = "TRT_ELEM:" + opr->name();
            elem->setName(layer_name.c_str());
            std::string output_name = "TRT_O:" + opr->output()[0]->name();
            elem->getOutput(0)->setName(output_name.c_str());
            varnode2itensor[opr->output(0)] = elem->getOutput(0);
            set_itensor_dynamic_range(opr->output(0), opr);
            if (mode == Mode::QFUSE_ADD_RELU) {
                auto act =
                        net->addActivation(*varnode2itensor[opr->output(0)],
                                           nvinfer1::ActivationType::kRELU);
                mgb_assert(act, "construct network failed");
                std::string layer_name = "TRT_ACTV:" + opr->name();
                act->setName(layer_name.c_str());
                std::string output_name = "TRT_O:" + opr->output()[0]->name() + "_act";
                act->getOutput(0)->setName(output_name.c_str());
                varnode2itensor[opr->output(0)] = act->getOutput(0);
                set_itensor_dynamic_range(opr->output(0), opr);
            }
        };

        auto replace_matmul_opr = [this](nvinfer1::INetworkDefinition* net,
                                         OperatorNodeBase* opr) {
            auto&& varnode2itensor =
                    m_tensorrt_graphs[m_graph_map[opr] - 1]->varnode2itensor;
            SmallVector<nvinfer1::DimensionType> dimtypes;
            bool transposeA = false, transposeB = false;
            if (opr->same_type<opr::MatrixMul>()) {
                dimtypes = {nvinfer1::DimensionType::kSPATIAL,
                            nvinfer1::DimensionType::kSPATIAL};
                transposeA = opr->cast_final_safe<opr::MatrixMul>()
                                     .param()
                                     .transposeA;
                transposeB = opr->cast_final_safe<opr::MatrixMul>()
                                     .param()
                                     .transposeB;
            } else {
                mgb_assert(opr->same_type<opr::BatchedMatrixMul>());
                dimtypes = {nvinfer1::DimensionType::kINDEX,
                            nvinfer1::DimensionType::kSPATIAL,
                            nvinfer1::DimensionType::kSPATIAL};
                transposeA = opr->cast_final_safe<opr::BatchedMatrixMul>()
                                     .param()
                                     .transposeA;
                transposeB = opr->cast_final_safe<opr::BatchedMatrixMul>()
                                     .param()
                                     .transposeB;
            }
            check_input(opr->input(0), opr, dimtypes);
            check_input(opr->input(1), opr, dimtypes);
#if NV_TENSOR_RT_VERSION >= 6001
            nvinfer1::MatrixOperation
                    opA = transposeA ? nvinfer1::MatrixOperation::kTRANSPOSE
                                     : nvinfer1::MatrixOperation::kNONE,
                    opB = transposeB ? nvinfer1::MatrixOperation::kTRANSPOSE
                                     : nvinfer1::MatrixOperation::kNONE;
            auto matmul = net->addMatrixMultiply(
                    *varnode2itensor[opr->input(0)], opA,
                    *varnode2itensor[opr->input(1)], opB);
#else
            auto matmul = net->addMatrixMultiply(
                    *varnode2itensor[opr->input(0)], transposeA,
                    *varnode2itensor[opr->input(1)], transposeB);
#endif
            std::string layer_name = "TRT_MATMUL:" + opr->name();
            matmul->setName(layer_name.c_str());
            std::string output_name = "TRT_O:" + opr->output()[0]->name();
            matmul->getOutput(0)->setName(output_name.c_str());
            varnode2itensor[opr->output(0)] = matmul->getOutput(0);
        };

        // megdnn matrix mul operator on cuda backend does not support quantized
        // data type
        m_opr_trait[opr::MatrixMul::typeinfo()].add_to_nvinfer = replace_matmul_opr;
        m_opr_trait[opr::BatchedMatrixMul::typeinfo()].add_to_nvinfer = replace_matmul_opr;

        // powc only support float32
        m_opr_trait[opr::PowC::typeinfo()]
                .add_to_nvinfer = [this](nvinfer1::INetworkDefinition* net,
                                         OperatorNodeBase* opr) {
            auto&& varnode2itensor =
                    m_tensorrt_graphs[m_graph_map[opr] - 1]->varnode2itensor;
            size_t tensor_ndim = opr->input(0)->shape().ndim;
            SmallVector<nvinfer1::DimensionType> dimtypes(tensor_ndim);
            for (size_t i = 0; i < tensor_ndim; i++) {
                dimtypes[i] = nvinfer1::DimensionType::kSPATIAL;
            }
            check_input(opr->input(0), opr, dimtypes);
            auto host_one = HostTensorND(opr->output(0)->comp_node(), {1},
                                         dtype::Float32()),
                 host_zero = HostTensorND(opr->output(0)->comp_node(), {1},
                                          dtype::Float32()),
                 host_exp = HostTensorND(opr->output(0)->comp_node(), {1},
                                         dtype::Float32());
            *(reinterpret_cast<float*>(host_one.raw_ptr())) = 1;
            *(reinterpret_cast<float*>(host_zero.raw_ptr())) = 0;
            *(reinterpret_cast<float*>(host_exp.raw_ptr())) =
                    opr->cast_final_safe<opr::PowC>().param().exp;
            auto ptr = opr->owner_graph()
                               ->options()
                               .user_data
                               .get_user_data_or_create<HostTensorKeeper>();
            ptr->htr.push_back(host_one);
            ptr->htr.push_back(host_zero);
            ptr->htr.push_back(host_exp);
            auto scale =
                    net->addScale(*varnode2itensor[opr->input(0)],
                                  nvinfer1::ScaleMode::kUNIFORM,
                                  nvinfer1::Weights{nvinfer1::DataType::kFLOAT,
                                                    host_zero.raw_ptr(), 1},
                                  nvinfer1::Weights{nvinfer1::DataType::kFLOAT,
                                                    host_one.raw_ptr(), 1},
                                  nvinfer1::Weights{nvinfer1::DataType::kFLOAT,
                                                    host_exp.raw_ptr(), 1});
            std::string layer_name = "TRT_SCALE:" + opr->name();
            scale->setName(layer_name.c_str());
            std::string output_name = "TRT_O:" + opr->output()[0]->name();
            scale->getOutput(0)->setName(output_name.c_str());
            varnode2itensor[opr->output(0)] = scale->getOutput(0);
        };

        m_opr_num = 0;
        m_opr_fail_num = 0;

        detect_replace();
        mark_varnode_format_nchw4();
        update_graph();
        if (!m_opr_fail.empty()) {
            std::string msg{"TRT replace summary:\n"};
            msg += ssprintf(" number of oprs: %zu\n", m_opr_num);
            msg += ssprintf(" number of unsupported oprs: %zu\n",
                            m_opr_fail_num);
            msg += ssprintf(" first %zu unsupported oprs:\n",
                            m_opr_fail.size());
            for (size_t i = 0; i < m_opr_fail.size(); ++i) {
                msg += ssprintf("   %s {%s}: %s\n", m_opr_fail[i].opr->cname(),
                                m_opr_fail[i].opr->dyn_typeinfo()->name,
                                m_opr_fail[i].fail_msg.c_str());
            }
            msg.pop_back();
            mgb_log("%s", msg.c_str());
        }
    }
};

MGB_TYPEINFO_OBJ_IMPL(TensorRTReplacePass::Impl::HostTensorKeeper);

Maybe<std::string> TensorRTReplacePass::Impl::has_fail_msg(
        OperatorNodeBase* opr) {
    auto iter = m_opr_trait.find(opr->dyn_typeinfo());
    if (iter != m_opr_trait.end()) {
        if (iter->second.get_replace_fail_msg) {
            return iter->second.get_replace_fail_msg(opr);
        }
        return None;
    }
    return "Opr not supported.";
}

VarNodeArray TensorRTReplacePass::Impl::find_parent_conv(
        OperatorNodeBase* inp_opr) {
    OperatorNodeBase* owner_opr;
    VarNodeArray vars_to_check, new_vars, rst;
    bool conv_output_found = false;
    VarNode* conv_output_var = nullptr;
    VarNode* bias_var = nullptr;
    VarNode* new_output_var = nullptr;

    if (m_const_var_propogate->is_const(inp_opr->input(0))) {
        vars_to_check.push_back(inp_opr->input(1));
        new_output_var = inp_opr->input(1);
        bias_var = inp_opr->input(0);
    } else if (m_const_var_propogate->is_const(inp_opr->input(1))) {
        vars_to_check.push_back(inp_opr->input(0));
        new_output_var = inp_opr->input(0);
        bias_var = inp_opr->input(1);
    } else {
        // No const input. return empty rst.
        return rst;
    }

    while (vars_to_check.size() != 0) {
        for (size_t i = 0; i < vars_to_check.size(); ++i) {
            owner_opr = vars_to_check[i]->owner_opr();
            if (owner_opr->same_type<opr::Convolution>() ||
                owner_opr->same_type<opr::ConvolutionBackwardData>()) {
                conv_output_found = true;
                conv_output_var = vars_to_check[i];
                break;
            }
            if (owner_opr->same_type<opr::Elemwise>() &&
                owner_opr->cast_final<opr::Elemwise>().param().mode ==
                        opr::Elemwise::Mode::ADD) {
                for (auto var2chk : owner_opr->input()) {
                    new_vars.push_back(var2chk);
                }
            }
        }
        vars_to_check.clear();
        if (conv_output_found)
            break;
        if (new_vars.size() != 0) {
            vars_to_check.insert(vars_to_check.end(), new_vars.begin(),
                                 new_vars.end());
            new_vars.clear();
        }
    }

    if (conv_output_found) {
        conv_output_found &= m_graph_map[inp_opr] ==
                             m_graph_map[conv_output_var->owner_opr()];
        auto&& trt_graph = m_tensorrt_graphs[m_graph_map[inp_opr] - 1];
        conv_output_found &= trt_graph->outputs.count(conv_output_var) == 0;
    }

    if (conv_output_found) {
        rst.push_back(conv_output_var);
        rst.push_back(bias_var);
        rst.push_back(new_output_var);
    }

    return rst;
}

bool TensorRTReplacePass::Impl::check_input(
        VarNode* var, OperatorNodeBase* opr,
        SmallVector<nvinfer1::DimensionType> dimtypes) {
    auto trt_graph = m_tensorrt_graphs[m_graph_map[opr] - 1];
    auto&& varnode2itensor = trt_graph->varnode2itensor;
    auto iter = trt_graph->inputs.find(var);
    if (iter == trt_graph->inputs.end())  // not a input of trt graph
        return false;
    for (auto i : trt_graph->trt_inputs)
        if (i == var)  // already added to input
            return false;
    trt_graph->trt_inputs.push_back(var);
    nvinfer1::ITensor* itensor;
    MGB_MARK_USED_VAR(mgb_dtype_to_trt_dtype);
    if (dimtypes.size() == 0) {
#if NV_TENSOR_RT_VERSION >= 6001
        mgb_assert(var->shape().ndim == 4 || (var->shape().ndim == 5 && var->shape()[4] == 4));
        nvinfer1::Dims4 dims{static_cast<int>(var->shape()[0]),
                             static_cast<int>(var->shape()[1]),
                             static_cast<int>(var->shape()[2]),
                             static_cast<int>(var->shape()[3])};
        if (var->shape().ndim == 5) {
            mgb_assert(var->shape()[4] == 4);
            dims.d[1] *= 4;
        }
        itensor = trt_graph->network->addInput(
                var->cname(), mgb_dtype_to_trt_dtype(var->dtype()),
                dims);
        if (trt_graph->mark_input_varnode_nchw4.count(var)) {
            itensor->setAllowedFormats(
                    1 << static_cast<int>(nvinfer1::TensorFormat::kCHW4));
        } else {
            itensor->setAllowedFormats(
                    1 << static_cast<int>(nvinfer1::TensorFormat::kLINEAR));
        }
#else
        if (var->shape().ndim == 4) {
            // the default input tensor is a NCHW tensor
            mgb_assert(var->shape().ndim == 4,
                       "Default input tensor should be NCHW or NCHW4 format.");
            itensor = trt_graph->network->addInput(
                    var->cname(), nvinfer1::DataType::kFLOAT,
                    nvinfer1::DimsNCHW{static_cast<int>(var->shape()[0]),
                                       static_cast<int>(var->shape()[1]),
                                       static_cast<int>(var->shape()[2]),
                                       static_cast<int>(var->shape()[3])});

        } else {
            mgb_assert(var->shape().ndim == 5 && var->shape()[4] == 4,
                       "Input tensor format is not NCHW4 (got %s)",
                       var->shape().to_string().c_str());
            itensor = trt_graph->network->addInput(
                    var->cname(), nvinfer1::DataType::kFLOAT,
                    nvinfer1::DimsNCHW{static_cast<int>(var->shape()[0]),
                                       static_cast<int>(var->shape()[1] * 4),
                                       static_cast<int>(var->shape()[2]),
                                       static_cast<int>(var->shape()[3])});
        }
#endif
    } else {
        nvinfer1::Dims dims;
        // process var node that marked as nchw4 format
        if (trt_graph->mark_input_varnode_nchw4.count(var)) {
            mgb_assert(var->shape().ndim == 5 && var->shape()[4] == 4,
                       "Input tensor format is not NCHW4 (got %s)",
                       var->shape().to_string().c_str());
            dims.nbDims = var->shape().ndim - 1;
            for (size_t i = 0; i < var->shape().ndim - 1; i++) {
                dims.d[i] = var->shape()[i];
#if NV_TENSOR_RT_VERSION < 6001
                dims.type[i] = dimtypes[i];
#endif
            }
            dims.d[1] *= 4;
            // process conventional var node
        } else {
            mgb_assert(var->shape().ndim == dimtypes.size());
            mgb_assert(var->shape().ndim <= nvinfer1::Dims::MAX_DIMS);
            dims.nbDims = var->shape().ndim;
            for (size_t i = 0; i < var->shape().ndim; i++) {
                dims.d[i] = var->shape()[i];
#if NV_TENSOR_RT_VERSION < 6001
                dims.type[i] = dimtypes[i];
#endif
            }
        }
#if NV_TENSOR_RT_VERSION >= 6001
        itensor = trt_graph->network->addInput(
                var->cname(), mgb_dtype_to_trt_dtype(var->dtype()), dims);
        if (trt_graph->mark_input_varnode_nchw4.count(var)) {
            itensor->setAllowedFormats(
                    1 << static_cast<int>(nvinfer1::TensorFormat::kCHW4));
        } else {
            itensor->setAllowedFormats(
                    1 << static_cast<int>(nvinfer1::TensorFormat::kLINEAR));
        }
#else
        itensor = trt_graph->network->addInput(
                var->cname(), nvinfer1::DataType::kFLOAT, dims);
#endif
    }
    varnode2itensor[var] = itensor;
    if (trt_graph->feature_bits == TensorRTGraphFeatureBits::NCHW4_QINT8)
        set_itensor_dynamic_range(var, opr);
    return true;
}

void TensorRTReplacePass::Impl::set_itensor_dynamic_range(
        VarNode* var, OperatorNodeBase* opr) {
    MGB_MARK_USED_VAR(var);
    MGB_MARK_USED_VAR(opr);
#if NV_TENSOR_RT_VERSION >= 5020
    auto&& varnode2itensor =
            m_tensorrt_graphs[m_graph_map[opr] - 1]->varnode2itensor;
    auto&& tensor = varnode2itensor[var];
    auto&& data_type = var->dtype();
    mgb_assert(data_type.enumv() == DTypeEnum::QuantizedS8);
    float scale = get_scale(data_type);
    tensor->setDynamicRange(-i8_max * scale, i8_max * scale);
#endif
}

HostTensorND TensorRTReplacePass::Impl::get_value(VarNode* var, ConvFormat format) {
    auto cg = m_opt_state.graph().comp_graph();
    auto inferred_val = HostTensorND(var->comp_node(), dtype::Float32());
    auto cb = [&](DeviceTensorND& val) { inferred_val.copy_from(val); };
    if (format == ConvFormat::NCHW) {
        mgb_assert(var->dtype() == dtype::Float32());
        auto orig_level = cg->options().log_level;
        cg->options().log_level = 0;
        MGB_TRY { cg->compile({{var, cb}})->execute(); }
        MGB_FINALLY(cg->options().log_level = orig_level);
    } else {
        mgb_assert(format == ConvFormat::NCHW4);
        if (var->shape().ndim == 5) {
            // assume nchw4 layout
            mgb_assert(var->shape()[4] == 4);
            auto x = SymbolVar(var);
            auto xshp = opr::GetVarShape::make(x);

            auto cv = [&x](int v) { return x.make_scalar(v); };
            auto sub = [&xshp, &cv](int idx) {
                return opr::IndexAt::make(xshp, {{0, cv(idx)}});
            };
            auto tshp =
                    opr::Concat::make({sub(0), sub(1) * 4, sub(2), sub(3)}, 0);
            auto y0 = opr::Dimshuffle::make(x, {0, 1, 4, 2, 3});
            auto y1 = opr::Reshape::make(y0, tshp);
            if (var->dtype().enumv() == DTypeEnum::QuantizedS8 ||
                var->dtype().enumv() == DTypeEnum::QuantizedS32) {
                y1 = opr::TypeCvt::make(y1, dtype::Float32());
            }
            auto orig_level = cg->options().log_level;
            cg->options().log_level = 0;
            cg->options().graph_opt.tensorrt = false;
            MGB_TRY { cg->compile({{y1.node(), cb}})->execute(); }
            MGB_FINALLY({
                cg->options().log_level = orig_level;
                cg->options().graph_opt.tensorrt = true;
            });
        } else if (var->shape().ndim == 6) {
            // assume nchw4 layout
            mgb_assert(var->shape()[5] == 4);
            mgb_assert(var->dtype().enumv() == DTypeEnum::QuantizedS8 ||
                       var->dtype() == dtype::Float32());
            auto x = SymbolVar(var);
            auto xshp = opr::GetVarShape::make(x);

            auto cv = [&x](int v) { return x.make_scalar(v); };
            auto sub = [&xshp, &cv](int idx) {
                return opr::IndexAt::make(xshp, {{0, cv(idx)}});
            };
            auto tshp = opr::Concat::make(
                    {sub(0), sub(1), sub(2) * 4, sub(3), sub(4)}, 0);
            auto y0 = opr::Dimshuffle::make(x, {0, 1, 2, 5, 3, 4});
            auto y1 = opr::Reshape::make(y0, tshp);
            if (var->dtype().enumv() == DTypeEnum::QuantizedS8) {
                y1 = opr::TypeCvt::make(y1, dtype::Float32());
            }
            auto orig_level = cg->options().log_level;
            cg->options().log_level = 0;
            cg->options().graph_opt.tensorrt = false;
            MGB_TRY { cg->compile({{y1.node(), cb}})->execute(); }
            MGB_FINALLY({
                cg->options().log_level = orig_level;
                cg->options().graph_opt.tensorrt = true;
            });
        }
    }
    auto ptr = var->owner_graph()
                       ->options()
                       .user_data.get_user_data_or_create<HostTensorKeeper>();
    ptr->htr.push_back(inferred_val);
    return inferred_val;
}

float TensorRTReplacePass::Impl::get_scale(DType data_type) {
    float scale = 1.f;
#define cb(_dt)                               \
    case DTypeTrait<_dt>::enumv:              \
        scale = data_type.param<_dt>().scale; \
        break;
    switch (data_type.enumv()) {
        MEGDNN_FOREACH_QUANTIZED_DTYPE(cb);
        default:
            mgb_throw(InternalError, "invalid quantized data type: %s",
                      data_type.name());
    }
    return scale;
#undef cb
}

bool TensorRTReplacePass::Impl::is_quantized_int8_operator(
        OperatorNodeBase* opr) {
    bool is_quantized = true;
    if (opr->same_type<opr::ConvBias>()) {
        is_quantized = opr->input(0)->dtype().enumv() == DTypeEnum::QuantizedS8;
        mgb_assert(!is_quantized ||
                   opr->output(0)->dtype().enumv() == DTypeEnum::QuantizedS8);
        return is_quantized;
    }
    for (auto&& inp : opr->input()) {
        if (inp->dtype().enumv() != DTypeEnum::QuantizedS8) {
            is_quantized = false;
            break;
        }
    }
    // assume all operator has only one output
    auto&& out = opr->output(0);
    if (out->dtype().enumv() != DTypeEnum::QuantizedS8) {
        is_quantized = false;
    }
    return is_quantized;
}

void TensorRTReplacePass::Impl::detect_replace() {
    auto cb = [this](OperatorNodeBase* opr) {
        m_const_var_propogate->add_opr(opr);
    };
    m_opt_state.graph().iter(cb);

    auto on_opr = [this](OperatorNodeBase* opr) {
        ++m_opr_num;
        Maybe<std::string> irreplaceable_msg = has_fail_msg(opr);
        TensorRTGraphFeatureBits feature_bits =
                is_quantized_int8_operator(opr)
                        ? TensorRTGraphFeatureBits::NCHW4_QINT8
                        : TensorRTGraphFeatureBits::NCHW_FLOAT;
        if (!irreplaceable_msg.valid()) {
            size_t max = 1;
            for (auto i : opr->input()) {
                if (!has_fail_msg(i->owner_opr()).valid())
                    update_max(max, m_graph_map[i->owner_opr()]);
                else
                    update_max(max, m_graph_map[i->owner_opr()] + 1);
            }

            size_t max_update = max;
            for (; max_update <= m_tensorrt_graphs.size(); max_update++) {
                TensorRTGraphFeatureBits trt_graph_feature_bits =
                        m_tensorrt_graphs[max_update - 1]->feature_bits;
                if (trt_graph_feature_bits == feature_bits)
                    break;
            }
            max = max_update;

            m_graph_map[opr] = max;
            if (max > m_tensorrt_graphs.size()) {
                opr->output(0)->comp_node().activate();
                m_tensorrt_graphs.push_back(
                        std::make_shared<TensorRTGraph>(feature_bits));
            }
            for (auto i : opr->input()) {
                if (m_graph_map[i->owner_opr()] != max) {
                    m_tensorrt_graphs[max - 1]->inputs.insert(i);
                    if (!has_fail_msg(i->owner_opr()).valid()) {
                        //! TODO: check
                        m_tensorrt_graphs[m_graph_map[i->owner_opr()] - 1]
                                ->outputs.insert(i);
                    }
                }
            }
        } else {
            static const ThinHashSet<Typeinfo*> ignore_types{
                    opr::SharedDeviceTensor::typeinfo(),
                    opr::ImmutableTensor::typeinfo(),
                    opr::Host2DeviceCopy::typeinfo(),
                    opr::MultipleDeviceTensorHolder::typeinfo()};
            if (!ignore_types.count(opr->dyn_typeinfo())) {
                ++m_opr_fail_num;
                if (m_opr_fail.size() < OPR_FAIL_LOG_NUM) {
                    FailInfo fail_info;
                    fail_info.opr = opr;
                    fail_info.fail_msg = irreplaceable_msg.val();
                    m_opr_fail.push_back(fail_info);
                }
            }
            size_t max = 0;
            for (auto i : opr->input()) {
                if (m_graph_map[i->owner_opr()] > max)
                    max = m_graph_map[i->owner_opr()];
                if (!has_fail_msg(i->owner_opr()).valid()) {
                    //! TODO: check
                    m_tensorrt_graphs[m_graph_map[i->owner_opr()] - 1]
                            ->outputs.insert(i);
                }
            }
            m_graph_map[opr] = max;
        }
    };
    m_opt_state.graph().iter(on_opr);

    for (auto i : m_opt_state.graph().endpoint_vars()) {
        auto var_node = i.node();
        if (!has_fail_msg(var_node->owner_opr()).valid()) {
            //! TODO: check
            m_tensorrt_graphs[m_graph_map[var_node->owner_opr()] - 1]
                    ->outputs.insert(var_node);
        }
    }
}

void TensorRTReplacePass::Impl::
        mark_varnode_format_nchw4() {
    for (auto trt_graph : m_tensorrt_graphs) {
        trt_graph->mark_varnode_format_nchw4();
    }
}

void TensorRTReplacePass::Impl::update_graph() {
    using GpuAllocator = opr::TensorRTOpr::GpuAllocator;
    using TensorRTOpr = opr::TensorRTOpr;

    std::shared_ptr<GpuAllocator> gpu_allocator;
    std::shared_ptr<ExtraDep> extra_dep = std::make_shared<ExtraDep>();

    // construct trt network
    auto construct_network = [this, &gpu_allocator, &extra_dep](OperatorNodeBase* opr) {
        if (!has_fail_msg(opr).valid()) {
            auto cn = opr->output(0)->comp_node();
            auto trt_graph = m_tensorrt_graphs[m_graph_map[opr] - 1];
            auto b = trt_graph->builder;
            mgb_assert(b != nullptr);
            if (!gpu_allocator) {
                gpu_allocator = std::make_shared<GpuAllocator>(cn);
                b->setGpuAllocator(gpu_allocator.get());
            } else {
                auto cn0 = gpu_allocator->comp_node();
                mgb_assert(cn0 == cn,
                           "multiple comp nodes for trt graph are not "
                           "supported: %s %s",
                           cn0.to_string().c_str(), cn.to_string().c_str());
            }

            if (!trt_graph->network) {
#if NV_TENSOR_RT_VERSION >= 6001
                nvinfer1::NetworkDefinitionCreationFlags flags;
                flags = 1 << static_cast<int>(
                                nvinfer1::NetworkDefinitionCreationFlag::
                                        kEXPLICIT_BATCH);
                trt_graph->network = b->createNetworkV2(flags);
#else
                trt_graph->network = b->createNetwork();
#endif
            }
            // make extra dep
            for (auto&& inp : trt_graph->inputs) {
                extra_dep->operator[](opr).push_back(inp);
            }

            auto iter = m_opr_trait.find(opr->dyn_typeinfo());
            if (iter != m_opr_trait.end()) {
                if (iter->second.add_to_nvinfer) {
                    iter->second.add_to_nvinfer(trt_graph->network, opr);
                }
            }
        }
    };
    m_opt_state.graph().iter(construct_network);

    // trt network markOutput
    for (auto trt_graph : m_tensorrt_graphs) {
        // record traverse order
        size_t idx = 0;
        auto&& varnode2itensor = trt_graph->varnode2itensor;
        for (auto output : trt_graph->outputs) {
            trt_graph->output2idx[output] = idx++;
            trt_graph->network->markOutput(*varnode2itensor[output]);
#if NV_TENSOR_RT_VERSION >= 6001
            if (output->dtype().enumv() == DTypeEnum::QuantizedS8) {
                varnode2itensor[output]->setType(nvinfer1::DataType::kINT8);
            }
            if (trt_graph->mark_output_varnode_nchw4.count(output)) {
                mgb_assert(output->dtype().enumv() == DTypeEnum::QuantizedS8);
                varnode2itensor[output]->setAllowedFormats(
                        1 << static_cast<int>(nvinfer1::TensorFormat::kCHW4));
            } else {
                varnode2itensor[output]->setAllowedFormats(
                        1 << static_cast<int>(nvinfer1::TensorFormat::kLINEAR));
            }
#endif
        }
    }

    ThinHashSet<OperatorNodeBase*> visited;
    // replace opr by trt
    auto update_opr = [this, &gpu_allocator,
                       &visited](OperatorNodeBase* opr) {
        if (!has_fail_msg(opr).valid()) {
            mgb_assert(gpu_allocator);
            auto trt_graph = m_tensorrt_graphs[m_graph_map[opr] - 1];
            for (auto&& inp : trt_graph->trt_inputs) {
                mgb_assert(visited.count(inp->owner_opr()));
            }
            if (trt_graph->trt_outputs.empty()) {
                // use updated varnode instead of old one
                auto inps = trt_graph->trt_inputs;
                VarNodeArray new_inps{inps.size()};
                for (size_t i = 0; i < inps.size(); i++) {
                    new_inps[i] = m_rewriter.get_var(inps[i]);
#if NV_TENSOR_RT_VERSION < 6001
                    if (trt_graph->mark_input_varnode_nchw4.count(inps[i])) {
                        auto x = SymbolVar(new_inps[i]);
                        auto xshp = opr::GetVarShape::make(x);
                        auto cv = [&x](int v) { return x.make_scalar(v); };
                        auto sub = [&xshp, &cv](int idx) {
                            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
                        };
                        auto tshp = opr::Concat::make(
                                {sub(0), sub(1) * 4, sub(2), sub(3)}, 0);
                        auto y0 = opr::Dimshuffle::make(x, {0, 1, 4, 2, 3});
                        auto y1 = opr::Reshape::make(y0, tshp);

                        new_inps[i] = y1.node();
                    }
                    if (inps[i]->dtype().enumv() == DTypeEnum::QuantizedS8) {
                        new_inps[i] = opr::TypeCvt::make(new_inps[i],
                                                         dtype::Float32())
                                              .node();
                    }
#endif
                }
                // now trt_graph does not own the unique_ptr of infer builder
                m_opt_state.call_with_opr(opr, [&] {
                    trt_graph->trt_outputs =
                            cg::to_var_node_array(TensorRTOpr::make(
                                    TensorRTOpr::to_shared_ptr_builder(
                                            trt_graph->builder),
                                    TensorRTOpr::to_shared_ptr_network(
                                            trt_graph->network),
                                    trt_graph->feature_bits, gpu_allocator,
                                    cg::to_symbol_var_array(new_inps)));
                });
                mgb_assert(trt_graph->trt_outputs.size() ==
                                   trt_graph->outputs.size(),
                           "mgb outputs number != tensorrt outputs number");
            }
            for (auto&& output : opr->output()) {
                if (trt_graph->outputs.count(output)) {
                    size_t output_idx = trt_graph->output2idx[output];
                    VarNode* output_var = trt_graph->trt_outputs[output_idx];
#if NV_TENSOR_RT_VERSION < 6001
                    if (trt_graph->mark_output_varnode_nchw4.count(output)) {
                        auto x = SymbolVar(output_var);
                        auto xshp = opr::GetVarShape::make(x);
                        auto cv = [&x](int v) { return x.make_scalar(v); };
                        auto sub = [&xshp, &cv](int idx) {
                            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
                        };
                        auto tshp = opr::Concat::make(
                                {sub(0), sub(1) / 4, cv(4), sub(2), sub(3)}, 0);
                        auto y0 = opr::Reshape::make(x, tshp);
                        auto y1 = opr::Dimshuffle::make(y0, {0, 1, 3, 4, 2});

                        output_var = y1.node();
                    }
                    if (output->dtype().enumv() == DTypeEnum::QuantizedS8) {
                        float scale = get_scale(output->dtype());
                        output_var =
                                opr::TypeCvt::make(output_var,
                                                   dtype::QuantizedS8{scale})
                                        .node();
                    }
#endif
                    m_rewriter.replace_var(
                            output, output_var,
                            mgb_ssprintf_log("replace opr: %s",
                                             output->owner_opr()->cname())
                                    .c_str());
                }
            }
            visited.insert(opr);
        } else {
            for (auto&& inp : opr->input()) {
                mgb_assert(visited.count(inp->owner_opr()));
            }

            visited.insert(opr);
            m_rewriter.auto_replace_outputs(opr);
        }
    };
    m_opt_state.graph().iter(update_opr, std::move(extra_dep));
    m_rewriter.apply_inplace();
}

const char* TensorRTReplacePass::name() const {
    return mgb_cstr_log("tensorrt_replace");
}

void TensorRTReplacePass::apply(OptState& opt) const {
    if (CompNode::get_device_count(CompNode::DeviceType::CUDA)) {
        opt.set_var_replace_check_flag(gopt::VarReplaceCheckFlag::CHECK_SHAPE |
                                       gopt::VarReplaceCheckFlag::CHECK_DTYPE);
        Impl(*this, opt);
    } else {
        mgb_log_debug("cuda is not available; TensorRTReplacePass is ignored");
    }
}

// ===================== TensorRTGraph =================
void TensorRTReplacePass::Impl::TensorRTGraph::mark_varnode_format_nchw4() {
    // consider TensorRT subgraph as a bi-directed graph and divide it into
    // multi connected components, mark the subgraph's inputs or outputs varnode
    // in format nchw4 iff the varnode belong to the connected components which
    // contains at least one NCHW4 operator(e.g. ConvBias, Pooling)

    // p[arrent] array use for Disjoint Set
    ThinHashMap<OperatorNodeBase*, OperatorNodeBase*> p;
    ThinHashSet<OperatorNodeBase*> outsides;

    thin_function<OperatorNodeBase*(OperatorNodeBase*)> get_root;
    get_root = [&](OperatorNodeBase* opr) -> OperatorNodeBase* {
        mgb_assert(p.count(opr));
        return p[opr] == opr ? opr : p[opr] = get_root(p[opr]);
    };

    auto is_format_nchw4 = [&](OperatorNodeBase* opr) {
        if (outsides.count(opr)) {
            return false;
        }
        if (opr->same_type<opr::ConvBias>()) {
            auto&& param = opr->cast_final_safe<opr::ConvBias>().param();
            if (param.format == opr::ConvBias::Param::Format::NCHW4)
                return true;
        }
        if (opr->same_type<opr::Pooling>()) {
            auto&& param = opr->cast_final_safe<opr::Pooling>().param();
            if (param.format == opr::Pooling::Param::Format::NCHW4)
                return true;
        }
        return false;
    };

    auto cb = [&](OperatorNodeBase* opr) {
        mgb_assert(!p.count(opr));
        p[opr] = opr;
        for (auto&& inp: opr->input()) {
            auto root = get_root(inp->owner_opr());
            // ensure that if one of oprs in tree is nchw4
            // the root of the tree must be nchw4
            if (is_format_nchw4(root)) {
                p[get_root(opr)] = root;
            } else {
                p[root] = get_root(opr);
            }
        }
    };

    DepOprIter iter{cb};
    for (auto&& inp : inputs) {
        p[inp->owner_opr()] = inp->owner_opr();
        iter.set_visited(inp->owner_opr());
        outsides.insert(inp->owner_opr());
    }

    for (auto&& out : outputs) {
        iter.add(out->owner_opr());
    }

    for (auto&& inp : inputs) {
        if (is_format_nchw4(get_root(inp->owner_opr()))) {
            mark_input_varnode_nchw4.insert(inp);
        }
    }

    for (auto&& out : outputs) {
        if (is_format_nchw4(get_root(out->owner_opr()))) {
            mark_output_varnode_nchw4.insert(out);
        }
    }
}

void mgb::tensorrt::transform_dest_vars_inplace(
        mgb::cg::VarNodeArray& dest_vars,
        cg::GraphCommonOptimizeOptions& options) {
    gopt::GraphOptimizer optimizer;
    //! As in megengine, the layout is NCHW, while tensorrt pass currently
    //! only support NCHW4(int8), so we transform layout to nchw4 firstly.
    if (options.has_set_nchw4()) {
        options.disable_nchw4();
        optimizer.add_pass<FuseConvBiasNonlinPass>();
        optimizer.add_pass(EnableNCHW4Pass::make_nchw4_converter());
    }
    optimizer.add_pass<ExpandFusedArithPass>();
    optimizer.add_pass<gopt::TensorRTReplacePass>();
    optimizer.add_pass<ArithFusePass>();
#if NV_TENSOR_RT_VERSION < 6001
    optimizer.add_pass<ShuffleShuffleRemovePass>();
    optimizer.add_pass<RemoveRedundantTypeCvtPass>();
#endif
    optimizer.apply_inplace(dest_vars);
}

#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
