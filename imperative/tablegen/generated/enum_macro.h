// clang-format off
#define FOR_EACH_ENUM_PARAM(cb) \
    cb(::megdnn::param::PoolingV0::Mode); \
    cb(::megdnn::param::Convolution::Format); \
    cb(::megdnn::param::Argsort::Order); \
    cb(::megdnn::param::ConvBiasV0::NonlineMode); \
    cb(::megdnn::param::ConvolutionV0::Mode); \
    cb(::megdnn::param::ConvolutionV0::Sparse); \
    cb(::megdnn::param::ConvolutionV1::ComputeMode); \
    cb(::megdnn::param::BN::ParamDim); \
    cb(::megdnn::param::BN::FwdMode); \
    cb(::megdnn::param::MatrixMulV1::ComputeMode); \
    cb(::megdnn::param::MatrixMul::Format); \
    cb(::megdnn::param::CollectiveComm::Mode); \
    cb(::megdnn::param::Convolution3D::Mode); \
    cb(::megdnn::param::Convolution3D::Sparse); \
    cb(::megdnn::param::Convolution3D::DataType); \
    cb(::megdnn::param::Convolution3D::Format); \
    cb(::megdnn::param::ConvolutionV0::Format); \
    cb(::megdnn::param::CvtColor::Mode); \
    cb(::megdnn::param::Elemwise::Mode); \
    cb(::megdnn::param::ElemwiseMultiType::Mode); \
    cb(::megdnn::param::WarpPerspectiveV1::BorderMode); \
    cb(::megdnn::param::MultiHeadAttn::AttnMaskType); \
    cb(::megdnn::param::MultiHeadAttn::TensorCombinationType); \
    cb(::megdnn::param::Padding::PaddingMode); \
    cb(::megdnn::param::RNNCell::NonlineMode); \
    cb(::megdnn::param::ROIAlignV0::Mode); \
    cb(::megdnn::param::ROIPooling::Mode); \
    cb(::megdnn::param::Reduce::Mode); \
    cb(::megdnn::param::Reduce::DataType); \
    cb(::megdnn::param::WarpPerspectiveV1::InterpolationMode); \
    cb(::megdnn::param::TopK::Mode);
#define FOR_EACH_BIT_COMBINED_ENUM_PARAM(cb) \
    cb(::megdnn::param::ExecutionPolicy::Strategy);
// clang-format on
