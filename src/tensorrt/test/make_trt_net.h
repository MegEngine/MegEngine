#include "megbrain/opr/blas.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/plugin/profiler.h"
#include "megbrain/test/helper.h"
#include "megbrain/utils/debug.h"

#if MGB_ENABLE_TENSOR_RT

#include "megbrain/tensorrt/tensorrt_opr.h"

#include <random>

using namespace mgb;
using namespace opr;
using namespace nvinfer1;

template <typename T>
using TensorRTUniquePtr = intl::TensorRTUniquePtr<T>;

namespace mgb {
namespace opr {
namespace intl {

struct SimpleTensorRTNetwork {
    HostTensorGenerator<> gen;
    std::shared_ptr<HostTensorND> host_x, host_w, host_b;
    std::shared_ptr<ComputingGraph> graph;
    SymbolVar x, y;

    HostTensorND host_z1;

    SimpleTensorRTNetwork();

    std::pair<nvinfer1::IBuilder*, INetworkDefinition*> create_trt_network(
            bool has_batch_dim);
};

struct BatchedTensorRTNetwork {
    HostTensorGenerator<> gen;
    std::shared_ptr<HostTensorND> host_x, host_w, host_b;
    std::shared_ptr<ComputingGraph> graph;
    SymbolVar x, y;

    HostTensorND host_z1;

    BatchedTensorRTNetwork();

    std::pair<nvinfer1::IBuilder*, INetworkDefinition*> create_trt_network(
            bool has_batch_dim);
};

struct SimpleQuantizedTensorRTNetwork {
    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM> weight_gen{
            1 * 1.1f, 127 * 1.1f};
    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM> range_gen{
            1 * 1.2f, 127 * 1.2f};
    std::shared_ptr<HostTensorND> host_x, host_w, host_b;
    std::shared_ptr<ComputingGraph> graph;
    SymbolVar x, y;
    SymbolVar quantized_x, quantized_y;

    SimpleQuantizedTensorRTNetwork();

    std::pair<nvinfer1::IBuilder*, INetworkDefinition*> create_trt_network(
            bool has_batch_dim);
};

struct ConcatConvTensorRTNetwork {
    HostTensorGenerator<> gen;
    std::shared_ptr<HostTensorND> host_x0, host_x1, host_x, host_w, host_b;
    std::shared_ptr<ComputingGraph> graph;
    SymbolVar x0, x1, y;

    HostTensorND host_z1;

    ConcatConvTensorRTNetwork();

    std::pair<nvinfer1::IBuilder*, INetworkDefinition*> create_trt_network(
            bool has_batch_dim);
};

struct ReshapeConcatTensorRTNetwork {
    HostTensorGenerator<> gen;
    std::shared_ptr<HostTensorND> host_x0, host_y0;
    std::shared_ptr<ComputingGraph> graph;
    SymbolVar x0, y0, z;

    ReshapeConcatTensorRTNetwork();

    std::pair<nvinfer1::IBuilder*, INetworkDefinition*> create_trt_network(
            bool has_batch_dim);
};

#if NV_TENSOR_RT_VERSION >= 6001
struct DynamicShapeTensorRTNetwork {
    HostTensorGenerator<> gen;
    std::shared_ptr<HostTensorND> host_x, host_w1, host_b1;
    std::shared_ptr<ComputingGraph> graph;
    SymbolVar x, y1;

    DynamicShapeTensorRTNetwork(size_t n, size_t c, size_t h, size_t w);

    TensorRTUniquePtr<ICudaEngine> create_trt_network();
};
#endif

}  // namespace intl
}  // namespace opr
}  // namespace mgb

#endif  // MGB_ENABLE_TENSOR_RT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
