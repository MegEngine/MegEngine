#include "./opr_impl.h"
#include "./kernel.cuh"
#include "src/common/utils.h"
#include "src/cuda/handle.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

namespace {
const char* status2str(curandStatus_t status) {
    switch (status) {
#define C(v) \
    case v:  \
        return #v
        C(CURAND_STATUS_SUCCESS);
        C(CURAND_STATUS_VERSION_MISMATCH);
        C(CURAND_STATUS_NOT_INITIALIZED);
        C(CURAND_STATUS_ALLOCATION_FAILED);
        C(CURAND_STATUS_TYPE_ERROR);
        C(CURAND_STATUS_OUT_OF_RANGE);
        C(CURAND_STATUS_LENGTH_NOT_MULTIPLE);
        C(CURAND_STATUS_DOUBLE_PRECISION_REQUIRED);
        C(CURAND_STATUS_LAUNCH_FAILURE);
        C(CURAND_STATUS_PREEXISTING_FAILURE);
        C(CURAND_STATUS_INITIALIZATION_FAILED);
        C(CURAND_STATUS_ARCH_MISMATCH);
        C(CURAND_STATUS_INTERNAL_ERROR);
#undef C
    }
    return "unknown";
}
#define CURAND_CHECK(expr)                                               \
    do {                                                                 \
        curandStatus_t status = (expr);                                  \
        MEGDNN_MARK_USED_VAR(&status2str);                               \
        if (status != CURAND_STATUS_SUCCESS) {                           \
            megdnn_throw(ssprintf(                                       \
                    "curand call failed: status=%d(%s) call=%s", status, \
                    status2str(status), #expr));                         \
        }                                                                \
    } while (0)

}  // namespace

CuRandHandle::CuRandHandle(cudaStream_t stream, uint64_t seed) {
    CURAND_CHECK(curandCreateGenerator(&m_gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetStream(m_gen, stream));
    this->seed(seed);
}

CuRandHandle::~CuRandHandle() {
    if (curandDestroyGenerator(m_gen) != CURAND_STATUS_SUCCESS) {
        megdnn_trap();
    }
}

void CuRandHandle::seed(uint64_t seed) {
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(m_gen, seed));
    m_seed = seed;
}

UniformRNGImpl::UniformRNGImpl(Handle* handle)
        : UniformRNG(handle), m_curand_handle(cuda_stream(handle)) {}

void UniformRNGImpl::exec(_megdnn_tensor_inout dst, _megdnn_workspace workspace) {
    check_exec(dst.layout, workspace.size);
    megdnn_assert(dst.layout.dtype == dtype::Float32(), "only float32 supported");
    m_curand_handle.ensure_seed(m_param.seed);
    CURAND_CHECK(curandGenerateUniform(
            m_curand_handle.gen(), dst.ptr<dt_float32>(), dst.layout.total_nr_elems()));
}

GaussianRNGImpl::GaussianRNGImpl(Handle* handle)
        : GaussianRNG(handle), m_curand_handle(cuda_stream(handle)) {}

void GaussianRNGImpl::exec(_megdnn_tensor_inout dst, _megdnn_workspace workspace) {
    check_exec(dst.layout, workspace.size);
    megdnn_assert(dst.layout.dtype == dtype::Float32(), "only float32 supported");
    auto ptr = dst.ptr<dt_float32>();
    auto size = dst.layout.total_nr_elems();
    megdnn_assert(size);
    m_curand_handle.ensure_seed(m_param.seed);
    auto gen = m_curand_handle.gen();
    if (size % 2) {
        auto wk = workspace.ptr<dt_float32>();
        CURAND_CHECK(curandGenerateNormal(gen, wk, 2, m_param.mean, m_param.std));
        cuda_check(cudaMemcpyAsync(
                ptr + size - 1, wk, sizeof(dt_float32), cudaMemcpyDeviceToDevice,
                cuda_stream(handle())));
        --size;
    }

    if (size) {
        CURAND_CHECK(curandGenerateNormal(gen, ptr, size, m_param.mean, m_param.std));
    }
}

size_t GaussianRNGImpl::get_workspace_in_bytes(const TensorLayout& layout) {
    if (layout.total_nr_elems() % 2)
        return 2 * layout.dtype.size();
    return 0;
}

GammaRNGImpl::GammaRNGImpl(Handle* handle)
        : GammaRNG(handle), m_seed(0), m_offset(0), m_stream(cuda_stream(handle)) {}

void GammaRNGImpl::exec(
        _megdnn_tensor_in shape, _megdnn_tensor_in scale, _megdnn_tensor_inout dst,
        _megdnn_workspace workspace) {
    check_exec(shape.layout, scale.layout, dst.layout, workspace.size);
    auto size = dst.layout.total_nr_elems();
    megdnn_assert(size);
    ensure_seed(m_param.seed);
    ElemwiseOpParamN<0> ele_param(size);
    switch (dst.layout.dtype.enumv()) {
#define cb(_dt)                                                              \
    case DTypeTrait<_dt>::enumv: {                                           \
        using ctype = DTypeTrait<_dt>::ctype;                                \
        run_elemwise<random::GammaKernel<ctype>, ctype, 0>(                  \
                ele_param, m_stream, {dst, shape, scale, m_seed, m_offset}); \
        break;                                                               \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
        default:
            megdnn_throw("bad dtype");
    }
    m_offset += 16;
}

PoissonRNGImpl::PoissonRNGImpl(Handle* handle)
        : PoissonRNG(handle), m_seed(0), m_offset(0), m_stream(cuda_stream(handle)) {}

void PoissonRNGImpl::exec(
        _megdnn_tensor_in lam, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(lam.layout, dst.layout, workspace.size);
    auto size = dst.layout.total_nr_elems();
    megdnn_assert(size);
    ensure_seed(m_param.seed);
    ElemwiseOpParamN<0> ele_param(size);
    switch (dst.layout.dtype.enumv()) {
#define cb(_dt)                                                     \
    case DTypeTrait<_dt>::enumv: {                                  \
        using ctype = DTypeTrait<_dt>::ctype;                       \
        run_elemwise<random::PoissonKernel<ctype>, ctype, 0>(       \
                ele_param, m_stream, {dst, lam, m_seed, m_offset}); \
        break;                                                      \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
        default:
            megdnn_throw("bad dtype");
    }
    m_offset += 20;
}

BetaRNGImpl::BetaRNGImpl(Handle* handle)
        : BetaRNG(handle), m_seed(0), m_offset(0), m_stream(cuda_stream(handle)) {}

void BetaRNGImpl::exec(
        _megdnn_tensor_in alpha, _megdnn_tensor_in beta, _megdnn_tensor_out dst,
        _megdnn_workspace workspace) {
    check_exec(alpha.layout, beta.layout, dst.layout, workspace.size);
    auto size = dst.layout.total_nr_elems();
    megdnn_assert(size);
    ensure_seed(m_param.seed);
    ElemwiseOpParamN<0> ele_param(size);
    switch (dst.layout.dtype.enumv()) {
#define cb(_dt)                                                             \
    case DTypeTrait<_dt>::enumv: {                                          \
        using ctype = DTypeTrait<_dt>::ctype;                               \
        run_elemwise<random::BetaKernel<ctype>, ctype, 0>(                  \
                ele_param, m_stream, {dst, alpha, beta, m_seed, m_offset}); \
        break;                                                              \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
        default:
            megdnn_throw("bad dtype");
    }
    m_offset += 32;
}

PermutationRNGImpl::PermutationRNGImpl(Handle* handle)
        : PermutationRNG(handle),
          m_seed(0),
          m_offset(0),
          m_stream(cuda_stream(handle)) {}

void PermutationRNGImpl::exec(_megdnn_tensor_inout dst, _megdnn_workspace workspace) {
    check_exec(dst.layout, workspace.size);
    auto size = dst.layout.total_nr_elems();
    megdnn_assert(size);
    ensure_seed(m_param.seed);

    auto wk = workspace.ptr<void>();
    switch (dst.layout.dtype.enumv()) {
#define cb(_dt)                                                          \
    case DTypeTrait<_dt>::enumv: {                                       \
        using ctype = DTypeTrait<_dt>::ctype;                            \
        ctype max_size = DTypeTrait<_dt>::max() - 1;                     \
        megdnn_assert(ctype(size) < max_size);                           \
        random::permutation_forward<ctype>(                              \
                dst.ptr<ctype>(), wk, size, m_seed, m_offset, m_stream); \
        break;                                                           \
    }
        cb(::megdnn::dtype::Float32) cb(::megdnn::dtype::Int32)
                cb(::megdnn::dtype::Int16)
#undef cb
                        default : megdnn_throw("bad dtype");
    }
    m_offset += 8;
}

size_t PermutationRNGImpl::get_workspace_in_bytes(const TensorLayout& layout) {
    size_t size = layout.total_nr_elems();
    return random::get_permutation_workspace_in_bytes(size);
}

ShuffleRNGForwardImpl::ShuffleRNGForwardImpl(Handle* handle)
        : ShuffleRNGForward(handle),
          m_seed(0),
          m_offset(0),
          m_stream(cuda_stream(handle)) {}

void ShuffleRNGForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_tensor_out indices,
        _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, indices.layout, workspace.size);
    ensure_seed(m_param.seed);
    auto wk = workspace.ptr<void>();
    const auto len = indices.layout[0];
    random::permutation_forward<dt_int32>(
            indices.ptr<dt_int32>(), wk, len, m_seed, m_offset, m_stream);
    size_t step = 0;
    for (size_t i = 1; i < src.layout.ndim; ++i) {
        step += src.layout[i];
    }
    if (step <= 0)
        step = 1;
    switch (src.layout.dtype.enumv()) {
#define cb(DType)                                                                  \
    case DTypeTrait<DType>::enumv:                                                 \
        random::shuffle_forward<DTypeTrait<DType>::ctype>(                         \
                src.ptr<DTypeTrait<DType>::ctype>(),                               \
                dst.ptr<DTypeTrait<DType>::ctype>(), indices.ptr<dt_int32>(), len, \
                step, m_stream);                                                   \
        break;
        ARGSORT_FOREACH_CTYPE(cb)
#undef cb
        default:
            megdnn_throw("bad dtype");
    }
    m_offset += 8;
}

size_t ShuffleRNGForwardImpl::get_workspace_in_bytes(
        const TensorLayout&, const TensorLayout&, const TensorLayout& indices) {
    size_t size = indices.total_nr_elems();
    return random::get_permutation_workspace_in_bytes(size);
}

ShuffleRNGBackwardImpl::ShuffleRNGBackwardImpl(Handle* handle)
        : ShuffleRNGBackward(handle), m_stream(cuda_stream(handle)) {}

void ShuffleRNGBackwardImpl::exec(
        _megdnn_tensor_in diff, _megdnn_tensor_in indices, _megdnn_tensor_out grad,
        _megdnn_workspace workspace) {
    const auto len = indices.layout[0];
    auto step = 0;
    for (size_t i = 1; i < diff.layout.ndim; ++i) {
        step += diff.layout[i];
    }
    if (step <= 0)
        step = 1;
    switch (diff.layout.dtype.enumv()) {
#define cb(DType)                                                              \
    case DTypeTrait<DType>::enumv:                                             \
        random::shuffle_backward<DTypeTrait<DType>::ctype>(                    \
                diff.ptr<DTypeTrait<DType>::ctype>(), indices.ptr<dt_int32>(), \
                grad.ptr<DTypeTrait<DType>::ctype>(), len, step, m_stream);    \
        break;
        ARGSORT_FOREACH_CTYPE(cb)
#undef cb
        default:
            megdnn_throw("bad dtype");
    }
}

// vim: syntax=cpp.doxygen
