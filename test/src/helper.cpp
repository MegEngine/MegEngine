/**
 * \file test/src/helper.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#include "./rng_seed.h"

#include "megbrain/test/helper.h"
#include "megbrain/utils/hash.h"
#include "megbrain/utils/debug.h"
#include "megbrain/utils/persistent_cache.h"
#include "megbrain/comp_node_env.h"

#include <atomic>
#include <random>

#include <cmath>
#include <cstring>
#include <cstdlib>

#if MGB_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

using namespace mgb;

const dt_qint8 UniformRNGDefaultRange<dtype::QuantizedS8>::LO = dt_qint8{-128};
const dt_qint8 UniformRNGDefaultRange<dtype::QuantizedS8>::HI = dt_qint8{127};

bool megdnn::operator == (const TensorLayout &a, const TensorLayout &b) {
    if (a.ndim != b.ndim)
        return false;
    // check all shapes and strides equal, including shape-1 dims
    for (size_t i = 0; i < a.ndim; ++ i) {
        if (a[i] != b[i] || a.stride[i] != b.stride[i])
            return false;
    }
    return true;
}

uint64_t mgb::next_rand_seed() {
    return RNGSeedManager::inst().next_seed();
}

void mgb::set_rand_seed(uint64_t seed) {
    RNGSeedManager::inst().set_seed(seed);
}

RNGxorshf::RNGxorshf(uint64_t seed) {
    std::mt19937_64 gen(seed);
    s[0] = gen();
    s[1] = gen();
}


/* ========================== HostTensorGenerator ========================== */
template<typename dtype>
std::shared_ptr<HostTensorND> HostTensorGenerator<
dtype, RandomDistribution::GAUSSIAN>::operator ()(
        const TensorShape &shape, CompNode cn) {
    if (!cn.valid())
        cn = CompNode::load("xpu0");
    std::shared_ptr<HostTensorND> ret =
        std::make_shared<HostTensorND>(cn, shape, dtype());
    auto ptr = ret->ptr<ctype>();
    auto mean = m_mean, std = m_std;
    for (size_t i = 0, it = shape.total_nr_elems(); i < it; i += 2) {
        ctype u1 = ctype((m_rng() + 1.0) / (m_rng.max() + 1.0)),
              u2 = ctype((m_rng() + 1.0) / (m_rng.max() + 1.0)),
              r = ctype(std * std::sqrt(-2 * std::log(u1))),
              theta = ctype(2 * M_PI * u2),
              z0 = ctype(r * std::cos(theta) + mean),
              z1 = ctype(r * std::sin(theta) + mean);
        ptr[i] = z0;
        ptr[std::min(i + 1, it - 1)] = z1;
    }
    return ret;
}

template<typename dtype>
std::shared_ptr<HostTensorND> HostTensorGenerator<
dtype, RandomDistribution::UNIFORM>::operator ()(
        const TensorShape &shape, CompNode cn) {
    if (!cn.valid())
        cn = CompNode::load("xpu0");
    std::shared_ptr<HostTensorND> ret =
        std::make_shared<HostTensorND>(cn, shape, dtype());
    auto ptr = ret->ptr<ctype>();
    double scale = (m_hi - m_lo) / (m_rng.max() + 1.0);
    for (size_t i = 0, it = shape.total_nr_elems(); i < it; ++ i) {
        ptr[i] = m_rng() * scale + m_lo;
    }
    return ret;
}

template<typename dtype>
std::shared_ptr<HostTensorND> HostTensorGenerator<
dtype, RandomDistribution::CONSTANT>::operator ()(
        const TensorShape &shape, CompNode cn) {
    if (!cn.valid())
        cn = CompNode::load("xpu0");
    std::shared_ptr<HostTensorND> ret =
        std::make_shared<HostTensorND>(cn, shape, dtype());
    auto ptr = ret->ptr<ctype>();
    for (size_t i = 0, it = shape.total_nr_elems(); i < it; ++ i) {
        ptr[i] = m_default_val;
    }
    return ret;
}

template<typename dtype>
std::shared_ptr<HostTensorND> HostTensorGenerator<
dtype, RandomDistribution::CONSECUTIVE>::operator ()(
        const TensorShape &shape, CompNode cn) {
    if (!cn.valid())
        cn = CompNode::load("xpu0");
    std::shared_ptr<HostTensorND> ret =
        std::make_shared<HostTensorND>(cn, shape, dtype());
    auto ptr = ret->ptr<ctype>();
    for (size_t i = 0, it = shape.total_nr_elems(); i < it; ++ i) {
        ptr[i] = m_val + i * m_delta;
    }
    return ret;
}

// explicit instantialization of HostTensorGenerator
namespace mgb {
    template class HostTensorGenerator<
        dtype::Float32, RandomDistribution::GAUSSIAN>;
    template class HostTensorGenerator<
        dtype::Float32, RandomDistribution::UNIFORM>;
    template class HostTensorGenerator<
        dtype::Float32, RandomDistribution::CONSTANT>;
    template class HostTensorGenerator<
        dtype::Float32, RandomDistribution::CONSECUTIVE>;
    template class HostTensorGenerator<
        dtype::Float16, RandomDistribution::GAUSSIAN>;
    template class HostTensorGenerator<
        dtype::Int8, RandomDistribution::UNIFORM>;
    template class HostTensorGenerator<
        dtype::Int8, RandomDistribution::CONSTANT>;
    template class HostTensorGenerator<
        dtype::Int8, RandomDistribution::CONSECUTIVE>;
    template class HostTensorGenerator<
        dtype::Uint8, RandomDistribution::UNIFORM>;
    template class HostTensorGenerator<
        dtype::Uint8, RandomDistribution::CONSTANT>;
    template class HostTensorGenerator<
        dtype::Int16, RandomDistribution::UNIFORM>;
    template class HostTensorGenerator<
        dtype::Int16, RandomDistribution::CONSTANT>;
    template class HostTensorGenerator<
        dtype::Int32, RandomDistribution::UNIFORM>;
    template class HostTensorGenerator<
        dtype::Int32, RandomDistribution::CONSTANT>;
    std::shared_ptr<HostTensorND>
    HostTensorGenerator<dtype::Bool, RandomDistribution::UNIFORM>::
    operator()(const TensorShape& shape, CompNode cn) {
        if (!cn.valid())
            cn = CompNode::load("xpu0");
        auto dtype = dtype::Bool();
        std::shared_ptr<HostTensorND> ret =
                std::make_shared<HostTensorND>(cn, shape, dtype);
        auto ptr = ret->ptr<dt_bool>();
        for (size_t i = 0, it = shape.total_nr_elems(); i < it; ++i) {
            ptr[i] = (i % 2 == 1);
        }
        return ret;
    }

    std::shared_ptr<HostTensorND>
    HostTensorGenerator<dtype::QuantizedS8, RandomDistribution::UNIFORM>::
    operator()(const TensorShape& shape, CompNode cn) {
        if (!cn.valid())
            cn = CompNode::load("xpu0");
        auto dtype = dtype::QuantizedS8(m_scale);
        auto param = dtype.param();
        std::shared_ptr<HostTensorND> ret =
                std::make_shared<HostTensorND>(cn, shape, dtype);
        auto ptr = ret->ptr<dt_qint8>();
        double scale = (param.dequantize(m_hi) - param.dequantize(m_lo)) /
                       (m_rng.max() + 1.0);
        for (size_t i = 0, it = shape.total_nr_elems(); i < it; ++i) {
            ptr[i] = param.quantize(m_rng() * scale + param.dequantize(m_lo));
        }
        return ret;
    }

    std::shared_ptr<HostTensorND>
    HostTensorGenerator<dtype::Quantized8Asymm, RandomDistribution::UNIFORM>::
    operator()(const TensorShape& shape, CompNode cn) {
        if (!cn.valid())
            cn = CompNode::load("xpu0");
        auto dtype = dtype::Quantized8Asymm(m_scale, m_zero_point);
        auto param = dtype.param();
        std::shared_ptr<HostTensorND> ret =
                std::make_shared<HostTensorND>(cn, shape, dtype);
        auto ptr = ret->ptr<dt_quint8>();
        double scale = (param.dequantize(m_hi) - param.dequantize(m_lo)) /
                       (m_rng.max() + 1.0);
        for (size_t i = 0, it = shape.total_nr_elems(); i < it; ++i) {
            ptr[i] = param.quantize(m_rng() * scale + param.dequantize(m_lo));           
        }
        return ret;
    }
}

::testing::AssertionResult mgb::__assert_float_equal(
        const char *expr0, const char *expr1, const char * /*expr_maxerr*/,
        float v0, float v1, float maxerr) {
    float err = fabs(v0 - v1) / std::max<float>(
            1, std::min(fabs(v0), fabs(v1)));
    if (std::isfinite(v0) && std::isfinite(v1) && err < maxerr) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << ssprintf(
            "Value of: %s\n"
            "  Actual: %.6g\n"
            "Expected: %s\n"
            "Which is: %.6g\n"
            "   Error: %.4e", expr1, v1, expr0, v0, err);
}

::testing::AssertionResult mgb::__assert_tensor_equal(
        const char *expr0, const char *expr1, const char * /*expr_maxerr*/,
        const HostTensorND &v0, const HostTensorND &v1, float maxerr) {
    auto ret = debug::compare_tensor_value(v0, expr0, v1, expr1, maxerr);
    if (ret.valid())
        return ::testing::AssertionFailure() << ret.val();
    return ::testing::AssertionSuccess();
}

::testing::AssertionResult mgb::__assert_shape_equal(const TensorShape& v0,
                                                const TensorShape& v1) {
    if (v0.eq_shape(v1))
        return ::testing::AssertionSuccess()
                << v0.to_string() << " == " << v1.to_string();
    else
        return ::testing::AssertionFailure()
                << v0.to_string() << " != " << v1.to_string();
}

#if WIN32
#include <io.h>
#include <fcntl.h>
#include <direct.h>
#define getcwd _getcwd
namespace {
    auto mkdir(const char *path, int) {
        return _mkdir(path);
    }

    int mkstemp(char *tpl){
        tpl = _mktemp(tpl);
        mgb_assert(tpl);
        auto fd = _open(tpl, _O_TEMPORARY | _O_RDWR);
        mgb_assert(fd > 0, "failed to open %s: %s", tpl, strerror(errno));
        return fd;
    }
}
#else
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif


NamedTemporaryFile::NamedTemporaryFile()
{
    char name[256];
    strcpy(name, output_file("mgb-test-XXXXXX", false).c_str());
    m_fd = mkstemp(name);
    mgb_throw_if(m_fd == -1, MegBrainError,
            "failed to open temp file `%s': %m", name);
    m_fpath = name;
    mgb_log_debug("opened temporary file: %s", name);
}

NamedTemporaryFile::~NamedTemporaryFile() {
#ifdef WIN32
    _unlink(m_fpath.c_str());
#else
    unlink(m_fpath.c_str());
#endif
}

#if defined(IOS)
#pragma message "build test on iOS; need ios_get_mgb_output_dir() to be defined"
extern "C" void ios_get_mgb_output_dir(char **dir);
#endif

std::string mgb::output_file(const std::string &fname, bool check_writable) {
    static std::string cwd;
    static std::mutex cwd_mtx;
    MGB_LOCK_GUARD(cwd_mtx);
    if (cwd.empty()) {
#if defined(IOS)
        char *buf = nullptr;
        ios_get_mgb_output_dir(&buf);
#else
        auto buf = getcwd(nullptr, 0);
#endif
        mgb_assert(buf);
        cwd = buf;
        free(buf);
        cwd.append("/output");
        mgb_log("use test output dir: %s", cwd.c_str());
        mkdir(cwd.c_str(), 0755);
    }
    if (fname.empty())
        return cwd;
    auto ret = cwd + "/" + fname;
    if (check_writable) {
        FILE *fout = fopen(ret.c_str(), "w");
        mgb_assert(fout, "failed to open %s: %s", ret.c_str(), strerror(errno));
        fclose(fout);
    }
    return ret;
}

std::vector<CompNode> mgb::load_multiple_xpus(size_t num) {
    auto cn0 = CompNode::load("xpu0");
    if (CompNode::get_device_count(cn0.device_type()) < num) {
        cn0 = CompNode::load("cpu0");
    }
    std::vector<CompNode> ret{cn0};
    auto loc = cn0.locator();
    for (size_t i = 1; i < num; ++ i) {
        loc.device = i;
        ret.push_back(CompNode::load(loc));
    }
    return ret;
}

bool mgb::check_gpu_available(size_t num) {
    if (CompNode::get_device_count(CompNode::DeviceType::CUDA) < num) {
        mgb_log_warn("skip test case that requires %zu GPU(s)", num);
        return false;
    }
    return true;
}

bool mgb::check_amd_gpu_available(size_t num) {
    if (CompNode::get_device_count(CompNode::DeviceType::ROCM) < num) {
        mgb_log_warn("skip test case that requires %zu AMD GPU(s)", num);
        return false;
    }
    return true;
}

bool mgb::check_cambricon_device_available(size_t num) {
    if (CompNode::get_device_count(CompNode::DeviceType::CAMBRICON) < num) {
        mgb_log_warn("skip test case that requires %zu cambricon device(s)",
                     num);
        return false;
    }
    return true;
}

bool mgb::check_device_type_avaiable(CompNode::DeviceType device_type) {
    switch (device_type) {
        case mgb::CompNode::DeviceType::CUDA:
        case mgb::CompNode::DeviceType::CPU:
        case mgb::CompNode::DeviceType::CAMBRICON:
        case mgb::CompNode::DeviceType::ATLAS:
        case mgb::CompNode::DeviceType::MULTITHREAD:
            return true;
        default:
            return false;
    }
    return false;
}

bool mgb::check_compute_capability(int major, int minor) {
#if MGB_CUDA
    int dev;
    MGB_CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    MGB_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    return prop.major > major || (prop.major == major && prop.minor >= minor);
#else
    MGB_MARK_USED_VAR(major);
    MGB_MARK_USED_VAR(minor);
    return false;
#endif
}

void mgb::write_tensor_to_file(const HostTensorND &hv,
        const char *fname, char mode) {
    mgb_assert(hv.layout().is_contiguous());
    char modefull[] = {mode, 'b', '\x00'};
    FILE *fout = fopen(fname, modefull);
    mgb_assert(fout, "failed to open %s: %s", fname, strerror(errno));
    fprintf(fout, "%s %zu", hv.dtype().name(), hv.shape().ndim);
    for (size_t i = 0; i < hv.shape().ndim; ++ i) {
        fprintf(fout, " %zu", hv.shape(i));
    }
    fprintf(fout, "\n");
    auto size = hv.layout().span().dist_byte();
    auto wr = fwrite(hv.raw_ptr(), 1, size, fout);
    mgb_assert(size == wr);
    mgb_log("write tensor: %zu bytes (%s) to %s", size,
            hv.shape().to_string().c_str(), fname);
    fclose(fout);
}

cg::ComputingGraph::OutputSpecItem
mgb::make_callback_copy(SymbolVar dev, HostTensorND &host, bool sync) {
    auto cb = [sync, &host](DeviceTensorND &d) {
        host.copy_from(d);
        if (sync) {
            host.sync();
        }
    };
    return {dev, cb};
}

/* ========================== PersistentCacheHook ========================== */
class PersistentCacheHook::HookedImpl final : public PersistentCache {
    GetHook m_on_get;

public:
    std::shared_ptr<PersistentCache> orig_impl;

    HookedImpl(GetHook on_get) : m_on_get{std::move(on_get)} {}

    Maybe<Blob> get(const std::string& category, const Blob& key) override {
        auto ret = orig_impl->get(category, key);
        m_on_get(category, key.ptr, key.size, ret.valid() ? ret->ptr : 0,
                 ret.valid() ? ret->size : 0);
        return ret;
    }

    void put(const std::string& category, const Blob& key,
             const Blob& value) override {
        orig_impl->put(category, key, value);
    }
};

PersistentCacheHook::PersistentCacheHook(GetHook on_get)
        : m_impl{std::make_shared<HookedImpl>(std::move(on_get))} {
    m_impl->orig_impl = PersistentCache::set_impl(m_impl);
}

PersistentCacheHook::~PersistentCacheHook() {
    PersistentCache::set_impl(std::move(m_impl->orig_impl));
}

#if !MGB_ENABLE_EXCEPTION
#pragma message "some tests would be disabled because exception is disabled"
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
