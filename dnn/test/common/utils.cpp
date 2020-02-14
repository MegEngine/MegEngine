/**
 * \file dnn/test/common/utils.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/common/utils.h"
#include "megdnn/basic_types.h"
#include "test/common/random_state.h"
#include "test/common/memory_manager.h"
#include "src/naive/handle.h"
#include "megcore.h"

#include <cmath>
#include <random>

using namespace megdnn;
using namespace test;

namespace {

void megdnn_memcpy_internal(Handle *handle, void *dst, const void *src,
        size_t size_in_bytes, megcoreMemcpyKind_t kind)
{
    auto comp_handle = handle->megcore_computing_handle();
    megcore_check(megcoreMemcpy(comp_handle, dst, src, size_in_bytes,
                kind));
    megcore_check(megcoreSynchronize(comp_handle));
}

class ErrorHandlerImpl final: public ErrorHandler {
    static ErrorHandlerImpl inst;
    void do_on_megdnn_error(const std::string &msg) override {
        fprintf(stderr, "megdnn error: %s\n", msg.c_str());
#if MEGDNN_ENABLE_EXCEPTIONS
        throw MegDNNError{msg};
#else
        megdnn_trap();
#endif
    }

    void do_on_tensor_reshape_error(const std::string &msg) override {
        fprintf(stderr, "tensor reshape error: %s\n", msg.c_str());
#if MEGDNN_ENABLE_EXCEPTIONS
        throw TensorReshapeError{msg};
#else
        megdnn_trap();
#endif
    }

    public:
        ErrorHandlerImpl() {
            ErrorHandler::set_handler(this);
        }
};

ErrorHandlerImpl ErrorHandlerImpl::inst;

} // anonymous namespace

CpuDispatchChecker::InstCounter CpuDispatchChecker::sm_inst_counter;

std::unique_ptr<Handle> test::create_cpu_handle(int debug_level,
                                                bool check_dispatch,
                                                TaskExecutorConfig* config) {
    std::shared_ptr<MegcoreCPUDispatcher> dispatcher(nullptr);
    if (check_dispatch) {
        dispatcher = CpuDispatchChecker::make(config);
    }
    return create_cpu_handle_with_dispatcher(debug_level, dispatcher);
}

std::unique_ptr<Handle> test::create_cpu_handle_with_dispatcher(int debug_level,
        const std::shared_ptr<MegcoreCPUDispatcher> &dispatcher)
{
    megcoreDeviceHandle_t dev_handle;
    megcore_check(megcoreCreateDeviceHandle(&dev_handle,
                megcorePlatformCPU));
    megcoreComputingHandle_t comp_handle;
    if (dispatcher) {
        megcore_check(megcoreCreateComputingHandleWithCPUDispatcher(
                    &comp_handle, dev_handle, dispatcher));
    } else {
        megcore_check(megcoreCreateComputingHandle(&comp_handle, dev_handle));
    }
    auto destructor = [=]() {
        megcore_check(megcoreDestroyComputingHandle(comp_handle));
        megcore_check(megcoreDestroyDeviceHandle(dev_handle));
    };
    auto ret = Handle::make(comp_handle, debug_level);
    ret->set_destructor(destructor);
    return ret;
}

void test::megdnn_sync(Handle *handle)
{
    auto comp_handle = handle->megcore_computing_handle();
    megcore_check(megcoreSynchronize(comp_handle));
}

void* test::megdnn_malloc(Handle *handle, size_t size_in_bytes)
{
    auto mm = MemoryManagerHolder::instance()->get(handle);
    return mm->malloc(size_in_bytes);
}

void test::megdnn_free(Handle *handle, void *ptr)
{
    auto mm = MemoryManagerHolder::instance()->get(handle);
    mm->free(ptr);
}

void test::megdnn_memcpy_D2H(Handle *handle, void *dst, const void *src,
        size_t size_in_bytes)
{
    megdnn_memcpy_internal(handle, dst, src, size_in_bytes,
            megcoreMemcpyDeviceToHost);
}

void test::megdnn_memcpy_H2D(Handle *handle, void *dst, const void *src,
        size_t size_in_bytes)
{
    megdnn_memcpy_internal(handle, dst, src, size_in_bytes,
            megcoreMemcpyHostToDevice);
}

void test::megdnn_memcpy_D2D(Handle *handle, void *dst, const void *src,
        size_t size_in_bytes)
{
    megdnn_memcpy_internal(handle, dst, src, size_in_bytes,
            megcoreMemcpyDeviceToDevice);
}

TensorND DynOutMallocPolicyImpl::alloc_output(
        size_t /*id*/, DType dtype, const TensorShape &shape,
        void * /*user_data*/) {
    auto ptr = megdnn_malloc(m_handle, dtype.size() * shape.total_nr_elems());
    return {ptr, TensorLayout{shape, dtype}};
}

void* DynOutMallocPolicyImpl::alloc_workspace(size_t sz, void * /*user_data*/) {
    return megdnn_malloc(m_handle, sz);
}

void DynOutMallocPolicyImpl::free_workspace(void *ptr, void * /*user_data*/) {
    megdnn_free(m_handle, ptr);
}

std::shared_ptr<void> DynOutMallocPolicyImpl::make_output_refholder(
        const TensorND &out) {
    using namespace std::placeholders;
    auto deleter = std::bind(megdnn_free, m_handle, _1);
    return {out.raw_ptr, deleter};
}

NaivePitchAlignmentScope::NaivePitchAlignmentScope(size_t alignment)
        : m_orig_val{naive::HandleImpl::exchange_image2d_pitch_alignment(
                  alignment)},
          m_new_val{alignment} {}

NaivePitchAlignmentScope::~NaivePitchAlignmentScope() {
    auto r = naive::HandleImpl::exchange_image2d_pitch_alignment(m_orig_val);
    megdnn_assert(r == m_new_val);
}

size_t test::get_cpu_count() {
    return std::max<size_t>(std::thread::hardware_concurrency(), 1_z);
}

// vim: syntax=cpp.doxygen
