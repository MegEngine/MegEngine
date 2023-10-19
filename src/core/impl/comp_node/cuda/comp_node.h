#pragma once

#include "../impl_helper.h"

namespace mgb {
class CudaCompNode final : public CompNodeImplHelper {
public:
    static constexpr Flag sm_flag =
            Flag::QUEUE_LIMITED | Flag::HAS_COPY_STREAM | Flag::SUPPORT_UNIFIED_ADDRESS;

    class CompNodeImpl;
    class EventImpl;

    //! whether cuda comp node is available
    static bool available();

    static void try_coalesce_all_free_memory();
    static void foreach (thin_function<void(CompNode)> callback);
    static void finalize();
    static size_t get_device_count(bool warn = true);
    static Impl* load_cuda(const Locator& locator, const Locator& locator_logical);
    static void sync_all();
    static DeviceProperties get_device_prop(int dev);
    static size_t get_device_left_memory(int dev);

    static void set_prealloc_config(
            size_t alignment, size_t min_req, size_t max_overhead,
            double growth_factor);
#if MGB_CUDA && defined(WIN32)
    //! FIXME: windows cuda driver shutdown before call atexit function
    //! even register atexit function after init cuda driver! as a
    //! workround recovery resource by OS temporarily, may need remove
    //! this after upgrade cuda runtime
    static bool is_into_atexit;
#endif
};
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
