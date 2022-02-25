#pragma once

#include "./impl.h"

namespace mgb {
namespace cg {

class StaticMemAllocPushdown final : public StaticMemAllocImplHelper {
    class BestfitPrealloc;

    size_t m_peak_usage = 0;

    /*!
     * intervals that lie directly below this interval; address of each interval
     * is max end address of those in below. Indexed by interval ID
     */
    std::vector<IntervalPtrArray> m_interval_below;

    /*!
     * \brief compute topology order of inervals; result represented in
     *      m_interval_below
     */
    void init_topo_order();

    size_t get_interval_addr_end(Interval* interval);

public:
    void do_solve() override;

    size_t tot_alloc() const override { return m_peak_usage; }
};

}  // namespace cg
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
