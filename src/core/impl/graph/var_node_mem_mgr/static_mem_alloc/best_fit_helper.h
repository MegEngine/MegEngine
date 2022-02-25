#pragma once

#include "./impl.h"
#include "megbrain/utils/thin/function.h"

namespace mgb {
namespace cg {

struct BestFitHelper {
    using Interval = StaticMemAllocImplHelper::Interval;
    thin_function<void(Interval*)> alloc;
    thin_function<void(Interval* dest, size_t offset, Interval*)> alloc_overwrite;
    thin_function<void(Interval*)> free;

    /*!
     * \brief run on intervals and call corresponding methods
     */
    void run(const StaticMemAllocImplHelper::IntervalPtrArray& intervals);
};

}  // namespace cg
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
