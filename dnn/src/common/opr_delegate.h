#pragma once

#include "megdnn/handle.h"
#include "megdnn/oprs/base.h"

#include "src/common/utils.h"

namespace megdnn {

/*!
 * \brief get a handle that dispatches to caller cpu thread
 *
 * Usually used for calling other opr impls from some opr impl. You probably
 * want to use CpuOprDelegationStorage instead.
 */
MGE_WIN_DECLSPEC_FUC const std::shared_ptr<Handle>& inplace_cpu_handle(
        int debug_level = 0);

/*!
 * \brief storage for oprs on inplace CPU handle
 *
 * This class takes care of thread safety and destruction order. Usage example:
 *
 *      MatrixMul* get_matmul() {
 *          static CpuOprDelegationStorage<> storage;
 *          return storage.get<MatrixMul>();
 *      }
 */
template <int nr_opr = 1>
class CpuOprDelegationStorage {
    DNN_MUTEX m_mtx;
    std::shared_ptr<Handle> m_handle;
    std::unique_ptr<OperatorBase> m_oprs[nr_opr];

public:
    ~CpuOprDelegationStorage();

    template <typename Opr, int idx = 0>
    Opr* get(const typename Opr::Param& param = {});
};

template <int nr_opr>
CpuOprDelegationStorage<nr_opr>::~CpuOprDelegationStorage() = default;

template <int nr_opr>
template <typename Opr, int idx>
Opr* CpuOprDelegationStorage<nr_opr>::get(const typename Opr::Param& param) {
    static_assert(idx < nr_opr, "invalid idx");
    if (!m_oprs[idx]) {
        MEGDNN_LOCK_GUARD(m_mtx);
        if (!m_oprs[idx]) {
            if (!m_handle) {
                m_handle = inplace_cpu_handle();
            }
            auto opr = m_handle->create_operator<Opr>();
            megdnn_assert(opr->is_thread_safe());
            opr->param() = param;
            m_oprs[idx] = std::move(opr);
        }
    }
    return static_cast<Opr*>(m_oprs[idx].get());
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen
