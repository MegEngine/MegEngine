#pragma once

#include <memory>
#include <vector>
#include "megdnn/oprs.h"
#include "src/common/conv_bias.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"
#include "test/common/checker.h"
#include "test/common/index.h"

namespace megdnn {
namespace test {

//! simulation the task dispatch progress
class CpuRecordDispatcher : public MegcoreCPUDispatcher {
    std::vector<MegcoreCPUDispatcher::Task> tasks;
    bool execute_inplace = false;

public:
    void dispatch(MultiThreadingTask&& task, size_t parallelism) override {
        if (execute_inplace) {
            for (size_t i = 0; i < parallelism; i++) {
                task(i, 0);
            }
        } else {
            tasks.push_back([task, parallelism]() {
                for (size_t i = 0; i < parallelism; i++) {
                    task(i, 0);
                }
            });
        }
    }

    void dispatch(Task&& task) override {
        // printf("dispatch one task with execute_inplace = %d\n", execute_inplace);
        if (execute_inplace) {
            task();
        } else {
            tasks.push_back(task);
        };
    }

    size_t nr_threads() override { return 1_z; }

    void sync() override {}

    void enable_execute_inplace() { execute_inplace = true; }

    void disable_execute_inplace() { execute_inplace = false; }

    void run_task() {
        // printf("size of task : %zu\n", tasks.size());
        for (auto&& task : tasks) {
            task();
        }
    }
    void clear_task() { tasks.clear(); }
};

template <typename Opr, typename Proxy = OprProxy<Opr>>
class TaskRecordChecker : public CheckerHelper {
    std::shared_ptr<CpuRecordDispatcher> m_dispatcher;
    std::unique_ptr<Handle> m_handle;
    Proxy m_naive_proxy, m_cur_proxy;

public:
    using Param = typename Opr::Param;
    using CheckerHelper::CheckerHelper;

    TaskRecordChecker(int debug_level = 0) {
        m_dispatcher = std::make_shared<CpuRecordDispatcher>();
        m_handle = create_cpu_handle_with_dispatcher(debug_level, m_dispatcher);
    }

    TensorLayoutArray make_layouts(const TensorShapeArray& shapes) {
        TensorLayoutArray layouts(shapes.size());
        for (size_t i = 0; i < shapes.size(); ++i) {
            DType dt =
                    (m_dtype.find(i) != m_dtype.end() ? m_dtype[i] : dtype::Float32());
            TensorFormat fmt =
                    (m_fmt.find(i) != m_fmt.end() ? m_fmt[i] : TensorFormat{});
            layouts[i] = TensorLayout(shapes[i], dt, fmt);
        }
        return layouts;
    }

    /*!
     * \brief execute opr on current param/dtype/rng config
     * \param shapes input/output shapes, which would be passed as
     *      arguments to Opr::deduce_layout
     *
     * Checker would construct TensorLayout vectors from shapes and dtypes,
     * and call exec(TensorLayoutArray &).
     */
    TaskRecordChecker& exec(const TensorShapeArray& shapes) {
        exec(make_layouts(shapes));
        return *this;
    }

    void exec(TensorLayoutArray layouts);

    //! explicitly require argument to be TensorShape
    TaskRecordChecker& execs(const TensorShapeArray& shapes) { return exec(shapes); }

    //! explicitly require argument to be TensorLayout
    TaskRecordChecker& execl(const TensorLayoutArray& layouts) {
        exec(layouts);
        return *this;
    }

    TaskRecordChecker& set_param(Param p) {
        m_param = p;
        opr()->param() = p;
        return *this;
    }
    TaskRecordChecker& set_dtype(size_t idx, DType dtype) {
        m_dtype[idx] = dtype;
        return *this;
    }
    TaskRecordChecker& set_rng(size_t idx, RNG* rng) {
        m_rng[idx] = rng;
        return *this;
    }

    TaskRecordChecker& set_epsilon(dt_float32 epsilon) {
        m_epsilon = epsilon;
        m_max_avg_error = epsilon;
        m_max_avg_biased_error = epsilon;
        return *this;
    }

    TaskRecordChecker& set_proxy(const Proxy& proxy) {
        m_naive_proxy = proxy;
        m_cur_proxy = proxy;
        return *this;
    }

    //! get the opr impl so setting other than param() can be modified
    Opr* opr() {
        if (!m_opr_cur) {
            m_opr_cur = m_handle->create_operator<Opr>();
        }
        return m_opr_cur.get();
    }

    void free_opr() {
        if (m_opr_cur) {
            m_opr_cur.reset();
        }
    }

    Handle* get_handle() {
        megdnn_assert(m_handle);
        return m_handle.get();
    }

    void copy_tensors(
            const CheckerHelper::TensorValueArray& dest,
            const CheckerHelper::TensorValueArray& src) {
        megdnn_assert(dest.size() == src.size());
        for (size_t i = 0; i < src.size(); i++) {
            auto&& tensor = src[i];
            if (tensor.layout.ndim == 0)
                continue;
            auto layout = tensor.layout;
            auto span = layout.span();
            auto dst_ptr = static_cast<dt_byte*>(dest[i].raw_ptr()) + span.low_byte;
            auto src_ptr =
                    static_cast<const dt_byte*>(src[i].raw_ptr()) + span.low_byte;
            memcpy(dst_ptr, src_ptr, span.dist_byte());
        }
    }

private:
    Param m_param;
    Proxy m_proxy;
    std::unique_ptr<Opr> m_opr_cur;
    std::shared_ptr<TensorValueArray> m_tensors_first, m_tensors_second,
            m_tensors_truth;

    std::vector<void*> m_recovery_ptrs;

    void init_host_values();

    void change_tensor_ptr(
            std::shared_ptr<TensorValueArray> des,
            std::shared_ptr<TensorValueArray> src, std::vector<void*>&);

    void recovery_tensor_ptr(
            std::shared_ptr<TensorValueArray> src, const std::vector<void*>&);
};

template <typename Opr, typename Proxy>
void TaskRecordChecker<Opr, Proxy>::exec(TensorLayoutArray layouts) {
    auto opr_cur = this->opr();
    opr_cur->param() = m_param;

    m_proxy.deduce_layout(opr_cur, layouts);
    for (size_t i = 0; i < layouts.size(); ++i) {
        if (layouts[i].dtype == dtype::Byte()) {
            layouts[i] = TensorLayout(layouts[i], dtype::Int8());
        }
    }

    // allocate input
    m_tensors_truth = alloc_tensors(m_handle.get(), layouts, 0);
    m_tensors_first = alloc_tensors(m_handle.get(), layouts, 0);
    m_tensors_second = alloc_tensors(m_handle.get(), layouts, 0);

    init_host_values();

    copy_tensors(*m_tensors_first, *m_tensors_truth);
    copy_tensors(*m_tensors_second, *m_tensors_truth);

    m_dispatcher->enable_execute_inplace();
    m_proxy.exec(opr_cur, *m_tensors_truth);

    m_dispatcher->clear_task();
    m_dispatcher->disable_execute_inplace();
    //! record the task
    m_proxy.exec(opr_cur, *m_tensors_first);
    m_dispatcher->run_task();

    //! if check record2, the opr should be free
    // free_opr();
    check_tensors(*m_tensors_truth, *m_tensors_first);

    //! change the src and out ptr and run again
    change_tensor_ptr(m_tensors_first, m_tensors_second, m_recovery_ptrs);
    m_dispatcher->run_task();
    check_tensors(*m_tensors_truth, *m_tensors_second);

    m_dispatcher->clear_task();
    recovery_tensor_ptr(m_tensors_first, m_recovery_ptrs);
    m_recovery_ptrs.clear();
}

template <typename Opr, typename Proxy>
void TaskRecordChecker<Opr, Proxy>::init_host_values() {
    for (size_t i = 0; i < m_tensors_truth->size(); ++i) {
        auto&& tensor = (*m_tensors_truth)[i];
        auto rng = m_rng[i];
        if (!rng)
            rng = m_default_rng.get();
        rng->gen(tensor);
    }
}
template <typename Opr, typename Proxy>
void TaskRecordChecker<Opr, Proxy>::change_tensor_ptr(
        std::shared_ptr<TensorValueArray> des, std::shared_ptr<TensorValueArray> src,
        std::vector<void*>& recovery_ptrs) {
    for (size_t i = 0; i < des->size(); ++i) {
        auto&& tensor_dest = (*des)[i];
        auto&& tensor_src = (*src)[i];
        megdnn_assert(tensor_dest.layout.eq_layout(tensor_src.layout));
        recovery_ptrs.push_back(tensor_dest.raw_ptr());
        tensor_dest.reset_ptr(tensor_src.raw_ptr());
    }
}

template <typename Opr, typename Proxy>
void TaskRecordChecker<Opr, Proxy>::recovery_tensor_ptr(
        std::shared_ptr<TensorValueArray> src,
        const std::vector<void*>& recovery_ptrs) {
    megdnn_assert(src->size() == recovery_ptrs.size());
    for (size_t i = 0; i < src->size(); ++i) {
        auto&& tensor_src = (*src)[i];
        tensor_src.reset_ptr(recovery_ptrs[i]);
    }
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
