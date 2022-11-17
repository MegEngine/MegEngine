#include <optional>
#include <type_traits>

#include "algo_chooser.h"
#include "megbrain/comp_node.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/imperative/blob_manager.h"
#include "megbrain/imperative/physical_tensor.h"
#include "megbrain/imperative/utils/helper.h"
#include "megbrain/imperative/utils/platform.h"
#include "megbrain/rdnn/management.h"
#include "megdnn/basic_types.h"

namespace mgb {
namespace imperative {

/*!
 * /brief Helps deduce layout and dtype
 */
template <typename Opr>
class DnnOprDeducer {
private:
    Opr* m_opr;

public:
    DnnOprDeducer(Opr* opr) : m_opr(opr) { mgb_assert(opr); }

    // FIXME: maybe in-place style deduction works better
    template <typename... TArgs>
    TensorLayout deduce_layout(TArgs&&... args) {
        // static_assert((std::is_convertible_v<TArgs, TensorLayout> && ...));
        TensorLayout output_layout;
        m_opr->deduce_layout(args..., output_layout);
        return output_layout;
    }

    template <typename... TArgs>
    TensorLayout deduce_layout_fallible(TArgs&&... args) {
        // static_assert((std::is_convertible_v<TArgs, TensorLayout> && ...));
        TensorLayout output_layout;
        bool success = (args.ndim * ...) > 0;
        if (success) {
            m_opr->deduce_layout(args..., output_layout);
        } else {
            m_opr->deduce_dtype(args.dtype..., output_layout.dtype);
        }
        return output_layout;
    }

    template <size_t nr_outputs, typename... TArgs>
    std::array<TensorLayout, nr_outputs> deduce_layouts(TArgs&&... args) {
        // static_assert((std::is_convertible_v<TArgs, TensorLayout> && ...));
        std::array<TensorLayout, nr_outputs> layouts;
        std::apply(
                [&](auto&&... outputs) { m_opr->deduce_layout(args..., outputs...); },
                layouts);
        return layouts;
    }
};

/*!
 * /brief Declare an abstract operator and initialize it's param
 */
template <typename Opr>
class DnnOprStub {
private:
    // TODO: make opr concrete
    std::aligned_storage_t<sizeof(Opr), alignof(Opr)> m_storage;

    using Param = typename Opr::Param;

private:
    DnnOprStub() { new (&param()) Param(); }

public:
    DnnOprStub(const Param& param) { this->param() = param; }

    // undefined behavior
    Opr& opr() { return *reinterpret_cast<Opr*>(&m_storage); }

    auto& param() { return opr().param(); }

    auto& param() const { return opr().param(); }

    ~DnnOprStub() { param().~Param(); }
};

/*!
 * /brief Deduce layout without create concrete opr
 */
template <typename Opr>
class DnnOprHelper : public DnnOprStub<Opr>, public DnnOprDeducer<Opr> {
private:
    using Stub = DnnOprStub<Opr>;
    using Deducer = DnnOprDeducer<Opr>;

public:
    DnnOprHelper(const typename Opr::Param& param)
            : Stub(param), Deducer(&Stub::opr()) {}
};

// hold a concrete operator in given comp_node
template <typename Opr>
class DnnOprHolder {
private:
    CompNode m_comp_node;
    opr::intl::UniqPtrWithCN<Opr> m_opr =
            opr::intl::create_megdnn_opr<Opr>(m_comp_node);

public:
    DnnOprHolder(CompNode comp_node) : m_comp_node(comp_node) {}

    auto& op() { return m_opr; }

    auto comp_node() { return m_comp_node; }

    auto& param() { return m_opr->param(); }

    auto& param() const { return m_opr->param(); }

    ~DnnOprHolder() {
        using DT = CompNode::DeviceType;

        if (m_comp_node.device_type() == DT::CPU &&
            m_comp_node != CompNode::default_cpu()) {
            CompNodeEnv::from_comp_node(m_comp_node)
                    .cpu_env()
                    .dispatch([p = m_opr.release()] { delete p; });
        }
    }
};

/*!
 * /brief Prevent binary float
 */
class DnnOprCallerBase {
protected:
    static auto&& get_layout(const megdnn::TensorND& tensor) { return tensor.layout; }

    static auto get_layout(const megdnn::TensorNDArray& tensors) {
        SmallVector<TensorLayout> layouts;
        for (auto&& tensor : tensors) {
            layouts.push_back(tensor.layout);
        }
        return layouts;
    }
};

/*!
 * \brief A struct for safely calling DNN oprs
 *
 * In some cases, op may be released before the complete of the execution
 * This destructor will prevent this
 */
template <typename Opr>
class DnnOprCaller final : public DnnOprHolder<Opr>,
                           public DnnOprDeducer<Opr>,
                           public DnnOprCallerBase {
private:
    using Holder = DnnOprHolder<Opr>;
    using Deducer = DnnOprDeducer<Opr>;
    using Base = DnnOprCallerBase;

    std::optional<DnnTensorND> m_workspace;
    std::optional<megdnn::param::ExecutionPolicy> m_policy;

    megdnn::Workspace create_workspace(size_t sz) {
        mgb_assert(
                !m_workspace, "workspace asked more than once by op: %s",
                demangled_typename<Opr>());
        dt_byte* ptr = nullptr;
        if (sz) {
            TensorLayout layout({sz}, dtype::Byte());
            m_workspace.emplace(
                    Tensor::make(layout, Holder::comp_node())->dnn_tensor());
            ptr = reinterpret_cast<dt_byte*>(m_workspace->raw_ptr());
        }
        return {ptr, sz};
    }

public:
    using Param = typename Opr::Param;

    DnnOprCaller(CompNode cn) : Holder(cn), Deducer(Holder::op().get()) {}

    DnnOprCaller(CompNode cn, const Param& param) : DnnOprCaller(cn) {
        Holder::param() = param;
    }

    DnnOprCaller(CompNode cn, const Param& param, megdnn::param::ExecutionPolicy policy)
            : DnnOprCaller(cn, param) {
        m_policy.emplace(policy);
    }

    /**
     * /brief Convert TensorPtr args to megdnn::TensorND and call f
     *
     */
    template <typename TFunctor, typename... TArgs>
    auto call_dnn(TFunctor&& f, TArgs&&... args) {
        std::optional<SmallVector<std::shared_ptr<dt_byte>>> input_ptrs;
        // recursive convert:
        // 1. TensorPtr to DnnTensorND (subclass of megdnn::TensorND) ;
        // 2. DeviceTensorND, HostTensorND to megdnn::TensorND ;
        // 3. SmallVector of above to SmallVector<megdnn::TensorND> .
        auto to_dnn = [&](auto&& arg, auto&& to_dnn) {
            using T = decltype(arg);
            if constexpr (std::is_convertible_v<T, TensorPtr>) {
                return arg->dnn_tensor();
            } else if constexpr (
                    std::is_convertible_v<T, DeviceTensorND> ||
                    std::is_convertible_v<T, HostTensorND>) {
                return arg.as_megdnn();
            } else if constexpr (
                    std::is_convertible_v<T, megdnn::TensorND> ||
                    std::is_convertible_v<T, SmallVector<megdnn::TensorND>>) {
                return std::forward<T>(arg);
            } else if constexpr (is_small_vector_v<std::decay_t<T>>) {
                using TItem = std::decay_t<decltype(to_dnn(arg[0], to_dnn))>;
                SmallVector<megdnn::TensorND> dnn_tensors;
                for (auto&& tensor : arg) {
                    if constexpr (std::is_same_v<TItem, DnnTensorND>) {
                        if (!input_ptrs) {
                            input_ptrs.emplace();
                        }
                        auto dnn_tensor = to_dnn(tensor, to_dnn);
                        input_ptrs->push_back(std::move(dnn_tensor.reference));
                        dnn_tensors.push_back(std::move(dnn_tensor));
                    } else if constexpr (std::is_same_v<TItem, megdnn::TensorND>) {
                        dnn_tensors.push_back(to_dnn(tensor, to_dnn));
                    } else {
                        static_assert(!std::is_same_v<TItem, TItem>);
                    }
                }
                return dnn_tensors;
            } else {
                static_assert(!std::is_same_v<T, T>);
            }
        };
        return f(to_dnn(std::forward<TArgs>(args), to_dnn)...);
    }

    // common execution (opr->exec(inputs..., outputs...))
    template <typename... TArgs>
    void exec(TArgs&&... args) {
        call_dnn(
                [this](auto&&... args) {
                    Holder::op()->exec(std::forward<decltype(args)>(args)...);
                },
                std::forward<TArgs>(args)...);
    }

    // execution fastrun opr
    // (opr->exec(inputs..., outputs..., create_ws(setup_algo(...))))
    template <typename... TArgs>
    void exec_fastrun(TArgs&&... args) {
        call_dnn(
                [&](auto&&... args) {
                    using FixedTensorLayouts =
                            typename rdnn::AlgoChooser<Opr>::FixedTensorLayouts;
                    SmallVector<megdnn::TensorND> dnn_inputs = {args...};
                    mgb_assert(m_policy, "policy not set");
                    size_t workspace_size = setup_algo<Opr>(
                            FixedTensorLayouts{args.layout...}, Holder::op().get(), 0,
                            false, false, Holder::comp_node(), *m_policy, false,
                            &dnn_inputs);
                    Holder::op()->exec(
                            std::forward<decltype(args)>(args)...,
                            create_workspace(workspace_size));
                },
                std::forward<TArgs>(args)...);
    }

    // execute with fixed workspace
    // (opr->exec(input..., outputs..., create_ws(get_workspace_in_bytes(...))))
    template <typename... TArgs>
    void exec_with_ws(TArgs&&... args) {
        call_dnn(
                [&](auto&&... args) {
                    size_t workspace_size =
                            Holder::op()->get_workspace_in_bytes(get_layout(args)...);
                    Holder::op()->exec(
                            std::forward<decltype(args)>(args)...,
                            create_workspace(workspace_size));
                },
                std::forward<TArgs>(args)...);
    }

    // execute dynamic out opr
    // (opr->exec(inputs..., outputs... create_ws(get_workspace_in_bytes(...)), alloc))
    template <size_t nr_out, typename... TArgs>
    auto exec_dynout(TArgs&&... args) {
        struct Alloc final : public megdnn::DynOutMallocPolicy {
            CompNode comp_node;
            std::array<TensorPtr, nr_out> output_tensors;
            std::array<std::optional<DnnTensorND>, nr_out> output_dnn_tensors;

        public:
            Alloc(CompNode comp_node) : comp_node(comp_node) {}
            megdnn::TensorND alloc_output(
                    size_t id, DType dtype, const TensorShape& shape,
                    void* user_data) override {
                TensorLayout layout(shape, dtype);
                output_tensors[id] = Tensor::make(layout, comp_node);
                output_dnn_tensors[id].emplace(
                        output_tensors[id]->dnn_tensor());  // pin output
                return *output_dnn_tensors[id];
            }

            void* alloc_workspace(size_t sz, void* user_data) override {
                mgb_assert(false);
            }

            void free_workspace(void* ptr, void* user_data) override {
                mgb_assert(false);
            }
        } alloc{Holder::comp_node()};
        call_dnn(
                [&](auto&&... args) {
                    size_t workspace_size =
                            Holder::op()->get_workspace_in_bytes(get_layout(args)...);
                    Holder::op()->exec(
                            std::forward<decltype(args)>(args)...,
                            create_workspace(workspace_size), &alloc);
                },
                std::forward<TArgs>(args)...);
        return alloc.output_tensors;
    }
};

}  // namespace imperative
}  // namespace mgb
