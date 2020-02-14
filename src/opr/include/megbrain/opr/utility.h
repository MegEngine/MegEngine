/**
 * \file src/opr/include/megbrain/opr/utility.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/graph.h"
#include "megbrain/graph/event.h"
#include "megbrain/opr/internal/identical_fwd.h"
#include "megbrain/opr/internal/param_tag_defs.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/opr/param_defs.h"

#include "megdnn/oprs/utils.h"

namespace mgb {
namespace opr {

#if !MGB_BUILD_SLIM_SERVING
/*!
 * \brief sleep for specific time on device
 */
MGB_DEFINE_OPR_CLASS(Sleep, cg::SingleCNIOSameShapeOperatorNodeBase) // {
    public:
        /*!
         * \brief directly sleep without constructing an opr
         */
        static void sleep(const CompNode &node, double seconds);

        //! sleep type: device or host or both
        struct Type {
            bool device, host;

            Type(bool d = true, bool h = false):
                device(d), host(h)
            {
            }
        };

        Sleep(VarNode *node, double seconds, Type type,
                const OperatorNodeConfig &config);

        static SymbolVar make(SymbolVar node, double seconds,
                Type type = {}, const OperatorNodeConfig &config = {});

        // for serialization
        struct Param {
            static constexpr auto TAG = param_tag::SLEEP;
            double seconds;
            Type type;
        };

        Param param() const {
            return {m_seconds, m_type};
        }

        static SymbolVar make(SymbolVar node, const Param &param,
                const OperatorNodeConfig &config) {
            return make(node, param.seconds, param.type, config);
        }
    private:
        double m_seconds;
        Type m_type;
        intl::UniqPtrWithCN<megdnn::Sleep> m_opr;

        void scn_do_execute() override;
        void record_execute_deps(ExecDependencyArray& deps) override;

};

/*!
 * \brief record the device time when this opr kernel is executed
 *
 * Note: the time is measured in seconds; time is only available after the
 * function has been waited. All Timestamp operators in the same graph on the
 * same computing node are based on the same reference point.
 *
 * \param dest the tensor to write timing information; it must be 1-dimensional
 *      and has float32 dtype
 * \param dest_off the offset on which \p dest should be modified; this helps
 *      multiple Timestamp operator instances
 */
MGB_DEFINE_OPR_CLASS(Timestamp, intl::ForwardInputToOutput) // {
    public:
        Timestamp(VarNode* node, std::shared_ptr<HostTensorND> dest,
                  size_t dest_off, const OperatorNodeConfig& config);

        static SymbolVar make(SymbolVar node,
                              std::shared_ptr<HostTensorND> dest,
                              size_t dest_off,
                              const OperatorNodeConfig& config = {});

    private:
        class GraphStorage;
        std::shared_ptr<HostTensorND> m_dest;
        size_t m_dest_off;
        CompNode::Event* m_first_event = nullptr;
        std::unique_ptr<CompNode::Event> m_event;

        void scn_do_execute_finish(const DeviceTensorND&) override;
        void on_output_comp_node_stream_changed() override;
        void add_input_layout_constraint() override;

        //! called from wait event handler to update timestamp values
        void update();
};

/*!
 * \brief To make sure inputs' owner oprs finished when executing this operator,
 *      and forwarding input(0) to output.
 */
MGB_DEFINE_OPR_CLASS(VirtualDep, intl::ForwardInputToOutput) // {
public:
    VirtualDep(const VarNodeArray& inputs, const OperatorNodeConfig& config);

    static SymbolVar make(const SymbolVarArray& inputs,
            const OperatorNodeConfig& config = {});

    NodeProp* do_make_node_prop() const override;
//    void add_input(std::initializer_list<VarNode*> list);
};

#endif  // MGB_BUILD_SLIM_SERVING

/*!
 * \brief do not provide any static infer on a var to mark it dynamic; used for
 *      debug purposes
 */
MGB_DEFINE_OPR_CLASS(MarkDynamicVar, cg::SingleCNOperatorNodeBase) // {
    void scn_do_execute() override;
    void init_output_static_infer_desc() override {}
    NodeProp* do_make_node_prop() const override;

    public:
        using Param = megdnn::param::Empty;

        MarkDynamicVar(VarNode *node, const OperatorNodeConfig &config);

        static SymbolVar make(
                SymbolVar node, const OperatorNodeConfig &config = {});

        // for serialization
        Param param() const {
            return {};
        }
        static SymbolVar make(SymbolVar node,
                const Param &, const OperatorNodeConfig &config) {
            return make(node, config);
        }
};

/*!
 * \brief inject a callback to be called whenever this operator is executed
 */
MGB_DEFINE_OPR_CLASS(CallbackInjector, intl::ForwardInputToOutput) // {

    void scn_do_execute_finish(const DeviceTensorND &val) override;
    cg::static_infer::ValueInferDesc mixin_get_static_infer_desc(OperatorNodeBase &opr) override;
    NodeProp* do_make_node_prop() const override;

    public:
        using Callback = thin_function<void(DeviceTensorND&)>;
        using MultiCallback = thin_function<void(SmallVector<DeviceTensorND>&)>;
        struct Param {
            //! whether to allow auto duplication (to be used with sublinear
            //! memory)
            bool allow_auto_dup = false;

            //! whether to ignore side effect (so this opr can be optimized out
            //! if input is constant)
            bool ignore_side_effect = false;

            //! whether to invoke the callback during static value inference
            bool invoke_for_static_infer = true;

            MultiCallback callback;

            explicit Param(Callback cb) :  callback{[cb](SmallVector<DeviceTensorND>& a){cb(a.at(0));}} {}

            Param(bool allow_auto_dup_, Callback cb)
                    : allow_auto_dup{allow_auto_dup_},
                      callback{[cb](SmallVector<DeviceTensorND>& a){cb(a.at(0));}} {}

            Param(bool allow_auto_dup_, bool ignore_side_effect_, Callback cb)
                    : allow_auto_dup{allow_auto_dup_},
                      ignore_side_effect{ignore_side_effect_},
                      callback{[cb](SmallVector<DeviceTensorND>& a){cb(a.at(0));}} {}

            explicit Param(MultiCallback cb) : callback{std::move(cb)} {}

            Param(bool allow_auto_dup_, MultiCallback cb)
                    : allow_auto_dup{allow_auto_dup_},
                      callback{std::move(cb)} {}

            Param(bool allow_auto_dup_, bool ignore_side_effect_, MultiCallback cb)
                    : allow_auto_dup{allow_auto_dup_},
                      ignore_side_effect{ignore_side_effect_},
                      callback{std::move(cb)} {}
        };

        CallbackInjector(VarNode *inp, const Param &param,
                const OperatorNodeConfig &config);

        CallbackInjector(VarNodeArray& inp, const Param &param,
                const OperatorNodeConfig &config);

        //! create the operator disallowing auto dup
        static SymbolVar make(SymbolVar inp, const Callback &cb,
                              const OperatorNodeConfig &config = {}) {
            return make((SymbolVarArray){inp}, Param{cb}, config);
        }

        static SymbolVar make(SymbolVar inp, const Param &param,
                      const OperatorNodeConfig &config = {}) {
            return make((SymbolVarArray){inp}, param, config);
        }

        static SymbolVar make(SymbolVar inp, const MultiCallback &cb,
                              const OperatorNodeConfig &config = {}) {
            return make((SymbolVarArray){inp}, Param{cb}, config);
        }

        static SymbolVar make(SymbolVarArray inp, const MultiCallback &cb,
                      const OperatorNodeConfig &config = {}) {
            return make(inp, Param{cb}, config);
        }

        static SymbolVar make(SymbolVarArray inp, const Param &param,
                      const OperatorNodeConfig &config = {});

        const Param& param() const {
            return m_param;
        }

    private:
        int m_warn_printed = 0;
        Param m_param;
};

/*!
 * \brief assert its output would not be broadcasted when involved in elemwise
 *      arith
 *
 * Useful for removing the reduce when computing grad, so graph optimizer can
 * work well.
 */
MGB_DEFINE_OPR_CLASS(MarkNoBroadcastElemwise, intl::ForwardInputToOutput) // {
    public:
        using Param = megdnn::param::Empty;
        MarkNoBroadcastElemwise(
                VarNode* input, const OperatorNodeConfig &config);

        static SymbolVar make(SymbolVar input,
                const OperatorNodeConfig &config = {});

        // for serialization
        Param param() const {
            return {};
        }
        static SymbolVar make(SymbolVar node,
                const Param &, const OperatorNodeConfig &config) {
            return make(node, config);
        }
};

/*!
 * \brief does nothing but forward input to output
 *
 * Currently only used for preventing graph optimizer from removing some var so
 * its gradient can be correctly computed.
 */
MGB_DEFINE_OPR_CLASS(Identity, intl::ForwardInputToOutput) // {
    public:
        using Param = megdnn::param::Empty;
        Identity(VarNode* input, const OperatorNodeConfig &config);

        static SymbolVar make(SymbolVar input,
                const OperatorNodeConfig &config = {});

        // for serialization
        Param param() const {
            return {};
        }
        static SymbolVar make(SymbolVar node,
                const Param &, const OperatorNodeConfig &config) {
            return make(node, config);
        }
};

/*!
 * \brief assert that two vars are equal; this opr would sync the stream when
 *      executing
 *
 * raise UnequalError during exec if tensor not equal
 */
MGB_DEFINE_OPR_CLASS(AssertEqual, intl::ForwardInputToOutput) // {
    bool m_throw_on_error = true;
    HostTensorND m_hv;

    void scn_do_execute_finish(const DeviceTensorND &) override;

    public:
        using Param = megdnn::param::AssertEqual;

        //! \p expect and \p get are only used for error message
        AssertEqual(VarNode *expect, VarNode *get, VarNode *err,
                const Param &param,
                const OperatorNodeConfig &config);

        static SymbolVar make(
                SymbolVar expect, SymbolVar get, const Param &param = {},
                const OperatorNodeConfig &config = {});

        //! do not throw exception on check error
        void disable_throw_on_error() {
            m_throw_on_error = false;
        }

        //! for serialization and shallow copy
        static SymbolVar make(
                SymbolVar expect, SymbolVar get, SymbolVar err,
                const Param &param, const OperatorNodeConfig &config);

        const Param& param() const {
            return m_param;
        }

        class UnequalError final: public MegBrainError {
            public:
                using MegBrainError::MegBrainError;
        };

    private:
        Param m_param;
};

#if MGB_ENABLE_GRAD

/*!
 * \brief output equals to input, but grad(input) would be replaced by return
 *      value of given callback at runtime
 */
MGB_DEFINE_OPR_CLASS(SetGrad, intl::ForwardInputToOutput) // {
    public:
        using GradGetter = thin_function<SymbolVar(const SetGrad &)>;

        SetGrad(VarNode* input, const GradGetter& grad_getter,
                const OperatorNodeConfig &config);

        static SymbolVar make(SymbolVar input, const GradGetter& grad_getter,
                const OperatorNodeConfig &config = {});

        const GradGetter& grad_getter() const {
            return m_grad_getter;
        }

        //! GradGetter for zero grad
        static SymbolVar zero_grad(const SetGrad &) {
            return {};
        }

    private:
        GradGetter m_grad_getter;

};

/*!
 * \brief get a special marker for a grad being invalid
 */
MGB_DEFINE_OPR_CLASS(InvalidGrad, cg::SingleCNIOSameShapeOperatorNodeBase) // {
    const OperatorNodeBase* m_grad_opr;
    size_t m_inp_idx;

    void add_input_layout_constraint() override;

    void scn_do_execute() override;

    public:
        //! \p vinp should be grad_opr.input(inp_idx), unless in shallow copy
        InvalidGrad(VarNode* vinp, const OperatorNodeBase* grad_opr,
                    size_t inp_idx);

        static VarNode* make(const OperatorNodeBase& grad_opr, size_t inp_idx);

        size_t inp_idx() const {
            return m_inp_idx;
        }

        const OperatorNodeBase* grad_opr() const {
            return m_grad_opr;
        }
};

/*!
 * \brief denote the gradient of a var w.r.t. another var, which would be
 *      expanded to real grad during the gopt::ExpandVirtualGradPass
 *
 * This operator exists so graph optimization can be performed without actual
 * grad oprs. This operator must be expanded before graph execution.
 */
MGB_DEFINE_OPR_CLASS(VirtualGrad, cg::OperatorNodeBase) // {
    void do_execute(ExecEnv &) override;
    void init_output_comp_node() override;
    void init_output_static_infer_desc() override;
    void on_output_comp_node_stream_changed() override;
    NodeProp* do_make_node_prop() const override;

    public:
        using Param = megdnn::param::Empty;

        VirtualGrad(VarNode *target, VarNode *wrt,
                const OperatorNodeConfig &config);


        Param param() const {
            return {};
        }
        static SymbolVar make(SymbolVar target, SymbolVar wrt,
                Param param = {}, const OperatorNodeConfig &config = {});
};

/*!
 * \brief Construct a loss var with specific gradients
 *
 * The gradient w.r.t. \p ys[i] would be \p y_grads[i]
 */
MGB_DEFINE_OPR_CLASS(VirtualLoss, cg::OperatorNodeBase) // {
    void do_execute(ExecEnv&) override;
    void init_output_comp_node() override;
    void init_output_static_infer_desc() override;
    void on_output_comp_node_stream_changed() override;
    NodeProp* do_make_node_prop() const override;

public:
    using Param = megdnn::param::Empty;

    //! the first half of \p inputs contain ys, and the remaining are y_grads
    VirtualLoss(const VarNodeArray& inputs, const OperatorNodeConfig& config);

    static SymbolVar make(const SymbolVarArray& ys,
                          const SymbolVarArray& y_grads,
                          Param param = {},
                          const OperatorNodeConfig& config = {});

    Param param() const { return {}; }
};

#else
class InvalidGrad {
public:
    using OperatorNodeBase = cg::OperatorNodeBase;
    [[noreturn]] static VarNode* make(const OperatorNodeBase& grad_opr,
                                      size_t inp_idx);
};
#endif // MGB_ENABLE_GRAD

/*!
 * \brief allocate output storage as a persistent storage
 *
 * This operator allocates a persistent storage (i.e. one that does not depend
 * on graph runtime memory allocator) prior to execution and copies input value
 * to it when the opr is executed. It is usually used for eliminating dynamic
 * memory allocation when multiple comp nodes are involved but some of them do
 * not support dynamic memory alloc/dealloc (e.g. hexagon).
 *
 * Note:
 *  1. Memory sharing is manually controlled by \p share_key
 *  2. Input shapes must be static.
 *
 * \see VarNode::Flag::NO_MEM_RECLAIM for eliminating only dynamic memory
 *      deallocation
 */
MGB_DEFINE_OPR_CLASS(PersistentOutputStorage,
                     cg::SingleCNIOSameShapeOperatorNodeBase) // {
public:
    using Param = megdnn::param::PersistentOutputStorage;

    PersistentOutputStorage(VarNode* inp, const Param& param,
                            const OperatorNodeConfig& config);

    static SymbolVar make(SymbolVar inp, const Param& param = {},
                          const OperatorNodeConfig& config = {});

    const Param& param() const { return m_param; }

private:
    class DevValueExecDep;
    class StorageHolder;

    DeviceTensorND m_dev_tensor;
    Param m_param;

    void init_output_mem_plan(bool dynamic) override final;
    void scn_do_execute() override final;
    void record_execute_deps(ExecDependencyArray& deps) override;

};

MGB_DEFINE_OPR_CLASS(RequireInputDynamicStorage,
                     intl::ForwardInputToOutput)  // {
public:
    RequireInputDynamicStorage(VarNode* input,
                               const OperatorNodeConfig& config);
    static SymbolVar make(SymbolVar input,
                          const OperatorNodeConfig& config = {});
};

} // namespace opr
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
