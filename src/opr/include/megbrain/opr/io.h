/**
 * \file src/opr/include/megbrain/opr/io.h
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
#include "megbrain/opr/internal/param_tag_defs.h"

#include "megdnn/opr_param_defs.h"

namespace mgb {
namespace opr {

namespace intl {
/*!
 * \brief base class for IO nodes between device and host
 */
class HostIONodeBase: public cg::SingleCNOperatorNodeBase {
    void init_output_static_infer_desc() override final;

    protected:
        using cg::SingleCNOperatorNodeBase::SingleCNOperatorNodeBase;

        /*!
         * \brief src_type for static shape and value infer
         */
        virtual cg::static_infer::SourceType static_infer_src_type() const;

        virtual const TensorShape& get_output_shape() = 0;

        /*!
         * \brief fill value in *dest* for static inference
         * \param dest static inference result; nullptr to check whether static
         *      infer enabled
         * \return whether static inference is successful; if *dest* is nullptr,
         *      return whether static inference is enabled
         */
        virtual bool fill_in_static_infer(DeviceTensorND *dest) = 0;
};

/*!
 * \brief base class for oprs that hold a device tensor
 */
class DeviceTensorHolder: public HostIONodeBase {
    class DevValueExecDep;

    void init_output_format() override;
    void init_output_mem_plan(bool dynamic) override final;
    void scn_do_execute() override final;
    void record_execute_deps(ExecDependencyArray& deps) override;

    protected:
        using HostIONodeBase::HostIONodeBase;

        virtual const DeviceTensorND& get_dev_tensor() const = 0;
        void add_output(DType dtype);
};

/*!
 * \brief base class for SharedDeviceTensor and VolatileSharedDeviceTensor
 *
 * Why differentiating SharedDeviceTensor/VolatileSharedDeviceTensor:
 * 1. SharedDeviceTensor has constant shape, so for graphs with lots of params,
 *    their shapes need not to be checked
 * 2. They have different load/dump strategies, so we use different operator
 *    classes rather than add meta-parameters.
 */
MGB_DEFINE_CLS_WITH_SUPER(SharedDeviceTensorBase, DeviceTensorHolder) // {
    std::shared_ptr<DeviceTensorND> m_dev_data;
    bool m_const_value;

    const TensorShape& get_output_shape() override;

    bool fill_in_static_infer(DeviceTensorND* dest) override {
        MGB_MARK_USED_VAR(dest);
        return false;
    }

    void init_output_comp_node() override;

    public:
        //! const_value marks whether the device value of this operator should
        //! be treated as constant during graph execution. Should be false in
        //! most cases.
        SharedDeviceTensorBase(ComputingGraph &graph,
                const std::shared_ptr<DeviceTensorND> &dev_data,
                bool const_value,
                const OperatorNodeConfig &config);

        const DeviceTensorND& get_dev_tensor() const override {
            return *m_dev_data;
        }

        const std::shared_ptr<DeviceTensorND>& dev_data() const {
            return m_dev_data;
        }

        bool const_value() const { return m_const_value; }
};

/*!
 * \brief Base class for producing multiple outputs corresponding to multiple
 * device tensors
 *
 * This opr is used to speed up inference by packing params together.
 * This operator assumes the device tensors are constant.
 */
MGB_DEFINE_CLS_WITH_SUPER(MultipleDeviceTensorHolderBase,
                          cg::OperatorNodeBase)  // {
    class DevValuesExecDep;
public:
    using ValueArray = SmallVector<std::shared_ptr<DeviceTensorND>>;
    MultipleDeviceTensorHolderBase(ComputingGraph& graph, ValueArray values,
                                   const OperatorNodeConfig& config);
    const ValueArray& values() const { return m_values; }

protected:
    ValueArray m_values;

private:
    void record_execute_deps(ExecDependencyArray& deps) override;
    void do_execute(ExecEnv &env) override;
    void init_output_mem_plan(bool dynamic) override;
    void on_output_comp_node_stream_changed() override;
    void init_output_comp_node() override;
    void init_output_static_infer_desc() override;
    NodeProp* do_make_node_prop() const override;
};

} // namespace intl

/*!
 * \brief copy from host to device
 *
 * Note:
 *
 * 1. If the underlying comp node of host tensor is on CPU and it has the same
 *    mem node of output comp node, and allow_cpu_mem_fwd is set tu true, then
 *    host data would be forwarded into graph directly.
 *    1.1 No synchronization is performed, meaning that even if host
 *        data is on cpu0 but Host2DeviceCopy is on cpu1, the memory is still
 *        directly forwarded.
 *    1.2 If host data pointer changes, static memory reallocation would be
 *        triggered.
 * 2. If host data is not contiguous, it would be relayouted on host.
 */
MGB_DEFINE_OPR_CLASS(Host2DeviceCopy, intl::HostIONodeBase) // {
    class HostValueExecDep;
    public:
        struct Param {
            static constexpr uint32_t TAG = param_tag::HOST2DEVICE_COPY;

            //! whether to enable static value inference; usually disabled in
            //! case of invalid initial host_data
            bool enable_value_infer;

            //! whether to dump current value in host_data when this opr is
            //! serialized
            bool dump_default_value;

            //! whether to forward memory of Host2DeviceCopy if it is on CPU
            bool allow_cpu_mem_fwd;

            Param(bool enable_value_infer_ = true,
                    bool dump_default_value_ = false,
                    bool allow_cpu_mem_fwd_ = true):
                enable_value_infer{enable_value_infer_},
                dump_default_value{dump_default_value_},
                allow_cpu_mem_fwd{allow_cpu_mem_fwd_}
            {}
        };

        Host2DeviceCopy(ComputingGraph &graph,
                const std::shared_ptr<HostTensorND> &host_data,
                const Param &param,
                const OperatorNodeConfig &config);

        static SymbolVar make(ComputingGraph &graph,
                const std::shared_ptr<HostTensorND> &host_data,
                const OperatorNodeConfig &config = {}) {
            return make(graph, host_data, {}, config);
        }

        static SymbolVar make_no_value_infer(ComputingGraph &graph,
                const std::shared_ptr<HostTensorND> &host_data,
                const OperatorNodeConfig &config = {}) {
            return make(graph, host_data, {false}, config);
        }

        static SymbolVar make_no_fwd(ComputingGraph &graph,
                const std::shared_ptr<HostTensorND> &host_data,
                const OperatorNodeConfig &config = {}) {
            Param p;
            p.allow_cpu_mem_fwd = false;
            return make(graph, host_data, p, config);
        }

        static SymbolVar make(ComputingGraph &graph,
                const std::shared_ptr<HostTensorND> &host_data,
                const Param &param,
                const OperatorNodeConfig &config);

        /*!
         * \brief get underlying host data
         */
        const std::shared_ptr<HostTensorND>& host_data() const {
            return m_host_data;
        }

        const Param& param() const {
            return m_param;
        }

        void record_execute_deps(ExecDependencyArray& deps) override;

    private:
        //! whether to forward memory in HostTensorND; used on CPU
        bool m_fwd_host_mem;
        const Param m_param;
        std::shared_ptr<HostTensorND> m_host_data;

        //! whether need to sync  to m_host_data_dev_cont in next exec
        mutable bool m_host_data_dev_cont_need_sync = false;
        //! cached DeviceTensorND used by get_dev_tensor()
        mutable DeviceTensorND m_host_data_dev_proxy, m_host_data_dev_cont;

        void init_output_mem_plan(bool dynamic) override final;
        void scn_do_execute() override;
        void init_output_comp_node() override;
        const TensorShape& get_output_shape() override;
        bool fill_in_static_infer(DeviceTensorND *dest) override;
        const DeviceTensorND& get_dev_tensor_in_mem_fwd() const;
        NodeProp* do_make_node_prop() const override;
};

/*!
 * \brief wrapper for device tensor to make it accessible in the computing graph
 *
 * This is mainly used for NN parameters.
 *
 * Note: after creating the node, the shape and mem pointer of dev_data should
 * not change again
 *
 * \see intl::SharedDeviceTensorBase and VolatileSharedDeviceTensor
 */
MGB_DEFINE_OPR_CLASS(SharedDeviceTensor, intl::SharedDeviceTensorBase) // {
    cg::static_infer::SourceType static_infer_src_type() const override;

    public:
        using Super::Super;

        static SymbolVar make(ComputingGraph& graph,
                              const std::shared_ptr<DeviceTensorND>& dev_data,
                              bool const_value,
                              const OperatorNodeConfig& config);

        static SymbolVar make(ComputingGraph& graph,
                              const std::shared_ptr<DeviceTensorND>& dev_data,
                              const OperatorNodeConfig& config = {}) {
            return make(graph, dev_data, false, config);
        }

        static SymbolVar make_const(
                ComputingGraph& graph,
                const std::shared_ptr<DeviceTensorND>& dev_data,
                const OperatorNodeConfig& config = {}) {
            return make(graph, dev_data, true, config);
        }

        /*!
         * \brief make a SharedDeviceTensor by first coping from host to device
         *
         * See SharedDeviceTensorBase::SharedDeviceTensorBase for const_value.
         */
        static SymbolVar make(ComputingGraph& graph, const HostTensorND& value,
                              bool const_value,
                              const OperatorNodeConfig& config);

        static SymbolVar make(ComputingGraph& graph, const HostTensorND& value,
                              const OperatorNodeConfig& config = {}) {
            return make(graph, value, false, config);
        }

        static SymbolVar make_const(ComputingGraph& graph,
                                    const HostTensorND& value,
                                    const OperatorNodeConfig& config = {}) {
            return make(graph, value, false, config);
        }
};

/*!
 * \brief a SharedDeviceTensor with non-default tensor format
 *
 * This opr is usually used in serialized models.
 */
MGB_DEFINE_OPR_CLASS(
        SharedDeviceTensorWithFormat, intl::SharedDeviceTensorBase) // {
    cg::static_infer::SourceType static_infer_src_type() const override;
public:
    using Super::Super;

    void init_output_format() override;

    static SymbolVar make(ComputingGraph& graph,
                          const std::shared_ptr<DeviceTensorND>& dev_data,
                          bool const_value, const OperatorNodeConfig& config);

    static SymbolVar make(ComputingGraph& graph,
                          const std::shared_ptr<DeviceTensorND>& dev_data,
                          const OperatorNodeConfig& config = {}) {
        return make(graph, dev_data, false, config);
    }

    static SymbolVar make_const(ComputingGraph& graph,
                                const std::shared_ptr<DeviceTensorND>& dev_data,
                                const OperatorNodeConfig& config = {}) {
        return make(graph, dev_data, true, config);
    }
};

/*!
 * \brief like SharedDeviceTensor but allows the mem ptr or shape to change
 *
 * This is mainly used for directly forwarding a given input pointer into the
 * computing graph.
 *
 * \see intl::SharedDeviceTensorBase and SharedDeviceTensor
 */
MGB_DEFINE_OPR_CLASS(
        VolatileSharedDeviceTensor, intl::SharedDeviceTensorBase) // {
    NodeProp* do_make_node_prop() const override;

    public:
        using Super::Super;

        static SymbolVar make(ComputingGraph &graph,
                const std::shared_ptr<DeviceTensorND> &dev_data,
                const OperatorNodeConfig &config = {});

        //! adapter for io.sereg.h: opr_shallow_copy_shared_device_tensor
        static SymbolVar make(ComputingGraph& graph,
                              const std::shared_ptr<DeviceTensorND>& dev_data,
                              bool const_value,
                              const OperatorNodeConfig& config) {
            mgb_assert(!const_value);
            return make(graph, dev_data, config);
        }
};

/*!
 * \brief tensor with immutable value
 */
MGB_DEFINE_OPR_CLASS(ImmutableTensor, intl::DeviceTensorHolder) // {
    public:
        class Value;
        class DevValueCache;

        ImmutableTensor(ComputingGraph &graph,
                const Value &value,
                const OperatorNodeConfig &config);
        ~ImmutableTensor() noexcept;

        static SymbolVar make(ComputingGraph &graph, const HostTensorND &val,
                const OperatorNodeConfig &config = {});

        //! make from DTypeScalar; comp node must be provided in config
        static SymbolVar make(ComputingGraph &graph, const DTypeScalar &val,
                const OperatorNodeConfig &config);

        //! get underlying value on device
        const DeviceTensorND& value() const;

        SymbolVar shallow_copy(
                ComputingGraph &graph, const OperatorNodeConfig &config) const {
            return make_from_value(graph, m_value, m_value_refkeep, config);
        }
    private:
        const Value &m_value;
        //! refkeep is used if value is not stored in DevValueCache
        std::shared_ptr<Value> m_value_refkeep;

        static SymbolVar make_from_value(
                ComputingGraph &graph, const Value &val,
                const std::shared_ptr<Value> &val_refkeep,
                const OperatorNodeConfig &config);

        void init_output_comp_node() override;
        const TensorShape& get_output_shape() override;
        bool fill_in_static_infer(DeviceTensorND *dest) override;
        const DeviceTensorND& get_dev_tensor() const override;
        cg::static_infer::SourceType static_infer_src_type() const override;
};

/*!
 * \brief copy a tensor on device, possibly across computing nodes
 *
 * To copy to different computing node, specify the destination in
 * OperatorNodeConfig.
 *
 * Output var would be placed on copy stream by default.
 */
MGB_DEFINE_OPR_CLASS(Copy, cg::SingleCNIOSameShapeOperatorNodeBase) // {

    bool m_mem_fwd_success = false;

    void scn_do_execute() override;
    NodeProp* do_make_node_prop() const override;
    void mem_plan_fwd_in2out_readonly() override;

    void add_input_layout_constraint() override;
    void init_output_comp_node() override;
    void init_output_static_infer_desc() override;
    void init_rt_force_dynamic_mem_alloc_imply_chain() override;

    public:
        Copy(VarNode *inp, const OperatorNodeConfig &config);
        static SymbolVar make(SymbolVar inp,
                const OperatorNodeConfig &config = {});

        // for serialization
        using Param = megdnn::param::Empty;
        Param param() const {
            return {};
        }
        static SymbolVar make(SymbolVar inp,
                Param, const OperatorNodeConfig &config) {
            return make(inp, config);
        }
};

/*!
 * \brief wrapper for multi device tensor
 *
 * \see intl::MultipleDeviceTensorHolderBase
 */
MGB_DEFINE_OPR_CLASS(MultipleDeviceTensorHolder,
                     intl::MultipleDeviceTensorHolderBase)  // {
public:
    using Super::Super;
    static SymbolVarArray make(ComputingGraph& graph, ValueArray values,
                               const OperatorNodeConfig& config = {});
};

/*!
 * \brief a MultipleDeviceTensorHolder with non-default tensor format
 *
 * \see intl::MultipleDeviceTensorHolderBase
 */
MGB_DEFINE_OPR_CLASS(MultipleDeviceTensorWithFormatHolder,
                     intl::MultipleDeviceTensorHolderBase)  // {
public:
    using Super::Super;
    static SymbolVarArray make(ComputingGraph& graph, ValueArray values,
                               const OperatorNodeConfig& config = {});

private:
    void init_output_format() override;
};

} // namespace opr
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
