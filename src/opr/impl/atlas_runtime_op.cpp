#include "megbrain/opr/atlas_runtime_op.h"
#include <memory>
#include "megbrain/common.h"
#include "megbrain/graph/operator_node.h"
#include "megdnn/basic_types.h"
#include "megdnn/dtype.h"

#if MGB_ATLAS
#include "acl/acl_mdl.h"

using namespace mgb;
using namespace opr;

namespace {
/**
 * \brief get mgb shape from acl shape, batch from mgb
 */
TensorShape acl_shape_to_mgb_shape_for_output(
        aclmdlDesc* model_desc, size_t output_idx, size_t output_dtype_size,
        aclmdlIODims acl_shape, size_t batch) {
    TensorShape ret;
    ret.ndim = acl_shape.dimCount;
    for (size_t i = 0; i < ret.ndim; ++i) {
        ret[i] = acl_shape.dims[i];
    }
    if (acl_shape.dims[0] == -1) {
        batch = aclmdlGetOutputSizeByIndex(model_desc, output_idx);
        size_t chw = output_dtype_size;
        for (size_t i = 1; i < ret.ndim; ++i) {
            chw *= ret[i];
        }
        mgb_assert(
                batch % chw == 0,
                "When the input batch is static and the output batch is dynamic, it is "
                "necessary to reconfigure the output batch. The output size obtained "
                "from the aclmdlGetOutputSizeByIndex interface should be evenly "
                "divided by "
                "shapes other than the batch. expect 0, but got %zu\n",
                batch % chw);
        batch /= chw;
        ret[0] = batch;
    }
    return ret;
}

/**
 * \brief deduce the input shape from aclFormat and aipp config.
 *
 * \param acl_shape shape from om file
 * \param batch batchsize from mgb
 * \param enable_dynamic_batch True if set dynamic batch size
 * \param om_format layout format from om file
 * \param aipp_input_fmt input_format in static aipp config of om file
 */
TensorShape acl_shape_to_mgb_shape_for_input(
        aclmdlIODims acl_shape, size_t batch, bool enable_dynamic_batch,
        aclFormat om_format, AtlasRuntimeOpr::AippInputFormat aipp_input_fmt) {
    MGB_MARK_USED_VAR(aipp_input_fmt);
    TensorShape ret;
    ret.ndim = acl_shape.dimCount;
    for (size_t i = 0; i < ret.ndim; ++i) {
        ret[i] = acl_shape.dims[i];
    }
    if (enable_dynamic_batch) {
        mgb_assert(
                ret[0] == static_cast<size_t>(-1),
                "batch size expected to be -1 when enable dynamic "
                "batchsize, got: %zu\n",
                ret[0]);
        ret[0] = batch;
    } else {
        mgb_assert(
                ret[0] == batch,
                "batchsize mismatch if no dynamic batchsize enabled, "
                "expected: %zu got: %zu\n",
                ret[0], batch);
    }

    mgb_assert(om_format != ACL_FORMAT_UNDEFINED, "om input format should be defined");

    return ret;
}

DType acl_dtype_to_mgb_dtype(aclDataType data_type) {
    switch (data_type) {
        case ACL_UINT8:
            return dtype::Uint8();
        case ACL_FLOAT16:
#if !MEGDNN_DISABLE_FLOAT16
            return dtype::Float16();
#else
            mgb_throw(MegBrainError, "Float16 support is disabled at compile time.");
#endif
        case ACL_FLOAT:
            return dtype::Float32();
        case ACL_INT8:
            return dtype::Int8();
        case ACL_INT16:
            return dtype::Int16();
        case ACL_INT32:
            return dtype::Int32();
        default:
            mgb_throw(
                    MegBrainError, "aclDataType %x is not supported by MegBrain.",
                    static_cast<int>(data_type));
    }
}

/**
 * \brief generate batch size which match the batch_choice
 */
SmallVector<size_t> gen_batch_vec(
        size_t origin_batch, const SmallVector<size_t>& batch_choices) {
    SmallVector<size_t> ret;
    size_t idx = 0;
    size_t nr_batch_choices = batch_choices.size();
    size_t batch = origin_batch;
    while (idx < nr_batch_choices) {
        size_t val = batch_choices[idx];
        while (batch >= batch_choices[idx]) {
            ret.push_back(val);
            batch -= val;
        }
        idx++;
    }
    mgb_assert(
            batch == 0, "Invalid batch size %zu, can not be generate by batch choices",
            origin_batch);

    return ret;
}

class PtrGetter {
public:
    PtrGetter(const VarNodeArray& vars) {
        for (auto&& var : vars) {
            m_ptrs.push_back(var->dev_tensor().raw_ptr());
            m_batch_in_bytes.push_back(
                    var->layout().stride[0] * var->layout().dtype.size());
        }
    }

    std::pair<void*, size_t> get(size_t batch, size_t idx) {
        std::pair<void*, size_t> ret;
        ret.first = m_ptrs[idx];
        ret.second = batch * m_batch_in_bytes[idx];
        m_ptrs[idx] = reinterpret_cast<void*>(
                reinterpret_cast<uintptr_t>(ret.first) + ret.second);
        return ret;
    }

private:
    SmallVector<void*> m_ptrs;
    SmallVector<size_t> m_batch_in_bytes;
};

};  // namespace

/* ====================== AtlasRuntimeOpr ==================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(AtlasRuntimeOpr);
AtlasRuntimeOpr::AtlasRuntimeOpr(
        SharedBuffer buf, const std::pair<uint32_t, aclmdlDesc*>& model,
        const VarNodeArray& inputs, const OperatorNodeConfig& config)
        : Super(inputs[0]->owner_graph(), config, "atlas_runtime", inputs),
          m_buffer{std::move(buf)},
          m_model_id{model.first},
          m_model_desc{model.second} {
    mgb_assert(
            inputs[0]->comp_node().device_type() == CompNode::DeviceType::ATLAS,
            "AtlasRuntimeOpr can only be used on atlas comp node; "
            "got %s",
            inputs[0]->comp_node().to_string().c_str());
    mgb_assert(
            m_buffer.data() != nullptr ||
            (m_model_id != INVALID_MODEL_ID && m_model_desc != nullptr));

    for (auto i : inputs) {
        add_input({i});
    }
    if (m_model_id == INVALID_MODEL_ID && m_model_desc == nullptr) {
        aclmdlConfigHandle* config_handle = aclmdlCreateConfigHandle();

        size_t mdl_load_type = ACL_MDL_LOAD_FROM_MEM;
        const void* mdl_mem_addr_ptr = m_buffer.data();
        size_t mdl_mem_size = m_buffer.size();
        MGB_ATLAS_CHECK(aclmdlSetConfigOpt(
                config_handle, ACL_MDL_LOAD_TYPE_SIZET, &mdl_load_type,
                sizeof(size_t)));
        MGB_ATLAS_CHECK(aclmdlSetConfigOpt(
                config_handle, ACL_MDL_MEM_ADDR_PTR, &mdl_mem_addr_ptr,
                sizeof(const void*)));
        MGB_ATLAS_CHECK(aclmdlSetConfigOpt(
                config_handle, ACL_MDL_MEM_SIZET, &mdl_mem_size, sizeof(size_t)));

        size_t mem_optimize_mode = true;
        MGB_ATLAS_CHECK(aclmdlSetConfigOpt(
                config_handle, ACL_MDL_WORKSPACE_MEM_OPTIMIZE, &mem_optimize_mode,
                sizeof(size_t)));

        MGB_ATLAS_CHECK(aclmdlLoadWithConfig(config_handle, &m_model_id));
        MGB_ATLAS_CHECK(aclmdlDestroyConfigHandle(config_handle));

        m_model_desc = aclmdlCreateDesc();
        MGB_ATLAS_CHECK(aclmdlGetDesc(m_model_desc, m_model_id));
        m_is_model_holder = true;
    }

    //! aipp input format
    m_aipp_input_format = SmallVector<AippInputFormat>(inputs.size());
    aclAippInfo aipp_info;
    for (size_t i = 0; i < inputs.size(); ++i) {
        aclError acl_err = aclmdlGetFirstAippInfo(m_model_id, i, &aipp_info);
        if (ACL_ERROR_NONE == acl_err) {
            switch (aipp_info.inputFormat) {
                case ACL_YUV420SP_U8:
                    m_aipp_input_format[i] = AippInputFormat::YUV420SP_U8;
                    break;
                case ACL_RGB888_U8:
                    m_aipp_input_format[i] = AippInputFormat::RGB888_U8;
                    break;
                default:
                    mgb_throw(
                            MegBrainError,
                            "Unsupported aclAippInputFormat for input %zu. ", i);
            }
        } else if (
                ACL_ERROR_NOT_STATIC_AIPP == acl_err ||
                ACL_ERROR_GE_AIPP_NOT_EXIST == acl_err) {
            m_aipp_input_format[i] = AippInputFormat::NO_AIPP;
        } else {
            MGB_ATLAS_CHECK(acl_err);
        }
    }

    size_t dynamic_index;
    auto errcode = aclmdlGetInputIndexByName(
            m_model_desc, ACL_DYNAMIC_TENSOR_NAME, &dynamic_index);
    if (errcode == ACL_ERROR_NONE) {
        aclmdlHW hw_info;
        MGB_ATLAS_CHECK(aclmdlGetDynamicHW(m_model_desc, dynamic_index, &hw_info));
        mgb_assert(hw_info.hwCount == 0, "Currently not support dynamic HW");
    }

    //! dynamic batch size
    aclmdlBatch acl_batch;
    MGB_ATLAS_CHECK(aclmdlGetDynamicBatch(m_model_desc, &acl_batch));
    if (acl_batch.batchCount) {
        size_t dynamic_data_size;
        dynamic_data_size = aclmdlGetInputSizeByIndex(m_model_desc, dynamic_index);
        m_dyn_batch_tensor = DeviceTensorND(
                inputs[0]->comp_node(), {{dynamic_data_size}, dtype::Uint8()});

        for (size_t i = 0; i < acl_batch.batchCount; ++i) {
            m_dyn_batch_choices.push_back(static_cast<size_t>(acl_batch.batch[i]));
        }
        std::sort(
                m_dyn_batch_choices.begin(), m_dyn_batch_choices.end(),
                std::greater<>());
    }

    //! add output
    size_t nr_outputs = aclmdlGetNumOutputs(m_model_desc);
    using F = VarNode::Flag;
    if (nr_outputs == 1) {
        add_output(None);
    } else {
        for (size_t i = 0; i < nr_outputs; ++i) {
            add_output(ssprintf("o%zu", i));
        }
    }
    if (!m_dyn_batch_choices.empty()) {
        /**
         * \warning If enable dynamic batchsize, the memory of output
         * should be the largest be the size with the largest batch_size, so we
         * set the flag to SYS_MEM_ALLOC.
         */
        for (size_t i = 0; i < nr_outputs; ++i) {
            output(i)->add_flag(F::NO_SYS_MEM_ALLOC).add_flag(F::NO_MEM_RECLAIM);
        }
    }
    add_equivalence_component<mgb::ScalarHash<const void*>>(m_buffer.data());
};

AtlasRuntimeOpr::~AtlasRuntimeOpr() {
    if (m_is_model_holder) {
        MGB_ATLAS_CHECK(aclmdlUnload(m_model_id));
        MGB_ATLAS_CHECK(aclmdlDestroyDesc(m_model_desc));
    }
}

void AtlasRuntimeOpr::scn_do_execute() {
    auto&& acl_env = CompNodeEnv::from_comp_node(input(0)->comp_node()).atlas_env();
    acl_env.activate();

    if (!m_dyn_batch_choices.empty()) {
        for (size_t i = 0; i < output().size(); i++) {
            auto output_size = aclmdlGetOutputSizeByIndex(m_model_desc, i);
            auto ovar = output(i);
            output_size = std::max<size_t>(
                    output_size, ovar->dtype().size(ovar->shape().total_nr_elems()));
            ovar->shape_alloc(ovar->shape(), output_size);
        }
    }

    PtrGetter input_getter(input());
    PtrGetter output_getter(output());

    bool enable_dynamic_batch = !m_dyn_batch_choices.empty();
    size_t nr_inputs = aclmdlGetNumInputs(m_model_desc);
    size_t nr_outputs = aclmdlGetNumOutputs(m_model_desc);
    size_t input_batch = input(0)->layout()[0];

    if (enable_dynamic_batch) {
        mgb_assert(
                nr_inputs == input().size() + 1,
                "nr inputs got from om model should be one more than got "
                "from megbrain");
    }
    SmallVector<size_t> batches_each_run;
    if (enable_dynamic_batch) {
        batches_each_run = gen_batch_vec(input_batch, m_dyn_batch_choices);
    } else {
        batches_each_run.push_back(input_batch);
    }

    for (auto&& batch : batches_each_run) {
        //! prepare input
        auto model_inputs = aclmdlCreateDataset();
        mgb_assert(model_inputs != nullptr, "failed to create atlas input dataset.");
        for (size_t i = 0; i < input().size(); i++) {
            auto value_pair = input_getter.get(batch, i);
            auto input_size = aclmdlGetInputSizeByIndex(m_model_desc, i);
            //! FIXME iff enable dynamic batchsize and dynamic aipp, the input
            //! size should be the size of aclmdlGetInputSizeByIndex.
            if (enable_dynamic_batch) {
                mgb_assert(
                        input_size ==
                                value_pair.second / batch * m_dyn_batch_choices[0],
                        "input %zu size mismatch, expected: %zu got: %zu", i,
                        input_size, value_pair.second / batch * m_dyn_batch_choices[0]);
            }
            aclDataBuffer* input_db =
                    aclCreateDataBuffer(value_pair.first, value_pair.second);
            mgb_assert(
                    input_db != nullptr,
                    "failed to create atlas input data buffer for input "
                    "%zu:%s.",
                    i, input(i)->cname());
            aclmdlAddDatasetBuffer(model_inputs, input_db);
        }
        //! append unit tensor for dynamic batch
        if (enable_dynamic_batch) {
            aclDataBuffer* input_db = aclCreateDataBuffer(
                    reinterpret_cast<void*>(m_dyn_batch_tensor.raw_ptr()),
                    m_dyn_batch_tensor.layout().span().dist_byte());
            mgb_assert(
                    input_db != nullptr,
                    "failed to create atlas input data buffer for dynamic "
                    "batch tensor.");
            MGB_ATLAS_CHECK(aclmdlAddDatasetBuffer(model_inputs, input_db));

            MGB_ATLAS_CHECK(aclmdlSetDynamicBatchSize(
                    m_model_id, model_inputs, input().size(),
                    static_cast<uint64_t>(batch)));
        }

        //! prepare output
        auto model_outputs = aclmdlCreateDataset();
        mgb_assert(model_outputs != nullptr, "failed to create atlas output dataset.");
        std::vector<aclTensorDesc*> tensor_desc_holder(nr_outputs, nullptr);
        for (size_t i = 0; i < nr_outputs; i++) {
            auto value_pair = output_getter.get(batch, i);
            size_t output_size = value_pair.second;
            if (enable_dynamic_batch || m_dyn_batch_output[i]) {
                output_size = aclmdlGetOutputSizeByIndex(m_model_desc, i);
            }
            aclDataBuffer* output_db =
                    aclCreateDataBuffer(value_pair.first, output_size);
            mgb_assert(
                    output_db != nullptr,
                    "failed to create atlas output data buffer for output "
                    "%zu:%s.",
                    i, output(i)->cname());
            aclmdlAddDatasetBuffer(model_outputs, output_db);

            if (!enable_dynamic_batch && m_dyn_batch_output[i]) {
                auto tensor_ndim = output(i)->shape().ndim;
                std::vector<int64_t> tensor_shape(tensor_ndim, 0);
                for (size_t j = 0; j < tensor_ndim; j++) {
                    tensor_shape[j] = output(i)->shape()[j];
                }
                aclTensorDesc* tensorDesc = aclCreateTensorDesc(
                        aclmdlGetOutputDataType(m_model_desc, i), tensor_ndim,
                        tensor_shape.data(), aclmdlGetOutputFormat(m_model_desc, i));
                aclmdlSetDatasetTensorDesc(model_outputs, tensorDesc, i);
                tensor_desc_holder[i] = tensorDesc;
            }
        }

#if MGB_USE_ATLAS_ASYNC_API
        MGB_ATLAS_CHECK(aclmdlExecuteAsync(
                m_model_id, model_inputs, model_outputs, acl_env.stream));
        MGB_ATLAS_CHECK(aclrtSynchronizeStream(acl_env.stream));
#else
        MGB_ATLAS_CHECK(aclmdlExecute(m_model_id, model_inputs, model_outputs));
#endif
        for (size_t i = 0; i < nr_inputs; ++i) {
            aclDataBuffer* db_ptr = aclmdlGetDatasetBuffer(model_inputs, i);
            MGB_ATLAS_CHECK(aclDestroyDataBuffer(db_ptr));
        }
        for (size_t i = 0; i < nr_outputs; ++i) {
            if (m_dyn_batch_output[i]) {
                const DeviceTensorND old_dev_tensor = output(i)->dev_tensor();

                auto new_output_desc = aclmdlGetDatasetTensorDesc(model_outputs, i);

                TensorShape new_shape;
                new_shape.ndim = aclGetTensorDescNumDims(new_output_desc);
                mgb_assert(
                        new_shape.ndim == old_dev_tensor.layout().ndim,
                        "for static input batch and dynamic output batch, the output "
                        "ndim should be consistent with the one before calling "
                        "aclmdlExecute(), so expect %zu, but got %zu",
                        old_dev_tensor.layout().ndim, new_shape.ndim);
                for (size_t j = 0; j < new_shape.ndim; j++) {
                    new_shape.shape[j] = aclGetTensorDescDim(new_output_desc, j);
                }

                TensorLayout new_layout{
                        new_shape, old_dev_tensor.dtype(), old_dev_tensor.format()};
                DeviceTensorND new_dev_tensor{
                        old_dev_tensor.comp_node(), new_layout, old_dev_tensor.dtype(),
                        old_dev_tensor.format()};
                new_dev_tensor.reset(old_dev_tensor.storage(), new_layout);
                output(i)->force_assign_dev_tensor_from_tensor(new_dev_tensor);
            }
            aclDataBuffer* db_ptr = aclmdlGetDatasetBuffer(model_outputs, i);
            MGB_ATLAS_CHECK(aclDestroyDataBuffer(db_ptr));
        }
        for (size_t i = 0; i < tensor_desc_holder.size(); ++i) {
            if (tensor_desc_holder[i] != nullptr) {
                aclDestroyTensorDesc(tensor_desc_holder[i]);
            }
        }
        MGB_ATLAS_CHECK(aclmdlDestroyDataset(model_inputs));
        MGB_ATLAS_CHECK(aclmdlDestroyDataset(model_outputs));
    }
}

void AtlasRuntimeOpr::get_output_var_shape(
        const TensorShapeArray& inp_shape, TensorShapeArray& out_shape) const {
    size_t nr_inputs = aclmdlGetNumInputs(m_model_desc);
    size_t batch_size = inp_shape[0][0];
    //! enable dynamic batchsize
    if (!m_dyn_batch_choices.empty()) {
        mgb_assert(!gen_batch_vec(batch_size, m_dyn_batch_choices).empty());
        mgb_assert(
                nr_inputs == inp_shape.size() + 1,
                "nr inputs got from om model should be one more than got "
                "from megbrain");
    }
    for (size_t i = 0; i < inp_shape.size(); ++i) {
        batch_size = inp_shape[i][0];
        aclmdlIODims input_dims;
        MGB_ATLAS_CHECK(aclmdlGetInputDimsV2(m_model_desc, i, &input_dims));
        auto om_format = aclmdlGetInputFormat(m_model_desc, i);
        TensorShape shape_from_om = acl_shape_to_mgb_shape_for_input(
                input_dims, batch_size, !m_dyn_batch_choices.empty(), om_format,
                m_aipp_input_format[i]);
        mgb_assert(
                inp_shape[i].ndim == input_dims.dimCount,
                "ndim mismatch of input %zu: expected %zu, "
                "got %zu",
                i, input_dims.dimCount, inp_shape[i].ndim);

        for (size_t j = 0; j < inp_shape[i].ndim; j++) {
            if (input_dims.dims[j] == -1) {
                continue;
            } else {
                mgb_assert(
                        inp_shape[i][j] == shape_from_om[j],
                        "The shape passed in by megbrain is inconsistent with the "
                        "shape of the wrapped om model, shape mismatch of input[%zu] "
                        ": om model expected %s, but got %s from megbrain",
                        i, shape_from_om.to_string().c_str(),
                        inp_shape[i].to_string().c_str());
            }
        }
    }

    m_dyn_batch_output.resize(out_shape.size());
    for (size_t i = 0; i < out_shape.size(); ++i) {
        aclmdlIODims output_dims;
        MGB_ATLAS_CHECK(aclmdlGetOutputDims(m_model_desc, i, &output_dims));
        out_shape[i] = acl_shape_to_mgb_shape_for_output(
                m_model_desc, i, output(i)->dtype().size(), output_dims, batch_size);
        m_dyn_batch_output[i] = output_dims.dims[0] == -1;
    }
}

void AtlasRuntimeOpr::add_input_layout_constraint() {
    //! default contiguous
    for (auto i : input()) {
        i->add_layout_constraint_contiguous();
    }
}

void AtlasRuntimeOpr::init_output_dtype() {
    DType dt_acl, dt_input;
    for (size_t i = 0; i < input().size(); ++i) {
        dt_acl = acl_dtype_to_mgb_dtype(aclmdlGetInputDataType(m_model_desc, i));
        dt_input = input(i)->dtype();
        mgb_assert(
                dt_acl.valid() && dt_input.valid() &&
                        dt_acl.enumv() == dt_input.enumv(),
                "dtype mismatch of input %zu: expected %s, "
                "got %s",
                i, dt_acl.name(), dt_input.name());
    }

    for (size_t i = 0; i < output().size(); ++i) {
        dt_acl = acl_dtype_to_mgb_dtype(aclmdlGetOutputDataType(m_model_desc, i));
        mgb_assert(
                dt_acl.valid(),
                "output dtype checking failed: invalid dtype returned.");
        if (dt_acl.enumv() == DTypeEnum::QuantizedS8) {
            mgb_assert(
                    output(i)->dtype().valid(),
                    "user should specify scale of output tensor of "
                    "AtlasRuntimeOpr.");
        }
        if (!output(i)->dtype().valid())
            output(i)->dtype(dt_acl);
    }
}

SymbolVarArray AtlasRuntimeOpr::make(
        SharedBuffer buf, const SymbolVarArray& src, const OperatorNodeConfig& config) {
    VarNodeArray var_node_array = cg::to_var_node_array(src);
    auto atlas_runtime_opr = std::make_unique<AtlasRuntimeOpr>(
            std::move(buf), std::pair<uint32_t, aclmdlDesc*>{INVALID_MODEL_ID, nullptr},
            var_node_array, config);
    auto ret =
            cg::to_symbol_var_array(src[0].node()
                                            ->owner_graph()
                                            ->insert_opr(std::move(atlas_runtime_opr))
                                            ->output());
    return ret;
}

SymbolVarArray AtlasRuntimeOpr::make(
        const void* buf, size_t size, const SymbolVarArray& src,
        const OperatorNodeConfig& config) {
    mgb_throw_if(
            !CompNode::get_device_count(CompNode::DeviceType::ATLAS), SystemError,
            "can not create AtlasRuntimeOpr when atlas is not "
            "available");
    std::shared_ptr<uint8_t> shptr{new uint8_t[size], [](uint8_t* p) { delete[] p; }};
    memcpy(shptr.get(), buf, size);
    SharedBuffer buffer{std::move(shptr), size};
    return make(std::move(buffer), src, config);
}

SymbolVarArray AtlasRuntimeOpr::make(
        const SharedBuffer buf, const std::pair<uint32_t, aclmdlDesc*>& model,
        const SymbolVarArray& src, const OperatorNodeConfig& config) {
    VarNodeArray var_node_array = cg::to_var_node_array(src);
    auto atlas_runtime_opr =
            std::make_unique<AtlasRuntimeOpr>(buf, model, var_node_array, config);
    auto ret =
            cg::to_symbol_var_array(src[0].node()
                                            ->owner_graph()
                                            ->insert_opr(std::move(atlas_runtime_opr))
                                            ->output());
    return ret;
}

constexpr uint32_t AtlasRuntimeOpr::INVALID_MODEL_ID;

#endif  // MGB_atlas

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
