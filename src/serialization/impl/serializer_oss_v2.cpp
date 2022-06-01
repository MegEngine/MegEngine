#if MGB_ENABLE_FBS_SERIALIZATION

#include "megbrain/comp_node_env.h"
#include "megbrain/opr/io.h"
#include "megbrain/serialization/helper.h"
#include "megbrain/serialization/internal/flatbuffers_helper.h"
#include "megbrain/serialization/internal/schema_v2_generated.h"
#include "megbrain/serialization/metadata.h"
#include "megbrain/serialization/opr_load_dump.h"
#include "megbrain/serialization/oss_opr_load_dump.h"
#include "megbrain/utils/hash_ct.h"
#include "megdnn/tensor_format.h"
#include "serializer_oss_common.h"

#include "megbrain/gopt/framework.h"

namespace mgb {
namespace serialization {

/*!
 * \brief replace the the opr who has the replace_opr methord in OprLoadDumpImplV2
 */
class PassConvertToCompatible : public gopt::Pass {
    ThinHashMap<
            Typeinfo*, thin_function<cg::OperatorNodeBase*(
                               cg::OperatorNodeBase*, const VarNodeArray&)>>
            m_opr_replace_func;
    gopt::VarReplaceCheckFlag m_var_replace_check_flag =
            gopt::VarReplaceCheckFlag::CHECK_ALL;

public:
    const char* name() const override { return "PassConvertToCompatible"; };

    PassConvertToCompatible& set_var_replace_check_flag(
            gopt::VarReplaceCheckFlag flag) {
        m_var_replace_check_flag = flag;
        return *this;
    }

    void apply(gopt::OptState& state) const override {
        state.set_var_replace_check_flag(m_var_replace_check_flag);
        auto rewriter = state.graph().make_rewriter();

        auto on_opr = [this, &rewriter](cg::OperatorNodeBase* opr) {
            auto it = m_opr_replace_func.find(opr->dyn_typeinfo());
            if (it != m_opr_replace_func.end()) {
                VarNodeArray new_inp;
                new_inp.clear();
                new_inp.reserve(opr->input().size());
                for (auto i : opr->input()) {
                    new_inp.push_back(rewriter.get_var(i));
                }
                auto new_opr = (it->second)(opr, new_inp);

                auto &&origin_out = opr->output(), &&cur_out = new_opr->output();
                for (size_t i = 0; i < std::min(origin_out.size(), cur_out.size());
                     i++) {
                    rewriter.replace_var(origin_out[i], cur_out[i], nullptr);
                }
            } else {
                rewriter.auto_replace_outputs(opr);
            }
        };
        state.graph().iter(on_opr);
        rewriter.apply_inplace();
    }

    static std::unique_ptr<PassConvertToCompatible> make(
            const SymbolVarArray& output_vars) {
        auto ret = std::make_unique<PassConvertToCompatible>();
        // iterate oprs to init
        auto on_opr = [&](cg::OperatorNodeBase* opr) {
            if (!GraphDumper::should_remove_in_dump(opr)) {
                auto registry = OprRegistryV2::versioned_find_by_typeinfo(
                        opr->dyn_typeinfo(), CURRENT_VERSION);
                mgb_throw_if(
                        !registry,
                        cg::OperatorNodeExcExtraInfo::ExcMaker{opr}.make<MegBrainError>,
                        "serialization as FlatBuffers is not supported for "
                        "operator %s, typeinfo %p",
                        opr->dyn_typeinfo()->name, opr->dyn_typeinfo());
                if (registry->converter) {
                    ret->m_opr_replace_func[opr->dyn_typeinfo()] = registry->converter;
                }
            }
        };
        cg::DepOprIter dep_opr_iter{on_opr};
        for (auto i : output_vars) {
            dep_opr_iter.add(i.node()->owner_opr());
        }
        return ret;
    };
};

namespace {
fbs::v2::TensorFormat get_flatbuffer_tensor_format_type(
        const TensorLayout::Format& format) {
    using Type = megdnn::TensorFormat::Type;
    switch (format.type()) {
        case Type::DEFAULT:
            return fbs::v2::TensorFormat::TensorFormat_DefaultTensorFormat;
        case Type::IMAGE2D_PACK4:
            return fbs::v2::TensorFormat::TensorFormat_Image2DPackedTensorFormat;
        case Type::LOWBITS_ALIGNED_TO_BYTE:
            return fbs::v2::TensorFormat::TensorFormat_LowbitsAlignedTensorFormat;
        default:
            mgb_throw(
                    SerializationError, "invalid tensor format type in serialization.");
    }
}
}  // namespace

flatbuffers::Offset<fbs::DType> GraphDumperOSSV2::build_dtype(DType dtype) {
    return fbs::intl::build_dtype(m_builder, dtype);
}

flatbuffers::Offset<void> GraphDumperOSSV2::build_tensor_format(
        const TensorLayout::Format& format) {
    using Type = megdnn::TensorFormat::Type;
    switch (format.type()) {
        case Type::DEFAULT:
            return fbs::v2::CreateDefaultTensorFormat(m_builder).Union();
        case Type::IMAGE2D_PACK4:
            return fbs::v2::CreateImage2DPackedTensorFormat(
                           m_builder, format.as_impl<megdnn::Image2DPack4TensorFormat>()
                                              .align_axis())
                    .Union();
        case Type::LOWBITS_ALIGNED_TO_BYTE: {
            auto size_bite = format.as_impl<megdnn::LowbitsAlignedToBytesTensorFormat>()
                                     .size_nbits();
            auto align_size_in_bits =
                    format.as_impl<megdnn::LowbitsAlignedToBytesTensorFormat>()
                            .align_size_in_bits();
            return fbs::v2::CreateLowbitsAlignedTensorFormat(
                           m_builder, size_bite, align_size_in_bits)
                    .Union();
        }
        default:
            mgb_throw(
                    SerializationError, "invalid tensor format type in serialization.");
    }
}

flatbuffers::Offset<fbs::v2::MiddleTensor> GraphDumperOSSV2::build_middle_tensor(
        const SymbolVar var) {
    mgb_assert(var.node());
    auto fbname = m_builder.CreateSharedString(var.node()->name());
    flatbuffers::Offset<fbs::v2::MiddleTensor> serialized_middle_tensor;
    if (var.node()->dev_tensor_valid()) {
        auto layout = var.node()->layout();
        auto fshape =
                m_builder.CreateVectorScalarCast<uint32_t>(layout.shape, layout.ndim);

        auto fcomp_node = fbs::v2::CreateCompNode(
                m_builder, m_builder.CreateSharedString(
                                   var.node()->comp_node().to_string_logical()));

        auto fdtype = build_dtype(layout.dtype);
        auto fformat_type = get_flatbuffer_tensor_format_type(layout.format);
        auto fformat = build_tensor_format(layout.format);
        serialized_middle_tensor = fbs::v2::CreateMiddleTensor(
                m_builder, fbname, fshape, fcomp_node, fdtype, fformat_type, fformat);
    }
    serialized_middle_tensor = fbs::v2::CreateMiddleTensor(m_builder, fbname);
    return serialized_middle_tensor;
}

flatbuffers::Offset<fbs::v2::OutputVar> GraphDumperOSSV2::build_output_var(
        const SymbolVar var) {
    auto out_node = var.node();
    if (m_var2midtensor_id.find(var.node()) == m_var2midtensor_id.end()) {
        mgb_assert(m_var_remove_in_dump.find(var.node()) != m_var_remove_in_dump.end());
        out_node = m_var_remove_in_dump[var.node()];
    }
    return fbs::v2::CreateOutputVar(
            m_builder, m_var2midtensor_id.at(out_node), var.node()->id());
}

void GraphDumperOSSV2::init_oprs_to_dump(const SymbolVarArray& endpoints) {
    m_oprs_to_dump.clear();

    // iterate oprs to init
    auto on_opr = [&](cg::OperatorNodeBase* opr) {
        if (should_remove_in_dump(opr)) {
            mgb_assert(opr->input().size() == 1);
            // Copy input ID to output
            for (auto i : opr->output()) {
                if (m_var_remove_in_dump.find(opr->input(0)) !=
                    m_var_remove_in_dump.end()) {
                    m_var_remove_in_dump[i] = m_var_remove_in_dump[opr->input(0)];
                } else {
                    m_var_remove_in_dump[i] = opr->input(0);
                }
            }
        } else {
            auto registry = OprRegistryV2::versioned_find_by_typeinfo(
                    opr->dyn_typeinfo(), m_version);
            if (!registry || !registry->dumper) {
                mgb_throw(
                        cg::OperatorNodeExcExtraInfo::ExcMaker{opr}.make<MegBrainError>,
                        "serialization as FlatBuffers is not supported for "
                        "operator %s",
                        opr->dyn_typeinfo()->name);
            }
            mgb_assert(
                    registry->version <= m_version,
                    "The Operator version should less than model version");
            m_oprs_to_dump.emplace_back(opr, registry);
        }
    };
    cg::DepOprIter dep_opr_iter{on_opr};
    for (auto i : endpoints) {
        dep_opr_iter.add(i.node()->owner_opr());
    }
}

flatbuffers::Offset<fbs::v2::Metadata> GraphDumperOSSV2::build_metadata(
        const Metadata& metadata) {
    auto user_info = m_builder.CreateSharedString(metadata.user_info);
    fbs::v2::MetadataBuilder builder(m_builder);
    builder.add_is_valid(metadata.is_valid);
    builder.add_graph_modified(metadata.graph_modified);
    builder.add_optimize_options(metadata.optimize_options);
    builder.add_user_info(user_info);
    return builder.Finish();
}

flatbuffers::Offset<fbs::v2::Operator> GraphDumperOSSV2::build_single_opr(
        cg::OperatorNodeBase* opr, const OprRegistryV2* registry) {
    m_cur_opr = opr;
    ++m_cur_rst.nr_opr;

    using namespace flatbuffers;
    Offset<Vector<uint32_t>> inputs;
    if (m_cur_opr->input().size()) {
        std::vector<uint32_t> v;
        v.reserve(m_cur_opr->input().size());
        for (auto inp : m_cur_opr->input()) {
            if (m_var2midtensor_id.find(inp) != m_var2midtensor_id.end()) {
                v.emplace_back(m_var2midtensor_id.at(inp));
            } else {
                mgb_assert(
                        m_var_remove_in_dump.find(inp) != m_var_remove_in_dump.end(),
                        "when dump the model, the dependence of var is wrong.");
                v.emplace_back(m_var2midtensor_id.at(m_var_remove_in_dump[inp]));
            }
        }
        inputs = m_builder.CreateVector(v);
    }

    m_cur_opr_tensor.clear();
    m_blobs.clear();
    m_cur_opr_param.clear();
    m_cur_opr_param_type.clear();
    registry->dumper(*this, *m_cur_opr);

    Offset<Vector<Offset<fbs::v2::CompNode>>> comp_node;
    auto& config = m_cur_opr->config();
    if (config.has_comp_node_set()) {
        std::vector<flatbuffers::Offset<fbs::v2::CompNode>> cns;
        for (const auto& cn : config.comp_node()) {
            cns.emplace_back(fbs::v2::CreateCompNode(
                    m_builder, m_builder.CreateSharedString(cn.to_string_logical())));
        }
        comp_node = m_builder.CreateVector(cns);
    }
    Offset<String> operator_name;
    if (m_config.keep_op_name) {
        operator_name = m_builder.CreateSharedString(m_cur_opr->name());
    }

    auto output_dtype = build_dtype(config.output_dtype());

    Offset<Vector<uint32_t>> outputs;
    if (m_cur_opr->output().size()) {
        std::vector<uint32_t> v;
        v.reserve(m_cur_opr->output().size());
        for (auto out : m_cur_opr->output()) {
            if (!out->contain_flag(VarNode::Flag::VOLATILE_CONTENT)) {
                auto fbs_out = build_middle_tensor(out);
                m_model_middle_tensors.push_back(fbs_out);
                m_var2midtensor_id[out] = m_model_middle_tensors.size() - 1;
                v.emplace_back(m_var2midtensor_id.at(out));
            }
        }
        outputs = m_builder.CreateVector(v);
    }

    Offset<Vector<Offset<fbs::v2::Tensor>>> tensors;
    if (m_cur_opr_tensor.size())
        tensors = m_builder.CreateVector(m_cur_opr_tensor);

    //! the blobs data is used by custom data
    //! m_blobs will be filled by the Operator dumper function
    Offset<Vector<Offset<fbs::v2::Blob>>> blobs;
    if (m_blobs.size())
        blobs = m_builder.CreateVector(m_blobs);

    Offset<Vector<uint8_t>> additional_params_type;
    Offset<Vector<Offset<void>>> additional_params;
    auto param_cnt = m_cur_opr_param_type.size();
    if (param_cnt > 1) {
        additional_params_type = m_builder.CreateVectorScalarCast<uint8_t>(
                m_cur_opr_param_type.data() + 1, param_cnt - 1);
        additional_params =
                m_builder.CreateVector(m_cur_opr_param.data() + 1, param_cnt - 1);
    }
    auto opr_type = m_builder.CreateSharedString(registry->name);

    fbs::v2::OperatorBuilder builder(m_builder);
    builder.add_type(opr_type);
    builder.add_type_id(registry->type_id);
    builder.add_inputs(inputs);
    builder.add_outputs(outputs);
    if (m_config.keep_opr_priority) {
        builder.add_priority(opr->node_prop().attribute().priority);
    }
    builder.add_comp_node(comp_node);
    builder.add_opr_version(registry->get_version());
    builder.add_name(operator_name);
    builder.add_output_dtype(output_dtype);
    if (param_cnt > 0) {
        builder.add_param_type(m_cur_opr_param_type[0]);
        builder.add_param(m_cur_opr_param[0]);
    }
    if (param_cnt > 1) {
        builder.add_additional_params_type(additional_params_type);
        builder.add_additional_params(additional_params);
    }
    builder.add_tensors(tensors);
    builder.add_custom_data(blobs);
    m_cur_opr = nullptr;
    return builder.Finish();
}

SymbolVarArray GraphDumperOSSV2::converter_all_opr_to_compatiable(
        const SymbolVarArray& output_vars) {
    gopt::GraphOptimizer optimizer;
    VarNodeArray rets_var;
    for (auto& symbolvar : output_vars) {
        rets_var.push_back(symbolvar.node());
    }
    optimizer.add_pass(PassConvertToCompatible::make(output_vars));
    optimizer.apply_inplace(rets_var);

    SymbolVarArray dst_vars;
    for (auto& var : rets_var) {
        dst_vars.push_back({var});
    }
    return dst_vars;
}

GraphDumper::DumpResult GraphDumperOSSV2::dump(
        const SymbolVarArray& output_vars, const DumpConfig& config,
        const Metadata& metadata) {
    mgb_throw_if(output_vars.empty(), SerializationError, "Can't dump empty graph");

    auto new_output_vars = output_vars;
    if (!config.no_change_graph) {
        new_output_vars = converter_all_opr_to_compatiable(output_vars);
    }

    auto begin_pos = m_file->tell();
    m_config = config;
    m_builder.Reset();

    m_output_vars.clear();
    m_cur_rst = {};
    m_used_input_names.clear();
    m_used_param_names.clear();
    m_var_remove_in_dump.clear();
    m_model_middle_tensors.clear();
    m_var2midtensor_id.clear();
    m_nr_shared_tensor = 0;

    // process output vars
    bool keep_output_var_name = m_config.keep_var_name >= 1;
    std::unordered_set<std::string> output_var_names;
    for (auto i : new_output_vars) {
        mgb_assert(
                !i.node()->contain_flag(VarNode::Flag::VOLATILE_CONTENT),
                "can not dump var with VOLATILE_CONTENT flag: %s",
                cg::dump_var_info({i.node()}).c_str());
        if (m_output_vars.insert(i.node()).second && keep_output_var_name) {
            auto name_ins = output_var_names.insert(i.node()->name()).second;
            mgb_assert(name_ins, "duplicated output var name: %s", i.node()->cname());
        }
    }

    // Dump metadata
    auto fbmeta = build_metadata(metadata);

    // Dump operators
    init_oprs_to_dump(new_output_vars);
    std::vector<flatbuffers::Offset<fbs::v2::Operator>> oprs;
    for (auto&& i : m_oprs_to_dump) {
        oprs.emplace_back(build_single_opr(i.first, i.second));
    }
    auto fb_oprs = m_builder.CreateVector(oprs);

    // Dump output vars
    std::vector<flatbuffers::Offset<fbs::v2::OutputVar>> output_vars_idx;
    output_vars_idx.reserve(new_output_vars.size());
    for (auto i : new_output_vars) {
        auto foutput_vars_idx = build_output_var(i);
        output_vars_idx.push_back(foutput_vars_idx);
    }
    auto fb_output_vars = m_builder.CreateVector(output_vars_idx);
    std::vector<flatbuffers::Offset<fbs::v2::OutputAlias>> output_vars_alias;
    if (m_config.alias_name_map.size() > 0) {
        for (auto&& pair : m_config.alias_name_map) {
            std::string name;
            SymbolVar var;
            std::tie(name, var) = pair;
            auto fbs_name = m_builder.CreateSharedString(name);
            output_vars_alias.push_back(
                    fbs::v2::CreateOutputAlias(m_builder, var.node()->id(), fbs_name));
        }
    }
    auto fbs_output_alias = m_builder.CreateVector(output_vars_alias);
    auto fb_mid_tensor = m_builder.CreateVector(m_model_middle_tensors);

    fbs::v2::ModelBuilder model(m_builder);
    model.add_mge_version(MGB_VERSION);
    model.add_model_version(m_version);
    model.add_oprs(fb_oprs);
    model.add_middle_tensors(fb_mid_tensor);
    model.add_output_vars_idx(fb_output_vars);
    model.add_output_alias(fbs_output_alias);
    model.add_nr_shared_tensor(m_nr_shared_tensor);
    model.add_metadata(fbmeta);
    m_builder.FinishSizePrefixed(model.Finish(), fbs::v2::ModelIdentifier());

    // Write serialized fbs::Graph
    m_file->write(m_builder.GetBufferPointer(), m_builder.GetSize());

    // Finalize DumpResult
    auto&& ret = m_cur_rst;
    for (size_t i = 0; i < new_output_vars.size(); i++) {
        ret.outputs.emplace_back(
                keep_output_var_name ? new_output_vars[i].node()->cname()
                                     : ssprintf("unnamed%zu", i));
    }
    std::sort(ret.inputs.begin(), ret.inputs.end());
    mgb_assert(ret.nr_opr == m_oprs_to_dump.size());
    ret.tot_bytes = m_file->tell() - begin_pos;
    return ret;
}

void GraphDumperOSSV2::dump_tensor(
        const std::string& name, const HostTensorND& tensor, TensorWriteMethod method) {
    using namespace flatbuffers;
    using Meth = TensorWriteMethod;
    mgb_assert(
            (method == Meth::VALUE_ANONYMOUS) ^ (!name.empty()),
            "name must be non-empty for non Meth::VALUE_ANONYMOUS tensors");

    bool has_value = method != Meth::META_INPUT;
    bool should_keep_name = true;
    switch (method) {
        case Meth::VALUE_ANONYMOUS:
            should_keep_name = false;
            break;
        case Meth::VALUE_SHARED:
            should_keep_name = m_config.keep_param_name;
            ++m_nr_shared_tensor;
            if (m_config.keep_param_name) {
                mgb_assert(
                        m_used_param_names.insert(name).second,
                        "duplicated VALUE_SHARED tensor name: %s", name.c_str());
                m_cur_rst.params.emplace_back(name);
            }
            break;
        case Meth::META_INPUT:
        case Meth::VALUE_INPUT:
            mgb_assert(!name.empty(), "empty input tensor name");
            mgb_assert(
                    m_used_input_names.insert(name).second,
                    "duplicated input tensor name: %s", name.c_str());
            m_cur_rst.inputs.emplace_back(name);
            break;
    }

    auto& layout = tensor.layout();
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> data;
    if (has_value) {
        check_tensor_value_valid(name, tensor);
        auto&& dumper = m_config.tensor_value_dumper;
        if (dumper) {
            mgb_log_warn(
                    "serialization v2 format is pure flatbuffer format, not support "
                    "user tensor value dumper callback.");
        }
        data = m_builder.CreateVector(
                reinterpret_cast<uint8_t*>(tensor.raw_ptr()), layout.span().high_byte);
        m_cur_rst.tensor_value_bytes += layout.span().high_byte;
    }

    auto fbname = should_keep_name ? m_builder.CreateSharedString(name) : 0;
    auto fshape = m_builder.CreateVectorScalarCast<uint32_t>(layout.shape, layout.ndim);
    auto fcomp_node = fbs::v2::CreateCompNode(
            m_builder,
            m_builder.CreateSharedString(tensor.comp_node().to_string_logical()));
    auto fdtype = build_dtype(layout.dtype);

    auto fformat_type = get_flatbuffer_tensor_format_type(layout.format);
    auto fformat = build_tensor_format(layout.format);
    auto serialized_tensor = fbs::v2::CreateTensor(
            m_builder, fbname, fshape, fcomp_node, fdtype, fformat_type, fformat, data);
    m_cur_opr_tensor.emplace_back(serialized_tensor);
}

void GraphDumperOSSV2::dump_buf_with_len(const void* data, uint32_t size) {
    auto blob = fbs::v2::CreateBlob(
            m_builder, m_builder.CreateVector(static_cast<const uint8_t*>(data), size));
    m_blobs.emplace_back(blob);
}

// ----------------------------- Loader --------------------------------------
CompNode GraphLoaderOSSV2::OprLoadContextImpl::load_comp_node(
        const fbs::v2::CompNode* comp_node) {
    mgb_assert(comp_node);
    if (!comp_node->logical_locator())
        return {};
    auto loc = CompNode::Locator::parse(comp_node->logical_locator()->str());
    m_loader->m_cur_load_config->comp_node_mapper(loc);
    return CompNode::load(loc);
}

TensorFormat load_tensor_format(
        const fbs::v2::TensorFormat fformat_type, const void* fformat,
        const CompNode& comp_node) {
    switch (fformat_type) {
        case fbs::v2::TensorFormat_DefaultTensorFormat:
            return megdnn::DefaultTensorFormat::make();
        case fbs::v2::TensorFormat_Image2DPackedTensorFormat: {
            auto image_format =
                    static_cast<const fbs::v2::Image2DPackedTensorFormat*>(fformat);
            auto handle =
                    MegDNNHandle::get(CompNodeEnv::from_comp_node(comp_node)).handle();
            return megdnn::Image2DPack4TensorFormat::make(
                    image_format->align_axis(), handle);
        }
        case fbs::v2::TensorFormat_LowbitsAlignedTensorFormat: {
            auto lowbit_format =
                    static_cast<const fbs::v2::LowbitsAlignedTensorFormat*>(fformat);
            return megdnn::LowbitsAlignedToBytesTensorFormat::make(
                    lowbit_format->size_nbits());
        }
        default:
            mgb_throw(
                    SerializationError, "invalid tensor format type in serialization.");
    }
}

TensorLayout load_tensor_layout(
        const fbs::v2::Tensor* tensor, const CompNode& comp_node) {
    TensorLayout layout;
    if (tensor->shape()) {
        layout.ndim = tensor->shape()->size();
        std::copy(tensor->shape()->begin(), tensor->shape()->end(), layout.shape);
    }
    if (tensor->dtype()) {
        // modify data type inplace for TensorLayout
        layout.modify_dtype_inplace(fbs::intl::load_dtype(tensor->dtype()));
    }
    if (tensor->format() && tensor->format_type()) {
        layout.format =
                load_tensor_format(tensor->format_type(), tensor->format(), comp_node);
    }
    layout.init_contiguous_stride();
    return layout;
}

//! the opr loader should make sure the exist of tensors and the number of
//! tensor, here just assert it.
std::shared_ptr<HostTensorND> GraphLoaderOSSV2::OprLoadContextImpl::load_tensor() {
    mgb_assert(
            m_current_opr->tensors() &&
            m_cur_opr_tensor_cnt < m_current_opr->tensors()->size());
    auto tensor = m_current_opr->tensors()->Get(m_cur_opr_tensor_cnt++);
    auto comp_node = load_comp_node(tensor->comp_node());
    auto layout = load_tensor_layout(tensor, comp_node);
    auto ret = std::make_shared<HostTensorND>(comp_node, layout);

    auto&& loader = m_loader->m_cur_load_config->tensor_value_loader;
    if (tensor->data() && tensor->data()->size() > 0) {
        if (loader) {
            mgb_log_warn(
                    "serialization v2 format is pure flatbuffer format, not support "
                    "user tensor value loader callback.");
        }
        memcpy(ret->raw_ptr(), tensor->data()->data(), tensor->data()->size());
    }
    if (tensor->name()) {
        m_tensor_map[tensor->name()->str()] = ret;
    }
    if (auto&& mod = m_loader->m_cur_load_config->tensor_modifier) {
        bool has_value = false;
        if (tensor && tensor->data()) {
            has_value = tensor->data()->size() != 0;
        }
        mod(tensor->name() ? tensor->name()->str() : "", has_value, *ret);
    }
    return ret;
}

std::shared_ptr<DeviceTensorND> GraphLoaderOSSV2::OprLoadContextImpl::
        load_tensor_shared() {
    mgb_assert(
            m_current_opr->tensors() &&
            m_cur_opr_tensor_cnt < m_current_opr->tensors()->size());
    auto tensor = m_current_opr->tensors()->Get(m_cur_opr_tensor_cnt++);
    auto comp_node = load_comp_node(tensor->comp_node());
    auto layout = load_tensor_layout(tensor, comp_node);
    mgb_assert(tensor->data());
    auto&& shared_pair = m_loader->m_shared_tensor_map.at(m_cur_shared_tensor_idx++);
    auto&& shared_tensor_ref = shared_pair.second[comp_node.mem_node()];
    if (shared_tensor_ref) {
        if (shared_tensor_ref->comp_node() == comp_node)
            return shared_tensor_ref;
        // same mem node but different comp node, change comp node and share
        // value
        auto ret = std::make_shared<DeviceTensorND>(*shared_tensor_ref);
        ret->comp_node(comp_node);
        return ret;
    }
    if (tensor->name()) {
        shared_pair.first = tensor->name()->str();
    }

    if (comp_node.mem_node() == CompNode::default_cpu().mem_node()) {
        // directly forward CPU memory
        HostTensorND hv{comp_node};
        if (tensor->data() && tensor->data()->size() > 0) {
            hv.dtype(layout.dtype).resize(layout);
            memcpy(hv.raw_ptr(), tensor->data()->data(), tensor->data()->size());
        }
        shared_tensor_ref = std::make_shared<DeviceTensorND>();
        *shared_tensor_ref = DeviceTensorND::make_proxy(hv);
    } else {
        // use lazy load for non-CPU devices
        HostTensorND hv{CompNode::default_cpu()};
        if (tensor->data() && tensor->data()->size() > 0) {
            hv.dtype(layout.dtype).resize(layout);
            memcpy(hv.raw_ptr(), tensor->data()->data(), tensor->data()->size());
        }
        shared_tensor_ref = m_device_value_loader.make(comp_node, std::move(hv));
    }
    return shared_tensor_ref;
}

Metadata GraphLoaderOSSV2::OprLoadContextImpl::load_metadata() {
    const auto* fbmeta = m_loader->m_model->metadata();
    Metadata ret;
    if (fbmeta) {
        ret.is_valid = fbmeta->is_valid();
        ret.graph_modified = fbmeta->graph_modified();
        if (fbmeta->user_info()) {
            ret.user_info = fbmeta->user_info()->str();
            ret.has_user_info = true;
        }
        if (fbmeta->optimize_options()) {
            ret.optimize_options = fbmeta->optimize_options();
            ret.optimized_for_inference = true;
        }
    }
    return ret;
}

void GraphLoaderOSSV2::OprLoadContextImpl::load_single_opr(
        const fbs::v2::Operator* fbopr) {
    m_cur_opr_tensor_cnt = 0;
    m_cur_opr_blob_cnt = 0;
    m_cur_opr_param_cnt = 0;

    OperatorNodeConfig config;
    if (fbopr->output_dtype()) {
        config.output_dtype(fbs::intl::load_dtype(fbopr->output_dtype()));
    }
    if (fbopr->name()) {
        config.name(fbopr->name()->str());
    }
    if (fbopr->comp_node()) {
        auto cnt = fbopr->comp_node()->size();
        cg::OperatorNodeConfig::CompNodeArray comp_node_arr(cnt);
        for (size_t i = 0; i < cnt; i++) {
            CompNode cn{};
            auto node = fbopr->comp_node()->Get(i);
            if (node) {
                cn = load_comp_node(node);
            }
            comp_node_arr[i] = cn;
        }
        config.comp_node_arr(comp_node_arr);
    }
    //! opr version must be exist
    uint8_t opr_version = fbopr->opr_version();
    auto type_id = fbopr->type_id();
    const OprRegistryV2* registry =
            OprRegistryV2::versioned_find_by_id(type_id, opr_version);
    mgb_throw_if(
            !registry, SerializationError,
            "failed to find opr with type %s and version %d.",
            fbopr->type()->str().c_str(), opr_version);

    // load inputs
    VarNodeArray inputs;
    if (fbopr->inputs()) {
        inputs.resize(fbopr->inputs()->size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            inputs[i] = m_id2varnode.at(fbopr->inputs()->Get(i));
        }
    }

    // call loader
    auto accessor = registry->loader(*this, inputs, config);
    auto opr = accessor.opr();

    // check opr type; note that:
    // 1. registry->type may be empty for dynamic opr loaders or legacy oprs
    // 2. due to some optimization, an opr may be replaced by ImmutableTensor
    mgb_assert(
            opr && (opr->dyn_typeinfo() == registry->type || !registry->type ||
                    opr->same_type<opr::ImmutableTensor>()),
            "got_type=%s expected_type=%s", opr ? opr->dyn_typeinfo()->name : nullptr,
            registry->type->name);
    // record output vars; read output names
    size_t i = 0;
    for (auto ovar : accessor.output()) {
        if (!ovar->contain_flag(VarNode::Flag::VOLATILE_CONTENT)) {
            m_id2varnode.push_back(ovar);
            if (fbopr->outputs()) {
                auto id = fbopr->outputs()->Get(i);
                mgb_assert(
                        m_id2varnode.size() - 1 == fbopr->outputs()->Get(i),
                        "id2var is %zu, fbs get id is %d\n", m_id2varnode.size() - 1,
                        fbopr->outputs()->Get(i));
                if (m_middle_tensors.size() > i) {
                    auto name = m_middle_tensors[id]->name()->str();
                    ovar->name(name);
                }
            }
            i++;
        }
    }

    opr->node_prop().attribute().priority = fbopr->priority();
}

GraphLoader::LoadResult GraphLoaderOSSV2::OprLoadContextImpl::load_oprs() {
    // load oprs
    const auto* oprs = m_loader->m_model->oprs();
    {
        // inplace arith graph optimization is disabled during opr load
        // it tries to restore the same graph as it was dumped
        // see test TestSerializer2.LOGEXP for example
        GraphLoader::ScopedGraphOptDisabler _(m_graph);
        for (flatbuffers::uoffset_t i = 0; i < oprs->size(); ++i) {
            m_current_opr = oprs->Get(i);
            load_single_opr(m_current_opr);
        }
    }

    // batched loading device values
    m_device_value_loader.apply();

    LoadResult ret;
    ret.graph = m_graph;
    ret.tensor_map = m_tensor_map;

    const auto* outputs = m_loader->m_model->output_vars_idx();
    ret.output_var_list.resize(outputs->size());
    for (flatbuffers::uoffset_t i = 0; i < outputs->size(); i++) {
        auto out = outputs->Get(i);
        auto var = m_id2varnode.at(out->compact_id());
        ret.output_var_map[var->name()] = var;
        ret.output_var_map_id[out->original_id()] = var;
        ret.output_var_list[i] = var;
    }
    mgb_assert(m_cur_shared_tensor_idx == m_loader->m_shared_tensor_map.size());
    return ret;
}

void GraphLoaderOSSV2::OprLoadContextImpl::load_middle_tensor() {
    auto model = m_loader->m_model;
    if (model->middle_tensors()) {
        for (unsigned int i = 0; i < m_loader->m_model->middle_tensors()->size(); i++) {
            m_middle_tensors.push_back(model->middle_tensors()->Get(i));
        }
    }
}

GraphLoader::LoadResult GraphLoaderOSSV2::load(const LoadConfig& config, bool rewind) {
    mgb_assert(m_file);
    m_cur_load_config = &config;
    if (rewind) {
        m_file->rewind();
    }
    // Read fbs::Graph
    uint32_t size;
    m_file->read(&size, sizeof(size));
    m_model_buf = m_file->read_shared(size);

    mgb_throw_if(
            !fbs::v2::ModelBufferHasIdentifier(m_model_buf.data()), SerializationError,
            "invalid fbs model");

    {
        flatbuffers::Verifier verifier(
                static_cast<const uint8_t*>(m_model_buf.data()), m_model_buf.size());
        mgb_throw_if(
                !fbs::v2::VerifyModelBuffer(verifier), SerializationError,
                "model verification failed (invalid or corrupted model?)");
    }

    m_model = fbs::v2::GetModel(m_model_buf.data());
    m_mgb_version = m_model->mge_version();
    m_model_version = m_model->model_version();
    if (m_model->mge_version() > MGB_VERSION) {
        mgb_log_warn(
                "loading model from future runtime: version=%u "
                "model_version=%u",
                MGB_VERSION, m_model->mge_version());
    }
    if (m_model_version > CURRENT_VERSION) {
        mgb_log_warn(
                "The model dump in the future version %d, try to load it, maybe case "
                "load error in %d version.",
                m_model_version, CURRENT_VERSION);
    }

    if (m_shared_tensor_map.empty()) {
        m_shared_tensor_map.resize(m_model->nr_shared_tensor());
    } else {
        mgb_assert(m_shared_tensor_map.size() == m_model->nr_shared_tensor());
    }

    OprLoadContextImpl ctx{this, m_model->mge_version()};
    ctx.load_middle_tensor();
    auto metadata = ctx.load_metadata();
    auto result = ctx.load_oprs();
    result.metadata = metadata;
    if (m_model->output_alias() && m_model->output_alias()->size() > 0) {
        auto nr_alias = m_model->output_alias()->size();
        result.output_var_list.resize(nr_alias);
        for (size_t i = 0; i < nr_alias; i++) {
            auto output_alias = m_model->output_alias()->Get(i);
            std::string name = output_alias->name()->str();
            size_t id = output_alias->id();
            result.output_var_map[name] = result.output_var_map_id[id];
            result.output_var_list[i] = result.output_var_map_id[id];
        }
    }
    m_model_loaded = true;
    result.graph_compile_ahead();
    return result;
}

std::unique_ptr<GraphDumper> make_fbs_v2_dumper(
        std::unique_ptr<OutputFile> file, int version) {
    return std::make_unique<GraphDumperOSSV2>(std::move(file), version);
}

std::unique_ptr<GraphLoader> make_fbs_v2_loader(std::unique_ptr<InputFile> file) {
    return std::make_unique<GraphLoaderOSSV2>(std::move(file));
}

bool is_fbs_v2_file(InputFile& file) {
    constexpr size_t identifier_length = 25;
    char identifier[identifier_length];
    file.read(identifier, identifier_length);
    file.skip(-identifier_length);
    //! skip the size in prefix of the file
    return fbs::v2::ModelBufferHasIdentifier(identifier + sizeof(uint32_t));
}

}  // namespace serialization
}  // namespace mgb

#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
