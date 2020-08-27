#include "megbrain/opr/standalone/nms_opr.h"

#if MGB_CUDA
#include "./nms_kern.cuh"
#endif
#include "./nms_cpu.h"

#include "megbrain/comp_node_env.h"
#include "megbrain/serialization/sereg.h"
#include "megbrain/utils/arith_helper.h"  // for get_aligned_power2

#if MGB_ENABLE_FBS_SERIALIZATION
#include "megbrain/serialization/internal/mgb_cpp_opr_generated.h"
#include "megbrain/serialization/internal/schema_generated.h"
#endif

using namespace mgb::opr::standalone;

MGB_DYN_TYPE_OBJ_FINAL_IMPL(NMSKeep);

class NMSKeep::Kern {
public:
    virtual ~Kern() = default;

    //! get workspace size in bytes
    virtual size_t get_workspace_size(const NMSKeep* opr,
                                      const TensorShape& boxes) = 0;
    virtual void exec(const NMSKeep* opr, const DeviceTensorND& inp,
                      const DeviceTensorND& out_idx,
                      const DeviceTensorND& out_size,
                      const DeviceTensorND& workspace) = 0;
};

// f{{{ cuda kernel begins
#if MGB_CUDA
class NMSKeep::CUDAKern final : public Kern {
    size_t m_workspace_overlap_mask_bytes, m_workspace_overlap_mask_bytes_align,
            m_workspace_rm_mask_bytes;

    void init(const NMSKeep* opr, const TensorShape& boxes) {
        auto align = opr->comp_node().get_mem_addr_alignment();
        size_t nr_boxes = boxes[1];
        m_workspace_overlap_mask_bytes =
                nr_boxes * DIVUP(nr_boxes, 64) * sizeof(uint64_t);
        m_workspace_overlap_mask_bytes_align =
                get_aligned_power2(m_workspace_overlap_mask_bytes, align);
        m_workspace_rm_mask_bytes = DIVUP(nr_boxes, 64) * sizeof(uint64_t);
    }

public:
    size_t get_workspace_size(const NMSKeep* opr,
                              const TensorShape& boxes) override {
        init(opr, boxes);
        return m_workspace_overlap_mask_bytes_align + m_workspace_rm_mask_bytes;
    }

    void exec(const NMSKeep* opr, const DeviceTensorND& inp,
              const DeviceTensorND& out_idx, const DeviceTensorND& out_size,
              const DeviceTensorND& workspace) override;
};

void NMSKeep::CUDAKern::exec(const NMSKeep* opr, const DeviceTensorND& inp,
                             const DeviceTensorND& out_idx,
                             const DeviceTensorND& out_size,
                             const DeviceTensorND& workspace) {
    // NOTE: input comp node might be different from output comp node (for
    // example, CUDA stream may be modified to overlap computations); a
    // SingleCNOperatorNodeBase is expected to execute on a single comp node,
    // and the comp node is defined as the output comp node
    CompNode comp_node = out_idx.comp_node();

    // comp ndoe is also accessible from SingleCNOperatorNode
    mgb_assert(comp_node == opr->comp_node());

    // CompNodeEnv contains platform-specific properties of a CompNode
    auto&& cuda_env = CompNodeEnv::from_comp_node(comp_node).cuda_env();
    mgb_assert(cuda_env.device_prop.warpSize == 32, "invalid warp size: %d",
               cuda_env.device_prop.warpSize);
    auto stream = cuda_env.stream;

    init(opr, inp.shape());

    auto inp_ptr = inp.ptr<float>();
    auto dev_overlap_mask = reinterpret_cast<uint64_t*>(workspace.raw_ptr()),
         dev_rm_mask = reinterpret_cast<uint64_t*>(
                 workspace.raw_ptr() + m_workspace_overlap_mask_bytes_align);
    auto out_idx_ptr = reinterpret_cast<uint32_t*>(out_idx.ptr<int32_t>()),
         out_size_ptr = reinterpret_cast<uint32_t*>(out_size.ptr<int32_t>());
    size_t batch = inp.shape(0), nr_boxes = inp.shape(1);

    MGB_CUDA_CHECK(cudaMemsetAsync(dev_overlap_mask, 0,
                                   m_workspace_overlap_mask_bytes, stream));

    auto max_output = opr->param().max_output;

    for (size_t i = 0; i < batch; ++i) {
        nms::launch_gen_mask(nr_boxes, opr->param().iou_thresh,
                             inp_ptr + i * nr_boxes * 4, DIVUP(nr_boxes, 64),
                             dev_overlap_mask, stream);

        MGB_CUDA_CHECK(cudaMemsetAsync(dev_rm_mask, 0,
                                       m_workspace_rm_mask_bytes, stream));
        nms::launch_gen_indices(nr_boxes, max_output, DIVUP(nr_boxes, 64),
                                dev_overlap_mask, dev_rm_mask,
                                out_idx_ptr + i * max_output, out_size_ptr + i,
                                stream);
    }
}

#endif  // MGB_CUDA for CUDAKern
// f}}} cuda kernel ends

// f{{{ cpu kernel begins
class NMSKeep::CPUKern final : public Kern {
public:
    ~CPUKern() = default;

    size_t get_workspace_size(const NMSKeep*,
                              const TensorShape& boxes) override {
        return nms::cpu_kern_workspace(boxes.shape[1]);
    }

    void exec(const NMSKeep* opr, const DeviceTensorND& inp,
              const DeviceTensorND& out_idx, const DeviceTensorND& out_size,
              const DeviceTensorND& workspace) override;
};
void NMSKeep::CPUKern::exec(const NMSKeep* opr, const DeviceTensorND& inp,
                            const DeviceTensorND& out_idx,
                            const DeviceTensorND& out_size,
                            const DeviceTensorND& workspace) {
    // See CUDAKern::exec for more explanation on output comp nodes.
    CompNode comp_node = out_idx.comp_node();

    auto inp_ptr = inp.ptr<float>();
    auto out_idx_ptr = reinterpret_cast<uint32_t*>(out_idx.ptr<int32_t>()),
         out_size_ptr = reinterpret_cast<uint32_t*>(out_size.ptr<int32_t>());
    size_t batch = inp.shape(0), nr_boxes = inp.shape(1);
    auto param = opr->param();

    auto workspace_ptr = workspace.raw_ptr();

    // NOTE: we must copy all the params into the kernel closure since it would
    // be dispatched on a different thread
    auto kern = [=]() {
        for (size_t i = 0; i < batch; ++i) {
            nms::cpu_kern(nr_boxes, param.max_output, param.iou_thresh,
                          inp_ptr + i * nr_boxes * 4,
                          out_idx_ptr + i * param.max_output, out_size_ptr + i,
                          workspace_ptr);
        }
    };

    // The kernel should not be invoked
    CompNodeEnv::from_comp_node(comp_node).cpu_env().dispatch(kern);
}

// f}}} cpu kernel ends

NMSKeep::NMSKeep(VarNode* boxes, const Param& param,
                 const OperatorNodeConfig& config)
        : Super(boxes->owner_graph(),  // owner graph
                config,                // OperatorNodeConfig
                "nms_keep",  // opr type name (used for generating opr name)
                {boxes}      // input vars for generating opr name
                ),
          m_param{param} {
    mgb_assert(boxes->dtype() == dtype::Float32(),
               "input should be float32; got %s", boxes->dtype().name());
    // setup m_kern according to device type
    switch (boxes->comp_node().device_type()) {
#if MGB_CUDA
        case CompNode::DeviceType::CUDA:
            m_kern = std::make_unique<CUDAKern>();
            break;
#endif
        case CompNode::DeviceType::CPU:
            m_kern = std::make_unique<CPUKern>();
            break;
        default:
            mgb_throw(MegBrainError, "NMSKeep: unsupported device type: %s",
                      boxes->comp_node().to_string().c_str());
    }

    add_input({boxes});
    add_output("indices")->dtype(dtype::Int32());
    add_output("sizes")->dtype(dtype::Int32());
    cg::add_workspace_output(this);  // workspace is also an output var

    // make the graph deduplication system consider m_param (so two oprs with
    // same input vars but different param values would not be deduplicated)
    add_equivalence_component<PODHash<Param>>(&m_param);
}

// impl dtor after Kern is defined
NMSKeep::~NMSKeep() noexcept = default;

mgb::SymbolVar NMSKeep::make(SymbolVar boxes, const Param& param,
                             const OperatorNodeConfig& config) {
    // SymbolVar is just a wrapper of VarNode*, with overloaded methods such as
    // operator+()
    auto bvar = boxes.node();
    // insert opr into the owner graph of boxes
    return boxes.insert_single_output_opr<NMSKeep>(bvar, param, config);
}

void NMSKeep::get_output_var_shape(const TensorShapeArray& inp_shape,
                                   TensorShapeArray& out_shape) const {
    auto boxes = inp_shape.at(0);
    mgb_assert(boxes.ndim == 3 && boxes.shape[2] == 4, "invalid box shape: %s",
               boxes.to_string().c_str());

    // out_shape should match the outputs added in the constructor
    mgb_assert(out_shape.size() == 3);

    auto batch = boxes[0];
    out_shape[0] = {batch, m_param.max_output};                // indices
    out_shape[1] = {batch};                                    // sizes
    out_shape[2] = {m_kern->get_workspace_size(this, boxes)};  // workspace
}

void NMSKeep::add_input_layout_constraint() {
    input(0)->add_layout_constraint_contiguous();
}

void NMSKeep::scn_do_execute() {
    DeviceTensorND empty_workspace;
    m_kern->exec(this, input(0)->dev_tensor(), output(0)->dev_tensor(),
                 output(1)->dev_tensor(),
                 // if workspace size is 0, output(2) would be invalid and its
                 // dev_tensor() can not be accessed
                 output(2)->dev_tensor_valid() ? output(2)->dev_tensor()
                                               : empty_workspace);
}

#if MGB_ENABLE_FBS_SERIALIZATION

namespace mgb {
namespace serialization {
namespace fbs {

template <>
struct ParamConverter<opr::standalone::NMSKeep::Param> {
    using FlatBufferType = param::NMSKeep;
    static opr::standalone::NMSKeep::Param to_param(const FlatBufferType* fb) {
        return {fb->iou_thresh(), fb->max_output()};
    }
    static flatbuffers::Offset<FlatBufferType> to_flatbuffer(
            flatbuffers::FlatBufferBuilder& builder,
            const opr::standalone::NMSKeep::Param& p) {
        return param::CreateNMSKeep(builder, p.iou_thresh, p.max_output);
    }
};

}  // namespace fbs
}  // namespace serialization
}  // namespace mgb

#endif

namespace mgb {

void _hack_pull_in_nms_opr_object() {}

}  // namespace mgb

// register serialization: the default implementation uses Opr::Param; it
// requires Param::TAG, Opr::param() and Opr::make(..., param) to exist
// Note: the second param 1 here means that this operator has one input
using NMSKeepMGB = NMSKeep;
MGB_SEREG_OPR(NMSKeepMGB, 1);

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
