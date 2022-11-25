#include "megbrain_build_config.h"
#include "megdnn/tensor_format.h"
#if MGB_JIT && MGB_OPENCL

#include "./codegen_opencl.h"
#include "./compiler.h"
#include "./utils.h"

#include "megbrain/common.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/rdnn/management.h"
#include "megbrain/utils/timer.h"

using namespace mgb;
using namespace jit;

/* ==================== OpenCLTinyCompiler ===================== */

OpenCLTinyCompiler::OpenCLTinyCompiler(CompNode::DeviceType device_type) {
    m_is_debug = ::std::getenv("OPENCL_JIT_DEBUG") ? true : false;
    mgb_assert(
            CompNode::DeviceType::OPENCL == device_type,
            "error init OpenCLTinyCompiler");
}

std::unique_ptr<Executable> OpenCLTinyCompiler::do_compile(
        const InternalGraph& graph, const JITExecutor::Args& args) {
    std::string source, kernel_name;
    std::tie(kernel_name, source) = codegen_opencl(graph, args);
    if (m_is_debug) {
        mgb_log_debug("kernel: name: %s\n%s", kernel_name.c_str(), source.c_str());
    }
    auto ret = std::make_unique<OpenCLExecutable>(
            std::move(source), std::move(kernel_name), m_is_debug);
    return ret;
}

size_t OpenCLTinyCompiler::get_nr_workspace_outputs(JITExecutor* opr) const {
    MGB_MARK_USED_VAR(opr);
    return 0;
}

void OpenCLTinyCompiler::init_workspace_size_infer(JITExecutor* opr) {
    MGB_MARK_USED_VAR(opr);
}

/* =================== OpenCLExecutable ==================== */

OpenCLExecutable::OpenCLExecutable(std::string source, std::string name, bool is_debug)
        : m_source{std::move(source)}, m_name{std::move(name)}, m_is_debug{is_debug} {}

void OpenCLExecutable::execute(JITExecutor* fusion_opr) {
    auto&& cn = fusion_opr->comp_node();
    auto& env = CompNodeEnv::from_comp_node(cn).opencl_env();
    auto handle = mgb::opr::intl::get_megdnn_handle(cn);
    auto mgr = env.opencl_mgr;
    auto&& ctx = mgr->context();
    auto& queue = mgr->command_queue();
    auto&& kernel = megdnn::opencl::OpenCLKernel(handle);
    auto& args = fusion_opr->args();

    static auto&& prop = megcore::opencl::OpenCLProp(mgr->device());
    bool is_adreno = prop.is_adreno();
    bool is_mali = prop.is_mali();
    auto max_work_group = static_cast<uint32_t>(prop.max_work_group_size());
    mgb_assert(
            prop.is_support_image(),
            "code issue happened, OpenCL jit only support device with support image");

    //! for debug
    MGB_MARK_USED_VAR(ctx);
    MGB_MARK_USED_VAR(queue);

    size_t WGSX = 0;
    size_t WGSY = 0;

    //! create cl args
    for (size_t i = 0; i < args.inputs.size(); i++) {
        if (TensorFormat::Type::IMAGE2D_PACK4 == args.inputs[i].layout.format.type()) {
            WGSX = std::max(
                    WGSX,
                    args.inputs[i]
                            .layout.format.as_impl<megdnn::Image2DPack4TensorFormat>()
                            .image_width(args.inputs[i].layout));
            WGSY = std::max(
                    WGSY,
                    args.inputs[i]
                            .layout.format.as_impl<megdnn::Image2DPack4TensorFormat>()
                            .image_height(args.inputs[i].layout));
        }
    }
    mgb_assert(WGSX > 0 && WGSY > 0, "invalid tensor for OpenCL jit");

    if (m_is_debug) {
        mgb_log_debug(
                "OpenCLExecutable init input tensor array with size: %zu, init output "
                "tensor array with size: %zu",
                args.inputs.size(), args.outputs.size());
        for (size_t i = 0; i < args.inputs.size(); i++) {
            mgb_log_debug(
                    "input(%zu) dim: %zu %s", i, args.inputs[i].layout.ndim,
                    args.inputs[i].layout.to_string().c_str());
        }
        for (size_t i = 0; i < args.outputs.size(); i++) {
            mgb_log_debug(
                    "output(%zu) dim: %zu %s", i, args.outputs[i].layout.ndim,
                    args.outputs[i].layout.to_string().c_str());
        }
    }
    mgb_assert(
            args.outputs.size() == 1, "OpenCL elemwise jit output size should be one");
    size_t h = args.outputs[0].layout[1];

    //! create kernel
    std::string compile_options;
    kernel.set_meta_data({compile_options, m_source});
    kernel.set_kernel_name(m_name);
    kernel.build_kernel();

    //! set tensor args
    for (size_t i = 0; i < args.inputs.size(); i++) {
        if (TensorFormat::Type::IMAGE2D_PACK4 == args.inputs[i].layout.format.type()) {
            kernel.add_tensor_image_args(
                    {{args.inputs[i].from->dev_tensor().raw_ptr(),
                      args.inputs[i].layout}});
        } else {
            //! scalar default format case
            kernel.add_tensor_arg(
                    {args.inputs[i].from->dev_tensor().raw_ptr(),
                     args.inputs[i].layout});
        }
    }
    kernel.add_tensor_image_args(
            {{args.outputs[0].from->dev_tensor().raw_ptr(), args.outputs[0].layout}});

    uint32_t block_w = 1, block_h = 1, dimx = 1, dimy = 1;
    auto config_super_parameter = [&] {
        if (is_adreno) {
            block_w = 1;
            dimx = 64;
            dimy = 1;
        } else if (is_mali) {
            block_w = 1;
            dimx = 96;
            dimy = 1;
        } else {
            //! unknown gpu case
            block_w = 1;
            dimx = 64;
            dimy = 1;
        }
        //! float16 case
        if (dtype::Float16() == args.inputs[0].layout.dtype) {
            dimx *= 2;
        }

        //! scaling dimx less than gws0, dimy less than gws1
        dimx = std::min(dimx, static_cast<uint32_t>((WGSX + block_w - 1) / block_w));
        dimy = std::min(dimy, static_cast<uint32_t>((WGSY + block_h - 1) / block_h));

        //! scaling dimx * dimy less than device max_work_group
        dimx = std::min(
                dimx, std::max(static_cast<uint32_t>(1), max_work_group / dimy));
    };

    config_super_parameter();

    //! set other args and config lws and gws
    int wc_size = WGSX;
    int hb_size = WGSY;
    WGSX = (WGSX + block_w - 1) / block_w;
    WGSY = (WGSY + block_h - 1) / block_h;
    int i_WGSX = safe_int<size_t>(WGSX);
    int i_WGSY = safe_int<size_t>(WGSY);
    kernel.add_args(
            {{&i_WGSX, sizeof(int)},
             {&i_WGSY, sizeof(int)},
             {&wc_size, sizeof(int)},
             {&hb_size, sizeof(int)},
             {&h, sizeof(int)}});
    //! have broadcasted_channel_like_input case
    int may_w_size = args.outputs[0].layout[3];
    kernel.add_arg({&may_w_size, sizeof(cl_uint)});
    mgb_log_debug(
            "config OpenCL jit kernel args: lws: (%d %d), i_WGSX: %d, i_WGSY: %d "
            "wc_size: %d, hb_size: %d, w_size: %d",
            dimx, dimy, i_WGSX, i_WGSY, wc_size, hb_size, may_w_size);

    kernel.set_local_size({dimx, dimy});
    kernel.set_global_size_divup_consider_uniform_gws({WGSX, WGSY});

    //! enqueue kernel
    kernel.run();
}

#endif  // MGB_OPENCL
