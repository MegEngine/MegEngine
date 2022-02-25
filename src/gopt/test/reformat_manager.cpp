#include "./helper.h"

#include "megbrain/gopt/reformat_manager.h"
#include "megbrain/graph/event.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/plugin/base.h"
#include "megbrain/plugin/profiler.h"

using namespace mgb;
using namespace gopt;

TEST(TestReformatManager, Feature) {
    constexpr size_t N = 16, C = 128, H = 7, W = 7;
    HostTensorGenerator<> gen;
    using ReformatKey = ReformatManager::ReformatKey;
    auto src_format = TensorFormats::NHWC, dst_format = TensorFormats::NCHWc64;
    ReformatKey key{src_format, dst_format};
    auto reformat = ReformatManager::instance().get(key);

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;

    auto r = [](VarNode* inp) {
        auto x = SymbolVar(inp);
        auto xshp = opr::GetVarShape::make(x);
        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp0 =
                opr::Concat::make({sub(0), sub(1), sub(2), sub(3) / 64, cv(64)}, 0);
        auto y0 = opr::Reshape::make(x, tshp0);
        auto y1 = opr::Dimshuffle::make(y0, {0, 3, 1, 2, 4});
        return y1;
    };

    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp)).rename(name);
    };
    auto x = mkvar("x", {N, H, W, C});
    auto y1 = SymbolVar(reformat({x.node()}));
    auto y2 = r(x.node());
    size_t nr_shapeof = 0;
    size_t nr_reshape = 0;
    cg::DepOprIter{[&nr_shapeof, &nr_reshape](cg::OperatorNodeBase* o) {
        if (o->same_type<opr::GetVarShape>())
            nr_shapeof++;
        if (o->same_type<opr::Reshape>())
            nr_reshape++;
    }}.add(y1.node()->owner_opr());
    ASSERT_EQ(nr_shapeof, 1);
    ASSERT_EQ(nr_reshape, 1);
    HostTensorND t1, t2;
    auto func1 = graph->compile({make_callback_copy(y1, t1)});
    func1->execute();
    auto func2 = graph->compile({make_callback_copy(y2, t2)});
    func2->execute();
    MGB_ASSERT_TENSOR_EQ(t1, t2);
}

TEST(TestReformatManager, Weight) {
    constexpr size_t G = 8, K = 128, C = 128, R = 3, S = 3;
    HostTensorGenerator<> gen;
    using ReformatKey = ReformatManager::ReformatKey;
    auto src_format = TensorFormats::GKCRS, dst_format = TensorFormats::GKCRSk4c4;
    ReformatKey key{src_format, dst_format};
    auto reformat = ReformatManager::instance().get(key);

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;

    auto r = [](VarNode* inp) {
        auto x = SymbolVar(inp);
        auto xshp = opr::GetVarShape::make(x);
        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp0 = opr::Concat::make(
                     {sub(0), sub(1) / 4, cv(4), sub(2) / 4, cv(4), sub(3), sub(4)}, 0),
             tshp1 = opr::Concat::make(
                     {sub(0), sub(1) / 4, sub(2) / 4, sub(3), sub(4), cv(4), cv(4)}, 0);
        auto y0 = opr::Reshape::make(x, tshp0);
        auto y1 = opr::Dimshuffle::make(y0, {0, 1, 3, 5, 6, 2, 4});
        auto y2 = opr::Reshape::make(y1, tshp1);
        return y2;
    };

    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp)).rename(name);
    };
    auto w = mkvar("w", {G, K / G, C / G, R, S});
    auto y1 = SymbolVar(reformat({w.node()}));
    auto y2 = r(w.node());
    size_t nr_shapeof = 0;
    size_t nr_reshape = 0;
    cg::DepOprIter{[&nr_shapeof, &nr_reshape](cg::OperatorNodeBase* o) {
        if (o->same_type<opr::GetVarShape>())
            nr_shapeof++;
        if (o->same_type<opr::Reshape>())
            nr_reshape++;
    }}.add(y1.node()->owner_opr());
    ASSERT_EQ(nr_shapeof, 1);
    ASSERT_EQ(nr_reshape, 1);
    HostTensorND t1, t2;
    auto func1 = graph->compile({make_callback_copy(y1, t1)});
    func1->execute();
    auto func2 = graph->compile({make_callback_copy(y2, t2)});
    func2->execute();
    MGB_ASSERT_TENSOR_EQ(t1, t2);
}

TEST(TestReformatManager, InvalidKey) {
    using ReformatKey = ReformatManager::ReformatKey;
    using Attribute = ReformatKey::Attribute;
    auto src_format = TensorFormats::GKCRS, dst_format = TensorFormats::GKCRSk4c4;
    Attribute attribute = Attribute::IMAGE2D;
    ReformatKey key{src_format, dst_format, attribute};
    ASSERT_THROW(ReformatManager::instance().get(key), AssertionError);
}

TEST(TestReformatManager, InputChannelSmall) {
    constexpr size_t N = 16, C = 3, H = 224, W = 224;
    auto cn = CompNode::load("cpux");
    HostTensorGenerator<> gen;
    using ReformatKey = ReformatManager::ReformatKey;
    using Attribute = ReformatKey::Attribute;
    auto src_format = TensorFormats::NCHW, dst_format = TensorFormats::NCHWc4;
    ReformatKey key{src_format, dst_format, Attribute::IC_SMALL};
    auto reformat = ReformatManager::instance().get(key);

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;

    auto r = [](VarNode* inp) {
        auto x = SymbolVar(inp);
        auto y = opr::RelayoutFormat::make(
                x, megdnn::param::RelayoutFormat::Mode::NCHW_NCHW4_IC_SMALL);
        return y;
    };

    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name);
    };
    auto x = mkvar("x", {N, C, H, W});
    auto y1 = SymbolVar(reformat({x.node()}));
    auto y2 = r(x.node());
    HostTensorND t1, t2;
    auto func1 = graph->compile({make_callback_copy(y1, t1)});
    func1->execute();
    auto func2 = graph->compile({make_callback_copy(y2, t2)});
    func2->execute();
    MGB_ASSERT_TENSOR_EQ(t1, t2);
}

TEST(TestReformatManager, AutoAlignedFeature) {
    constexpr size_t N = 16, C = 22, H = 55, W = 55;
    HostTensorGenerator<> gen;
    using ReformatKey = ReformatManager::ReformatKey;
    auto src_format = TensorFormats::NCHWc4, dst_format = TensorFormats::NCHWc32;
    ReformatKey key{src_format, dst_format};

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;

    std::shared_ptr<HostTensorND> host_orig_x = gen({N, C, H, W});
    std::shared_ptr<HostTensorND> host_x = gen({N, (C + 3) / 4, H, W, 4});
    auto mkvar = [&](const char* name, const std::shared_ptr<HostTensorND>& host_val) {
        return opr::Host2DeviceCopy::make(*graph, host_val).rename(name);
    };
    auto orig_x = mkvar("orig_x", host_orig_x);
    auto x = mkvar("x", host_x);
    auto builder = ReformatManager::instance().auto_aligned_reformat_featrue(
            orig_x.node(), TensorFormats::NCHW, key);
    auto y = builder({x.node()});
    HostTensorND t;
    auto func = graph->compile({make_callback_copy(y, t)});
    func->execute();
    *host_x = *gen({(N + 5), (C + 3) / 4, H, W, 4});
    func->execute();
    *host_x = *gen({(N - 5), (C + 3) / 4, H, W, 4});
    func->execute();
    auto shp = TensorShape{(N - 5), (C + 31) / 32, H, W, 32};
    ASSERT_TRUE(shp.eq_shape(t.shape()));
}

TEST(TestReformatManager, AutoAlignedFeatureB4) {
    constexpr size_t N = 16, C = 94, H = 55, W = 55;
    HostTensorGenerator<> gen;
    using ReformatKey = ReformatManager::ReformatKey;
    auto src_format = TensorFormats::NCHWc4, dst_format = TensorFormats::NCHWc64;
    ReformatKey key{src_format, dst_format};

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;

    std::shared_ptr<HostTensorND> host_orig_x = gen({N, C, H, W});
    std::shared_ptr<HostTensorND> host_x = gen({N, (C + 3) / 4, H, W, 4});
    auto mkvar = [&](const char* name, const std::shared_ptr<HostTensorND>& host_val,
                     const DType& dtype) {
        return opr::TypeCvt::make(
                opr::Host2DeviceCopy::make(*graph, host_val).rename(name), dtype);
    };
    auto orig_x =
            mkvar("orig_x", host_orig_x,
                  dtype::Quantized4Asymm(20.f, static_cast<uint8_t>(8)));
    auto x = mkvar("x", host_x, dtype::Quantized4Asymm(25.f, static_cast<uint8_t>(4)));
    auto builder = ReformatManager::instance().auto_aligned_reformat_featrue(
            orig_x.node(), TensorFormats::NCHW, key);
    auto y = builder({x.node()});
    HostTensorND t;
    auto func = graph->compile({make_callback_copy(y, t)});
    func->execute();
}

TEST(TestReformatManager, AutoAlignedWeight) {
    constexpr size_t K = 32, C = 32, R = 3, S = 3;
    HostTensorGenerator<> gen;
    using ReformatKey = ReformatManager::ReformatKey;
    auto src_format = TensorFormats::NCHW, dst_format = TensorFormats::NCHWc64;
    ReformatKey key{src_format, dst_format};

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;

    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp)).rename(name);
    };
    auto w = mkvar("w", {K, C, R, S});
    auto builder = ReformatManager::instance().auto_aligned_reformat_weight(
            w.node(), key,
            ReformatManager::AlignmentDesc{megdnn::Dimension::Name::N, 64});
    auto y = builder({w.node()});
    HostTensorND t;
    auto func = graph->compile({make_callback_copy(y, t)});
    func->execute();
}

#if MGB_CUDA
#include "megbrain/comp_node_env.h"
namespace {
class ReformatProfiler : public PluginBase {
    using CompNodeEventPtr = std::unique_ptr<CompNode::Event>;

public:
    class MarkInputContiguous;
    ReformatProfiler(
            cg::ComputingGraph* graph, cg::OperatorNodeBase* opr_start,
            cg::OperatorNodeBase* opr_end);
    ~ReformatProfiler() noexcept;
    double duration() const;

private:
    CompNodeEventPtr m_start, m_end;
    cg::OperatorNodeBase *m_opr_start, *m_opr_end;
};

ReformatProfiler::ReformatProfiler(
        cg::ComputingGraph* graph, cg::OperatorNodeBase* opr_start,
        cg::OperatorNodeBase* opr_end)
        : PluginBase(graph), m_opr_start(opr_start), m_opr_end(opr_end) {
    using namespace cg::event;
    auto on_reformat_start = [this](BeforeKernel const& event) {
        auto opr = event.opr;
        if (opr != m_opr_start)
            return;
        if (m_start == nullptr) {
            m_start = event.comp_node.create_event(CompNode::Event::NEED_TIMER);
        }
        m_start->record();
    };
    auto on_reformat_end = [this](AfterKernel const& event) {
        auto opr = event.opr;
        if (opr != m_opr_end)
            return;
        if (m_end == nullptr) {
            m_end = event.comp_node.create_event(CompNode::Event::NEED_TIMER);
        }
        m_end->record();
    };
    auto&& ev = graph->event();
    add_event_handler(ev.register_receiver<BeforeKernel>(on_reformat_start));
    add_event_handler(ev.register_receiver<AfterKernel>(on_reformat_end));
}

ReformatProfiler::~ReformatProfiler() noexcept {
    if (m_start)
        m_start->host_wait();
    if (m_end)
        m_end->host_wait();
}

double ReformatProfiler::duration() const {
    mgb_assert(m_end);
    m_end->host_wait();
    return m_start->elapsed_time_until(*m_end) - m_start->elapsed_time_until(*m_start);
}

MGB_DEFINE_OPR_CLASS(
        ReformatProfiler::MarkInputContiguous, cg::SingleCNOperatorNodeBase) // {
    void scn_do_execute() override{};
    void init_output_static_infer_desc() override;
    void add_input_layout_constraint() override;

public:
    MarkInputContiguous(VarNode* node, const OperatorNodeConfig& config);

    static SymbolVar make(SymbolVar node, const OperatorNodeConfig& config = {});
};  // namespace

MGB_DYN_TYPE_OBJ_FINAL_IMPL(ReformatProfiler::MarkInputContiguous);

ReformatProfiler::MarkInputContiguous::MarkInputContiguous(
        VarNode* node, const OperatorNodeConfig& config)
        : Super(node->owner_graph(), config, "mark_contiguous", {node}) {
    add_input({node});
    add_output(None);
}

SymbolVar ReformatProfiler::MarkInputContiguous::make(
        SymbolVar node, const OperatorNodeConfig& config) {
    return node.insert_single_output_opr<MarkInputContiguous>(node.node(), config);
}

void ReformatProfiler::MarkInputContiguous::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto&& mgr = owner_graph()->static_infer_manager();
    mgr.register_shape_infer(output(0), ShapeInferDesc::make_identity(input(0)));
}

void ReformatProfiler::MarkInputContiguous::add_input_layout_constraint() {
    input(0)->add_layout_constraint_contiguous();
}

class CUTimer {
public:
    CUTimer(cudaStream_t& stream, cudaEvent_t& evt0, cudaEvent_t& evt1)
            : m_stream{stream}, m_evt0{evt0}, m_evt1{evt1} {
        reset();
    }

    void reset() {
        m_started = false;
        m_stopped = false;
    }
    void start() {
        mgb_assert(!m_started);
        mgb_assert(!m_stopped);
        m_started = true;
        cudaEventRecord(m_evt0, m_stream);
    }
    void stop() {
        mgb_assert(m_started);
        mgb_assert(!m_stopped);
        m_stopped = true;
        cudaEventRecord(m_evt1, m_stream);
    }
    size_t get_time_in_us() const {
        cudaStreamSynchronize(m_stream);
        float t = -1;
        cudaEventElapsedTime(&t, m_evt0, m_evt1);
        return static_cast<size_t>(t * 1e3);
    }

private:
    bool m_started, m_stopped;
    size_t m_start_point, m_stop_point;
    cudaStream_t& m_stream;
    cudaEvent_t &m_evt0, &m_evt1;
};

}  // namespace

TEST(TestReformatManager, AutoAlignedFeatureProfiling) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpux");
    using ReformatKey = ReformatManager::ReformatKey;
    auto dtype = dtype::Quantized4Asymm(20.f, static_cast<uint8_t>(4));
    HostTensorND hval(cn, dtype);
    constexpr size_t N = 16, C = 18, H = 55, W = 55;
    hval.resize({N, (C + 63) / 64, H, W, 64});
    std::shared_ptr<DeviceTensorND> dval = std::make_shared<DeviceTensorND>(cn, dtype);
    dval->copy_from(hval).sync();
    std::shared_ptr<DeviceTensorND> dprime =
            std::make_shared<DeviceTensorND>(cn, dtype);
    dprime->resize({N, C, H, W});

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    graph->options().var_sanity_check_first_run = false;

    auto x = opr::VolatileSharedDeviceTensor::make(*graph, dval);
    auto xprime = opr::VolatileSharedDeviceTensor::make(*graph, dprime);
    ReformatKey key{TensorFormats::NCHWc64, TensorFormats::NCHW};
    auto builder = ReformatManager::instance().auto_aligned_reformat_featrue(
            xprime.node(), TensorFormats::NCHW, key);
    auto y = builder({x.node()});
    auto mark = ReformatProfiler::MarkInputContiguous::make(SymbolVar(y));
    auto cb = [](DeviceTensorND& d) { MGB_MARK_USED_VAR(d); };
    auto output_spec = std::make_pair(mark, cb);
    auto func = graph->compile({output_spec});
    static constexpr size_t RUNS = 100;
    cn.activate();
    auto stream = CompNodeEnv::from_comp_node(cn).cuda_env().stream;
    cudaEvent_t evt0;
    cudaEvent_t evt1;
    MGB_CUDA_CHECK(cudaEventCreate(&evt0));
    MGB_CUDA_CHECK(cudaEventCreate(&evt1));
    CUTimer timer(stream, evt0, evt1);
    timer.start();
    for (size_t i = 0; i < RUNS; ++i)
        func->execute();
    timer.stop();
    double time_cuda_evt = timer.get_time_in_us() / static_cast<double>(RUNS);

    OperatorNodeBase* start = x.node()->owner_opr();
    OperatorNodeBase* end = y->owner_opr();
    std::unique_ptr<ReformatProfiler> profiler =
            std::make_unique<ReformatProfiler>(graph.get(), start, end);
    ASSERT_TRUE(y->shape().eq_shape(TensorShape{N, C, H, W}));
    for (size_t i = 0; i < RUNS; ++i)
        func->execute();
    double time_profiler = profiler->duration() * 1e6;
    printf("time: %f, %f\n", time_cuda_evt, time_profiler);
    MGB_CUDA_CHECK(cudaEventDestroy(evt0));
    MGB_CUDA_CHECK(cudaEventDestroy(evt1));
}
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
