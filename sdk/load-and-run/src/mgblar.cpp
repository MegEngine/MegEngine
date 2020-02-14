/**
 * \file sdk/load-and-run/src/mgblar.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./mgblar.h"
#include "./infile_persistent_cache.h"

#include "megbrain/utils/debug.h"
#include "megbrain/serialization/serializer.h"
#include "megbrain/serialization/extern_c_opr.h"
#include "megbrain/plugin/opr_io_dump.h"
#include "megbrain/plugin/profiler.h"
#include "megbrain/plugin/num_range_checker.h"
#include "megbrain/plugin/cpu_dispatch_checker.h"
#include "megbrain/plugin/var_value_checker.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/utility.h"
#include "megbrain/gopt/inference.h"
#include "megbrain/comp_node_env.h"

#include "megbrain/system.h"
#include "megbrain/version.h"
#include "megdnn/version.h"

#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <cstdio>
#include <sstream>

#if defined(_WIN32)
#include <io.h>
#define F_OK 0
#define access(a, b) _access(a, b)
#elif __linux__ || __unix__ || __APPLE__
#include <unistd.h>
#include <dlfcn.h>
#endif

#if MGB_ENABLE_TENSOR_RT
#include "megbrain/tensorrt/tensorrt_engine_cache.h"
#endif

using namespace mgb;

namespace {

const char* OPTIONS_DESC =
R"__usage__(
  --cpu|--cpu-default
)__usage__"
R"__usage__(
    Require to compute on CPU or OpenCL. By default CUDA is used if available,
    and CPU is used if CUDA is not available. Use --cpu-default to compute on
    CPU and dispatch all tasks in the caller thread.
  --multithread|--multithread-default <nr_thread>
    Use --multithread to compute on CPU with multi threads.
    Use --multithread-default to compute on CPU with multi threads and
    the caller thread is main thread of the multi thread pool, follow by
    thread number
  --multi-thread-core-ids
    The multi thread affinity core set, separated with ',', the number of digital
    will be the thread number. for example:--multi-thread-core-ids "0,1,2,3", the
    number thread if 4,the main thread binding the last core '3',
    for best performance, the main thread should binding to the fast core.
  --profile|--profile-host <output>
    Write profiling result to given file. The output file is in JSON format and
    can be processed by scripts in MegHair/utils/debug.
    Note:
        For some backends (like opencl), special options need to be enabled for
        profiling device time, which may cause additional overhead and make it
        hard to profile host time. Use --profile-host to focus on host time
        profiling.
  --io-dump <output> | --bin-io-dump <output dir>
    Dump input/output values of all internal variables to output file or
    directory, in text or binary format. The binary file can be parsed by
    `megbrain.plugin.load_tensor_binary`.
  --bin-out-dump <output dir>
    Dump output tensor values in binary format to given directory.
  --iter <num>
    Number of iterations to run for each testcase.
  --warmup-iter <num>
    Number of warm-up iterations, which are not included in the time statistics.
  --range <value>
    Enable tensor value range check. Exception would be raised if the absolute
    value of any element of any variable does not fit in given range. This can
    be used to debug NaN values.
  --check-dispatch
    Enable CPU dispatch checker, which prints a warning message if on operator
    does not the dispatch function. This is used to find potential bugs in
    MegDNN.
  --check-var-value <switch_interval[:start_i dx]>
    Enable VarValueChecker plugin. Refer to its doc for more details.
  --no-sanity-check
    Disable var sanity check on the first run. Var sanity check is enabled on
    the first-time execution by default, and can be used to find some potential
    memory access errors in the operator implementation.
  --disable-mem-opt
    Disable memory optimizations. This is used to check whether memory
    optimization is the cause for unexpected behavior.
  --fake-first
    Enable fake exec for the first run. In fake exec mode, some initialization
    job would be done, but no actual computing is performed. This can be used in
    an SDK right after loading the model to reduce execution latency in the real
    fist-time computing. It requires input shapes to be correctly setup.
  --const-shape
    Set `GraphLoadConfig::const_var_shape` to true before loading the graph.
    This can be used to reduce memory usage since some static inference data
    structures can be omitted.
  --share-param-mem
    Share the memory used by model params with model storage. This can be used
    to reduce memory usage when computing on CPU.
  --record-comp-seq | --record-comp-seq2
    Record the computing sequence, in level 1 or 2. It reduces overhead of API
    calls of some asynchronous computing devices, especially for OpenCL. In
    level 2 the computing graph can be destructed to reduce memory usage. Read
    the doc of `ComputingGraph::Options::comp_node_seq_record_level` for more
    details.
)__usage__"
#if MGB_ENABLE_FASTRUN
R"__usage__(
  --fast-run
    Enable fast-run mode. Operators with multiple algorithms would be profiled
    on the real device with actual input shapes.
    See `mgb::gopt::enable_opr_algo_profiling_inplace` for more details.
)__usage__"
#endif
R"__usage__(
  --fast-run-algo-policy <path>
    It will read the cache file before profile, and save new fastrun in cache file.
  --wait-gdb
    Print PID and wait for a line from stdin before starting execution. Useful
    for waiting for gdb attach.
  --c-opr-lib <path>
    Load external operator library. It must implement `mgb_c_opr_init` as the
    entry point.
  --thread <num>
    Number of threads to run concurrently. All threads perform the same work of
    loading and executing models. This is used for test thread safety, not for
    speed up on multiple cores.
  --disable-assert-throw
    Do not throw exception in case AssertEqual fails. Note that the exit code
    would also be zero if this option is enabled. This should only be used for
    debug.
  --copy-to-host
    Whether copy output from device to host.
    This is used for checking the performance in real scenarios including output copy.
  --workspace-limit <num>
    set workspace_limit for execution strategy for oprs with multiple algorithms.
    The default is SIZE_MAX(bytes).
  --verbose
    Increase verbosity for megbrain log.
)__usage__"
#if MGB_ENABLE_TENSOR_RT
R"__usage__(
  --tensorrt
    Execute supported operators with TensorRT. Can only be used on Nvidia GPUs,
    i.e. comp node is xpu or gpu.
  --tensorrt-cache <path>
    Set the TensorRT engine cache path for serialized prebuilt ICudaEngine
)__usage__"
#endif
R"__usage__(
  --enable-jit
    Execute supported operators with JIT(now only support NVRTC). Can only be used on Nvidia GPUs.
)__usage__"
R"__usage__(
  --winograd-transform
    Execute opr replace, replace weights by winograd transform. Currently support on conv bias opr.
)__usage__"
R"__usage__(
  --enable-chwn4
    Execute operators with kernels implemented in MegDNN with CHWN4 tensor format. Can only be used
    on Nvidia GPUs, whose compute capability is above 6.1.
)__usage__"

;

struct Args {
    int args_parse_ret = 0;

    std::string model_path;

    bool disable_assert_throw = false;
    bool share_param_mem = false;
#if MGB_ENABLE_FASTRUN
    bool use_fast_run = false;
#endif
    std::string fast_run_cache_path;
    bool copy_to_host = false;
    int nr_run = 10;
    int nr_warmup = 1;
    int nr_thread = 1;
    int multithread_number = 1;
    size_t workspace_limit = SIZE_MAX;
    serialization::GraphLoader::LoadResult load_ret;
#if MGB_ENABLE_JSON
    std::unique_ptr<GraphProfiler> profiler;
#endif
    std::string profiler_output;
    std::string bin_out_dump;

    std::unique_ptr<OprIODumpBase> iodump;
    std::unique_ptr<NumRangeChecker> num_range_checker;
    std::unique_ptr<CPUDispatchChecker> cpu_dispatch_checker;
    std::unique_ptr<VarValueChecker> var_value_checker;
    serialization::GraphLoader::LoadConfig load_config;
    thin_function<void(size_t)> affinity_cb;

    static Args from_argv(int argc, char **argv);
};

uint32_t read_nr_test(serialization::InputFile &fin) {
    char magic[8];
    fin.read(magic, sizeof(magic));
    if (strncmp(magic, "mgbtest0", 8)) {
        fin.rewind();
        return 0;
    }
    uint32_t ret;
    fin.read(&ret, sizeof(ret));
    return ret;
}

size_t get_file_size(FILE *fptr) {
    fseek(fptr, 0, SEEK_END);
    size_t size = ftell(fptr);
    fseek(fptr, 0, SEEK_SET);
    return size;
}

/**
 * \brief dump output tensor.
 *
 * graph would be destructed if comp_node_seq_record_level == 2; so we should
 * store graph info before graph_compile().
 */
class OutputDumper {
    struct DumpInfo {
        HostTensorND hv = {};
        std::string var_info;
        std::string owner_inputs_info;
        size_t id;
    };
    SmallVector<DumpInfo> m_infos;

    size_t m_run_id = 0;
    size_t m_bind_id = 0;
    const Args& m_env;

public:
    OutputDumper(const Args& env) : m_env{env} {
        for (auto&& i : m_env.load_ret.output_var_list) {
            auto&& var = i.node();
            DumpInfo info;
            info.var_info = cg::dump_var_info({var});
            info.owner_inputs_info =
                    cg::dump_var_info(var->owner_opr()->input());
            info.id = var->id();
            m_infos.push_back(info);
        }
    }

    ComputingGraph::Callback bind() {
        auto& info = m_infos.at(m_bind_id++);
        ComputingGraph::Callback cb = [&info](const DeviceTensorND& dv) {
            info.hv.copy_from(dv);
        };
        return cb;
    }

    void write_to_file() {
        if (!m_env.bin_out_dump.empty()) {
            for (auto&& info : m_infos) {
                auto value = debug::dump_tensor(
                        info.hv, ssprintf("var=%s owner_opr_inputs=%s",
                                          info.var_info.c_str(),
                                          info.owner_inputs_info.c_str()));
                debug::write_to_file(
                        ssprintf("%s/run%zu-var%zd", m_env.bin_out_dump.c_str(),
                                 m_run_id, info.id)
                                .c_str(),
                        value);
            }

        }
        m_run_id ++;
    }
};

void run_test_st(Args &env) {
    std::unique_ptr<serialization::InputFile> inp_file;

    if (env.share_param_mem) {
        FILE *fin = fopen(env.model_path.c_str(), "rb");
        mgb_assert(fin, "failed to open %s: %s", env.model_path.c_str(),
                strerror(errno));
        auto size = get_file_size(fin);
        void *ptr = malloc(size);
        std::shared_ptr<void> buf{ptr, free};
        auto nr = fread(buf.get(), 1, size, fin);
        mgb_assert(nr == size);
        fclose(fin);
        inp_file = serialization::InputFile::make_mem_proxy(buf, size);
    } else {
        inp_file = serialization::InputFile::make_fs(
                env.model_path.c_str());
    }
    auto nr_test = read_nr_test(*inp_file);

    auto format =
            serialization::GraphLoader::identify_graph_dump_format(*inp_file);
    if (!format.valid()) {
        printf("invalid model: unknown model format, please make sure input "
               "file is generated by GraphDumper\n");
        return;
    }
    auto loader =
            serialization::GraphLoader::make(std::move(inp_file), format.val());
    RealTimer timer;
    env.load_ret = loader->load(env.load_config, false);

    // graph is no longer needed; reset so memory can be reclaimed
    env.load_config.comp_graph.reset();

    printf("load model: %.3fms\n", timer.get_msecs_reset());

    // compile function to compute all outputs
    ComputingGraph::OutputSpec out_spec;
    std::string output_names;

    OutputDumper output_dumper(env);
    for (auto&& i : env.load_ret.output_var_list) {
        if (&i != env.load_ret.output_var_list.data()) {
            output_names += " ";
        }
        output_names.append(i.node()->name() + i.shape().to_string());
        ComputingGraph::Callback cb;
        if (!env.bin_out_dump.empty()) {
            cb = output_dumper.bind();
        } else if (env.copy_to_host) {
            HostTensorND val;
            cb = [val](const DeviceTensorND& dv) mutable {
                val.copy_from(dv);
            };
        }
        out_spec.emplace_back(i, std::move(cb));
    }

    if (env.disable_assert_throw) {
        auto on_opr = [](cg::OperatorNodeBase* opr) {
            if (opr->same_type<opr::AssertEqual>()) {
                opr->cast_final<opr::AssertEqual>().disable_throw_on_error();
            }
        };
        cg::DepOprIter iter{on_opr};
        for (auto&& i : out_spec) {
            iter.add(i.first.node()->owner_opr());
        }
    }

    SymbolVarArray vars;
    for (auto i : out_spec) {
        vars.push_back(i.first);
    }

    mgb::gopt::set_opr_algo_workspace_limit_inplace(vars, env.workspace_limit);
#if MGB_ENABLE_FASTRUN
    if (env.use_fast_run)
        mgb::gopt::enable_opr_algo_profiling_inplace(vars);
#endif
    if (!env.fast_run_cache_path.empty()) {
#if MGB_ENABLE_FASTRUN
        if (!access(env.fast_run_cache_path.c_str(), F_OK)) {
#else
        mgb_assert(access(env.fast_run_cache_path.c_str(), F_OK) == 0,
                   "fast-run cache file can't be accessed");
#endif
            FILE* fin = fopen(env.fast_run_cache_path.c_str(), "rb");
            auto flen = get_file_size(fin);
            std::unique_ptr<uint8_t[]> buf{new uint8_t[flen]};
            size_t ret = fread(buf.get(), flen, 1, fin);
            MGB_MARK_USED_VAR(ret);
            mgb_assert(ret == 1, "read 1 block (got %zu), and block size %zu.",
                       ret, flen);
            fclose(fin);
            PersistentCache::set_impl(
                    std::make_shared<InFilePersistentCache>(buf.get(), flen));
#if MGB_ENABLE_FASTRUN
        } else {
            mgb_assert(env.use_fast_run, "fast-run should be enabled");
            PersistentCache::set_impl(
                    std::make_shared<InFilePersistentCache>());
        }
        if (!env.use_fast_run)
#endif
            mgb::gopt::enable_opr_use_profiling_cache_inplace(vars);
    }

    auto func = env.load_ret.graph_compile(out_spec);
    auto warmup = [&]() {
        printf("=== prepare: %.3fms; going to warmup\n",
               timer.get_msecs_reset());
        for (int run = 0; run < env.nr_warmup; ++run) {
            func->execute().wait();
            printf("warmup %d: %.3fms\n", run, timer.get_msecs_reset());
        }
    };

    if (nr_test) {
        // run testcase, generated by dump_with_testcase.py

        std::vector<std::pair<std::string, HostTensorND*>> inp_tensors;
        for (auto &&i: env.load_ret.tensor_map) {
            inp_tensors.emplace_back(i.first, i.second.get());
        }
        std::sort(inp_tensors.begin(), inp_tensors.end());

        printf("=== going to run %u testcases; output vars: %s\n", nr_test,
                output_names.c_str());
        double tot_time = 0;
        for (uint32_t i = 0; i < nr_test; ++ i) {
            loader = serialization::GraphLoader::make(
                    loader->reset_file(), loader->format());
            auto testcase = loader->load(env.load_config, false);
            mgb_assert(testcase.output_var_list.size() == inp_tensors.size());
            for (size_t i = 0; i < inp_tensors.size(); ++ i) {
                auto &&opr = testcase.output_var_list[i].node()->owner_opr()->
                    cast_final_safe<opr::SharedDeviceTensor>();
                inp_tensors[i].second->copy_from(
                        HostTensorND::make_proxy(*opr.dev_data()));
            }

            if (!i) {
                warmup();
            }

            timer.reset();
            printf("=== going to run test #%u for %d times\n", i, env.nr_run);
            if (!env.nr_run) {
                continue;
            }
            double time_sqrsum = 0, time_sum = 0,
                   min_time = std::numeric_limits<double>::max(), max_time = 0;
            for (int run = 0; run < env.nr_run; ++ run) {
                mgb_log_debug("load_and_run: before running iter %d", run);
                timer.reset();
                func->execute();
                mgb_log_debug("load_and_run: before waiting iter %d", run);
                auto exec_time = timer.get_msecs();
                func->wait();
                output_dumper.write_to_file();
                auto cur = timer.get_msecs();
                printf("iter %d/%d: %.3fms (exec=%.3f,device=%.3f)\n", run,
                       env.nr_run, cur, exec_time,
                       func->get_prev_exec_time() * 1e3);
                time_sum += cur;
                time_sqrsum += cur * cur;
                fflush(stdout);
                if (cur < min_time) {
                    min_time = cur;
                }
                if (cur > max_time) {
                    max_time = cur;
                }
            }
            tot_time += time_sum;
            printf("=== finished test #%u: time=%.3fms avg_time=%.3fms "
                   "sd=%.3fms minmax=%.3f,%.3f\n\n",
                   i, time_sum, time_sum / env.nr_run,
                   std::sqrt((time_sqrsum * env.nr_run - time_sum * time_sum) /
                             (env.nr_run * (env.nr_run - 1))),
                   min_time, max_time);
        }

        printf("=== total time: %.3fms\n", tot_time);
    } else {
        // run speed test for a raw mgb graph
        mgb_assert(env.load_ret.tensor_map.empty(),
                "model should not require input values; input vars should be "
                "replaced by SharedDeviceTensor "
                "(i.e. megskull.opr.ParamProvider)");

        warmup();
        timer.reset();
        printf("=== going to run for %d times; output vars: %s\n",
                env.nr_run, output_names.c_str());
        for (int i = 0; i < env.nr_run; ++ i) {
            mgb_log_debug("load_and_run: before benchmark iter %d", i);
            auto start = timer.get_msecs();
            func->execute().wait();
            output_dumper.write_to_file();
            printf("=== finished run #%d: time=%.3fms\n", i,
                    timer.get_msecs() - start);
            fflush(stdout);
        }
        printf("avg time: %.3fms\n", timer.get_msecs() / env.nr_run);
    }

#if MGB_ENABLE_JSON
    if (env.profiler) {
        env.profiler->to_json_full(func.get())->writeto_fpath(
                env.profiler_output);
        mgb_log("profiling result written to %s", env.profiler_output.c_str());
    }
#endif
#if MGB_ENABLE_FASTRUN
    if (!env.fast_run_cache_path.empty()) {
        static_cast<InFilePersistentCache&>(PersistentCache::inst())
                .dump_cache(env.fast_run_cache_path.c_str());
    }
#endif
#if MGB_ENABLE_TENSOR_RT
    if (TensorRTEngineCache::enable_engine_cache()) {
        TensorRTEngineCache::inst().dump_cache();
    }
#endif
}

}  // anonymous namespace

int mgb_load_and_run_main(int argc, char** argv) {
    {
        auto v0 = get_version();
        auto v1 = megdnn::get_version();
        printf("mgb load-and-run: using MegBrain "
               "%d.%d.%d(%d) and MegDNN %d.%d.%d\n",
               v0.major, v0.minor, v0.patch, v0.is_dev, v1.major, v1.minor,
               v1.patch);
    }
    auto env = Args::from_argv(argc, argv);

    if (env.args_parse_ret != 0) {
        return env.args_parse_ret;
    }

    if (env.nr_thread == 1) {
        run_test_st(env);
    } else {
#if MGB_HAVE_THREAD
        mgb_log_warn("use %d threads", env.nr_thread);
        std::vector<std::thread> threads;
        auto run = [argc, argv]() {
            auto env = Args::from_argv(argc, argv);
            run_test_st(env);
        };

        for (int i = 0; i < env.nr_thread; ++i) {
            threads.emplace_back(run);
        }

        for (auto&& i : threads) {
            i.join();
        }
#else
        mgb_log_error("%d threads requested, but load-and-run was compiled "
                      "without thread support.");
#endif
    }

    return 0;
}

Args Args::from_argv(int argc, char **argv) {
    Args ret;
    if (argc < 2) {
        printf("usage: %s <model file> [options...]\nWhere options are:%s",
               argv[0], OPTIONS_DESC);
        ret.args_parse_ret = -1;
        return ret;
    }
    set_log_level(LogLevel::WARN);
    ret.model_path = argv[1];
    ret.load_config.comp_graph = ComputingGraph::make();
    auto &&graph_opt = ret.load_config.comp_graph->options();
    graph_opt.graph_opt_level = 0;

    for (int i = 2; i < argc; ++ i) {
        if (!strcmp(argv[i], "--cpu")) {
            mgb_log_warn("use cpu mode");
            ret.load_config.comp_node_mapper = [](CompNode::Locator &loc) {
                loc.type = CompNode::DeviceType::CPU;
            };
            continue;
        }
        if (!strcmp(argv[i], "--cpu-default")) {
            mgb_log_warn("use cpu:default mode");
            ret.load_config.comp_node_mapper = [](CompNode::Locator &loc) {
                loc.type = CompNode::DeviceType::CPU;
                loc.device = CompNode::Locator::DEVICE_CPU_DEFAULT;
            };
            continue;
        }
        if (!strcmp(argv[i], "--multithread")) {
            mgb_log_warn("use multithread mode");
            ++ i;
            ret.multithread_number = std::stoi(argv[i]);
            ret.load_config.comp_node_mapper =
                    [nr_thread =
                             ret.multithread_number](CompNode::Locator& loc) {
                        loc.type = CompNode::DeviceType::MULTITHREAD;
                        loc.device = 0;
                        loc.stream = nr_thread;
                    };
            continue;
        }
        if (!strcmp(argv[i], "--multithread-default")) {
            mgb_log_warn("use multithread:default mode");
            ++i;
            ret.multithread_number = std::stoi(argv[i]);
            ret.load_config.comp_node_mapper = [nr_thread =
                             ret.multithread_number](CompNode::Locator& loc) {
                loc.type = CompNode::DeviceType::MULTITHREAD;
                loc.device = CompNode::Locator::DEVICE_MULTITHREAD_DEFAULT;
                loc.stream = nr_thread;
            };
            continue;
        }
        if (!strcmp(argv[i], "--multi-thread-core-ids")) {
            ++i;
            std::string core_id_string = argv[i];
            std::stringstream input_stringstream(core_id_string);
            std::string id;
            size_t nr_threads = 0;
            std::vector<int> core_ids;
            mgb_log_warn("multi thread core ids: %s", core_id_string.c_str());
            while(getline(input_stringstream, id, ',')) {
                nr_threads++;
                core_ids.push_back(atoi(id.c_str()));
            }
            mgb_assert(ret.multithread_number > 0 &&
                               ret.load_config.comp_node_mapper,
                       "the core id should set behind the --multithread param");
            mgb_assert(static_cast<size_t>(ret.multithread_number) ==
                               core_ids.size(),
                       "the core id should equal to the multi thread number");
            auto affinity_cb = [core_ids](int thread_id) {
                mgb::sys::set_cpu_affinity({core_ids[thread_id]});
            };
            CompNode::Locator loc;
            ret.load_config.comp_node_mapper(loc);
            mgb_assert(loc.type == CompNode::DeviceType::MULTITHREAD,
                       "core id only set on multithread compnode");
            auto cn = CompNode::load(loc);
            CompNodeEnv::from_comp_node(cn).cpu_env().set_affinity(affinity_cb);
            continue;
        }
#if MGB_ENABLE_TENSOR_RT
        if (!strcmp(argv[i], "--tensorrt")) {
            mgb_log_warn("use tensorrt mode");
            graph_opt.graph_opt.tensorrt = true;
            continue;
        }
        if (!strcmp(argv[i], "--tensorrt-cache")) {
            ++i;
            mgb_assert(i < argc, "value not given for --tensorrt-cache");
            char* tensorrt_cache_path = argv[i];
            mgb_log_warn("use tensorrt cache: %s", tensorrt_cache_path);
            TensorRTEngineCache::enable_engine_cache(true);
            TensorRTEngineCache::set_impl(
                    std::make_shared<TensorRTEngineCacheIO>(
                            tensorrt_cache_path));
            continue;
        }
#endif
        if (!strcmp(argv[i], "--enable-chwn4")) {
            mgb_log_warn("enable chwn4 optimization");
            graph_opt.graph_opt.enable_chwn4 = true;
            continue;
        }
#if MGB_ENABLE_JSON
        if (!strcmp(argv[i], "--profile") ||
            !strcmp(argv[i], "--profile-host")) {
            if (!strcmp(argv[i], "--profile")) {
                mgb_log_warn("enable profiling");
            } else {
                mgb_log_warn("enable profiling for host");
            }
            ++i;
            mgb_assert(i < argc, "output file not given for --profile");
            ret.profiler = std::make_unique<GraphProfiler>(
                    ret.load_config.comp_graph.get());
            ret.profiler_output = argv[i];
            continue;
        }
#endif
        if (!strcmp(argv[i], "--io-dump")) {
            mgb_log_warn("enable opr io dump");
            ++ i;
            mgb_assert(i < argc, "output file not given for --io-dump");
            auto iodump = std::make_unique<TextOprIODump>(
                    ret.load_config.comp_graph.get(), argv[i]);
            iodump->print_addr(false);
            ret.iodump = std::move(iodump);
            continue;
        }
        if (!strcmp(argv[i], "--bin-io-dump")) {
            mgb_log_warn("enable opr binary io dump");
            ++ i;
            mgb_assert(i < argc,
                    "output directory not given for --bin-io-dump");
            ret.iodump = std::make_unique<BinaryOprIODump>(
                    ret.load_config.comp_graph.get(), argv[i]);
            continue;
        }
        if (!strcmp(argv[i], "--bin-out-dump")) {
            ++i;
            mgb_assert(i < argc,
                    "output directory not given for --bin-out-dump");
            ret.bin_out_dump = argv[i];
            continue;
        }
        if (!strcmp(argv[i], "--iter")) {
            ++ i;
            mgb_assert(i < argc, "value not given for --iter");
            ret.nr_run = std::stoi(argv[i]);
            mgb_assert(ret.nr_run >= 0);
            continue;
        }
        if (!strcmp(argv[i], "--warmup-iter")) {
            ++ i;
            mgb_assert(i < argc, "value not given for --warmup-iter");
            ret.nr_warmup = std::stoi(argv[i]);
            mgb_assert(ret.nr_warmup >= 0);
            continue;
        }
        if (!strcmp(argv[i], "--range")) {
            ++ i;
            mgb_assert(i < argc, "value not given for --range");
            auto range = std::atof(argv[i]);
            mgb_assert(range > 0);
            ret.num_range_checker = std::make_unique<NumRangeChecker>(
                    ret.load_config.comp_graph.get(), range);
            continue;
        }
        if (!strcmp(argv[i], "--check-dispatch")) {
            ret.cpu_dispatch_checker =
                std::make_unique<CPUDispatchChecker>(
                        ret.load_config.comp_graph.get());
            continue;
        }
        if (!strcmp(argv[i], "--disable-mem-opt")) {
            graph_opt.seq_opt.enable_mem_reuse_alloc = false;
            graph_opt.seq_opt.enable_mem_plan_opt = false;
            continue;
        }
        if (!strcmp(argv[i], "--copy-to-host")) {
            ret.copy_to_host = true;
            continue;
        }
        if (!strcmp(argv[i], "--verbose")) {
            graph_opt.log_level = 2;
            set_log_level(LogLevel::DEBUG);
            continue;
        }
        if (!strcmp(argv[i], "--check-var-value")) {
            ++ i;
            mgb_assert(i < argc, "value not given for --check-var-value");
            std::string arg(argv[i]);
            auto sep = arg.find(':');
            size_t switch_interval, start = 0;
            if (sep != std::string::npos) {
                switch_interval = std::stoul(arg.substr(0, sep));
                start = std::stoul(arg.substr(sep + 1));
            } else {
                switch_interval = std::stoul(arg);
            }
            ret.var_value_checker = std::make_unique<VarValueChecker>(
                    ret.load_config.comp_graph.get(), switch_interval, start);
            continue;
        }
        if (!strcmp(argv[i], "--no-sanity-check")) {
            graph_opt.var_sanity_check_first_run = false;
            continue;
        }
        if (!strcmp(argv[i], "--fake-first")) {
            graph_opt.fake_next_exec = true;
            continue;
        }
        if (!strcmp(argv[i], "--record-comp-seq")) {
            graph_opt.comp_node_seq_record_level = 1;
            continue;
        }
        if (!strcmp(argv[i], "--record-comp-seq2")) {
            graph_opt.comp_node_seq_record_level = 2;
            continue;
        }
#if MGB_ENABLE_FASTRUN
        if (!strcmp(argv[i], "--fast-run")) {
            ret.use_fast_run = true;
            continue;
        }
#endif
        if (!strcmp(argv[i], "--fast-run-algo-policy")) {
            ++i;
            ret.fast_run_cache_path = argv[i];
            continue;
        }
        if (!strcmp(argv[i], "--const-shape")) {
            ret.load_config.const_var_shape = true;
            continue;
        }
        if (!strcmp(argv[i], "--share-param-mem")) {
            ret.share_param_mem = true;
            continue;
        }
        if (!strcmp(argv[i], "--disable-assert-throw")) {
            ret.disable_assert_throw = true;
            continue;
        }
        if (!strcmp(argv[i], "--workspace-limit")) {
            ++i;
            ret.workspace_limit = std::stoll(argv[i]);
            continue;
        }
#if __linux__ || __unix__
        if (!strcmp(argv[i], "--wait-gdb")) {
            printf("wait for gdb attach (pid=%d): ", getpid());
            getchar();
            continue;
        }
        if (!strcmp(argv[i], "--c-opr-lib")) {
            ++ i;
            mgb_assert(i < argc, "value not given for --c-opr-lib");
            auto handle = dlopen(argv[i], RTLD_LAZY);
            mgb_assert(handle, "failed to open c opr lib %s: %s",
                    argv[i], dlerror());
            const char* entry = "mgb_c_opr_init";
            auto func = dlsym(handle, entry);
            mgb_assert(func, "can not resolve %s: %s", entry, dlerror());
            typedef void (*entry_f_t)(void*);
            reinterpret_cast<entry_f_t>(func)(
                    reinterpret_cast<void*>(
                        &mgb_get_extern_c_opr_api_versioned));
            printf("loaded C opr library: %s\n", argv[i]);
            continue;
        }
#endif
        if (!strcmp(argv[i], "--thread")) {
            ++ i;
            mgb_assert(i < argc, "value not given for --thread");
            ret.nr_thread = std::stoi(argv[i]);
            continue;
        }
        if (!strcmp(argv[i], "--enable-jit")) {
            graph_opt.graph_opt.jit = 1;
            continue;
        }
        if (!strcmp(argv[i], "--winograd-transform")) {
            mgb_log_warn("enable winograd transform");
            graph_opt.graph_opt.winograd_transform = true;
            continue;
        }

        fprintf(stderr, "invalid arg: %s\n", argv[i]);
        ret.args_parse_ret = -1;
        return ret;
    }

    return ret;
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
