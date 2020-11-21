/**
 * \file src/gopt/impl/fuse_nchw4_int8_preprocess.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain/gopt/inference.h"
#include "megbrain/gopt/misc.h"
#include "megbrain/graph/grad_impl.h"
#include "megbrain/opr/cond.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"
#include "megbrain/serialization/opr_shallow_copy.h"
#include "megbrain/serialization/serializer.h"
#include "megbrain/opr/imgproc.h"

using namespace mgb;
using namespace gopt;
namespace {
#define RETURN_IF_FALSE(ok) \
    {                       \
        if (!ok)            \
            return ok;      \
    }

struct SubGraphMatcher {
    struct Node {
        using CallBack = std::function<bool(OperatorNodeBase* opr)>;
        Node(Typeinfo* in_op_type) : op_type(in_op_type){};
        Node(Typeinfo* in_op_type, CallBack func)
                : op_type(in_op_type), cbk(func){};
        Node(Typeinfo* in_op_type, std::vector<Node> in_pre_node)
                : op_type(in_op_type), pre_node(in_pre_node){};
        Node(Typeinfo* in_op_type, std::vector<Node> in_pre_node, CallBack func)
                : op_type(in_op_type), pre_node(in_pre_node), cbk(func){};

        Typeinfo* op_type{nullptr};
        std::vector<Node> pre_node;
        //! cbk used to check param and gather args for creating fusion op
        CallBack cbk;
    };

    bool match(Node& root, OperatorNodeBase* opr) {
        if (opr == nullptr) {
            return false;
        }
        //! match nullptr node always
        if (root.op_type == nullptr || root.op_type == opr->dyn_typeinfo()) {
            bool match_ok = true;
            if (root.cbk)
                match_ok &= root.cbk(opr);
            RETURN_IF_FALSE(match_ok);
            auto& inp = opr->input();
            for (size_t node_idx = 0; node_idx < root.pre_node.size();
                 ++node_idx) {
                bool valid_node_idx = node_idx < inp.size();
                RETURN_IF_FALSE(valid_node_idx);
                match_ok &= match(root.pre_node[node_idx],
                                  inp[node_idx]->owner_opr());
                RETURN_IF_FALSE(match_ok);
            }
            return match_ok;
        } else {
            return false;
        }
    }
};
#undef RETURN_IF_FALSE

struct SubGraphChecker {
    using DepType = cg::OperatorNodeProp::DepType;
    using ReaderType =
            ThinHashMap<OperatorNodeBase*,
                        SmallVector<std::pair<OperatorNodeBase*, DepType>>>;
    SubGraphChecker() {}

    bool check(ThinHashSet<OperatorNodeBase*> used_input,
               OperatorNodeBase* start_opr, OperatorNodeBase* stop_opr,
               ReaderType& readers, bool ignore_immutable = true) {
        bool is_all_inp_used = check_all_inp_used(used_input, start_opr,
                                                  stop_opr, ignore_immutable);
        bool is_all_dep_inside =
                check_all_dep_inside_node(start_opr, stop_opr, readers);
        return is_all_inp_used && is_all_dep_inside;
    }

    bool check_all_inp_used(ThinHashSet<OperatorNodeBase*>& used_input,
                            OperatorNodeBase* start_opr,
                            OperatorNodeBase* stop_opr,
                            bool ignore_immutable = true) {
        ThinHashSet<OperatorNodeBase*> leaf_set;
        get_leaf_node(start_opr, stop_opr, leaf_set);
        for (auto in_opr : leaf_set) {
            bool skip = in_opr->same_type<opr::ImmutableTensor>() &&
                        ignore_immutable;
            if (used_input.find(in_opr) == used_input.end() && !skip) {
                return false;
            }
        }
        return true;
    }

    bool check_all_dep_inside_node(OperatorNodeBase* start_opr,
                                   OperatorNodeBase* stop_opr,
                                   ReaderType& readers) {
        ThinHashSet<OperatorNodeBase*> mid_set;
        get_mid_node(start_opr, start_opr, stop_opr, mid_set);
        for (auto inner_opr : mid_set) {
            if (readers.find(inner_opr) != readers.end()) {
                for (auto& out_node : readers[inner_opr]) {
                    if (mid_set.find(out_node.first) == mid_set.end() &&
                        out_node.first != start_opr &&
                        out_node.second ==
                                cg::OperatorNodeProp::DepType::DEV_VALUE) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    void get_mid_node(OperatorNodeBase* opr, OperatorNodeBase* start_opr,
                      OperatorNodeBase* stop_opr,
                      ThinHashSet<OperatorNodeBase*>& mid_set) {
        if (opr == nullptr) {
            return;
        }
        if (opr != start_opr) {
            mid_set.insert(opr);
        }
        if (opr == stop_opr) {
            return;
        }
        for (auto& tensor : opr->input()) {
            auto pre_opr = tensor->owner_opr();
            get_mid_node(pre_opr, start_opr, stop_opr, mid_set);
        }
    }

    void get_leaf_node(OperatorNodeBase* opr, OperatorNodeBase* stop_opr,
                       ThinHashSet<OperatorNodeBase*>& leaf_set) {
        if (opr == nullptr) {
            return;
        }
        if (opr == stop_opr || opr->input().size() == 0) {
            leaf_set.insert(opr);
        }
        if (opr == stop_opr) {
            return;
        }
        for (auto& tensor : opr->input()) {
            auto pre_opr = tensor->owner_opr();
            get_leaf_node(pre_opr, stop_opr, leaf_set);
        }
    }
};

static inline bool is_shape_nchw(const TensorShape& shape) {
    return shape.ndim == 4;
}

static inline bool is_shape_before_nchw4(const TensorShape& shape) {
    return shape.ndim == 5 && shape[2] == 4;
}

static inline bool is_nchw_nchw4_shuffle_vec(
        const opr::Dimshuffle::Param param) {
    return param.ndim == 5 && param.pattern[0] == 0 && param.pattern[1] == 1 &&
           param.pattern[2] == 3 && param.pattern[3] == 4 &&
           param.pattern[4] == 2;
}

template <typename T>
static inline bool is_immutable_equal(OperatorNodeBase* opr, T val,
                                      DTypeEnum dtype_enum) {
    auto const_opr = opr->try_cast_final<opr::ImmutableTensor>();
    if (!const_opr) {
        return false;
    }
    auto& host_value = const_opr->host_value();
    bool ok_value = host_value.layout().total_nr_elems() == 1 &&
                    host_value.dtype().enumv() == dtype_enum &&
                    host_value.ptr<T>()[0] == val;
    return ok_value;
}

template <typename T>
static inline bool is_immutable_all_equal(OperatorNodeBase* opr,
                                          typename DTypeTrait<T>::ctype val) {
    auto const_opr = opr->try_cast_final<opr::ImmutableTensor>();
    if (!const_opr) {
        return false;
    }
    auto& host_value = const_opr->host_value();
    bool ok_value = host_value.dtype().enumv() == DTypeTrait<T>::enumv;
    if (!ok_value) {
        return false;
    }
    size_t nr_elem = host_value.layout().total_nr_elems();
    for (size_t i = 0; i < nr_elem; ++i) {
        if (host_value.ptr<typename DTypeTrait<T>::ctype>()[i] != val) {
            ok_value = false;
            break;
        }
    }
    return ok_value;
}

}  // namespace

const char* FuseNCHW4Int8Preprocess::name() const {
    return "fuse_pre_process_pass";
}

std::unique_ptr<FuseNCHW4Int8Preprocess> FuseNCHW4Int8Preprocess::make() {
    using SGM = SubGraphMatcher;
    auto gen_pad_dimshuffle_graph = [&](SGM::Node& in_node,
                                        SGM::Node::CallBack& pad_cbk,
                                        SGM::Node::CallBack& shape_cbk) {
        SGM::Node::CallBack check_pad = [&](OperatorNodeBase* opr) {
            SGM sub_matcher;
            SGM::Node immu_node{opr::ImmutableTensor::typeinfo(), pad_cbk};
            if (opr->same_type<opr::ImmutableTensor>()) {
                return sub_matcher.match(immu_node, opr);
            } else if (opr->same_type<opr::Broadcast>()) {
                return sub_matcher.match(immu_node,
                                         opr->input()[0]->owner_opr());
            } else {
                return false;
            }
        };
        SGM::Node broadcast_or_immutable{nullptr, check_pad};
        SGM::Node broadcast_concat{
                opr::Concat::typeinfo(),
                {in_node, broadcast_or_immutable},
                [](OperatorNodeBase* opr) {
                    auto concat_pad = opr->try_cast_final<opr::Concat>();
                    return concat_pad->axis() == 1;
                }};

        SGM::Node nchwx_reshape{opr::Reshape::typeinfo(),
                                {broadcast_concat, SGM::Node(nullptr)},
                                [](OperatorNodeBase* opr) {
                                    auto inp0 = opr->input()[0];
                                    return is_shape_nchw(inp0->shape());
                                }};
        SGM::Node shuffle_root{
                opr::Dimshuffle::typeinfo(),
                {nchwx_reshape},
                [](OperatorNodeBase* opr) {
                    auto& shuffle_opr = opr->cast_final<opr::Dimshuffle>();
                    auto& input_vec = shuffle_opr.input();
                    return is_shape_before_nchw4(input_vec[0]->shape()) &&
                           is_nchw_nchw4_shuffle_vec(shuffle_opr.param());
                }};
        return shuffle_root;
    };
    auto replace_shuffle_opr = [&](OperatorNodeBase* opr,
                                   const VarNodeArray& new_inp,
                                   SubGraph::Rewriter& rewriter,
                                   ReaderType& reader) {
        SGM matcher;
        OperatorNodeBase* src_node = nullptr;
        SGM::Node input_data_cp{
                nullptr, [&](OperatorNodeBase* opr) {
                    auto src_dtype = opr->output()[0]->dtype();
                    if (src_dtype.enumv() == DTypeEnum::Quantized8Asymm) {
                        src_node = opr;
                        return true;
                    } else {
                        return false;
                    }
                }};
        SGM::Node type_cvt{opr::TypeCvt::typeinfo(), {input_data_cp}};
        SGM::Node::CallBack const_pad_cbk = [&](OperatorNodeBase* opr) {
            bool is_fp32_pad = is_immutable_all_equal<dtype::Float32>(opr, 0);
            bool is_i32_pad = is_immutable_all_equal<dtype::Int32>(opr, 0);
            bool is_q8_pad = is_immutable_all_equal<dtype::QuantizedS8>(
                    opr, dt_qint8(0));
            return is_fp32_pad || is_i32_pad || is_q8_pad;
        };
        SGM::Node::CallBack const_reshape_cbk = [](OperatorNodeBase* opr) {
            return true;
        };
        auto&& shuffle_root = gen_pad_dimshuffle_graph(type_cvt, const_pad_cbk,
                                                       const_reshape_cbk);
        bool match = matcher.match(shuffle_root, opr);
        bool check_ok = false;
        if (match) {
            check_ok =
                    SubGraphChecker().check({src_node}, opr, src_node, reader);
        }
        if (match && check_ok) {
            opr::RelayoutFormat::Param param;
            param.mode = opr::RelayoutFormat::Param::Mode::NCHW_NCHW4;
            OperatorNodeConfig config(opr->output()[0]->dtype());
            auto out_node = opr::RelayoutFormat::make(
                    rewriter.get_var(src_node->output()[0]), param.mode,
                    config);
            return out_node.node()->owner_opr();
        } else {
            return serialization::copy_opr_shallow(*opr, new_inp,
                                                   opr->config());
        }
    };

    auto replace_astype_opr = [&](OperatorNodeBase* opr,
                                  const VarNodeArray& new_inp,
                                  SubGraph::Rewriter& rewriter,
                                  ReaderType& reader) {
        SGM matcher;
        OperatorNodeBase* src_node = nullptr;
        OperatorNodeBase* neg_128_immu_node = nullptr;
        OperatorNodeBase* pad0_immu_node = nullptr;
        OperatorNodeBase* const_reshape_last_dim_node = nullptr;
        SGM::Node input_data_cp{nullptr, [&](OperatorNodeBase* opr) {
                                    auto src_dtype = opr->output()[0]->dtype();
                                    if (src_dtype.enumv() == DTypeEnum::Uint8) {
                                        src_node = opr;
                                        return true;
                                    } else {
                                        return false;
                                    }
                                }};
        SGM::Node cvt_fp32{opr::TypeCvt::typeinfo(),
                           {input_data_cp},
                           [](OperatorNodeBase* opr) {
                               auto cvt_op =
                                       opr->try_cast_final<opr::TypeCvt>();
                               bool is_fp32 = cvt_op->param().enumv() ==
                                              DTypeEnum::Float32;
                               return is_fp32;
                           }};
        SGM::Node sub_128{
                opr::Elemwise::typeinfo(),
                {cvt_fp32},
                [&](OperatorNodeBase* opr) {
                    auto elem_op = opr->try_cast_final<opr::Elemwise>();
                    bool is_add_op = elem_op->param().mode ==
                                     opr::Elemwise::Param::Mode::ADD;
                    auto neg_128_op = elem_op->input()[1]->owner_opr();
                    bool is_neg_128 = is_immutable_equal(neg_128_op, -128.f,
                                                         DTypeEnum::Float32);
                    neg_128_immu_node = is_neg_128 ? neg_128_op : nullptr;
                    return is_add_op && is_neg_128;
                }};
        SGM::Node::CallBack const_pad_cbk = [&](OperatorNodeBase* opr) {
            pad0_immu_node = opr;
            bool is_fp32_pad = is_immutable_all_equal<dtype::Float32>(opr, 0);
            bool is_i32_pad = is_immutable_all_equal<dtype::Int32>(opr, 0);
            return is_fp32_pad || is_i32_pad;
        };
        SGM::Node::CallBack const_reshape_cbk = [&](OperatorNodeBase* opr) {
            const_reshape_last_dim_node = opr;
            return true;
        };
        auto&& shuffle_root = gen_pad_dimshuffle_graph(sub_128, const_pad_cbk,
                                                       const_reshape_cbk);

        SGM::Node astype_root{opr::TypeCvt::typeinfo(), {shuffle_root}};
        bool match = matcher.match(astype_root, opr);
        bool check_ok = false;
        if (match) {
            check_ok = SubGraphChecker().check(
                    {src_node, neg_128_immu_node, pad0_immu_node,
                     const_reshape_last_dim_node},
                    opr, src_node, reader);
        }
        if (match && check_ok) {
            opr::RelayoutFormat::Param param;
            param.mode = opr::RelayoutFormat::Param::Mode::NCHW_NCHW4;
            OperatorNodeConfig config(opr->output()[0]->dtype());
            auto out_node = opr::RelayoutFormat::make(
                    rewriter.get_var(src_node->output()[0]), param.mode,
                    config);
            return out_node.node()->owner_opr();
        } else {
            return serialization::copy_opr_shallow(*opr, new_inp,
                                                   opr->config());
        }
    };
    auto ret = std::make_unique<FuseNCHW4Int8Preprocess>();
    auto&& replace_func = ret->m_opr_replace_func;

    MGB_MARK_USED_VAR(replace_astype_opr);
    MGB_MARK_USED_VAR(replace_shuffle_opr);
    replace_func[opr::Dimshuffle::typeinfo()] = replace_shuffle_opr;
    replace_func[opr::TypeCvt::typeinfo()] = replace_astype_opr;
    return ret;
}

void FuseNCHW4Int8Preprocess::apply(OptState& state) const {
    state.set_var_replace_check_flag(VarReplaceCheckFlag::CHECK_DTYPE |
                                     VarReplaceCheckFlag::CHECK_SHAPE);
    auto rewriter = state.graph().make_rewriter();
    VarNodeArray new_inp_cache;

    ReaderType readers;
    state.graph().iter([&readers](OperatorNodeBase* opr) {
        for (auto&& i : opr->node_prop().dep_map()) {
            readers[i.first->owner_opr()].emplace_back(opr, i.second);
        }
    });

    auto on_opr = [this, &rewriter, &new_inp_cache,
                   &readers](OperatorNodeBase* opr) {
        auto it = m_opr_replace_func.find(opr->dyn_typeinfo());

        if (it != m_opr_replace_func.end()) {
            auto&& new_inp = new_inp_cache;
            new_inp.clear();
            new_inp.reserve(opr->input().size());
            for (auto i : opr->input()) {
                new_inp.push_back(rewriter.get_var(i));
            }
            auto new_opr = (it->second)(opr, new_inp, rewriter, readers);
            if (new_opr->try_cast_final<opr::RelayoutFormat>()) {
                auto &&origin_out = opr->output(),
                     &&cur_out = new_opr->output();
                rewriter.replace_var(origin_out[0], cur_out[0], nullptr);
            } else {
                auto &&origin_out = opr->output(),
                     &&cur_out = new_opr->output();
                mgb_assert(origin_out.size() == cur_out.size(),
                           "bad opr replace: src=%s{%s} dst=%s{%s}, %zu != %zu",
                           opr->cname(), opr->dyn_typeinfo()->name,
                           new_opr->cname(), new_opr->dyn_typeinfo()->name,
                           origin_out.size(), cur_out.size());
                for (size_t i = 0; i < origin_out.size(); i++) {
                    rewriter.replace_var(origin_out[i], cur_out[i], nullptr);
                }
            }
        } else {
            rewriter.auto_replace_outputs(opr);
        }
    };
    state.graph().iter(on_opr);
    rewriter.apply_inplace();
}

/* ==================== FuseWarpPerspectiveDimshufflePass ================= */
const char* FuseWarpPerspectiveDimshufflePass::name() const {
    return mgb_cstr_log("Fuse warp perspective dimshuffle pass");
}

void FuseWarpPerspectiveDimshufflePass::apply(OptState& opt) const {
    auto rewriter = opt.graph().make_rewriter();
    auto uniq_reader_check = UniqReaderCheck{opt.graph()};

    auto make_new_warp = [&rewriter](opr::WarpPerspective* warp,
                                     opr::WarpPerspective::Param new_param,
                                     megdnn::DType dst_dtype,
                                     SymbolVar& new_warp) {
        OperatorNodeConfig new_config(dst_dtype);
        if (warp->input().size() == 3) {
            auto src = rewriter.get_var(warp->input(0)),
                 mat = rewriter.get_var(warp->input(1)),
                 out_shape = rewriter.get_var(warp->input(2));
            new_warp = opr::WarpPerspective::make(src, mat, out_shape,
                                                  new_param, new_config);
        } else {
            mgb_assert(warp->input().size() == 4);
            auto src = rewriter.get_var(warp->input(0)),
                 mat = rewriter.get_var(warp->input(1)),
                 mat_idx = rewriter.get_var(warp->input(2)),
                 out_shape = rewriter.get_var(warp->input(3));
            new_warp = opr::WarpPerspective::make(src, mat, mat_idx, out_shape,
                                                  new_param, new_config);
        }
    };

    auto is_warp_nchw = [&uniq_reader_check](OperatorNodeBase* bottom_opr,
                                             OperatorNodeBase*& top_opr) {
        // check warp
        auto warp = try_cast_as_op<opr::WarpPerspective>(bottom_opr);
        if (warp == nullptr)
            return false;
        auto inp_dtype = warp->input(0)->dtype();
        bool is_u8_or_qu8 = inp_dtype.enumv() == DTypeEnum::Quantized8Asymm ||
                            inp_dtype.enumv() == DTypeEnum::Uint8;

        bool is_nchw = warp->param().format ==
                       megdnn::param::WarpPerspective::Format::NCHW;
        if (!(is_u8_or_qu8 && is_nchw))
            return false;
        if (!uniq_reader_check(warp->input(0)))
            return false;

        top_opr = warp;
        return true;
    };

    auto is_warp_nhwc2nchw = [&uniq_reader_check](OperatorNodeBase* bottom_opr,
                                                  OperatorNodeBase*& top_opr) {
        // check shuffle
        auto shuffle = try_cast_as_op<opr::Dimshuffle>(bottom_opr);
        if (shuffle == nullptr)
            return false;
        auto&& shuffle_param = shuffle->param();
        if (shuffle_param.pattern_len != 4)
            return false;
        bool is_nhwc2nchw = shuffle_param.pattern[0] == 0 &&
                            shuffle_param.pattern[1] == 3 &&
                            shuffle_param.pattern[2] == 1 &&
                            shuffle_param.pattern[3] == 2;
        if (!is_nhwc2nchw)
            return false;
        if (!uniq_reader_check(shuffle->input(0)))
            return false;

        // check warp
        auto warp = try_cast_as_op<opr::WarpPerspective>(
                shuffle->input(0)->owner_opr());
        if (warp == nullptr)
            return false;
        auto inp_dtype = warp->input(0)->dtype();
        bool is_u8_or_qu8 = inp_dtype.enumv() == DTypeEnum::Quantized8Asymm ||
                            inp_dtype.enumv() == DTypeEnum::Uint8;
        bool is_nhwc = warp->param().format ==
                       megdnn::param::WarpPerspective::Format::NHWC;
        if (!(is_u8_or_qu8 && is_nhwc))
            return false;

        top_opr = warp;
        return true;
    };

    auto try_warp_nchw_typecvt = [&rewriter, &uniq_reader_check, &is_warp_nchw,
                                  &make_new_warp](OperatorNodeBase* opr) {
        // check typecvt
        auto typecvt = try_cast_as_op<opr::TypeCvt>(opr);
        if (typecvt == nullptr)
            return false;
        bool is_to_f32 =
                typecvt->output(0)->dtype().enumv() == DTypeEnum::Float32;
        if (!is_to_f32)
            return false;
        if (!uniq_reader_check(typecvt->input(0)))
            return false;

        OperatorNodeBase* top_opr = nullptr;
        if (!is_warp_nchw(typecvt->input(0)->owner_opr(), top_opr))
            return false;
        auto warp = try_cast_as_op<opr::WarpPerspective>(top_opr);
        SymbolVar new_warp;
        make_new_warp(warp, warp->param(), opr->output()[0]->dtype(), new_warp);

        rewriter.replace_var(opr->output(0), new_warp.node(),
                             mgb_cstr_log("replace warp + typecvt"
                                          "fuse warp_dimshuffle(NCHW)"));

        return true;
    };

    auto try_warp_nhwc2nchw_typecvt = [&rewriter, &uniq_reader_check,
                                       &is_warp_nhwc2nchw,
                                       &make_new_warp](OperatorNodeBase* opr) {
        // check typecvt
        auto typecvt = try_cast_as_op<opr::TypeCvt>(opr);
        if (typecvt == nullptr)
            return false;
        bool is_to_f32 =
                typecvt->output(0)->dtype().enumv() == DTypeEnum::Float32;
        if (!is_to_f32)
            return false;
        if (!uniq_reader_check(typecvt->input(0)))
            return false;

        OperatorNodeBase* top_opr = nullptr;
        if (!is_warp_nhwc2nchw(typecvt->input(0)->owner_opr(), top_opr))
            return false;
        auto warp = try_cast_as_op<opr::WarpPerspective>(top_opr);
        opr::WarpPerspective::Param new_param = warp->param();
        new_param.format = megdnn::param::WarpPerspective::Format::NHWC_NCHW;
        SymbolVar new_warp;
        make_new_warp(warp, new_param, opr->output()[0]->dtype(), new_warp);

        rewriter.replace_var(
                opr->output(0), new_warp.node(),
                mgb_cstr_log("replace conv_bias + dimshuffle + "
                             "typecvt to warp_dimshuffle(NHWC_NCHW)"));

        return true;
    };

    auto try_warp_nhwc2nchw4_typecvt = [&rewriter, &uniq_reader_check,
                                        &is_warp_nhwc2nchw,
                                        &make_new_warp](OperatorNodeBase* opr) {
        // check relayout
        auto relayout = try_cast_as_op<opr::RelayoutFormat>(opr);
        if (relayout == nullptr)
            return false;
        bool is_to_q8 =
                relayout->output(0)->dtype().enumv() == DTypeEnum::QuantizedS8;
        bool is_to_nchw2nchw4 = relayout->param().mode ==
                                opr::RelayoutFormat::Param::Mode::NCHW_NCHW4;
        if (!(is_to_q8 && is_to_nchw2nchw4))
            return false;
        if (!uniq_reader_check(relayout->input(0)))
            return false;

        OperatorNodeBase* top_opr = nullptr;
        if (!is_warp_nhwc2nchw(relayout->input(0)->owner_opr(), top_opr))
            return false;

        auto warp = try_cast_as_op<opr::WarpPerspective>(top_opr);

        bool is_small_chn = warp->input(0)->shape()[3] < 4;
        if (!is_small_chn)
            return false;

        opr::WarpPerspective::Param new_param = warp->param();
        new_param.format =
                megdnn::param::WarpPerspective::Format::NHWC_NCHW4_IC_SMALL;

        SymbolVar new_warp;
        make_new_warp(warp, new_param, opr->output()[0]->dtype(), new_warp);

        rewriter.replace_var(
                opr->output(0), new_warp.node(),
                mgb_cstr_log("replace warp + dimshuffle + relayout(NCHW_NCHW4)"
                             "to warp_dimshuffle(NHWC_NCHW4_IC_SMALL)"));

        return true;
    };

    auto try_warp_nchw2nchw4_typecvt = [&rewriter, &uniq_reader_check,
                                        &is_warp_nchw,
                                        &make_new_warp](OperatorNodeBase* opr) {
        // check relayout
        auto relayout = try_cast_as_op<opr::RelayoutFormat>(opr);
        if (relayout == nullptr)
            return false;
        bool is_to_q8 =
                relayout->output(0)->dtype().enumv() == DTypeEnum::QuantizedS8;
        bool is_to_nchw2nchw4 = relayout->param().mode ==
                                opr::RelayoutFormat::Param::Mode::NCHW_NCHW4;
        if (!(is_to_q8 && is_to_nchw2nchw4))
            return false;
        if (!uniq_reader_check(relayout->input(0)))
            return false;

        OperatorNodeBase* top_opr = nullptr;
        if (!is_warp_nchw(relayout->input(0)->owner_opr(), top_opr))
            return false;

        auto warp = try_cast_as_op<opr::WarpPerspective>(top_opr);

        bool is_small_chn = warp->input(0)->shape()[1] < 4;
        if (!is_small_chn)
            return false;

        opr::WarpPerspective::Param new_param = warp->param();
        new_param.format =
                megdnn::param::WarpPerspective::Format::NCHW_NCHW4_IC_SMALL;

        SymbolVar new_warp;
        make_new_warp(warp, new_param, opr->output()[0]->dtype(), new_warp);

        rewriter.replace_var(
                opr->output(0), new_warp.node(),
                mgb_cstr_log("replace warp + relayout(NCHW_NCHW4)"
                             "to warp_dimshuffle(NCHW_NCHW4_IC_SMALL)"));

        return true;
    };

    auto on_opr = [&try_warp_nchw_typecvt, &try_warp_nhwc2nchw_typecvt,
                   &try_warp_nhwc2nchw4_typecvt, &try_warp_nchw2nchw4_typecvt,
                   &rewriter](OperatorNodeBase* opr) {
        if (!try_warp_nchw_typecvt(opr) && !try_warp_nhwc2nchw_typecvt(opr) &&
            !try_warp_nhwc2nchw4_typecvt(opr) &&
            !try_warp_nchw2nchw4_typecvt(opr)) {
            rewriter.auto_replace_outputs(opr);
        }
    };
    opt.graph().iter(on_opr);
    rewriter.apply_inplace();
}