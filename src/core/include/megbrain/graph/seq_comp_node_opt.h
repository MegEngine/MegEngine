/**
 * \file src/core/include/megbrain/graph/seq_comp_node_opt.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/comp_node.h"

namespace mgb {
namespace cg {

class VarNode;

/*!
 * \brief class that manages optimizing strategies for computing nodes in a
 *      computing sequence
 */
class SeqCompNodeOptimizer {
    protected:
        ~SeqCompNodeOptimizer() = default;

    public:
        //! stream propagation type
        struct StreamPropType {
            enum PropType {
                NONE,   //!< used for stream_prop_type() return value
                WEAK,   //!< move opr to stream if all of its inputs is
                        //!< moved
                STRONG  //!< move opr to stream if any of its inputs are
                        //!< moved
            };
            int stream;  //!< stream to change
            PropType prop_type;
        };
        using PropFunction = thin_function<void(
                StreamPropType& /* dest */,
                const SmallVector<StreamPropType>& /* srcs */)>;

        //! register a var that should be placed on the stream
        virtual void register_stream_var(
                VarNode* var, StreamPropType prop_type) = 0;

        //! register a propagate function on given var_node
        virtual void register_propagate_function(
                VarNode* var, PropFunction prop_func) = 0;

        //! check if a var has been registered in stream and get its
        //! propagation type
        virtual StreamPropType stream_prop_type(VarNode *var) = 0;
};

}
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
