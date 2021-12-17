/**
 * \file src/opr/include/training/dataview.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 */

#pragma once

#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/tensor_manip.h"

#include <type_traits>

namespace mgb {
using DataPair = std::pair<
        std::shared_ptr<mgb::HostTensorND>, std::shared_ptr<mgb::HostTensorND>>;

//! The interface of the dataset.
class IDataView {
public:
    /*!
     * The method to get an item in dataset with index.
     */
    virtual DataPair get_item(int idx) = 0;

    /*!
     * The method to get the size of the dataset.
     */
    virtual size_t size() = 0;

    virtual ~IDataView() = default;
};

//! The definition of dataloader, which is corresponding to the <DataLoader> of
//! Python API of MegEngine.
class DataLoader {
public:
    DataLoader(
            std::shared_ptr<IDataView> dataview, mgb::CompNode compnode,
            unsigned long batchsize = 1U, bool shuffle = false, bool drop_last = true);
    /*!
     * Get the next pair of data of the dataset.
     */
    DataPair next();
    /*!
     * Get the size of the dataloader.
     */
    size_t size();

private:
    std::shared_ptr<IDataView> m_dataview;
    mgb::CompNode m_comp_node;
    unsigned long m_batchsize;
    bool m_shuffle;
    bool m_drop_last;
    size_t m_idx;
    mgb::TensorShape m_data_shape;
    mgb::TensorShape m_label_shape;
    mgb::DType m_data_type;
    mgb::DType m_label_type;

    // Only used in the temp solution for shuffle
    std::vector<int> m_index_collection;
};

}  // namespace mgb
