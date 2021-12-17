/**
 * \file src/opr/impl/training/dataview.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 */

#include "megbrain/opr/training/dataview.h"

#include "megbrain/exception.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/tensor.h"

#include <random>

namespace mgb {
DataLoader::DataLoader(
        std::shared_ptr<IDataView> dataview, mgb::CompNode comp_node,
        unsigned long batchsize, bool shuffle, bool drop_last)
        : m_dataview(dataview),
          m_comp_node(comp_node),
          m_batchsize(batchsize),
          m_shuffle(shuffle),
          m_drop_last(drop_last),
          m_idx(0) {
    if (!m_comp_node.valid()) {
        m_comp_node = CompNode::load("xpu0");
    }
    for (size_t i = 0; i < m_dataview->size(); i++) {
        m_index_collection.push_back(i);
    }

    if (m_dataview->size() > 0) {
        auto data_sample = m_dataview->get_item(0);
        SmallVector<size_t> dshape;
        dshape.push_back(static_cast<size_t>(batchsize));
        for (size_t i = 0; i < data_sample.first->layout().ndim; i++) {
            dshape.push_back(data_sample.first->shape()[i]);
        }
        m_data_shape = dshape;
        SmallVector<size_t> lshape;
        lshape.push_back(m_batchsize);
        for (size_t i = 1; i < data_sample.second->layout().ndim; i++) {
            lshape.push_back(data_sample.second->shape()[i]);
        }
        m_label_shape = lshape;

        m_data_type = data_sample.first->dtype();
        m_label_type = data_sample.second->dtype();
    } else {
        mgb_throw(AssertionError, "The dataset is empty.");
    }
}

size_t DataLoader::size() {
    return m_dataview->size() / m_batchsize;
}

DataPair DataLoader::next() {
    if (m_idx == 0 && m_shuffle) {
        std::shuffle(
                m_index_collection.begin(), m_index_collection.end(),
                std::default_random_engine());
    }
    if (m_idx >= m_index_collection.size() - m_batchsize) {
        m_idx = 0;
    }

    auto data = std::make_shared<HostTensorND>(m_comp_node, m_data_shape, m_data_type);
    auto label =
            std::make_shared<HostTensorND>(m_comp_node, m_label_shape, m_label_type);
    size_t data_bytes = m_dataview->get_item(m_index_collection.at(m_idx))
                                .first->layout()
                                .access_bytes();
    size_t label_bytes = m_dataview->get_item(m_index_collection.at(m_idx))
                                 .second->layout()
                                 .access_bytes();

    auto data_ptr = data->raw_ptr();
    auto label_ptr = label->raw_ptr();
    for (unsigned int i = 0; i < m_batchsize; i++) {
        auto item = m_dataview->get_item(m_index_collection.at(m_idx));
        auto pre_data = item.first;
        auto pre_label = item.second;
        auto pre_data_ptr = pre_data->raw_ptr();
        auto pre_label_ptr = pre_label->raw_ptr();

        memcpy(data_ptr + data_bytes * i, pre_data_ptr,
               sizeof(megdnn::dt_byte) * data_bytes);
        memcpy(label_ptr + label_bytes * i, pre_label_ptr,
               sizeof(megdnn::dt_byte) * label_bytes);
        m_idx++;
    }
    return {data, label};
}
}  // namespace mgb
