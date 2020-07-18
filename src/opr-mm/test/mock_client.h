/**
 * \file src/opr-mm/test/mock_client.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/group_manager.h"

namespace mgb {
namespace test {

class MockGroupClient final : public opr::GroupClient {
    public:
        using RegisterInfo = opr::GroupManager::RegisterInfo;

        MockGroupClient(const std::string& server_addr = "mock_addr") :
            m_addr(server_addr) {
        }

        ~MockGroupClient() override = default;

        const std::string& get_addr() const {
            return m_addr;
        }

	RegisterInfo opr_register(const std::string& key, size_t nr_devices,
                                  bool is_root, int rank, uint64_t comp_node_hash) override {
            return m_mgr.opr_register(key, nr_devices, is_root, rank, comp_node_hash);
        }

        void bcast_addr(std::string& master_ip, int& port,
                        const std::string& key, uint32_t size,
                        uint32_t rank, uint32_t root) override {
            return m_mgr.bcast_addr(master_ip, port, key, size, rank, root);
        }

        void set_output_shape(const std::string& key,
                              const TensorShape& shape) override {
            m_mgr.set_output_shape(key, shape);
        }

        TensorShape get_output_shape(const std::string& key) override {
            return m_mgr.get_output_shape(key);
        }

        uint32_t group_barrier(uint32_t size, uint32_t rank) override {
            return m_mgr.group_barrier(size, rank);
        }

    private:
        const std::string m_addr;
        opr::GroupManager m_mgr;
};

} // namespace test
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
