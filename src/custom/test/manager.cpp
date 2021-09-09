/**
 * \file src/custom/test/manager.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/custom/manager.h"
#include "megbrain/custom/custom.h"
#include "gtest/gtest.h"

#define MANAGER_TEST_LOG 0

namespace custom {

TEST(TestOpManager, TestOpManager) {
    CustomOpManager *com = CustomOpManager::inst();
    com->insert("Op1", CUSTOM_OP_VERSION);
    com->insert("Op2", CUSTOM_OP_VERSION);
    std::shared_ptr<CustomOp> ptr = com->find_or_reg("Op3", CUSTOM_OP_VERSION);
    ASSERT_TRUE(ptr != nullptr);

    std::vector<std::string> op_names = com->op_name_list();
    std::vector<RunTimeId> op_ids = com->op_id_list();

    ASSERT_TRUE(op_names.size() == 3);
    ASSERT_TRUE(op_ids.size() == 3);

#if MANAGER_TEST_LOG
    for (std::string &name: op_names) {
        std::cout << name << std::endl;
    }
#endif

    for (std::string &name: op_names) {
        std::shared_ptr<const CustomOp> op = com->find(name);
        ASSERT_TRUE(op != nullptr);
        ASSERT_TRUE(op->op_type() == name);
        RunTimeId id = com->to_id(name);
        ASSERT_TRUE(com->find(id) == op);
    }

    for (RunTimeId &id: op_ids) {
        std::shared_ptr<const CustomOp> op = com->find(id);
        ASSERT_TRUE(op != nullptr);
        ASSERT_TRUE(op->runtime_id() == id);
        std::string name = com->to_name(id);
        ASSERT_TRUE(com->find(name) == op);
    }

    ASSERT_FALSE(com->erase("Op0"));
#if MANAGER_TEST_LOG
    for (auto &name: com->op_name_list()) {
        std::cout << name << std::endl;
    }
#endif
    ASSERT_TRUE(com->erase("Op1"));
    ASSERT_TRUE(com->erase(com->to_id("Op2")));
    ASSERT_TRUE(com->op_id_list().size() == 1);
    ASSERT_TRUE(com->op_name_list().size() == 1);
    ASSERT_TRUE(com->op_name_list()[0] == "Op3");
    ptr.reset();
    ASSERT_TRUE(com->erase("Op3"));
}

TEST(TestOpManager, TestOpReg) {
    CUSTOM_OP_REG(Op1)
        .add_inputs(2)
        .add_outputs(3)
        .add_input("lhs")
        .add_param("param1", 1)
        .add_param("param2", 3.45);
    
    CUSTOM_OP_REG(Op2)
        .add_input("lhs")
        .add_input("rhs")
        .add_output("out")
        .add_param("param1", "test")
        .add_param("param2", true)
        .add_param("", "no name");

    (void)_Op1;
    (void)_Op2;
        
#if MANAGER_TEST_LOG
    for (const auto &name: CustomOpManager::inst()->op_name_list()) {
        std::cout << CustomOpManager::inst()->find(name)->str() << std::endl;
    }
#endif
}

}
