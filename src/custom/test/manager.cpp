#include "megbrain_build_config.h"

#if MGB_CUSTOM_OP

#include "gtest/gtest.h"
#include "megbrain/custom/custom.h"
#include "megbrain/custom/manager.h"

#define MANAGER_TEST_LOG 0

namespace custom {

TEST(TestOpManager, TestOpManager) {
    CustomOpManager* com = CustomOpManager::inst();
    std::vector<std::string> builtin_op_names = com->op_name_list();
    size_t builtin_op_num = builtin_op_names.size();

    com->insert("Op1", CUSTOM_OP_VERSION);
    com->insert("Op2", CUSTOM_OP_VERSION);

    std::vector<std::string> op_names = com->op_name_list();
    std::vector<RunTimeId> op_ids = com->op_id_list();

    ASSERT_TRUE(op_names.size() == builtin_op_num + 2);
    ASSERT_TRUE(op_ids.size() == builtin_op_num + 2);

#if MANAGER_TEST_LOG
    for (std::string& name : op_names) {
        std::cout << name << std::endl;
    }
#endif

    for (std::string& name : op_names) {
        std::shared_ptr<const CustomOp> op = com->find(name);
        ASSERT_TRUE(op != nullptr);
        ASSERT_TRUE(op->op_type() == name);
        RunTimeId id = com->to_id(name);
        ASSERT_TRUE(com->find(id) == op);
    }

    for (RunTimeId& id : op_ids) {
        std::shared_ptr<const CustomOp> op = com->find(id);
        ASSERT_TRUE(op != nullptr);
        ASSERT_TRUE(op->runtime_id() == id);
        std::string name = com->to_name(id);
        ASSERT_TRUE(com->find(name) == op);
    }

    ASSERT_FALSE(com->erase("Op0"));
#if MANAGER_TEST_LOG
    for (auto& name : com->op_name_list()) {
        std::cout << name << std::endl;
    }
#endif
    ASSERT_TRUE(com->erase("Op1"));
    ASSERT_TRUE(com->op_id_list().size() == builtin_op_num + 1);
    ASSERT_TRUE(com->op_name_list().size() == builtin_op_num + 1);
    ASSERT_TRUE(com->erase("Op2"));
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
    for (const auto& name : CustomOpManager::inst()->op_name_list()) {
        std::cout << CustomOpManager::inst()->find(name)->str() << std::endl;
    }
#endif
}

}  // namespace custom

#endif
