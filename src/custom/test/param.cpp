#include "megbrain_build_config.h"

#if MGB_CUSTOM_OP

#include <iostream>
#include "gtest/gtest.h"
#include "megbrain/custom/param.h"

#define PARAM_TEST_LOG 0

namespace custom {

#define SchemaDef                                                               \
    ParamSchema schema_bool("param_bool", true, "bool");                        \
    ParamSchema schema_flt("param_flt", 2.3f, "float");                         \
    ParamSchema schema_int("param_int", 4, "int");                              \
    ParamSchema schema_str("param_str", "test", "string");                      \
    ParamSchema schema_bool_list("param_bl", {true, false, true}, "bool list"); \
    ParamSchema schema_flt_list("param_fl", {1.1f, 2.2f, 3.3f}, "float list");  \
    ParamSchema schema_int_list("param_il", {1, 2, 3}, "int list");             \
    ParamSchema schema_str_list("param_sl", {"test1", "test2", "test3"}, "string list")

#define InfoDef                                 \
    info.meta().emplace_back(schema_bool);      \
    info.meta().emplace_back(schema_flt);       \
    info.meta().emplace_back(schema_int);       \
    info.meta().emplace_back(schema_str);       \
    info.meta().emplace_back(schema_bool_list); \
    info.meta().emplace_back(schema_flt_list);  \
    info.meta().emplace_back(schema_int_list);  \
    info.meta().emplace_back(schema_str_list)

TEST(TestParam, TestParamScheme) {
#if PARAM_TEST_LOG
    SchemaDef;
    ParamSchema new_schema = schema_int;

    std::cout << schema_bool.str() << std::endl;
    std::cout << schema_flt.str() << std::endl;
    std::cout << schema_int.str() << std::endl;
    std::cout << schema_str.str() << std::endl;
    std::cout << schema_bool_list.str()
              << "len: " << schema_bool_list.default_val().size() << std::endl;
    std::cout << schema_flt_list.str()
              << "len: " << schema_flt_list.default_val().size() << std::endl;
    std::cout << schema_int_list.str()
              << "len: " << schema_int_list.default_val().size() << std::endl;
    std::cout << schema_str_list.str()
              << "len: " << schema_str_list.default_val().size() << std::endl;

    std::cout << new_schema.str() << std::endl;
#endif
}

TEST(TestParam, TestParamVal) {
    ParamVal pv1 = 1.2f, pv2 = true, pv3 = "test", pv4 = {0, 1, 2},
             pv5 = {true, false, true};

#if PARAM_TEST_LOG
    ParamVal pv6 = {"test1", "test2", "test3"};
    std::cout << pv1.str() << std::endl;
    std::cout << pv2.str() << std::endl;
    std::cout << pv3.str() << std::endl;
    std::cout << pv4.str() << std::endl;
    std::cout << pv5.str() << std::endl;
    std::cout << pv6.str() << std::endl;
#endif

    ParamVal pv_manip = pv1;
    ASSERT_TRUE(pv_manip.type() == pv1.type());
    ASSERT_TRUE(pv_manip == pv1);
    pv_manip = 1.3;
    ASSERT_TRUE(pv_manip.type() != pv1.type());
    ASSERT_TRUE(pv_manip != pv1);
    ASSERT_TRUE(pv_manip > pv1);
    pv_manip = pv_manip + pv1;
    ASSERT_TRUE(pv_manip.type() == ParamDynType::Float64);
    ASSERT_TRUE(pv_manip == 1.3 + 1.2f);
    pv_manip = 1.3f + 1.2f;
    ASSERT_TRUE(pv_manip.type() == pv1.type());

    pv_manip = false;
    ASSERT_TRUE(pv_manip.type() == pv2.type());
    ASSERT_TRUE(pv_manip.type() == ParamDynType::Bool);
    ASSERT_TRUE(pv_manip != pv2);

    pv_manip = "test";
    ASSERT_TRUE(pv_manip.type() == pv3.type());
    ASSERT_TRUE(pv_manip.type() == ParamDynType::String);
    ASSERT_TRUE(pv_manip == pv3);
    pv_manip = "test1";
    ASSERT_TRUE(pv_manip > pv3);
    pv_manip = pv_manip + pv3;
    ASSERT_TRUE(pv_manip == "test1test");

    pv_manip = {0, 1, 2};
    ASSERT_TRUE(pv_manip.type() == pv4.type());
    ASSERT_TRUE(pv_manip.type() == ParamDynType::Int32List);
    ASSERT_TRUE(pv_manip == pv4);
    pv_manip = {3, 2, 1};
    ASSERT_TRUE(pv_manip != pv4);
    ASSERT_TRUE(pv_manip > pv4);

    pv_manip = {true, false, true};
    ASSERT_TRUE(pv_manip.type() == pv5.type());
    ASSERT_TRUE(pv_manip.type() == ParamDynType::BoolList);
    ASSERT_TRUE(pv_manip == pv5);
    pv_manip = {false, true, false};
    ASSERT_TRUE(pv_manip != pv5);
}

TEST(TestParam, TestParamInfo) {
    ParamInfo info;
    info.set_tag("Test");
#if PARAM_TEST_LOG
    uint32_t tag = info.tag();
    std::cout << tag << std::endl;
#endif

    SchemaDef;
    InfoDef;

    ParamInfo new_info1, new_info2;
    new_info1.set_meta(info.meta());
    new_info2.meta() = info.meta();

#if PARAM_TEST_LOG
    for (auto ele : new_info1.meta()) {
        std::cout << ele.str() << std::endl;
    }
    for (auto ele : new_info2.meta()) {
        std::cout << ele.str() << std::endl;
    }
#endif
}

TEST(TestParam, TestParam) {
    ParamInfo info;
    SchemaDef;
    InfoDef;

    Param param(info);

#if PARAM_TEST_LOG
    std::vector<std::string> names = {"param_bool", "param_flt", "param_int",
                                      "param_str",  "param_bl",  "param_fl",
                                      "param_il",   "param_sl"};
    for (auto& name : names) {
        std::cout << param[name].str() << std::endl;
        ;
    }
#endif
    ASSERT_TRUE(param["param_bool"] == true);
    ASSERT_TRUE(param["param_flt"] == 2.3f);
    ASSERT_TRUE(param["param_int"] == 4);
    ASSERT_TRUE(param["param_str"] == "test");
    ASSERT_TRUE(param["param_bl"] == ParamVal({true, false, true}));
    ASSERT_TRUE(param["param_fl"] == ParamVal({1.1f, 2.2f, 3.3f}));
    ASSERT_TRUE(param["param_il"] == ParamVal({1, 2, 3}));
    ASSERT_TRUE(param["param_sl"] == ParamVal({"test1", "test2", "test3"}));

    param["param_bool"] = false;
    param["param_flt"] = 3.4f;
    param["param_int"] = 5;
    param["param_str"] = "tset";
    param["param_bl"] = {false, true, false, true};
    param["param_fl"] = {7.6f, 6.5f};
    param["param_il"] = {5, 4, 3, 2, 1};
    param["param_sl"] = {"1tset", "2tset", "3tset", "4tset", "5tset"};

    ASSERT_TRUE(param["param_bool"] != true);
    ASSERT_TRUE(param["param_flt"] != 2.3f);
    ASSERT_TRUE(param["param_int"] != 4);
    ASSERT_TRUE(param["param_str"] != "test");
    ASSERT_TRUE(param["param_bl"] != ParamVal({true, false, true}));
    ASSERT_TRUE(param["param_fl"] != ParamVal({1.1f, 2.2f, 3.3f}));
    ASSERT_TRUE(param["param_il"] != ParamVal({1, 2, 3}));
    ASSERT_TRUE(param["param_sl"] != ParamVal({"test1", "test2", "test3"}));

    ASSERT_TRUE(param["param_bool"] == false);
    ASSERT_TRUE(param["param_flt"] == 3.4f);
    ASSERT_TRUE(param["param_int"] == 5);
    ASSERT_TRUE(param["param_str"] == "tset");
    ASSERT_TRUE(param["param_bl"] == ParamVal({false, true, false, true}));
    ASSERT_TRUE(param["param_fl"] == ParamVal({7.6f, 6.5f}));
    ASSERT_TRUE(param["param_il"] == ParamVal({5, 4, 3, 2, 1}));
    ASSERT_TRUE(
            param["param_sl"] ==
            ParamVal({"1tset", "2tset", "3tset", "4tset", "5tset"}));

#if PARAM_TEST_LOG
    Param copy_param = param;
    for (auto& name : names) {
        std::cout << copy_param[name].str() << std::endl;
    }
#endif

    Param loaded_param(info);
    std::string bytes = param.to_bytes();
    loaded_param.from_bytes(bytes);

#if PARAM_TEST_LOG
    for (auto& kv : loaded_param.raw()) {
        std::cout << kv.first << ":\n" << kv.second.str() << std::endl;
    }
#endif
}

}  // namespace custom

#endif
