#include <limits>
#include "megdnn/dtype.h"
#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs.h"
#include "megdnn/oprs/nn.h"
#include "megdnn/thin/small_vector.h"
#include "test/common/checker.h"
#include "test/naive/fixture.h"

namespace megdnn {
namespace test {

template <typename Checker, typename T>
void mha_test_main(Checker& checker, T type) {
    MultiHeadAttn::Param param;
    double neginf = -std::numeric_limits<double>::infinity();
    param.attn_mask_type = param::MultiHeadAttn::AttnMaskType::NO_MASK;
    param.tensor_combination_type = param::MultiHeadAttn::TensorCombinationType::NONE;
    param.seed = 1686221026;
    param.attn_prob = 0.0;
    param.out_prob = 0.0;
    param.embeding_size = 2;
    param.v_size = 2;
    param.k_size = 3;
    param.kproj_size = 2;
    param.oproj_size = 2;
    param.qproj_size = 2;
    param.vproj_size = 2;
    param.training = false;
    TensorShape _qshape{2, 2, 2};
    std::initializer_list<float> _query{0.14335328, 0.9446689, 0.5218483,  0.41466194,
                                        0.2645556,  0.7742337, 0.45615032, 0.56843394};
    TensorShape _kshape{2, 2, 3};
    std::initializer_list<float> _key{0.0187898,  0.6176355,  0.6120957, 0.616934,
                                      0.94374806, 0.6818203,  0.3595079, 0.43703195,
                                      0.6976312,  0.06022547, 0.6667667, 0.67063785};
    TensorShape _vshape{2, 2, 2};
    std::initializer_list<float> _value{0.21038257, 0.12892629, 0.31542835, 0.36371076,
                                        0.57019675, 0.43860152, 0.9883738,  0.10204481};
    TensorShape _wbshape{18};
    std::initializer_list<float> _io_weight_bias{
            0.5488135,  0.71518934, 0.60276335,  0.5448832, 0.4236548,   0.6458941,
            0.4375872,  0.891773,   0.96366274,  0.3834415, 0.79172504,  0.5288949,
            0.56804454, 0.92559665, 0.071036056, 0.0871293, 0.020218397, 0.83261985};
    TensorShape _amshape{2, 2};
    std::initializer_list<float> _attn_mask{0.0, static_cast<float>(neginf), 0.0, 0.0};
    // num_heads = 1
    {
        param.num_heads = 1;
        param.sm_scaler = 0.7071067;
        auto query = TensorValue(_qshape, type, _query);
        auto key = TensorValue(_kshape, type, _key);
        auto value = TensorValue(_vshape, type, _value);
        auto io_weight_bias = TensorValue(_wbshape, type, _io_weight_bias);
        auto out = TensorValue(
                {2, 2, 2}, type,
                {0.034801412, 0.36719304, 0.03457721, 0.3645532, 0.068083055,
                 0.61898696, 0.06808451, 0.6189755});
        auto attn_weight = TensorValue(
                {2, 2, 2}, type,
                {0.3729596, 0.6270404, 0.3836875, 0.61631244, 0.50552285, 0.49447718,
                 0.50534284, 0.49465716});
        auto othr_reservespace = TensorValue(
                {33}, type, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        checker.set_param(param).set_bypass(9).set_bypass(10).exect(
                Testcase{query, key, value, io_weight_bias, {}, {}, {}, {}, {}, {}, {}},
                Testcase{
                        {},
                        {},
                        {},
                        {},
                        {},
                        {},
                        {},
                        out,
                        attn_weight,
                        {},
                        othr_reservespace});
    }

    // num_heads = 2
    {
        param.num_heads = 2;
        param.sm_scaler = 1.0;
        auto query = TensorValue(_qshape, type, _query);
        auto key = TensorValue(_kshape, type, _key);
        auto value = TensorValue(_vshape, type, _value);
        auto io_weight_bias = TensorValue(_wbshape, type, _io_weight_bias);
        auto out = TensorValue(
                {2, 2, 2}, type,
                {0.03388246, 0.3616359, 0.0336703, 0.3607049, 0.068048045, 0.61852735,
                 0.06805048, 0.61852974});
        auto attn_weight = TensorValue(
                {4, 2, 2}, type,
                {0.42549428, 0.5745057, 0.438196, 0.561804, 0.39305627, 0.6069437,
                 0.39609915, 0.60390085, 0.50799584, 0.49200416, 0.5077489, 0.49225107,
                 0.49981424, 0.5001858, 0.49980664, 0.50019336});
        auto othr_reservespace =
                TensorValue({33}, type, {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.});
        checker.set_param(param).set_bypass(9).set_bypass(10).exect(
                Testcase{query, key, value, io_weight_bias, {}, {}, {}, {}, {}, {}, {}},
                Testcase{
                        {},
                        {},
                        {},
                        {},
                        {},
                        {},
                        {},
                        out,
                        attn_weight,
                        {},
                        othr_reservespace});
    }

    param.attn_mask_type = param::MultiHeadAttn::AttnMaskType::DEFAULT_MASK;
    param.tensor_combination_type =
            param::MultiHeadAttn::TensorCombinationType::ONLY_MASK;
    // attn_mask1
    {
        param.sm_scaler = 0.7071067;
        param.num_heads = 1;
        auto query = TensorValue(_qshape, type, _query);
        auto key = TensorValue(_kshape, type, _key);
        auto value = TensorValue(_vshape, type, _value);
        auto io_weight_bias = TensorValue(_wbshape, type, _io_weight_bias);
        auto attn_mask = TensorValue(_amshape, type, _attn_mask);
        auto out = TensorValue(
                {2, 2, 2}, type,
                {0.021696962, 0.21289916, 0.03457721, 0.3645532, 0.06407211, 0.6501551,
                 0.06808451, 0.6189755});
        auto attn_weight = TensorValue(
                {2, 2, 2}, type,
                {1.0, 0.0, 0.3836875, 0.61631244, 1.0, 0.0, 0.50534284, 0.49465716});
        auto othr_reservespace =
                TensorValue({33}, type, {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.});

        checker.set_param(param).set_bypass(9).set_bypass(10).exect(
                Testcase{
                        query,
                        key,
                        value,
                        io_weight_bias,
                        attn_mask,
                        {},
                        {},
                        {},
                        {},
                        {},
                        {}},
                Testcase{
                        {},
                        {},
                        {},
                        {},
                        {},
                        {},
                        {},
                        out,
                        attn_weight,
                        {},
                        othr_reservespace});
    }

    // attn_mask2
    {
        param.sm_scaler = 1.0;
        param.num_heads = 2;
        auto query = TensorValue(_qshape, type, _query);
        auto key = TensorValue(_kshape, type, _key);
        auto value = TensorValue(_vshape, type, _value);
        auto io_weight_bias = TensorValue(_wbshape, type, _io_weight_bias);
        auto attn_mask = TensorValue(_amshape, type, _attn_mask);
        auto out = TensorValue(
                {2, 2, 2}, type,
                {0.021696962, 0.21289916, 0.0336703, 0.3607049, 0.06407211, 0.6501551,
                 0.06805048, 0.61852974});
        auto attn_weight = TensorValue(
                {4, 2, 2}, type,
                {1.0, 0.0, 0.438196, 0.561804, 1.0, 0.0, 0.39609915, 0.60390085, 1.0,
                 0.0, 0.5077489, 0.49225107, 1.0, 0.0, 0.49980664, 0.50019336});
        auto othr_reservespace =
                TensorValue({33}, type, {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.});
        checker.set_param(param).set_bypass(9).set_bypass(10).exect(
                Testcase{
                        query,
                        key,
                        value,
                        io_weight_bias,
                        attn_mask,
                        {},
                        {},
                        {},
                        {},
                        {},
                        {}},
                Testcase{
                        {},
                        {},
                        {},
                        {},
                        {},
                        {},
                        {},
                        out,
                        attn_weight,
                        {},
                        othr_reservespace});
    }

    // big
    {
        param.embeding_size = 4;
        param.k_size = 4;
        param.kproj_size = 4;
        param.oproj_size = 4;
        param.qproj_size = 4;
        param.v_size = 4;
        param.vproj_size = 4;
        param.sm_scaler = 0.7071067;
        param.num_heads = 2;
        auto query = TensorValue(
                {3, 4, 4}, type,
                {0.31798318, 0.41426298,  0.064147495, 0.6924721,  0.56660146,
                 0.2653895,  0.5232481,   0.09394051,  0.5759465,  0.9292962,
                 0.31856894, 0.6674104,   0.13179787,  0.7163272,  0.2894061,
                 0.18319136, 0.5865129,   0.020107547, 0.82894003, 0.004695476,
                 0.6778165,  0.27000797,  0.735194,    0.96218854, 0.24875315,
                 0.57615733, 0.5920419,   0.5722519,   0.22308163, 0.952749,
                 0.44712538, 0.84640867,  0.6994793,   0.29743695, 0.81379783,
                 0.39650574, 0.8811032,   0.5812729,   0.8817354,  0.6925316,
                 0.7252543,  0.50132436,  0.95608366,  0.6439902,  0.42385504,
                 0.6063932,  0.019193199, 0.30157483});
        auto key = TensorValue(
                {3, 5, 4}, type,
                {0.66017354,  0.2900776,   0.6180154,  0.4287687,   0.13547407,
                 0.29828233,  0.5699649,   0.59087276, 0.57432526,  0.6532008,
                 0.65210325,  0.43141845,  0.8965466,  0.36756188,  0.43586493,
                 0.89192337,  0.806194,    0.7038886,  0.10022689,  0.9194826,
                 0.7142413,   0.998847,    0.1494483,  0.86812603,  0.16249293,
                 0.6155596,   0.123819984, 0.8480082,  0.807319,    0.56910074,
                 0.4071833,   0.069166996, 0.69742876, 0.45354268,  0.7220556,
                 0.8663823,   0.9755215,   0.8558034,  0.011714084, 0.35997805,
                 0.72999054,  0.17162968,  0.5210366,  0.05433799,  0.19999653,
                 0.018521795, 0.7936977,   0.22392468, 0.34535167,  0.9280813,
                 0.7044144,   0.03183893,  0.16469416, 0.6214784,   0.5772286,
                 0.23789282,  0.934214,    0.6139659,  0.5356328,   0.58991});
        auto value = TensorValue(
                {3, 5, 4}, type,
                {0.730122,    0.311945,    0.39822108, 0.20984375, 0.186193,
                 0.9443724,   0.73955077,  0.49045882, 0.22741462, 0.25435647,
                 0.05802916,  0.43441662,  0.3117959,  0.6963435,  0.37775183,
                 0.17960368,  0.024678728, 0.06724963, 0.67939276, 0.45369685,
                 0.5365792,   0.8966713,   0.9903389,  0.21689698, 0.6630782,
                 0.26332238,  0.020651,    0.7583786,  0.32001716, 0.3834639,
                 0.5883171,   0.8310484,   0.6289818,  0.8726507,  0.27354205,
                 0.7980468,   0.18563594,  0.95279163, 0.68748826, 0.21550767,
                 0.9473706,   0.7308558,   0.25394166, 0.21331197, 0.5182007,
                 0.025662718, 0.20747007,  0.42468548, 0.37416998, 0.46357542,
                 0.27762872,  0.58678436,  0.8638556,  0.11753186, 0.5173791,
                 0.13206811,  0.7168597,   0.3960597,  0.5654213,  0.18327984});
        auto io_weight_bias = TensorValue(
                {64}, type,
                {0.5488135,  0.71518934,  0.60276335, 0.5448832,  0.4236548,
                 0.6458941,  0.4375872,   0.891773,   0.96366274, 0.3834415,
                 0.79172504, 0.5288949,   0.56804454, 0.92559665, 0.071036056,
                 0.0871293,  0.020218397, 0.83261985, 0.77815676, 0.87001216,
                 0.9786183,  0.7991586,   0.46147937, 0.7805292,  0.11827443,
                 0.639921,   0.14335328,  0.9446689,  0.5218483,  0.41466194,
                 0.2645556,  0.7742337,   0.45615032, 0.56843394, 0.0187898,
                 0.6176355,  0.6120957,   0.616934,   0.94374806, 0.6818203,
                 0.3595079,  0.43703195,  0.6976312,  0.06022547, 0.6667667,
                 0.67063785, 0.21038257,  0.12892629, 0.31542835, 0.36371076,
                 0.57019675, 0.43860152,  0.9883738,  0.10204481, 0.20887676,
                 0.16130951, 0.6531083,   0.2532916,  0.46631077, 0.2444256,
                 0.15896958, 0.11037514,  0.6563296,  0.13818295});
        auto attn_mask = TensorValue(
                {4, 5}, type, {0.0,    neginf, neginf, neginf, neginf, 0.0,   0.0,
                               neginf, neginf, neginf, 0.0,    0.0,    0.0,   neginf,
                               neginf, 0.0,    0.0,    0.0,    0.0,    neginf});
        auto out = TensorValue(
                {3, 4, 4}, type,
                {1.6912086, 0.6261238,  1.4156955, 0.7555092,  2.177456,  0.81336236,
                 1.7710469, 0.9678083,  1.5905006, 0.5963271,  1.3028629, 0.7127416,
                 1.7710252, 0.66557235, 1.4539132, 0.79054594, 3.0321078, 1.1345028,
                 2.4570594, 1.3310935,  2.7499804, 1.0264602,  2.2247908, 1.2149017,
                 2.5934792, 0.96562505, 2.0686975, 1.1467543,  2.7249372, 1.0191544,
                 2.206509,  1.213203,   2.366218,  0.89146763, 2.0670228, 1.0715914,
                 1.9098943, 0.7156819,  1.644923,  0.86424935, 1.897423,  0.7128644,
                 1.5815731, 0.8578857,  1.7762156, 0.66100645, 1.4773674, 0.79827106});
        auto attn_weight = TensorValue(
                {6, 4, 5}, type,
                {1.0,        0.0,         0.0,        0.0,        0.0,
                 0.5467959,  0.45320415,  0.0,        0.0,        0.0,
                 0.27896783, 0.1846584,   0.5363738,  0.0,        0.0,
                 0.21476158, 0.17719562,  0.30043945, 0.30760336, 0.0,
                 1.0,        0.0,         0.0,        0.0,        0.0,
                 0.60972804, 0.390272,    0.0,        0.0,        0.0,
                 0.34474388, 0.18311267,  0.47214353, 0.0,        0.0,
                 0.22621259, 0.15230283,  0.2759932,  0.34549138, 0.0,
                 1.0,        0.0,         0.0,        0.0,        0.0,
                 0.81995606, 0.18004394,  0.0,        0.0,        0.0,
                 0.6023512,  0.20229074,  0.19535813, 0.0,        0.0,
                 0.4820769,  0.1220997,   0.1253118,  0.27051157, 0.0,
                 1.0,        0.0,         0.0,        0.0,        0.0,
                 0.75752157, 0.24247849,  0.0,        0.0,        0.0,
                 0.52649254, 0.1970557,   0.27645174, 0.0,        0.0,
                 0.36510885, 0.114216655, 0.16812822, 0.35254624, 0.0,
                 1.0,        0.0,         0.0,        0.0,        0.0,
                 0.62315977, 0.3768402,   0.0,        0.0,        0.0,
                 0.16396031, 0.10346532,  0.73257434, 0.0,        0.0,
                 0.20188671, 0.15875323,  0.3786241,  0.26073596, 0.0,
                 1.0,        0.0,         0.0,        0.0,        0.0,
                 0.65460557, 0.34539443,  0.0,        0.0,        0.0,
                 0.31592765, 0.17300747,  0.51106495, 0.0,        0.0,
                 0.24820071, 0.19006157,  0.32281628, 0.23892137, 0.0});
        auto othr_reservespace = TensorValue(
                {217}, type,
                {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.});
        checker.set_param(param).set_bypass(9).set_bypass(10).exect(
                Testcase{
                        query,
                        key,
                        value,
                        io_weight_bias,
                        attn_mask,
                        {},
                        {},
                        {},
                        {},
                        {},
                        {}},
                Testcase{
                        {},
                        {},
                        {},
                        {},
                        {},
                        {},
                        {},
                        out,
                        attn_weight,
                        {},
                        othr_reservespace});
    }
}

TEST_F(NAIVE, MULTIHEADATTN_FORWARD) {
    Checker<MultiHeadAttn> checker(handle(), true);
    mha_test_main(checker, dtype::Float32());
    mha_test_main(checker, dtype::Float16());
}

}  // namespace test
}  // namespace megdnn
