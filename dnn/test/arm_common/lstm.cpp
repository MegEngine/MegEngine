#include "test/arm_common/fixture.h"

#include "megdnn/oprs.h"
#include "megdnn/oprs/general.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/task_record_check.h"

using namespace megdnn;
using namespace test;

namespace {
//! in arm_common the reserve tensor is not used
void output_canonizer(const CheckerHelper::TensorValueArray& arr) {
    const TensorND& reserve = arr.back();
    TensorND& modif_reserve = const_cast<TensorND&>(reserve);
    modif_reserve.layout = TensorLayout();
}
}  // namespace

TEST_F(ARM_COMMON, LSTMCell) {
    Checker<LSTMCell> checker(handle());
    checker.set_output_canonizer(output_canonizer);
    checker.exec(
            {{1, 10},
             {40, 10},
             {1, 40},
             {1, 10},
             {40, 10},
             {1, 40},
             {1, 10},
             {},
             {},
             {}});
    for (size_t batch : {2})
        for (size_t n : {3, 4, 5, 23, 100})
            for (size_t out : {3, 6, 25, 100}) {
                checker.exec(
                        {{batch, n},
                         {out * 4, n},
                         {1, out * 4},
                         {batch, out},
                         {out * 4, out},
                         {1, out * 4},
                         {batch, out},
                         {},
                         {},
                         {}});
                checker.exec(
                        {{batch, n},
                         {out * 4, n},
                         {batch, out * 4},
                         {batch, out},
                         {out * 4, out},
                         {batch, out * 4},
                         {batch, out},
                         {},
                         {},
                         {}});
            }
}

TEST_F(ARM_COMMON, LSTMCellRecord) {
    TaskRecordChecker<LSTMCell> checker(0);
    checker.exec(
            {{1, 10},
             {40, 10},
             {1, 40},
             {1, 10},
             {40, 10},
             {1, 40},
             {1, 10},
             {},
             {},
             {}});
}

namespace {
void test_lstm(bool bias, bool direction, Handle* handle) {
    Checker<LSTM> checker(handle, true);
    //! because lstm has tanh, exp mathematical compute, after more iteration,
    //! the error will more than 1e-3
    checker.set_epsilon(1e-2);
    checker.set_output_canonizer(output_canonizer);
    for (size_t input_size : {2, 8, 13})
        for (size_t hidden_size : {1, 4, 17}) {
            size_t dir_size = direction == false ? 1 : 2;
            LSTM::Param param;
            param.bidirectional = direction;
            size_t gate_hidden_size = 4 * hidden_size;
            param.bias = bias;
            param.hidden_size = hidden_size;
            for (size_t seq_len : {1, 3, 5})
                for (size_t batch_size : {1, 2, 4})
                    for (size_t number_layer : {1, 2, 4, 5, 8}) {
                        size_t flatten_size = 0;
                        for (size_t layer = 0; layer < number_layer; layer++) {
                            for (size_t dir = 0; dir < dir_size; dir++) {
                                flatten_size += layer == 0
                                                      ? input_size
                                                      : dir_size * hidden_size;  // ih
                                flatten_size += hidden_size;                     // hh
                            }
                        }
                        if (bias) {
                            flatten_size += 2 * dir_size * number_layer;
                        }
                        param.num_layers = number_layer;
                        checker.set_param(param).exec(
                                {{seq_len, batch_size, input_size},  // input
                                 {number_layer * dir_size, batch_size,
                                  hidden_size},  // hx
                                 {number_layer * dir_size, batch_size,
                                  hidden_size},                     // hy
                                 {gate_hidden_size, flatten_size},  // flat weight
                                 {},
                                 {},
                                 {},
                                 {}});
                    }
        }
}

}  // namespace

TEST_F(ARM_COMMON, LSTM_FORWARD_NO_BIAS_NO_DIRCTION) {
    test_lstm(false, false, handle());
}

TEST_F(ARM_COMMON, LSTM_FORWARD_BIAS_NO_DIRCTION) {
    test_lstm(true, false, handle());
}

TEST_F(ARM_COMMON, LSTM_FORWARD_DIRECTION_NO_BIAS) {
    test_lstm(false, true, handle());
}

TEST_F(ARM_COMMON, LSTM_FORWARD_DIRECTION_BIAS) {
    test_lstm(true, true, handle());
}

TEST_F(ARM_COMMON, LSTM_FORWARD_RECORD) {
    TaskRecordChecker<LSTM> checker(0);
    size_t input_size = 2;
    size_t hidden_size = 2;
    size_t gate_hidden_size = 4 * hidden_size;
    LSTM::Param param;
    param.bidirectional = false;
    param.bias = false;
    param.hidden_size = hidden_size;

    // checker.set_output_canonizer(output_canonizer);
    for (size_t seq_len : {1, 3, 5})
        for (size_t batch_size : {1, 2, 4})
            for (size_t number_layer : {1, 2, 4, 5, 8}) {
                param.num_layers = number_layer;
                checker.set_param(param).exec(
                        {{seq_len, batch_size, input_size},        // input
                         {number_layer, batch_size, hidden_size},  // hx
                         {number_layer, batch_size, hidden_size},  // hy
                         {number_layer, gate_hidden_size,
                          input_size + hidden_size},  // flat weight
                         {},
                         {},
                         {},
                         {}});
            }
}

#if MEGDNN_WITH_BENCHMARK

TEST_F(ARM_COMMON, BENCHMARK_LSTM_FORWARD) {
    Benchmarker<LSTM> optimized_bench(handle());
    auto run = [&](size_t hidden_size, size_t input_size) {
        optimized_bench.set_times(20).set_display(true);
        size_t gate_hidden_size = 4 * hidden_size;
        for (bool direction : {false, true}) {
            LSTM::Param param;
            param.hidden_size = hidden_size;
            param.bidirectional = direction;
            param.bias = false;
            size_t dir_size = direction == false ? 1 : 2;
            for (size_t seq_len : {1, 5, 8})
                for (size_t batch_size : {1, 8, 16})
                    for (size_t number_layer : {1}) {
                        param.num_layers = number_layer;
                        size_t flatten_size = 0;
                        for (size_t layer = 0; layer < number_layer; layer++) {
                            for (size_t dir = 0; dir < dir_size; dir++) {
                                flatten_size += layer == 0
                                                      ? input_size
                                                      : dir_size * hidden_size;  // ih
                                flatten_size += hidden_size;                     // hh
                            }
                        }
                        optimized_bench.set_param(param).exec(
                                {{seq_len, batch_size, input_size},  // input
                                 {number_layer * dir_size, batch_size,
                                  hidden_size},  // hx
                                 {number_layer * dir_size, batch_size,
                                  hidden_size},                     // hy
                                 {gate_hidden_size, flatten_size},  // flat weight
                                 {},
                                 {},
                                 {},
                                 {}});
                    }
        }
    };
    run(512, 256);
}

#endif
// vim: syntax=cpp.doxygen
