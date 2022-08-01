#pragma once
#include <gflags/gflags.h>
#include <memory>
#include <unordered_map>
DECLARE_int32(thread);
namespace lar {
/*!
 * \brief: state of model running
 */
enum class RunStage {

    BEFORE_MODEL_LOAD = 0,

    AFTER_NETWORK_CREATED = 1,

    AFTER_MODEL_LOAD = 2,

    BEFORE_OUTSPEC_SET = 3,

    //! using for dump static memory information svg file
    AFTER_OUTSPEC_SET = 4,

    //! using for external c opr library
    MODEL_RUNNING = 5,

    //! using for output dumper
    AFTER_RUNNING_WAIT = 6,

    //! using for external c opr library
    AFTER_RUNNING_ITER = 7,

    AFTER_MODEL_RUNNING = 8,

    GLOBAL_OPTIMIZATION = 9,

    UPDATE_IO = 10,
};
/*!
 * \brief: type of different model
 */
enum class ModelType {
    LITE_MODEL = 0,
    MEGDL_MODEL,
    UNKNOWN,
};
/*!
 * \brief: param for running model
 */
struct RuntimeParam {
    RunStage stage = RunStage::AFTER_MODEL_LOAD;
    size_t warmup_iter;             //! warm up number before running model
    size_t run_iter;                //! iteration number for running model
    size_t threads = FLAGS_thread;  //! thread number for running model (NOTE:it's
                                    //! different from multithread device )
    size_t testcase_num = 1;        //! testcase number for model with testcase
};
/*!
 * \brief:layout type  for running model optimization
 */
enum class OptLayoutType {
    NCHW4 = 1 << 0,
    CHWN4 = 1 << 1,
    NCHW44 = 1 << 2,
    NCHW88 = 1 << 3,
    NCHW32 = 1 << 4,
    NCHW64 = 1 << 5,
    NHWCD4 = 1 << 6,
    NCHW44_DOT = 1 << 7
};
/**
 *  base class to story option value
 */
enum class JsonValueType {
    Bool = 0,
    Number,
    NumberInt32,
    NumberUint64,
    String,

};
struct Value {
    virtual JsonValueType get_type() const = 0;
    virtual std::string type_string() const = 0;
    virtual void reset_value() = 0;
    virtual ~Value() = default;
};

/**
 * class for double option
 */
struct Number final : public Value {
    Number(double v) : m_val(v), m_default_val(v) {}
    static std::shared_ptr<Number> make(double v) {
        return std::make_shared<Number>(v);
    }
    void set_value(double v) { m_val = v; }
    double get_value() { return m_val; }
    double get_default() { return m_default_val; }
    void reset_value() override { m_val = m_default_val; }
    JsonValueType get_type() const override { return JsonValueType::Number; }
    std::string type_string() const override { return "Number"; }

private:
    double m_val;
    double m_default_val;
};

/**
 * class for int32_t option
 */
struct NumberInt32 final : public Value {
    NumberInt32(int32_t v) : m_val(v), m_default_val(v) {}
    static std::shared_ptr<NumberInt32> make(int32_t v) {
        return std::make_shared<NumberInt32>(v);
    }
    void set_value(int32_t v) { m_val = v; }
    int32_t get_value() { return m_val; }
    int32_t get_default() { return m_default_val; }
    void reset_value() override { m_val = m_default_val; }
    JsonValueType get_type() const override { return JsonValueType::NumberInt32; }
    std::string type_string() const override { return "NumberInt32"; }

private:
    int32_t m_val;
    int32_t m_default_val;
};
/**
 * class for uint64 option
 */
struct NumberUint64 final : public Value {
    NumberUint64(uint64_t v) : m_val(v), m_default_val(v) {}
    static std::shared_ptr<NumberUint64> make(uint64_t v) {
        return std::make_shared<NumberUint64>(v);
    }
    void set_value(uint64_t v) { m_val = v; }
    uint64_t get_value() { return m_val; }
    uint64_t get_default() { return m_default_val; }
    void reset_value() override { m_val = m_default_val; }
    JsonValueType get_type() const override { return JsonValueType::NumberUint64; }
    std::string type_string() const override { return "NumberUint64"; }

private:
    uint64_t m_val;
    uint64_t m_default_val;
};

/**
 * class for boolean option
 */
struct Bool final : public Value {
    Bool(bool v) : m_val(v), m_default_val(v) {}
    static std::shared_ptr<Bool> make(bool v) { return std::make_shared<Bool>(v); }
    void set_value(bool v) { m_val = v; }
    bool get_value() { return m_val; }
    bool get_default() { return m_default_val; }
    void reset_value() override { m_val = m_default_val; }
    JsonValueType get_type() const override { return JsonValueType::Bool; }
    std::string type_string() const override { return "Bool"; }

private:
    bool m_val;
    bool m_default_val;
};

/**
 * class for string option
 */
struct String final : public Value {
    String(std::string v) : m_val(v), m_default_val(v) {}
    static std::shared_ptr<String> make(const std::string& v) {
        return std::make_shared<String>(v);
    }
    void set_value(const std::string& v) { m_val = v; }
    std::string& get_value() { return m_val; }
    std::string get_default() { return m_default_val; }
    void reset_value() override { m_val = m_default_val; }
    JsonValueType get_type() const override { return JsonValueType::String; }
    std::string type_string() const override { return "String"; }

private:
    std::string m_val;
    std::string m_default_val;
};

using OptionValMap = std::unordered_map<std::string, std::shared_ptr<lar::Value>>;

}  // namespace lar
// vim: syntax=cpp.doxygen
