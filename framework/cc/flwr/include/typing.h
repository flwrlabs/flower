/***********************************************************************************************************
 *
 * @file typing.h
 *
 * @brief C++ Flower type definitions
 *
 * @author Lekang Jiang
 *
 * @version 1.0
 *
 * @date 03/09/2021
 *
 * ********************************************************************************************************/

#pragma once
#include <cstdint>
#include <list>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace flwr_local {
/**
 * This class contains C++ types corresponding to ProtoBuf types that
 * ProtoBuf considers to be "Scalar Value Types", even though some of them
 * arguably do not conform to other definitions of what a scalar is. There is no
 * "bytes" type in C++, so "string" is used instead of bytes in Python (char*
 * can also be used if needed) In C++, Class is easier to use than Union (can be
 * changed if needed) Source:
 * https://developers.google.com/protocol-buffers/docs/overview#scalar
 *
 */
class Scalar {
public:
  // Getters
  std::optional<bool> getBool() { return b; }
  std::optional<std::string> getBytes() { return bytes; }
  std::optional<double> getDouble() { return d; }
  std::optional<int> getInt() { return i; }
  std::optional<std::string> getString() { return string; }

  // Setters
  void setBool(bool b) { this->b = b; }
  void setBytes(const std::string &bytes) { this->bytes = bytes; }
  void setDouble(double d) { this->d = d; }
  void setInt(int i) { this->i = i; }
  void setString(const std::string &string) { this->string = string; }

private:
  std::optional<bool> b = std::nullopt;
  std::optional<std::string> bytes = std::nullopt;
  std::optional<double> d = std::nullopt;
  std::optional<int> i = std::nullopt;
  std::optional<std::string> string = std::nullopt;
};

typedef std::map<std::string, flwr_local::Scalar> Metrics;

/**
 * Model parameters
 */
class Parameters {
public:
  Parameters() {}
  Parameters(const std::list<std::string> &tensors,
             const std::string &tensor_type)
      : tensors(tensors), tensor_type(tensor_type) {}

  // Getters
  const std::list<std::string> getTensors() const { return tensors; }
  const std::string getTensor_type() const { return tensor_type; }

  // Setters
  void setTensors(const std::list<std::string> &tensors) {
    this->tensors = tensors;
  }
  void setTensor_type(const std::string &tensor_type) {
    this->tensor_type = tensor_type;
  }

private:
  std::list<std::string> tensors;
  std::string tensor_type;
};

/**
 * Response when asked to return parameters
 */
class ParametersRes {
public:
  explicit ParametersRes(const Parameters &parameters)
      : parameters(parameters) {}

  const Parameters getParameters() const { return parameters; }
  void setParameters(const Parameters &p) { parameters = p; }

private:
  Parameters parameters;
};

/**
 * Fit instructions for a client
 */
class FitIns {
public:
  FitIns(const Parameters &parameters,
         const std::map<std::string, flwr_local::Scalar> &config)
      : parameters(parameters), config(config) {}

  // Getters
  Parameters getParameters() { return parameters; }
  std::map<std::string, Scalar> getConfig() { return config; }

  // Setters
  void setParameters(const Parameters &p) { parameters = p; }
  void setConfig(const std::map<std::string, Scalar> &config) {
    this->config = config;
  }

private:
  Parameters parameters;
  std::map<std::string, Scalar> config;
};

/**
 * Fit response from a client
 */
class FitRes {
public:
  FitRes() {}
  FitRes(const Parameters &parameters, int num_examples, int num_examples_ceil,
         float fit_duration, const Metrics &metrics)
      : _parameters(parameters), _num_examples(num_examples),
        _fit_duration(fit_duration), _metrics(metrics) {}

  // Getters
  const Parameters getParameters() const { return _parameters; }
  const int getNum_example() const { return _num_examples; }
  const std::optional<float> getFit_duration() const { return _fit_duration; }
  const std::optional<Metrics> getMetrics() const { return _metrics; }

  // Setters
  void setParameters(const Parameters &p) { _parameters = p; }
  void setNum_example(int n) { _num_examples = n; }
  void setFit_duration(float f) { _fit_duration = f; }
  void setMetrics(const flwr_local::Metrics &m) { _metrics = m; }

private:
  Parameters _parameters;
  int _num_examples;
  std::optional<float> _fit_duration = std::nullopt;
  std::optional<Metrics> _metrics = std::nullopt;
};

/**
 * Evaluate instructions for a client
 */
class EvaluateIns {
public:
  EvaluateIns(const Parameters &parameters,
              const std::map<std::string, Scalar> &config)
      : parameters(parameters), config(config) {}

  // Getters
  Parameters getParameters() { return parameters; }
  std::map<std::string, Scalar> getConfig() { return config; }

  // Setters
  void setParameters(const Parameters &p) { parameters = p; }
  void setConfig(const std::map<std::string, Scalar> &config) {
    this->config = config;
  }

private:
  Parameters parameters;
  std::map<std::string, Scalar> config;
};

/**
 * Evaluate response from a client
 */
class EvaluateRes {
public:
  EvaluateRes() {}
  EvaluateRes(float loss, int num_examples, float accuracy,
              const Metrics &metrics)
      : loss(loss), num_examples(num_examples), metrics(metrics) {}

  // Getters
  const float getLoss() const { return loss; }
  const int getNum_example() const { return num_examples; }
  const std::optional<Metrics> getMetrics() const { return metrics; }

  // Setters
  void setLoss(float f) { loss = f; }
  void setNum_example(int n) { num_examples = n; }
  void setMetrics(const Metrics &m) { metrics = m; }

private:
  float loss;
  int num_examples;
  std::optional<Metrics> metrics = std::nullopt;
};

typedef std::map<std::string, flwr_local::Scalar> Config;
typedef std::map<std::string, flwr_local::Scalar> Properties;

class PropertiesIns {
public:
  PropertiesIns() {}

  std::map<std::string, flwr_local::Scalar> getPropertiesIns() {
    return static_cast<std::map<std::string, flwr_local::Scalar>>(config);
  }

  void setPropertiesIns(const Config &c) { config = c; }

private:
  Config config;
};

class PropertiesRes {
public:
  PropertiesRes() {}

  Properties getPropertiesRes() { return properties; }

  void setPropertiesRes(const Properties &p) { properties = p; }

private:
  Properties properties;
};

struct Array {
  std::string dtype;
  std::vector<int32_t> shape;
  std::string stype;
  std::string data; // use string to represent bytes
};

// Wrapper to distinguish bytes from string in ConfigRecord variants
struct Bytes {
  std::string data;
};

using ArrayRecord = std::map<std::string, Array>;

using MetricRecordValue =
    std::variant<int64_t, uint64_t, double, std::vector<int64_t>,
                 std::vector<uint64_t>, std::vector<double>>;
using MetricRecord = std::map<std::string, MetricRecordValue>;

using ConfigRecordValue =
    std::variant<int64_t, uint64_t, double, bool, std::string, Bytes,
                 std::vector<int64_t>, std::vector<uint64_t>,
                 std::vector<double>, std::vector<bool>,
                 std::vector<std::string>, std::vector<Bytes>>;
using ConfigRecord = std::map<std::string, ConfigRecordValue>;

using RecordDictValue = std::variant<ArrayRecord, MetricRecord, ConfigRecord>;

class RecordDict {
public:
  RecordDict() = default;
  RecordDict(const std::map<std::string, RecordDictValue> &items)
      : _items(items) {}

  const std::map<std::string, RecordDictValue> &getItems() const {
    return _items;
  }
  void setItems(const std::map<std::string, RecordDictValue> &items) {
    _items = items;
  }

  // Convenience accessors that filter by type
  std::map<std::string, ArrayRecord> getArrayRecords() const {
    std::map<std::string, ArrayRecord> result;
    for (const auto &[key, value] : _items) {
      if (std::holds_alternative<ArrayRecord>(value)) {
        result[key] = std::get<ArrayRecord>(value);
      }
    }
    return result;
  }

  std::map<std::string, MetricRecord> getMetricRecords() const {
    std::map<std::string, MetricRecord> result;
    for (const auto &[key, value] : _items) {
      if (std::holds_alternative<MetricRecord>(value)) {
        result[key] = std::get<MetricRecord>(value);
      }
    }
    return result;
  }

  std::map<std::string, ConfigRecord> getConfigRecords() const {
    std::map<std::string, ConfigRecord> result;
    for (const auto &[key, value] : _items) {
      if (std::holds_alternative<ConfigRecord>(value)) {
        result[key] = std::get<ConfigRecord>(value);
      }
    }
    return result;
  }

private:
  std::map<std::string, RecordDictValue> _items;
};

struct Error {
  int64_t code = 0;
  std::string reason;
};

struct Metadata {
  uint64_t run_id = 0;
  std::string message_id;
  uint64_t src_node_id = 0;
  uint64_t dst_node_id = 0;
  std::string reply_to_message_id;
  std::string group_id;
  double ttl = 0.0;
  std::string message_type;
  double created_at = 0.0;
};

struct Message {
  Metadata metadata;
  std::optional<RecordDict> content;
  std::optional<Error> error;
};

} // namespace flwr_local
