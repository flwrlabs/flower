#include "recorddict_serde.h"

#include <stdexcept>

namespace flwr_quickstart {
namespace {

flwr::proto::Array array_to_proto(const flwr_local::Array &array) {
  flwr::proto::Array out;
  out.set_dtype(array.dtype);
  for (int32_t dim : array.shape) {
    out.add_shape(dim);
  }
  out.set_stype(array.stype);
  out.set_data(array.data);
  return out;
}

flwr_local::Array array_from_proto(const flwr::proto::Array &array) {
  flwr_local::Array out;
  out.dtype = array.dtype();
  out.shape.assign(array.shape().begin(), array.shape().end());
  out.stype = array.stype();
  out.data = array.data();
  return out;
}

flwr::proto::ArrayRecord
array_record_to_proto(const flwr_local::ParametersRecord &record) {
  flwr::proto::ArrayRecord out;
  for (const auto &[key, value] : record) {
    auto *item = out.add_items();
    item->set_key(key);
    *item->mutable_value() = array_to_proto(value);
  }
  return out;
}

flwr_local::ParametersRecord
array_record_from_proto(const flwr::proto::ArrayRecord &record) {
  flwr_local::ParametersRecord out;
  for (const auto &item : record.items()) {
    out[item.key()] = array_from_proto(item.value());
  }
  return out;
}

flwr::proto::MetricRecord
metric_record_to_proto(const flwr_local::MetricsRecord &record) {
  flwr::proto::MetricRecord out;
  for (const auto &[key, value] : record) {
    auto *item = out.add_items();
    item->set_key(key);
    auto *dst = item->mutable_value();
    if (std::holds_alternative<int>(value)) {
      dst->set_sint64(std::get<int>(value));
    } else if (std::holds_alternative<double>(value)) {
      dst->set_double_(std::get<double>(value));
    } else if (std::holds_alternative<std::vector<int>>(value)) {
      auto *list = dst->mutable_sint_list();
      for (int val : std::get<std::vector<int>>(value)) {
        list->add_vals(val);
      }
    } else if (std::holds_alternative<std::vector<double>>(value)) {
      auto *list = dst->mutable_double_list();
      for (double val : std::get<std::vector<double>>(value)) {
        list->add_vals(val);
      }
    }
  }
  return out;
}

flwr_local::MetricsRecord
metric_record_from_proto(const flwr::proto::MetricRecord &record) {
  flwr_local::MetricsRecord out;
  for (const auto &item : record.items()) {
    const auto &value = item.value();
    if (value.has_sint64()) {
      out[item.key()] = static_cast<int>(value.sint64());
    } else if (value.has_uint64()) {
      out[item.key()] = static_cast<int>(value.uint64());
    } else if (value.has_double_()) {
      out[item.key()] = value.double_();
    } else if (value.has_sint_list()) {
      std::vector<int> vals;
      for (auto val : value.sint_list().vals()) {
        vals.push_back(static_cast<int>(val));
      }
      out[item.key()] = vals;
    } else if (value.has_uint_list()) {
      std::vector<int> vals;
      for (auto val : value.uint_list().vals()) {
        vals.push_back(static_cast<int>(val));
      }
      out[item.key()] = vals;
    } else if (value.has_double_list()) {
      std::vector<double> vals(value.double_list().vals().begin(),
                               value.double_list().vals().end());
      out[item.key()] = vals;
    }
  }
  return out;
}

flwr::proto::ConfigRecord
config_record_to_proto(const flwr_local::ConfigsRecord &record) {
  flwr::proto::ConfigRecord out;
  for (const auto &[key, value] : record) {
    auto *item = out.add_items();
    item->set_key(key);
    auto *dst = item->mutable_value();
    if (std::holds_alternative<int>(value)) {
      dst->set_sint64(std::get<int>(value));
    } else if (std::holds_alternative<double>(value)) {
      dst->set_double_(std::get<double>(value));
    } else if (std::holds_alternative<std::string>(value)) {
      dst->set_string(std::get<std::string>(value));
    } else if (std::holds_alternative<bool>(value)) {
      dst->set_bool_(std::get<bool>(value));
    } else if (std::holds_alternative<std::vector<int>>(value)) {
      auto *list = dst->mutable_sint_list();
      for (int val : std::get<std::vector<int>>(value)) {
        list->add_vals(val);
      }
    } else if (std::holds_alternative<std::vector<double>>(value)) {
      auto *list = dst->mutable_double_list();
      for (double val : std::get<std::vector<double>>(value)) {
        list->add_vals(val);
      }
    } else if (std::holds_alternative<std::vector<std::string>>(value)) {
      auto *list = dst->mutable_string_list();
      for (const auto &val : std::get<std::vector<std::string>>(value)) {
        list->add_vals(val);
      }
    } else if (std::holds_alternative<std::vector<bool>>(value)) {
      auto *list = dst->mutable_bool_list();
      for (bool val : std::get<std::vector<bool>>(value)) {
        list->add_vals(val);
      }
    }
  }
  return out;
}

flwr_local::ConfigsRecord
config_record_from_proto(const flwr::proto::ConfigRecord &record) {
  flwr_local::ConfigsRecord out;
  for (const auto &item : record.items()) {
    const auto &value = item.value();
    if (value.has_sint64()) {
      out[item.key()] = static_cast<int>(value.sint64());
    } else if (value.has_uint64()) {
      out[item.key()] = static_cast<int>(value.uint64());
    } else if (value.has_double_()) {
      out[item.key()] = value.double_();
    } else if (value.has_bool_()) {
      out[item.key()] = value.bool_();
    } else if (value.has_string()) {
      out[item.key()] = value.string();
    } else if (value.has_bytes()) {
      out[item.key()] = value.bytes();
    } else if (value.has_sint_list()) {
      std::vector<int> vals;
      for (auto val : value.sint_list().vals()) {
        vals.push_back(static_cast<int>(val));
      }
      out[item.key()] = vals;
    } else if (value.has_uint_list()) {
      std::vector<int> vals;
      for (auto val : value.uint_list().vals()) {
        vals.push_back(static_cast<int>(val));
      }
      out[item.key()] = vals;
    } else if (value.has_double_list()) {
      std::vector<double> vals(value.double_list().vals().begin(),
                               value.double_list().vals().end());
      out[item.key()] = vals;
    } else if (value.has_string_list()) {
      std::vector<std::string> vals(value.string_list().vals().begin(),
                                    value.string_list().vals().end());
      out[item.key()] = vals;
    } else if (value.has_bool_list()) {
      std::vector<bool> vals;
      for (bool val : value.bool_list().vals()) {
        vals.push_back(val);
      }
      out[item.key()] = vals;
    }
  }
  return out;
}

flwr_local::Parameters
parameters_record_to_parameters(const flwr_local::ParametersRecord &record) {
  std::list<std::string> tensors;
  std::string tensor_type;
  for (const auto &[_, array] : record) {
    tensors.push_back(array.data);
    if (tensor_type.empty()) {
      tensor_type = array.stype;
    }
  }
  return flwr_local::Parameters(tensors, tensor_type);
}

flwr_local::ParametersRecord
parameters_to_parameters_record(const flwr_local::Parameters &parameters) {
  flwr_local::ParametersRecord out;
  int idx = 0;
  for (const auto &tensor : parameters.getTensors()) {
    flwr_local::Array array;
    array.dtype = parameters.getTensor_type();
    array.stype = parameters.getTensor_type();
    array.data = tensor;
    out[std::to_string(idx++)] = array;
  }
  return out;
}

flwr_local::Scalar scalar_from_config_value(const flwr_local::ConfigsRecord::mapped_type &value) {
  flwr_local::Scalar scalar;
  std::visit(
      [&scalar](auto &&arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, int>) {
          scalar.setInt(arg);
        } else if constexpr (std::is_same_v<T, double>) {
          scalar.setDouble(arg);
        } else if constexpr (std::is_same_v<T, std::string>) {
          scalar.setString(arg);
        } else if constexpr (std::is_same_v<T, bool>) {
          scalar.setBool(arg);
        }
      },
      value);
  return scalar;
}

flwr_local::ConfigsRecord metrics_to_config_record(const flwr_local::Metrics &metrics) {
  flwr_local::ConfigsRecord out;
  for (const auto &[key, value] : metrics) {
    flwr_local::Scalar scalar = value;
    if (scalar.getBool().has_value()) {
      out[key] = scalar.getBool().value();
    } else if (scalar.getDouble().has_value()) {
      out[key] = scalar.getDouble().value();
    } else if (scalar.getInt().has_value()) {
      out[key] = scalar.getInt().value();
    } else if (scalar.getString().has_value()) {
      out[key] = scalar.getString().value();
    } else if (scalar.getBytes().has_value()) {
      out[key] = scalar.getBytes().value();
    }
  }
  return out;
}

template <typename MapT>
const typename MapT::mapped_type &required_record(const MapT &records,
                                                  const std::string &key,
                                                  const std::string &kind) {
  const auto it = records.find(key);
  if (it == records.end()) {
    throw std::runtime_error("RecordSet missing required " + kind +
                             " record: " + key);
  }
  return it->second;
}

} // namespace

flwr_local::RecordSet
recorddict_from_proto(const flwr::proto::RecordDict &recorddict) {
  std::map<std::string, flwr_local::ParametersRecord> parameters_records;
  std::map<std::string, flwr_local::MetricsRecord> metrics_records;
  std::map<std::string, flwr_local::ConfigsRecord> configs_records;

  for (const auto &item : recorddict.items()) {
    if (item.has_array_record()) {
      parameters_records[item.key()] = array_record_from_proto(item.array_record());
    } else if (item.has_metric_record()) {
      metrics_records[item.key()] = metric_record_from_proto(item.metric_record());
    } else if (item.has_config_record()) {
      configs_records[item.key()] = config_record_from_proto(item.config_record());
    }
  }

  return flwr_local::RecordSet(parameters_records, metrics_records, configs_records);
}

flwr::proto::RecordDict
recorddict_to_proto(const flwr_local::RecordSet &recordset) {
  flwr::proto::RecordDict out;
  for (const auto &[key, value] : recordset.getParametersRecords()) {
    auto *item = out.add_items();
    item->set_key(key);
    *item->mutable_array_record() = array_record_to_proto(value);
  }
  for (const auto &[key, value] : recordset.getMetricsRecords()) {
    auto *item = out.add_items();
    item->set_key(key);
    *item->mutable_metric_record() = metric_record_to_proto(value);
  }
  for (const auto &[key, value] : recordset.getConfigsRecords()) {
    auto *item = out.add_items();
    item->set_key(key);
    *item->mutable_config_record() = config_record_to_proto(value);
  }
  return out;
}

flwr_local::FitIns recorddict_to_fit_ins(const flwr_local::RecordSet &recordset) {
  const auto &parameters_record =
      required_record(recordset.getParametersRecords(), "fitins.parameters",
                      "parameters");
  const auto &configs_record =
      required_record(recordset.getConfigsRecords(), "fitins.config", "config");
  flwr_local::Config config;
  for (const auto &[key, value] : configs_record) {
    config[key] = scalar_from_config_value(value);
  }
  return flwr_local::FitIns(parameters_record_to_parameters(parameters_record),
                            config);
}

flwr_local::EvaluateIns
recorddict_to_evaluate_ins(const flwr_local::RecordSet &recordset) {
  const auto &parameters_record =
      required_record(recordset.getParametersRecords(), "evaluateins.parameters",
                      "parameters");
  const auto &configs_record =
      required_record(recordset.getConfigsRecords(), "evaluateins.config",
                      "config");
  flwr_local::Config config;
  for (const auto &[key, value] : configs_record) {
    config[key] = scalar_from_config_value(value);
  }
  return flwr_local::EvaluateIns(
      parameters_record_to_parameters(parameters_record), config);
}

flwr_local::RecordSet
recorddict_from_get_parameters_res(const flwr_local::ParametersRes &parameters_res) {
  flwr_local::RecordSet out;
  out.setParametersRecords({{"getparametersres.parameters",
                             parameters_to_parameters_record(
                                 parameters_res.getParameters())}});
  out.setConfigsRecords({{"getparametersres.status",
                          {{"code", 0}, {"message", std::string("Success")}}}});
  return out;
}

flwr_local::RecordSet
recorddict_from_get_properties_res(const flwr_local::PropertiesRes &properties_res) {
  flwr_local::ConfigsRecord properties;
  flwr_local::PropertiesRes props = properties_res;
  for (const auto &[key, value] : props.getPropertiesRes()) {
    flwr_local::Scalar scalar = value;
    if (scalar.getBool().has_value()) {
      properties[key] = scalar.getBool().value();
    } else if (scalar.getDouble().has_value()) {
      properties[key] = scalar.getDouble().value();
    } else if (scalar.getInt().has_value()) {
      properties[key] = scalar.getInt().value();
    } else if (scalar.getString().has_value()) {
      properties[key] = scalar.getString().value();
    } else if (scalar.getBytes().has_value()) {
      properties[key] = scalar.getBytes().value();
    }
  }
  flwr_local::RecordSet out;
  out.setConfigsRecords({{"getpropertiesres.properties", properties},
                         {"getpropertiesres.status",
                          {{"code", 0}, {"message", std::string("Success")}}}});
  return out;
}

flwr_local::RecordSet recorddict_from_fit_res(const flwr_local::FitRes &fit_res) {
  flwr_local::RecordSet out;
  out.setParametersRecords({{"fitres.parameters",
                             parameters_to_parameters_record(
                                 fit_res.getParameters())}});
  out.setMetricsRecords({{"fitres.num_examples",
                          {{"num_examples", fit_res.getNum_example()}}}});
  flwr_local::ConfigsRecord metrics;
  if (fit_res.getMetrics().has_value()) {
    metrics = metrics_to_config_record(fit_res.getMetrics().value());
  }
  out.setConfigsRecords({{"fitres.metrics", metrics},
                         {"fitres.status",
                          {{"code", 0}, {"message", std::string("Success")}}}});
  return out;
}

flwr_local::RecordSet
recorddict_from_evaluate_res(const flwr_local::EvaluateRes &evaluate_res) {
  flwr_local::RecordSet out;
  out.setMetricsRecords({{"evaluateres.loss", {{"loss", evaluate_res.getLoss()}}},
                         {"evaluateres.num_examples",
                          {{"num_examples", evaluate_res.getNum_example()}}}});
  flwr_local::ConfigsRecord metrics;
  if (evaluate_res.getMetrics().has_value()) {
    metrics = metrics_to_config_record(evaluate_res.getMetrics().value());
  }
  out.setConfigsRecords({{"evaluateres.metrics", metrics},
                         {"evaluateres.status",
                          {{"code", 0}, {"message", std::string("Success")}}}});
  return out;
}

} // namespace flwr_quickstart
