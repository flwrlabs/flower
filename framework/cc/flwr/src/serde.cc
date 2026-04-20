#include "serde.h"
#include "flwr/proto/recorddict.pb.h"
#include "typing.h"

/**
 * Serialize client parameters to protobuf parameters message
 */
flwr::proto::Parameters parameters_to_proto(flwr_local::Parameters parameters) {
  flwr::proto::Parameters mp;
  mp.set_tensor_type(parameters.getTensor_type());

  for (auto &i : parameters.getTensors()) {
    mp.add_tensors(i);
  }
  return mp;
}

/**
 * Deserialize client protobuf parameters message to client parameters
 */
flwr_local::Parameters parameters_from_proto(flwr::proto::Parameters msg) {
  std::list<std::string> tensors;
  for (int i = 0; i < msg.tensors_size(); i++) {
    tensors.push_back(msg.tensors(i));
  }

  return flwr_local::Parameters(tensors, msg.tensor_type());
}

/**
 * Serialize client scalar type to protobuf scalar type
 */
flwr::proto::Scalar scalar_to_proto(flwr_local::Scalar scalar_msg) {
  flwr::proto::Scalar s;
  if (scalar_msg.getBool() != std::nullopt) {
    s.set_bool_(scalar_msg.getBool().value());
    return s;
  }
  if (scalar_msg.getBytes() != std::nullopt) {
    s.set_bytes(scalar_msg.getBytes().value());
    return s;
  }
  if (scalar_msg.getDouble() != std::nullopt) {
    s.set_double_(scalar_msg.getDouble().value());
    return s;
  }
  if (scalar_msg.getInt() != std::nullopt) {
    s.set_sint64(scalar_msg.getInt().value());
    return s;
  }
  if (scalar_msg.getString() != std::nullopt) {
    s.set_string(scalar_msg.getString().value());
    return s;
  } else {
    throw "Scalar to Proto failed";
  }
}

/**
 * Deserialize protobuf scalar type to client scalar type
 */
flwr_local::Scalar scalar_from_proto(flwr::proto::Scalar scalar_msg) {
  flwr_local::Scalar scalar;
  switch (scalar_msg.scalar_case()) {
  case 1:
    scalar.setDouble(scalar_msg.double_());
    return scalar;
  case 8:
    scalar.setInt(scalar_msg.sint64());
    return scalar;
  case 13:
    scalar.setBool(scalar_msg.bool_());
    return scalar;
  case 14:
    scalar.setString(scalar_msg.string());
    return scalar;
  case 15:
    scalar.setBytes(scalar_msg.bytes());
    return scalar;
  case 6:
    scalar.setDouble(scalar_msg.uint64());
    return scalar;
  case 0:
    break;
  }
  throw "Error scalar type";
}

/**
 * Serialize client metrics type to protobuf metrics type
 * "Any" is used in Python, this part might be changed if needed
 */
google::protobuf::Map<std::string, flwr::proto::Scalar>
metrics_to_proto(flwr_local::Metrics metrics) {
  google::protobuf::Map<std::string, flwr::proto::Scalar> proto;

  for (auto &[key, value] : metrics) {
    proto[key] = scalar_to_proto(value);
  }

  return proto;
}

/**
 * Deserialize protobuf metrics type to client metrics type
 * "Any" is used in Python, this part might be changed if needed
 */
flwr_local::Metrics metrics_from_proto(
    google::protobuf::Map<std::string, flwr::proto::Scalar> proto) {
  flwr_local::Metrics metrics;

  for (auto &[key, value] : proto) {
    metrics[key] = scalar_from_proto(value);
  }
  return metrics;
}

/**
 * Serialize client ParametersRes type to protobuf ParametersRes type
 */
flwr::proto::ClientMessage_GetParametersRes
parameters_res_to_proto(flwr_local::ParametersRes res) {
  flwr::proto::Parameters mp = parameters_to_proto(res.getParameters());
  flwr::proto::ClientMessage_GetParametersRes cpr;
  *(cpr.mutable_parameters()) = mp;
  return cpr;
}

/**
 * Deserialize protobuf FitIns type to client FitIns type
 */
flwr_local::FitIns fit_ins_from_proto(flwr::proto::ServerMessage_FitIns msg) {
  flwr_local::Parameters parameters = parameters_from_proto(msg.parameters());
  flwr_local::Metrics config = metrics_from_proto(msg.config());
  return flwr_local::FitIns(parameters, config);
}

/**
 * Serialize client FitRes type to protobuf FitRes type
 */
flwr::proto::ClientMessage_FitRes fit_res_to_proto(flwr_local::FitRes res) {
  flwr::proto::ClientMessage_FitRes cres;

  flwr::proto::Parameters parameters_proto =
      parameters_to_proto(res.getParameters());
  google::protobuf::Map<::std::string, ::flwr::proto::Scalar> metrics_msg;
  if (res.getMetrics() != std::nullopt) {
    metrics_msg = metrics_to_proto(res.getMetrics().value());
  }

  // Forward - compatible case
  *(cres.mutable_parameters()) = parameters_proto;
  cres.set_num_examples(res.getNum_example());
  if (!metrics_msg.empty()) {
    *cres.mutable_metrics() = metrics_msg;
  }
  return cres;
}

/**
 * Deserialize protobuf EvaluateIns type to client EvaluateIns type
 */
flwr_local::EvaluateIns
evaluate_ins_from_proto(flwr::proto::ServerMessage_EvaluateIns msg) {
  flwr_local::Parameters parameters = parameters_from_proto(msg.parameters());
  flwr_local::Metrics config = metrics_from_proto(msg.config());
  return flwr_local::EvaluateIns(parameters, config);
}

/**
 * Serialize client EvaluateRes type to protobuf EvaluateRes type
 */
flwr::proto::ClientMessage_EvaluateRes
evaluate_res_to_proto(flwr_local::EvaluateRes res) {
  flwr::proto::ClientMessage_EvaluateRes cres;
  google::protobuf::Map<::std::string, ::flwr::proto::Scalar> metrics_msg;
  if (res.getMetrics() != std::nullopt) {
    metrics_msg = metrics_to_proto(res.getMetrics().value());
  }
  // Forward - compatible case
  cres.set_loss(res.getLoss());
  cres.set_num_examples(res.getNum_example());
  if (!metrics_msg.empty()) {
    auto &map = *cres.mutable_metrics();

    for (auto &[key, value] : metrics_msg) {
      map[key] = value;
    }
  }

  return cres;
}

flwr::proto::Array array_to_proto(const flwr_local::Array &array) {
  flwr::proto::Array protoArray;
  protoArray.set_dtype(array.dtype);
  for (int32_t dim : array.shape) {
    protoArray.add_shape(dim);
  }
  protoArray.set_stype(array.stype);
  protoArray.set_data({array.data.begin(), array.data.end()});
  return protoArray;
}

flwr_local::Array array_from_proto(const flwr::proto::Array &protoArray) {
  flwr_local::Array array;
  array.dtype = protoArray.dtype();
  array.shape.assign(protoArray.shape().begin(), protoArray.shape().end());
  array.stype = protoArray.stype();

  const std::string &protoData = protoArray.data();
  array.data.assign(protoData.begin(), protoData.end());

  return array;
}

flwr::proto::ArrayRecord
parameters_record_to_proto(const flwr_local::ParametersRecord &record) {
  flwr::proto::ArrayRecord protoRecord;
  for (const auto &[key, value] : record) {
    *protoRecord.add_data_keys() = key;
    *protoRecord.add_data_values() = array_to_proto(value);
  }
  return protoRecord;
}

flwr_local::ParametersRecord
parameters_record_from_proto(const flwr::proto::ArrayRecord &protoRecord) {
  flwr_local::ParametersRecord record;

  auto keys = protoRecord.data_keys();
  auto values = protoRecord.data_values();
  for (size_t i = 0; i < keys.size(); ++i) {
    record[keys[i]] = array_from_proto(values[i]);
  }
  return record;
}

flwr::proto::MetricRecord
metrics_record_to_proto(const flwr_local::MetricsRecord &record) {
  flwr::proto::MetricRecord protoRecord;

  for (const auto &[key, value] : record) {
    auto &data = (*protoRecord.mutable_data())[key];

    if (std::holds_alternative<int>(value)) {
      data.set_sint64(std::get<int>(value));
    } else if (std::holds_alternative<double>(value)) {
      data.set_double_(std::get<double>(value));
    } else if (std::holds_alternative<std::vector<int>>(value)) {
      auto &int_list = std::get<std::vector<int>>(value);
      auto *list = data.mutable_sint_list();
      for (int val : int_list) {
        list->add_vals(val);
      }
    } else if (std::holds_alternative<std::vector<double>>(value)) {
      auto &double_list = std::get<std::vector<double>>(value);
      auto *list = data.mutable_double_list();
      for (double val : double_list) {
        list->add_vals(val);
      }
    }
  }

  return protoRecord;
}

flwr_local::MetricsRecord
metrics_record_from_proto(const flwr::proto::MetricRecord &protoRecord) {
  flwr_local::MetricsRecord record;

  for (const auto &[key, value] : protoRecord.data()) {
    if (value.has_sint64()) {
      record[key] = (int)value.sint64();
    } else if (value.has_double_()) {
      record[key] = (double)value.double_();
    } else if (value.has_sint_list()) {
      std::vector<int> int_list;
      for (const auto sint : value.sint_list().vals()) {
        int_list.push_back((int)sint);
      }
      record[key] = int_list;
    } else if (value.has_double_list()) {
      std::vector<double> double_list;
      for (const auto proto_double : value.double_list().vals()) {
        double_list.push_back((double)proto_double);
      }
      record[key] = double_list;
    }
  }
  return record;
}

flwr::proto::ConfigRecord
configs_record_to_proto(const flwr_local::ConfigsRecord &record) {
  flwr::proto::ConfigRecord protoRecord;

  for (const auto &[key, value] : record) {
    auto &data = (*protoRecord.mutable_data())[key];

    if (std::holds_alternative<int>(value)) {
      data.set_sint64(std::get<int>(value));
    } else if (std::holds_alternative<double>(value)) {
      data.set_double_(std::get<double>(value));
    } else if (std::holds_alternative<bool>(value)) {
      data.set_bool_(std::get<bool>(value));
    } else if (std::holds_alternative<std::string>(value)) {
      data.set_string(std::get<std::string>(value));
    } else if (std::holds_alternative<std::vector<int>>(value)) {
      auto &list = *data.mutable_sint_list();
      for (int val : std::get<std::vector<int>>(value)) {
        list.add_vals(val);
      }
    } else if (std::holds_alternative<std::vector<double>>(value)) {
      auto &list = *data.mutable_double_list();
      for (double val : std::get<std::vector<double>>(value)) {
        list.add_vals(val);
      }
    } else if (std::holds_alternative<std::vector<bool>>(value)) {
      auto &list = *data.mutable_bool_list();
      for (bool val : std::get<std::vector<bool>>(value)) {
        list.add_vals(val);
      }
    } else if (std::holds_alternative<std::vector<std::string>>(value)) {
      auto &list = *data.mutable_string_list();
      for (const auto &val : std::get<std::vector<std::string>>(value)) {
        list.add_vals(val);
      }
    }
  }

  return protoRecord;
}

flwr_local::ConfigsRecord
configs_record_from_proto(const flwr::proto::ConfigRecord &protoRecord) {
  flwr_local::ConfigsRecord record;

  for (const auto &[key, value] : protoRecord.data()) {
    if (value.has_sint_list()) {
      std::vector<int> int_list;
      for (const auto sint : value.sint_list().vals()) {
        int_list.push_back((int)sint);
      }
      record[key] = int_list;
    } else if (value.has_double_list()) {
      std::vector<double> double_list;
      for (const auto proto_double : value.double_list().vals()) {
        double_list.push_back((double)proto_double);
      }
      record[key] = double_list;
    } else if (value.has_bool_list()) {
      std::vector<bool> tmp_list;
      for (const auto proto_val : value.bool_list().vals()) {
        tmp_list.push_back((bool)proto_val);
      }
      record[key] = tmp_list;
    } else if (value.has_bytes_list()) {
      std::vector<std::string> tmp_list;
      for (const auto proto_val : value.bytes_list().vals()) {
        tmp_list.push_back(proto_val);
      }
      record[key] = tmp_list;
    } else if (value.has_string_list()) {
      std::vector<std::string> tmp_list;
      for (const auto proto_val : value.bytes_list().vals()) {
        tmp_list.push_back(proto_val);
      }
      record[key] = tmp_list;
    } else if (value.has_sint64()) {
      record[key] = (int)value.sint64();
    } else if (value.has_double_()) {
      record[key] = (double)value.double_();
    } else if (value.has_bool_()) {
      record[key] = value.bool_();
    } else if (value.has_bytes()) {
      record[key] = value.bytes();
    } else if (value.has_string()) {
      record[key] = value.string();
    }
  }
  return record;
}

flwr_local::Parameters
parametersrecord_to_parameters(const flwr_local::ParametersRecord &record,
                               bool keep_input) {
  std::list<std::string> tensors;
  std::string tensor_type;

  for (const auto &[key, array] : record) {
    tensors.push_back(array.data);

    if (tensor_type.empty()) {
      tensor_type = array.stype;
    }
  }

  return flwr_local::Parameters(tensors, tensor_type);
}

flwr_local::EvaluateIns
recordset_to_evaluate_ins(const flwr_local::RecordSet &recordset,
                          bool keep_input) {
  auto parameters_record =
      recordset.getParametersRecords().at("evaluateins.parameters");

  flwr_local::Parameters params =
      parametersrecord_to_parameters(parameters_record, keep_input);

  auto configs_record = recordset.getConfigsRecords().at("evaluateins.config");
  flwr_local::Config config_dict;

  for (const auto &[key, value] : configs_record) {
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
          } else if constexpr (std::is_same_v<T, std::vector<int>>) {
          } else if constexpr (std::is_same_v<T, std::vector<double>>) {
          } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
          } else if constexpr (std::is_same_v<T, std::vector<bool>>) {
          }
        },
        value);

    config_dict[key] = scalar;
  }

  return flwr_local::EvaluateIns(params, config_dict);
}

flwr_local::ConfigsRecord
metrics_to_config_record(const flwr_local::Metrics metrics) {
  flwr_local::ConfigsRecord config_record;
  for (const auto &[key, value] : metrics) {
    flwr_local::Scalar scalar_value = value;
    if (scalar_value.getBool().has_value()) {
      config_record[key] = scalar_value.getBool().value();
    } else if (scalar_value.getBytes().has_value()) {
      config_record[key] = scalar_value.getBytes().value();
    } else if (scalar_value.getDouble().has_value()) {
      config_record[key] = scalar_value.getDouble().value();
    } else if (scalar_value.getInt().has_value()) {
      config_record[key] = scalar_value.getInt().value();
    } else if (scalar_value.getString().has_value()) {
      config_record[key] = scalar_value.getString().value();
    } else {
      config_record[key] = "";
    }
  }
  return config_record;
}

flwr_local::FitIns recordset_to_fit_ins(const flwr_local::RecordSet &recordset,
                                        bool keep_input) {
  auto parameters_record =
      recordset.getParametersRecords().at("fitins.parameters");

  flwr_local::Parameters params =
      parametersrecord_to_parameters(parameters_record, keep_input);

  auto configs_record = recordset.getConfigsRecords().at("fitins.config");
  flwr_local::Config config_dict;

  for (const auto &[key, value] : configs_record) {
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
          } else if constexpr (std::is_same_v<T, std::vector<int>>) {
          } else if constexpr (std::is_same_v<T, std::vector<double>>) {
          } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
          } else if constexpr (std::is_same_v<T, std::vector<bool>>) {
          }
        },
        value);

    config_dict[key] = scalar;
  }

  return flwr_local::FitIns(params, config_dict);
}

flwr_local::ParametersRecord
parameters_to_parametersrecord(const flwr_local::Parameters &parameters) {
  flwr_local::ParametersRecord record;
  const std::list<std::string> tensors = parameters.getTensors();
  const std::string tensor_type = parameters.getTensor_type();

  int idx = 0;
  for (const auto &tensor : tensors) {
    flwr_local::Array array{tensor_type, std::vector<int32_t>(), tensor_type,
                            tensor};
    record[std::to_string(idx++)] = array;
  }

  return record;
}

flwr_local::RecordSet recordset_from_get_parameters_res(
    const flwr_local::ParametersRes &get_parameters_res) {
  std::map<std::string, flwr_local::ParametersRecord> parameters_record = {
      {"getparametersres.parameters",
       parameters_to_parametersrecord(get_parameters_res.getParameters())}};

  std::map<std::string, flwr_local::ConfigsRecord> configs_record = {
      {"getparametersres.status", {{"code", 0}, {"message", "Success"}}}};

  flwr_local::RecordSet recordset = flwr_local::RecordSet();

  recordset.setParametersRecords(parameters_record);
  recordset.setConfigsRecords(configs_record);

  return recordset;
}

flwr_local::RecordSet recordset_from_fit_res(const flwr_local::FitRes &fitres) {
  std::map<std::string, flwr_local::ParametersRecord> parameters_record = {
      {"fitres.parameters",
       parameters_to_parametersrecord(fitres.getParameters())}};

  std::map<std::string, flwr_local::MetricsRecord> metrics_record = {
      {"fitres.num_examples", {{"num_examples", fitres.getNum_example()}}}};

  std::map<std::string, flwr_local::ConfigsRecord> configs_record = {
      {"fitres.status", {{"code", 0}, {"message", "Success"}}}};

  if (fitres.getMetrics() != std::nullopt) {
    configs_record["fitres.metrics"] =
        metrics_to_config_record(fitres.getMetrics().value());
  } else {
    configs_record["fitres.metrics"] = {};
  }
  flwr_local::RecordSet recordset = flwr_local::RecordSet();

  recordset.setParametersRecords(parameters_record);
  recordset.setMetricsRecords(metrics_record);
  recordset.setConfigsRecords(configs_record);

  return recordset;
}

flwr_local::RecordSet
recordset_from_evaluate_res(const flwr_local::EvaluateRes &evaluate_res) {
  std::map<std::string, flwr_local::MetricsRecord> metrics_record = {
      {"evaluateres.loss", {{"loss", evaluate_res.getLoss()}}},
      {"evaluateres.num_examples",
       {{"num_examples", evaluate_res.getNum_example()}}}};

  std::map<std::string, flwr_local::ConfigsRecord> configs_record = {
      {"evaluateres.status", {{"code", 0}, {"message", "Success"}}}};

  if (evaluate_res.getMetrics() != std::nullopt) {
    configs_record["evaluateres.metrics"] =
        metrics_to_config_record(evaluate_res.getMetrics().value());
  } else {
    configs_record["evaluateres.metrics"] = {};
  }

  flwr_local::RecordSet recordset = flwr_local::RecordSet();

  recordset.setMetricsRecords(metrics_record);
  recordset.setConfigsRecords(configs_record);

  return recordset;
}

flwr_local::RecordSet
recordset_from_proto(const flwr::proto::RecordDict &recordset) {

  std::map<std::string, flwr_local::ParametersRecord> parametersRecords;
  std::map<std::string, flwr_local::MetricsRecord> metricsRecords;
  std::map<std::string, flwr_local::ConfigsRecord> configsRecords;

  for (const auto &[key, param_record] : recordset.arrays()) {
    parametersRecords[key] = parameters_record_from_proto(param_record);
  }

  for (const auto &[key, metrics_record] : recordset.metrics()) {
    metricsRecords[key] = metrics_record_from_proto(metrics_record);
  }

  for (const auto &[key, configs_record] : recordset.configs()) {
    configsRecords[key] = configs_record_from_proto(configs_record);
  }

  return flwr_local::RecordSet(parametersRecords, metricsRecords,
                               configsRecords);
}

flwr::proto::RecordDict
recordset_to_proto(const flwr_local::RecordSet &recordset) {
  flwr::proto::RecordDict proto_recordset;

  for (const auto &[key, param_record] : recordset.getParametersRecords()) {
    (*(proto_recordset.mutable_arrays()))[key] =
        parameters_record_to_proto(param_record);
  }

  for (const auto &[key, metrics_record] : recordset.getMetricsRecords()) {
    (*(proto_recordset.mutable_metrics()))[key] =
        metrics_record_to_proto(metrics_record);
  }

  for (const auto &[key, configs_record] : recordset.getConfigsRecords()) {
    (*(proto_recordset.mutable_configs()))[key] =
        configs_record_to_proto(configs_record);
  }

  return proto_recordset;
}
