#include "serde.h"
#include "flwr/proto/message.pb.h"
#include "flwr/proto/recorddict.pb.h"
#include "typing.h"

#include <openssl/sha.h>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>

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
  case 6:
    scalar.setInt(static_cast<int>(scalar_msg.uint64()));
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
  case 0:
    break;
  }
  throw "Error scalar type";
}

/**
 * Serialize client metrics type to protobuf metrics type
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
 */
flwr_local::Metrics metrics_from_proto(
    google::protobuf::Map<std::string, flwr::proto::Scalar> proto) {
  flwr_local::Metrics metrics;

  for (auto &[key, value] : proto) {
    metrics[key] = scalar_from_proto(value);
  }
  return metrics;
}

///////////////////////////////////////////////////////////////////////////////
// Array serde
///////////////////////////////////////////////////////////////////////////////

flwr::proto::Array array_to_proto(const flwr_local::Array &array) {
  flwr::proto::Array proto;
  proto.set_dtype(array.dtype);
  for (int32_t dim : array.shape) {
    proto.add_shape(dim);
  }
  proto.set_stype(array.stype);
  proto.set_data({array.data.begin(), array.data.end()});
  return proto;
}

flwr_local::Array array_from_proto(const flwr::proto::Array &proto) {
  flwr_local::Array array;
  array.dtype = proto.dtype();
  array.shape.assign(proto.shape().begin(), proto.shape().end());
  array.stype = proto.stype();
  const std::string &data = proto.data();
  array.data.assign(data.begin(), data.end());
  return array;
}

///////////////////////////////////////////////////////////////////////////////
// ArrayRecord serde (items-based)
///////////////////////////////////////////////////////////////////////////////

flwr::proto::ArrayRecord
array_record_to_proto(const flwr_local::ArrayRecord &record) {
  flwr::proto::ArrayRecord proto;
  for (const auto &[key, value] : record) {
    auto *item = proto.add_items();
    item->set_key(key);
    *item->mutable_value() = array_to_proto(value);
  }
  return proto;
}

flwr_local::ArrayRecord
array_record_from_proto(const flwr::proto::ArrayRecord &proto) {
  flwr_local::ArrayRecord record;
  for (const auto &item : proto.items()) {
    record[item.key()] = array_from_proto(item.value());
  }
  return record;
}

///////////////////////////////////////////////////////////////////////////////
// MetricRecord serde (items-based)
///////////////////////////////////////////////////////////////////////////////

flwr::proto::MetricRecord
metric_record_to_proto(const flwr_local::MetricRecord &record) {
  flwr::proto::MetricRecord proto;

  for (const auto &[key, value] : record) {
    auto *item = proto.add_items();
    item->set_key(key);

    std::visit(
        [&](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, int64_t>) {
            item->mutable_value()->set_sint64(arg);
          } else if constexpr (std::is_same_v<T, uint64_t>) {
            item->mutable_value()->set_uint64(arg);
          } else if constexpr (std::is_same_v<T, double>) {
            item->mutable_value()->set_double_(arg);
          } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
            auto *list = item->mutable_value()->mutable_sint_list();
            for (auto v : arg)
              list->add_vals(v);
          } else if constexpr (std::is_same_v<T, std::vector<uint64_t>>) {
            auto *list = item->mutable_value()->mutable_uint_list();
            for (auto v : arg)
              list->add_vals(v);
          } else if constexpr (std::is_same_v<T, std::vector<double>>) {
            auto *list = item->mutable_value()->mutable_double_list();
            for (auto v : arg)
              list->add_vals(v);
          }
        },
        value);
  }

  return proto;
}

flwr_local::MetricRecord
metric_record_from_proto(const flwr::proto::MetricRecord &proto) {
  flwr_local::MetricRecord record;

  for (const auto &item : proto.items()) {
    const auto &value = item.value();
    switch (value.value_case()) {
    case flwr::proto::MetricRecordValue::kSint64:
      record[item.key()] = value.sint64();
      break;
    case flwr::proto::MetricRecordValue::kUint64:
      record[item.key()] = value.uint64();
      break;
    case flwr::proto::MetricRecordValue::kDouble:
      record[item.key()] = value.double_();
      break;
    case flwr::proto::MetricRecordValue::kSintList: {
      std::vector<int64_t> vals;
      for (const auto v : value.sint_list().vals())
        vals.push_back(v);
      record[item.key()] = vals;
      break;
    }
    case flwr::proto::MetricRecordValue::kUintList: {
      std::vector<uint64_t> vals;
      for (const auto v : value.uint_list().vals())
        vals.push_back(v);
      record[item.key()] = vals;
      break;
    }
    case flwr::proto::MetricRecordValue::kDoubleList: {
      std::vector<double> vals;
      for (const auto v : value.double_list().vals())
        vals.push_back(v);
      record[item.key()] = vals;
      break;
    }
    default:
      break;
    }
  }
  return record;
}

///////////////////////////////////////////////////////////////////////////////
// ConfigRecord serde (items-based)
///////////////////////////////////////////////////////////////////////////////

flwr::proto::ConfigRecord
config_record_to_proto(const flwr_local::ConfigRecord &record) {
  flwr::proto::ConfigRecord proto;

  for (const auto &[key, value] : record) {
    auto *item = proto.add_items();
    item->set_key(key);

    std::visit(
        [&](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, int64_t>) {
            item->mutable_value()->set_sint64(arg);
          } else if constexpr (std::is_same_v<T, uint64_t>) {
            item->mutable_value()->set_uint64(arg);
          } else if constexpr (std::is_same_v<T, double>) {
            item->mutable_value()->set_double_(arg);
          } else if constexpr (std::is_same_v<T, bool>) {
            item->mutable_value()->set_bool_(arg);
          } else if constexpr (std::is_same_v<T, std::string>) {
            item->mutable_value()->set_string(arg);
          } else if constexpr (std::is_same_v<T, flwr_local::Bytes>) {
            item->mutable_value()->set_bytes(arg.data);
          } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
            auto *list = item->mutable_value()->mutable_sint_list();
            for (auto v : arg)
              list->add_vals(v);
          } else if constexpr (std::is_same_v<T, std::vector<uint64_t>>) {
            auto *list = item->mutable_value()->mutable_uint_list();
            for (auto v : arg)
              list->add_vals(v);
          } else if constexpr (std::is_same_v<T, std::vector<double>>) {
            auto *list = item->mutable_value()->mutable_double_list();
            for (auto v : arg)
              list->add_vals(v);
          } else if constexpr (std::is_same_v<T, std::vector<bool>>) {
            auto *list = item->mutable_value()->mutable_bool_list();
            for (bool v : arg)
              list->add_vals(v);
          } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
            auto *list = item->mutable_value()->mutable_string_list();
            for (const auto &v : arg)
              list->add_vals(v);
          } else if constexpr (std::is_same_v<T,
                                              std::vector<flwr_local::Bytes>>) {
            auto *list = item->mutable_value()->mutable_bytes_list();
            for (const auto &v : arg)
              list->add_vals(v.data);
          }
        },
        value);
  }

  return proto;
}

flwr_local::ConfigRecord
config_record_from_proto(const flwr::proto::ConfigRecord &proto) {
  flwr_local::ConfigRecord record;

  for (const auto &item : proto.items()) {
    const auto &value = item.value();
    switch (value.value_case()) {
    case flwr::proto::ConfigRecordValue::kSint64:
      record[item.key()] = value.sint64();
      break;
    case flwr::proto::ConfigRecordValue::kUint64:
      record[item.key()] = value.uint64();
      break;
    case flwr::proto::ConfigRecordValue::kDouble:
      record[item.key()] = value.double_();
      break;
    case flwr::proto::ConfigRecordValue::kBool:
      record[item.key()] = value.bool_();
      break;
    case flwr::proto::ConfigRecordValue::kString:
      record[item.key()] = value.string();
      break;
    case flwr::proto::ConfigRecordValue::kBytes:
      record[item.key()] = flwr_local::Bytes{value.bytes()};
      break;
    case flwr::proto::ConfigRecordValue::kSintList: {
      std::vector<int64_t> vals;
      for (const auto v : value.sint_list().vals())
        vals.push_back(v);
      record[item.key()] = vals;
      break;
    }
    case flwr::proto::ConfigRecordValue::kUintList: {
      std::vector<uint64_t> vals;
      for (const auto v : value.uint_list().vals())
        vals.push_back(v);
      record[item.key()] = vals;
      break;
    }
    case flwr::proto::ConfigRecordValue::kDoubleList: {
      std::vector<double> vals;
      for (const auto v : value.double_list().vals())
        vals.push_back(v);
      record[item.key()] = vals;
      break;
    }
    case flwr::proto::ConfigRecordValue::kBoolList: {
      std::vector<bool> vals;
      for (const auto v : value.bool_list().vals())
        vals.push_back(v);
      record[item.key()] = vals;
      break;
    }
    case flwr::proto::ConfigRecordValue::kStringList: {
      std::vector<std::string> vals;
      for (const auto &v : value.string_list().vals())
        vals.push_back(v);
      record[item.key()] = vals;
      break;
    }
    case flwr::proto::ConfigRecordValue::kBytesList: {
      std::vector<flwr_local::Bytes> vals;
      for (const auto &v : value.bytes_list().vals())
        vals.push_back(flwr_local::Bytes{v});
      record[item.key()] = vals;
      break;
    }
    default:
      break;
    }
  }
  return record;
}

///////////////////////////////////////////////////////////////////////////////
// RecordDict serde
///////////////////////////////////////////////////////////////////////////////

flwr::proto::RecordDict
recorddict_to_proto(const flwr_local::RecordDict &rd) {
  flwr::proto::RecordDict proto;

  for (const auto &[key, value] : rd.getItems()) {
    auto *item = proto.add_items();
    item->set_key(key);

    std::visit(
        [&](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, flwr_local::ArrayRecord>) {
            *item->mutable_array_record() = array_record_to_proto(arg);
          } else if constexpr (std::is_same_v<T, flwr_local::MetricRecord>) {
            *item->mutable_metric_record() = metric_record_to_proto(arg);
          } else if constexpr (std::is_same_v<T, flwr_local::ConfigRecord>) {
            *item->mutable_config_record() = config_record_to_proto(arg);
          }
        },
        value);
  }

  return proto;
}

flwr_local::RecordDict
recorddict_from_proto(const flwr::proto::RecordDict &proto) {
  std::map<std::string, flwr_local::RecordDictValue> items;

  for (const auto &item : proto.items()) {
    switch (item.value_case()) {
    case flwr::proto::RecordDict_Item::kArrayRecord:
      items[item.key()] = array_record_from_proto(item.array_record());
      break;
    case flwr::proto::RecordDict_Item::kMetricRecord:
      items[item.key()] = metric_record_from_proto(item.metric_record());
      break;
    case flwr::proto::RecordDict_Item::kConfigRecord:
      items[item.key()] = config_record_from_proto(item.config_record());
      break;
    default:
      break;
    }
  }

  return flwr_local::RecordDict(items);
}

///////////////////////////////////////////////////////////////////////////////
// Metadata serde
///////////////////////////////////////////////////////////////////////////////

flwr::proto::Metadata metadata_to_proto(const flwr_local::Metadata &meta) {
  flwr::proto::Metadata proto;
  proto.set_run_id(meta.run_id);
  proto.set_message_id(meta.message_id);
  proto.set_src_node_id(meta.src_node_id);
  proto.set_dst_node_id(meta.dst_node_id);
  proto.set_reply_to_message_id(meta.reply_to_message_id);
  proto.set_group_id(meta.group_id);
  proto.set_ttl(meta.ttl);
  proto.set_message_type(meta.message_type);
  proto.set_created_at(meta.created_at);
  return proto;
}

flwr_local::Metadata metadata_from_proto(const flwr::proto::Metadata &proto) {
  flwr_local::Metadata meta;
  meta.run_id = proto.run_id();
  meta.message_id = proto.message_id();
  meta.src_node_id = proto.src_node_id();
  meta.dst_node_id = proto.dst_node_id();
  meta.reply_to_message_id = proto.reply_to_message_id();
  meta.group_id = proto.group_id();
  meta.ttl = proto.ttl();
  meta.message_type = proto.message_type();
  meta.created_at = proto.created_at();
  return meta;
}

///////////////////////////////////////////////////////////////////////////////
// Message serde
///////////////////////////////////////////////////////////////////////////////

flwr::proto::Message message_to_proto(const flwr_local::Message &msg) {
  flwr::proto::Message proto;
  *proto.mutable_metadata() = metadata_to_proto(msg.metadata);
  if (msg.content) {
    *proto.mutable_content() = recorddict_to_proto(*msg.content);
  }
  if (msg.error) {
    proto.mutable_error()->set_code(msg.error->code);
    proto.mutable_error()->set_reason(msg.error->reason);
  }
  return proto;
}

flwr_local::Message message_from_proto(const flwr::proto::Message &proto) {
  flwr_local::Message msg;
  msg.metadata = metadata_from_proto(proto.metadata());
  if (proto.has_content()) {
    msg.content = recorddict_from_proto(proto.content());
  }
  if (proto.has_error()) {
    msg.error =
        flwr_local::Error{proto.error().code(), proto.error().reason()};
  }
  return msg;
}

///////////////////////////////////////////////////////////////////////////////
// Legacy type conversions (RecordDict <-> FitIns/FitRes/EvaluateIns/etc.)
///////////////////////////////////////////////////////////////////////////////

flwr_local::Parameters
parametersrecord_to_parameters(const flwr_local::ArrayRecord &record,
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

flwr_local::ArrayRecord
parameters_to_parametersrecord(const flwr_local::Parameters &parameters) {
  flwr_local::ArrayRecord record;
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

flwr_local::ConfigRecord
metrics_to_config_record(const flwr_local::Metrics metrics) {
  flwr_local::ConfigRecord config_record;
  for (const auto &[key, value] : metrics) {
    flwr_local::Scalar scalar_value = value;
    if (scalar_value.getBool().has_value()) {
      config_record[key] = scalar_value.getBool().value();
    } else if (scalar_value.getBytes().has_value()) {
      config_record[key] = flwr_local::Bytes{scalar_value.getBytes().value()};
    } else if (scalar_value.getDouble().has_value()) {
      config_record[key] = scalar_value.getDouble().value();
    } else if (scalar_value.getInt().has_value()) {
      config_record[key] = static_cast<int64_t>(scalar_value.getInt().value());
    } else if (scalar_value.getString().has_value()) {
      config_record[key] = scalar_value.getString().value();
    } else {
      config_record[key] = std::string("");
    }
  }
  return config_record;
}

flwr_local::FitIns recorddict_to_fit_ins(const flwr_local::RecordDict &rd,
                                         bool keep_input) {
  auto array_records = rd.getArrayRecords();
  auto config_records = rd.getConfigRecords();

  std::cerr << "[DEBUG] recorddict_to_fit_ins: array_records keys:";
  for (const auto &[k, v] : array_records)
    std::cerr << " '" << k << "'";
  std::cerr << "; config_records keys:";
  for (const auto &[k, v] : config_records)
    std::cerr << " '" << k << "'";
  std::cerr << "; all items keys:";
  for (const auto &[k, v] : rd.getItems())
    std::cerr << " '" << k << "'";
  std::cerr << std::endl;

  auto parameters_record = array_records.at("fitins.parameters");
  flwr_local::Parameters params =
      parametersrecord_to_parameters(parameters_record, keep_input);

  auto configs_record = config_records.at("fitins.config");
  flwr_local::Config config_dict;

  for (const auto &[key, value] : configs_record) {
    flwr_local::Scalar scalar;

    std::visit(
        [&scalar](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, int64_t>) {
            scalar.setInt(static_cast<int>(arg));
          } else if constexpr (std::is_same_v<T, double>) {
            scalar.setDouble(arg);
          } else if constexpr (std::is_same_v<T, std::string>) {
            scalar.setString(arg);
          } else if constexpr (std::is_same_v<T, bool>) {
            scalar.setBool(arg);
          }
        },
        value);

    config_dict[key] = scalar;
  }

  return flwr_local::FitIns(params, config_dict);
}

flwr_local::EvaluateIns
recorddict_to_evaluate_ins(const flwr_local::RecordDict &rd, bool keep_input) {
  auto array_records = rd.getArrayRecords();
  auto config_records = rd.getConfigRecords();

  auto parameters_record = array_records.at("evaluateins.parameters");
  flwr_local::Parameters params =
      parametersrecord_to_parameters(parameters_record, keep_input);

  auto configs_record = config_records.at("evaluateins.config");
  flwr_local::Config config_dict;

  for (const auto &[key, value] : configs_record) {
    flwr_local::Scalar scalar;

    std::visit(
        [&scalar](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, int64_t>) {
            scalar.setInt(static_cast<int>(arg));
          } else if constexpr (std::is_same_v<T, double>) {
            scalar.setDouble(arg);
          } else if constexpr (std::is_same_v<T, std::string>) {
            scalar.setString(arg);
          } else if constexpr (std::is_same_v<T, bool>) {
            scalar.setBool(arg);
          }
        },
        value);

    config_dict[key] = scalar;
  }

  return flwr_local::EvaluateIns(params, config_dict);
}

flwr_local::RecordDict recorddict_from_get_parameters_res(
    const flwr_local::ParametersRes &get_parameters_res) {
  std::map<std::string, flwr_local::RecordDictValue> items;

  items["getparametersres.parameters"] = flwr_local::ArrayRecord(
      parameters_to_parametersrecord(get_parameters_res.getParameters()));

  flwr_local::ConfigRecord status_record;
  status_record["code"] = static_cast<int64_t>(0);
  status_record["message"] = std::string("Success");
  items["getparametersres.status"] = status_record;

  return flwr_local::RecordDict(items);
}

flwr_local::RecordDict
recorddict_from_fit_res(const flwr_local::FitRes &fitres) {
  std::map<std::string, flwr_local::RecordDictValue> items;

  items["fitres.parameters"] = flwr_local::ArrayRecord(
      parameters_to_parametersrecord(fitres.getParameters()));

  flwr_local::MetricRecord num_examples_record;
  num_examples_record["num_examples"] =
      static_cast<int64_t>(fitres.getNum_example());
  items["fitres.num_examples"] = num_examples_record;

  flwr_local::ConfigRecord status_record;
  status_record["code"] = static_cast<int64_t>(0);
  status_record["message"] = std::string("Success");
  items["fitres.status"] = status_record;

  if (fitres.getMetrics() != std::nullopt) {
    items["fitres.metrics"] =
        metrics_to_config_record(fitres.getMetrics().value());
  } else {
    items["fitres.metrics"] = flwr_local::ConfigRecord{};
  }

  return flwr_local::RecordDict(items);
}

flwr_local::RecordDict
recorddict_from_evaluate_res(const flwr_local::EvaluateRes &evaluate_res) {
  std::map<std::string, flwr_local::RecordDictValue> items;

  flwr_local::MetricRecord loss_record;
  loss_record["loss"] = static_cast<double>(evaluate_res.getLoss());
  items["evaluateres.loss"] = loss_record;

  flwr_local::MetricRecord num_examples_record;
  num_examples_record["num_examples"] =
      static_cast<int64_t>(evaluate_res.getNum_example());
  items["evaluateres.num_examples"] = num_examples_record;

  flwr_local::ConfigRecord status_record;
  status_record["code"] = static_cast<int64_t>(0);
  status_record["message"] = std::string("Success");
  items["evaluateres.status"] = status_record;

  if (evaluate_res.getMetrics() != std::nullopt) {
    items["evaluateres.metrics"] =
        metrics_to_config_record(evaluate_res.getMetrics().value());
  } else {
    items["evaluateres.metrics"] = flwr_local::ConfigRecord{};
  }

  return flwr_local::RecordDict(items);
}

///////////////////////////////////////////////////////////////////////////////
// Inflatable object utilities (Flower 1.27+ protocol)
///////////////////////////////////////////////////////////////////////////////

// SHA-256 hex digest of raw bytes
std::string compute_sha256(const std::string &data) {
  uint8_t hash[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const uint8_t *>(data.data()), data.size(), hash);
  std::ostringstream ss;
  for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
    ss << std::hex << std::setw(2) << std::setfill('0')
       << static_cast<int>(hash[i]);
  }
  return ss.str();
}

// Build inflatable object bytes: "ClassName children_csv bodylen\x00body"
static std::string build_object_bytes(const std::string &class_name,
                                      const std::vector<std::string> &children,
                                      const std::string &body) {
  std::string children_str;
  for (size_t i = 0; i < children.size(); i++) {
    if (i > 0)
      children_str += ",";
    children_str += children[i];
  }
  std::string header =
      class_name + " " + children_str + " " + std::to_string(body.size());
  return header + '\x00' + body;
}

// Parse object header from bytes: splits on first \x00, returns {class, children, body}
struct ParsedObject {
  std::string class_name;
  std::vector<std::string> children;
  std::string body;
};

static ParsedObject parse_object_bytes(const std::string &bytes) {
  auto null_pos = bytes.find('\x00');
  if (null_pos == std::string::npos)
    throw std::runtime_error("Invalid object bytes: no null separator");
  std::string header = bytes.substr(0, null_pos);
  std::string body = bytes.substr(null_pos + 1);

  // Header: "ClassName children_csv bodylen"
  auto sp1 = header.find(' ');
  auto sp2 = header.find(' ', sp1 + 1);

  ParsedObject obj;
  obj.class_name = header.substr(0, sp1);
  obj.body = body;

  std::string children_str = header.substr(sp1 + 1, sp2 - sp1 - 1);
  if (!children_str.empty()) {
    size_t pos = 0;
    while (pos < children_str.size()) {
      auto comma = children_str.find(',', pos);
      if (comma == std::string::npos) {
        obj.children.push_back(children_str.substr(pos));
        break;
      }
      obj.children.push_back(children_str.substr(pos, comma - pos));
      pos = comma + 1;
    }
  }
  return obj;
}

// Collect all object IDs from an ObjectTree (recursive, depth-first)
void collect_object_ids(const flwr::proto::ObjectTree &tree,
                        std::vector<std::string> &out) {
  if (!tree.object_id().empty()) {
    out.push_back(tree.object_id());
  }
  for (const auto &child : tree.children()) {
    collect_object_ids(child, out);
  }
}

// Simple JSON string-to-string map parser: {"key": "value", ...}
static std::map<std::string, std::string>
parse_json_str_map(const std::string &json) {
  std::map<std::string, std::string> result;
  size_t pos = 0;
  while (pos < json.size()) {
    // Find next '"'
    pos = json.find('"', pos);
    if (pos == std::string::npos)
      break;
    size_t key_start = pos + 1;
    size_t key_end = json.find('"', key_start);
    if (key_end == std::string::npos)
      break;
    std::string key = json.substr(key_start, key_end - key_start);
    pos = key_end + 1;

    // Find ':' then '"'
    pos = json.find('"', pos);
    if (pos == std::string::npos)
      break;
    size_t val_start = pos + 1;
    size_t val_end = json.find('"', val_start);
    if (val_end == std::string::npos)
      break;
    std::string val = json.substr(val_start, val_end - val_start);
    pos = val_end + 1;
    result[key] = val;
  }
  return result;
}

// Parse "arraychunk_ids": [0, 1, ...] from Array body JSON
static std::vector<int> parse_arraychunk_ids(const std::string &json) {
  std::vector<int> result;
  auto idx = json.find("arraychunk_ids");
  if (idx == std::string::npos)
    return result;
  auto lb = json.find('[', idx);
  auto rb = json.find(']', lb);
  if (lb == std::string::npos || rb == std::string::npos)
    return result;
  std::string arr_str = json.substr(lb + 1, rb - lb - 1);
  size_t pos = 0;
  while (pos < arr_str.size()) {
    while (pos < arr_str.size() && (arr_str[pos] == ' ' || arr_str[pos] == ','))
      pos++;
    if (pos >= arr_str.size())
      break;
    size_t num_end = pos;
    while (num_end < arr_str.size() && std::isdigit(arr_str[num_end]))
      num_end++;
    if (num_end > pos) {
      result.push_back(std::stoi(arr_str.substr(pos, num_end - pos)));
      pos = num_end;
    } else {
      break;
    }
  }
  return result;
}

// Inflate a RecordDict from objects map.
// recorddict_obj_id: SHA-256 of the RecordDict object bytes
// objects: map object_id -> raw bytes
flwr_local::RecordDict
inflate_recorddict(const std::string &recorddict_obj_id,
                   const std::map<std::string, std::string> &objects) {
  // Get RecordDict object bytes
  auto rd_it = objects.find(recorddict_obj_id);
  if (rd_it == objects.end())
    throw std::runtime_error("RecordDict object not found: " +
                             recorddict_obj_id);

  auto rd_parsed = parse_object_bytes(rd_it->second);
  // body is JSON: {"key": "child_object_id", ...}
  std::cerr << "[DEBUG inflate] rd class_name='" << rd_parsed.class_name
            << "' body_len=" << rd_parsed.body.size()
            << " body_first_100='" << rd_parsed.body.substr(0, 100) << "'" << std::endl;
  auto rd_refs = parse_json_str_map(rd_parsed.body);
  std::cerr << "[DEBUG inflate] rd_refs size=" << rd_refs.size() << std::endl;

  std::map<std::string, flwr_local::RecordDictValue> items;

  for (const auto &[record_name, child_id] : rd_refs) {
    auto child_it = objects.find(child_id);
    if (child_it == objects.end())
      throw std::runtime_error("Child object not found: " + child_id);

    auto child_parsed = parse_object_bytes(child_it->second);

    if (child_parsed.class_name == "ArrayRecord") {
      // ArrayRecord body: JSON {"array_key": "array_object_id", ...}
      auto array_refs = parse_json_str_map(child_parsed.body);
      flwr_local::ArrayRecord array_record;

      for (const auto &[array_key, array_id] : array_refs) {
        auto arr_it = objects.find(array_id);
        if (arr_it == objects.end())
          throw std::runtime_error("Array object not found: " + array_id);

        auto arr_parsed = parse_object_bytes(arr_it->second);
        // Array children = unique chunk object_ids (in order of unique_children)
        // Array body JSON: {"dtype":..., "shape":..., "stype":..., "arraychunk_ids":[...]}
        auto chunk_indices = parse_arraychunk_ids(arr_parsed.body);

        // Extract stype
        std::string stype = "numpy.ndarray";
        {
          auto s_idx = arr_parsed.body.find("\"stype\"");
          if (s_idx != std::string::npos) {
            auto q1 = arr_parsed.body.find('"', s_idx + 7);
            auto q2 = arr_parsed.body.find('"', q1 + 1);
            auto q3 = arr_parsed.body.find('"', q2 + 1);
            auto q4 = arr_parsed.body.find('"', q3 + 1);
            if (q3 != std::string::npos && q4 != std::string::npos)
              stype = arr_parsed.body.substr(q3 + 1, q4 - q3 - 1);
          }
        }

        // Concatenate chunk data in order specified by arraychunk_ids
        std::string raw_data;
        for (int ci : chunk_indices) {
          if (ci < 0 || ci >= (int)arr_parsed.children.size())
            throw std::runtime_error("Invalid chunk index");
          const std::string &chunk_id = arr_parsed.children[ci];
          auto chunk_it = objects.find(chunk_id);
          if (chunk_it == objects.end())
            throw std::runtime_error("ArrayChunk not found: " + chunk_id);
          auto chunk_parsed = parse_object_bytes(chunk_it->second);
          raw_data += chunk_parsed.body;
        }

        flwr_local::Array array;
        array.dtype = "";
        array.stype = stype;
        array.data = raw_data;
        array_record[array_key] = array;
      }
      items[record_name] = array_record;

    } else if (child_parsed.class_name == "ConfigRecord") {
      // ConfigRecord body: serialized flwr::proto::ConfigRecord
      flwr::proto::ConfigRecord proto;
      proto.ParseFromString(child_parsed.body);
      items[record_name] = config_record_from_proto(proto);

    } else if (child_parsed.class_name == "MetricRecord") {
      // MetricRecord body: serialized flwr::proto::MetricRecord
      flwr::proto::MetricRecord proto;
      proto.ParseFromString(child_parsed.body);
      items[record_name] = metric_record_from_proto(proto);

    } else {
      throw std::runtime_error("Unknown record class: " + child_parsed.class_name);
    }
  }

  return flwr_local::RecordDict(items);
}

// Deflate a RecordDict and Message into inflatable object bytes.
// Returns DeflatedContent with objects map + ObjectTree + message_id.
DeflatedContent deflate_message(const flwr_local::RecordDict &rd,
                                const flwr_local::Metadata &reply_metadata) {
  DeflatedContent result;

  // ---- Deflate RecordDict bottom-up ----
  std::vector<std::string> rd_children; // children of RecordDict

  for (const auto &[record_name, record_val] : rd.getItems()) {
    std::string child_id;

    if (std::holds_alternative<flwr_local::ArrayRecord>(record_val)) {
      const auto &ar = std::get<flwr_local::ArrayRecord>(record_val);
      std::vector<std::string> ar_children;

      for (const auto &[array_key, array] : ar) {
        // ArrayChunk: body = raw tensor data
        std::string chunk_body(array.data.begin(), array.data.end());
        std::string chunk_bytes =
            build_object_bytes("ArrayChunk", {}, chunk_body);
        std::string chunk_id = compute_sha256(chunk_bytes);
        result.objects[chunk_id] = chunk_bytes;

        // Array: JSON body referencing the chunk
        // unique_children = [chunk_id], arraychunk_ids = [0]
        std::ostringstream arr_body;
        arr_body << "{\"dtype\": \"\", \"shape\": [], \"stype\": \""
                 << array.stype << "\", \"arraychunk_ids\": [0]}";
        std::string array_bytes =
            build_object_bytes("Array", {chunk_id}, arr_body.str());
        std::string array_id = compute_sha256(array_bytes);
        result.objects[array_id] = array_bytes;
        ar_children.push_back(array_id);

        // ArrayRecord JSON entry: "array_key": "array_id"
        // (we build the JSON below)
        (void)array_key;
        // Store array_key→array_id for JSON construction
        // We'll reconstruct below
        (void)ar_children; // will be used
      }

      // Build ArrayRecord JSON body: {"0": "array_id", ...}
      std::string ar_body = "{";
      bool first = true;
      size_t arr_idx = 0;
      std::vector<std::string> ar_child_ids;
      for (const auto &[array_key, array] : ar) {
        // re-compute array_id from the same data (same chunk was computed above)
        std::string chunk_body(array.data.begin(), array.data.end());
        std::string chunk_bytes =
            build_object_bytes("ArrayChunk", {}, chunk_body);
        std::string chunk_id = compute_sha256(chunk_bytes);
        std::ostringstream arr_body_s;
        arr_body_s << "{\"dtype\": \"\", \"shape\": [], \"stype\": \""
                   << array.stype << "\", \"arraychunk_ids\": [0]}";
        std::string array_bytes =
            build_object_bytes("Array", {chunk_id}, arr_body_s.str());
        std::string array_id = compute_sha256(array_bytes);
        ar_child_ids.push_back(array_id);

        if (!first)
          ar_body += ", ";
        ar_body += "\"" + array_key + "\": \"" + array_id + "\"";
        first = false;
        arr_idx++;
      }
      ar_body += "}";

      std::string ar_bytes =
          build_object_bytes("ArrayRecord", ar_child_ids, ar_body);
      std::string ar_id = compute_sha256(ar_bytes);
      result.objects[ar_id] = ar_bytes;
      rd_children.push_back(ar_id);
      child_id = ar_id;

    } else if (std::holds_alternative<flwr_local::ConfigRecord>(record_val)) {
      const auto &cr = std::get<flwr_local::ConfigRecord>(record_val);
      flwr::proto::ConfigRecord proto = config_record_to_proto(cr);
      std::string cr_body;
      proto.SerializeToString(&cr_body);
      std::string cr_bytes = build_object_bytes("ConfigRecord", {}, cr_body);
      std::string cr_id = compute_sha256(cr_bytes);
      result.objects[cr_id] = cr_bytes;
      rd_children.push_back(cr_id);
      child_id = cr_id;

    } else if (std::holds_alternative<flwr_local::MetricRecord>(record_val)) {
      const auto &mr = std::get<flwr_local::MetricRecord>(record_val);
      flwr::proto::MetricRecord proto = metric_record_to_proto(mr);
      std::string mr_body;
      proto.SerializeToString(&mr_body);
      std::string mr_bytes = build_object_bytes("MetricRecord", {}, mr_body);
      std::string mr_id = compute_sha256(mr_bytes);
      result.objects[mr_id] = mr_bytes;
      rd_children.push_back(mr_id);
      child_id = mr_id;
    }

    (void)child_id; // used indirectly via rd_children
  }

  // Build RecordDict JSON body: {"record_name": "child_id", ...}
  std::string rdbody = "{";
  {
    bool first = true;
    size_t i = 0;
    for (const auto &[record_name, record_val] : rd.getItems()) {
      if (!first)
        rdbody += ", ";
      rdbody += "\"" + record_name + "\": \"" + rd_children[i] + "\"";
      first = false;
      i++;
    }
  }
  rdbody += "}";

  std::string rd_bytes = build_object_bytes("RecordDict", rd_children, rdbody);
  std::string rd_id = compute_sha256(rd_bytes);
  result.objects[rd_id] = rd_bytes;

  // ---- Deflate Message ----
  // Build metadata proto WITHOUT message_id (as Python does)
  flwr_local::Metadata meta_no_id = reply_metadata;
  meta_no_id.message_id = "";
  flwr::proto::Metadata meta_proto = metadata_to_proto(meta_no_id);

  // Build Message proto body (metadata only, no content, no error)
  flwr::proto::Message msg_proto_body;
  *msg_proto_body.mutable_metadata() = meta_proto;
  std::string msg_body;
  msg_proto_body.SerializeToString(&msg_body);

  // Message object: one child = RecordDict
  std::string msg_bytes = build_object_bytes("Message", {rd_id}, msg_body);
  std::string msg_id = compute_sha256(msg_bytes);
  result.objects[msg_id] = msg_bytes;
  result.message_id = msg_id;

  // ---- Build ObjectTree ----
  // Helper to build tree recursively from objects map
  // We need to know the tree structure. Use the parsed children from the objects.
  std::function<flwr::proto::ObjectTree(const std::string &)> build_tree;
  build_tree = [&](const std::string &obj_id) -> flwr::proto::ObjectTree {
    flwr::proto::ObjectTree tree;
    tree.set_object_id(obj_id);
    const auto &bytes = result.objects.at(obj_id);
    auto parsed = parse_object_bytes(bytes);
    for (const auto &child_id : parsed.children) {
      *tree.add_children() = build_tree(child_id);
    }
    return tree;
  };
  result.message_tree = build_tree(msg_id);

  return result;
}
