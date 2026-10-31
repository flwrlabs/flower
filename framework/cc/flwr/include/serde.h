/***********************************************************************************************************
 *
 * @file serde.h
 *
 * @brief ProtoBuf serialization and deserialization
 *
 * @author Lekang Jiang
 *
 * @version 1.0
 *
 * @date 03/09/2021
 *
 * ********************************************************************************************************/

#pragma once
#include "flwr/proto/message.pb.h"
#include "flwr/proto/recorddict.pb.h"
#include "flwr/proto/transport.pb.h"
#include "typing.h"
#include <functional>
#include <map>

// Legacy Scalar/Parameters/Metrics serde (used by transport.proto types)
flwr::proto::Parameters parameters_to_proto(flwr_local::Parameters parameters);
flwr_local::Parameters parameters_from_proto(flwr::proto::Parameters msg);
flwr::proto::Scalar scalar_to_proto(flwr_local::Scalar scalar_msg);
flwr_local::Scalar scalar_from_proto(flwr::proto::Scalar scalar_msg);
google::protobuf::Map<std::string, flwr::proto::Scalar>
metrics_to_proto(flwr_local::Metrics metrics);
flwr_local::Metrics metrics_from_proto(
    google::protobuf::Map<std::string, flwr::proto::Scalar> proto);

// Array serde
flwr::proto::Array array_to_proto(const flwr_local::Array &array);
flwr_local::Array array_from_proto(const flwr::proto::Array &proto);

// Record serde (items-based format)
flwr::proto::ArrayRecord
array_record_to_proto(const flwr_local::ArrayRecord &record);
flwr_local::ArrayRecord
array_record_from_proto(const flwr::proto::ArrayRecord &proto);

flwr::proto::MetricRecord
metric_record_to_proto(const flwr_local::MetricRecord &record);
flwr_local::MetricRecord
metric_record_from_proto(const flwr::proto::MetricRecord &proto);

flwr::proto::ConfigRecord
config_record_to_proto(const flwr_local::ConfigRecord &record);
flwr_local::ConfigRecord
config_record_from_proto(const flwr::proto::ConfigRecord &proto);

// RecordDict serde
flwr::proto::RecordDict
recorddict_to_proto(const flwr_local::RecordDict &rd);
flwr_local::RecordDict
recorddict_from_proto(const flwr::proto::RecordDict &proto);

// Message/Metadata serde
flwr::proto::Message message_to_proto(const flwr_local::Message &msg);
flwr_local::Message message_from_proto(const flwr::proto::Message &proto);
flwr::proto::Metadata metadata_to_proto(const flwr_local::Metadata &meta);
flwr_local::Metadata metadata_from_proto(const flwr::proto::Metadata &proto);

// Legacy type conversion (RecordDict <-> FitIns/FitRes/EvaluateIns/EvaluateRes)
flwr_local::FitIns recorddict_to_fit_ins(const flwr_local::RecordDict &rd,
                                         bool keep_input);
flwr_local::EvaluateIns
recorddict_to_evaluate_ins(const flwr_local::RecordDict &rd, bool keep_input);
flwr_local::RecordDict
recorddict_from_fit_res(const flwr_local::FitRes &fit_res);
flwr_local::RecordDict
recorddict_from_evaluate_res(const flwr_local::EvaluateRes &evaluate_res);
flwr_local::RecordDict recorddict_from_get_parameters_res(
    const flwr_local::ParametersRes &parameters_res);

// Inflatable object utilities (Flower 1.27+ protocol)
// Compute SHA-256 hex digest of bytes
std::string compute_sha256(const std::string &data);

// Collect all object_ids from an ObjectTree (recursive)
void collect_object_ids(const flwr::proto::ObjectTree &tree,
                        std::vector<std::string> &out);

// Inflate a RecordDict from pulled objects map.
// objects: map from object_id (hex SHA-256) to raw object bytes.
// recorddict_obj_id: the object_id of the RecordDict to inflate.
flwr_local::RecordDict
inflate_recorddict(const std::string &recorddict_obj_id,
                   const std::map<std::string, std::string> &objects);

// Result of deflating a RecordDict for sending via PushMessages.
struct DeflatedContent {
  // All objects: object_id -> bytes (includes RecordDict, records, arrays, chunks, Message)
  std::map<std::string, std::string> objects;
  // Object tree rooted at the Message
  flwr::proto::ObjectTree message_tree;
  // The message_id (= Message object_id, to set in metadata)
  std::string message_id;
};

// Deflate a RecordDict reply content along with the reply metadata into
// inflatable object bytes. Returns DeflatedContent with all objects + tree.
DeflatedContent deflate_message(const flwr_local::RecordDict &rd,
                                const flwr_local::Metadata &reply_metadata);
