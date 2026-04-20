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
#include "flwr/proto/fleet.pb.h"
#include "flwr/proto/transport.pb.h"
#include "typing.h"

/**
 * Serialize client parameters to protobuf parameters message
 */
flwr::proto::Parameters parameters_to_proto(flwr_local::Parameters parameters);

/**
 * Deserialize client protobuf parameters message to client parameters
 */
flwr_local::Parameters parameters_from_proto(flwr::proto::Parameters msg);

/**
 * Serialize client scalar type to protobuf scalar type
 */
flwr::proto::Scalar scalar_to_proto(flwr_local::Scalar scalar_msg);

/**
 * Deserialize protobuf scalar type to client scalar type
 */
flwr_local::Scalar scalar_from_proto(flwr::proto::Scalar scalar_msg);

/**
 * Serialize client metrics type to protobuf metrics type
 * "Any" is used in Python, this part might be changed if needed
 */
google::protobuf::Map<std::string, flwr::proto::Scalar>
metrics_to_proto(flwr_local::Metrics metrics);

/**
 * Deserialize protobuf metrics type to client metrics type
 * "Any" is used in Python, this part might be changed if needed
 */
flwr_local::Metrics metrics_from_proto(
    google::protobuf::Map<std::string, flwr::proto::Scalar> proto);

/**
 * Serialize client ParametersRes type to protobuf ParametersRes type
 */
flwr::proto::ClientMessage_GetParametersRes
parameters_res_to_proto(flwr_local::ParametersRes res);

/**
 * Deserialize protobuf FitIns type to client FitIns type
 */
flwr_local::FitIns fit_ins_from_proto(flwr::proto::ServerMessage_FitIns msg);

/**
 * Serialize client FitRes type to protobuf FitRes type
 */
flwr::proto::ClientMessage_FitRes fit_res_to_proto(flwr_local::FitRes res);

/**
 * Deserialize protobuf EvaluateIns type to client EvaluateIns type
 */
flwr_local::EvaluateIns
evaluate_ins_from_proto(flwr::proto::ServerMessage_EvaluateIns msg);

/**
 * Serialize client EvaluateRes type to protobuf EvaluateRes type
 */
flwr::proto::ClientMessage_EvaluateRes
evaluate_res_to_proto(flwr_local::EvaluateRes res);

flwr_local::RecordSet
recordset_from_proto(const flwr::proto::RecordDict &recordset);

flwr_local::FitIns recordset_to_fit_ins(const flwr_local::RecordSet &recordset,
                                        bool keep_input);

flwr_local::EvaluateIns
recordset_to_evaluate_ins(const flwr_local::RecordSet &recordset,
                          bool keep_input);

flwr_local::RecordSet
recordset_from_evaluate_res(const flwr_local::EvaluateRes &evaluate_res);

flwr_local::RecordSet recordset_from_fit_res(const flwr_local::FitRes &fit_res);

flwr_local::RecordSet recordset_from_get_parameters_res(
    const flwr_local::ParametersRes &parameters_res);

flwr::proto::RecordDict
recordset_to_proto(const flwr_local::RecordSet &recordset);
