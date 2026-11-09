#pragma once

#include "flwr/proto/recorddict.pb.h"
#include "typing.h"

namespace flwr_quickstart {

flwr_local::RecordSet
recorddict_from_proto(const flwr::proto::RecordDict &recorddict);

flwr::proto::RecordDict
recorddict_to_proto(const flwr_local::RecordSet &recordset);

flwr_local::FitIns recorddict_to_fit_ins(const flwr_local::RecordSet &recordset);

flwr_local::EvaluateIns
recorddict_to_evaluate_ins(const flwr_local::RecordSet &recordset);

flwr_local::RecordSet
recorddict_from_get_parameters_res(const flwr_local::ParametersRes &parameters_res);

flwr_local::RecordSet
recorddict_from_get_properties_res(const flwr_local::PropertiesRes &properties_res);

flwr_local::RecordSet
recorddict_from_fit_res(const flwr_local::FitRes &fit_res);

flwr_local::RecordSet
recorddict_from_evaluate_res(const flwr_local::EvaluateRes &evaluate_res);

} // namespace flwr_quickstart
