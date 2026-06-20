# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""RecordDict utilities."""


from collections.abc import Mapping
from typing import cast, get_args

from flwr.app import Array, ArrayRecord, ConfigRecord, MetricRecord, RecordDict
from flwr.app.typing import ConfigRecordValues, MetricRecordValues

from .typing import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    Scalar,
    Status,
)

EMPTY_TENSOR_KEY = "_empty"


def arrayrecord_to_parameters(record: ArrayRecord, keep_input: bool) -> Parameters:
    """Convert ParameterRecord to legacy Parameters.

    Warnings
    --------
    Because `Array`s in `ArrayRecord` encode more information of the
    array-like or tensor-like data (e.g their datatype, shape) than `Parameters` it
    might not be possible to reconstruct such data structures from `Parameters` objects
    alone. Additional information or metadata must be provided from elsewhere.

    Parameters
    ----------
    record : ArrayRecord
        The record to be conveted into Parameters.
    keep_input : bool
        A boolean indicating whether entries in the record should be deleted from the
        input dictionary immediately after adding them to the record.

    Returns
    -------
    parameters : Parameters
        The parameters in the legacy format Parameters.
    """
    parameters = Parameters(tensors=[], tensor_type="")

    for key in list(record.keys()):
        if key != EMPTY_TENSOR_KEY:
            parameters.tensors.append(record[key].data)

        if not parameters.tensor_type:
            # Setting from first array in record. Recall the warning in the docstrings
            # of this function.
            parameters.tensor_type = record[key].stype

        if not keep_input:
            del record[key]

    return parameters


def parameters_to_arrayrecord(parameters: Parameters, keep_input: bool) -> ArrayRecord:
    """Convert legacy Parameters into a single ArrayRecord.

    Because there is no concept of names in the legacy Parameters, arbitrary keys will
    be used when constructing the ArrayRecord. Similarly, the shape and data type
    won't be recorded in the Array objects.

    Parameters
    ----------
    parameters : Parameters
        Parameters object to be represented as a ArrayRecord.
    keep_input : bool
        A boolean indicating whether parameters should be deleted from the input
        Parameters object (i.e. a list of serialized NumPy arrays) immediately after
        adding them to the record.

    Returns
    -------
    ArrayRecord
        The ArrayRecord containing the provided parameters.
    """
    tensor_type = parameters.tensor_type

    num_arrays = len(parameters.tensors)
    array_dict = {}
    for idx in range(num_arrays):
        if keep_input:
            tensor = parameters.tensors[idx]
        else:
            tensor = parameters.tensors.pop(0)
        array_dict[str(idx)] = Array(data=tensor, dtype="", stype=tensor_type, shape=())

    if num_arrays == 0:
        array_dict[EMPTY_TENSOR_KEY] = Array(
            data=b"", dtype="", stype=tensor_type, shape=()
        )
    return ArrayRecord(array_dict, keep_input=keep_input)


def _check_mapping_from_recordscalartype_to_scalar(
    record_data: Mapping[str, ConfigRecordValues | MetricRecordValues],
) -> dict[str, Scalar]:
    """Check mapping `common.*RecordValues` into `common.Scalar` is possible."""
    for value in record_data.values():
        if not isinstance(value, get_args(Scalar)):
            raise TypeError(
                "There is not a 1:1 mapping between `common.Scalar` types and those "
                "supported in `flwr.app.ConfigRecordValues` or "
                "`flwr.app.MetricRecordValues`. Consider casting your values to a type "
                "supported by the `common.RecordDict` infrastructure. "
                f"You used type: {type(value)}"
            )
    return cast(dict[str, Scalar], record_data)


def _recorddict_to_fit_or_evaluate_ins_components(
    recorddict: RecordDict,
    ins_str: str,
    keep_input: bool,
) -> tuple[Parameters, dict[str, Scalar]]:
    """Derive Fit/Evaluate Ins from a RecordDict."""
    # get Array and construct Parameters
    array_record = recorddict.array_records[f"{ins_str}.parameters"]

    parameters = arrayrecord_to_parameters(array_record, keep_input=keep_input)

    # get config dict
    config_record = recorddict.config_records[f"{ins_str}.config"]
    # pylint: disable-next=protected-access
    config_dict = _check_mapping_from_recordscalartype_to_scalar(config_record)

    return parameters, config_dict


def _fit_or_evaluate_ins_to_recorddict(
    ins: FitIns | EvaluateIns, keep_input: bool
) -> RecordDict:
    recorddict = RecordDict()

    ins_str = "fitins" if isinstance(ins, FitIns) else "evaluateins"
    arr_record = parameters_to_arrayrecord(ins.parameters, keep_input)
    recorddict.array_records[f"{ins_str}.parameters"] = arr_record

    recorddict.config_records[f"{ins_str}.config"] = ConfigRecord(
        ins.config  # type: ignore
    )

    return recorddict


def _embed_status_into_recorddict(
    res_str: str, status: Status, recorddict: RecordDict
) -> RecordDict:
    status_dict: dict[str, ConfigRecordValues] = {
        "code": int(status.code.value),
        "message": status.message,
    }
    # we add it to a `ConfigRecord` because the `status.message` is a string
    # and `str` values aren't supported in `MetricRecords`
    recorddict.config_records[f"{res_str}.status"] = ConfigRecord(status_dict)
    return recorddict


def _extract_status_from_recorddict(res_str: str, recorddict: RecordDict) -> Status:
    status = recorddict.config_records[f"{res_str}.status"]
    code = cast(int, status["code"])
    return Status(code=Code(code), message=str(status["message"]))


def recorddict_to_fitins(recorddict: RecordDict, keep_input: bool) -> FitIns:
    """Derive FitIns from a RecordDict object."""
    parameters, config = _recorddict_to_fit_or_evaluate_ins_components(
        recorddict,
        ins_str="fitins",
        keep_input=keep_input,
    )

    return FitIns(parameters=parameters, config=config)


def fitins_to_recorddict(fitins: FitIns, keep_input: bool) -> RecordDict:
    """Construct a RecordDict from a FitIns object."""
    return _fit_or_evaluate_ins_to_recorddict(fitins, keep_input)


def recorddict_to_fitres(recorddict: RecordDict, keep_input: bool) -> FitRes:
    """Derive FitRes from a RecordDict object."""
    ins_str = "fitres"
    parameters = arrayrecord_to_parameters(
        recorddict.array_records[f"{ins_str}.parameters"], keep_input=keep_input
    )

    num_examples = cast(
        int, recorddict.metric_records[f"{ins_str}.num_examples"]["num_examples"]
    )
    config_record = recorddict.config_records[f"{ins_str}.metrics"]
    # pylint: disable-next=protected-access
    metrics = _check_mapping_from_recordscalartype_to_scalar(config_record)
    status = _extract_status_from_recorddict(ins_str, recorddict)

    return FitRes(
        status=status, parameters=parameters, num_examples=num_examples, metrics=metrics
    )


def fitres_to_recorddict(fitres: FitRes, keep_input: bool) -> RecordDict:
    """Construct a RecordDict from a FitRes object."""
    recorddict = RecordDict()

    res_str = "fitres"

    recorddict.config_records[f"{res_str}.metrics"] = ConfigRecord(
        fitres.metrics  # type: ignore
    )
    recorddict.metric_records[f"{res_str}.num_examples"] = MetricRecord(
        {"num_examples": fitres.num_examples},
    )
    recorddict.array_records[f"{res_str}.parameters"] = parameters_to_arrayrecord(
        fitres.parameters,
        keep_input,
    )

    # status
    recorddict = _embed_status_into_recorddict(res_str, fitres.status, recorddict)

    return recorddict


def recorddict_to_evaluateins(recorddict: RecordDict, keep_input: bool) -> EvaluateIns:
    """Derive EvaluateIns from a RecordDict object."""
    parameters, config = _recorddict_to_fit_or_evaluate_ins_components(
        recorddict,
        ins_str="evaluateins",
        keep_input=keep_input,
    )

    return EvaluateIns(parameters=parameters, config=config)


def evaluateins_to_recorddict(evaluateins: EvaluateIns, keep_input: bool) -> RecordDict:
    """Construct a RecordDict from a EvaluateIns object."""
    return _fit_or_evaluate_ins_to_recorddict(evaluateins, keep_input)


def recorddict_to_evaluateres(recorddict: RecordDict) -> EvaluateRes:
    """Derive EvaluateRes from a RecordDict object."""
    ins_str = "evaluateres"

    loss = cast(int, recorddict.metric_records[f"{ins_str}.loss"]["loss"])

    num_examples = cast(
        int, recorddict.metric_records[f"{ins_str}.num_examples"]["num_examples"]
    )
    config_record = recorddict.config_records[f"{ins_str}.metrics"]

    # pylint: disable-next=protected-access
    metrics = _check_mapping_from_recordscalartype_to_scalar(config_record)
    status = _extract_status_from_recorddict(ins_str, recorddict)

    return EvaluateRes(
        status=status, loss=loss, num_examples=num_examples, metrics=metrics
    )


def evaluateres_to_recorddict(evaluateres: EvaluateRes) -> RecordDict:
    """Construct a RecordDict from a EvaluateRes object."""
    recorddict = RecordDict()

    res_str = "evaluateres"
    # loss
    recorddict.metric_records[f"{res_str}.loss"] = MetricRecord(
        {"loss": evaluateres.loss},
    )

    # num_examples
    recorddict.metric_records[f"{res_str}.num_examples"] = MetricRecord(
        {"num_examples": evaluateres.num_examples},
    )

    # metrics
    recorddict.config_records[f"{res_str}.metrics"] = ConfigRecord(
        evaluateres.metrics,  # type: ignore
    )

    # status
    recorddict = _embed_status_into_recorddict(
        f"{res_str}", evaluateres.status, recorddict
    )

    return recorddict


def recorddict_to_getparametersins(recorddict: RecordDict) -> GetParametersIns:
    """Derive GetParametersIns from a RecordDict object."""
    config_record = recorddict.config_records["getparametersins.config"]
    # pylint: disable-next=protected-access
    config_dict = _check_mapping_from_recordscalartype_to_scalar(config_record)

    return GetParametersIns(config=config_dict)


def getparametersins_to_recorddict(getparameters_ins: GetParametersIns) -> RecordDict:
    """Construct a RecordDict from a GetParametersIns object."""
    recorddict = RecordDict()

    recorddict.config_records["getparametersins.config"] = ConfigRecord(
        getparameters_ins.config,  # type: ignore
    )
    return recorddict


def getparametersres_to_recorddict(
    getparametersres: GetParametersRes, keep_input: bool
) -> RecordDict:
    """Construct a RecordDict from a GetParametersRes object."""
    recorddict = RecordDict()
    res_str = "getparametersres"
    array_record = parameters_to_arrayrecord(
        getparametersres.parameters, keep_input=keep_input
    )
    recorddict.array_records[f"{res_str}.parameters"] = array_record

    # status
    recorddict = _embed_status_into_recorddict(
        res_str, getparametersres.status, recorddict
    )

    return recorddict


def recorddict_to_getparametersres(
    recorddict: RecordDict, keep_input: bool
) -> GetParametersRes:
    """Derive GetParametersRes from a RecordDict object."""
    res_str = "getparametersres"
    parameters = arrayrecord_to_parameters(
        recorddict.array_records[f"{res_str}.parameters"], keep_input=keep_input
    )

    status = _extract_status_from_recorddict(res_str, recorddict)
    return GetParametersRes(status=status, parameters=parameters)


def recorddict_to_getpropertiesins(recorddict: RecordDict) -> GetPropertiesIns:
    """Derive GetPropertiesIns from a RecordDict object."""
    config_record = recorddict.config_records["getpropertiesins.config"]
    # pylint: disable-next=protected-access
    config_dict = _check_mapping_from_recordscalartype_to_scalar(config_record)

    return GetPropertiesIns(config=config_dict)


def getpropertiesins_to_recorddict(getpropertiesins: GetPropertiesIns) -> RecordDict:
    """Construct a RecordDict from a GetPropertiesRes object."""
    recorddict = RecordDict()
    recorddict.config_records["getpropertiesins.config"] = ConfigRecord(
        getpropertiesins.config,  # type: ignore
    )
    return recorddict


def recorddict_to_getpropertiesres(recorddict: RecordDict) -> GetPropertiesRes:
    """Derive GetPropertiesRes from a RecordDict object."""
    res_str = "getpropertiesres"
    config_record = recorddict.config_records[f"{res_str}.properties"]
    # pylint: disable-next=protected-access
    properties = _check_mapping_from_recordscalartype_to_scalar(config_record)

    status = _extract_status_from_recorddict(res_str, recorddict=recorddict)

    return GetPropertiesRes(status=status, properties=properties)


def getpropertiesres_to_recorddict(getpropertiesres: GetPropertiesRes) -> RecordDict:
    """Construct a RecordDict from a GetPropertiesRes object."""
    recorddict = RecordDict()
    res_str = "getpropertiesres"
    recorddict.config_records[f"{res_str}.properties"] = ConfigRecord(
        getpropertiesres.properties,  # type: ignore
    )
    # status
    recorddict = _embed_status_into_recorddict(
        res_str, getpropertiesres.status, recorddict
    )

    return recorddict
