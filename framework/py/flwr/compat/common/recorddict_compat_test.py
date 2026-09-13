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
"""RecordDict from legacy messages tests."""


from collections.abc import Callable
from copy import deepcopy

import numpy as np
import pytest

from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    NDArrays,
    Scalar,
    Status,
    ndarrays_to_parameters,
)

from .recorddict_compat import (
    evaluateins_to_recorddict,
    evaluateres_to_recorddict,
    fitins_to_recorddict,
    fitres_to_recorddict,
    getparametersins_to_recorddict,
    getparametersres_to_recorddict,
    getpropertiesins_to_recorddict,
    getpropertiesres_to_recorddict,
    recorddict_to_evaluateins,
    recorddict_to_evaluateres,
    recorddict_to_fitins,
    recorddict_to_fitres,
    recorddict_to_getparametersins,
    recorddict_to_getparametersres,
    recorddict_to_getpropertiesins,
    recorddict_to_getpropertiesres,
)


def get_ndarrays() -> NDArrays:
    """Return list of NumPy arrays."""
    arr1 = np.array([[1.0, 2.0], [3.0, 4], [5.0, 6.0]])
    arr2 = np.eye(2, 7, 3)

    return [arr1, arr2]


##################################################
#  Testing conversion: *Ins --> RecordDict --> *Ins
#  Testing conversion: *Res <-- RecordDict <-- *Res
##################################################


def _get_valid_fitins() -> FitIns:
    arrays = get_ndarrays()
    return FitIns(parameters=ndarrays_to_parameters(arrays), config={"a": 1.0, "b": 0})


def _get_valid_fitins_with_empty_ndarrays() -> FitIns:
    pp = ndarrays_to_parameters([])
    return FitIns(parameters=pp, config={"a": 1.0, "b": 0})


def _get_valid_fitres() -> FitRes:
    """Returnn Valid parameters but potentially invalid config."""
    arrays = get_ndarrays()
    metrics: dict[str, Scalar] = {"a": 1.0, "b": 0}
    return FitRes(
        parameters=ndarrays_to_parameters(arrays),
        num_examples=1,
        status=Status(code=Code(0), message=""),
        metrics=metrics,
    )


def _get_valid_evaluateins() -> EvaluateIns:
    fit_ins = _get_valid_fitins()
    return EvaluateIns(parameters=fit_ins.parameters, config=fit_ins.config)


def _get_valid_evaluateres() -> EvaluateRes:
    """Return potentially invalid config."""
    metrics: dict[str, Scalar] = {"a": 1.0, "b": 0}
    return EvaluateRes(
        num_examples=1,
        loss=0.1,
        status=Status(code=Code(0), message=""),
        metrics=metrics,
    )


def _get_valid_getparametersins() -> GetParametersIns:
    config_dict: dict[str, Scalar] = {
        "a": 1.0,
        "b": 3,
        "c": True,
    }  # valid since both Ins/Res communicate over ConfigRecord

    return GetParametersIns(config_dict)


def _get_valid_getparametersres() -> GetParametersRes:
    arrays = get_ndarrays()
    return GetParametersRes(
        status=Status(code=Code(0), message=""),
        parameters=ndarrays_to_parameters(arrays),
    )


def _get_valid_getpropertiesins() -> GetPropertiesIns:
    getparamsins = _get_valid_getparametersins()
    return GetPropertiesIns(config=getparamsins.config)


def _get_valid_getpropertiesres() -> GetPropertiesRes:
    config_dict: dict[str, Scalar] = {
        "a": 1.0,
        "b": 3,
        "c": True,
    }  # valid since both Ins/Res communicate over ConfigRecord

    return GetPropertiesRes(
        status=Status(code=Code(0), message=""), properties=config_dict
    )


@pytest.mark.parametrize(
    "keep_input, validate_freed_fn, fn",
    [
        (
            False,
            lambda x, x_copy, y: len(x.parameters.tensors) == 0 and x_copy == y,
            _get_valid_fitins,
        ),  # check tensors were freed
        (True, lambda x, x_copy, y: x == y, _get_valid_fitins),
        (
            False,
            lambda x, x_copy, y: len(x.parameters.tensors) == 0 and x_copy == y,
            _get_valid_fitins_with_empty_ndarrays,
        ),  # check tensors were freed
        (True, lambda x, x_copy, y: x == y, _get_valid_fitins_with_empty_ndarrays),
    ],
)
def test_fitins_to_recorddict_and_back(
    keep_input: bool,
    validate_freed_fn: Callable[[FitIns, FitIns, FitIns], bool],
    fn: Callable[[], FitIns],
) -> None:
    """Test conversion FitIns --> RecordDict --> FitIns."""
    fitins = fn()

    fitins_copy = deepcopy(fitins)

    recorddict = fitins_to_recorddict(fitins, keep_input=keep_input)

    fitins_ = recorddict_to_fitins(recorddict, keep_input=keep_input)

    assert validate_freed_fn(fitins, fitins_copy, fitins_)


@pytest.mark.parametrize(
    "keep_input, validate_freed_fn",
    [
        (
            False,
            lambda x, x_copy, y: len(x.parameters.tensors) == 0 and x_copy == y,
        ),  # check tensors were freed
        (
            True,
            lambda x, x_copy, y: x == y,
        ),
    ],
)
def test_fitres_to_recorddict_and_back(
    keep_input: bool, validate_freed_fn: Callable[[FitRes, FitRes, FitRes], bool]
) -> None:
    """Test conversion FitRes --> RecordDict --> FitRes."""
    fitres = _get_valid_fitres()

    fitres_copy = deepcopy(fitres)

    recorddict = fitres_to_recorddict(fitres, keep_input=keep_input)
    fitres_ = recorddict_to_fitres(recorddict, keep_input=keep_input)

    assert validate_freed_fn(fitres, fitres_copy, fitres_)


@pytest.mark.parametrize(
    "keep_input, validate_freed_fn",
    [
        (
            False,
            lambda x, x_copy, y: len(x.parameters.tensors) == 0 and x_copy == y,
        ),  # check tensors were freed
        (
            True,
            lambda x, x_copy, y: x == y,
        ),
    ],
)
def test_evaluateins_to_recorddict_and_back(
    keep_input: bool,
    validate_freed_fn: Callable[[EvaluateIns, EvaluateIns, EvaluateIns], bool],
) -> None:
    """Test conversion EvaluateIns --> RecordDict --> EvaluateIns."""
    evaluateins = _get_valid_evaluateins()

    evaluateins_copy = deepcopy(evaluateins)

    recorddict = evaluateins_to_recorddict(evaluateins, keep_input=keep_input)

    evaluateins_ = recorddict_to_evaluateins(recorddict, keep_input=keep_input)

    assert validate_freed_fn(evaluateins, evaluateins_copy, evaluateins_)


def test_evaluateres_to_recorddict_and_back() -> None:
    """Test conversion EvaluateRes --> RecordDict --> EvaluateRes."""
    evaluateres = _get_valid_evaluateres()

    evaluateres_copy = deepcopy(evaluateres)

    recorddict = evaluateres_to_recorddict(evaluateres)
    evaluateres_ = recorddict_to_evaluateres(recorddict)

    assert evaluateres_copy == evaluateres_


def test_get_properties_ins_to_recorddict_and_back() -> None:
    """Test conversion GetPropertiesIns --> RecordDict --> GetPropertiesIns."""
    getproperties_ins = _get_valid_getpropertiesins()

    getproperties_ins_copy = deepcopy(getproperties_ins)

    recorddict = getpropertiesins_to_recorddict(getproperties_ins)
    getproperties_ins_ = recorddict_to_getpropertiesins(recorddict)

    assert getproperties_ins_copy == getproperties_ins_


def test_get_properties_res_to_recorddict_and_back() -> None:
    """Test conversion GetPropertiesRes --> RecordDict --> GetPropertiesRes."""
    getproperties_res = _get_valid_getpropertiesres()

    getproperties_res_copy = deepcopy(getproperties_res)

    recorddict = getpropertiesres_to_recorddict(getproperties_res)
    getproperties_res_ = recorddict_to_getpropertiesres(recorddict)

    assert getproperties_res_copy == getproperties_res_


def test_get_parameters_ins_to_recorddict_and_back() -> None:
    """Test conversion GetParametersIns --> RecordDict --> GetParametersIns."""
    getparameters_ins = _get_valid_getparametersins()

    getparameters_ins_copy = deepcopy(getparameters_ins)

    recorddict = getparametersins_to_recorddict(getparameters_ins)
    getparameters_ins_ = recorddict_to_getparametersins(recorddict)

    assert getparameters_ins_copy == getparameters_ins_


@pytest.mark.parametrize(
    "keep_input, validate_freed_fn",
    [
        (
            False,
            lambda x, x_copy, y: len(x.parameters.tensors) == 0 and x_copy == y,
        ),  # check tensors were freed
        (
            True,
            lambda x, x_copy, y: x == y,
        ),
    ],
)
def test_get_parameters_res_to_recorddict_and_back(
    keep_input: bool,
    validate_freed_fn: Callable[
        [GetParametersRes, GetParametersRes, GetParametersRes], bool
    ],
) -> None:
    """Test conversion GetParametersRes --> RecordDict --> GetParametersRes."""
    getparameteres_res = _get_valid_getparametersres()

    getparameters_res_copy = deepcopy(getparameteres_res)

    recorddict = getparametersres_to_recorddict(
        getparameteres_res, keep_input=keep_input
    )
    getparameteres_res_ = recorddict_to_getparametersres(
        recorddict, keep_input=keep_input
    )

    assert validate_freed_fn(
        getparameteres_res, getparameters_res_copy, getparameteres_res_
    )
