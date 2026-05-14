# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""VerticalPartitioner utils.py."""

# noqa: E501
# pylint: disable=C0301
from typing import Any, Literal


def _list_split(lst: list[Any], num_sublists: int) -> list[list[Any]]:
    """Split a list into n nearly equal-sized sublists.

    Parameters
    ----------
    lst : list[Any]
        The list to split.
    num_sublists : int
        Number of sublists to create.

    Returns
    -------
    subslist: list[list[Any]]
        A list containing num_sublists sublists.
    """
    if num_sublists <= 0:
        raise ValueError("Number of splits must be greater than 0")
    chunk_size, remainder = divmod(len(lst), num_sublists)
    sublists = []
    start_index = 0
    for i in range(num_sublists):
        end_index = start_index + chunk_size
        if i < remainder:
            end_index += 1
        sublists.append(lst[start_index:end_index])
        start_index = end_index
    return sublists


def _add_active_party_columns(  # pylint: disable=R0912
    active_party_columns: str | list[str],
    active_party_columns_mode: (
        Literal[
            "add_to_first",
            "add_to_last",
            "create_as_first",
            "create_as_last",
            "add_to_all",
        ]
        | int
    ),
    partition_columns: list[list[str]],
) -> list[list[str]]:
    """Add active party columns to the partition columns based on the mode.

    Parameters
    ----------
    active_party_columns : Union[str, list[str]]
        List of active party columns.
    active_party_columns_mode : Union[Literal["add_to_first", "add_to_last", "create_as_first", "create_as_last", "add_to_all"], int]
        Mode to add active party columns to partition columns.

    Returns
    -------
    partition_columns: list[list[str]]
        List of partition columns after the modification.
    """
    if isinstance(active_party_columns, str):
        active_party_columns = [active_party_columns]
    if isinstance(active_party_columns_mode, int):
        partition_id = active_party_columns_mode
        if partition_id < 0 or partition_id >= len(partition_columns):
            raise ValueError(
                f"Invalid partition index {partition_id} for active_party_columns_mode."
                f"Must be in the range [0, {len(partition_columns) - 1}]"
                f"but given {partition_id}"
            )
        for column in active_party_columns:
            partition_columns[partition_id].append(column)
    else:
        if active_party_columns_mode == "add_to_first":
            for column in active_party_columns:
                partition_columns[0].append(column)
        elif active_party_columns_mode == "add_to_last":
            for column in active_party_columns:
                partition_columns[-1].append(column)
        elif active_party_columns_mode == "create_as_first":
            partition_columns.insert(0, active_party_columns)
        elif active_party_columns_mode == "create_as_last":
            partition_columns.append(active_party_columns)
        elif active_party_columns_mode == "add_to_all":
            for column in active_party_columns:
                for partition in partition_columns:
                    partition.append(column)
    return partition_columns


def _init_optional_str_or_list_str(parameter: str | list[str] | None) -> list[str]:
    """Initialize a parameter as a list of strings.

    Parameters
    ----------
    parameter : Union[str, list[str], None]
        A parameter that should be a string, a list of strings, or None.

    Returns
    -------
    parameter: list[str]
        The parameter as a list of strings.
    """
    if parameter is None:
        return []
    if not isinstance(parameter, (str | list)):
        raise TypeError("Parameter must be a string or a list of strings")
    if isinstance(parameter, list) and not all(
        isinstance(single_param, str) for single_param in parameter
    ):
        raise TypeError("All elements in the list must be strings")
    if isinstance(parameter, str):
        return [parameter]
    return parameter
