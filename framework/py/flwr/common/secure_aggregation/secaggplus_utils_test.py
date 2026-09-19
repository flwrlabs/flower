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
"""SecAgg+ pseudo-random generator tests."""


import numpy as np
import pytest

from .secaggplus_utils import pseudo_rand_gen


def test_pseudo_rand_gen_determinism() -> None:
    """Test that identical seeds produce identical masks."""
    seed = b"0123456789abcdef0123456789abcdef"
    dims = [(10,)]
    m1 = pseudo_rand_gen(seed, 2**32, dims)
    m2 = pseudo_rand_gen(seed, 2**32, dims)
    assert len(m1) == len(m2) == 1
    assert np.array_equal(m1[0], m2[0])


def test_pseudo_rand_gen_entropy_sensitivity() -> None:
    """Test that different seeds produce different outputs."""
    seed_a = b"\x00" * 32
    seed_b = b"\x01" + b"\x00" * 31
    m_a = pseudo_rand_gen(seed_a, 2**32, [(1000,)])
    m_b = pseudo_rand_gen(seed_b, 2**32, [(1000,)])
    assert len(m_a) == len(m_b) == 1
    assert not np.array_equal(m_a[0], m_b[0])


def test_pseudo_rand_gen_value_range() -> None:
    """Test that values are within the expected range."""
    mask_list = pseudo_rand_gen(b"test", 2**22, [(1000,)])
    assert len(mask_list) == 1
    mask = mask_list[0]
    assert mask.min() >= 0
    assert mask.max() < 2**22


def test_pseudo_rand_gen_power_of_two() -> None:
    """Test that non-power-of-two num_range raises an error."""
    with pytest.raises(ValueError):
        pseudo_rand_gen(b"test", 100, [(10,)])


def test_pseudo_rand_gen_empty_dimension() -> None:
    """Test behavior with scalar shape."""
    seed = b"scalar_test"
    m = pseudo_rand_gen(seed, 2**32, [()])
    assert len(m) == 1
    assert m[0].shape == ()


def test_pseudo_rand_gen_shape_fidelity() -> None:
    """Test that all returned arrays have exact shapes and dtype."""
    dims = [(10, 5), (3,), (), (2, 2, 2)]
    result = pseudo_rand_gen(b"shape_test", 2**32, dims)
    assert len(result) == len(dims)
    for res_arr, expected_shape in zip(result, dims):
        assert res_arr.shape == expected_shape
        assert res_arr.dtype == np.int64
