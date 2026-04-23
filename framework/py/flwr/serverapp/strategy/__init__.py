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
"""ServerApp strategies."""


from .bulyan import Bulyan
from .dp_adaptive_clipping import (
    DifferentialPrivacyClientSideAdaptiveClipping,
    DifferentialPrivacyServerSideAdaptiveClipping,
)
from .dp_fixed_clipping import (
    DifferentialPrivacyClientSideFixedClipping,
    DifferentialPrivacyServerSideFixedClipping,
)
from .fedadagrad import FedAdagrad
from .fedadam import FedAdam
from .fedavg import FedAvg
from .fedavgm import FedAvgM
from .fedmedian import FedMedian
from .fedprox import FedProx
from .fedrdf import FedRDF as FedRDF
from .fedtrimmedavg import FedTrimmedAvg
from .fedxgb_bagging import FedXgbBagging
from .fedxgb_cyclic import FedXgbCyclic
from .fedyogi import FedYogi
from .krum import Krum
from .multikrum import MultiKrum
from .qfedavg import QFedAvg
from .result import Result
from .strategy import Strategy

__all__ = [
    "Bulyan",
    "DifferentialPrivacyClientSideAdaptiveClipping",
    "DifferentialPrivacyClientSideFixedClipping",
    "DifferentialPrivacyServerSideAdaptiveClipping",
    "DifferentialPrivacyServerSideFixedClipping",
    "FedAdagrad",
    "FedAdam",
    "FedAvg",
    "FedAvgM",
    "FedMedian",
    "FedProx",
    "FedRDF",
    "FedTrimmedAvg",
    "FedXgbBagging",
    "FedXgbCyclic",
    "FedYogi",
    "Krum",
    "MultiKrum",
    "QFedAvg",
    "Result",
    "Strategy",
]
