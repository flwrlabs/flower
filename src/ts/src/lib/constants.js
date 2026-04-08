"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.FLWR_HOME = exports.FAB_CONFIG_FILE = exports.APP_DIR = exports.PING_MAX_INTERVAL = exports.PING_RANDOM_RANGE = exports.PING_BASE_MULTIPLIER = exports.PING_CALL_TIMEOUT = exports.PING_DEFAULT_INTERVAL = void 0;
// Copyright 2024 Flower Labs GmbH. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
exports.PING_DEFAULT_INTERVAL = 30;
exports.PING_CALL_TIMEOUT = 5;
exports.PING_BASE_MULTIPLIER = 0.8;
exports.PING_RANDOM_RANGE = [-0.1, 0.1];
exports.PING_MAX_INTERVAL = 1e300;
exports.APP_DIR = "apps";
exports.FAB_CONFIG_FILE = "pyproject.toml";
exports.FLWR_HOME = "FLWR_HOME";
