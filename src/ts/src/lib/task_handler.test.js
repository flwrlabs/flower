"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
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
const task_handler_1 = require("./task_handler");
const recordset_1 = require("./recordset");
const task_1 = require("../protos/flwr/proto/task");
const fleet_1 = require("../protos/flwr/proto/fleet");
const serde_1 = require("./serde");
// Test for validateTaskIns: No task inside TaskIns
describe("validateTaskIns - No task", () => {
    it("should return false when task is null", () => {
        const taskIns = task_1.TaskIns.create({ task: {} });
        expect((0, task_handler_1.validateTaskIns)(taskIns)).toBe(false);
    });
});
// Test for validateTaskIns: No content inside Task
describe("validateTaskIns - No content", () => {
    it("should return false when recordset is null", () => {
        const taskIns = task_1.TaskIns.create({ task: task_1.Task.create() });
        expect((0, task_handler_1.validateTaskIns)(taskIns)).toBe(false);
    });
});
// Test for validateTaskIns: Valid TaskIns
describe("validateTaskIns - Valid TaskIns", () => {
    it("should return true when task contains a valid recordset", () => {
        const recordSet = new recordset_1.RecordSet();
        const taskIns = task_1.TaskIns.create({ task: { recordset: (0, serde_1.recordSetToProto)(recordSet) } });
        expect((0, task_handler_1.validateTaskIns)(taskIns)).toBe(true);
    });
});
// Test for getTaskIns: Empty response
describe("getTaskIns - Empty response", () => {
    it("should return null when task_ins_list is empty", () => {
        const res = fleet_1.PullTaskInsResponse.create({ taskInsList: [] });
        const taskIns = (0, task_handler_1.getTaskIns)(res);
        expect(taskIns).toBeNull();
    });
});
// Test for getTaskIns: Single TaskIns in response
describe("getTaskIns - Single TaskIns", () => {
    it("should return the task ins when task_ins_list contains one task", () => {
        const expectedTaskIns = task_1.TaskIns.create({ taskId: "123", task: task_1.Task.create() });
        const res = fleet_1.PullTaskInsResponse.create({ taskInsList: [expectedTaskIns] });
        const actualTaskIns = (0, task_handler_1.getTaskIns)(res);
        expect(actualTaskIns).toEqual(expectedTaskIns);
    });
});
// Test for getTaskIns: Multiple TaskIns in response
describe("getTaskIns - Multiple TaskIns", () => {
    it("should return the first task ins when task_ins_list contains multiple tasks", () => {
        const expectedTaskIns = task_1.TaskIns.create({ taskId: "123", task: task_1.Task.create() });
        const res = fleet_1.PullTaskInsResponse.create({
            taskInsList: [expectedTaskIns, task_1.TaskIns.create(), task_1.TaskIns.create()],
        });
        const actualTaskIns = (0, task_handler_1.getTaskIns)(res);
        expect(actualTaskIns).toEqual(expectedTaskIns);
    });
});
