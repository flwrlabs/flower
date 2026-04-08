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
import { getTaskIns, validateTaskIns } from "./task_handler";
import { RecordSet } from "./recordset";
import { TaskIns, Task } from "../protos/flwr/proto/task";
import { PullTaskInsResponse } from "../protos/flwr/proto/fleet";
import { recordSetToProto } from "./serde";

// Test for validateTaskIns: No task inside TaskIns
describe("validateTaskIns - No task", () => {
  it("should return false when task is null", () => {
    const taskIns = TaskIns.create({ task: {} });
    expect(validateTaskIns(taskIns)).toBe(false);
  });
});

// Test for validateTaskIns: No content inside Task
describe("validateTaskIns - No content", () => {
  it("should return false when recordset is null", () => {
    const taskIns = TaskIns.create({ task: Task.create() });
    expect(validateTaskIns(taskIns)).toBe(false);
  });
});

// Test for validateTaskIns: Valid TaskIns
describe("validateTaskIns - Valid TaskIns", () => {
  it("should return true when task contains a valid recordset", () => {
    const recordSet = new RecordSet();
    const taskIns = TaskIns.create({ task: { recordset: recordSetToProto(recordSet) } });
    expect(validateTaskIns(taskIns)).toBe(true);
  });
});

// Test for getTaskIns: Empty response
describe("getTaskIns - Empty response", () => {
  it("should return null when task_ins_list is empty", () => {
    const res = PullTaskInsResponse.create({ taskInsList: [] });
    const taskIns = getTaskIns(res);
    expect(taskIns).toBeNull();
  });
});

// Test for getTaskIns: Single TaskIns in response
describe("getTaskIns - Single TaskIns", () => {
  it("should return the task ins when task_ins_list contains one task", () => {
    const expectedTaskIns = TaskIns.create({ taskId: "123", task: Task.create() });
    const res = PullTaskInsResponse.create({ taskInsList: [expectedTaskIns] });
    const actualTaskIns = getTaskIns(res);
    expect(actualTaskIns).toEqual(expectedTaskIns);
  });
});

// Test for getTaskIns: Multiple TaskIns in response
describe("getTaskIns - Multiple TaskIns", () => {
  it("should return the first task ins when task_ins_list contains multiple tasks", () => {
    const expectedTaskIns = TaskIns.create({ taskId: "123", task: Task.create() });
    const res = PullTaskInsResponse.create({
      taskInsList: [expectedTaskIns, TaskIns.create(), TaskIns.create()],
    });
    const actualTaskIns = getTaskIns(res);
    expect(actualTaskIns).toEqual(expectedTaskIns);
  });
});
