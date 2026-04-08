"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.getTaskIns = exports.validateTaskIns = void 0;
const validateTaskIns = (taskIns) => {
    if (!(taskIns.task && taskIns.task.recordset)) {
        return false;
    }
    return true;
};
exports.validateTaskIns = validateTaskIns;
const getTaskIns = (pullTaskInsResponse) => {
    if (pullTaskInsResponse.taskInsList.length === 0) {
        return null;
    }
    return pullTaskInsResponse.taskInsList[0];
};
exports.getTaskIns = getTaskIns;
