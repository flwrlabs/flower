"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.RecordSet = exports.ConfigsRecord = exports.MetricsRecord = exports.ParametersRecord = exports.ArrayData = void 0;
class ArrayData {
    dtype;
    shape;
    stype;
    data;
    constructor(dtype, shape, stype, data) {
        this.dtype = dtype;
        this.shape = shape;
        this.stype = stype;
        this.data = data;
    }
}
exports.ArrayData = ArrayData;
class ParametersRecord {
    constructor(data = {}) {
        Object.assign(this, data);
    }
}
exports.ParametersRecord = ParametersRecord;
class MetricsRecord {
    constructor(data = {}) {
        Object.assign(this, data);
    }
}
exports.MetricsRecord = MetricsRecord;
class ConfigsRecord {
    constructor(data = {}) {
        Object.assign(this, data);
    }
}
exports.ConfigsRecord = ConfigsRecord;
class RecordSet {
    parametersRecords = {};
    metricsRecords = {};
    configsRecords = {};
    constructor(parametersRecords = {}, metricsRecords = {}, configsRecords = {}) {
        this.parametersRecords = parametersRecords;
        this.metricsRecords = metricsRecords;
        this.configsRecords = configsRecords;
    }
}
exports.RecordSet = RecordSet;
