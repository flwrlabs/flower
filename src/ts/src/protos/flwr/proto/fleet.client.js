"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.FleetClient = void 0;
const fleet_1 = require("./fleet");
const runtime_rpc_1 = require("@protobuf-ts/runtime-rpc");
/**
 * @generated from protobuf service flwr.proto.Fleet
 */
class FleetClient {
    _transport;
    typeName = fleet_1.Fleet.typeName;
    methods = fleet_1.Fleet.methods;
    options = fleet_1.Fleet.options;
    constructor(_transport) {
        this._transport = _transport;
    }
    /**
     * @generated from protobuf rpc: CreateNode(flwr.proto.CreateNodeRequest) returns (flwr.proto.CreateNodeResponse);
     */
    createNode(input, options) {
        const method = this.methods[0], opt = this._transport.mergeOptions(options);
        return (0, runtime_rpc_1.stackIntercept)("unary", this._transport, method, opt, input);
    }
    /**
     * @generated from protobuf rpc: DeleteNode(flwr.proto.DeleteNodeRequest) returns (flwr.proto.DeleteNodeResponse);
     */
    deleteNode(input, options) {
        const method = this.methods[1], opt = this._transport.mergeOptions(options);
        return (0, runtime_rpc_1.stackIntercept)("unary", this._transport, method, opt, input);
    }
    /**
     * @generated from protobuf rpc: Ping(flwr.proto.PingRequest) returns (flwr.proto.PingResponse);
     */
    ping(input, options) {
        const method = this.methods[2], opt = this._transport.mergeOptions(options);
        return (0, runtime_rpc_1.stackIntercept)("unary", this._transport, method, opt, input);
    }
    /**
     * Retrieve one or more tasks, if possible
     *
     * HTTP API path: /api/v1/fleet/pull-task-ins
     *
     * @generated from protobuf rpc: PullTaskIns(flwr.proto.PullTaskInsRequest) returns (flwr.proto.PullTaskInsResponse);
     */
    pullTaskIns(input, options) {
        const method = this.methods[3], opt = this._transport.mergeOptions(options);
        return (0, runtime_rpc_1.stackIntercept)("unary", this._transport, method, opt, input);
    }
    /**
     * Complete one or more tasks, if possible
     *
     * HTTP API path: /api/v1/fleet/push-task-res
     *
     * @generated from protobuf rpc: PushTaskRes(flwr.proto.PushTaskResRequest) returns (flwr.proto.PushTaskResResponse);
     */
    pushTaskRes(input, options) {
        const method = this.methods[4], opt = this._transport.mergeOptions(options);
        return (0, runtime_rpc_1.stackIntercept)("unary", this._transport, method, opt, input);
    }
    /**
     * @generated from protobuf rpc: GetRun(flwr.proto.GetRunRequest) returns (flwr.proto.GetRunResponse);
     */
    getRun(input, options) {
        const method = this.methods[5], opt = this._transport.mergeOptions(options);
        return (0, runtime_rpc_1.stackIntercept)("unary", this._transport, method, opt, input);
    }
    /**
     * Get FAB
     *
     * @generated from protobuf rpc: GetFab(flwr.proto.GetFabRequest) returns (flwr.proto.GetFabResponse);
     */
    getFab(input, options) {
        const method = this.methods[6], opt = this._transport.mergeOptions(options);
        return (0, runtime_rpc_1.stackIntercept)("unary", this._transport, method, opt, input);
    }
}
exports.FleetClient = FleetClient;
