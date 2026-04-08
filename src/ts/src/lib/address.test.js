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
const address_1 = require("./address");
describe("parseAddress", () => {
    test("parses a valid IPv4 address", () => {
        const result = (0, address_1.parseAddress)("127.0.0.1:8080");
        expect(result).toEqual({
            host: "127.0.0.1",
            port: 8080,
            version: false, // IPv4 address
        });
    });
    test("parses a valid IPv6 address", () => {
        const result = (0, address_1.parseAddress)("[::1]:8080");
        expect(result).toEqual({
            host: "::1",
            port: 8080,
            version: true, // IPv6 address
        });
    });
    test("returns null for an invalid port number", () => {
        const result = (0, address_1.parseAddress)("127.0.0.1:70000"); // Invalid port
        expect(result).toBeNull();
    });
    test("returns null for missing port", () => {
        const result = (0, address_1.parseAddress)("127.0.0.1"); // No port provided
        expect(result).toBeNull();
    });
    test("returns null for an invalid address format", () => {
        const result = (0, address_1.parseAddress)("notAnAddress");
        expect(result).toBeNull();
    });
    test("parses domain names correctly", () => {
        const result = (0, address_1.parseAddress)("example.com:8080");
        expect(result).toEqual({
            host: "example.com",
            port: 8080,
            version: null, // Domain names do not have IP versions
        });
    });
    test("parses IPv6 with brackets and returns proper version", () => {
        const result = (0, address_1.parseAddress)("[2001:db8::ff00:42:8329]:9090");
        expect(result).toEqual({
            host: "2001:db8::ff00:42:8329",
            port: 9090,
            version: true, // IPv6 address
        });
    });
});
