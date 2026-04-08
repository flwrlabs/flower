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
import { isIP } from "net";

const IPV6 = 6;
const IPV4 = 4;

interface ParsedAddress {
  host: string;
  port: number;
  version: boolean | null;
}

export function parseAddress(address: string): ParsedAddress | null {
  try {
    const lastColonIndex = address.lastIndexOf(":");

    if (lastColonIndex === -1) {
      throw new Error("No port was provided.");
    }

    // Split the address into host and port.
    const rawHost = address.slice(0, lastColonIndex);
    const rawPort = address.slice(lastColonIndex + 1);

    const port = parseInt(rawPort, 10);

    if (port > 65535 || port < 1) {
      throw new Error("Port number is invalid.");
    }

    let host = rawHost.replace(/[\[\]]/g, ""); // Remove brackets for IPv6
    let version: boolean | null = null;

    const ipVersion = isIP(host);
    if (ipVersion === IPV6) {
      version = true;
    } else if (ipVersion === IPV4) {
      version = false;
    }

    return {
      host,
      port,
      version,
    };
  } catch (err) {
    return null;
  }
}
