#pragma once

#include "client.h"
#include <string>

namespace flwr_quickstart {

constexpr int kGrpcMaxMessageLength = 536870912;

// Throws on fatal transport or message-processing errors. Callers must report
// failure instead of treating an interrupted client as a successful run.
void start_client(const std::string &server_address, flwr_local::Client *client,
                  int grpc_max_message_length = kGrpcMaxMessageLength);

} // namespace flwr_quickstart
