#pragma once

#include "client.h"
#include <string>

namespace flwr_quickstart {

constexpr int kGrpcMaxMessageLength = 536870912;

void start_client(const std::string &server_address, flwr_local::Client *client,
                  int grpc_max_message_length = kGrpcMaxMessageLength);

} // namespace flwr_quickstart
