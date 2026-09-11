/*************************************************************************************************
 *
 * @file start.h
 *
 * @brief Create a gRPC channel to connect to the server and enable message
 *communication
 *
 * @author Lekang Jiang
 *
 * @version 1.0
 *
 * @date 06/09/2021
 *
 *************************************************************************************************/

#ifndef START_H
#define START_H
#pragma once
#include "client.h"
#include "communicator.h"
#include "grpc_rere.h"
#include "message_handler.h"
#include <grpcpp/grpcpp.h>
#include <thread>

#define GRPC_MAX_MESSAGE_LENGTH 536870912 //  == 512 * 1024 * 1024

/**
 * @brief Start a C++ Flower Client which connects to a gRPC server
 * @param  server_address
 *                        The IPv6 address of the server. If the Flower server
 * runs on the same machine on port 8080, then `server_address` would be
 * `"[::]:8080"`.
 *
 *         client
 *                        An implementation of the abstract base class
 * `flwr::Client`
 *
 *         grpc_max_message_length
 *                        int (default: 536_870_912, this equals 512MB).
 *                        The maximum length of gRPC messages that can be
 * exchanged with the Flower server. The default should be sufficient for most
 * models. Users who train very large models might need to increase this value.
 * Note that the Flower server needs to be started with the same value (see
 * `flwr.server.start_server`), otherwise it will not know about the increased
 * limit and block larger messages.
 *
 */

class start {
public:
  static void
  start_client(std::string server_address, flwr_local::Client *client,
               int grpc_max_message_length = GRPC_MAX_MESSAGE_LENGTH);
};
#endif
